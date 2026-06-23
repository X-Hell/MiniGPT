#!/usr/bin/env python3
"""monitor.py — live rich TUI monitor for the MiniGPT training run.

Tails a training log, parses per-step metrics, and renders a refreshing
dashboard: header, status table, loss-regime indicator, sparkline, tripwire
panel, and alerts. Press 's' to dump a snapshot to status_snapshot.txt; 'q'
or Ctrl+C to quit.

Handles BOTH log formats:
  real : [2026-06-23 09:01:52] step=  940 loss=7.8291 grad_norm=10.57 lr=1.41e-04 tok/s=15063.5
         [2026-06-23 09:05:59] eval step= 1000 val_loss=7.9072 best_val_loss=7.7983
  spec : step 920 | loss 7.82 | val_loss 7.80 | lr 1.38e-4 | grad_norm 1.61 | tok/s 14900

Usage:
    .venv/bin/python monitor.py --log outputs/modern_30m/training.log \
        --pid 19314 --total-steps 36000 --start-step 0
"""
import argparse, os, sys, re, time, threading
from collections import deque
from datetime import datetime, timedelta

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Two-eval + spike-cluster logic lives in minigpt.stability (pure-Python classes).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
try:
    from minigpt.stability import EvaluationMonitor, GradientSpikeDetector
except Exception:                       # degrade gracefully if src/ isn't importable
    EvaluationMonitor = GradientSpikeDetector = None

# ── loss regime thresholds (nats), vocab=16384 ──────────────────────────────
RANDOM, UNIGRAM, BIGRAM, TARGET = 9.70, 7.50, 6.30, 4.50
TRIPWIRE_STEP = 4000            # val_loss must be < UNIGRAM by here (warmup ends 3500)
VRAM_WARN_MB, VRAM_CRIT_MB = 11_000, 11_500    # I-7: batch-32 ceiling (~11.3GB measured)
GRAD_WARN, GRAD_CRIT = 12.0, 40.0              # I-1: clustering thresholds
VRAM_RE = re.compile(r"\[VRAM[^\]]*\]\s*([0-9.]+)\s*/\s*([0-9.]+)\s*MB")
SPARK = "▁▂▃▄▅▆▇█"

# Per-field regexes — tolerant of `key=val`, `key: val`, `key val`, and the
# leading spaces the real log pads step numbers with. \b before `loss`/`val_loss`
# means `loss` won't match inside `val_loss`/`best_val_loss` (underscore is a
# word char, so there is no boundary there).
NUM = r"([0-9][0-9.eE+\-]*)"
FIELD_RE = {
    "step":      re.compile(r"\bstep[=:\s]+(\d+)"),
    "loss":      re.compile(r"\bloss[=:\s]+" + NUM),
    "val_loss":  re.compile(r"\bval_loss[=:\s]+" + NUM),
    "grad_norm": re.compile(r"\bgrad_norm[=:\s]+" + NUM),
    "lr":        re.compile(r"\blr[=:\s]+" + NUM),
    "tok_s":     re.compile(r"\btok/s[=:\s]+" + NUM),
}
TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")


class State:
    """Thread-safe rolling store of parsed log entries."""

    def __init__(self, maxlen=500):
        self.entries = deque(maxlen=maxlen)    # merged per-step dicts
        self.val_history = []                  # (step, val_loss) on each eval
        self.crossed = {}                      # regime name -> step first crossed
        self.lock = threading.Lock()
        # Run-2 detectors (None if minigpt.stability could not be imported).
        self.eval_mon = EvaluationMonitor() if EvaluationMonitor else None
        self.spike_det = GradientSpikeDetector(warn=GRAD_WARN, crit=GRAD_CRIT) if GradientSpikeDetector else None
        self.last_spike = None                 # latest GradientSpikeDetector status
        self.vram = None                       # (used_mb, total_mb) from [VRAM ...] lines

    def update(self, line):
        ts = None
        m = TS_RE.match(line)
        if m:
            try:
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                ts = None

        mv = VRAM_RE.search(line)              # [VRAM ...] lines carry no step= field
        if mv:
            try:
                with self.lock:
                    self.vram = (float(mv.group(1)), float(mv.group(2)))
            except ValueError:
                pass
        vals = {}
        for key, rx in FIELD_RE.items():
            mm = rx.search(line)
            if mm:
                try:
                    vals[key] = float(mm.group(1))
                except ValueError:
                    pass
        if "step" not in vals:
            return                             # not a metrics line — ignore
        step = int(vals["step"])
        with self.lock:
            if self.entries and self.entries[-1]["step"] == step:
                e = self.entries[-1]           # merge eval+train for same step
            else:
                e = {"step": step, "loss": None, "val_loss": None,
                     "grad_norm": None, "lr": None, "tok_s": None, "t": ts}
                self.entries.append(e)
            for k in ("loss", "val_loss", "grad_norm", "lr", "tok_s"):
                if k in vals:
                    e[k] = vals[k]
            if ts:
                e["t"] = ts
            if "val_loss" in vals:
                self.val_history.append((step, vals["val_loss"]))
            cur = e["val_loss"] if e["val_loss"] is not None else e["loss"]
            if cur is not None:
                for name, thr in (("random", RANDOM), ("unigram", UNIGRAM),
                                  ("bigram", BIGRAM), ("target", TARGET)):
                    if cur < thr and name not in self.crossed:
                        self.crossed[name] = step
            # Feed the Run-2 detectors as lines stream (backfill replays history).
            if self.eval_mon is not None:
                if "loss" in vals:
                    self.eval_mon.record_train(step, vals["loss"])
                if "grad_norm" in vals and self.spike_det is not None:
                    self.last_spike = self.spike_det.update(vals["grad_norm"])
                if "val_loss" in vals:
                    self.eval_mon.record_eval(step, vals["val_loss"])

    def snapshot(self):
        with self.lock:
            return (list(self.entries), list(self.val_history), dict(self.crossed))


# ── helpers ─────────────────────────────────────────────────────────────────
def pid_alive(pid):
    if not pid:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True


def regime(loss):
    if loss is None:
        return ("UNKNOWN", "white")
    if loss >= RANDOM:
        return ("RANDOM INIT", "bold red")
    if loss >= UNIGRAM:
        return ("FREQ FIT", "yellow")
    if loss >= BIGRAM:
        return ("UNIGRAM FLOOR", "cyan")
    if loss >= TARGET:
        return ("CONTEXT LEARNING", "green")
    return ("TARGET RANGE", "bold green")


def sparkline(vals, width=72):
    vals = [v for v in vals if v is not None][-width:]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return "".join(SPARK[min(7, int((v - lo) / rng * 7))] for v in vals)


def rate_and_eta(entries, total_steps):
    timed = [e for e in entries if e.get("t")]
    cur = entries[-1]["step"] if entries else 0
    sps = 0.0
    if len(timed) >= 2:
        ds = timed[-1]["step"] - timed[0]["step"]
        dt = (timed[-1]["t"] - timed[0]["t"]).total_seconds()
        sps = ds / dt if dt > 0 else 0.0
    eta = (total_steps - cur) / sps if sps > 0 and cur < total_steps else None
    return sps, eta, cur


def fmt_dur(sec):
    return "—" if sec is None else str(timedelta(seconds=int(sec)))


def compute_alerts(state, cur_step, pid):
    """Run-2 alerts: grad clustering (not isolated spikes), two-eval confirmed
    val regression, the unigram tripwire, VRAM budget, and PID liveness."""
    entries, val_history, _ = state.snapshot()
    out = []
    last = entries[-1] if entries else None

    # Gradient clustering (I-1) — isolated spikes are ignored by the detector.
    spike = getattr(state, "last_spike", None)
    g = last.get("grad_norm") if last else None
    if spike == "CRITICAL":
        out.append(f"CRITICAL grad cluster (grad_norm {g:.1f}; >{GRAD_CRIT:.0f} or 5×>{GRAD_WARN:.0f})"
                   if g is not None else "CRITICAL grad cluster")
    elif spike == "WARN":
        out.append(f"WARN grad cluster: ≥3 consecutive logged points > {GRAD_WARN:.0f}")

    # Two-eval confirmed regression (I-6) — a new train/val low cancels pending.
    em = getattr(state, "eval_mon", None)
    if em is not None and em.status == "CONFIRMED":
        out.append(f"CONFIRMED val regression: {em.detail}")

    # Unigram tripwire (I-1) — warmup ends 3500; allow 500 steps to cross.
    last_val = val_history[-1][1] if val_history else None
    if cur_step >= TRIPWIRE_STEP and last_val is not None and last_val > UNIGRAM:
        out.append(f"step {cur_step} ≥ {TRIPWIRE_STEP} but val_loss "
                   f"{last_val:.2f} > {UNIGRAM}  (not learning context)")

    # VRAM budget (I-7).
    vram = getattr(state, "vram", None)
    if vram:
        used, total = vram
        if used >= VRAM_CRIT_MB:
            out.append(f"CRITICAL VRAM {used:.0f}/{total:.0f} MB (≥ {VRAM_CRIT_MB}) — OOM risk")
        elif used >= VRAM_WARN_MB:
            out.append(f"WARN VRAM {used:.0f}/{total:.0f} MB (≥ {VRAM_WARN_MB})")

    if not pid_alive(pid):
        out.append(f"PID {pid} not found — training process may have CRASHED")
    return out


# ── rendering ─────────────────────────────────────────────────────────────────
def metric_cell(label, value, color):
    return Text(value, style=color)


def build_status_table(last, last_val=None):
    t = Table(show_header=True, header_style="bold", expand=True)
    for col in ("step", "train loss", "val loss", "grad_norm", "tok/s", "lr"):
        t.add_column(col, justify="right")
    if not last:
        t.add_row("—", "—", "—", "—", "—", "—")
        return t
    loss = last.get("loss")
    # carry the most recent eval forward (val checkpoints are every 500 steps)
    vl = last.get("val_loss") if last.get("val_loss") is not None else last_val
    gn = last.get("grad_norm")
    toks = last.get("tok_s")
    lr = last.get("lr")

    def loss_color(x):
        if x is None:
            return "white"
        return "green" if x < UNIGRAM else ("yellow" if x < RANDOM else "red")

    gn_color = "white" if gn is None else ("green" if gn < 2 else ("yellow" if gn < 5 else "bold red"))
    t.add_row(
        f"{last['step']:,}",
        Text(f"{loss:.4f}" if loss is not None else "—", style=loss_color(loss)),
        Text(f"{vl:.4f}" if vl is not None else "—", style=loss_color(vl)),
        Text(f"{gn:.2f}" if gn is not None else "—", style=gn_color),
        Text(f"{toks:,.0f}" if toks is not None else "—", style="cyan"),
        f"{lr:.2e}" if lr is not None else "—",
    )
    return t


def build_tripwire_table(crossed, cur_step, val_history):
    t = Table(show_header=True, header_style="bold", expand=True)
    t.add_column("threshold")
    t.add_column("status", justify="left")
    rows = [("Random init  (9.70)", "random"),
            ("Unigram floor (7.50)", "unigram"),
            ("Bigram floor  (6.30)", "bigram"),
            ("Target range  (4.50)", "target")]
    for label, key in rows:
        if key in crossed:
            t.add_row(label, Text(f"✅ crossed @ step {crossed[key]:,}", style="green"))
        else:
            t.add_row(label, Text("⏳ pending", style="dim"))
    last_val = val_history[-1][1] if val_history else None
    if "unigram" in crossed:
        t.add_row("Step-3500 tripwire", Text("✅ unigram beaten before 3500", style="green"))
    elif cur_step < TRIPWIRE_STEP:
        t.add_row("Step-3500 tripwire",
                  Text(f"⏳ pending (at step {cur_step:,}/{TRIPWIRE_STEP})", style="dim"))
    else:
        t.add_row("Step-3500 tripwire",
                  Text(f"❌ val_loss {last_val:.2f} > 7.50 at step {cur_step:,}", style="bold red"))
    return t


def render(state, args, start_wall, alerts):
    entries, val_history, crossed = state.snapshot()
    last = entries[-1] if entries else None
    sps, eta, cur = rate_and_eta(entries, args.total_steps)
    cur_loss = (last.get("val_loss") or last.get("loss")) if last else None
    reg_name, reg_color = regime(last.get("loss") if last else None)

    pct = 100.0 * cur / args.total_steps if args.total_steps else 0.0
    elapsed = time.time() - start_wall
    tokens = cur * args.tokens_per_step
    finish = (datetime.now() + timedelta(seconds=eta)).strftime("%b %d %H:%M") if eta else "—"
    alive = pid_alive(args.pid)

    header = Text()
    header.append(f" {args.name} ", style="bold white on blue")
    header.append(f"  PID {args.pid} ")
    header.append("● alive" if alive else "● DEAD", style="green" if alive else "bold red")
    header.append(f"   elapsed {fmt_dur(elapsed)}   ETA {fmt_dur(eta)}   finish ~{finish}")

    bar_stops = ["RANDOM INIT", "FREQ FIT", "UNIGRAM FLOOR", "CONTEXT LEARNING", "TARGET RANGE"]
    bar = Text("  ")
    for s in bar_stops:
        if s == reg_name:
            bar.append(f"[{s}]", style=f"bold {reg_color}")
        else:
            bar.append(f" {s} ", style="dim")
        bar.append(" → " if s != bar_stops[-1] else "")
    regime_panel = Panel(
        Group(bar, Text(f"\n  current loss {cur_loss:.4f} nats" if cur_loss is not None else "  —",
                        style=reg_color)),
        title="loss regime", border_style=reg_color)

    losses = [e.get("loss") for e in entries][-200:]
    spark = sparkline(losses)
    lo = min([v for v in losses if v is not None], default=0)
    hi = max([v for v in losses if v is not None], default=0)
    spark_panel = Panel(
        Group(Text(spark, style="cyan"),
              Text(f"last {len([v for v in losses if v is not None])} steps   "
                   f"min {lo:.3f}  max {hi:.3f}", style="dim")),
        title="train loss (last 200 steps)", border_style="cyan")

    if alerts:
        alert_body = Group(*[Text("  ❌ " + a, style="bold red") for a in alerts])
        alert_panel = Panel(alert_body, title="⚠ ALERTS", border_style="bold red")
    else:
        alert_panel = Panel(Text("  ✅ all systems healthy — no alerts", style="green"),
                            title="alerts", border_style="green")

    vram = getattr(state, "vram", None)
    vram_str = f"VRAM {vram[0]:.0f}/{vram[1]:.0f}MB   " if vram else ""
    footer = Text(
        f"  progress {cur:,}/{args.total_steps:,} ({pct:.1f}%)   "
        f"{sps:.3f} steps/s   {tokens/1e6:,.1f}M tokens   {vram_str}"
        f"peak LR {args.peak_lr:g} @ step {args.warmup}   [s]=snapshot  [q]=quit", style="dim")

    last_val = val_history[-1][1] if val_history else None
    return Group(
        Panel(header, border_style="blue"),
        build_status_table(last, last_val),
        regime_panel,
        spark_panel,
        build_tripwire_table(crossed, cur, val_history),
        alert_panel,
        footer,
    )


def snapshot_text(state, args):
    entries, val_history, crossed = state.snapshot()
    last = entries[-1] if entries else {}
    sps, eta, cur = rate_and_eta(entries, args.total_steps)
    reg_name, _ = regime(last.get("loss"))
    last_val = last.get("val_loss")
    if last_val is None and val_history:
        last_val = val_history[-1][1]
    lines = [
        "=" * 56,
        f" {args.name} — status snapshot {datetime.now():%Y-%m-%d %H:%M:%S}",
        "=" * 56,
        f"  PID {args.pid} alive : {pid_alive(args.pid)}",
        f"  step             : {cur:,} / {args.total_steps:,} "
        f"({100.0*cur/args.total_steps:.1f}%)",
        f"  train loss       : {last.get('loss')}",
        f"  val loss         : {last_val}",
        f"  grad_norm        : {last.get('grad_norm')}",
        f"  tok/s            : {last.get('tok_s')}",
        f"  lr               : {last.get('lr')}",
        f"  regime           : {reg_name}",
        f"  steps/s          : {sps:.3f}",
        f"  ETA              : {fmt_dur(eta)}",
        f"  thresholds       : " + (", ".join(f"{k}@{v}" for k, v in crossed.items()) or "none"),
        "=" * 56,
    ]
    return "\n".join(lines)


# ── background threads ────────────────────────────────────────────────────────
def tail(path, state, stop):
    while not stop.is_set() and not os.path.exists(path):
        time.sleep(1.0)
    if stop.is_set():
        return
    with open(path, "r") as f:
        for line in f:                         # backfill existing history
            state.update(line.rstrip("\n"))
        while not stop.is_set():
            line = f.readline()
            if not line:
                time.sleep(0.5)
                continue
            state.update(line.rstrip("\n"))


def key_listener(flag, stop):
    try:
        import termios, tty, select
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
    except Exception:
        return                                 # not an interactive tty
    try:
        tty.setcbreak(fd)
        while not stop.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.4)
            if r:
                ch = sys.stdin.read(1)
                if ch == "s":
                    flag["snap"] = True
                elif ch in ("q", "Q"):
                    stop.set()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    p = argparse.ArgumentParser(description="Live rich TUI monitor for MiniGPT training.")
    p.add_argument("--log", default="outputs/modern_30m/training.log")
    p.add_argument("--pid", type=int, default=None)
    p.add_argument("--total-steps", type=int, default=36000)
    p.add_argument("--start-step", type=int, default=0)
    p.add_argument("--name", default="modern_30m")
    p.add_argument("--tokens-per-step", type=int, default=128 * 512,
                   help="effective batch * seq_len (default 65536)")
    p.add_argument("--peak-lr", type=float, default=2e-4, help="peak LR (footer display only)")
    p.add_argument("--warmup", type=int, default=3500, help="warmup steps (footer display only)")
    args = p.parse_args()

    console = Console()
    state = State()
    stop = threading.Event()
    flag = {"snap": False}
    threading.Thread(target=tail, args=(args.log, state, stop), daemon=True).start()
    threading.Thread(target=key_listener, args=(flag, stop), daemon=True).start()

    start_wall = time.time()
    prev_alerts = set()
    try:
        with Live(console=console, auto_refresh=False, screen=False) as live:
            while not stop.is_set():
                ents, _, _ = state.snapshot()
                cur = ents[-1]["step"] if ents else 0
                alerts = compute_alerts(state, cur, args.pid)
                live.update(render(state, args, start_wall, alerts), refresh=True)

                now = set(alerts)
                if now - prev_alerts:               # ring bell only on NEW alert
                    sys.stdout.write("\a")
                    sys.stdout.flush()
                prev_alerts = now

                if flag["snap"]:
                    flag["snap"] = False
                    txt = snapshot_text(state, args)
                    try:
                        with open("status_snapshot.txt", "w") as fh:
                            fh.write(txt + "\n")
                    except OSError:
                        pass
                    live.console.print(Panel(txt, title="snapshot → status_snapshot.txt",
                                             border_style="magenta"))
                time.sleep(2.0)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        console.print("[dim]monitor stopped.[/dim]")


if __name__ == "__main__":
    main()
