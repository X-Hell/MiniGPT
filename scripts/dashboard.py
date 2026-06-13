#!/usr/bin/env python3
"""Live training dashboard for the MiniGPT Jetson run.

Merges metrics parsed from the training log with live Jetson hardware stats
(via the jtop service) into a single auto-refreshing terminal dashboard.

Usage:
    python3 scripts/dashboard.py [--log PATH] [--interval SECONDS] [--once]

Defaults to the active gpt1_jetson_10h run. Ctrl-C to quit.
"""
import argparse
import os
import re
import sys
import time
from datetime import datetime

# ---- ANSI helpers ---------------------------------------------------------
CSI = "\x1b["
RESET = CSI + "0m"
BOLD = CSI + "1m"
DIM = CSI + "2m"
HIDE_CUR = CSI + "?25l"
SHOW_CUR = CSI + "?25h"
CLEAR = CSI + "2J" + CSI + "H"
HOME = CSI + "H"


def c(text, code):
    return f"{CSI}{code}m{text}{RESET}"


def color_for(value, warn, crit):
    """Green below warn, yellow up to crit, red above (for temps/mem)."""
    if value >= crit:
        return "31"  # red
    if value >= warn:
        return "33"  # yellow
    return "32"      # green


def util_color(value):
    """For utilization, high is desirable: green when busy, red when idle."""
    if value >= 70:
        return "32"  # green
    if value >= 30:
        return "33"  # yellow
    return "31"      # red


SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values, width=40):
    if not values:
        return ""
    vals = values[-width:]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return "".join(SPARK[min(len(SPARK) - 1, int((v - lo) / span * (len(SPARK) - 1)))] for v in vals)


def bar(frac, width, code="32"):
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return c("█" * filled, code) + c("░" * (width - filled), "2")


def fmt_dur(secs):
    secs = int(max(0, secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


# ---- Log parsing ----------------------------------------------------------
STEP_RE = re.compile(
    r"\[(?P<ts>[\d\-: ]+)\]\s+step=\s*(?P<step>\d+)\s+loss=(?P<loss>[\d.]+)\s+"
    r"grad_norm=(?P<gn>[\d.]+)\s+lr=(?P<lr>[\deE.\-+]+)\s+tok/s=(?P<toks>[\d.]+)"
)
CFG_RE = re.compile(r"steps=(\d+)")
PARAM_RE = re.compile(r"Model params:\s*([\d,]+)")
RUNCFG_RE = re.compile(r"Run config:\s*(.+)")
TS_FMT = "%Y-%m-%d %H:%M:%S"


def parse_log(path):
    info = {
        "steps_total": None, "params": None, "runcfg": None,
        "start_ts": None, "rows": [],
    }
    if not os.path.exists(path):
        return info
    with open(path, "r", errors="replace") as f:
        for line in f:
            if info["steps_total"] is None:
                m = CFG_RE.search(line)
                if m and "Run config" in line:
                    info["steps_total"] = int(m.group(1))
            if info["params"] is None:
                m = PARAM_RE.search(line)
                if m:
                    info["params"] = int(m.group(1).replace(",", ""))
            if info["runcfg"] is None:
                m = RUNCFG_RE.search(line)
                if m:
                    info["runcfg"] = m.group(1).strip()
            if info["start_ts"] is None:
                ts = re.match(r"\[([\d\-: ]+)\]", line)
                if ts:
                    try:
                        info["start_ts"] = datetime.strptime(ts.group(1), TS_FMT)
                    except ValueError:
                        pass
            m = STEP_RE.search(line)
            if m:
                info["rows"].append({
                    "ts": datetime.strptime(m.group("ts"), TS_FMT),
                    "step": int(m.group("step")),
                    "loss": float(m.group("loss")),
                    "gn": float(m.group("gn")),
                    "lr": float(m.group("lr")),
                    "toks": float(m.group("toks")),
                })
    return info


# ---- Hardware (jtop) ------------------------------------------------------
def get_hw(jet):
    if jet is None or not jet.ok():
        return None
    g = jet.gpu.get("gpu", {})
    hw = {
        "gpu_load": g.get("status", {}).get("load"),
        "gpu_freq": (g.get("freq", {}).get("cur") or 0) / 1000.0,  # MHz
        "gpu_fmax": (g.get("freq", {}).get("max") or 0) / 1000.0,
        "temps": {k: v.get("temp") for k, v in jet.temperature.items() if v.get("online")},
        "power": jet.power["tot"]["power"] / 1000.0,       # W
        "power_avg": jet.power["tot"].get("avg", 0) / 1000.0,
        "ram_used": jet.memory["RAM"]["used"] / 1024.0 / 1024.0,  # GiB
        "ram_tot": jet.memory["RAM"]["tot"] / 1024.0 / 1024.0,
    }
    # per-core CPU load average
    loads = []
    for k, v in jet.stats.items():
        if k.startswith("CPU") and isinstance(v, (int, float)):
            loads.append(v)
    hw["cpu_load"] = sum(loads) / len(loads) if loads else None
    fan = jet.stats.get("Fan pwmfan0")
    hw["fan"] = fan
    return hw


# ---- Render ---------------------------------------------------------------
def render(info, hw, log_path):
    W = 64
    out = []
    now = datetime.now()
    rows = info["rows"]
    last = rows[-1] if rows else None
    total = info["steps_total"] or 0

    out.append(c("╔" + "═" * (W - 2) + "╗", "36"))
    title = " MiniGPT · GPT-1 · Jetson Orin — Live Training Dashboard "
    pad = (W - 2 - len(title)) // 2
    out.append(c("║", "36") + " " * pad + c(title.strip(), "1;36").center(0) +
               " " * (W - 2 - pad - len(title)) + c("║", "36"))
    out.append(c("╠" + "═" * (W - 2) + "╣", "36"))

    def row(s):
        # s may contain ANSI; pad on visible length
        visible = re.sub(r"\x1b\[[0-9;]*m", "", s)
        padlen = max(0, (W - 2) - len(visible))
        out.append(c("║", "36") + s + " " * padlen + c("║", "36"))

    # --- clock / status
    status = c("● RUNNING", "1;32") if last else c("● WAITING", "1;33")
    row(f" {status}   {DIM}clock{RESET} {now.strftime('%H:%M:%S')}   "
        f"{DIM}{now.strftime('%Y-%m-%d')}{RESET}")

    # --- progress
    if last and total:
        frac = last["step"] / total
        # rate from recent rows
        if len(rows) >= 2:
            dt = (rows[-1]["ts"] - rows[0]["ts"]).total_seconds()
            dstep = rows[-1]["step"] - rows[0]["step"]
            sps = dt / dstep if dstep else 0
        else:
            sps = 0
        remain = (total - last["step"]) * sps
        eta = (now + __import__("datetime").timedelta(seconds=remain)).strftime("%H:%M") if sps else "--:--"
        elapsed = (now - info["start_ts"]).total_seconds() if info["start_ts"] else 0
        step_str = c(str(last["step"]), "1;37")
        pct_str = c(f"{frac*100:5.1f}%", "1;32")
        eta_str = c(eta, "1;36")
        row("")
        row(f" {BOLD}TRAINING PROGRESS{RESET}")
        row(f"   {bar(frac, 46, '32')} {pct_str}")
        row(f"   step {step_str}/{total}   {DIM}·{RESET}   ~{sps:0.1f}s/step")
        row(f"   elapsed {fmt_dur(elapsed)}   {DIM}·{RESET}   "
            f"ETA {eta_str}   {DIM}·{RESET}   {fmt_dur(remain)} left")
    else:
        row("")
        row(f" {DIM}waiting for first logged step...{RESET}")

    # --- training metrics
    out.append(c("╟" + "─" * (W - 2) + "╢", "36"))
    if last:
        loss_str = c(f"{last['loss']:.4f}", "1;35")
        row(f" {BOLD}LOSS{RESET}      {loss_str}"
            f"   {DIM}grad_norm{RESET} {last['gn']:.3f}"
            f"   {DIM}lr{RESET} {last['lr']:.2e}"
            f"   {DIM}tok/s{RESET} {last['toks']:.0f}")
        spark = sparkline([r["loss"] for r in rows], 46)
        row(f"   {c(spark,'35')}")
        if len(rows) >= 2:
            row(f"   {DIM}first {rows[0]['loss']:.3f} → now {last['loss']:.3f}"
                f"  (Δ {rows[0]['loss']-last['loss']:+.3f}){RESET}")

    # --- hardware
    out.append(c("╟" + "─" * (W - 2) + "╢", "36"))
    row(f" {BOLD}JETSON HARDWARE{RESET}")
    if hw:
        # GPU
        gl = hw["gpu_load"] or 0
        gl_str = c(f"{gl:4.0f}%", util_color(gl))
        row(f"   GPU  {bar(gl/100, 28, util_color(gl))} {gl_str}"
            f"  {DIM}{hw['gpu_freq']:.0f}/{hw['gpu_fmax']:.0f}MHz{RESET}")
        # CPU (high CPU isn't bad here, keep neutral cyan scale via color_for off)
        if hw["cpu_load"] is not None:
            cl = hw["cpu_load"]
            cl_str = c(f"{cl:4.0f}%", "36")
            row(f"   CPU  {bar(cl/100, 28, '36')} {cl_str}")
        # RAM
        rf = hw["ram_used"] / hw["ram_tot"] if hw["ram_tot"] else 0
        ram_used_str = c(f"{hw['ram_used']:.1f}", color_for(rf * 100, 80, 93))
        row(f"   RAM  {bar(rf, 28, color_for(rf*100,80,93))} "
            f"{ram_used_str}/{hw['ram_tot']:.1f}GiB")
        # Temps
        t = hw["temps"]
        def tcol(v):
            return c(f"{v:.0f}°C", color_for(v, 70, 85)) if v is not None else "--"
        tline = (f"   {DIM}TEMP{RESET} "
                 f"cpu {tcol(t.get('cpu'))}  gpu {tcol(t.get('gpu'))}  "
                 f"tj {tcol(t.get('tj'))}  soc {tcol(t.get('soc0'))}")
        row(tline)
        # Power + fan
        fan = f"{hw['fan']:.0f}%" if isinstance(hw["fan"], (int, float)) else str(hw["fan"])
        power_str = c(f"{hw['power']:.1f}W", "1;33")
        row(f"   {DIM}POWER{RESET} {power_str} "
            f"{DIM}(avg {hw['power_avg']:.1f}W){RESET}   {DIM}FAN{RESET} {fan}")
    else:
        row(f"   {c('jtop service unavailable','1;31')}")

    out.append(c("╚" + "═" * (W - 2) + "╝", "36"))
    if info["runcfg"]:
        out.append(c(f" {info['runcfg']}", "2"))
    out.append(c(f" log: {log_path}   refresh: live   (Ctrl-C to quit)", "2"))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="outputs/gpt1_jetson_10h/training.log")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--once", action="store_true", help="render a single frame and exit")
    args = ap.parse_args()

    jet = None
    try:
        from jtop import jtop
        jet = jtop()
        jet.start()
    except Exception:
        jet = None

    try:
        if not args.once:
            sys.stdout.write(HIDE_CUR + CLEAR)
        while True:
            info = parse_log(args.log)
            hw = None
            try:
                hw = get_hw(jet)
            except Exception:
                hw = None
            frame = render(info, hw, args.log)
            if args.once:
                sys.stdout.write(frame + "\n")
                break
            sys.stdout.write(HOME + frame + CSI + "0J")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if not args.once:
            sys.stdout.write(SHOW_CUR + "\n")
            sys.stdout.flush()
        if jet is not None:
            try:
                jet.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
