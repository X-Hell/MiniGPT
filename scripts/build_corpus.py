#!/usr/bin/env python3
"""
Scrape Wikipedia ML/AI articles via MediaWiki API and build a plain-prose corpus.
Output: data/train_data.txt  (~500K+ characters of clean English prose)
Documents are separated by \n---\n for EOS boundary markers.
"""

import urllib.request
import urllib.parse
import json
import re
import os
import time

TOPICS = [
    "Neural network (machine learning)",
    "Deep learning",
    "Machine learning",
    "Natural language processing",
    "Transformer (deep learning architecture)",
    "Attention (machine learning)",
    "Backpropagation",
    "Gradient descent",
    "Convolutional neural network",
    "Recurrent neural network",
    "Reinforcement learning",
    "Generative adversarial network",
    "Autoencoder",
    "Word embedding",
    "Tokenization (lexical analysis)",
    "Loss function",
    "Overfitting",
    "Regularization (mathematics)",
    "Batch normalization",
    "Dropout (neural networks)",
    "Transfer learning",
    "BERT (language model)",
    "GPT-3",
    "Language model",
    "Perceptron",
    "Activation function",
    "Stochastic gradient descent",
    "Adam (optimizer)",  # Note: might redirect
    "Self-supervised learning",
    "Few-shot learning (natural language processing)",
    "Artificial intelligence",
    "Turing test",
    "Knowledge representation and reasoning",
    "Bayesian inference",
    "Dimensionality reduction",
    "Principal component analysis",
    "Support vector machine",
    "Random forest",
    "Decision tree learning",
    "Logistic regression",
    "Artificial neural network",
    "Multilayer perceptron",
    "Long short-term memory",
    "Vanishing gradient problem",
    "Softmax function",
    "Cross-entropy",
    "Cosine similarity",
    "Word2vec",
    "Recurrent neural network",
    "Feedforward neural network",
]


def fetch_article(title: str) -> str:
    """Fetch plain text extract of a Wikipedia article."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "1",
        "exsectionformat": "plain",
        "format": "json",
    }
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)

    req = urllib.request.Request(url, headers={"User-Agent": "MiniGPT-CorpusBuilder/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    pages = data.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if page_id == "-1":
            return ""
        return page.get("extract", "")
    return ""


def clean_text(text: str) -> str:
    """Clean Wikipedia article text into plain prose."""
    # Remove section headers that are ALL CAPS or == styled ==
    text = re.sub(r"^=+\s*.*?\s*=+$", "", text, flags=re.MULTILINE)

    # Remove reference markers like [1], [23], [citation needed]
    text = re.sub(r"\[[\d,\s]+\]", "", text)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # ---- MATH / LATEX CLEANUP ----
    # Remove \displaystyle and similar LaTeX commands
    text = re.sub(r"\\displaystyle\b", "", text)
    text = re.sub(r"\\operatorname\{[^}]*\}", "", text)
    text = re.sub(r"\\text(rm|bf|it|tt|sf)?\{[^}]*\}", "", text)
    text = re.sub(r"\\(begin|end)\{[^}]*\}", "", text)
    text = re.sub(r"\\(frac|sqrt|sum|prod|int|lim|log|exp|sin|cos|tan|max|min|arg|sup|inf)\b", "", text)
    # Remove LaTeX commands like \mathbf, \mathrm, \boldsymbol, etc.
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    # Remove curly braces, subscripts, superscripts
    text = re.sub(r"[{}_^]", " ", text)
    # Remove standalone math operators and symbols
    text = re.sub(r"(?<!\w)[=+\-*/|<>≤≥≈∈∑∏∫]+(?!\w)", " ", text)
    # Remove leftover parentheses with only whitespace/symbols inside
    text = re.sub(r"\(\s*\)", "", text)
    # Collapse multiple spaces
    text = re.sub(r"  +", " ", text)

    # Remove lines that are just whitespace
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip very short lines (likely headers or stubs)
        if len(stripped) < 20:
            continue
        # Skip lines that look like list markers without substance
        if re.match(r"^[\d\.\-\*]+\s*$", stripped):
            continue
        # Skip lines that are >30% non-alpha (likely formulas)
        alpha_count = sum(1 for c in stripped if c.isalpha())
        if alpha_count < len(stripped) * 0.55:
            continue
        cleaned.append(stripped)

    return "\n".join(cleaned)


def main():
    os.makedirs("data", exist_ok=True)
    output_path = "data/train_data.txt"

    # Deduplicate topics
    seen = set()
    unique_topics = []
    for t in TOPICS:
        key = t.lower().strip()
        if key not in seen:
            seen.add(key)
            unique_topics.append(t)

    documents = []
    total_chars = 0

    for i, topic in enumerate(unique_topics):
        print(f"[{i+1}/{len(unique_topics)}] Fetching: {topic}...", end=" ")
        try:
            raw = fetch_article(topic)
            if not raw or len(raw) < 200:
                print(f"SKIP (too short: {len(raw)} chars)")
                continue
            cleaned = clean_text(raw)
            if len(cleaned) < 200:
                print(f"SKIP (cleaned too short: {len(cleaned)} chars)")
                continue
            documents.append(cleaned)
            total_chars += len(cleaned)
            print(f"OK ({len(cleaned):,} chars, total: {total_chars:,})")
        except Exception as e:
            print(f"ERROR: {e}")

        # Be polite to Wikipedia API
        time.sleep(0.5)

    # Join with document boundary marker
    corpus = "\n---\n".join(documents)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(corpus)

    word_count = len(corpus.split())
    print(f"\n=== Corpus Built ===")
    print(f"Articles: {len(documents)}")
    print(f"Characters: {total_chars:,}")
    print(f"Words: {word_count:,}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
