"""
src/eval/metrics.py
────────────────────
All classification and escalation metrics for the evaluation harness.
Three baselines are provided:
  B0 — Trivial : always predict ORDER_STATUS
  B1 — TF-IDF  : TF-IDF + Logistic Regression (keyword-based)
  Agent        : our Gemini + Qdrant RAG pipeline
"""

import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for server/CI)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from intent_taxonomy import INTENT_NAMES  # noqa: E402

console = Console()


# ── Intent classification metrics ─────────────────────────────────────────

def compute_intent_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: Optional[list[str]] = None,
) -> dict:
    """
    Compute full intent classification metrics.

    Returns:
        dict with: accuracy, macro_f1, weighted_f1, per_class (dict), confusion_matrix (list), labels
    """
    if labels is None:
        labels = INTENT_NAMES

    accuracy     = accuracy_score(y_true, y_pred)
    macro_f1     = f1_score(y_true, y_pred, average="macro",    labels=labels, zero_division=0)
    weighted_f1  = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    cm           = confusion_matrix(y_true, y_pred, labels=labels)

    per_class: dict[str, dict] = {}
    for label in labels:
        p = precision_score(y_true, y_pred, labels=[label], average="macro", zero_division=0)
        r = recall_score   (y_true, y_pred, labels=[label], average="macro", zero_division=0)
        f = f1_score       (y_true, y_pred, labels=[label], average="macro", zero_division=0)
        per_class[label] = {
            "precision": round(p, 3),
            "recall":    round(r, 3),
            "f1":        round(f, 3),
        }

    return {
        "accuracy":         round(accuracy,    4),
        "macro_f1":         round(macro_f1,    4),
        "weighted_f1":      round(weighted_f1, 4),
        "per_class":        per_class,
        "confusion_matrix": cm.tolist(),
        "labels":           labels,
    }


def compute_escalation_metrics(
    y_true: list[bool],
    y_pred: list[bool],
) -> dict:
    """
    Compute binary escalation metrics.
    Recall is the most safety-critical metric (never miss an escalation).
    false_auto_handle_rate = share of should-escalate cases the system
    auto-handled (the dangerous error; = 1 - recall). Reported by name.
    """
    p   = precision_score(y_true, y_pred, zero_division=0)
    r   = recall_score   (y_true, y_pred, zero_division=0)
    f   = f1_score       (y_true, y_pred, zero_division=0)
    acc = accuracy_score (y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=[False, True]).tolist()
    return {
        "accuracy":  round(acc, 4),
        "precision": round(p,   4),
        "recall":    round(r,   4),
        "f1":        round(f,   4),
        "false_auto_handle_rate": round(1.0 - r, 4),
        "confusion_matrix": cm,   # [[tn, fp], [fn, tp]]
        "labels": ["auto_handle", "escalate"],
    }


def bootstrap_ci(
    y_true: list,
    y_pred: list,
    metric: str = "accuracy",
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Nonparametric bootstrap 95% CI for a headline metric (numpy only).
    metric: "accuracy" | "macro_f1" | "recall" (escalate class).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        if metric == "accuracy":
            vals.append(accuracy_score(yt, yp))
        elif metric == "macro_f1":
            vals.append(f1_score(yt, yp, average="macro", zero_division=0))
        elif metric == "recall":
            vals.append(recall_score(yt, yp, zero_division=0))
        else:
            raise ValueError(f"unknown metric: {metric}")
    lo, hi = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    return {"metric": metric, "n": n, "n_boot": n_boot,
            "lo_95": round(lo, 4), "hi_95": round(hi, 4)}


# ── Console display ────────────────────────────────────────────────────────

def print_intent_table(metrics: dict) -> None:
    """Print per-class classification metrics as a Rich table."""
    table = Table(title="Intent Classification — Per-Class Metrics", show_header=True)
    table.add_column("Intent",    style="cyan",  min_width=26)
    table.add_column("Precision", justify="right")
    table.add_column("Recall",    justify="right")
    table.add_column("F1",        justify="right")

    for intent, scores in metrics["per_class"].items():
        table.add_row(
            intent,
            f"{scores['precision']:.3f}",
            f"{scores['recall']:.3f}",
            f"{scores['f1']:.3f}",
        )

    console.print(table)
    console.print(
        f"  Accuracy : [bold]{metrics['accuracy']:.4f}[/bold] | "
        f"Macro F1 : [bold]{metrics['macro_f1']:.4f}[/bold] | "
        f"Weighted F1 : [bold]{metrics['weighted_f1']:.4f}[/bold]"
    )


# ── Confusion matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix(
    metrics:     dict,
    output_path: Optional[Path] = None,
) -> None:
    """Save and/or display an annotated intent confusion matrix heatmap."""
    cm     = np.array(metrics["confusion_matrix"])
    labels = metrics["labels"]

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm,
        annot    = True,
        fmt      = "d",
        cmap     = "Blues",
        xticklabels = [l.replace("_", "\n") for l in labels],
        yticklabels = [l.replace("_", "\n") for l in labels],
        ax       = ax,
        linewidths = 0.5,
    )
    ax.set_xlabel("Predicted Intent", fontsize=12, labelpad=10)
    ax.set_ylabel("True Intent",      fontsize=12, labelpad=10)
    ax.set_title("Intent Classification — Confusion Matrix", fontsize=14, pad=15)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        console.print(f"[green]OK Confusion matrix -> {output_path}[/green]")

    plt.close(fig)


# ── Baselines ──────────────────────────────────────────────────────────────

def _trivial_baseline(n: int) -> list[str]:
    """B0: Always predict ORDER_STATUS (majority class heuristic)."""
    return ["ORDER_STATUS"] * n


def _heuristic_label(text: str) -> str:
    """Keyword-vote intent label (same signal as the retrieval index builder)."""
    from intent_taxonomy import INTENT_BY_NAME, INTENT_NAMES
    t = text.lower()
    best, best_score = "GENERAL_COMPLAINT", 0
    for name in INTENT_NAMES:
        score = sum(1 for kw in INTENT_BY_NAME[name].trigger_keywords if kw in t)
        if score > best_score:
            best_score, best_name = score, name
            best = best_name
    return best


def _tfidf_baseline(
    texts:  list[str],
    labels: list[str],
) -> list[str]:
    """
    B1: TF-IDF bigrams + Logistic Regression, trained FAIRLY.

    Training data = weak-labelled (keyword-heuristic) non-golden threads;
    the golden texts are TEST-ONLY. An earlier revision fit on the golden
    texts themselves (train=test contamination, 0.92 accuracy) — that number
    is withdrawn; see README "misleading" section.
    """
    import json
    import random
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    if len(set(labels)) < 2:
        return list(labels)

    threads_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "amazon_threads.jsonl"
    golden_dir = Path(__file__).resolve().parents[2] / "golden_set"
    eval_ids: set[str] = set()
    for csv_name in ("golden_250.csv", "heldout_50.csv"):
        p = golden_dir / csv_name
        if p.exists():
            eval_ids.update(pd.read_csv(p)["thread_id"].astype(str).tolist())

    X_train, y_train = [], []
    random.seed(42)
    with open(threads_path, encoding="utf-8") as f:
        pool = [json.loads(line) for line in f]
    random.shuffle(pool)
    for t in pool:
        if str(t.get("thread_id")) in eval_ids:
            continue
        msg = (t.get("first_customer_message") or "").strip()
        if not msg:
            continue
        X_train.append(msg)
        y_train.append(_heuristic_label(msg))
        if len(X_train) >= 8000:
            break

    clf = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=8_000, sublinear_tf=True)),
            ("lr",    LogisticRegression(C=1.0, max_iter=1_000, random_state=42)),
        ]
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(texts).tolist()
    provenance = {
        "source": "amazon_threads.jsonl first_customer_message",
        "n_train": len(X_train),
        "labels": "keyword-heuristic (never golden/heldout human labels)",
        "excluded_eval_threads": len(eval_ids),
        "seed": 42,
        "vectorizer": "TfidfVectorizer(1-2gram, 8000, sublinear)",
        "model": "LogisticRegression(C=1.0, max_iter=1000)",
    }
    return preds, provenance


def run_baseline_comparison(
    golden_df:        pd.DataFrame,
    agent_predictions: list[str],
) -> dict:
    """
    Compare B0 (trivial), B1 (TF-IDF), and our agent on the golden set.

    Returns:
        dict with keys: trivial_b0, tfidf_b1, agent — each is a metrics dict
    """
    y_true = golden_df["true_intent"].tolist()
    texts  = golden_df["text"].tolist()
    n      = len(y_true)

    b0_preds = _trivial_baseline(n)
    b1_preds, b1_prov = _tfidf_baseline(texts, y_true)

    b0 = compute_intent_metrics(y_true, b0_preds)
    b1 = compute_intent_metrics(y_true, b1_preds)
    ag = compute_intent_metrics(y_true, agent_predictions)

    # Print comparison table
    table = Table(title="Baseline Comparison", show_header=True)
    table.add_column("System",    style="cyan", min_width=30)
    table.add_column("Accuracy",  justify="right")
    table.add_column("Macro F1",  justify="right")

    table.add_row("B0: Trivial (always ORDER_STATUS)",
                  f"{b0['accuracy']:.3f}", f"{b0['macro_f1']:.3f}")
    table.add_row("B1: TF-IDF + Logistic Regression",
                  f"{b1['accuracy']:.3f}", f"{b1['macro_f1']:.3f}")
    table.add_row("[bold]Our Agent (Gemini + Qdrant RAG)[/bold]",
                  f"[bold]{ag['accuracy']:.3f}[/bold]",
                  f"[bold]{ag['macro_f1']:.3f}[/bold]")
    console.print(table)

    return {
        "trivial_b0": b0,
        "tfidf_b1":   b1,
        "b1_training": b1_prov,
        "agent":      ag,
    }


# ── Smoke-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    y_t = ["ORDER_STATUS", "RETURN_REFUND", "ACCOUNT_ACCESS", "ORDER_STATUS"]
    y_p = ["ORDER_STATUS", "RETURN_REFUND", "ORDER_STATUS",   "ORDER_STATUS"]
    m = compute_intent_metrics(y_t, y_p, labels=["ORDER_STATUS", "RETURN_REFUND", "ACCOUNT_ACCESS"])
    print_intent_table(m)
    print(compute_escalation_metrics([True, False, True, False], [True, False, False, True]))
