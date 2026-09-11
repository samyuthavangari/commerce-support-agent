"""Recompute headline metrics offline over golden-250 (no API).

Reads golden_250.csv (250 rows) + agent_replies.csv + judge_scores.csv,
rewrites eval_report.json metrics, confusion matrix PNG, and agreement.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from config import cfg  # noqa: E402
from eval.llm_judge import compute_agreement  # noqa: E402
from eval.metrics import (bootstrap_ci, compute_escalation_metrics,  # noqa: E402
                          compute_intent_metrics, plot_confusion_matrix,
                          run_baseline_comparison)


def main() -> None:
    golden = pd.read_csv("golden_set/golden_250.csv")
    replies = pd.read_csv("results/agent_replies.csv")
    assert len(golden) == len(replies) == 250
    m = replies.set_index("message_id")
    assert (m.loc[golden["message_id"], "true_intent"].tolist()
            == golden["true_intent"].tolist())

    y_true = golden["true_intent"].tolist()
    y_pred = replies.set_index("message_id").loc[
        golden["message_id"], "pred_intent"].tolist()
    intent_metrics = compute_intent_metrics(y_true, y_pred)
    try:
        plot_confusion_matrix(intent_metrics,
                              output_path=cfg.results_dir / "confusion_matrix.png")
    except OSError as exc:
        print(f"! confusion matrix locked ({exc}) — rerun later")

    esc_true = golden["escalate_yn"].astype(bool).tolist()
    esc_pred = (replies.set_index("message_id").loc[
        golden["message_id"], "escalation_decision"] == "escalate").tolist()
    esc_metrics = compute_escalation_metrics(esc_true, esc_pred)
    baseline_metrics = run_baseline_comparison(golden, y_pred)

    judge = pd.read_csv("results/judge_scores.csv")
    judge_metrics = {
        "mean_total": round(float(judge["total"].mean()), 2),
        "std_total": round(float(judge["total"].std()), 2),
        "mean_per_dimension": {
            d: round(float(judge[d].mean()), 2)
            for d in ["groundedness", "helpfulness", "empathy", "safety", "brevity"]},
    }
    agreement = {}
    if Path("results/human_scores.csv").exists():
        agreement = compute_agreement(judge, Path("results/human_scores.csv"))
        agreement["n"] = 40

    rep = json.load(open("results/eval_report.json"))
    rep.update({
        "num_examples": 250,
        "intent_classification": intent_metrics,
        "intent_ci": {
            "accuracy_95": bootstrap_ci(y_true, y_pred, "accuracy"),
            "macro_f1_95": bootstrap_ci(y_true, y_pred, "macro_f1")},
        "escalation": esc_metrics,
        "escalation_ci": {
            "recall_95": bootstrap_ci(esc_true, esc_pred, "recall")},
        "baselines": baseline_metrics,
        "reply_quality_judge": judge_metrics,
        "human_judge_agreement": agreement,
    })
    json.dump(rep, open("results/eval_report.json", "w"), indent=2, allow_nan=False)
    print("intent:", intent_metrics["accuracy"], intent_metrics["macro_f1"])
    print("esc:", {k: esc_metrics[k] for k in ("precision", "recall", "f1", "false_auto_handle_rate")})
    b = baseline_metrics
    print("B0:", b["trivial_b0"]["accuracy"], round(b["trivial_b0"]["macro_f1"], 4),
          "| B1:", b["tfidf_b1"]["accuracy"], round(b["tfidf_b1"]["macro_f1"], 4))
    print("judge mean:", judge_metrics["mean_total"])


if __name__ == "__main__":
    main()
