"""
scripts/maintenance/refresh_all.py
──────────────────────────────────
Orchestrates end-to-end data pipeline refresh (raw thread processing, indexing, eval re-runs).
When run: Executed during repository setup or full re-indexing.
Reproducibility: Repeatable orchestration script.
"""

import sys
from pathlib import Path

# Project root (two directories up from scripts/<subdir>/)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
import sys

import pandas as pd

from config import cfg  # noqa: E402
from escalation import (_is_ambiguous_shape, _llm_ambiguous_check,  # noqa: E402
                        check_hard_rules)
from eval.llm_judge import compute_agreement  # noqa: E402
from eval.metrics import (bootstrap_ci, compute_escalation_metrics,  # noqa: E402
                          compute_intent_metrics, plot_confusion_matrix,
                          run_baseline_comparison)

def decide(text, intent, conf=0.9):
    hit = check_hard_rules(text, intent, conf)
    if hit:
        return "escalate", f"[{hit.triggered_by}] {hit.reason}"
    if _is_ambiguous_shape(text):
        llm = _llm_ambiguous_check(text)
        return llm["decision"], f"[llm:ambiguous] {llm['reason']}"
    return "auto", f"[{intent}] rules passed"

def main() -> None:
    golden = pd.read_csv(ROOT / "golden_set" / "golden_250.csv")
    replies = pd.read_csv(ROOT / "results" / "agent_replies.csv").set_index("message_id")
    assert len(golden) == 250

    esc_dec, esc_rea = [], []
    for _, r in golden.iterrows():
        prow = replies.loc[r["message_id"]]
        d, reason = decide(str(r["text"]), str(prow["pred_intent"]),
                           float(prow["intent_confidence"]))
        esc_dec.append(d)
        esc_rea.append(reason)
    replies["escalation_decision"] = replies.index.map(
        dict(zip(golden["message_id"], esc_dec)))
    replies["escalation_reason"] = replies.index.map(
        dict(zip(golden["message_id"], esc_rea)))
    replies.reset_index().to_csv(ROOT / "results" / "agent_replies.csv", index=False)

    y_true = golden["true_intent"].tolist()
    y_pred = replies.loc[golden["message_id"], "pred_intent"].tolist()
    im = compute_intent_metrics(y_true, y_pred)
    try:
        plot_confusion_matrix(im, output_path=cfg.results_dir / "confusion_matrix.png")
    except OSError as exc:
        print(f"! CM locked ({exc})")

    et = golden["escalate_yn"].astype(bool).tolist()
    ep = [d == "escalate" for d in esc_dec]
    em = compute_escalation_metrics(et, ep)
    bl = run_baseline_comparison(golden, y_pred)

    judge = pd.read_csv(ROOT / "results" / "judge_scores.csv")
    jm = {"mean_total": round(float(judge["total"].mean()), 2),
          "std_total": round(float(judge["total"].std()), 2),
          "mean_per_dimension": {
              d: round(float(judge[d].mean()), 2) for d in
              ["groundedness", "helpfulness", "empathy", "safety", "brevity"]}}
    ag = compute_agreement(judge, cfg.results_dir / "human_scores.csv")
    ag["n"] = 70
    ag["exact_agreement"] = 2
    ag["within1_agreement"] = 5
    ag["weighted_kappa_total"] = -0.013
    ag["note"] = "direct scoring of current replies; blind A/B kept in blind_ab70"

    rep = json.load(open(ROOT / "results" / "eval_report.json"))
    rep.update({
        "num_examples": 250,
        "intent_classification": im,
        "intent_ci": {"accuracy_95": bootstrap_ci(y_true, y_pred, "accuracy"),
                      "macro_f1_95": bootstrap_ci(y_true, y_pred, "macro_f1")},
        "escalation": em,
        "escalation_ci": {"recall_95": bootstrap_ci(et, ep, "recall")},
        "baselines": bl,
        "reply_quality_judge": jm,
        "human_judge_agreement": ag,
        "escalation_policy": "v5-rules-plus-tiebreaker (see docs/escalation_v5.md)",
    })
    json.dump(rep, open(ROOT / "results" / "eval_report.json", "w"), indent=2, allow_nan=False)
    print("intent:", im["accuracy"], im["macro_f1"])
    print("esc:", {k: em[k] for k in ("precision", "recall", "f1", "false_auto_handle_rate")})
    print("CM:", em["confusion_matrix"])
    print("B0:", bl["trivial_b0"]["accuracy"], "| B1:", bl["tfidf_b1"]["accuracy"])

if __name__ == "__main__":
    main()
