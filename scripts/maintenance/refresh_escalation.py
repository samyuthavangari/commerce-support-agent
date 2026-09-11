"""
scripts/maintenance/refresh_escalation.py
─────────────────────────────────────────
Re-runs escalation evaluation across all datasets and outputs summary metrics.
When run: Executed whenever rules in src/escalation.py are updated.
Reproducibility: Repeatable verification script.
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

from eval.metrics import compute_escalation_metrics  # noqa: E402
from escalation import decide_escalation  # noqa: E402

def main() -> None:
    golden = pd.read_csv(ROOT / "golden_set" / "golden_250.csv").set_index("message_id")
    replies = pd.read_csv(ROOT / "results" / "agent_replies.csv").set_index("message_id")
    assert len(replies) == len(golden) == 200

    for mid, row in replies.iterrows():
        dec = decide_escalation(str(golden.loc[mid, "text"]),
                                str(row["pred_intent"]),
                                float(row["intent_confidence"]))
        replies.loc[mid, "escalation_decision"] = dec.decision
        replies.loc[mid, "escalation_reason"] = (
            f"[{dec.triggered_by}] {dec.reason}")
    replies.reset_index().to_csv(ROOT / "results" / "agent_replies.csv", index=False)

    y_true = golden["escalate_yn"].astype(bool).tolist()
    y_pred = (replies["escalation_decision"] == "escalate").tolist()
    esc = compute_escalation_metrics(y_true, y_pred)
    print("refreshed escalation:", esc)

    rep = json.load(open(ROOT / "results" / "eval_report.json"))
    rep["escalation"] = esc
    rep["escalation_policy"] = "v4-deterministic-rules (see docs/escalation_v4.md)"
    json.dump(rep, open(ROOT / "results" / "eval_report.json", "w"), indent=2,
              allow_nan=False)
    print("eval_report.json updated")

if __name__ == "__main__":
    main()
