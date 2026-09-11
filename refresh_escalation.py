"""Refresh escalation columns after the v4 policy freeze (offline, exact).

decide_escalation() is fully deterministic, so re-running it over the stored
golden rows reproduces the live path bit-for-bit (thread_length=1, same as
run_agent). Updates agent_replies.csv + eval_report.json escalation block.
No API calls.
"""

import json
import sys

import pandas as pd

sys.path.insert(0, "src")
from eval.metrics import compute_escalation_metrics  # noqa: E402
from escalation import decide_escalation  # noqa: E402


def main() -> None:
    golden = pd.read_csv("golden_set/golden_250.csv").set_index("message_id")
    replies = pd.read_csv("results/agent_replies.csv").set_index("message_id")
    assert len(replies) == len(golden) == 200

    for mid, row in replies.iterrows():
        dec = decide_escalation(str(golden.loc[mid, "text"]),
                                str(row["pred_intent"]),
                                float(row["intent_confidence"]))
        replies.loc[mid, "escalation_decision"] = dec.decision
        replies.loc[mid, "escalation_reason"] = (
            f"[{dec.triggered_by}] {dec.reason}")
    replies.reset_index().to_csv("results/agent_replies.csv", index=False)

    y_true = golden["escalate_yn"].astype(bool).tolist()
    y_pred = (replies["escalation_decision"] == "escalate").tolist()
    esc = compute_escalation_metrics(y_true, y_pred)
    print("refreshed escalation:", esc)

    rep = json.load(open("results/eval_report.json"))
    rep["escalation"] = esc
    rep["escalation_policy"] = "v4-deterministic-rules (see docs/escalation_v4.md)"
    json.dump(rep, open("results/eval_report.json", "w"), indent=2,
              allow_nan=False)
    print("eval_report.json updated")


if __name__ == "__main__":
    main()
