"""Fill escalate_reason on human-escalated rows with the firing rule trigger.

Human review flipped escalate_yn without touching the LLM's reason text,
leaving True rows with empty reasons. This stamps each human-True row with
the deterministic rule that fires on it (audit trail for the 7-rule
definition). Idempotent: skips rows already stamped.
"""

import sys

import pandas as pd

sys.path.insert(0, "src")
from escalation import check_hard_rules  # noqa: E402

STAMP = "[human_review_v1:"


def main() -> None:
    path = "golden_set/golden_250.csv"
    df = pd.read_csv(path)
    n = 0
    for idx, row in df.iterrows():
        if not bool(row["escalate_yn"]):
            continue
        reason = "" if pd.isna(row["escalate_reason"]) else str(row["escalate_reason"])
        if STAMP in reason:
            continue
        hit = check_hard_rules(str(row["text"]), str(row["true_intent"]), 0.9)
        trigger = hit.triggered_by if hit else "reviewer-judgment"
        df.loc[idx, "escalate_reason"] = f"{STAMP}{trigger}] " + reason
        n += 1
    df.to_csv(path, index=False)
    print(f"Stamped {n} rows; sample:")
    sub = df[df["escalate_yn"] == True][["message_id", "escalate_reason"]].head(8)
    print(sub.to_string())


if __name__ == "__main__":
    main()
