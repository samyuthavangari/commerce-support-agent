"""Human escalation labels for CAL2-100 (frozen 7-rule definition).

Labelled row-by-row, blind to rule output, in the same session as cal-100
intent labels. Cal-2 threads were never seen during v1–v4 rule development.
Rule IDs: 1 legal · 2 fraud/security · 3 safety · 4 public PII/order-ID ·
5 repeat-contact failure · 6 explicit human-ask · 7 uninterpretable.
"""

import pandas as pd

CAL2_TRUE = {
    "C2_0001": 5, "C2_0003": 5, "C2_0009": 7, "C2_0010": 2, "C2_0011": 5,
    "C2_0015": 5, "C2_0016": 5, "C2_0019": 4, "C2_0022": 2, "C2_0026": 5,
    "C2_0030": 5, "C2_0032": 5, "C2_0033": 5, "C2_0034": 5, "C2_0036": 4,
    "C2_0040": 7, "C2_0042": 5, "C2_0055": 4, "C2_0058": 4, "C2_0060": 7,
    "C2_0064": 5, "C2_0078": 5, "C2_0079": 5, "C2_0083": 2, "C2_0084": 5,
    "C2_0088": 5, "C2_0094": 5, "C2_0095": 5, "C2_0096": 5,
}


def main() -> None:
    df = pd.read_csv("golden_set/cal2_texts.csv")
    assert len(df) == 100
    df["escalate_yn"] = df["message_id"].isin(CAL2_TRUE)
    df["escalate_rule"] = df["message_id"].map(lambda m: CAL2_TRUE.get(m, ""))
    df["labelled_by"] = "human_cal2_session_v1"
    df["human_verified"] = True
    df.to_csv("golden_set/cal2_100.csv", index=False)
    n = int(df["escalate_yn"].sum())
    print(f"cal2: {len(df)} rows, {n} positives ({n/len(df):.2f})")


if __name__ == "__main__":
    main()
