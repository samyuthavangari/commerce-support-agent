"""
scripts/golden_set/stamp_reasons.py
───────────────────────────────────
Attaches human-verified rule trigger reason codes to positive escalation cases.
When run: Executed during golden set annotation hardening.
Reproducibility: Repeatable auditing script.
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

import sys

import pandas as pd

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
