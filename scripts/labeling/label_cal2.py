"""
scripts/labeling/label_cal2.py
──────────────────────────────
Generates human escalation ground truth for CAL2-100 calibration set (golden_set/cal2_100.csv).
When run: Executed prior to v5 rule development to create an untouched tuning split.
Reproducibility: Repeatable dataset generation script.
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
    df = pd.read_csv(ROOT / "golden_set" / "cal2_texts.csv")
    assert len(df) == 100
    df["escalate_yn"] = df["message_id"].isin(CAL2_TRUE)
    df["escalate_rule"] = df["message_id"].map(lambda m: CAL2_TRUE.get(m, ""))
    df["labelled_by"] = "human_cal2_session_v1"
    df["human_verified"] = True
    df.to_csv(ROOT / "golden_set" / "cal2_100.csv", index=False)
    n = int(df["escalate_yn"].sum())
    print(f"cal2: {len(df)} rows, {n} positives ({n/len(df):.2f})")

if __name__ == "__main__":
    main()
