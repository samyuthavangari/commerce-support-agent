"""
scripts/eval/eval_5set.py
─────────────────────────
Definitive escalation evaluation table over all 5 labelled sets (cal, cal2, golden, heldout A, heldout B).
When run: Executed to benchmark escalation precision, recall, and false-auto rates.
Reproducibility: Repeatable evaluation script.
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

FILES = [
    str(ROOT / "golden_set" / "cal_100.csv"),
    str(ROOT / "golden_set" / "cal2_100.csv"),
    str(ROOT / "golden_set" / "golden_250.csv"),
    str(ROOT / "golden_set" / "heldout_50.csv"),
    str(ROOT / "golden_set" / "heldoutB_50.csv"),
]

def main() -> None:
    for f in FILES:
        df = pd.read_csv(f).reset_index(drop=True)
        y = df["escalate_yn"].astype(bool).tolist()
        out = [check_hard_rules(str(t), "GENERAL_COMPLAINT", 0.9)
               for t in df["text"].astype(str)]
        p = [o is not None for o in out]
        tp = sum(1 for a, b in zip(y, p) if a and b)
        fp = sum(1 for a, b in zip(y, p) if not a and b)
        fn = sum(1 for a, b in zip(y, p) if a and not b)
        print(f"{f.split('/')[1]:18s} P={tp/max(tp+fp,1):.3f} "
              f"R={tp/max(tp+fn,1):.3f} fa={fn/max(tp+fn,1):.3f} "
              f"(tp={tp} fp={fp} fn={fn} npos={sum(y)})")
        for i, (a, b) in enumerate(zip(y, p)):
            mid = df.loc[i, "message_id"]
            if a and not b:
                print(f"    FN {mid}")
            elif b and not a:
                print(f"    FP {mid} [{out[i].triggered_by}]")

if __name__ == "__main__":
    main()
