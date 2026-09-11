"""
scripts/eval/eval_4set.py
─────────────────────────
Computes deterministic escalation metrics across 4 splits (cal, golden, heldout A, heldout B).
When run: Executed during rule development to evaluate generalization.
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

def main() -> None:
    print("set              P      R      falseauto  (tp fp fn npos)")
    for f in [str(ROOT / "golden_set" / "cal_100.csv"), str(ROOT / "golden_set" / "golden_250.csv"),
              str(ROOT / "golden_set" / "heldout_50.csv"), str(ROOT / "golden_set" / "heldoutB_50.csv")]:
        df = pd.read_csv(f)
        y = df["escalate_yn"].astype(bool).tolist()
        p = [check_hard_rules(str(t), "GENERAL_COMPLAINT", 0.9) is not None
             for t in df["text"].astype(str)]
        tp = sum(1 for a, b in zip(y, p) if a and b)
        fp = sum(1 for a, b in zip(y, p) if not a and b)
        fn = sum(1 for a, b in zip(y, p) if a and not b)
        name = f.split("/")[1].ljust(14)
        print(f"{name} P={tp/max(tp+fp,1):.3f} R={tp/max(tp+fn,1):.3f} "
              f"fa={fn/max(tp+fn,1):.3f}  ({tp} {fp} {fn} {sum(y)})")
        mids = df["message_id"].tolist()
        print("  FN:", [m for m, a, b in zip(mids, y, p) if a and not b])
        print("  FP:", [m for m, a, b in zip(mids, y, p) if b and not a])

if __name__ == "__main__":
    main()
