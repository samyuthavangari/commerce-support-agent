"""Definitive escalation table over all 5 labelled sets (single careful pass).

Prints per-set P/R/false-auto plus trigger attribution for every miss.
The LLM tiebreaker (temp 0.0) can still wobble ±1 row between runs;
re-run twice before quoting.
"""

import sys

import pandas as pd

sys.path.insert(0, "src")
from escalation import check_hard_rules  # noqa: E402

FILES = [
    "golden_set/cal_100.csv",
    "golden_set/cal2_100.csv",
    "golden_set/golden_250.csv",
    "golden_set/heldout_50.csv",
    "golden_set/heldoutB_50.csv",
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
