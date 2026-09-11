"""
scripts/checks/boundary_eval.py
───────────────────────────────
Evaluates class boundary confusion, specifically PRIME_SUBSCRIPTION vs ACCOUNT_ACCESS.
When run: Executed to diagnose classifier confusion bottlenecks.
Reproducibility: Repeatable diagnostic script.
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

from intent_taxonomy import INTENT_BY_NAME  # noqa: E402

PAIRS = [
    ("ORDER_STATUS", "DELIVERY_DAMAGE"),
    ("PRIME_SUBSCRIPTION", "ACCOUNT_ACCESS"),
    ("ORDER_STATUS", "GENERAL_COMPLAINT"),
    ("ACCOUNT_ACCESS", "GENERAL_COMPLAINT"),
]

def dual_match(text, a, b):
    t = text.lower()
    ka = INTENT_BY_NAME[a].trigger_keywords
    kb = INTENT_BY_NAME[b].trigger_keywords
    return any(k in t for k in ka) and any(k in t for k in kb)

def main() -> None:
    golden = pd.read_csv(ROOT / "golden_set" / "golden_250.csv")
    replies = pd.read_csv(ROOT / "results" / "agent_replies.csv").set_index("message_id")
    golden["pred"] = golden["message_id"].map(
        lambda m: replies.loc[m, "pred_intent"])
    print("pair                        n   acc")
    for a, b in PAIRS:
        sub = golden[golden["true_intent"].isin([a, b])]
        acc = (sub["pred"] == sub["true_intent"]).mean()
        print(f"{a[:12]:12} vs {b[:12]:12} {len(sub):3d} {acc:.3f}")
        for _, r in sub[sub["pred"] != sub["true_intent"]].head(6).iterrows():
            print(f"    miss {r['message_id']}: true={r['true_intent']} "
                  f"pred={r['pred']} | {' '.join(str(r['text']).split())[:100]}")

if __name__ == "__main__":
    main()
