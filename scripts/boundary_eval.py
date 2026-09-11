"""Boundary-slice evaluation (offline): accuracy on the 4 noisy intent pairs.

Design note: an earlier revision filled slices errors-first, which measures
error concentration, not pair accuracy (it scored ~0.00 by construction).
This version reports accuracy over ALL golden rows in each pair — unbiased.
Miss examples are still listed for analysis. Uses stored predictions — no API.
"""

import sys

import pandas as pd

sys.path.insert(0, "src")
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
    golden = pd.read_csv("golden_set/golden_250.csv")
    replies = pd.read_csv("results/agent_replies.csv").set_index("message_id")
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
