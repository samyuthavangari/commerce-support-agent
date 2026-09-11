"""Sample calibration-100 + heldout-B-50 from threads excluding golden/heldout-A.

cal-100: stratified-ish (30 hinted + 70 plain) — the ONLY set rules may be
tuned against from here on.
heldout-B-50: same recipe as heldout-A (30 hinted + 20 plain), frozen on arrival.
"""

import json
import random

import pandas as pd

random.seed(99)

HINTS = [
    "contacted", "called", "twice", "thrice", "still no", "no update",
    "fraud", "dispute", "chargeback", "unauthorized", "hacked", "cheat",
    "sued", "lawsuit", "lawyer", "refund", "real person", "human",
    "supervisor", "robot", "helpline", "helpless", "ignoring",
    "405-", "406-", "407-", "408-", "171-", "order id", "order no",
    "order #", "apology", "weeks", "tired", "spoke", "charged twice",
    "heard back",
]

threads = [json.loads(line) for line in
           open("data/processed/amazon_threads.jsonl", encoding="utf-8")]
used = (set(pd.read_csv("golden_set/golden_250.csv")["thread_id"].astype(str))
        | set(pd.read_csv("golden_set/heldout_50.csv")["thread_id"].astype(str)))
pool = [t for t in threads if str(t.get("thread_id")) not in used
        and (t.get("first_customer_message") or "").strip()]
print(f"pool: {len(pool)}")

hinted = [t for t in pool if any(h.lower() in
          t["first_customer_message"].lower() for h in HINTS)]
plain = [t for t in pool if t not in hinted]
random.shuffle(hinted)
random.shuffle(plain)

cal = hinted[:30] + plain[:70]
rest_h = hinted[30:]
rest_p = [t for t in plain[70:] if t not in cal]
hb = rest_h[:30] + rest_p[:20]
random.shuffle(cal)
random.shuffle(hb)


def dump(rows, prefix):
    out = [{"message_id": f"{prefix}_{i+1:04d}", "thread_id": t["thread_id"],
            "text": " ".join(t["first_customer_message"].split())} for i, t in enumerate(rows)]
    pd.DataFrame(out).to_csv(f"golden_set/{prefix.lower()}_texts.csv", index=False)
    print(f"{prefix}: {len(out)}")


dump(cal, "CAL")
dump(hb, "HB")
