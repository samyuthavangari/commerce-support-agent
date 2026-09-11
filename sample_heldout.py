"""Sample 50 held-out threads for escalation validation.

Excludes every thread_id already in golden_250.csv. Oversamples
repeat/fraud/PII-ish texts 2x so the 50-row set contains enough positives
to measure recall (else ~12 positives by base rate).
"""

import json
import random

import pandas as pd

random.seed(7)

HINTS = [
    "contacted", "called", "twice", "thrice", "still no", "no update",
    "fraud", "dispute", "chargeback", "unauthorized", "hacked", "cheat",
    "sued", "lawsuit", "lawyer", "refund", "real person", "human",
    "supervisor", "robot", " Rant", "helpless", "ignoring",
    "405-", "406-", "407-", "408-", "171-", "order id", "order no",
    "order #", "apology", "weeks", "tired",
]

threads = [json.loads(line) for line in
           open("data/processed/amazon_threads.jsonl", encoding="utf-8")]
golden_ids = set(pd.read_csv("golden_set/golden_250.csv")["thread_id"].astype(str))
pool = [t for t in threads if str(t.get("thread_id")) not in golden_ids
        and (t.get("first_customer_message") or "").strip()]
print(f"held-out pool: {len(pool)} threads")

hinted = [t for t in pool if any(h.lower() in
          t["first_customer_message"].lower() for h in HINTS)]
plain = [t for t in pool if t not in hinted]
random.shuffle(hinted)
random.shuffle(plain)
sel = (hinted[:30] + plain[:20])
random.shuffle(sel)

rows = [{"message_id": f"H_{i+1:04d}", "thread_id": t["thread_id"],
         "text": " ".join(t["first_customer_message"].split())}
        for i, t in enumerate(sel)]
pd.DataFrame(rows).to_csv("golden_set/heldout_50_texts.csv", index=False)
print(f"wrote {len(rows)} texts ({len(hinted[:30])} hinted + 20 plain)")
