"""
scripts/golden_set/sample_topup.py
──────────────────────────────────
Samples 50 rare-class rows from outside the 10k indexed corpus to prevent vector leakage.
When run: Executed prior to merge_topup.py.
Reproducibility: Repeatable sampling script.
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

import json
import random
import sys
from pathlib import Path

import pandas as pd

from data_prep import _filter_amazon, _load_csv, _reconstruct_threads  # noqa: E402
from intent_taxonomy import INTENT_BY_NAME  # noqa: E402

random.seed(123)

DEV_KWS = INTENT_BY_NAME["DEVICE_TECH_SUPPORT"].trigger_keywords
PRM_KWS = INTENT_BY_NAME["PRIME_SUBSCRIPTION"].trigger_keywords

def match(text, kws):
    t = text.lower()
    return any(k in t for k in kws)

def main() -> None:
    df = _load_csv()
    df_amazon = _filter_amazon(df)
    threads = _reconstruct_threads(df_amazon, max_threads=10**9)
    print(f"full pool threads: {len(threads)}")

    used = set()
    with open(ROOT / "data" / "processed/amazon_threads.jsonl", encoding="utf-8") as f:
        for line in f:
            used.add(str(json.loads(line)["thread_id"]))
    for csv_name in ("golden_250.csv", "heldout_50.csv", "heldoutB_50.csv",
                     "cal_100.csv", "topup_texts.csv"):
        p = Path("golden_set") / csv_name
        if p.exists():
            used.update(pd.read_csv(p)["thread_id"].astype(str).tolist())
    print(f"excluded known ids: {len(used)}")

    dev, prm = [], []
    random.shuffle(threads)
    N_DEV, N_PRM = 34, 16   # oversampled: keyword pools leak (~40% precision)
    for t in threads:
        if str(t.get("thread_id")) in used:
            continue
        msg = (t.get("first_customer_message") or "").strip()
        if not msg:
            continue
        if len(dev) < N_DEV and match(msg, DEV_KWS):
            dev.append(t)
        elif len(prm) < N_PRM and match(msg, PRM_KWS):
            prm.append(t)
        if len(dev) >= N_DEV and len(prm) >= N_PRM:
            break
    print(f"device candidates: {len(dev)}, prime candidates: {len(prm)}")

    rows = []
    for i, t in enumerate(dev + prm):
        rows.append({"message_id": f"GS_T{i+1:03d}",
                     "thread_id": t["thread_id"],
                     "text": " ".join(t["first_customer_message"].split())})
    pd.DataFrame(rows).to_csv(ROOT / "golden_set" / "topup_texts.csv", index=False)
    print(f"wrote {len(rows)} top-up texts")

if __name__ == "__main__":
    main()
