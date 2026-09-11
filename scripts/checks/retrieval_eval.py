"""
scripts/checks/retrieval_eval.py
────────────────────────────────
Measures semantic retrieval Hit@K and intent consistency across candidate depths.
When run: Executed to validate intent-filtered retrieval vs unfiltered baseline.
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

import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from qdrant_store import get_client, retrieve  # noqa: E402


def main() -> None:
    golden = pd.read_csv(ROOT / "golden_set" / "golden_250.csv")
    client = get_client()
    agree1 = agree3 = agree5 = 0
    sims: list[float] = []
    per_intent: dict[str, dict[str, int]] = {}
    for _, r in tqdm(list(golden.iterrows()), total=len(golden), desc="Retrieval eval"):
        hits = retrieve(client, str(r["text"]), top_k=5)  # unfiltered: honest test
        intents = [h.get("intent") for h in hits]
        true = r["true_intent"]
        agree1 += int(bool(intents) and intents[0] == true)
        agree3 += int(true in intents[:3])
        agree5 += int(true in intents)
        sims.extend([h.get("_similarity", 0.0) for h in hits])
        d = per_intent.setdefault(true, {"n": 0, "a1": 0})
        d["n"] += 1
        d["a1"] += int(bool(intents) and intents[0] == true)

    n = len(golden)
    out = {
        "n": n, "k": 5, "unfiltered": True,
        "intent_consistency@1": round(agree1 / n, 4),
        "intent_consistency@3": round(agree3 / n, 4),
        "intent_consistency@5": round(agree5 / n, 4),
        "mean_top5_similarity": round(sum(sims) / max(len(sims), 1), 4),
        "per_intent_top1": {k: {"n": v["n"], "acc": round(v["a1"] / v["n"], 3)}
                            for k, v in sorted(per_intent.items())},
    }
    with open(ROOT / "results" / "retrieval_eval.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
