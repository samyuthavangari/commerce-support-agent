"""Retrieval evaluation (contract §13): is retrieval actually useful?

For each golden query, retrieve top-5 WITHOUT intent filter and measure:
  - intent-consistency@1/3/5 vs the golden true_intent
  - mean cosine similarity of top-5
Saves results/retrieval_eval.json. Needs embedding API only (no LLM gen).
"""

import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qdrant_store import get_client, retrieve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


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
