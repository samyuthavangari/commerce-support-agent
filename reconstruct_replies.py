"""Reconstruct results/agent_replies.csv (250 live rows) after an offline run
clobbered it. Drafts are preserved verbatim in judge_scores.csv; only
pred_intent/confidence (classifier) and reference_reply/top1-sim (retrieval)
are re-fetched — both deterministic given the frozen index. Escalation is
recomputed exactly via decide_escalation. No drafts are regenerated, so the
judged texts stay bit-identical.
"""

import sys
import time

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, "src")
from agent import classify_intent  # noqa: E402
from escalation import decide_escalation  # noqa: E402
from qdrant_store import get_client, retrieve  # noqa: E402


def main() -> None:
    golden = pd.read_csv("golden_set/golden_250.csv")
    judge = pd.read_csv("results/judge_scores.csv").set_index("message_id")
    assert len(golden) == 250 and len(judge) == 250
    client = get_client()

    rows = []
    for _, r in tqdm(list(golden.iterrows()), total=len(golden), desc="Reconstruct"):
        mid = r["message_id"]
        clf = classify_intent(str(r["text"]))
        hits = retrieve(client, str(r["text"]),
                        intent_filter=clf["intent"])
        ref = hits[0].get("amazon_reply", "") if hits else ""
        top1 = hits[0].get("_similarity") if hits else None
        dec = decide_escalation(str(r["text"]), clf["intent"],
                                clf["confidence"])
        draft = str(judge.loc[mid, "agent_reply"])
        rows.append({
            "message_id": mid, "text": r["text"],
            "true_intent": r["true_intent"], "pred_intent": clf["intent"],
            "intent_confidence": clf["confidence"],
            "draft_reply": draft, "reply_char_count": len(draft),
            "reference_reply": ref, "retrieval_top1_sim": top1,
            "escalation_decision": dec.decision,
            "escalation_reason": f"[{dec.triggered_by}] {dec.reason}"})
        time.sleep(0.1)
    pd.DataFrame(rows).to_csv("results/agent_replies.csv", index=False)
    print("reconstructed 250 rows")


if __name__ == "__main__":
    main()
