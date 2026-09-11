"""
scripts/eval/eval_topup.py
──────────────────────────
Measures classifier and escalation metrics specifically on the 50 rare-class top-up rows.
When run: Executed to assess rare-intent performance.
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
import time

import pandas as pd
from tqdm import tqdm

from agent import run_agent  # noqa: E402
from eval.llm_judge import judge_reply  # noqa: E402
from qdrant_store import get_client  # noqa: E402

def main() -> None:
    golden = pd.read_csv(ROOT / "golden_set" / "golden_250.csv")
    assert len(golden) == 250
    replies = pd.read_csv(ROOT / "results" / "agent_replies.csv")
    have = set(replies["message_id"])
    new = golden[~golden["message_id"].isin(have)].reset_index(drop=True)
    assert len(new) == 50, len(new)
    print(f"evaluating {len(new)} new rows")

    client = get_client()
    rows = []
    for _, r in tqdm(list(new.iterrows()), total=len(new), desc="Agent topup"):
        res = run_agent(str(r["text"]), qdrant_client=client, verbose=False)
        ref = (res.retrieved_examples[0].get("amazon_reply", "")
               if res.retrieved_examples else "")
        top1 = (res.retrieved_examples[0].get("_similarity")
                if res.retrieved_examples else None)
        rows.append({
            "message_id": r["message_id"], "text": r["text"],
            "true_intent": r["true_intent"], "pred_intent": res.intent,
            "intent_confidence": res.intent_confidence,
            "draft_reply": res.draft_reply,
            "reply_char_count": res.reply_char_count,
            "reference_reply": ref, "retrieval_top1_sim": top1,
            "escalation_decision": res.escalation_decision,
            "escalation_reason": res.escalation_reason})
        time.sleep(0.15)
    replies = pd.concat([replies, pd.DataFrame(rows)], ignore_index=True)
    replies.to_csv(ROOT / "results" / "agent_replies.csv", index=False)

    judged = pd.read_csv(ROOT / "results" / "judge_scores.csv")
    jhave = set(judged["message_id"])
    jrecs = []
    for _, r in tqdm(list(replies[replies["message_id"].isin(
            set(new["message_id"]))].iterrows()), total=50, desc="Judge topup"):
        s = judge_reply(str(r["text"]), str(r["true_intent"]),
                        str(r["draft_reply"]), str(r.get("reference_reply", "")))
        s["message_id"] = r["message_id"]
        s["text"] = r["text"]
        s["true_intent"] = r["true_intent"]
        s["agent_reply"] = r["draft_reply"]
        jrecs.append(s)
        time.sleep(0.05)
    judged = pd.concat([judged, pd.DataFrame(jrecs)], ignore_index=True)
    judged.to_csv(ROOT / "results" / "judge_scores.csv", index=False)
    print(f"replies={len(replies)} judged={len(judged)}")

if __name__ == "__main__":
    main()
