"""Re-judge all 250 replies with judge v2 (cross-generation, anchored rubric).

Checkpointed: re-running resumes from results/judge_scores_v2.csv.
Usage: python scripts/rejudge_v2.py [--templates]
(--templates also re-judges the 70 blind template replies.)
"""

import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval.llm_judge import judge_reply  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "judge_scores_v2.csv"
TOUT = ROOT / "results" / "blind_template_judge_v2.csv"
TPL = ("Thanks for reaching out — please send us a DM with your "
       "order number so we can look into this for you.")


def score_batch(items, done_ids, desc):
    recs = []
    todo = [it for it in items if it[0] not in done_ids]
    for mid, text, intent, reply, ref in tqdm(todo, desc=desc):
        s = judge_reply(str(text), str(intent), str(reply), str(ref))
        s["message_id"] = mid
        recs.append(s)
        time.sleep(0.05)
    return recs


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", action="store_true")
    args = ap.parse_args()

    replies = pd.read_csv(ROOT / "results" / "agent_replies.csv")
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["message_id"])
    done_ids = set(done["message_id"].astype(str).tolist())
    items = [(r["message_id"], r["text"], r["true_intent"], r["draft_reply"],
              "" if pd.isna(r.get("reference_reply", "")) else r.get("reference_reply", ""))
             for _, r in replies.iterrows()]
    recs = score_batch(items, done_ids, "Re-judge v2")
    if recs or not done.empty:
        out = pd.concat([done, pd.DataFrame(recs)], ignore_index=True)
        out.to_csv(OUT, index=False)
    print(f"agent judged: {len(pd.read_csv(OUT))}/250")

    if args.templates:
        b = pd.concat([pd.read_csv(ROOT / "results" / "blind_40.csv"),
                       pd.read_csv(ROOT / "results" / "blind_30.csv")],
                      ignore_index=True)
        tdone = pd.read_csv(TOUT) if TOUT.exists() else pd.DataFrame(columns=["message_id"])
        tids = set(tdone["message_id"].astype(str).tolist())
        trecs = score_batch(
            [(r["message_id"], r["text"], r["true_intent"], TPL, "")
             for _, r in b.iterrows()], tids, "Templates v2")
        if trecs or not tdone.empty:
            tout = pd.concat([tdone, pd.DataFrame(trecs)], ignore_index=True)
            tout.to_csv(TOUT, index=False)
        print(f"templates judged: {len(pd.read_csv(TOUT))}/70")


if __name__ == "__main__":
    main()
