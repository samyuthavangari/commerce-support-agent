"""Ablation + blind-baseline replies (contract §15/§21).

- ABLATION (n=50, stratified): RAG draft (reuse agent_replies.csv) vs
  NO-RETRIEVAL draft (same pred_intent, examples=[]). Judge both.
- BLIND BASELINE (n=40 subset): canned template reply (offline, no API)
  vs agent RAG reply. Judge scores both WITHOUT source labels
  (judge() never receives system identity); human scores blind pairs.
Saves results/ablation_50.csv and results/blind_40.csv
(with source column for analysis only -- never shown to judge/human).
"""

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import draft_reply  # noqa: E402
from eval.llm_judge import judge_reply  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ("Thanks for reaching out — please send us a DM with your "
            "order number so we can look into this for you.")


def stratified_ids(golden: pd.DataFrame, n: int, seed: int) -> list[str]:
    import random
    random.seed(seed)
    picked: list[str] = []
    by_intent: dict[str, list[str]] = {}
    for _, r in golden.iterrows():
        by_intent.setdefault(r["true_intent"], []).append(r["message_id"])
    per = max(1, n // len(by_intent))
    for intent, ids in sorted(by_intent.items()):
        random.shuffle(ids)
        picked.extend(ids[:per])
    # Distribute the remainder (integer division under-fills, e.g. 50//7*7=49).
    k = 0
    order = sorted(by_intent)
    while len(picked) < n:
        ids = by_intent[order[k % len(order)]]
        nxt = ids[per + (k // len(order))]
        if nxt not in picked:
            picked.append(nxt)
        k += 1
        if k > n * len(order) + 10:
            break
    random.shuffle(picked)
    return picked[:n]


def main() -> None:
    golden = pd.read_csv(ROOT / "golden_set" / "golden_250.csv")
    replies = pd.read_csv(ROOT / "results" / "agent_replies.csv").set_index("message_id")
    judge_in = pd.read_csv(ROOT / "results" / "judge_scores.csv").set_index("message_id")

    abl_ids = stratified_ids(golden, 50, seed=11)
    print(f"[ablate] subset: {len(abl_ids)} rows")

    rows = []
    for mid in tqdm(abl_ids, desc="No-retrieval drafts"):
        g = golden[golden.message_id == mid].iloc[0]
        pred = replies.loc[mid, "pred_intent"]
        draft = draft_reply(str(g["text"]), str(pred), [])  # NO examples
        rows.append({"message_id": mid, "true_intent": g["true_intent"],
                     "text": g["text"], "pred_intent": pred,
                     "rag_reply": replies.loc[mid, "draft_reply"],
                     "norag_reply": draft})
    abl = pd.DataFrame(rows)

    # Judge the no-retrieval drafts (RAG scores reused from judge_scores.csv).
    # The judge never receives system identity -> blind by construction.
    scored = []
    for _, r in tqdm(list(abl.iterrows()), total=len(abl), desc="Judging no-RAG"):
        ref = replies.loc[r["message_id"], "reference_reply"]
        if pd.isna(ref):
            ref = ""
        s = judge_reply(str(r["text"]), str(r["true_intent"]),
                        str(r["norag_reply"]), str(ref))
        s["message_id"] = r["message_id"]
        scored.append(s)
    nj = pd.DataFrame(scored).set_index("message_id")
    abl["rag_total"] = [float(judge_in.loc[m, "total"]) for m in abl["message_id"]]
    abl["norag_total"] = [float(nj.loc[m, "total"]) for m in abl["message_id"]]
    abl.to_csv(ROOT / "results" / "ablation_50.csv", index=False)
    print(f"[ablate] RAG mean={abl['rag_total'].mean():.2f} "
          f"vs no-RAG mean={abl['norag_total'].mean():.2f} (judge totals)")

    # Blind baseline pairs on a 40-subset (15 prior human rows + 25 new).
    prior = pd.read_csv(ROOT / "results" / "human_scores.csv")["message_id"].tolist()
    extra = [m for m in stratified_ids(golden, 60, seed=23) if m not in prior][:25]
    blind_ids = prior + extra
    bpairs = []
    for mid in blind_ids:
        g = golden[golden.message_id == mid].iloc[0]
        bpairs.append({"message_id": mid, "true_intent": g["true_intent"],
                       "text": g["text"],
                       "agent_reply": replies.loc[mid, "draft_reply"],
                       "template_reply": TEMPLATE})
    pd.DataFrame(bpairs).to_csv(ROOT / "results" / "blind_40.csv", index=False)
    print(f"[ablate] blind pairs: {len(bpairs)} "
          f"(human scores them WITHOUT source labels next)")


if __name__ == "__main__":
    main()
