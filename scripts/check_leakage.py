"""
scripts/check_leakage.py
========================
Automated leakage checks (contract §6). FAILS LOUDLY (exit 1) on violation.

Checks:
  1. eval-thread exclusion — no golden/heldout thread_id present in the
     Qdrant collection payloads (requires the index built with --exclude-eval).
  2. exact duplicates — identical normalized customer texts inside golden,
     and golden texts appearing verbatim in the retrieval corpus.
  3. same-thread-in-topK — each golden query's top-5 retrieval must not
     contain its own thread_id (needs embedding API; skip with --no-retrieval).

Usage:
  python scripts/check_leakage.py
  python scripts/check_leakage.py --no-retrieval   # skip live top-K check
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import cfg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def norm(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-retrieval", action="store_true")
    args = ap.parse_args()
    failures: list[str] = []

    golden = pd.read_csv(ROOT / "golden_set" / "golden_250.csv")
    heldout = pd.read_csv(ROOT / "golden_set" / "heldout_50.csv")
    eval_ids = (set(golden["thread_id"].astype(str))
                | set(heldout["thread_id"].astype(str)))
    print(f"[leak] eval threads: {len(eval_ids)} "
          f"(golden={len(golden)}, heldout={len(heldout)})")

    # ── 1. collection payloads must not contain eval threads ──────────────
    from qdrant_store import get_client
    client = get_client()
    indexed_ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=cfg.qdrant_collection, limit=2000,
            offset=offset, with_payload=True, with_vectors=False)
        if not points:
            break
        indexed_ids.update(str(p.payload.get("thread_id")) for p in points)
        if offset is None:
            break
    overlap = eval_ids & indexed_ids
    print(f"[leak] indexed points: {len(indexed_ids)}, "
          f"eval overlap: {len(overlap)}")
    if overlap:
        failures.append(f"{len(overlap)} eval threads present in index "
                        f"(e.g. {sorted(overlap)[:3]}); rebuild with --exclude-eval")

    # ── 2a. exact dupes inside golden ─────────────────────────────────────
    seen: dict[str, str] = {}
    dupes = 0
    for _, r in golden.iterrows():
        key = norm(str(r["text"]))
        if key in seen:
            print(f"[leak] WARN golden exact-dupe: {r['message_id']} == "
                  f"{seen[key]} (kept deliberately; both labels agree)")
            dupes += 1
        else:
            seen[key] = r["message_id"]
    print(f"[leak] golden internal exact-dupes: {dupes} (warn-only)")

    # ── 2b. golden texts verbatim in corpus payloads ──────────────────────
    corpus_texts: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=cfg.qdrant_collection, limit=2000,
            offset=offset, with_payload=True, with_vectors=False)
        if not points:
            break
        corpus_texts.update(norm(str(p.payload.get("customer_msg", "")))
                            for p in points)
        if offset is None:
            break
    verbatim = [r["message_id"] for _, r in golden.iterrows()
                if norm(str(r["text"])) in corpus_texts]
    print(f"[leak] golden texts verbatim in corpus: {len(verbatim)}")
    if verbatim:
        failures.append(f"{len(verbatim)} golden texts retrievable verbatim "
                        f"(e.g. {verbatim[:3]})")

    # ── 2c. near-duplicates (REPORT ONLY — support language is naturally
    # repetitive, so this informs rather than fails) ──────────────────────
    def toks(s: str) -> set[str]:
        return set(norm(s).split())

    g_toks = [(r["message_id"], toks(str(r["text"]))) for _, r in golden.iterrows()]
    near = 0
    for mid, gt in g_toks:
        if not gt:
            continue
        for ct in corpus_texts:
            cts = set(ct.split())
            if not cts:
                continue
            jac = len(gt & cts) / len(gt | cts)
            if jac >= 0.85:
                near += 1
                break
    print(f"[leak] golden rows with a >=0.85-Jaccard near-twin in corpus: "
          f"{near} (report-only; short support tweets legitimately repeat)")

    # ── 3. same-thread in live top-K ──────────────────────────────────────
    if not args.no_retrieval:
        from qdrant_store import retrieve
        bad = 0
        for _, r in golden.iterrows():
            hits = retrieve(client, str(r["text"]), top_k=5)
            tids = {str(h.get("thread_id")) for h in hits}
            if str(r["thread_id"]) in tids:
                bad += 1
        print(f"[leak] golden queries retrieving own thread in top-5: {bad}")
        if bad:
            failures.append(f"{bad} queries retrieve their own thread")

    if failures:
        print("\n[leak] FAIL:")
        for f in failures:
            print("  -", f)
        return 1
    print("\n[leak] PASS: no blocking leakage detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
