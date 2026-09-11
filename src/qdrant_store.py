"""
src/qdrant_store.py
────────────────────
Build and query the Qdrant Cloud vector index of AmazonHelp support pairs.

Each stored vector = embedding of:
    "<customer_message> [SEP] <amazon_reply>"

Payload per point:
    intent          : str  (heuristic-classified, refined by agent)
    customer_msg    : str
    amazon_reply    : str
    resolution_type : str  (redirected_DM | resolved_refund | acknowledged | escalated | info_provided)
    thread_id       : str
    created_at      : str

Usage:
    # Build the cloud index (run ONCE after data_prep.py)
    python src/qdrant_store.py --build

    # Test a semantic search
    python src/qdrant_store.py --query "Where is my package?"

    # Test with intent filter
    python src/qdrant_store.py --query "Kindle won't turn on" --intent DEVICE_TECH_SUPPORT
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from rich.console import Console
from tqdm import tqdm

from config import cfg
from embedder import embed_texts, embed_query
from intent_taxonomy import INTENT_NAMES

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console(safe_box=True)

# ── Qdrant Cloud client (singleton) ───────────────────────────────────────

def get_client() -> QdrantClient:
    """
    Returns a QdrantClient for QDRANT_URL (+ QDRANT_API_KEY if set).
    Works against local Docker (default http://localhost:6333, no key)
    as well as Qdrant Cloud.
    """
    console.print(f"[cyan]Connecting to Qdrant -> {cfg.qdrant_url}[/cyan]")
    kwargs: dict = {"url": cfg.qdrant_url, "timeout": 60}
    if cfg.qdrant_api_key:
        kwargs["api_key"] = cfg.qdrant_api_key
    client = QdrantClient(**kwargs)
    # Verify connection
    try:
        client.get_collections()
        console.print("[green]OK Qdrant connected[/green]")
    except Exception as e:
        raise ConnectionError(
            f"Cannot reach Qdrant at {cfg.qdrant_url}.\n"
            f"  -> Start local Qdrant: docker run -d -p 6333:6333 qdrant/qdrant\n"
            f"  -> Or set QDRANT_URL + QDRANT_API_KEY in your .env file.\n"
            f"  -> Original error: {e}"
        )
    return client


# Backwards-compat alias (app.py imports this name).
COLLECTION_NAME = cfg.qdrant_collection


# ── Heuristic helpers ─────────────────────────────────────────────────────

def _infer_resolution_type(amazon_reply: str) -> str:
    """Label how Amazon resolved the issue based on reply text."""
    r = amazon_reply.lower()
    if any(k in r for k in ["dm", "direct message", "send us a dm", "private message"]):
        return "redirected_DM"
    if any(k in r for k in ["refund", "reimburse", "credited", "money back"]):
        return "resolved_refund"
    if any(k in r for k in ["team", "specialist", "department", "escalat"]):
        return "escalated"
    if any(k in r for k in ["sorry", "apologize", "apologies", "understand"]):
        return "acknowledged"
    return "info_provided"


def _classify_intent_heuristic(text: str) -> str:
    """
    Fast keyword-based intent guess for index-time labelling.
    The LLM classifier (agent.py) overrides this at query time.
    """
    from intent_taxonomy import INTENTS
    t = text.lower()
    best_name, best_score = "GENERAL_COMPLAINT", 0
    for intent in INTENTS:
        score = sum(1 for kw in intent.trigger_keywords if kw in t)
        if score > best_score:
            best_score, best_name = score, intent.name
    return best_name


# ── Index building ────────────────────────────────────────────────────────

def build_index(client: QdrantClient, max_points: int = 10_000,
                exclude_eval: bool = False) -> None:
    """
    Read amazon_threads.jsonl -> embed (customer+reply) pairs -> upsert to Qdrant Cloud.

    Steps:
        1. Delete existing collection (clean rebuild)
        2. Create collection with cosine distance, dim=768
        3. Load threads, build text + payload lists
        4. Batch-embed with Gemini
        5. Batch-upsert to Qdrant Cloud (256 points per request)

    If exclude_eval is True, threads whose IDs appear in
    golden_set/golden_250.csv or golden_set/heldout_50.csv are SKIPPED,
    so evaluation examples never sit in the retrieval index (§6 leakage).
    """
    if not cfg.threads_jsonl.exists():
        raise FileNotFoundError(
            f"{cfg.threads_jsonl} not found.\n"
            "  -> Run: python src/data_prep.py --source hf"
        )

    eval_ids: set[str] = set()
    if exclude_eval:
        import pandas as pd
        for csv_name in ("golden_250.csv", "heldout_50.csv"):
            p = cfg.threads_jsonl.parent.parent.parent / "golden_set" / csv_name
            if p.exists():
                eval_ids.update(pd.read_csv(p)["thread_id"].astype(str).tolist())
        console.print(f"[cyan]Excluding {len(eval_ids)} eval threads from index[/cyan]")

    # ── 1. Recreate collection ─────────────────────────────────────────────
    if client.collection_exists(cfg.qdrant_collection):
        client.delete_collection(cfg.qdrant_collection)
        console.print(f"[yellow]Deleted existing collection '{cfg.qdrant_collection}'[/yellow]")

    client.create_collection(
        collection_name=cfg.qdrant_collection,
        vectors_config=VectorParams(size=cfg.embed_dim, distance=Distance.COSINE),
    )
    console.print(f"[green]OK Created collection '{cfg.qdrant_collection}' (dim={cfg.embed_dim}, cosine)[/green]")

    # Payload index on `intent` — required for filtered retrieval
    # (without this, search with query_filter raises 400 "Index required").
    client.create_payload_index(
        collection_name=cfg.qdrant_collection,
        field_name="intent",
        field_schema="keyword",
    )
    console.print("[green]OK Created payload index on field 'intent' (keyword)[/green]")

    # ── 2. Load threads ────────────────────────────────────────────────────
    threads = []
    with open(cfg.threads_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            threads.append(json.loads(line.strip()))
    if exclude_eval:
        before = len(threads)
        threads = [t for t in threads if str(t.get("thread_id")) not in eval_ids]
        console.print(f"[cyan]Dropped {before - len(threads)} eval threads "
                      f"({len(threads)} remain)[/cyan]")
    threads = threads[:max_points]
    console.print(f"[cyan]Loaded {len(threads):,} threads for indexing[/cyan]")

    # ── 3. Build (text, payload) lists ────────────────────────────────────
    texts: list[str] = []
    payloads: list[dict] = []

    for t in threads:
        cust_msg = t.get("first_customer_message", "").strip()
        if not cust_msg:
            continue
        # Use first Amazon reply only (most relevant response)
        amazon_reply = (t.get("amazon_replies") or [""])[0].strip()
        if not amazon_reply:
            continue

        combined = f"{cust_msg} [SEP] {amazon_reply}"
        texts.append(combined)
        payloads.append(
            {
                "intent":           _classify_intent_heuristic(cust_msg),
                "customer_msg":     cust_msg,
                "amazon_reply":     amazon_reply,
                "resolution_type":  _infer_resolution_type(amazon_reply),
                "thread_id":        t.get("thread_id", ""),
                "created_at":       t.get("created_at", ""),
                "num_turns":        t.get("num_turns", 1),
            }
        )

    console.print(f"[cyan]Embedding {len(texts):,} pairs with Gemini...[/cyan]")

    # ── 4. Embed ───────────────────────────────────────────────────────────
    vectors = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT", show_progress=True)

    # ── 5. Upsert to Qdrant Cloud ──────────────────────────────────────────
    BATCH_SIZE = 256
    total_upserted = 0

    for start in tqdm(range(0, len(vectors), BATCH_SIZE), desc="Upserting to Qdrant Cloud"):
        batch_vecs  = vectors [start : start + BATCH_SIZE]
        batch_pays  = payloads[start : start + BATCH_SIZE]
        points = [
            PointStruct(id=start + j, vector=vec, payload=pay)
            for j, (vec, pay) in enumerate(zip(batch_vecs, batch_pays))
        ]
        client.upsert(collection_name=cfg.qdrant_collection, points=points, wait=True)
        total_upserted += len(points)

    info = client.get_collection(cfg.qdrant_collection)
    console.print(
        f"\n[bold green]OK Index built successfully![/bold green]\n"
        f"   Collection : {cfg.qdrant_collection}\n"
        f"   Points     : {info.points_count:,}\n"
        f"   Cluster    : {cfg.qdrant_url}"
    )

    # Build manifest: lets run_eval verify (without trust) that the index
    # excludes exactly the current eval threads. See scripts/check_leakage.py.
    manifest = {
        "collection": cfg.qdrant_collection,
        "points": info.points_count,
        "exclude_eval": exclude_eval,
        "eval_ids": sorted(eval_ids) if exclude_eval else [],
        "built_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path = cfg.threads_jsonl.parent / "index_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    console.print(f"[green]OK Manifest -> {manifest_path}[/green]")


# ── Semantic retrieval ────────────────────────────────────────────────────

def retrieve(
    client: QdrantClient,
    query: str,
    intent_filter: Optional[str] = None,
    top_k: Optional[int] = None,
) -> list[dict]:
    """
    Retrieve top-k historical (customer_msg, amazon_reply) pairs from Qdrant Cloud.

    Args:
        client        : connected QdrantClient
        query         : incoming customer message text
        intent_filter : if provided, restrict search to this intent label
        top_k         : number of results (defaults to cfg.top_k_retrieval)

    Returns:
        List of payload dicts, each with: intent, customer_msg, amazon_reply, resolution_type
    """
    k = top_k if top_k is not None else cfg.top_k_retrieval
    query_vec = embed_query(query)

    qdrant_filter = None
    if intent_filter and intent_filter in INTENT_NAMES:
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="intent",
                    match=MatchValue(value=intent_filter),
                )
            ]
        )

    hits = client.search(
        collection_name=cfg.qdrant_collection,
        query_vector=query_vec,
        query_filter=qdrant_filter,
        limit=k,
        with_payload=True,
    )

    # Traceability (§12/§20): keep the similarity score alongside the payload
    # so eval can store per-example retrieval evidence, not just text.
    results = []
    for hit in hits:
        payload = dict(hit.payload)
        payload["_similarity"] = round(float(hit.score), 4)
        results.append(payload)
    return results


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qdrant Cloud index manager for Amazon Support Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/qdrant_store.py --build
  python src/qdrant_store.py --query "Where is my package?"
  python src/qdrant_store.py --query "Kindle won't charge" --intent DEVICE_TECH_SUPPORT
  python src/qdrant_store.py --info
        """,
    )
    parser.add_argument("--build",  action="store_true",  help="Build (or rebuild) the Qdrant Cloud index")
    parser.add_argument("--query",  type=str,             help="Run a semantic search query")
    parser.add_argument("--intent", type=str,             help="Filter retrieval by intent (use with --query)")
    parser.add_argument("--top-k",  type=int, default=5,  help="Number of results to return (default: 5)")
    parser.add_argument("--max-points", type=int, default=10_000, help="Max threads to index (default: 10000)")
    parser.add_argument("--exclude-eval", action="store_true",
                        help="Skip golden/heldout threads (leakage control; recommended)")
    parser.add_argument("--info",   action="store_true",  help="Show collection info")
    args = parser.parse_args()

    client = get_client()

    if args.build:
        build_index(client, max_points=args.max_points, exclude_eval=args.exclude_eval)

    if args.info:
        if client.collection_exists(cfg.qdrant_collection):
            info = client.get_collection(cfg.qdrant_collection)
            console.print(f"[bold]Collection:[/bold] {cfg.qdrant_collection}")
            console.print(f"  Points   : {info.points_count:,}")
            console.print(f"  Status   : {info.status}")
        else:
            console.print(f"[red]Collection '{cfg.qdrant_collection}' does not exist.[/red]")

    if args.query:
        console.print(f"\n[bold cyan]Query:[/bold cyan] {args.query}")
        if args.intent:
            console.print(f"[cyan]Intent filter:[/cyan] {args.intent}")
        hits = retrieve(client, args.query, intent_filter=args.intent, top_k=args.top_k)
        if not hits:
            console.print("[yellow]No results found.[/yellow]")
        for i, hit in enumerate(hits, 1):
            console.print(f"\n[bold]── Result {i} ──[/bold]")
            console.print(f"  Intent       : [yellow]{hit.get('intent')}[/yellow]")
            console.print(f"  Customer     : {hit.get('customer_msg', '')[:120]}")
            console.print(f"  Amazon reply : {hit.get('amazon_reply', '')[:160]}")
            console.print(f"  Resolution   : {hit.get('resolution_type')}")


if __name__ == "__main__":
    main()
