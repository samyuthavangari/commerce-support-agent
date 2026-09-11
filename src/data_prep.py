"""
src/data_prep.py
────────────────
Download twcs.csv -> filter to AmazonHelp -> reconstruct threads -> save JSONL.

Sources (pick one):
  --source hf      HuggingFace mirror (no auth needed, recommended)
  --source kaggle  Kaggle API (needs ~/.kaggle/kaggle.json)
  --source skip    Already have data/raw/twcs.csv

Output: data/processed/amazon_threads.jsonl

Each line is a JSON object:
  {
    "thread_id":              str,
    "first_customer_message": str,
    "customer_messages":      [str, ...],
    "amazon_replies":         [str, ...],
    "num_turns":              int,
    "created_at":             str
  }

Usage:
  python src/data_prep.py --source hf --max-threads 10000
"""

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from tqdm import tqdm

from config import cfg

console = Console()

AMAZON_HANDLE = "AmazonHelp"


# ── Download helpers ───────────────────────────────────────────────────────

def _download_hf() -> None:
    """Download via HuggingFace datasets (no credentials required)."""
    try:
        from datasets import load_dataset
    except ImportError:
        console.print("[red]Install datasets: pip install datasets[/red]")
        sys.exit(1)

    # Try multiple known mirrors in order (first one that works wins)
    candidates = [
        ("SunidhiSriram/twcs",                              "train"),
        ("TNE-AI/customer-support-on-twitter-conversation", "train"),
        ("MohammadOthman/mo-customer-support-tweets-945k",  "train"),
    ]

    cfg.raw_csv.parent.mkdir(parents=True, exist_ok=True)

    for hf_id, split in candidates:
        try:
            console.print(f"[cyan]Trying HuggingFace dataset: {hf_id} ...[/cyan]")
            ds = load_dataset(hf_id, split=split)
            df = ds.to_pandas()
            df.to_csv(cfg.raw_csv, index=False)
            console.print(f"[green]OK  {hf_id}: {len(df):,} rows saved to {cfg.raw_csv}[/green]")
            return
        except Exception as exc:
            console.print(f"[yellow]  Skipping {hf_id}: {exc}[/yellow]")

    console.print(
        "[red]All HuggingFace mirrors failed.\n"
        "  Falling back to direct CSV download...[/red]"
    )
    _download_direct_csv()


def _download_direct_csv() -> None:
    """Last-resort: download twcs.csv directly from a public URL."""
    import urllib.request
    url = (
        "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv"
    )
    console.print(f"[cyan]Downloading CSV directly from:\n  {url}[/cyan]")
    cfg.raw_csv.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, cfg.raw_csv)
    console.print(f"[green]OK  Downloaded to {cfg.raw_csv}[/green]")


def _download_kaggle() -> None:
    """Download via Kaggle API (requires ~/.kaggle/kaggle.json)."""
    import os
    console.print("[cyan]Downloading via Kaggle API...[/cyan]")
    cfg.raw_csv.parent.mkdir(parents=True, exist_ok=True)
    ret = os.system(
        "kaggle datasets download thoughtvector/customer-support-on-twitter "
        f"--unzip -p {cfg.raw_csv.parent}"
    )
    if ret != 0 or not cfg.raw_csv.exists():
        console.print(
            "[red]Kaggle download failed.\n"
            "  -> Ensure kaggle.json is at ~/.kaggle/kaggle.json with correct credentials.\n"
            "  -> Or use --source hf (no credentials needed).[/red]"
        )
        sys.exit(1)
    console.print(f"[green]OK Saved -> {cfg.raw_csv}[/green]")


# ── Core processing ────────────────────────────────────────────────────────

def _load_csv() -> pd.DataFrame:
    """Load twcs.csv with only the columns we need."""
    console.print(f"[cyan]Loading {cfg.raw_csv} ...[/cyan]")
    df = pd.read_csv(
        cfg.raw_csv,
        dtype=str,
        usecols=[
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
            "response_tweet_id",
            "in_response_to_tweet_id",
        ],
        low_memory=False,
    )
    df["text"]    = df["text"].fillna("").str.strip()
    # NOTE (bug fix): twcs.csv stores tweet_id as "272" but
    # in_response_to_tweet_id as float-formatted "272.0". Without
    # normalizing, tweet_id.isin(parent_ids) matches 0 rows and
    # thread reconstruction yields 0 threads. Strip trailing ".0".
    for _col in ["tweet_id", "in_response_to_tweet_id", "response_tweet_id"]:
        df[_col] = (
            df[_col].fillna("").str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )
    console.print(f"[green]OK Loaded {len(df):,} total tweets[/green]")
    return df


def _filter_amazon(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only AmazonHelp reply rows + the customer messages they replied to."""
    amazon_rows = df[df["author_id"] == AMAZON_HANDLE]
    parent_ids  = set(amazon_rows["in_response_to_tweet_id"].dropna())
    parent_ids.discard("")

    relevant = df[
        (df["author_id"] == AMAZON_HANDLE) | (df["tweet_id"].isin(parent_ids))
    ].copy()

    console.print(
        f"[green]OK AmazonHelp rows: {len(amazon_rows):,} | "
        f"Customer parents: {len(parent_ids):,} | "
        f"Total relevant: {len(relevant):,}[/green]"
    )
    return relevant


def _reconstruct_threads(df: pd.DataFrame, max_threads: int) -> list[dict]:
    """
    Build multi-turn conversation threads from a (reply-chain) DataFrame.

    Strategy:
        1. Build a tweet_id -> row lookup map.
        2. For each AmazonHelp reply, find its parent customer tweet.
        3. Group by the first customer tweet (thread anchor).
        4. Stratified sample: keep top multi-turn threads + random single-turn.
    """
    tweet_map   = df.set_index("tweet_id").to_dict(orient="index")
    amazon_rows = (
        df[df["author_id"] == AMAZON_HANDLE]
        .dropna(subset=["in_response_to_tweet_id"])
        .copy()
    )

    threads: dict[str, dict] = {}

    for _, row in tqdm(
        amazon_rows.iterrows(), total=len(amazon_rows), desc="Reconstructing threads"
    ):
        parent_id = row["in_response_to_tweet_id"].strip()
        if not parent_id:
            continue
        parent = tweet_map.get(parent_id)
        if not parent:
            continue

        anchor = parent_id    # use first customer tweet as thread key
        if anchor not in threads:
            threads[anchor] = {
                "thread_id":              anchor,
                "customer_messages":      [],
                "amazon_replies":         [],
                "created_at":             parent.get("created_at", ""),
            }
        threads[anchor]["customer_messages"].append(parent["text"])
        threads[anchor]["amazon_replies"].append(row["text"])

    # Post-process
    result: list[dict] = []
    for t in threads.values():
        if not t["customer_messages"]:
            continue
        t["first_customer_message"] = t["customer_messages"][0]
        t["num_turns"]              = len(t["amazon_replies"])
        result.append(t)

    console.print(
        f"[green]OK Reconstructed {len(result):,} threads "
        f"(avg turns: {sum(t['num_turns'] for t in result) / max(len(result),1):.1f})[/green]"
    )

    # Stratified sample: keep richest multi-turn threads + random single-turn
    if len(result) > max_threads:
        multi  = sorted([t for t in result if t["num_turns"] > 1],
                        key=lambda x: x["num_turns"], reverse=True)[:500]
        single = [t for t in result if t["num_turns"] == 1]
        random.seed(42)
        sample_n = min(max_threads - len(multi), len(single))
        single   = random.sample(single, sample_n)
        result   = multi + single
        random.shuffle(result)
        console.print(f"[cyan]Sampled to {len(result):,} threads[/cyan]")

    return result


def _save_jsonl(threads: list[dict]) -> None:
    """Write threads to data/processed/amazon_threads.jsonl."""
    cfg.threads_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.threads_jsonl, "w", encoding="utf-8") as f:
        for t in threads:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    console.print(f"[green]OK Saved {len(threads):,} threads -> {cfg.threads_jsonl}[/green]")


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare AmazonHelp dataset from twcs.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/data_prep.py --source hf
  python src/data_prep.py --source kaggle --max-threads 5000
  python src/data_prep.py --source skip   # twcs.csv already downloaded
        """,
    )
    parser.add_argument(
        "--source",
        choices=["hf", "kaggle", "skip"],
        default="hf",
        help="Download source (default: hf — no credentials needed)",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=cfg.max_sample_threads,
        help=f"Max threads to keep (default: {cfg.max_sample_threads})",
    )
    args = parser.parse_args()

    # Step 1: Ensure raw CSV exists
    if args.source == "hf":
        _download_hf()
    elif args.source == "kaggle":
        _download_kaggle()
    else:
        if not cfg.raw_csv.exists():
            console.print(
                f"[red]✗ {cfg.raw_csv} not found. "
                "Run with --source hf or --source kaggle.[/red]"
            )
            sys.exit(1)
        console.print(f"[yellow]Using existing {cfg.raw_csv}[/yellow]")

    # Step 2: Load + filter + reconstruct
    df            = _load_csv()
    df_amazon     = _filter_amazon(df)
    threads       = _reconstruct_threads(df_amazon, max_threads=args.max_threads)

    # Step 3: Save
    _save_jsonl(threads)
    console.print(f"\n[bold green]OK Data prep complete -> {cfg.threads_jsonl}[/bold green]")


if __name__ == "__main__":
    main()
