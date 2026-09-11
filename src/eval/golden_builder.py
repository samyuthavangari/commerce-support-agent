"""
src/eval/golden_builder.py
───────────────────────────
Build the 200-example golden evaluation set.

Two-pass labelling:
  Pass 1 (automated): Gemini Flash assigns intent + escalate_yn for each sample.
  Pass 2 (human)    : Open golden_set/golden_250.csv in Excel/Sheets, review and
                      correct 'true_intent' and 'escalate_yn', then set
                      'human_verified' = True for each reviewed row.

Usage:
  # Build the golden set
  python src/eval/golden_builder.py

  # Check Cohen's Kappa after human review
  python src/eval/golden_builder.py --kappa
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
from rich.console import Console
from tqdm import tqdm

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import google.generativeai as genai  # noqa: E402
from config import cfg               # noqa: E402
from intent_taxonomy import get_classifier_block, INTENT_NAMES  # noqa: E402

genai.configure(api_key=cfg.google_api_key)
console = Console()

# ── Sampling targets (total = 200) ─────────────────────────────────────────

SAMPLE_TARGETS: dict[str, int] = {
    "ORDER_STATUS":        35,
    "RETURN_REFUND":       35,
    "ACCOUNT_ACCESS":      25,
    "PRIME_SUBSCRIPTION":  25,
    "DEVICE_TECH_SUPPORT": 25,
    "DELIVERY_DAMAGE":     25,
    "GENERAL_COMPLAINT":   30,
}
assert sum(SAMPLE_TARGETS.values()) == 200, "Targets must sum to 200"


# ── Keyword pre-filter ─────────────────────────────────────────────────────

def _keyword_match(text: str, intent_name: str) -> bool:
    """Return True if any keyword for this intent appears in text."""
    from intent_taxonomy import INTENT_BY_NAME
    return any(
        kw in text.lower()
        for kw in INTENT_BY_NAME[intent_name].trigger_keywords
    )


# ── LLM first-pass label ──────────────────────────────────────────────────

_LABEL_PROMPT = """\
You are labelling a customer support tweet for a golden evaluation set.

{intent_block}

Also decide: should this message be ESCALATED to a human agent?
Escalate if: legal threat, extreme distress, ambiguous intent, PII present, or complex edge case.

Customer tweet: "{message}"

Respond ONLY as strict JSON (no markdown):
{{
  "intent": "INTENT_NAME",
  "confidence": <0.0-1.0>,
  "escalate": <true or false>,
  "escalate_reason": "<short reason, or empty string>"
}}"""


def _llm_label(message: str) -> dict:
    """Call Gemini Flash to label a single message. Returns safe defaults on error."""
    model  = genai.GenerativeModel(cfg.judge_model)
    prompt = _LABEL_PROMPT.format(
        intent_block = get_classifier_block(),
        message      = message[:500],
    )
    try:
        resp = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        result = json.loads(resp.text)
        # Validate intent
        intent = str(result.get("intent", "GENERAL_COMPLAINT")).upper().strip()
        if intent not in INTENT_NAMES:
            intent = "GENERAL_COMPLAINT"
        return {
            "intent":          intent,
            "confidence":      float(result.get("confidence", 0.5)),
            "escalate":        bool(result.get("escalate", False)),
            "escalate_reason": str(result.get("escalate_reason", "")),
        }
    except Exception as exc:
        console.print(f"[red]LLM label error: {exc}[/red]")
        return {
            "intent": "GENERAL_COMPLAINT",
            "confidence": 0.0,
            "escalate": False,
            "escalate_reason": "",
        }


# ── Stratified sampler ────────────────────────────────────────────────────

def _stratified_sample(threads: list[dict]) -> dict[str, list[dict]]:
    """
    Pre-filter and stratify threads into per-intent pools,
    then sample to SAMPLE_TARGETS counts.
    """
    buckets: dict[str, list[dict]] = {name: [] for name in SAMPLE_TARGETS}

    for t in threads:
        msg = t.get("first_customer_message", "")
        if not msg:
            continue
        for intent_name in SAMPLE_TARGETS:
            if _keyword_match(msg, intent_name):
                buckets[intent_name].append(t)

    sampled: dict[str, list[dict]] = {}
    random.seed(42)
    for intent_name, target in SAMPLE_TARGETS.items():
        pool = buckets[intent_name]
        random.shuffle(pool)
        sampled[intent_name] = pool[:target]
        console.print(
            f"  [cyan]{intent_name:25s}[/cyan] pool={len(pool):4d} -> sampled={len(sampled[intent_name])}"
        )

    return sampled


# ── Main builder ──────────────────────────────────────────────────────────

def build_golden_set() -> None:
    """Sample 200 examples, LLM-label them, and save to golden_set/golden_250.csv."""
    random.seed(42)

    if not cfg.threads_jsonl.exists():
        console.print(
            f"[red]✗ {cfg.threads_jsonl} not found.\n"
            "  -> Run: python src/data_prep.py --source hf[/red]"
        )
        sys.exit(1)

    # Load threads
    threads = []
    with open(cfg.threads_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            threads.append(json.loads(line.strip()))
    console.print(f"[cyan]Loaded {len(threads):,} threads from {cfg.threads_jsonl}[/cyan]\n")

    # Stratified sample
    console.print("[bold]Stratified sampling:[/bold]")
    sampled = _stratified_sample(threads)

    # LLM first-pass labelling
    rows = []
    msg_id = 1

    for proposed_intent, thread_list in sampled.items():
        console.print(f"\n[bold cyan]Labelling {proposed_intent} ({len(thread_list)} examples)...[/bold cyan]")
        for t in tqdm(thread_list, desc=proposed_intent):
            msg = t.get("first_customer_message", "").strip()
            if not msg:
                continue
            label = _llm_label(msg)
            rows.append(
                {
                    "message_id":      f"GS_{msg_id:04d}",
                    "thread_id":       t.get("thread_id", ""),
                    "text":            msg,
                    "proposed_intent": proposed_intent,
                    "llm_intent":      label["intent"],
                    "llm_confidence":  label["confidence"],
                    # human fills this in during second pass:
                    "true_intent":     label["intent"],
                    "escalate_yn":     label["escalate"],
                    "escalate_reason": label["escalate_reason"],
                    "labelled_by":     "llm_first_pass",
                    "human_verified":  False,
                }
            )
            msg_id += 1
            time.sleep(0.05)    # gentle rate-limit buffer

    df = pd.DataFrame(rows)
    cfg.golden_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.golden_csv, index=False)

    console.print(f"\n[bold green]OK Golden set saved: {len(df)} rows -> {cfg.golden_csv}[/bold green]")
    console.print(
        "\n[yellow]!  NEXT STEP — Human second pass:[/yellow]\n"
        f"  1. Open [cyan]{cfg.golden_csv}[/cyan] in Excel / Google Sheets\n"
        "  2. Review 'true_intent' and 'escalate_yn' columns for each row\n"
        "  3. Correct any mislabelled rows\n"
        "  4. Set 'human_verified' = True for every reviewed row\n"
        "  5. Run: python src/eval/golden_builder.py --kappa\n"
        "     (target: Cohen's Kappa ≥ 0.75)"
    )

    # Distribution preview
    console.print("\n[bold]LLM first-pass intent distribution:[/bold]")
    dist = df["llm_intent"].value_counts()
    for intent, count in dist.items():
        bar = "█" * count
        console.print(f"  {intent:25s}: {count:3d}  {bar}")


# ── Kappa checker ─────────────────────────────────────────────────────────

def compute_kappa() -> None:
    """
    Compute Cohen's Kappa between 'llm_intent' and 'true_intent'
    on human-verified rows in the golden set.
    """
    from sklearn.metrics import cohen_kappa_score

    if not cfg.golden_csv.exists():
        console.print(f"[red]✗ {cfg.golden_csv} not found. Build it first.[/red]")
        sys.exit(1)

    df       = pd.read_csv(cfg.golden_csv)
    verified = df[df["human_verified"].astype(str).str.lower().isin(["true", "1", "yes"])]

    if len(verified) < 20:
        console.print(
            f"[yellow]Only {len(verified)} human-verified rows. "
            "Review more rows in the CSV before computing Kappa.[/yellow]"
        )
        return

    kappa = cohen_kappa_score(verified["llm_intent"], verified["true_intent"])
    status = "OK PASS" if kappa >= 0.75 else "! BELOW TARGET (≥0.75)"
    console.print(
        f"\n[bold]Cohen's Kappa (LLM vs Human) on {len(verified)} verified rows:[/bold] "
        f"{kappa:.3f}  {status}"
    )


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the 200-example golden evaluation set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/eval/golden_builder.py           # Build golden set
  python src/eval/golden_builder.py --kappa   # Check human-LLM agreement
        """,
    )
    parser.add_argument("--kappa", action="store_true",
                        help="Compute Cohen's Kappa on human-verified rows")
    args = parser.parse_args()

    if args.kappa:
        compute_kappa()
    else:
        build_golden_set()


if __name__ == "__main__":
    main()
