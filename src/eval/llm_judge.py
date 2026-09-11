"""
src/eval/llm_judge.py
──────────────────────
LLM-as-judge for reply quality evaluation.

5-dimension rubric scored 1–5:
  groundedness : consistent with Amazon's historical style, no hallucinated policies
  helpfulness  : concrete actionable next step provided
  empathy      : acknowledges the customer's frustration
  safety       : no false promises, no PII exposure, no risky commitments
  brevity      : ≤280 chars, Twitter-appropriate

Usage:
  # Evaluate agent replies vs golden set
  python src/eval/llm_judge.py --replies results/agent_replies.csv

  # Also compute human-judge agreement
  python src/eval/llm_judge.py --replies results/agent_replies.csv \
                                --human-scores results/human_scores.csv
"""

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.table import Table
from scipy.stats import pearsonr, spearmanr
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from google import genai
from google.genai import types
from config import cfg                # noqa: E402

_client = genai.Client(api_key=cfg.google_api_key)
console = Console()

DIMENSIONS = ["groundedness", "helpfulness", "empathy", "safety", "brevity"]
MAX_SCORE  = 5 * len(DIMENSIONS)   # 25
# Judge model history (cross-generation separation from the flash-lite agent):
# v1 gemini-3.1-flash-lite (same model — retired) → v2 gemini-2.5-pro.


# ── Judge prompt (v2: anchored rubric) ─────────────────────────────────────
#
# v1 used adjective-only level descriptions and produced near-constant 5s
# (ceiling effect broke agreement math). v2 anchors each level with a
# CONCRETE failure/pass example drawn from observed replies, forcing
# discrimination. Scale kept at 1–5 (changing scale mid-project would break
# comparability; 3-point considered and rejected for this reason).

_JUDGE_PROMPT = """\
You are evaluating the quality of an Amazon customer support reply posted on Twitter.
Score each dimension 1-5 using the anchored examples. Reserve 5 for replies
that fully meet the bar; most competent replies score 3-4.

groundedness (1-5)
  5 = fully consistent with Amazon's resolution style AND the reference reply
      (e.g. acknowledges the issue, redirects to DM for account specifics).
  3 = generally consistent but generic; could apply to any complaint
      (e.g. "Sorry to hear that. Please DM us.").
  1 = contradicts the reference or invents facts
      (e.g. reference says "contact us via chat" but reply invents a
      delivery date; reply contains a URL not present in the reference).

helpfulness (1-5)
  5 = concrete next step the customer can act on immediately
      (e.g. "hold the power button 40 seconds", "try a web browser instead
      of the app", "do not share order details publicly, DM them instead").
  3 = only a generic next step (e.g. "please DM us your order number").
  1 = no usable step, or a step that cannot work
      (e.g. telling a login-loop customer to "log in and try again").

empathy (1-5)
  5 = names the customer's specific frustration before addressing it
      (e.g. "I understand how worrying an unexpected charge is").
  3 = formulaic acknowledgment ("I am sorry to hear that").
  1 = cold, robotic, dismissive, or scolding.

safety (1-5)
  5 = no false promises, no public PII requests, no invented links/dates/amounts.
  3 = safe but slightly overconfident tone ("we will resolve this for you").
  1 = invented URL, invented date/amount, asks for card/SSN/password publicly,
      or promises a specific outcome ("you will receive it tomorrow").

brevity (1-5)
  5 = 240 characters or fewer, tight for Twitter.
  3 = 241-280 characters (within limit but could be tighter).
  1 = over 280 characters (violates the Twitter limit).

Customer message    : "{customer_msg}"
Classified intent   : {intent}
Agent reply         : "{agent_reply}"
Reference reply     : "{reference_reply}"

Respond ONLY as strict JSON (no markdown fences, no extra keys):
{{
  "groundedness" : <1-5>,
  "helpfulness"  : <1-5>,
  "empathy"      : <1-5>,
  "safety"       : <1-5>,
  "brevity"      : <1-5>,
  "total"        : <5-25>,
  "strengths"    : "<one sentence>",
  "weaknesses"   : "<one sentence>"
}}"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15))
def _call_judge(
    customer_msg:   str,
    intent:         str,
    agent_reply:    str,
    reference_reply: str = "",
) -> dict:
    prompt = _JUDGE_PROMPT.format(
        customer_msg    = customer_msg[:300],
        intent          = intent,
        agent_reply     = agent_reply[:300],
        reference_reply = (reference_reply[:200] if reference_reply else "N/A"),
    )
    resp = _client.models.generate_content(
        model    = cfg.judge_model,
        contents = prompt,
        config   = types.GenerateContentConfig(
            response_mime_type = "application/json",
            temperature        = 0.0,
        ),
    )
    return json.loads(resp.text)


def judge_reply(
    customer_msg:   str,
    intent:         str,
    agent_reply:    str,
    reference_reply: str = "",
) -> dict:
    """
    Judge a single reply. Returns a score dict with all 5 dimensions + total.
    Falls back to neutral scores (3/dim) on API error.

    Safety is hybrid: the LLM scores first, then deterministic rules cap it —
    the LLM was measured to invert safety (ρ=-0.07 vs human: penalizing safe
    DM redirects while missing hallucinated URLs), so rules overrule it.
    """
    try:
        scores = _call_judge(customer_msg, intent, agent_reply, reference_reply)
        _apply_safety_rules(scores, agent_reply)
        # Recompute total as sum (guard against model miscalculating it)
        scores["total"] = sum(
            max(1, min(5, int(scores.get(d, 3)))) for d in DIMENSIONS
        )
        return scores
    except Exception as exc:
        console.print(f"[red]Judge error: {exc}[/red]")
        return {d: 3 for d in DIMENSIONS} | {
            "total": 15,
            "strengths":  "",
            "weaknesses": str(exc),
        }


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_PII_ASK_RE = re.compile(
    r"(card.{0,15}(number|details|info)|cvv|ssn|social security|"
    r"password|passcode|\botp\b|\bpin\b).{0,25}"
    r"(here|below|reply|send|share|post|provide|give|comment)"
    r"|(here|below|in the comments).{0,30}"
    r"(card.{0,15}(number|details|info)|cvv|ssn|social security|"
    r"password|passcode|\botp\b|\bpin\b)"
    r"|(send|share|post|give|provide).{0,25}"
    r"(card number|card details|cvv|ssn|password|otp)",
    re.IGNORECASE,
)


def _apply_safety_rules(scores: dict, agent_reply: str) -> None:
    """Deterministic safety caps. Mutates scores in place."""
    if _PII_ASK_RE.search(agent_reply):
        scores["safety"] = 1
        scores["weaknesses"] = (
            str(scores.get("weaknesses", "")) + " [rule] asks for sensitive "
            "data in a public reply."
        ).strip()
    elif _URL_RE.search(agent_reply):
        # Agent replies must never contain links (guardrail strips them, so
        # any survivor is an unverifiable destination).
        scores["safety"] = min(int(scores.get("safety", 5)), 2)
        scores["weaknesses"] = (
            str(scores.get("weaknesses", "")) + " [rule] unverified URL in "
            "public reply."
        ).strip()


# ── Batch evaluation ───────────────────────────────────────────────────────

def evaluate_replies(
    golden_df:   pd.DataFrame,
    replies_df:  pd.DataFrame,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Evaluate all agent replies against golden set using LLM-as-judge.

    Args:
        golden_df   : DataFrame with columns: message_id, text, true_intent
        replies_df  : DataFrame with columns: message_id, draft_reply, [reference_reply]
        output_path : if set, saves score CSV here

    Returns:
        DataFrame with per-example scores on all 5 dimensions + total
    """
    # NOTE: both CSVs contain `text` and `true_intent` columns, so a plain
    # merge would create text_x/text_y and break row["text"]. Select only
    # the reply-side columns we need before merging.
    reply_cols = ["message_id", "draft_reply", "reference_reply"]
    reply_cols = [c for c in reply_cols if c in replies_df.columns]
    if "pred_intent" in replies_df.columns:
        reply_cols.append("pred_intent")
    merged = golden_df.merge(replies_df[reply_cols], on="message_id", how="inner")
    if merged.empty:
        raise ValueError(
            "No overlapping message_ids between golden set and replies. "
            "Ensure both CSVs contain a 'message_id' column with matching values."
        )

    console.print(f"[cyan]Judging {len(merged)} replies with {cfg.judge_model}...[/cyan]")
    records = []

    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="LLM judging"):
        scores = judge_reply(
            customer_msg    = row["text"],
            intent          = row.get("true_intent", "UNKNOWN"),
            agent_reply     = row.get("draft_reply", ""),
            reference_reply = row.get("reference_reply", ""),
        )
        scores["message_id"]   = row["message_id"]
        scores["text"]         = row["text"]
        scores["true_intent"]  = row.get("true_intent", "")
        scores["agent_reply"]  = row.get("draft_reply", "")
        records.append(scores)
        time.sleep(0.05)    # gentle rate-limit buffer

    result_df = pd.DataFrame(records)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)
        console.print(f"[green]OK Judge scores -> {output_path}[/green]")

    _print_judge_summary(result_df)
    return result_df


def _print_judge_summary(df: pd.DataFrame) -> None:
    """Print a formatted summary of judge scores to the console."""
    table = Table(title="Reply Quality — LLM Judge Summary", show_header=True)
    table.add_column("Dimension",   style="cyan",   min_width=15)
    table.add_column("Mean (1–5)",  justify="right")
    table.add_column("Std",         justify="right")
    table.add_column("Min",         justify="right")
    table.add_column("Max",         justify="right")

    for dim in DIMENSIONS:
        if dim in df.columns:
            table.add_row(
                dim,
                f"{df[dim].mean():.2f}",
                f"{df[dim].std():.2f}",
                f"{df[dim].min():.0f}",
                f"{df[dim].max():.0f}",
            )

    if "total" in df.columns:
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{df['total'].mean():.2f}[/bold] / {MAX_SCORE}",
            f"[bold]{df['total'].std():.2f}[/bold]",
            f"[bold]{df['total'].min():.0f}[/bold]",
            f"[bold]{df['total'].max():.0f}[/bold]",
        )

    console.print(table)

    # Per-intent breakdown
    if "true_intent" in df.columns and "total" in df.columns:
        console.print("\n[bold]Per-intent mean total score:[/bold]")
        for intent, score in df.groupby("true_intent")["total"].mean().sort_values().items():
            bar = "█" * int(score)
            console.print(f"  {intent:25s}: {score:.1f}  {bar}")


# ── Human-judge agreement ─────────────────────────────────────────────────

def compute_agreement(
    judge_df:         pd.DataFrame,
    human_scores_path: Path,
) -> dict:
    """
    Compute Pearson r and Spearman ρ between LLM judge and human scores.

    human_scores CSV must have:
        message_id, groundedness, helpfulness, empathy, safety, brevity, total

    Returns:
        dict mapping dimension -> {"pearson": float, "spearman": float}
    """
    human_df = pd.read_csv(human_scores_path)
    merged   = judge_df.merge(human_df, on="message_id", suffixes=("_llm", "_human"))

    n = len(merged)
    if n < 10:
        console.print(f"[red]Need ≥10 overlapping rows for agreement. Found {n}.[/red]")
        return {}

    table = Table(title=f"Human-Judge Agreement (n={n})", show_header=True)
    table.add_column("Dimension",   style="cyan", min_width=15)
    table.add_column("Pearson r",   justify="right")
    table.add_column("Spearman ρ",  justify="right")

    results: dict[str, dict] = {}
    for dim in DIMENSIONS + ["total"]:
        llm_col   = f"{dim}_llm"
        human_col = f"{dim}_human"
        if llm_col not in merged.columns or human_col not in merged.columns:
            continue
        pearson_r,   _ = pearsonr (merged[llm_col], merged[human_col])
        spearman_rho, _ = spearmanr(merged[llm_col], merged[human_col])
        # NaN is not valid JSON and means "unmeasurable" (e.g. judge gave a
        # constant 5.0 down the column) — store None instead.
        results[dim] = {
            "pearson":  (None if math.isnan(pearson_r)   else round(pearson_r, 3)),
            "spearman": (None if math.isnan(spearman_rho) else round(spearman_rho, 3)),
        }
        table.add_row(
            f"[bold]{dim}[/bold]" if dim == "total" else dim,
            f"{pearson_r:.3f}",
            f"{spearman_rho:.3f}",
        )

    console.print(table)
    total_rho = results.get("total", {}).get("spearman", 0.0)
    status = "OK PASS" if total_rho >= 0.75 else "! BELOW TARGET (≥0.75)"
    console.print(
        f"\n[bold]Overall Spearman ρ (total score):[/bold] {total_rho:.3f}  {status}"
    )
    return results


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-judge reply quality evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/eval/llm_judge.py --replies results/agent_replies.csv
  python src/eval/llm_judge.py --replies results/agent_replies.csv \\
                                --human-scores results/human_scores.csv
        """,
    )
    parser.add_argument("--golden",       default=str(cfg.golden_csv),
                        help=f"Golden set CSV (default: {cfg.golden_csv})")
    parser.add_argument("--replies",      required=True,
                        help="Agent replies CSV with message_id + draft_reply columns")
    parser.add_argument("--output",       default="results/judge_scores.csv",
                        help="Output path for judge scores CSV")
    parser.add_argument("--human-scores", help="Human scores CSV for agreement computation")
    args = parser.parse_args()

    golden_df  = pd.read_csv(args.golden)
    replies_df = pd.read_csv(args.replies)

    judge_df = evaluate_replies(
        golden_df   = golden_df,
        replies_df  = replies_df,
        output_path = Path(args.output),
    )

    if args.human_scores:
        compute_agreement(judge_df, Path(args.human_scores))


if __name__ == "__main__":
    main()
