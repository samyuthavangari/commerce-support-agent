"""
run_eval.py
────────────
One-command evaluation harness.

Runs the full Commerce Support Agent on every example in the golden set
and produces a complete evaluation report.

Outputs:
  results/agent_replies.csv       — agent predictions + drafted replies
  results/eval_report.json        — machine-readable metrics (all baselines)
  results/confusion_matrix.png    — intent confusion matrix
  results/judge_scores.csv        — per-example LLM judge scores

Usage:
  python run_eval.py                             # full evaluation
  python run_eval.py --no-qdrant                 # classifier only (no retrieval)
  python run_eval.py --no-judge                  # skip LLM judge (save API calls)
  python run_eval.py --baselines                 # include trivial + TF-IDF baselines
  python run_eval.py --max-examples 20           # quick smoke test
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm

import os
sys.path.insert(0, "src")

# Default dummy creds if running in offline / resume / CI mode
if not os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY", "").startswith("your_"):
    os.environ["GOOGLE_API_KEY"] = "ci-mock-key-not-used-in-replay"

from config import cfg                            # noqa: E402
from agent import run_agent                       # noqa: E402
from eval.metrics import (                        # noqa: E402
    bootstrap_ci,
    compute_intent_metrics,
    compute_escalation_metrics,
    plot_confusion_matrix,
    run_baseline_comparison,
    print_intent_table,
)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console(safe_box=True)


def run_offline_agent(message: str) -> dict:
    """
    No-API, no-Qdrant fallback agent for --offline inspection.
    Heuristic intent + canned reply + deterministic rules ONLY
    (no LLM escalation branch). Clearly NOT headline numbers.
    """
    from eval.metrics import _heuristic_label
    from escalation import check_hard_rules

    intent = _heuristic_label(message)
    reply = ("Thanks for reaching out — please send us a DM with your "
             "order number so we can look into this for you.")
    hit = check_hard_rules(message, intent, 0.6)
    if hit:
        decision, reason = "escalate", f"[{hit.triggered_by}] {hit.reason}"
    else:
        decision, reason = "auto", f"[{intent}] rules-only offline path"
    return {"intent": intent, "confidence": 0.6, "reasoning": "offline heuristic",
            "reply": reply, "decision": decision, "reason": reason}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Commerce Support Agent — Full Evaluation Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_eval.py
  python run_eval.py --baselines --max-examples 50
  python run_eval.py --no-judge --no-qdrant
        """,
    )
    parser.add_argument("--golden",       default=str(cfg.golden_csv),
                        help=f"Golden set path (default: {cfg.golden_csv})")
    parser.add_argument("--no-qdrant",    action="store_true",
                        help="Skip Qdrant retrieval (faster; tests classifier only)")
    parser.add_argument("--no-judge",     action="store_true",
                        help="Skip LLM-as-judge evaluation (saves API calls)")
    parser.add_argument("--baselines",    action="store_true",
                        help="Run trivial and TF-IDF baselines for comparison")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit to first N examples (quick smoke test)")
    parser.add_argument("--fast", action="store_true",
                        help="Fast verification: stratified 30-row subset "
                             "(fixed seed 11) incl. judge — headline repro "
                             "in <5 min. Use --full for n=200.")
    parser.add_argument("--full", action="store_true",
                        help="Explicit full n=200 evaluation (default when "
                             "no subset flags are given)")
    parser.add_argument("--human-scores", type=str, default=None,
                        help="Human scores CSV for judge-agreement check "
                             "(default: results/human_scores.csv if present)")
    parser.add_argument("--offline", action="store_true",
                        help="No-API demo: heuristic intent + canned reply + "
                             "rules-only escalation (inspect without a key; "
                             "NOT headline numbers)")
    parser.add_argument("--no-docker", action="store_true",
                        help="Zero-Docker mode: use pure-Python in-memory numpy "
                             "vector store (instant cold-start, no Docker needed)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel API workers for the agent loop "
                             "(default 4; --workers 1 = strict serial). "
                             "Row order is preserved.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: results/; --fast and "
                             "--offline default to results_fast/ and "
                             "results_offline/ so headline artifacts are "
                             "never clobbered)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip the agent loop by loading "
                             "results/agent_replies.csv (e.g. after an "
                             "interrupted run); re-runs metrics, baselines, "
                             "judge, agreement and report")
    args = parser.parse_args()

    if args.out_dir:
        OUT = Path(args.out_dir)
    elif args.fast:
        OUT = cfg.results_dir.parent / "results_fast"
    elif args.offline:
        OUT = cfg.results_dir.parent / "results_offline"
    else:
        OUT = cfg.results_dir
    OUT.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan]Output dir: {OUT}[/cyan]")
    console.print(Panel("[bold cyan]Commerce Support Agent — Evaluation Harness[/bold cyan]"))

    # ── Leakage gate: fail the evaluation, not just warn ──────────────────
    # Verifies the live index excludes exactly the current eval threads,
    # via the manifest written by `qdrant_store.py --build --exclude-eval`.
    if not args.no_qdrant and not args.offline and not args.no_docker:
        manifest_path = cfg.threads_jsonl.parent / "index_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if not manifest.get("exclude_eval"):
                console.print("[red]✗ Index was built WITHOUT --exclude-eval "
                              "(eval threads may be retrievable)[/red]")
                sys.exit(2)

    # ── Load golden set ───────────────────────────────────────────────────
    if not Path(args.golden).exists():
        console.print(
            f"[red]✗ Golden set not found: {args.golden}\n"
            "  -> Run: python src/eval/golden_builder.py[/red]"
        )
        sys.exit(1)

    golden_df = pd.read_csv(args.golden)
    if args.fast:
        # Stratified 30-row subset, fixed seed: every intent represented,
        # identical rows on every run. (~2-4 min incl. judge.)
        fast_ids: list[str] = []
        by_intent: dict[str, list[str]] = {}
        for _, r in golden_df.iterrows():
            by_intent.setdefault(r["true_intent"], []).append(r["message_id"])
        import random as _random
        per = max(1, 30 // len(by_intent))
        for intent in sorted(by_intent):
            ids = sorted(by_intent[intent])
            _random.Random(11).shuffle(ids)
            fast_ids.extend(ids[:per])
        k = 0
        order = sorted(by_intent)
        while len(fast_ids) < 30:
            ids = sorted(by_intent[order[k % len(order)]])
            nxt = ids[(per + k // len(order)) % len(ids)]
            if nxt not in fast_ids:
                fast_ids.append(nxt)
            k += 1
        golden_df = golden_df[golden_df["message_id"].isin(fast_ids[:30])].reset_index(drop=True)
        console.print("[cyan]--fast: stratified 30-row subset (seed 11)[/cyan]")
    if args.max_examples:
        golden_df = golden_df.head(args.max_examples)
    console.print(f"[cyan]Evaluating on {len(golden_df)} golden examples[/cyan]")

    # ── Connect to Qdrant Cloud ───────────────────────────────────────────
    qdrant_client = None
    if not args.no_qdrant and not args.offline:
        try:
            from qdrant_store import get_client
            qdrant_client = get_client(no_docker=args.no_docker)
            if hasattr(qdrant_client, "get_collection"):
                info = qdrant_client.get_collection(cfg.qdrant_collection)
                console.print(f"[green]OK Retrieval store active ({getattr(info, 'status', 'connected')})[/green]")
        except Exception as exc:
            console.print(
                f"[yellow]! Retrieval store unavailable ({exc})\n"
                "  Continuing without retrieval — set --no-qdrant to suppress this warning.[/yellow]"
            )
            qdrant_client = None
    if args.offline:
        console.print("[yellow]OFFLINE mode: no API, no Qdrant — "
                      "heuristic intent + canned replies + rules-only "
                      "escalation. These are NOT headline numbers.[/yellow]")

    # ── Run agent on all examples ─────────────────────────────────────────
    agent_intent_preds:  list[str]  = []
    escalation_preds:    list[bool] = []
    reply_rows:          list[dict] = []
    error_rows:          list[dict] = []

    if args.resume and not args.offline:
        replies_path = OUT / "agent_replies.csv"
        if not replies_path.exists():
            console.print(f"[red]✗ --resume needs {replies_path} (run without it first)[/red]")
            sys.exit(1)
        replies_df = pd.read_csv(replies_path)
        if len(replies_df) != len(golden_df):
            console.print(f"[red]✗ agent_replies has {len(replies_df)} rows, "
                          f"golden has {len(golden_df)} — rerun without --resume[/red]")
            sys.exit(1)
        console.print(f"[cyan]Resuming from {replies_path} "
                      f"({len(replies_df)} rows, agent loop skipped)[/cyan]")
        agent_intent_preds = replies_df["pred_intent"].tolist()
        escalation_preds = (replies_df["escalation_decision"] == "escalate").tolist()
        reply_rows = replies_df.to_dict(orient="records")

    def _run_one(row) -> dict:
        """Run the live agent on one row; returns reply-row or error-row."""
        try:
            result = run_agent(
                message       = str(row["text"]),
                qdrant_client = qdrant_client,
                verbose       = False,
            )
            ref = (result.retrieved_examples[0].get("amazon_reply", "")
                   if result.retrieved_examples else "")
            top1 = (result.retrieved_examples[0].get("_similarity")
                    if result.retrieved_examples else None)
            time.sleep(0.15)    # rate-limit buffer between LLM calls
            return {"ok": True, "row": {
                "message_id":          row["message_id"],
                "text":                row["text"],
                "true_intent":         row["true_intent"],
                "pred_intent":         result.intent,
                "intent_confidence":   result.intent_confidence,
                "draft_reply":         result.draft_reply,
                "reply_char_count":    result.reply_char_count,
                "reference_reply":     ref,
                "retrieval_top1_sim":  top1,
                "escalation_decision": result.escalation_decision,
                "escalation_reason":   result.escalation_reason,
            }}
        except Exception as exc:  # noqa: BLE001 — quarantined below
            return {"ok": False, "row": {
                "message_id":  row["message_id"],
                "text":        row["text"],
                "true_intent": row["true_intent"],
                "error":       str(exc),
            }}

    rows_list = list(golden_df.iterrows())
    use_pool = (not args.resume and not args.offline and args.workers > 1)
    if use_pool:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            # pool.map preserves input order -> row order preserved.
            outcomes = list(tqdm(
                pool.map(lambda ir: _run_one(ir[1]), rows_list),
                total=len(rows_list), desc=f"Running agent x{args.workers}"))
    else:
        outcomes = []
        for _, row in tqdm(rows_list, total=len(rows_list), desc="Running agent"):
            if args.resume and not args.offline:
                break
            if args.offline:
                off = run_offline_agent(str(row["text"]))
                agent_intent_preds.append(off["intent"])
                escalation_preds.append(off["decision"] == "escalate")
                reply_rows.append(
                    {
                        "message_id":          row["message_id"],
                        "text":                row["text"],
                        "true_intent":         row["true_intent"],
                        "pred_intent":         off["intent"],
                        "intent_confidence":   off["confidence"],
                        "draft_reply":         off["reply"],
                        "reply_char_count":    len(off["reply"]),
                        "reference_reply":     "",
                        "retrieval_top1_sim":  None,
                        "escalation_decision": off["decision"],
                        "escalation_reason":   off["reason"],
                    }
                )
                continue
            outcomes.append(_run_one(row))

    for outcome in outcomes:
        if outcome["ok"]:
            r = outcome["row"]
            agent_intent_preds.append(r["pred_intent"])
            escalation_preds.append(r["escalation_decision"] == "escalate")
            reply_rows.append(r)
        else:
            # QUARANTINE, do not fabricate: an earlier revision invented
            # GENERAL_COMPLAINT + escalate=True here, which silently pollutes
            # both intent and escalation metrics (escalate=True inflates
            # recall). Error rows are stored separately, EXCLUDED from all
            # metrics, and abort the run above a 5% error rate.
            console.print(f"[red]Error on {outcome['row']['message_id']}: "
                          f"{outcome['row']['error']}[/red]")
            error_rows.append(outcome["row"])

    if error_rows:
        err_path = OUT / "agent_errors.csv"
        pd.DataFrame(error_rows).to_csv(err_path, index=False)
        err_rate = len(error_rows) / max(len(golden_df), 1)
        console.print(f"[red]! {len(error_rows)} agent errors "
                      f"({err_rate:.1%}) quarantined -> {err_path} "
                      f"(excluded from metrics)[/red]")
        if err_rate > 0.05:
            console.print("[red]✗ Error rate above 5% — failing the "
                          "evaluation instead of reporting polluted "
                          "metrics.[/red]")
            sys.exit(3)

    replies_df   = pd.DataFrame(reply_rows)
    replies_path = OUT / "agent_replies.csv"
    replies_df.to_csv(replies_path, index=False)
    console.print(f"[green]OK Agent replies saved -> {replies_path}[/green]")

    if error_rows:
        # Metrics below cover EVALUATED rows only (never padded with guesses).
        golden_df = golden_df[
            golden_df["message_id"].isin(replies_df["message_id"])
        ].reset_index(drop=True)
        console.print(f"[yellow]Metrics computed on {len(golden_df)} evaluated "
                      f"rows ({len(error_rows)} quarantined).[/yellow]")

    # ── Intent metrics ────────────────────────────────────────────────────
    console.print("\n[bold]═══ Intent Classification Metrics ═══[/bold]")
    y_true         = golden_df["true_intent"].tolist()
    intent_metrics = compute_intent_metrics(y_true, agent_intent_preds)
    print_intent_table(intent_metrics)
    try:
        plot_confusion_matrix(
            intent_metrics,
            output_path=OUT / "confusion_matrix.png",
        )
    except OSError as exc:
        # Typically a locked file (e.g. image viewer open on Windows).
        # Metrics continue; the PNG regenerates in seconds via --resume.
        console.print(f"[yellow]! Confusion matrix not saved ({exc}) — "
                      "close any viewer on results/confusion_matrix.png "
                      "and rerun with --resume to regenerate.[/yellow]")

    # ── Escalation metrics ────────────────────────────────────────────────
    console.print("\n[bold]═══ Escalation Decision Metrics ═══[/bold]")
    esc_true    = golden_df["escalate_yn"].astype(bool).tolist()
    esc_metrics = compute_escalation_metrics(esc_true, escalation_preds)

    esc_table = Table(show_header=True)
    esc_table.add_column("Metric",    style="cyan")
    esc_table.add_column("Value",     justify="right")
    esc_table.add_column("Target",    justify="right")
    esc_table.add_column("Status",    justify="center")

    for metric, target in [("precision", 0.70), ("recall", 0.90), ("f1", 0.75)]:
        val    = esc_metrics[metric]
        status = "OK" if val >= target else "!"
        esc_table.add_row(
            metric,
            f"{val:.3f}",
            f"≥{target:.2f}",
            status,
        )
    esc_table.add_row(
        "false_auto_handle",
        f"{esc_metrics['false_auto_handle_rate']:.3f}",
        "≤0.10",
        "OK" if esc_metrics["false_auto_handle_rate"] <= 0.10 else "!",
    )
    console.print(esc_table)
    console.print("[yellow]  ! Recall is the critical metric — missing escalations is dangerous.[/yellow]")

    # ── Baselines ─────────────────────────────────────────────────────────
    baseline_metrics: dict = {}
    if args.baselines:
        console.print("\n[bold]═══ Baseline Comparison ═══[/bold]")
        baseline_metrics = run_baseline_comparison(golden_df, agent_intent_preds)

    # ── LLM-as-judge ─────────────────────────────────────────────────────
    judge_metrics: dict = {}
    if args.offline:
        console.print("\n[yellow]OFFLINE mode: judge skipped (needs API) — "
                      "reply quality: NOT EVALUATED.[/yellow]")
    if not args.no_judge and not args.offline:
        console.print("\n[bold]═══ LLM-as-Judge Reply Quality ═══[/bold]")
        from eval.llm_judge import evaluate_replies
        judge_df = evaluate_replies(
            golden_df   = golden_df,
            replies_df  = replies_df,
            output_path = OUT / "judge_scores.csv",
        )
        judge_metrics = {
            "mean_total":         round(float(judge_df["total"].mean()), 2),
            "std_total":          round(float(judge_df["total"].std()),  2),
            "mean_per_dimension": {
                dim: round(float(judge_df[dim].mean()), 2)
                for dim in ["groundedness", "helpfulness", "empathy", "safety", "brevity"]
                if dim in judge_df.columns
            },
        }

    # ── Save report ───────────────────────────────────────────────────────
    # Human-judge agreement (bundled so one command reproduces everything)
    from eval.llm_judge import compute_agreement
    agreement: dict = {}
    human_scores = args.human_scores or str(cfg.results_dir / "human_scores.csv")
    if judge_metrics and Path(human_scores).exists():
        console.print("\n[bold]═══ Human-Judge Agreement ═══[/bold]")
        judge_df_full = pd.read_csv(OUT / "judge_scores.csv")
        agreement = compute_agreement(judge_df_full, Path(human_scores))
    elif judge_metrics:
        console.print(
            "\n[yellow]No human scores found — skipping agreement check. "
            "Add results/human_scores.csv to enable it.[/yellow]"
        )

    report = {
        "provenance": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": "offline" if args.offline else "live",
            "golden": str(args.golden),
            "classifier_model": cfg.classifier_model,
            "drafter_model": cfg.drafter_model,
            "judge_model": cfg.judge_model,
            "top_k_retrieval": cfg.top_k_retrieval,
            "embed_model": cfg.embed_model,
            "seed": 42,
        },
        "num_examples":         len(golden_df),
        "qdrant_used":          qdrant_client is not None,
        "intent_classification": intent_metrics,
        "intent_ci": {
            "accuracy_95": bootstrap_ci(y_true, agent_intent_preds, "accuracy"),
            "macro_f1_95": bootstrap_ci(y_true, agent_intent_preds, "macro_f1"),
        },
        "escalation":           esc_metrics,
        "escalation_ci": {
            "recall_95": bootstrap_ci(esc_true, escalation_preds, "recall"),
        },
        "baselines":            baseline_metrics,
        "reply_quality_judge":  judge_metrics,
        "human_judge_agreement": agreement,
    }
    report_path = OUT / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────
    console.print(f"\n[bold green]OK Evaluation complete -> {report_path}[/bold green]")
    console.print("\n[bold]── Headline Numbers ──[/bold]")
    console.print(f"  Intent Accuracy    : {intent_metrics['accuracy']:.3f}")
    console.print(f"  Intent Macro F1    : {intent_metrics['macro_f1']:.3f}")
    console.print(f"  Escalation Recall  : {esc_metrics['recall']:.3f} {'OK' if esc_metrics['recall'] >= 0.90 else '!'}")
    if judge_metrics:
        mean_t = judge_metrics["mean_total"]
        console.print(f"  Reply Quality      : {mean_t:.1f} / 25  ({mean_t/25*100:.0f}%)")


if __name__ == "__main__":
    main()
