"""
src/agent.py
────────────
Core agent pipeline using the new google-genai SDK.

Pipeline: Classify -> Retrieve -> Draft -> Escalation Decide

Usage:
    python src/agent.py --message "Where is my order?"
    python src/agent.py --message "I want a refund" --no-qdrant
"""

import argparse
import json
import re
from dataclasses import asdict, dataclass
from typing import Optional

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
from rich.console import Console
from rich.panel import Panel

from config import cfg
from escalation import EscalationDecision, decide_escalation
from intent_taxonomy import INTENT_NAMES, get_classifier_block

_client  = genai.Client(api_key=cfg.google_api_key)
console  = Console()


# ── Response dataclass ─────────────────────────────────────────────────────

@dataclass
class AgentResponse:
    message:                 str
    intent:                  str
    intent_confidence:       float
    calibrated_confidence:   float
    intent_reasoning:        str
    retrieved_examples:      list[dict]
    draft_reply:             str
    reply_char_count:        int
    escalation_decision:     str    # "auto" | "escalate"
    escalation_reason:       str
    escalation_confidence:   float
    escalation_triggered_by: str


def calibrate_confidence(raw_conf: float) -> float:
    """
    Map raw LLM verbalized confidence to empirical calibrated probability.
    Derived from the calibration set (cal_100 + cal2_100).
    Eliminates ECE overconfidence compression in [0.80, 1.00].
    """
    if raw_conf >= 0.98:
        return 0.962
    elif raw_conf >= 0.93:
        return 0.773
    elif raw_conf >= 0.88:
        return 0.571
    elif raw_conf >= 0.82:
        return 0.655
    else:
        return 0.800


# ══════════════════════════════════════════════════════════════════════════
# Step 1 — Intent Classifier (gemini-3.1-flash-lite)
# ══════════════════════════════════════════════════════════════════════════

_CLASSIFIER_PROMPT = """\
You are an intent classifier for Amazon customer support on Twitter.

Classify the customer message into EXACTLY ONE of these intents:
{intent_block}

Customer message: "{message}"

Disambiguation Rules:
- Pick the single most specific matching intent.
- If nothing specific fits, use GENERAL_COMPLAINT.
- PRIME_SUBSCRIPTION vs ACCOUNT_ACCESS: Any unexpected Prime membership fee, accidental subscription charge, Prime trial, or Prime cancellation is PRIME_SUBSCRIPTION, UNLESS the customer explicitly reports stolen credentials or locked account login (ACCOUNT_ACCESS).
- ORDER_STATUS vs RETURN_REFUND: Tracking a replacement package or delayed shipment belongs to ORDER_STATUS. Requesting a return, refund, or asking how to exchange an item belongs to RETURN_REFUND.
- Confidence: Score your calibrated certainty (0.0 to 1.0). If the message is ambiguous or spans multiple intents, reflect this with confidence <= 0.75 rather than default high confidence.

Respond ONLY as strict JSON (no markdown, no extra keys):
{{"intent": "INTENT_NAME", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def classify_intent(message: str) -> dict:
    """Classify customer message into one of 7 Amazon intents."""
    response = _client.models.generate_content(
        model    = cfg.classifier_model,
        contents = _CLASSIFIER_PROMPT.format(
            intent_block = get_classifier_block(),
            message      = message[:500],
        ),
        config = types.GenerateContentConfig(
            response_mime_type = "application/json",
            temperature        = 0.0,
        ),
    )
    result = json.loads(response.text)

    intent = str(result.get("intent", "GENERAL_COMPLAINT")).upper().strip()
    if intent not in INTENT_NAMES:
        intent = "GENERAL_COMPLAINT"

    raw_conf = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
    cal_conf = calibrate_confidence(raw_conf)

    return {
        "intent":                intent,
        "confidence":            raw_conf,
        "calibrated_confidence": cal_conf,
        "reasoning":             str(result.get("reasoning", "")),
    }


# ══════════════════════════════════════════════════════════════════════════
# Step 2 — Qdrant Retriever
# ══════════════════════════════════════════════════════════════════════════

def retrieve_examples(qdrant_client, message: str, intent: str) -> list[dict]:
    """Retrieve top-K historical AmazonHelp reply pairs filtered by intent."""
    from qdrant_store import retrieve
    return retrieve(client=qdrant_client, query=message, intent_filter=intent)


def _format_examples(examples: list[dict]) -> str:
    if not examples:
        return "No historical examples available."
    blocks = []
    for i, ex in enumerate(examples, 1):
        blocks.append(
            f"[Example {i}]\n"
            f"  Customer : {ex.get('customer_msg', '')[:200]}\n"
            f"  Amazon   : {ex.get('amazon_reply', '')[:220]}\n"
            f"  Resolved : {ex.get('resolution_type', 'unknown')}"
        )
    return "\n\n".join(blocks)


# ══════════════════════════════════════════════════════════════════════════
# Step 3 — Reply Drafter (gemini-3.1-flash-lite)
# ══════════════════════════════════════════════════════════════════════════

_DRAFTER_PROMPT = """\
You are an Amazon customer support agent replying on Twitter (@AmazonHelp).

TONE   : empathetic, concise, helpful, professional
RULES  :
  1. Reply MUST be 280 characters or fewer — Twitter hard limit.
  2. NEVER promise specific refund amounts or delivery dates you cannot verify.
  3. NEVER reference or ask for sensitive personal data in the public tweet.
  4. ALWAYS acknowledge the customer's frustration or issue first.
  5. ALWAYS give a concrete next step.
  6. If resolution requires private account info, invite them to DM.
  7. NEVER include URLs or links of any kind — you have no verified links
     to give, and invented short-links are a hallucination. Say "send us
     a DM" with plain words instead.

Customer issue:
  Intent  : {intent}
  Message : "{message}"

Historical examples of how Amazon resolved similar {intent} issues:
{examples}

Write ONLY the reply text. No quotes, no prefix, no explanation:"""


_URL_RE = re.compile(r"https?://\S+|www\.\S+")


def sanitize_reply(reply: str) -> str:
    """
    Deterministic URL guardrail (pure function — unit-tested).
    The model has no verified links, so any URL it emits is a hallucination
    (observed: invented t.co DM/chat links). Strip them rather than trusting
    prompt compliance alone.
    """
    reply = _URL_RE.sub("", reply)
    return re.sub(r"\s{2,}", " ", reply).strip()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def draft_reply(message: str, intent: str, examples: list[dict]) -> str:
    """Draft a <=280-char Twitter reply grounded in retrieved examples."""
    # temperature=0.0: deterministic drafts. Rationale: run-to-run variance
    # would silently invalidate human scores and judge comparisons across
    # eval runs; consistency beats marginal creativity here.
    response = _client.models.generate_content(
        model    = cfg.drafter_model,
        contents = _DRAFTER_PROMPT.format(
            intent   = intent,
            message  = message,
            examples = _format_examples(examples),
        ),
        config = types.GenerateContentConfig(temperature=0.0),
    )
    reply = response.text.strip().strip('"').strip("'")
    reply = sanitize_reply(reply)

    # Hard-trim safety net
    if len(reply) > 280:
        reply = reply[:277] + "..."

    return reply


# ══════════════════════════════════════════════════════════════════════════
# Full pipeline
# ══════════════════════════════════════════════════════════════════════════

def run_agent(
    message: str,
    qdrant_client=None,
    thread_length: int = 1,
    verbose: bool = False,
) -> AgentResponse:
    """
    Run the full Amazon Support Agent pipeline.

    Args:
        message        : raw customer tweet text
        qdrant_client  : connected QdrantClient (None = skip retrieval)
        thread_length  : conversation turns so far (affects escalation rules)
        verbose        : print step-by-step to console

    Returns:
        AgentResponse with all pipeline outputs
    """
    if verbose:
        console.print(Panel(f"[bold cyan]Processing:[/bold cyan] {message}"))

    # Step 1: Classify
    clf        = classify_intent(message)
    intent     = clf["intent"]
    confidence = clf["confidence"]
    if verbose:
        console.print(
            f"[Step 1] [yellow]Intent:[/yellow] {intent} "
            f"({confidence:.0%}) — {clf['reasoning']}"
        )

    # Step 2: Retrieve
    examples: list[dict] = []
    if qdrant_client is not None:
        examples = retrieve_examples(qdrant_client, message, intent)
        if verbose:
            console.print(f"[Step 2] [yellow]Retrieved:[/yellow] {len(examples)} examples")
    elif verbose:
        console.print("[Step 2] [dim]Retrieval skipped (no Qdrant client)[/dim]")

    # Step 3: Draft
    reply = draft_reply(message, intent, examples)
    if verbose:
        color = "green" if len(reply) <= 240 else "yellow" if len(reply) <= 280 else "red"
        console.print(f"[Step 3] [yellow]Reply:[/yellow] {reply}")
        console.print(f"         [{color}]({len(reply)}/280 chars)[/{color}]")

    # Step 4: Escalation
    esc: EscalationDecision = decide_escalation(message, intent, confidence, thread_length)
    if verbose:
        color = "red" if esc.decision == "escalate" else "green"
        console.print(
            f"[Step 4] [{color}]{esc.decision.upper()}[/{color}] "
            f"({esc.triggered_by}) — {esc.reason}"
        )

    return AgentResponse(
        message                  = message,
        intent                   = intent,
        intent_confidence        = confidence,
        calibrated_confidence    = clf.get("calibrated_confidence", confidence),
        intent_reasoning         = clf["reasoning"],
        retrieved_examples       = examples,
        draft_reply              = reply,
        reply_char_count         = len(reply),
        escalation_decision      = esc.decision,
        escalation_reason        = esc.reason,
        escalation_confidence    = esc.confidence,
        escalation_triggered_by  = esc.triggered_by,
    )


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Amazon Support Agent — single message")
    parser.add_argument("--message",       required=True, type=str)
    parser.add_argument("--no-qdrant",     action="store_true")
    parser.add_argument("--thread-length", type=int, default=1)
    args = parser.parse_args()

    qdrant_client = None
    if not args.no_qdrant:
        from qdrant_store import get_client
        qdrant_client = get_client()

    result = run_agent(
        message       = args.message,
        qdrant_client = qdrant_client,
        thread_length = args.thread_length,
        verbose       = True,
    )
    import pprint
    console.print("\n[bold]Final AgentResponse:[/bold]")
    pprint.pprint(asdict(result), sort_dicts=False)


if __name__ == "__main__":
    main()
