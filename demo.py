"""
demo.py
───────
Interactive CLI demo for the Amazon Support AI Agent.

Generates comprehensive diagnostic reports for incoming customer queries:
  - Predicted intent & confidence
  - Top-k retrieved historical examples with similarity scores
  - Retrieval agreement & max similarity
  - Grounding verification & score
  - Risk trigger detection (deterministic rules)
  - Final decision ([AUTO-HANDLE] vs [ESCALATE TO HUMAN])
  - Reason code & explanation
  - Grounded <=280-char drafted reply

Usage:
  python demo.py                               # Interactive mode (prompts for input or presets)
  python demo.py --query "Where is my order?"  # CLI query input
  python demo.py --preset amz-001              # Run preset query (amz-001 to amz-005)
  python demo.py --no-qdrant                   # Run without vector retrieval
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

# Suppress noisy library version mismatch warnings
warnings.filterwarnings("ignore")

# UTF-8 encoding configuration for Windows console compatibility
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import cfg
from agent import classify_intent, draft_reply, sanitize_reply, calibrate_confidence
from escalation import check_hard_rules, EscalationDecision
from intent_taxonomy import INTENT_NAMES

# ── Standard Presets ────────────────────────────────────────────────────────
PRESETS = {
    "amz-001": (
        "say they put my parcel through my letterbox on saturday. "
        "Still waiting for it to come out the other side 😠 <URL>"
    ),
    "amz-002": (
        "I will sue your company and contact my lawyer if this unauthorized "
        "charge of $149 is not refunded immediately."
    ),
    "amz-003": (
        "Order 408-1899080-9287553 says delivered yesterday but nothing "
        "arrived at my door. Can you check where it was left?"
    ),
    "amz-004": (
        "I have called support three times and chatted twice. Still no update "
        "on my replacement. Is there a real human I can speak to?"
    ),
    "amz-005": (
        "How do I exchange my running shoes for a size 9? The current ones "
        "are too small for me."
    ),
    "amz-006": (
        "Where is my replacement item that was supposed to arrive today?"
    ),
    "amz-007": (
        "One fladuent activity happen in my card from amazon at name of prime membership with transaction of 6850 rs"
    ),
    "amz-008": (
        "Wow amazing job Amazon! Threw my package directly into my swimming pool on a sunny day! 🏊 Great service!"
    ),
    "amz-009": (
        "Amazon charged my card $99 for prime without permission!"
    ),
}



def get_qdrant_client():
    """Attempt to connect to Qdrant with graceful fallback."""
    try:
        from qdrant_store import get_client
        client = get_client()
        client.get_collection(cfg.qdrant_collection)
        return client
    except Exception as exc:
        print(f"[Note] Qdrant not reachable ({exc}). Running in offline retrieval mode.\n")
        return None


def calculate_grounding(
    draft: str,
    query: str,
    examples: list[dict],
    max_sim: float,
    agreement: float,
) -> tuple[bool, float]:
    """
    Evaluate reply grounding against retrieved historical precedent.
    Returns (passed: bool, score: float on 1.0 - 5.0 scale).
    """
    # Base grounded score
    score = 4.5

    # Check for hallucinated URLs
    if "http" in draft or "www." in draft or "t.co" in draft:
        score -= 2.5

    # If retrieval is weak or contradictory, grounding cannot be verified
    if max_sim < 0.50:
        score -= 2.0
    elif max_sim < 0.65:
        score -= 0.8

    if agreement < 0.30 and len(examples) >= 3:
        score -= 1.5
    elif agreement < 0.60 and len(examples) >= 3:
        score -= 0.6

    # Penalty if reply asks for sensitive card/SSN data
    lower_draft = draft.lower()
    if any(k in lower_draft for k in ["credit card", "password", "ssn", "cvv"]):
        score = 1.0

    score = max(1.0, min(5.0, round(score, 2)))
    passed = score >= 3.0
    return passed, score


def map_reason_code(
    trigger: Optional[str],
    agreement: float,
    confidence: float,
    calibrated_confidence: float,
    grounding_passed: bool,
    max_sim: float = 0.0,
) -> tuple[str, str]:
    """Map trigger or evidence signals to standardized reason codes and descriptions."""
    if trigger == "rule:legal":
        return "LEGAL_RISK", "Legal or regulatory action mentioned; requires specialized human support."
    elif trigger == "rule:fraud":
        return "FRAUD_DETECTION", "Fraud, unauthorized charge, or security breach reported."
    elif trigger == "rule:public_pii":
        return "PUBLIC_PII", "Order number or phone posted publicly in tweet; human must triage and advise removal."
    elif trigger == "rule:pii":
        return "PII_EXPOSURE", "Customer sensitive credentials (card/SSN) detected in public message."
    elif trigger == "rule:repeat_contact":
        return "REPEAT_CONTACT", "Prior failed support contact referenced; customer is looping without resolution."
    elif trigger == "rule:human_ask":
        return "HUMAN_REQUEST", "Customer explicitly asked for a human agent or rejected automated assistance."
    elif trigger == "rule:safety":
        return "SAFETY_CRITICAL", "Safety, self-harm, or emergency distress signal detected."
    elif trigger == "rule:unclear":
        return "UNCLEAR_MESSAGE", "Message contains only links/mentions without interpretable customer issue."
    elif trigger == "rule:thread_length":
        return "UNRESOLVED_THREAD", "Conversation exceeded turn threshold without resolution."
    elif trigger in ("llm:ambiguous", "rule:ambiguous"):
        return "AMBIGUOUS_QUERY", "Message lacks necessary order details or context for automated resolution."
    elif trigger == "rule:low_confidence":
        return "LOW_CONFIDENCE", "Intent classification confidence below safety threshold."

    # Evidence & certainty calibration checks
    if not grounding_passed:
        return "UNGROUNDED_REPLY", "Draft reply failed grounding check against historical brand precedents."
    if agreement < 0.30:
        return "WEAK_EVIDENCE", f"precedent agrees {agreement:.2f} < 0.30 with predicted intent; likely misclassified"
    if agreement < 0.60:
        return (
            "CONTESTED_PRECEDENT",
            f"Precedent is contested (agreement {agreement:.2f} < 0.60); majority of historical precedents disagree."
        )
    if calibrated_confidence < 0.70:
        return "LOW_CONFIDENCE", f"Calibrated intent certainty {calibrated_confidence:.2f} < 0.70 safety threshold."

    return "SAFE_TO_HANDLE", "Grounded query confirmed by majority precedent agreement; safe for automated reply."


def run_demo_pipeline(
    query_text: str,
    query_id: str = "amz-custom",
    client=None,
    top_k: int = 5,
) -> dict:
    """Run full agent pipeline with detailed diagnostic telemetry."""
    from escalation import decide_escalation

    # 1. Intent Classification & Calibration Mapping
    clf = classify_intent(query_text)
    intent = clf["intent"]
    raw_confidence = clf["confidence"]
    calibrated_confidence = clf.get("calibrated_confidence", calibrate_confidence(raw_confidence))

    # 2. Retrieval (unfiltered top-k to test natural semantic precedent agreement)
    retrieved: list[dict] = []
    max_sim = 0.0
    agreement = 1.0

    if client is not None:
        try:
            from qdrant_store import retrieve
            # Open semantic retrieval to verify natural intent alignment
            retrieved = retrieve(client=client, query=query_text, intent_filter=None, top_k=top_k)
            if retrieved:
                max_sim = max((r.get("_similarity", 0.0) for r in retrieved), default=0.0)
                matching = sum(1 for r in retrieved if r.get("intent") == intent)
                agreement = matching / len(retrieved)
        except Exception:
            retrieved = []

    # 3. Deterministic Hard Rules / Risk Trigger
    esc = decide_escalation(query_text, intent, raw_confidence)
    risk_trigger = esc.triggered_by if (esc and esc.decision == "escalate") else "None"

    # 4. Draft Reply
    draft = draft_reply(query_text, intent, retrieved)

    # 5. Grounding Evaluation
    grounding_passed, grounding_score = calculate_grounding(draft, query_text, retrieved, max_sim, agreement)

    # 6. Calibrated Decision & Reason Code (Locked Policy)
    # Safe Auto-Handle requires:
    #   (1) No deterministic safety triggers (esc.decision != 'escalate')
    #   (2) Grounding verification passed (score >= 3.0)
    #   (3) Precedent agreement not contradictory (agreement >= 0.30)
    #   (4) If agreement is contested (< 0.60), require strict raw_confidence == 1.00 AND similarity >= 0.80
    #   (5) Calibrated confidence >= 0.70 (eliminating uncalibrated LLM 0.90/0.85 traps)
    if esc and esc.decision == "escalate":
        decision = "[ESCALATE TO HUMAN]"
        reason_code, reason = map_reason_code(esc.triggered_by, agreement, raw_confidence, calibrated_confidence, grounding_passed, max_sim)
        if esc.reason and reason_code in ("LEGAL_RISK", "PUBLIC_PII", "AMBIGUOUS_QUERY"):
            reason = esc.reason
    elif not grounding_passed:
        decision = "[ESCALATE TO HUMAN]"
        reason_code, reason = map_reason_code(None, agreement, raw_confidence, calibrated_confidence, grounding_passed, max_sim)
    elif agreement < 0.30:
        decision = "[ESCALATE TO HUMAN]"
        reason_code, reason = map_reason_code(None, agreement, raw_confidence, calibrated_confidence, grounding_passed, max_sim)
    elif agreement < 0.60 and (raw_confidence < 1.00 or max_sim < 0.80):
        decision = "[ESCALATE TO HUMAN]"
        reason_code, reason = map_reason_code(None, agreement, raw_confidence, calibrated_confidence, grounding_passed, max_sim)
    elif calibrated_confidence < 0.70:
        decision = "[ESCALATE TO HUMAN]"
        reason_code, reason = map_reason_code(None, agreement, raw_confidence, calibrated_confidence, grounding_passed, max_sim)
    else:
        decision = "[AUTO-HANDLE]"
        reason_code = "SAFE_TO_HANDLE"
        reason = f"High-certainty {intent} query confirmed by majority precedent agreement ({agreement:.0%})."

    return {
        "query_id": query_id,
        "query_text": query_text,
        "intent": intent,
        "confidence": raw_confidence,
        "calibrated_confidence": calibrated_confidence,
        "retrieval_sim": max_sim,
        "agreement": agreement,
        "risk_trigger": risk_trigger,
        "grounding_passed": grounding_passed,
        "grounding_score": grounding_score,
        "retrieved_examples": retrieved,
        "decision": decision,
        "reason_code": reason_code,
        "reason": reason,
        "draft_reply": draft,
    }


def print_report(res: dict, show_examples: bool = True) -> None:
    """Print the exact diagnostic report format requested."""
    sep = "-" * 65
    double_sep = "=" * 65

    clean_query = " ".join(str(res["query_text"]).replace("\r", " ").replace("\n", " ").split())
    clean_draft = " ".join(str(res["draft_reply"]).replace("\r", " ").replace("\n", " ").split())

    print(f"\n{double_sep}")
    print(f'Customer Query [{res["query_id"]}]: "{clean_query}"')
    print(sep)
    print(f'Predicted Intent : {res["intent"]} (raw conf: {res["confidence"]:.2f} -> calibrated: {res["calibrated_confidence"]:.2f})')
    print(f'Retrieval Sim    : {res["retrieval_sim"]:.3f} | Agreement: {res["agreement"]:.2f}')
    print(f'Risk Trigger     : {res["risk_trigger"]}')
    print(f'Grounding Passed : {res["grounding_passed"]} (score: {res["grounding_score"]:.2f})')
    print(sep)

    if show_examples and res["retrieved_examples"]:
        print(f"Top-{len(res['retrieved_examples'])} Retrieved Examples:")
        for idx, ex in enumerate(res["retrieved_examples"], 1):
            c_msg = " ".join(str(ex.get("customer_msg", "")).replace("\r", " ").replace("\n", " ").split())[:110]
            r_msg = " ".join(str(ex.get("amazon_reply", "")).replace("\r", " ").replace("\n", " ").split())[:110]
            sim = ex.get("_similarity", 0.0)
            ex_intent = ex.get("intent", "UNKNOWN")
            print(f"  [{idx}] (sim: {sim:.3f} | intent: {ex_intent})")
            print(f'      Cust: "{c_msg}..."')
            print(f'      Amzn: "{r_msg}..."')
        print(sep)

    print(f'Decision         : {res["decision"]}')
    print(f'Reason Code      : {res["reason_code"]}')
    print(f'Reason           : {res["reason"]}')
    print(sep)
    print("Drafted Reply:")
    print(f'  "{clean_draft}"')
    print(f"{double_sep}\n")


def interactive_menu(client) -> None:
    """Run user-friendly prompt loop."""
    print("\n" + "=" * 65)
    print(" 📦 Amazon Support AI Agent — Interactive Diagnostic Console")
    print("=" * 65)
    print("Select a sample query or type your own:\n")
    for pid, text in PRESETS.items():
        snippet = text if len(text) <= 65 else text[:62] + "..."
        print(f"  [{pid}] {snippet}")
    print("  [custom ] Type any customer message directly")
    print("-" * 65)

    while True:
        try:
            choice = input("Enter query text or preset ID [default: amz-001, 'q' to quit]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting demo.")
            break

        if choice.lower() in ("q", "quit", "exit"):
            break

        if not choice:
            query_id = "amz-001"
            query_text = PRESETS["amz-001"]
        elif choice in PRESETS:
            query_id = choice
            query_text = PRESETS[choice]
        else:
            query_id = "custom"
            query_text = choice

        print(f"\n[Processing query: '{query_text[:50]}...']")
        res = run_demo_pipeline(query_text=query_text, query_id=query_id, client=client)
        print_report(res)


def main() -> None:
    parser = argparse.ArgumentParser(description="Amazon Support Agent Diagnostic Demo")
    parser.add_argument("--query", type=str, default=None, help="Customer message text to analyze")
    parser.add_argument("--preset", type=str, default=None, choices=list(PRESETS.keys()), help="Run specific preset (amz-001 to amz-005)")
    parser.add_argument("--no-qdrant", action="store_true", help="Skip Qdrant vector retrieval")
    parser.add_argument("--hide-examples", action="store_true", help="Hide retrieved examples list in report")
    parser.add_argument("--k", type=int, default=5, help="Number of retrieved historical examples")
    args = parser.parse_args()

    client = None if args.no_qdrant else get_qdrant_client()

    if args.preset:
        res = run_demo_pipeline(query_text=PRESETS[args.preset], query_id=args.preset, client=client, top_k=args.k)
        print_report(res, show_examples=not args.hide_examples)
    elif args.query:
        res = run_demo_pipeline(query_text=args.query, query_id="amz-cli", client=client, top_k=args.k)
        print_report(res, show_examples=not args.hide_examples)
    else:
        interactive_menu(client)


if __name__ == "__main__":
    main()
