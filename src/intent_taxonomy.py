"""
src/intent_taxonomy.py
──────────────────────
Single source of truth for intent definitions.
All classifiers, evaluators, and label builders import from here.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Intent:
    name: str
    label: int
    description: str
    trigger_keywords: List[str]
    examples: List[str]


# ── 7 Amazon-specific intents derived from EDA ─────────────────────────────

INTENTS: List[Intent] = [
    Intent(
        name="ORDER_STATUS",
        label=0,
        description=(
            "Customer asking about order tracking, delivery time, shipment "
            "updates, estimated arrival, or order not showing in their account."
        ),
        trigger_keywords=[
            "order", "tracking", "shipped", "delivery", "arrive", "package",
            "where is", "when will", "estimated", "dispatch",
        ],
        examples=[
            "Where is my order? It's been 5 days!",
            "My tracking number says delivered but I got nothing.",
            "Can you tell me when my package will arrive?",
            "Order #112-XXXXXXX hasn't shipped yet.",
        ],
    ),
    Intent(
        name="RETURN_REFUND",
        label=1,
        description=(
            "Customer requesting a return, refund, replacement, or reporting "
            "a wrong item / incorrect product delivered."
        ),
        trigger_keywords=[
            "return", "refund", "wrong item", "incorrect", "replacement",
            "money back", "exchange", "send back", "reimburs",
        ],
        examples=[
            "I want to return this product.",
            "I received the wrong item. How do I get a refund?",
            "My refund still hasn't hit my account after 10 days.",
            "Can I exchange this for a different size?",
        ],
    ),
    Intent(
        name="ACCOUNT_ACCESS",
        label=2,
        description=(
            "Customer unable to log in, account locked/suspended, "
            "password reset issues, 2FA problems, or unauthorized account activity."
        ),
        trigger_keywords=[
            "log in", "login", "sign in", "password", "locked", "suspended",
            "account", "access", "verify", "2FA", "hacked", "unauthorized",
        ],
        examples=[
            "I can't log into my Amazon account.",
            "My account has been locked and I don't know why.",
            "Someone got into my account – please help!",
            "I'm not receiving the OTP for sign-in.",
        ],
    ),
    Intent(
        name="PRIME_SUBSCRIPTION",
        label=3,
        description=(
            "Customer asking about Amazon Prime membership: billing, "
            "cancellation, benefits not working, free trial, or accidental charge."
        ),
        trigger_keywords=[
            "prime", "subscription", "membership", "cancel", "free trial",
            "charged", "benefit", "prime video", "prime music", "renew",
        ],
        examples=[
            "How do I cancel my Prime subscription?",
            "I was charged for Prime but I thought I cancelled.",
            "My Prime free shipping isn't working.",
            "Prime Video isn't showing in my benefits.",
        ],
    ),
    Intent(
        name="DEVICE_TECH_SUPPORT",
        label=4,
        description=(
            "Customer reporting issues with Amazon hardware devices: "
            "Alexa, Echo, Kindle, Fire TV, Ring, or related apps."
        ),
        trigger_keywords=[
            "alexa", "echo", "kindle", "fire tv", "fire stick", "ring",
            "device", "not working", "won't turn on", "app", "update",
            "reset", "connect", "wifi", "bluetooth",
        ],
        examples=[
            "My Alexa isn't responding to commands.",
            "Kindle won't charge even with the original cable.",
            "Fire TV keeps crashing after the latest update.",
            "Echo dot won't connect to my WiFi.",
        ],
    ),
    Intent(
        name="DELIVERY_DAMAGE",
        label=5,
        description=(
            "Customer reporting that a delivered package or item inside "
            "was damaged, missing items from the box, or delivered to wrong address."
        ),
        trigger_keywords=[
            "damaged", "broken", "crushed", "missing from box", "open box",
            "wrong address", "tampered", "seal broken", "arrived damaged",
        ],
        examples=[
            "My package arrived completely crushed.",
            "The box was open and one item is missing.",
            "Delivered to wrong address – neighbour has my package.",
            "Product inside was broken, box looked fine.",
        ],
    ),
    Intent(
        name="GENERAL_COMPLAINT",
        label=6,
        description=(
            "Vague frustration or anger with no clearly actionable request. "
            "Customer venting, threatening to leave, or expressing disappointment "
            "without specifying a concrete resolvable issue."
        ),
        trigger_keywords=[
            "worst", "terrible", "horrible", "never again", "awful",
            "disappointed", "frustrated", "disgusted", "fed up",
        ],
        examples=[
            "Amazon has the WORST customer service ever.",
            "I'm done with Amazon after this experience.",
            "This is absolutely ridiculous!",
            "I am beyond frustrated right now.",
        ],
    ),
]

# Convenience lookups
INTENT_BY_NAME = {i.name: i for i in INTENTS}
INTENT_BY_LABEL = {i.label: i for i in INTENTS}
INTENT_NAMES = [i.name for i in INTENTS]
NUM_INTENTS = len(INTENTS)


def get_classifier_block() -> str:
    """Return formatted intent definitions for LLM classifier prompt."""
    lines = ["INTENTS — pick exactly one:"]
    for idx, intent in enumerate(INTENTS, 1):
        lines.append(f"{idx}. {intent.name} — {intent.description}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(get_classifier_block())
    print(f"\nTotal intents: {NUM_INTENTS}")
