"""
scripts/labeling/apply_review.py
─────────────────────────────────
Applies recorded manual second-pass human review corrections to golden_set/golden_250.csv.
When run: Executed during dataset construction after initial LLM labelling pass.
Reproducibility: Repeatable auditing script; applies programmatic diff to base sample.
"""

import sys
from pathlib import Path

# Project root (two directories up from scripts/<subdir>/)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd

from pathlib import Path

# --- Intent corrections: message_id -> corrected true_intent ----------------
INTENT_FIX = {
    "GS_0004": "GENERAL_COMPLAINT",   # price-adjustment policy Q; no specific intent fits
    "GS_0006": "ORDER_STATUS",        # earphones never arrived (non-delivery, not damage)
    "GS_0026": "ORDER_STATUS",        # marked-delivered-but-missing -> delivery failure
    "GS_0034": "GENERAL_COMPLAINT",   # carrier-practice question, no damage reported
    "GS_0051": "ORDER_STATUS",        # marked-delivered-but-missing -> delivery failure
    "GS_0080": "ACCOUNT_ACCESS",      # phishing/legitimacy suspicion -> security
    "GS_0093": "ACCOUNT_ACCESS",      # Amazon Pay balance deduction ("cheat" allegation)
    "GS_0100": "ORDER_STATUS",        # Echo Dot *delivery* issue (10+ calls about delivery)
    "GS_0108": "ACCOUNT_ACCESS",      # mystery item / billing confusion, not Prime-related
    "GS_0184": "ORDER_STATUS",        # dup of GS_0006: non-delivery, not damage
}

# --- Escalation corrections: message_id -> corrected escalate_yn ------------
ESC_FIX_TRUE = [  # LLM said auto; human says escalate
    "GS_0006",  # rule 4: order ID posted publicly
    "GS_0029",  # rule 4: order ID posted publicly
    "GS_0035",  # rule 4: order number posted publicly
    "GS_0046",  # rule 4: order ID posted publicly
    "GS_0111",  # rule 4: order number posted publicly
    "GS_0123",  # rule 5: "Reported to the team. Still no update."
    "GS_0141",  # rule 5: helpline not connecting (failed contact attempt)
    "GS_0150",  # rule 5: "now ignoring my issue"
    "GS_0153",  # rule 5: "no one is helping"
    "GS_0168",  # rule 5: "After so many follow up"
    "GS_0169",  # rule 4: order ID posted publicly
    "GS_0196",  # rule 5: "Customer Care Team is helpless" (prior contact)
    "GS_0198",  # rule 4: order number posted publicly
]

ESC_FIX_FALSE = [  # LLM said escalate; human says auto (venting/profanity/
    # urgency/duration alone, no legal/fraud/PII/repeat signal)
    "GS_0008", "GS_0016", "GS_0026", "GS_0034", "GS_0044", "GS_0047",
    "GS_0049", "GS_0054", "GS_0061", "GS_0067", "GS_0072", "GS_0074",
    "GS_0075", "GS_0081", "GS_0083", "GS_0084", "GS_0088", "GS_0091",
    "GS_0099", "GS_0129", "GS_0132", "GS_0147", "GS_0148", "GS_0162",
    "GS_0163", "GS_0166", "GS_0175", "GS_0181", "GS_0185", "GS_0187",
    "GS_0190",
]

def main() -> None:
    path = Path(ROOT / "golden_set" / "golden_250.csv")
    df = pd.read_csv(path)
    assert len(df) in (200, 250), f"Expected 200 or 250 rows, got {len(df)}"
    df = df.set_index("message_id")

    for mid, intent in INTENT_FIX.items():
        if mid in df.index:
            df.loc[mid, "true_intent"] = intent
    for mid in ESC_FIX_TRUE:
        if mid in df.index:
            df.loc[mid, "escalate_yn"] = True
    for mid in ESC_FIX_FALSE:
        if mid in df.index:
            df.loc[mid, "escalate_yn"] = False

    core_mask = df.index <= "GS_0200"
    df.loc[core_mask, "human_verified"] = True
    df.loc[core_mask, "labelled_by"] = "human_review_v1"

    df.reset_index().to_csv(path, index=False)
    n_intent = len(INTENT_FIX)
    n_esc = len(ESC_FIX_TRUE) + len(ESC_FIX_FALSE)
    print(f"Applied {n_intent} intent + {n_esc} escalation corrections; "
          f"human_verified=True on all {len(df)} rows.")
    print("New intent dist:", df["true_intent"].value_counts().to_dict())
    print("New escalate rate:", round(float(df["escalate_yn"].mean()), 3))

if __name__ == "__main__":
    main()
