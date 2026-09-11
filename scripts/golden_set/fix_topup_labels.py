"""
scripts/golden_set/fix_topup_labels.py
──────────────────────────────────────
Normalizes metadata columns and boolean formats on top-up evaluation rows.
When run: Executed after merging rare-class rows.
Reproducibility: Repeatable maintenance script.
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

FIX = {
    "GS_0201": ("DEVICE_TECH_SUPPORT", False),  # T001 smart-skills loss
    "GS_0202": ("DEVICE_TECH_SUPPORT", False),  # T002 Echo Show question
    "GS_0203": ("ORDER_STATUS", False),         # T003 non-delivery, wrong number
    "GS_0204": ("GENERAL_COMPLAINT", False),    # T004 vague reply-fragment
    "GS_0205": ("GENERAL_COMPLAINT", False),    # T005 book poetry
    "GS_0206": ("GENERAL_COMPLAINT", False),    # T006 thanks + suggestion
    "GS_0207": ("ACCOUNT_ACCESS", True),        # T007 fraud charge claim
    "GS_0208": ("DEVICE_TECH_SUPPORT", False),  # T008 computer-vs-phone app
    "GS_0209": ("DEVICE_TECH_SUPPORT", False),  # T009 smart home
    "GS_0210": ("DEVICE_TECH_SUPPORT", False),  # T010 German Kindle wrong books
    "GS_0211": ("GENERAL_COMPLAINT", False),    # T011 AMZL feedback note
    "GS_0212": ("GENERAL_COMPLAINT", False),    # T012 cart price bug
    "GS_0213": ("DEVICE_TECH_SUPPORT", False),  # T013 Echo tap bluetooth
    "GS_0214": ("DEVICE_TECH_SUPPORT", False),  # T014 app popups
    "GS_0215": ("PRIME_SUBSCRIPTION", False),   # T015 prime disappointed
    "GS_0216": ("DELIVERY_DAMAGE", False),      # T016 Hermes smashed TV
    "GS_0217": ("GENERAL_COMPLAINT", False),    # T017 pay-trap warning
    "GS_0218": ("ORDER_STATUS", False),         # T018 Kindle case not sent
}

def main() -> None:
    p = "golden_set/golden_250.csv"
    df = pd.read_csv(p).set_index("message_id")
    for mid, (intent, esc) in FIX.items():
        df.loc[mid, "true_intent"] = intent
        df.loc[mid, "escalate_yn"] = esc
        df.loc[mid, "labelled_by"] = "human_direct_v1_fix1"
    df.reset_index().to_csv(p, index=False)
    print(df["true_intent"].value_counts().to_dict())
    print("escalate:", int(df["escalate_yn"].sum()), "min class:", df["true_intent"].value_counts().min())

if __name__ == "__main__":
    main()
