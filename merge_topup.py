"""Merge the 50 rare-class top-up rows into golden_250.csv (200 -> 250).

Same two-pass pipeline as the original 200: LLM first-pass labels, then the
recorded human corrections below (labelled row-by-row, frozen definition).
New thread_ids come from OUTSIDE the 10k corpus (never indexed).
"""

import json
import sys
import time

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, "src")
from eval.golden_builder import _llm_label  # noqa: E402

# message_id -> (true_intent, escalate_yn); all 50 rows listed explicitly.
HUMAN = {
    "GS_T001": ("DEVICE_TECH_SUPPORT", True),
    "GS_T002": ("DEVICE_TECH_SUPPORT", False),
    "GS_T003": ("DELIVERY_DAMAGE", False),
    "GS_T004": ("PRIME_SUBSCRIPTION", False),
    "GS_T005": ("GENERAL_COMPLAINT", False),
    "GS_T006": ("ACCOUNT_ACCESS", False),
    "GS_T007": ("DELIVERY_DAMAGE", True),
    "GS_T008": ("GENERAL_COMPLAINT", False),
    "GS_T009": ("DEVICE_TECH_SUPPORT", False),
    "GS_T010": ("DEVICE_TECH_SUPPORT", False),
    "GS_T011": ("ORDER_STATUS", True),
    "GS_T012": ("ORDER_STATUS", True),
    "GS_T013": ("DEVICE_TECH_SUPPORT", False),
    "GS_T014": ("ORDER_STATUS", True),
    "GS_T015": ("PRIME_SUBSCRIPTION", False),
    "GS_T016": ("PRIME_SUBSCRIPTION", False),
    "GS_T017": ("ORDER_STATUS", True),
    "GS_T018": ("ORDER_STATUS", False),
    "GS_T019": ("GENERAL_COMPLAINT", False),
    "GS_T020": ("ORDER_STATUS", False),
    "GS_T021": ("DEVICE_TECH_SUPPORT", False),
    "GS_T022": ("DEVICE_TECH_SUPPORT", False),
    "GS_T023": ("DEVICE_TECH_SUPPORT", False),
    "GS_T024": ("DEVICE_TECH_SUPPORT", False),
    "GS_T025": ("ORDER_STATUS", False),
    "GS_T026": ("GENERAL_COMPLAINT", False),
    "GS_T027": ("GENERAL_COMPLAINT", False),
    "GS_T028": ("DELIVERY_DAMAGE", False),
    "GS_T029": ("DEVICE_TECH_SUPPORT", False),
    "GS_T030": ("DEVICE_TECH_SUPPORT", False),
    "GS_T031": ("DEVICE_TECH_SUPPORT", False),
    "GS_T032": ("ORDER_STATUS", False),
    "GS_T033": ("DEVICE_TECH_SUPPORT", False),
    "GS_T034": ("ORDER_STATUS", False),
    "GS_T035": ("PRIME_SUBSCRIPTION", False),
    "GS_T036": ("PRIME_SUBSCRIPTION", False),
    "GS_T037": ("PRIME_SUBSCRIPTION", False),
    "GS_T038": ("PRIME_SUBSCRIPTION", False),
    "GS_T039": ("PRIME_SUBSCRIPTION", False),
    "GS_T040": ("PRIME_SUBSCRIPTION", False),
    "GS_T041": ("PRIME_SUBSCRIPTION", False),
    "GS_T042": ("PRIME_SUBSCRIPTION", False),
    "GS_T043": ("PRIME_SUBSCRIPTION", False),
    "GS_T044": ("PRIME_SUBSCRIPTION", False),
    "GS_T045": ("DEVICE_TECH_SUPPORT", False),
    "GS_T046": ("ACCOUNT_ACCESS", True),
    "GS_T047": ("PRIME_SUBSCRIPTION", False),
    "GS_T048": ("ORDER_STATUS", True),
    "GS_T049": ("PRIME_SUBSCRIPTION", False),
    "GS_T050": ("PRIME_SUBSCRIPTION", False),
}


def main() -> None:
    cand = pd.read_csv("golden_set/topup_texts.csv").set_index("message_id")
    assert len(cand) == 50 and set(cand.index) == set(HUMAN)
    golden = pd.read_csv("golden_set/golden_250.csv")
    assert len(golden) == 200
    assert not set(cand["thread_id"].astype(str)) & set(
        golden["thread_id"].astype(str))

    rows = []
    for mid in tqdm(sorted(cand.index,
                           key=lambda m: int(m.split("_T")[1])), desc="LLM first pass"):
        text = str(cand.loc[mid, "text"])
        lab = _llm_label(text)
        intent, esc = HUMAN[mid]
        rows.append({
            "message_id": f"GS_{200+int(mid.split('_T')[1]):04d}",
            "thread_id": cand.loc[mid, "thread_id"],
            "text": text,
            "proposed_intent": "TOPUP_RARE_CLASS",
            "llm_intent": lab["intent"],
            "llm_confidence": lab["confidence"],
            "true_intent": intent,
            "escalate_yn": esc,
            "escalate_reason": lab["escalate_reason"],
            "labelled_by": "human_direct_v1",
            "human_verified": True,
        })
        time.sleep(0.05)
    out = pd.concat([golden, pd.DataFrame(rows)], ignore_index=True)
    assert len(out) == 250
    out.to_csv("golden_set/golden_250.csv", index=False)
    print("golden now:", len(out))
    print(out["true_intent"].value_counts().to_dict())
    print("escalate rate:", round(float(out["escalate_yn"].mean()), 3))


if __name__ == "__main__":
    main()
