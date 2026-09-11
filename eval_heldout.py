"""Label + evaluate the held-out 50 with FROZEN rules (no tuning on this set).

Human labels use the same frozen 7-rule definition as the golden review.
The escalation patterns in src/escalation.py were NOT touched after seeing
these texts -- this measures generalization, not calibration.
"""

import sys

import pandas as pd

sys.path.insert(0, "src")
from escalation import check_hard_rules  # noqa: E402

# message_id -> True (all others False). Labelled blind to rule output,
# using the frozen definition from apply_review.py.
HELDOUT_TRUE = {
    "H_0006",  # hacked account mentioned
    "H_0008",  # "three times actually" -- repeat contact (truncated context)
    "H_0014",  # "contacted them by chat 3 times"
    "H_0015",  # order number posted publicly
    "H_0018",  # fraud/hacked-cards report
    "H_0019",  # duplicate charge ("charged me twice")
    "H_0022",  # hacked + email changed
    "H_0026",  # "still not heard back" -- failed contact
    "H_0028",  # order number posted publicly
    "H_0029",  # "spoke to you guys" + delayed again
    "H_0034",  # cannot reach a human (failed contact + human-ask)
    "H_0035",  # "still no call back" + prior CS contact
    "H_0036",  # order number posted publicly + broken promise
    "H_0047",  # partial refund, "told to wait" (prior contact)
    "H_0050",  # order number posted publicly
}


def main() -> None:
    df = pd.read_csv("golden_set/heldout_50_texts.csv")
    df["escalate_yn"] = df["message_id"].isin(HELDOUT_TRUE)
    df.to_csv("golden_set/heldout_50.csv", index=False)

    y = df["escalate_yn"].tolist()
    preds, trig = [], []
    for _, r in df.iterrows():
        hit = check_hard_rules(str(r["text"]), "GENERAL_COMPLAINT", 0.9)
        preds.append(hit is not None)
        trig.append(hit.triggered_by if hit else "-")

    tp = sum(1 for a, b in zip(y, preds) if a and b)
    fp = sum(1 for a, b in zip(y, preds) if not a and b)
    fn = sum(1 for a, b in zip(y, preds) if a and not b)
    print(f"HELD-OUT (frozen rules, n=50, {sum(y)} positives): "
          f"P={tp/max(tp+fp,1):.3f} R={tp/max(tp+fn,1):.3f} "
          f"(tp={tp} fp={fp} fn={fn})")
    print("\nMisses (rule gaps for future work -- NOT patched here):")
    for m, t, a, p, tr in zip(df["message_id"], df["text"], y, preds, trig):
        if a != p:
            print(f" {'FN' if a else 'FP'} {m} [{tr}] | "
                  + " ".join(str(t).split())[:115])


if __name__ == "__main__":
    main()
