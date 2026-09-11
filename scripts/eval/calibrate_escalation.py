"""
scripts/eval/calibrate_escalation.py
────────────────────────────────────
Offline calibration sweep of regex rules and confidence thresholds against human labels.
When run: Executed during v4->v5 rule calibration.
Reproducibility: Repeatable offline calibration script.
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

import re
import sys

import pandas as pd

from escalation import check_hard_rules  # noqa: E402

ORDER_ID = re.compile(r"\b\d{3}[\s\-]?\d{7}[\s\-]?\d{7}\b")

FRAUD = re.compile(
    r"(fraud|scam|phish|legit|suspicious|spam|unauthorized|hacked"
    r"|disput\w*|chargeback"
    r"|cheat\w*.{0,60}(money|charge|refund|account|bank|payment|balance)"
    r"|(money|charge|refund|balance).{0,60}cheat\w*"
    r"|duplicate (charge|payment)|randomly charg\w+"
    r"|charg\w+.{0,30}(nothing.{0,10}ordered|never ordered|didn.?t order|not mine)"
    r"|never.{0,20}ordered|didn.?t order)",
    re.IGNORECASE,
)

# Explicit prior support contact / failed handling (requires a support actor
# or handling noun nearby -- bare "second time" / "promised" excluded).
REPEAT = re.compile(
    r"(contacted|called|emailed|chatted|reported).{0,25}(support|service|team|"
    r"helpline|call cent(er|re)|tech support|customer care|bank|you|them|amazon)"
    r"|(wrote|sent|write).{0,20}(email|mail|letter|message)"
    r"|\bCC\b.{0,25}(rude|unprof|behav|call)"
    r"|(contact|call|email)\w*.{0,15}(twice|thrice|\d+\s?times)"
    r"|(several|\d+|ten).{0,10}calls|plus calls|multiple tweets"
    r"|thrice|follow\w*.{0,3}up"
    r"|(already|aready|alredy).{0,15}(said|told|asked|contacted|reported)"
    r"|(said|told).{0,15}(already|aready|alredy)"
    r"|no[ -]?(one|body).{0,20}(help|answer|handle|respond|resolve|care)"
    r"|talking to a robot|like a robot|not.*robot"
    r"|disconnect\w+|on hold|supervisor|hung up|raccrocher|appeler.{0,10}fois"
    r"|helpline.{0,20}not connect|call back.{0,20}(not|isn.?t|fail)"
    r"|apolog\w+.{0,10}(mail|email|letter)|assur\w+.{0,15}(resolution|resolve|refund)"
    r"|trying.{0,15}for\s+(two|three|\d+)"
    r"|for (two|three|\d+) weeks"
    r"|tired of (calling|chatting|waiting)"
    r"|tri\w+ getting.{0,20}(refund|response|answer|through)"
    r"|can.?t get.{0,20}(refund|replacement|response|answer|through)"
    r"|ignor\w+.{0,15}(my|me|issue|claim|request|ticket|us)"
    r"|(denied|refused|refusing).{0,25}(refund|return|replacement|help|report)"
    r"|(support|service|team|care).{0,15}(helpless|useless|hopeless)"
    r"|escalat\w*|harass\w*|supervisor",
    re.IGNORECASE,
)

HUMAN_ASK = re.compile(
    r"\b(real (person|human)|speak to (someone|a human)|phone number to call|"
    r"don.?t want generic|talk to (anyone|somebody))\b",
    re.IGNORECASE,
)

def pr(y_true, y_pred, name):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    print(f"{name:22s} P={prec:.3f} R={rec:.3f} (tp={tp} fp={fp} fn={fn})")
    return prec, rec

def main():
    g = pd.read_csv(ROOT / "golden_set" / "golden_250.csv").set_index("message_id")
    r = pd.read_csv(ROOT / "results" / "agent_replies.csv").set_index("message_id")
    texts = g["text"].astype(str).tolist()
    y = g["escalate_yn"].astype(bool).tolist()

    cur = [
        check_hard_rules(t, r.loc[mid, "pred_intent"],
                         float(r.loc[mid, "intent_confidence"])) is not None
        for mid, t in zip(g.index, texts)
    ]
    pr(y, cur, "current hard rules")
    full = [c or bool(ORDER_ID.search(t)) or bool(FRAUD.search(t))
            or bool(REPEAT.search(t)) or bool(HUMAN_ASK.search(t))
            for c, t in zip(cur, texts)]
    pr(y, full, "current+ALL v2")

    print("\nFalse negatives:")
    for mid, t, yt, p in zip(g.index, texts, y, full):
        if yt and not p:
            print(" -", mid, "|", " ".join(t.split())[:130])
    print("\nFalse positives (rule attribution):")
    for mid, t, yt, p in zip(g.index, texts, y, full):
        if p and not yt:
            fired = [n for n, rx in [("cur", None), ("oid", ORDER_ID),
                                     ("fraud", FRAUD), ("rep", REPEAT),
                                     ("ask", HUMAN_ASK)]
                     if (rx is None and cur[list(g.index).index(mid)])
                     or (rx is not None and rx.search(t))]
            print(" -", mid, fired, "|", " ".join(t.split())[:110])

if __name__ == "__main__":
    main()
