"""
scripts/checks/inspect_fps.py
─────────────────────────────
Inspects and categorizes false positive escalation triggers.
When run: Executed to analyze customer delay venting vs true repeat tickets.
Reproducibility: Repeatable diagnostic script.
"""

import sys
from pathlib import Path

# Project root (two directories up from scripts/<subdir>/)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sys
import pandas as pd
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path("d:/NExp2-13")
fps = {
    'golden_250.csv': ['GS_0022', 'GS_0086', 'GS_0134', 'GS_0191', 'GS_0200', 'GS_0234', 'GS_0242', 'GS_0244'],
    'heldout_50.csv': ['H_0021', 'H_0031', 'H_0045', 'H_0046'],
    'heldoutB_50.csv': ['HB_0048', 'HB_0049']
}

for f, ids in fps.items():
    df = pd.read_csv(ROOT / "golden_set" / f).set_index('message_id')
    print("=" * 60)
    print(f, f"({len(ids)} false escalations)")
    print("=" * 60)
    for mid in ids:
        if mid in df.index:
            r = df.loc[mid]
            print(f"[{mid}] {r['text']}")
            print(f"   Gold Intent: {r.get('true_intent', 'N/A')} | Escalate Gold: {r.get('escalate_yn', 'N/A')}")
