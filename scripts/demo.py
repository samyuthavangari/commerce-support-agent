"""
scripts/demo.py
───────────────
Convenience wrapper pointing to the root demo.py diagnostic script.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import demo

if __name__ == "__main__":
    demo.main()
