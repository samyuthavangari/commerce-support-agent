"""
scripts/retrieval_eval.py
Compatibility forwarder -> scripts/checks/retrieval_eval.py
"""
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "checks" / "retrieval_eval.py"
if __name__ == "__main__":
    import runpy
    runpy.run_path(str(TARGET), run_name="__main__")
