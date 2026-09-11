"""
scripts/check_leakage.py
Compatibility forwarder -> scripts/checks/check_leakage.py
"""
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "checks" / "check_leakage.py"
if __name__ == "__main__":
    import runpy
    runpy.run_path(str(TARGET), run_name="__main__")
