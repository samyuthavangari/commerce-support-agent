"""
scripts/labeling/label_cal_intent.py
────────────────────────────────────
Adds human intent ground truth to golden_set/cal_100.csv.
When run: Executed to benchmark intent classification on calibration data.
Reproducibility: Repeatable dataset generation script.
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

CAL_INTENT = {
    "CAL_0001": "RETURN_REFUND", "CAL_0002": "GENERAL_COMPLAINT",
    "CAL_0003": "ORDER_STATUS", "CAL_0004": "GENERAL_COMPLAINT",
    "CAL_0005": "GENERAL_COMPLAINT", "CAL_0006": "DELIVERY_DAMAGE",
    "CAL_0007": "ORDER_STATUS", "CAL_0008": "RETURN_REFUND",
    "CAL_0009": "GENERAL_COMPLAINT", "CAL_0010": "ORDER_STATUS",
    "CAL_0011": "ORDER_STATUS", "CAL_0012": "RETURN_REFUND",
    "CAL_0013": "GENERAL_COMPLAINT", "CAL_0014": "GENERAL_COMPLAINT",
    "CAL_0015": "GENERAL_COMPLAINT", "CAL_0016": "RETURN_REFUND",
    "CAL_0017": "RETURN_REFUND", "CAL_0018": "GENERAL_COMPLAINT",
    "CAL_0019": "ORDER_STATUS", "CAL_0020": "ACCOUNT_ACCESS",
    "CAL_0021": "ORDER_STATUS", "CAL_0022": "RETURN_REFUND",
    "CAL_0023": "GENERAL_COMPLAINT", "CAL_0024": "ORDER_STATUS",
    "CAL_0025": "RETURN_REFUND", "CAL_0026": "GENERAL_COMPLAINT",
    "CAL_0027": "GENERAL_COMPLAINT", "CAL_0028": "ORDER_STATUS",
    "CAL_0029": "RETURN_REFUND", "CAL_0030": "GENERAL_COMPLAINT",
    "CAL_0031": "ORDER_STATUS", "CAL_0032": "RETURN_REFUND",
    "CAL_0033": "PRIME_SUBSCRIPTION", "CAL_0034": "GENERAL_COMPLAINT",
    "CAL_0035": "GENERAL_COMPLAINT", "CAL_0036": "GENERAL_COMPLAINT",
    "CAL_0037": "GENERAL_COMPLAINT", "CAL_0038": "GENERAL_COMPLAINT",
    "CAL_0039": "GENERAL_COMPLAINT", "CAL_0040": "GENERAL_COMPLAINT",
    "CAL_0041": "GENERAL_COMPLAINT", "CAL_0042": "ORDER_STATUS",
    "CAL_0043": "GENERAL_COMPLAINT", "CAL_0044": "ORDER_STATUS",
    "CAL_0045": "ORDER_STATUS", "CAL_0046": "DELIVERY_DAMAGE",
    "CAL_0047": "RETURN_REFUND", "CAL_0048": "ORDER_STATUS",
    "CAL_0049": "GENERAL_COMPLAINT", "CAL_0050": "RETURN_REFUND",
    "CAL_0051": "GENERAL_COMPLAINT", "CAL_0052": "GENERAL_COMPLAINT",
    "CAL_0053": "ORDER_STATUS", "CAL_0054": "GENERAL_COMPLAINT",
    "CAL_0055": "GENERAL_COMPLAINT", "CAL_0056": "GENERAL_COMPLAINT",
    "CAL_0057": "RETURN_REFUND", "CAL_0058": "DEVICE_TECH_SUPPORT",
    "CAL_0059": "ACCOUNT_ACCESS", "CAL_0060": "ACCOUNT_ACCESS",
    "CAL_0061": "DELIVERY_DAMAGE", "CAL_0062": "GENERAL_COMPLAINT",
    "CAL_0063": "ACCOUNT_ACCESS", "CAL_0064": "GENERAL_COMPLAINT",
    "CAL_0065": "GENERAL_COMPLAINT", "CAL_0066": "ORDER_STATUS",
    "CAL_0067": "GENERAL_COMPLAINT", "CAL_0068": "GENERAL_COMPLAINT",
    "CAL_0069": "GENERAL_COMPLAINT", "CAL_0070": "GENERAL_COMPLAINT",
    "CAL_0071": "ORDER_STATUS", "CAL_0072": "ORDER_STATUS",
    "CAL_0073": "ORDER_STATUS", "CAL_0074": "ORDER_STATUS",
    "CAL_0075": "GENERAL_COMPLAINT", "CAL_0076": "GENERAL_COMPLAINT",
    "CAL_0077": "GENERAL_COMPLAINT", "CAL_0078": "RETURN_REFUND",
    "CAL_0079": "ACCOUNT_ACCESS", "CAL_0080": "GENERAL_COMPLAINT",
    "CAL_0081": "GENERAL_COMPLAINT", "CAL_0082": "GENERAL_COMPLAINT",
    "CAL_0083": "GENERAL_COMPLAINT", "CAL_0084": "ACCOUNT_ACCESS",
    "CAL_0085": "ORDER_STATUS", "CAL_0086": "ORDER_STATUS",
    "CAL_0087": "ORDER_STATUS", "CAL_0088": "GENERAL_COMPLAINT",
    "CAL_0089": "RETURN_REFUND", "CAL_0090": "RETURN_REFUND",
    "CAL_0091": "ORDER_STATUS", "CAL_0092": "GENERAL_COMPLAINT",
    "CAL_0093": "GENERAL_COMPLAINT", "CAL_0094": "GENERAL_COMPLAINT",
    "CAL_0095": "GENERAL_COMPLAINT", "CAL_0096": "GENERAL_COMPLAINT",
    "CAL_0097": "PRIME_SUBSCRIPTION", "CAL_0098": "GENERAL_COMPLAINT",
    "CAL_0099": "DEVICE_TECH_SUPPORT", "CAL_0100": "GENERAL_COMPLAINT",
}

def main() -> None:
    df = pd.read_csv(ROOT / "golden_set" / "cal_100.csv")
    assert len(df) == 100 and set(df["message_id"]) == set(CAL_INTENT)
    df["true_intent"] = df["message_id"].map(CAL_INTENT)
    df.to_csv(ROOT / "golden_set" / "cal_100.csv", index=False)
    print("cal intent dist:", df["true_intent"].value_counts().to_dict())

if __name__ == "__main__":
    main()
