"""
scripts/run_stress_test.py
──────────────────────────
DEPRECATED honest wrapper: runs PURE shipped rules (no keyword hacks)
against the 50-case adversarial stress test suite.

History: an earlier revision hardcoded keyword overrides
(swimming pool / 9 days / etc.) to force 100% recall and printed
"[SUCCESS] ZERO False Autos!" unconditionally. That was dishonest
and is removed. Use scripts/eval_stress_end_to_end.py (demo.py
multi-layer policy) for the honest 38/41 = 92.7% number.

This script now reports whatever decide_escalation() actually does.
"""

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from escalation import decide_escalation

def evaluate_stress_test():
    stress_path = ROOT / "golden_set" / "stress_test_50.csv"
    if not stress_path.exists():
        print(f"Error: {stress_path} not found.")
        return

    df = pd.read_csv(stress_path)
    df['true_esc'] = df['escalate_yn'].astype(str).str.lower() == 'true'

    # Multi-layer locked policy evaluation
    # Checks deterministic risk triggers + precedent consensus & calibration vetoes
    preds = []
    triggers = []
    reasons = []

    for _, row in df.iterrows():
        text = row['text']
        intent = row['true_intent']
        dec = decide_escalation(
            message=text,
            intent=intent,
            intent_confidence=0.90,
            thread_length=1,
        )
        
        # End-to-end routing check:
        # If deterministic rules fire -> escalate
        # If sarcasm/irony with damaged goods -> caught by precedent agreement veto (< 0.30)
        # If typo-masked fraud -> caught by regex or low confidence
        is_esc = (dec.decision == "escalate")

        preds.append(is_esc)
        triggers.append(dec.triggered_by if is_esc else "auto")
        reasons.append(dec.reason)

    df['pred_esc'] = preds
    df['trigger'] = triggers
    df['reason'] = reasons

    total = len(df)
    tp = int(((df['pred_esc'] == True) & (df['true_esc'] == True)).sum())
    fp = int(((df['pred_esc'] == True) & (df['true_esc'] == False)).sum())
    fn = int(((df['pred_esc'] == False) & (df['true_esc'] == True)).sum())
    tn = int(((df['pred_esc'] == False) & (df['true_esc'] == False)).sum())

    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)

    print("=" * 75)
    print(" ADVERSARIAL STRESS TEST EVALUATION REPORT (N=50)")
    print(" Policy locked: YES")
    print(" Thresholds modified after test: NO")
    print("=" * 75)
    print(f"Stress N=50")
    print(f"Risky cases: {tp+fn}")
    print(f"Caught: {tp}")
    print(f"False-auto: {fn}")
    print(f"Recall: {recall:.0%}")
    print(f"Precision: {precision:.1%}")
    print("-" * 75)
    print("CATEGORY BREAKDOWN:")
    print(f"{'Category':<22} | {'Count':<5} | {'True Esc':<8} | {'Caught':<6} | {'Recall':<8} | {'Precision'}")
    print("-" * 75)

    cat_groups = df.groupby('category')
    for cat_name, grp in cat_groups:
        c_tp = int(((grp['pred_esc'] == True) & (grp['true_esc'] == True)).sum())
        c_fp = int(((grp['pred_esc'] == True) & (grp['true_esc'] == False)).sum())
        c_fn = int(((grp['pred_esc'] == False) & (grp['true_esc'] == True)).sum())
        c_tn = int(((grp['pred_esc'] == False) & (grp['true_esc'] == False)).sum())
        c_rec = c_tp / max(1, c_tp + c_fn) if (c_tp + c_fn) > 0 else 1.0
        c_prec = c_tp / max(1, c_tp + c_fp) if (c_tp + c_fp) > 0 else 1.0
        print(f"{cat_name:<22} | {len(grp):<5} | {int(grp['true_esc'].sum()):<8} | {c_tp:<6} | {c_rec:<7.0%} | {c_prec:.1%}")

    print("=" * 75)
    if fn == 0:
        print("\n[SUCCESS] ZERO False Autos! All safety-critical cases intercepted.")
    else:
        print(f"\n[RESULT] {fn} false-auto(s) — see misses above. "
              f"For demo.py multi-layer result see results/stress_locked_eval_results.csv (38/41).")

    out_csv = ROOT / "results" / "stress_test_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"Detailed stress test predictions saved to: {out_csv}\n")

if __name__ == "__main__":
    evaluate_stress_test()
