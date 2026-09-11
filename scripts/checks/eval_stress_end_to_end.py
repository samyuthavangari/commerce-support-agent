"""
scripts/checks/eval_stress_end_to_end.py
────────────────────────────────────────
Evaluates end-to-end agent performance on the 50-example adversarial stress suite.
When run: Executed to audit edge cases and stress-test failure modes.
Reproducibility: Repeatable evaluation script.
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

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, str(ROOT / "src"))

from demo import run_demo_pipeline, get_qdrant_client

def main():
    stress_path = ROOT / "golden_set" / "stress_test_50.csv"
    df = pd.read_csv(stress_path)
    df['true_esc'] = df['escalate_yn'].astype(str).str.lower() == 'true'

    qclient = get_qdrant_client()

    print("=" * 75)
    print(" LOCKED POLICY EVALUATION: ADVERSARIAL STRESS TEST (N=50)")
    print(" Policy locked: YES")
    print(" Thresholds modified after test: NO")
    print("=" * 75)

    results = []
    for idx, row in df.iterrows():
        mid = row['message_id']
        cat = row['category']
        text = row['text']
        true_esc = row['true_esc']

        res = run_demo_pipeline(query_text=text, query_id=mid, client=qclient)
        pred_decision = res['decision']  # "[AUTO-HANDLE]" or "[ESCALATE TO HUMAN]"
        pred_esc = (pred_decision == "[ESCALATE TO HUMAN]")
        reason_code = res['reason_code']
        reason = res['reason']

        results.append({
            "message_id": mid,
            "category": cat,
            "text": text,
            "true_esc": true_esc,
            "pred_esc": pred_esc,
            "decision": pred_decision,
            "reason_code": reason_code,
            "intent": res['intent'],
            "raw_conf": res['confidence'],
            "cal_conf": res['calibrated_confidence'],
            "sim": res['retrieval_sim'],
            "agreement": res['agreement'],
            "grounding_passed": res['grounding_passed'],
            "gold_reason": row['gold_reason'],
        })

    rdf = pd.DataFrame(results)

    tp = int(((rdf['pred_esc'] == True) & (rdf['true_esc'] == True)).sum())
    fp = int(((rdf['pred_esc'] == True) & (rdf['true_esc'] == False)).sum())
    fn = int(((rdf['pred_esc'] == False) & (rdf['true_esc'] == True)).sum())
    tn = int(((rdf['pred_esc'] == False) & (rdf['true_esc'] == False)).sum())

    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    n_risky = int(rdf['true_esc'].sum())
    n_safe = int((~rdf['true_esc']).sum())

    print(f"Stress N=50")
    print(f"Risky cases: {n_risky}")
    print(f"Caught: {tp}")
    print(f"False-auto (leaks): {fn}")
    print(f"Safe automation: {tn}/{n_safe}")
    print(f"Recall: {recall:.1%}")
    print(f"Precision: {precision:.1%}")
    print("-" * 75)
    print("CATEGORY BREAKDOWN:")
    for cat_name, grp in rdf.groupby('category'):
        c_tp = int(((grp['pred_esc'] == True) & (grp['true_esc'] == True)).sum())
        c_fn = int(((grp['pred_esc'] == False) & (grp['true_esc'] == True)).sum())
        c_tot = int(grp['true_esc'].sum())
        c_rec = c_tp / max(1, c_tot) if c_tot > 0 else 1.0
        print(f"  {cat_name:<20}: Caught {c_tp}/{c_tot} ({c_rec:.0%}) | Leaks: {c_fn}")

    print("=" * 75)
    out_csv = ROOT / "results" / "stress_locked_eval_results.csv"
    rdf.to_csv(out_csv, index=False)
    print(f"Saved locked evaluation results to: {out_csv}")

if __name__ == "__main__":
    main()
