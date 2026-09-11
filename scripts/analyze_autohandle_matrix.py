"""
scripts/analyze_autohandle_matrix.py
───────────────────────────────────
Constructs the comprehensive multi-dimensional confusion/error table
for auto-handled cases across the golden test set:
  Confidence × Retrieval Top-1 Sim × Grounding × Actual Human Label × Escalation Decision
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

df_replies = pd.read_csv(ROOT / "results" / "agent_replies.csv")
df_gold = pd.read_csv(ROOT / "golden_set" / "golden_250.csv")
df_judge = pd.read_csv(ROOT / "results" / "judge_scores.csv")

# Merge complete evaluation records
m = pd.merge(df_replies, df_gold[['message_id', 'escalate_yn', 'true_intent', 'escalate_reason']], on='message_id')
m = pd.merge(m, df_judge[['message_id', 'groundedness', 'safety', 'helpfulness', 'total']], on='message_id')

m['intent_correct'] = m['pred_intent'] == m['true_intent_x']
m['is_auto'] = m['escalation_decision'] == 'auto'
m['is_escalate'] = m['escalation_decision'] == 'escalate'

# 1. Overall Pipeline Outcome Matrix
# True Auto: Gold Escalate=False AND System=Auto AND Intent=Correct
# Risky Auto (Intent Error): Gold Escalate=False AND System=Auto AND Intent=Wrong
# False Auto (Safety Leak): Gold Escalate=True AND System=Auto (CRITICAL BUG)
# True Escalate: Gold Escalate=True AND System=Escalate
# False Escalate (Over-cautious): Gold Escalate=False AND System=Escalate

def categorize_outcome(row):
    if row['escalation_decision'] == 'auto':
        if row['escalate_yn']:
            return "FALSE_AUTO (Safety Leak)"
        elif not row['intent_correct']:
            return "MISCLASSIFIED_AUTO (Wrong Intent)"
        else:
            return "TRUE_AUTO (Safe & Correct)"
    else:
        if row['escalate_yn']:
            return "TRUE_ESCALATE (Correct Escalation)"
        else:
            return "FALSE_ESCALATE (Over-cautious)"

m['outcome_category'] = m.apply(categorize_outcome, axis=1)

print("=" * 70)
print("1. OVERALL PIPELINE OUTCOME BREAKDOWN (N=250)")
print("=" * 70)
outcome_counts = m['outcome_category'].value_counts()
for cat, count in outcome_counts.items():
    pct = count / len(m) * 100
    print(f"  {cat:<35} : {count:3d} ({pct:5.1f}%)")

print("\n" + "=" * 70)
print("2. CONFIDENCE CALIBRATION & ERROR PROFILE BY CONFIDENCE BIN")
print("=" * 70)
conf_bins = pd.cut(m['intent_confidence'], bins=[0.79, 0.89, 0.94, 0.99, 1.01], labels=['0.80-0.85', '0.90', '0.95', '1.00'])
m['conf_bin'] = conf_bins

conf_table = m.groupby('conf_bin', observed=False).agg(
    Total_N=('message_id', 'count'),
    True_Intent_Acc=('intent_correct', 'mean'),
    Auto_N=('is_auto', 'sum'),
    Auto_Intent_Acc=('is_auto', lambda s: m.loc[s[s].index, 'intent_correct'].mean()),
    Misclassified_Auto=('outcome_category', lambda s: (s == "MISCLASSIFIED_AUTO (Wrong Intent)").sum()),
    Safety_Leak_Auto=('outcome_category', lambda s: (s == "FALSE_AUTO (Safety Leak)").sum()),
    Escalate_N=('is_escalate', 'sum'),
)
print(conf_table.to_string())

print("\n" + "=" * 70)
print("3. AUTO-HANDLED SLICE TABLE (N=190 Auto-Handled Cases)")
print("Confidence × Intent Correctness × Safety × Judge Groundedness")
print("=" * 70)
auto_cases = m[m['is_auto']]
slice_table = auto_cases.groupby(['conf_bin', 'intent_correct'], observed=False).agg(
    Count=('message_id', 'count'),
    Mean_Top1_Sim=('retrieval_top1_sim', 'mean'),
    Mean_Groundedness=('groundedness', 'mean'),
    Mean_Safety_Score=('safety', 'mean'),
    Safety_Leaks=('escalate_yn', 'sum'),
)
print(slice_table.to_string())

print("\n" + "=" * 70)
print("4. DETAILED BREAKDOWN OF ALL 38 MISCLASSIFIED AUTO-HANDLED CASES")
print("=" * 70)
wrong_auto = auto_cases[~auto_cases['intent_correct']]
print(f"Total misclassified auto-handled cases: {len(wrong_auto)} / {len(auto_cases)} ({len(wrong_auto)/len(auto_cases):.1%})\n")
intent_pairs = pd.crosstab(wrong_auto['true_intent_x'], wrong_auto['pred_intent'], margins=True)
print(intent_pairs.to_string())

print("\n" + "=" * 70)
print("5. THE CRITICAL SAFETY LEAK (FALSE AUTO)")
print("=" * 70)
leak = m[m['outcome_category'] == "FALSE_AUTO (Safety Leak)"]
for _, r in leak.iterrows():
    print(f"Message ID   : {r['message_id']}")
    print(f"Customer Text: \"{r['text']}\"")
    print(f"True Intent  : {r['true_intent_x']} | True Escalate: {r['escalate_yn']}")
    print(f"Pred Intent  : {r['pred_intent']} (conf: {r['intent_confidence']:.2f})")
    print(f"Gold Reason  : {r['escalate_reason']}")
    print(f"Draft Reply  : \"{r['draft_reply']}\"")
    print(f"Judge Scores : Grounded={r['groundedness']}/5, Safety={r['safety']}/5, Helpfulness={r['helpfulness']}/5")
