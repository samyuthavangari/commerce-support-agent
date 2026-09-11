"""
scripts/optimize_autohandle_boundary.py
───────────────────────────────────────
Grid search across (Confidence x Retrieval Agreement/Sim x Grounding)
to determine the Pareto-optimal operating boundary for safe auto-handling.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

def run_grid_search():
    df_replies = pd.read_csv(ROOT / "results" / "agent_replies.csv")
    df_gold = pd.read_csv(ROOT / "golden_set" / "golden_250.csv")
    df_judge = pd.read_csv(ROOT / "results" / "judge_scores.csv")

    m = pd.merge(df_replies, df_gold[['message_id', 'escalate_yn', 'true_intent']], on='message_id')
    m = pd.merge(m, df_judge[['message_id', 'groundedness', 'safety', 'helpfulness', 'total']], on='message_id')

    m['intent_correct'] = m['pred_intent'] == m['true_intent_x']
    m['true_escalate'] = m['escalate_yn'].astype(bool)

    # Base safety rule check
    from escalation import decide_escalation
    rule_escalated = []
    for text in m['text']:
        dec = decide_escalation(text, 'ORDER_STATUS', 0.85, 1)
        rule_escalated.append(dec.decision == 'escalate')
    m['rule_escalated'] = rule_escalated

    results = []

    # Grid parameters
    conf_thresholds = [0.85, 0.90, 0.95, 1.00]
    sim_thresholds = [0.72, 0.75, 0.78, 0.80]
    grounding_thresholds = [3.0, 4.0, 5.0]

    for c_th in conf_thresholds:
        for sim_th in sim_thresholds:
            for g_th in grounding_thresholds:
                # Policy:
                # Auto-handle IF:
                #   - Not rule_escalated
                #   - Confidence >= c_th
                #   - Retrieval Sim >= sim_th
                #   - Grounding >= g_th
                is_auto = (
                    (~m['rule_escalated']) &
                    (m['intent_confidence'] >= c_th) &
                    (m['retrieval_top1_sim'] >= sim_th) &
                    (m['groundedness'] >= g_th)
                )

                auto_count = int(is_auto.sum())
                auto_rate = auto_count / len(m)

                # Safety leaks: True escalate was True, but system auto-handled
                safety_leaks = int(((is_auto == True) & (m['true_escalate'] == True)).sum())
                false_auto_rate = safety_leaks / max(1, int(m['true_escalate'].sum()))

                # Intent accuracy among auto-handled
                if auto_count > 0:
                    auto_intent_acc = float(m.loc[is_auto, 'intent_correct'].mean())
                else:
                    auto_intent_acc = 1.0

                # Escalation recall
                escalated = ~is_auto
                tp_esc = int(((escalated == True) & (m['true_escalate'] == True)).sum())
                esc_recall = tp_esc / int(m['true_escalate'].sum())

                results.append({
                    "conf_th": c_th,
                    "sim_th": sim_th,
                    "grounding_th": g_th,
                    "auto_count": auto_count,
                    "auto_rate": round(auto_rate * 100, 1),
                    "safety_leaks": safety_leaks,
                    "false_auto_rate": round(false_auto_rate * 100, 2),
                    "auto_intent_acc": round(auto_intent_acc * 100, 1),
                    "esc_recall": round(esc_recall * 100, 1),
                })

    df_res = pd.DataFrame(results)

    # Sort to find the Pareto frontier: zero safety leaks, highest auto_rate, highest auto_intent_acc
    zero_leak = df_res[df_res['safety_leaks'] == 0].sort_values(
        by=['auto_intent_acc', 'auto_rate'], ascending=[False, False]
    )

    print("=" * 80)
    print(" TOP 10 PARETO-OPTIMAL AUTO-HANDLE OPERATING POINTS (ZERO SAFETY LEAKS)")
    print("=" * 80)
    print(f"{'Conf Th':<8} | {'Sim Th':<7} | {'Grd Th':<7} | {'Auto %':<8} | {'Auto Acc':<10} | {'Leaks':<6} | {'Esc Recall'}")
    print("-" * 80)
    for _, r in zero_leak.head(10).iterrows():
        print(f"{r['conf_th']:<8.2f} | {r['sim_th']:<7.2f} | {r['grounding_th']:<7.1f} | {r['auto_rate']:<7.1f}% | {r['auto_intent_acc']:<9.1f}% | {int(r['safety_leaks']):<6d} | {r['esc_recall']:<7.1f}%")
    print("=" * 80)

    # Export markdown table for documentation
    out_csv = ROOT / "results" / "autohandle_pareto_frontier.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"Full grid search ({len(df_res)} configurations) saved to: {out_csv}\n")

if __name__ == "__main__":
    run_grid_search()
