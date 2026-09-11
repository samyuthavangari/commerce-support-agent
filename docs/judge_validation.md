# Judge validation — how (little) the LLM judge agrees with a human

## Setup
- Rubric: groundedness / helpfulness / empathy / safety / brevity, 1–5 each
  (brevity scale fixed: 5 ≤240 chars, 3 = 241–280, 1 = over limit).
- Judge (`src/eval/llm_judge.py`): receives customer text, intent, agent reply,
  reference reply. Strict-JSON, retried, recomputed totals. **Never receives
  system identity** — agent vs template scored blind by construction.
- Safety hybrid: deterministic caps overrule the LLM (URL → ≤2, PII-ask → 1),
  added after measuring safety ρ<0.
- Human round 1 (blind): author, same rubric, 70 pairs (agent vs canned
  template, order shuffled, keys in `results/blind_key_70.json`), established
  the A/B direction result — then frozen.
- Human round 2 (direct): same rubric scored against the CURRENT 70 replies
  (`results/human_scores.csv`), because regenerated drafts invalidated round-1
  grades. Template familiarity means round 2 is not blind — disclosed; the
  blind direction claim rests on round 1 only.

## Results (round 2, n=70, current replies)
- Total: Pearson -0.136, Spearman **-0.062**, exact 2/70, within-±1 5/70,
  quadratic-weighted κ -0.013. Target was ≥0.75: **missed, reported as-is**.
- Per-dimension ρ: empathy 0.440, helpfulness 0.301, brevity -0.135,
  groundedness/safety NaN (judge near-constant — ceiling effect breaks the math).
- Bias: judge mean 24.6 vs human 20.7 (**+4 systematic over-score**).
- Discrimination check (round 1, frozen): blind human ranks agent 20.75 >
  template 19.00 (+1.75); blind judge ranks agent 24.62 > template 22.12 (+2.50).
  **Same direction.** The judge detects *which system is better* while failing
  to score *how good a reply is* in absolute terms.

## Reading guidance
Trust the judge for A/B direction (empathy/helpfulness deltas). Do not trust
24.44/25 as an absolute quality grade. Agreement fell across rounds (total ρ
0.287 → 0.242 → -0.062 as replies regenerated and samples grew): grade
agreement is fragile, direction agreement held twice. That asymmetry is the
finding this file exists to record.
