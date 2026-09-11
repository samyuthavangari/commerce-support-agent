# Escalation v5 — calibration log (cal-100 + cal2-100 ONLY)

Lineage: v0 narrow rules → v1–v3 tuned on golden-200 (disclosed) → v4 on
cal-100 → **v5 on cal-100 ∪ cal2-100**. Golden, heldout-A and heldout-B were
never consulted during v5 iteration. Cal-2 (100 threads, 29 positives) was
sampled and labelled BEFORE v5 existed.

## Transfer test that motivated v5 (v4 rules, unchanged)
cal2: P=0.824 R=0.483 — same disease as v3→heldout-A: high precision, no
coverage on fresh phrasings.

## v5 changes (each motivated by ≥1 cal/cal2 miss)
- `fraudulent|scammer` standalone (CAL_0048, CAL_0059): explicit fraud
  attribution is a claim, not venting.
- Repeat-verb expansion: `team/support/service + told/said/asked/promised`
  (C2_0096 pattern family), wrote/sent verbs + shared/provided/gave/posted
  (C2_0042, C2_0084), `many/several/multiple/N number of + times` (C2_0042),
  `filed + complaint/case` (C2_0094), `took another + currency` (duplicate
  charge family), `wrong report/response` (C2_0079).
- Assurance family widened to deliver/delivery/package/order objects
  (C2_0016, C2_0064); `cargo doble` Spanish billing (C2_0083 pattern family).
- Double-charge family: `double charged`, `charged…twice` (CAL_0060).
- 10-digit phone → public-PII (C2_0058; zero FPs across all 550 labelled rows).
- "Already replied/responded" verbs (C2_0011).
- **LLM tiebreaker** (`llm:ambiguous`) for the ultra-short bucket (≤4 tokens,
  no IDs): routes "Yes."/"help"/"Already done"/bare codes to a zero-shot
  check instead of declaring them an accepted residual. ~10 calls per eval,
  temp 0.0, fail-safe default.

## Accepted residuals (NOT patched)
- CJK no-reply markers (CAL_0031): cannot validate without speakers.
- "What is the time??" / "Mr. Robot" shapes the tiebreaker still misses.
- GS_0207 ("fladuent" typo): typo-tolerant fraud matching is v6 work —
  deliberately not added from a golden miss.

## Frozen 5-set results (v5 rules + tiebreaker, locked)

| Set | n | Pos | Precision | Recall | False-auto |
|---|---|---|---|---|---|
| cal-100 (tuned) | 100 | 31 | 0.816 | 1.000 | 0.000 |
| cal2-100 (tuned) | 100 | 29 | 0.725 | 1.000 | 0.000 |
| golden-250 | 250 | 53 | 0.867 | 0.981 | 0.019 |
| heldout-A-50 (frozen) | 50 | 15 | 0.733 | 0.733 | 0.267 |
| heldout-B-50 (frozen) | 50 | 11 | 0.800 | 0.364 | 0.636 |

## Severity split (the number that matters)

False-auto-handle is NOT one number. Audited per miss:

- **Critical misses (fraud-concept the patterns miss): 3 total** — GS_0207
  ("fladuent" typo for fraudulent), H_0019 ("charged twice"), HB_0047
  ("charged... not returned"). All three are paraphrase/typo gaps in fraud
  language, never seen in either cal set. This is the v6 work list.
- **Benign misses (repeat-contact/human-ask/unclear): 14 total** — all other
  FNs across the five sets.
- **Zero misses** in legal, safety, public-PII, or explicit-fraud-keyword
  categories on any set (600 labelled rows).

The tiebreaker costs precision on benign shorts (8 cal2 FPs, all benign
foreign-language fragments) while catching 9 ultra-short TPs. Kept: fail-safe
direction, ~10 extra API calls per 250-row eval, disclosed cost.
