# Escalation v4 — calibration log (cal-100 ONLY)

Policy lineage: v0 narrow rules (legal/PII/safety/thread/lowconf) → v1–v3
iterated on golden-200 (disclosed tuned-on-test) → **v4 iterated ONLY on
cal-100 misses**. Golden, heldout-A and heldout-B were never consulted during
v4 iteration (their misses below were seen once in a transfer table, then
ignored until freeze). Heldout-B was labelled before v4 existed.

## Transfer test that motivated v4 (v3 rules, unchanged)
cal-100: P=1.000 R=0.226 · heldout-B: P=1.000 R=0.364. High precision, no
coverage: v3 memorized golden phrasings.

## v4 changes (each motivated by ≥1 cal-100 miss)
- `fraudulent|scammer` standalone (CAL_0048, CAL_0059): explicit fraud
  attribution is a claim, not venting — no money-context needed.
- Repeat-verb expansion: furnished/provided-details (CAL_0014, CAL_0049),
  looking-into-matter (CAL_0010), closed-complaint (CAL_0012), false-info
  (CAL_0024), tried+gerund (CAL_0028), executive/agent-told (CAL_0029),
  spoke-to-rep (CAL_0039), spent-hr-on-phone (CAL_0043), web-chat object
  (CAL_0079), can't-reach (CAL_0093), refusal-to-talk (CAL_0065),
  cursed/swore staff misconduct (CAL_0070), readymade/canned answers
  (CAL_0076), ordinal-effort "15th time" (CAL_0064), CC-rude kept.
- Spanish phone-failure (CAL_0038): `comunica…teléfono`, `no solucionaban`.
- `_UNCLEAR` rule for link/mention-only messages (CAL_0037).
- Dead LLM branch (`conf<0.72`) REMOVED: classifier never scores below 0.80
  over 400+ rated messages; untestable dead code is worse than an explicit
  deterministic policy. Low confidence still escalates via rule (<0.45).

## Accepted residuals (NOT patched)
- CJK no-reply markers (CAL_0031): cannot validate without speakers.
- Ultra-short ambiguous texts (CAL_0052 "Yes.", CAL_0056 "i did",
  CAL_0075, CAL_0077): no safe pattern separates contentless-escalate from
  contentless-benign ("Gracias", "WINNERS?" are auto by the same shape).

## Frozen 4-set results (rules locked; sets measured once)
| Set | n | Pos | Precision | Recall | False-auto |
|---|---|---|---|---|---|
| cal-100 (tuned here) | 100 | 31 | 0.963 | 0.839 | 0.161 |
| golden-250 | 250 | 58 | 0.944 | 0.879 | 0.121 |
| heldout-A-50 | 50 | 15 | 0.818 | 0.600 | 0.400 |
| heldout-B-50 | 50 | 11 | 1.000 | 0.364 | 0.636 |

v4 cost 2 new golden FPs (GS_0134 "2nd time" ordinal clause, GS_0200 "wont
help" refusal clause) — accepted without tightening, because tightening
against golden misses would repeat the v1–v3 mistake. Next round (v5) may use
cal-100 + a fresh cal-2 set only.
