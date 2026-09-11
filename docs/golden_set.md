# Golden set — sampling, labelling, quality

Files: `golden_set/golden_250.csv` (eval, now **250 rows** — filename kept for
pipeline compat), `golden_set/cal_100.csv` (tuning ONLY),
`golden_set/heldout_50.csv` + `heldoutB_50.csv` (frozen validation).

## Sampling (golden_200)
Source: `data/processed/amazon_threads.jsonl` (10,000 threads). First customer
message per thread, keyword pre-filtered into 7 buckets, stratified sample
(pools: ORDER 2810, RETURN 695, ACCOUNT 496, PRIME 1141, DEVICE 1524,
DAMAGE 108, COMPLAINT 299). Hard cases (sarcasm, ultra-short, multi-intent)
kept where they fell; 3 exact-duplicate pairs retained deliberately with
agreeing labels (GS_0006/0184, GS_0074/0129, GS_0109/0197 — pinned by test).

## Labelling
1. **First pass (LLM)**: Gemini flash assigns `llm_intent` + `escalate_yn`.
   Stored, never overwritten — `llm_intent` vs `true_intent` gives κ=0.939 (LLM-vs-human-reviewer agreement, single reviewer, no adjudication pass).
2. **Second pass (human, all 200 rows read)**: 10 intent + 45 escalation
   corrections recorded as code in `apply_review.py` (`human_verified=True`,
   `labelled_by=human_review_v1`). Escalation judged against the 7-rule
   definition frozen in that file BEFORE labelling (legal / fraud-security /
   safety / public-PII / repeat-contact-failure / explicit-human-ask /
   uninterpretable). Single reviewer, one session — no adjudication pass
   (documented limitation).
3. `escalate_reason` on human-True rows stamped with the firing rule
   (`stamp_reasons.py`); LLM reasons retained alongside, not erased.
4. **Rare-class top-up** (`merge_topup.py`, +50 rows sampled OUTSIDE the 10k
   corpus so no index rebuild was needed): DEVICE 8→22, PRIME 14→30.
   Same two-pass pipeline (LLM first-pass, then recorded human corrections in
   `merge_topup.py`). Golden total 250 (assignment cap), min class 22.

## Held-out 50
Drawn from 9,803 threads excluding all golden IDs, oversampled 30 hinted + 20
plain for positives (15/50 escalate). Labelled with the same frozen definition;
rules locked before labelling. Purpose: generalization check, never tuning.

## Contract-schema mapping (§8 suggested fields)
`id`→message_id · `customer_message`→text · `context`→thread_id (first message
only; multi-turn context out of scope) · `intent`→true_intent ·
`expected_action`→escalate_yn · `gold_reply_guidance`→(reference_reply in
results, not golden — many valid replies exist) · `labeler`→labelled_by ·
`notes`→escalate_reason.
