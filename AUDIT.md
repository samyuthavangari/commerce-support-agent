# AUDIT.md — Repository audit vs the submission contract (2026-09-10)

> **Resolution status (end of 10/10 pass):** P0 items 1–3 closed (leakage gated
> + manifest, B1 retrained fair at 0.496/0.453, escalation re-calibrated on
> cal-100 with 4-set reporting + dead branch removed). P1 largely closed:
> retrieval eval + ablation shipped, agreement at blind n=70, 26 tests green,
> DECISIONS/docs/REPORT/REVIEW written, `--fast` measured 2:59, UI boot +
> interaction tested. Remaining: clean-room clone run, second reviewer,
> cross-family judge (all owner/future work). See REVIEW.md for the final
> PASS/WARN/FAIL ledger.

Scope: `D:\NExp2-13` (AmazonHelp support agent). Method: full tree read, all of
`src/` + root scripts + README + `golden_set/` + `results/` inspected; two
measurement probes executed (thread-overlap probe, artifact inventory).
No code was modified for this audit.

Legend: P0 = submission blocker · P1 = important · P2 = polish.
"Contract §N" refers to the lead-engineer contract sections.

---

## 1. What already works (keep; do not rewrite)

- **End-to-end agent** (`src/agent.py::run_agent`): classify → intent-filtered
  Qdrant retrieval → draft (280-char + URL guardrail) → escalation with
  reason/trigger. Live-probed on unseen messages; 200/200 golden rows, 0 errors.
- **Golden set**: 200 rows, all `human_verified=True`, corrections recorded as
  code (`scripts/labeling/apply_review.py`), intent κ=0.939 (LLM-vs-human-reviewer agreement, single reviewer, no adjudication pass), frozen 7-rule escalation definition.
  Bonus: 50-row frozen held-out set (`golden_set/heldout_50.csv`).
- **Harness**: `run_eval.py` one-command (agent + B0/B1 + judge + agreement →
  `eval_report.json`); per-class intent metrics + confusion matrix; baselines
  compared on the same set; judge with safety overrides; 15-row human agreement.
- **Honesty artifacts**: measured (not claimed) numbers everywhere; misleading
  section, failure analysis with real IDs, tuned-on-test disclosures.
- **Secrets hygiene**: `.env` gitignored, `.env.example` documents local-first
  Qdrant; no secrets in code.

## 2. What partially works

| Area | State | Gap |
|---|---|---|
| Escalation | P/R 1.00 golden; 0.80/0.53 frozen held-out | Rules tuned on test; `conf<0.72` LLM branch is dead code (confidence never <0.80); thresholds unjustified empirically |
| Judge | Runs, safety-capped, 0 URLs in outputs | Agreement ρ=0.287 (n=15); groundedness/brevity unmeasurable (judge constant); baselines never judged; judge not blinded |
| B1 baseline | Runs | **Trained on the test set** (contamination) — unfair per §10/§35 |
| Reproducibility | One command, local-Docker path real | **~16 min > 15**; no fast/offline/mock path; reviewer needs paid Gemini key to see anything |
| Decision log | 15 rows, good content | Lives in README, not `DECISIONS.md`; rows lack ALTERNATIVES/TRADEOFF structure |
| Report content | All 6 assignment sections in README | No `REPORT.md` ≤6pp; no architecture diagram; no spec-format result table |
| Failure analysis | Top-5 + real examples + fixes | Missing frequency counts and EXPECTED-vs-ACTUAL structure per failure |

## 3. What is missing (no evidence exists)

- `scripts/check_leakage.py` and any leakage analysis (§6).
- Retrieval evaluation: no Recall@K / intent-consistency at K=1,3,5 (§13).
- Ablations: no-retrieval / no-intent / no-escalation comparisons (§21).
- Tests: zero automated tests (§28).
- Offline/mock mode (§25); cost/latency accounting (§27); experiment tracking (§26).
- `docs/brand_selection.md`, `docs/intent_taxonomy.md`, `docs/golden_set.md`,
  `data/intent_schema.json`, `DECISIONS.md`, `REPORT.md`, `REVIEW.md` (§4,§7,§8,§30,§31,§38).
- Escalation confusion matrix + explicitly named **false-auto-handle rate** (§18).
- Confidence intervals on headline metrics (§11).
- Human agreement at n=40–60 with exact/±1 agreement and (weighted) κ (§16: have n=15, Pearson+Spearman only).
- PII-handling strategy note (order IDs persist in golden CSV + Qdrant payloads) (§29).

## 4. What is scientifically weak

1. **Same-thread leakage (measured, §6 violation).** Probe result: **197/197
   golden threads and 50/50 held-out threads are inside the 10k retrieval
   corpus.** Eval queries retrieve from an index containing their own thread
   (and its reference reply). Intent-filtered top-5 over 10k dilutes but does
   not remove this. Golden∩heldout = 0 (good). Known exact-duplicate pair
   inside golden itself: GS_0006/GS_0184 (identical earphones text, both kept).
2. **B1 train-on-test** inflates the "simple baseline" to 0.92 accuracy; the
   agent looks worse than it is on accuracy (macro-F1 comparison is fair).
3. **Judge validation underpowered and failing**: n=15, total ρ=0.287, safety
   ρ<0. Reply-quality headline (24.5/25) is therefore unproven.
4. **No retrieval grounding proof**: nothing shows retrieved cases are relevant
   (no intent-consistency@K, no similarity scores stored — `retrieve()` drops
   Qdrant scores, violating §12/§20 traceability).
5. **Dead decision branch**: `intent_confidence<0.72` LLM-escalation path never
   fires (observed min 0.80); effective policy = regexes only, undocumented.
6. **Golden labels**: LLM-first-pass + single-reviewer second pass. Defensible
   as "reviewed", but §8 forbids calling auto-labelled sets hand-labelled —
   our wording must stay precisely on "human-reviewed", never "hand-labelled
   from scratch". No adjudication pass; `escalate_reason` on auto rows is
   stale LLM text.
7. **Brand justification is thin** (§4): AmazonHelp chosen for volume/familiarity,
   no resolution-density or ambiguity analysis, no `docs/brand_selection.md`.

## 5. What is not reproducible

- **~16 min > 15-min contract** (11:17 agent + 4:09 judge, n=200).
- No fast path (`--fast` on a committed fixture), no `--offline`/mock mode;
  without `GOOGLE_API_KEY` nothing runs — a reviewer cannot inspect behavior.
- Fresh-clone path repaired by inspection only (dep pin, `COLLECTION_NAME`,
  optional Qdrant creds) — never executed clean.
- Qdrant index lives on the author's private Cloud cluster; local rebuild
  (~5 min embed + quota) is required but untested end-to-end this session.
- `notebooks/` is empty; root helper scripts (`apply_review.py`,
  `calibrate_escalation.py`, …) are unlabeled method-vs-scratch.

## 6. What violates the assignment/contract (must fix or disclose)

- §6 leakage controls: absent (P0).
- §10 fair simple baseline: B1 contaminated (P0 — decontaminate or relabel).
- §16 agreement n=15 vs 40–60 minimum (P1).
- §23 exact title: ours reads "⚠️ What Is Misleading About My Headline Number?"
  (P2 — retitle exactly).
- §30: decisions in README table, not `DECISIONS.md` with WHAT/WHY/ALTERNATIVES/
  TRADEOFF (P1).
- §8 prompt/retrieval hygiene: eval threads indexed (P0, same fix as leakage).
- Assignment delivery itself: **no git repo → no submittable link** (P0, owner action).

## 7. What cannot currently be proven

- That retrieval helps (no ablation, no retrieval metrics).
- That the judge measures quality (agreement fails; no blind comparison).
- That escalation generalizes (held-out R=0.53 IS the honest number; 1.00 is not).
- That results reproduce fresh (no clean-room run).
- Production behavior: latency, cost/request, failure under API timeout,
  empty/absurd inputs (no tests, no edge-case policy).

## 8. Recommended changes (priority order)

**P0 — blockers**
1. Leakage: exclude golden+heldout thread_ids from the retrieval corpus
   (rebuild index minus ~250 threads), add `scripts/check_leakage.py`
   (exact-dup, near-dup, same-thread-in-topK) failing loudly; document in
   README ("how leakage was prevented").
2. Baselines: retrain B1 on a held-out split (or 3k weak-labelled sample à la
   standard practice) — never on the golden 200.
3. Repro: add `--fast` eval path (committed 30-row fixture + cached replies,
   no key) and `--offline` agent path (TF-IDF + canned retrieval); state
   measured full-run time honestly; clean-room verify once.
4. Owner: `git init`, push, submit link (not done by agent per instructions).

**P1 — important**
5. Retrieval eval: intent-consistency@1/3/5 + score storage in
   `agent_replies.csv`; one ablation (no-retrieval draft vs RAG) on judge means.
6. Agreement: grow human scores to n≥40 with exact/±1 + weighted-κ; blind the
   judge (score agent + B1-retrieval replies without source labels).
7. Tests: preprocessing, taxonomy schema, leakage, escalation policy
   (incl. the 50 golden + 15 held-out expectations), metric sanity, JSON
   validity of judge outputs.
8. Docs: `DECISIONS.md` (15, structured), `docs/{brand_selection,intent_taxonomy,
   golden_set,judge_validation}.md`, `data/intent_schema.json` (with
   include/exclude + confusing neighbors), `REPORT.md` (≤6pp) + architecture
   diagram + spec-format result table, `REVIEW.md` (PASS/WARN/FAIL).
9. Escalation reporting: confusion matrix, false-auto-handle rate by name,
   threshold justification (or remove dead branch), PII-handling note.
10. CIs (bootstrap, n=200) on accuracy/macro-F1/recall; experiment provenance
    block (model/prompt/seed/config/timestamp) in `eval_report.json`.

**P2 — polish**
11. Exact §23 title; failure-analysis frequency + EXPECTED/ACTUAL fields;
    `escalate_reason` refresh on auto rows; remove empty `notebooks/`;
    file the scratch scripts under `scripts/` with purpose headers;
    cost/latency table; prompt versioning.

## 9. Phase plan (contract §40 order, adapted — no blind rewrites)

1. Leakage (exclude + check script + re-verify) → 2. Baselines (decontaminate,
   re-run affected metrics only) → 3. Retrieval eval + score plumbing →
4. Judge blinding + agreement expansion → 5. Tests → 6. Fast/offline repro +
   clean-room verification → 7. Docs + report + decisions + review audit →
8. Final cleanup. Re-run full eval once after P0 items 1–2 (threshold/rules
   unchanged, so intent/judge numbers carry over; escalation re-measured on
   the decontaminated setup).
