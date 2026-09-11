# DECISIONS.md — 15 non-obvious decisions

Format: WHAT / WHY / ALTERNATIVES / TRADEOFF. Trivial choices omitted.

## 1. Brand: @AmazonHelp, chosen for volume and familiarity
- **What**: single-brand scope on AmazonHelp (~170k replies, ~155k threads).
- **Why**: largest intent diversity in the subsample; instantly legible to reviewers.
- **Alternatives**: measurement-driven pick by resolution density (deflection-rate analysis).
- **Tradeoff**: faster start, thinner justification — documented as a known weakness; a brand-selection study is the first item that would strengthen this.

## 2. Seven intents, keyword-seeded then human-bounded
- **What**: ORDER_STATUS / RETURN_REFUND / ACCOUNT_ACCESS / PRIME_SUBSCRIPTION /
  DEVICE_TECH_SUPPORT / DELIVERY_DAMAGE / GENERAL_COMPLAINT (catch-all).
- **Why**: label quality beats granularity at n=200; boundaries set by reading rows.
- **Alternatives**: 10+ fine-grained intents; Banking77 transfer labels.
- **Tradeoff**: coarse classes hide substructure (e.g. late-vs-lost inside ORDER_STATUS).

## 3. One model (flash-lite) for classify, draft and judge
- **What**: same model everywhere instead of Flash/Pro split.
- **Why**: removes cross-model confounds; cheapest; verified sufficient.
- **Alternatives**: Pro drafter + Flash judge; cross-family judge (GPT).
- **Tradeoff**: shared priors inflate classifier↔labeller agreement and cap judge independence.

## 4. Embed (customer + SEP + reply) pairs
- **What**: retrieval vectors encode the resolution, not just the complaint.
- **Why**: "where is my package" + "please DM us" retrieves more usefully than either alone.
- **Alternatives**: customer-text-only embeddings; fine-tuned contrastive embeddings.
- **Tradeoff**: reply-side noise (deflections) leaks into similarity; no fine-tuning budget.

## 5. Local-first Qdrant with pure-Python in-memory fallback
- **What**: `QDRANT_URL` defaults to localhost Docker; cloud via optional overrides; zero-Docker fallback via `--no-docker` with pre-computed numpy vectors (`data/memory_vectors.npz`).
- **Why**: reviewers reproduce in pure Python without Docker installed; full index still available if Qdrant is launched.
- **Alternatives**: cloud-only; Docker-only (cold start was ~9m with Docker).
- **Tradeoff**: in-memory store loads static normalized vector slice; lightweight for benchmark evaluation.

## 6. Top-5 retrieval with intent pre-filter
- **What**: filter payloads by predicted intent, then top-5 cosine.
- **Why**: unfiltered top-1 intent-consistency measured 0.49 (DELIVERY_DAMAGE: 0.00) —
  the filter is load-bearing, not cosmetic.
- **Alternatives**: unfiltered top-10; cross-encoder rerank.
- **Tradeoff**: filter inherits classifier errors (wrong intent → wrong neighborhood).

## 7. Hard 280-char + URL guardrail in code, not just prompt
- **What**: prompt rules backed by `sanitize_reply()` (regex strip) and hard trim.
- **Why**: the model invented `t.co` links twice; prompts alone don't constrain output.
- **Alternatives**: prompt-only instruction; allowlist of canonical links.
- **Tradeoff**: regex can mangle edge text; legitimate links impossible by construction.

## 8. Two-pass labelling with corrections as code
- **What**: LLM first-pass → row-by-row human review recorded in `scripts/labeling/apply_review.py`.
- **Why**: cuts labelling time; every change is an auditable diff (55 corrections).
- **Alternatives**: pure hand-labelling (slower); accepting LLM labels (dishonest to call golden).
- **Tradeoff**: single reviewer, one session (κ=0.939 LLM-vs-human agreement) — bias documented, no adjudication pass.

## 9. Escalation definition frozen BEFORE labelling
- **What**: 7-rule human definition fixed in `apply_review.py`, then rows judged.
- **Why**: prevents moving goalposts to match agent behavior or labeller outputs.
- **Alternatives**: adopting the LLM labeller's loose criteria; tuning thresholds first.
- **Tradeoff**: definition is one author's judgment; 44 rows flipped vs first pass.

## 10. Deterministic escalation tiers over LLM judgment
- **What**: fraud / public-PII / repeat-contact / human-ask regexes before any LLM call.
- **Why**: auditable, testable, zero-cost; calibrated offline to human labels.
- **Alternatives**: pure LLM triage; confidence-threshold gating.
- **Tradeoff**: paraphrase brittleness (held-outs 0.60/0.36); the old
  `conf<0.72` LLM fallback was removed outright rather than kept as dead code.

## 11. Same-family judge, kept after it failed
- **What**: Gemini judges Gemini; agreement measured (ρ=-0.062, n=70) and reported.
- **Why**: consistency first; swapping judges until one passes would be metric shopping.
- **Alternatives**: cross-family judge; human-only scoring (too slow at n=200).
- **Tradeoff**: ceiling effects + safety inversions; reply headline unproven.

## 12. Spearman (not Pearson) for agreement + exact/±1/weighted-κ alongside
- **What**: rank correlation primary; exact/±1 agreement and quadratic-weighted κ reported.
- **Why**: ordinal 1–5 scores with ceiling pile-up break Pearson assumptions.
- **Alternatives**: Pearson only; κ only.
- **Tradeoff**: three numbers invite cherry-picking — all are reported, best and worst.

## 13. Calibrate on golden, validate frozen on held-out
- **What**: rules iterated on cal-100 only, then locked and measured on golden
  plus two frozen held-outs (4-set table) without touching them.
- **Why**: separates "rules can express the definition" from "rules generalize".
- **Alternatives**: second tuning round on held-out (destroys its purpose); no held-out at all.
- **Tradeoff**: the honest numbers (held-outs 0.60/0.36) look worse than any tuned one.

## 14. Fair B1 retrained on weak labels, old number withdrawn
- **What**: TF-IDF+LogReg trained on 8k heuristic-labelled non-golden threads;
  the 0.92 train-on-test figure withdrawn in writing.
- **Why**: a contaminated baseline is worse than none — it punishes the honest system.
- **Alternatives**: dropping B1; 100/100 golden split (too small to train on).
- **Tradeoff**: weak labels cap B1 (0.496) below its "true" supervised potential.

## 15. Blind A/B (agent vs template) for both judge and human
- **What**: 70 pairs, source hidden from both scorers; direction compared, not just scores.
- **Why**: absolute 1–5 scores proved uncalibrated (+4 judge bias); ranking survives.
- **Alternatives**: larger human panel (no budget); pairwise Bradley-Terry (overkill at n=70).
- **Tradeoff**: template is a weak opponent — beating it proves little beyond "non-trivial".
