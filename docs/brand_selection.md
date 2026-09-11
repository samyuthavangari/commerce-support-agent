# Brand selection — @AmazonHelp

## Candidates considered
The subsample pipeline (`src/data_prep.py`) filters `twcs.csv` (2,811,774 rows)
by `author_id`. AmazonHelp contributed **169,840 reply rows / ~155k parent
threads** — the largest single-brand footprint in the dump, reconstructed into
**154,985 threads** (audit-measured).

## Why AmazonHelp
1. **Volume**: largest thread count → 10k retrieval corpus + 200 golden + 50
   held-out with room to spare, all disjoint where it matters.
2. **Intent diversity**: order, return, account, Prime, device, damage and open
   venting all present at measurable rates (golden spans 8–48 per class).
3. **Reviewer legibility**: Amazon support behavior is familiar; failures are
   checkable without domain expertise.
4. **English-majority**: keeps labelling tractable (non-English rows retained
   where frequent, e.g. French/Spanish/Portuguese/Hindi/Japanese examples in golden).

## Honest limitations
- Selection was by **volume and familiarity, not by measured resolution
  density**. A stronger procedure (not done): score brands by
  `groundable_pairs = volume × (1 − deflection_rate)` and pick the max.
  Amazon's heavy "please DM us" style means the retrieval corpus teaches
  redirection more than resolution — the drafter's concreteness comes from
  the minority of specific replies.
- One brand, one 2017 time window: no cross-brand or temporal generalization
  is claimed or measured.
