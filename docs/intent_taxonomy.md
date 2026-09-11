# Intent taxonomy — process note

Machine-readable schema: `data/intent_schema.json` (v1-2026-09-10).
Code source of truth: `src/intent_taxonomy.py`.

## How the 7 intents were derived
1. Keyword-clustered EDA over reconstructed AmazonHelp threads surfaced
   recurring families (tracking/delivery, returns/money-back, login/account,
   Prime, devices, damage, open venting).
2. Families were merged/split by **routing consequence**: two complaints share
   an intent iff the same resolution path handles them. This split
   late-vs-lost only at the ORDER_STATUS/DELIVERY_DAMAGE boundary
   (marked-delivered-never-arrived → ORDER_STATUS; damage/missing-contents/
   wrong-address → DELIVERY_DAMAGE) and kept refund-timing under RETURN_REFUND.
3. `GENERAL_COMPLAINT` is the **explicit catch-all** (classifier rule: "if
   nothing specific fits"), not a dumping ground by accident.
4. Banking77 was deliberately NOT used: its 77 banking intents do not transfer
   to Twitter commerce support and would have laundered an alien taxonomy in.

## Known noisy boundaries (expect classifier confusion here)
- ORDER_STATUS ↔ DELIVERY_DAMAGE (late vs lost vs damaged)
- RETURN_REFUND ↔ DELIVERY_DAMAGE (damage once money is the ask)
- PRIME_SUBSCRIPTION ↔ ACCOUNT_ACCESS (billing confusion)
- PRIME_SUBSCRIPTION ↔ ORDER_STATUS (late Prime delivery)

Measured class distribution (golden-250, human-reviewed): ORDER_STATUS 58,
RETURN_REFUND 45, GENERAL_COMPLAINT 43, PRIME_SUBSCRIPTION 30,
DELIVERY_DAMAGE 27, ACCOUNT_ACCESS 25, DEVICE_TECH_SUPPORT 22. After rare-class
top-up all classes have n≥20; per-class CI bands remain wide for the
smallest classes — reported, not hidden.
