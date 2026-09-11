# Live Technical Walkthrough & Code Defense Guide

This guide prepares the candidate for live technical interviews and code walkthroughs. For each of the **5 critical files** in the repository, it provides a strict **<60-second spoken explanation**:
1. **What it does**
2. **Why it exists**
3. **What breaks if you delete it**
4. **The reviewer question you will get & how to answer it**

---

## 1. `src/agent.py` — Pipeline Coordinator & Routing Engine

### Spoken 60-Second Walkthrough
> "`src/agent.py` is the central orchestrator that links classification, vector retrieval, grounded drafting, and escalation policy into a unified, reproducible callable.
> 
> It exists to decouple the business pipeline from the evaluation harness. When a message comes in, it classifies intent with strict JSON schema parsing, retrieves the top-5 historical resolutions filtered by that intent, drafts a 280-character Twitter reply using few-shot RAG with a deterministic URL stripping guardrail, and passes the context to the escalation gate to make the final routing decision (`AUTO_HANDLE` vs `ESCALATE_TO_HUMAN`).
> 
> If you delete this file, the entire end-to-end support system breaks. Neither `run_eval.py`, `app.py`, nor `demo.py` can process customer inquiries, and you have no cohesive pipeline to benchmark."

### Anticipated Reviewer Question & Defense
- **Q: "Why do you store both raw confidence and calibrated confidence in `AgentResponse` if the active gate still uses raw confidence?"**
- **Defense**: *"Our diagnostic on golden-250 proved the raw LLM is severely overconfident (ECE 0.1518; a 0.90 verbalized confidence only corresponds to 57.1% accuracy). We implemented isotonic calibration in `calibrate_confidence()` and store it per response. We chose not to change the locked routing gate to use calibrated confidence on the test benchmark because that would violate our frozen evaluation protocol — doing so is scheduled for v6."*

---

## 2. `src/escalation.py` — Deterministic Multi-Tier Safety Gate

### Spoken 60-Second Walkthrough
> "`src/escalation.py` is the deterministic safety barrier that decides whether an inquiry is safe to auto-handle or must be escalated to a human agent with a structured reason code.
> 
> It exists because in enterprise customer support, false negatives on safety, legal threats, and account fraud carry catastrophic brand and liability risk. Rather than relying on an uncalibrated LLM prompt to self-police risk, it applies a prioritized cascade of deterministic regex rules: legal threats, fraud/unauthorized charges, public PII (phone/email/order IDs), physical safety, explicit human agent requests, repeat contact loops, and long conversation threads.
> 
> If you delete it, the agent will auto-reply to customers claiming fraud or threatening lawsuits with canned bot responses, destroying compliance and safety guarantees."

### Anticipated Reviewer Question & Defense
- **Q: "Why does your escalation recall drop from 0.981 on golden-250 down to 0.733 on Heldout-A and 0.364 on Heldout-B?"**
- **Defense**: *"Because regex rules memorize the exact lexical patterns seen during calibration (e.g. `unauthorized transaction`), but fail on novel paraphrases (`charged twice`, `empty box`) or typos (`fladuent`). That is why we explicitly state we would not ship this system without closing this gap via semantic vector gating."*

---

## 3. `src/qdrant_store.py` — Dual-Engine Vector Retrieval (Docker + Pure-Python)

### Spoken 60-Second Walkthrough
> "`src/qdrant_store.py` handles indexing and filtered vector retrieval over historical `@AmazonHelp` customer-reply pairs, offering dual execution modes: production Qdrant or pure-Python in-memory numpy.
> 
> It exists because intent-filtered retrieval is load-bearing: our ablation showed unfiltered top-1 retrieval has only 49% intent-consistency. To guarantee zero-Docker instant reproducibility, it includes `InMemoryVectorStore`, which loads pre-computed normalized embeddings from `data/memory_vectors.npz` and runs cosine similarity via numpy matrix multiplication in under 2ms.
> 
> If you delete it, RAG reply drafting falls back to ungrounded zero-shot generation (which scored 23.46 on ablation vs 24.34 grounded), and anyone running without Docker cannot execute evaluation."

### Anticipated Reviewer Question & Defense
- **Q: "Why embed `customer + SEP + reply` instead of customer inquiry alone?"**
- **Defense**: *"Because customer queries are ambiguous. 'Where is my order?' paired with 'Please DM us your tracking number' captures resolution style and conversational context far better than query text alone."*

---

## 4. `run_eval.py` — Unified Benchmark & Baselines Harness

### Spoken 60-Second Walkthrough
> "`run_eval.py` is the single-command evaluation harness that runs the agent against the 250-row human-reviewed benchmark, executes trivial (B0) and fair TF-IDF (B1) baselines, and generates `eval_report.json`.
> 
> It exists to provide reproducible verification of all headline claims. It includes an automated data-leakage gate that asserts zero overlap between eval threads and vector store points, supports parallel multi-worker execution (full eval in 9:38, fast 30-sample run in 2:59), and accepts `--no-docker` for pure-Python execution.
> 
> If you delete it, there is no standardized, reproducible way to measure intent accuracy, escalation recall, or baseline comparisons."

### Anticipated Reviewer Question & Defense
- **Q: "Why did your earlier TF-IDF baseline achieve 92% accuracy, and why is it now 50.8%?"**
- **Defense**: *"The earlier baseline was contaminated because it was fitted on the golden set itself (train=test). We discovered the leak, withdrew the number, and retrained a fair baseline on 8,000 weak-labelled off-golden threads. That lowered baseline accuracy to 50.8%, making our agent's 82.0% an honest +31.2% improvement."*

---

## 5. `src/eval/llm_judge.py` — 5D Evaluation Rubric & Human Agreement

### Spoken 60-Second Walkthrough
> "`src/eval/llm_judge.py` implements the automated 5-dimensional evaluation rubric (Groundedness, Helpfulness, Empathy, Safety, Brevity) and computes correlation against blind human ratings.
> 
> It exists to evaluate reply quality systematically, but more importantly, to document LLM-as-judge failure modes. It enforces hard deterministic safety caps: overriding the judge to score Safety=1 if a bot asks for PII in public. Crucially, it computes Spearman correlation against human scores, demonstrating that the judge has near-zero rank correlation (ρ = −0.062) and runs ~+4 points lenient.
> 
> If you delete it, you lose the automated quality evaluation and the empirical evidence that LLM judges cannot be trusted for absolute grading without human calibration."

### Anticipated Reviewer Question & Defense
- **Q: "If the judge doesn't correlate with human grades (ρ = −0.062), why keep it at all?"**
- **Defense**: *"Because in blind A/B evaluation, the judge agrees with human ranking on system direction: human scores agent 20.7 vs template 19.0 (+1.7), while judge scores agent 24.6 vs template 22.1 (+2.5). The judge discriminates systems, but suffers from severe ceiling leniency and cannot grade absolute individual quality."*

---

## Summary Cheat Sheet for Live Defense

| Metric / Claim | What to Say | What NOT to Say |
|---|---|---|
| **Escalation Recall** | "0.981 on golden-250, but 0.733 and 0.364 on frozen held-outs. We would not ship without closing this gap." | "Our system achieves 98.1% recall in production." |
| **Intent Accuracy** | "0.820 overall on a hard 7-intent benchmark with rare-class top-up (+31.2% over fair TF-IDF)." | "State-of-the-art 88% intent classifier." |
| **LLM Judge Score** | "24.44/25, but that reflects judge leniency (+4 bias). Blind human score is 20.7/25." | "The LLM judge proves near-perfect reply quality." |
| **Agreement Metric** | "κ=0.939 (LLM-vs-human-reviewer agreement, single reviewer, no adjudication pass)." | "Inter-annotator agreement κ=0.939." |
| **Cold-Start Setup** | "4:15 in pure-Python `--no-docker` mode; 8:40 if pulling Qdrant Docker." | "Instant zero-second cold start." |
