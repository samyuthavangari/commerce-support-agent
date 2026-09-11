# REVIEW.md — Skeptical reviewer audit (2026-09-10, final pass)

Verdict per assignment requirement. Evidence paths given; nothing taken on trust.

| # | Requirement | Verdict | Evidence / note |
|---|---|---|---|
| 1 | One brand, all 3 tasks | PASS | @AmazonHelp; `src/agent.py::run_agent` live-probed; Streamlit UI boot + interaction tested |
| 2 | Runnable repo | PASS | `--fast` 2:59, full 9:38 (4 workers), `--offline` keyless; leakage gate + out-dir routing; 27/27 pytest green (bare `pytest` and `python -m pytest`) |
| 3 | Golden 150–250 reviewed + note | PASS | 250 rows, all verified, min class 24; cal-100 + cal2-100 + 2 frozen held-outs; `docs/golden_set.md` |
| 4 | Automated harness | PASS | `run_eval.py` (+`--fast`/`--resume`/`--offline`/`--workers`) → `eval_report.json` with provenance + CIs + error quarantine |
| 5 | LLM-as-judge rubric | PASS | 5 dims, strict JSON, safety caps, source-blind |
| 6 | Judge↔human agreement evidence | WARN | n=70 direct ρ=-0.062 (target missed, honestly reported); frozen blind A/B direction agrees |
| 7 | Trivial baseline | PASS | B0 0.220/0.0515, same golden set |
| 8 | Simple baseline | PASS | Fair B1 0.508/0.464, 8k weak threads, provenance; contaminated 0.92 withdrawn |
| 9 | Failure analysis top-5 + real examples | PASS | README + REPORT §12 incl. reverted prompt experiment |
| 10 | "Misleading" section | PASS | Exact title, REPORT §13 + README |
| 11 | One-week next steps | PASS | Concrete: v6 calibration, adjudication, boundary features, confidence calibration |
| 12 | Decision log 10–15 | PASS | `DECISIONS.md`, exactly 15, WHAT/WHY/ALTERNATIVES/TRADEOFF |
| 13 | Repro <15 min | PASS | Full 9:38, fast 2:59 — both measured, both under |
| 14 | Citations | PASS | README "Borrowed & Cited" |
| 15 | No leakage | PASS | Eval threads excluded; manifest gate fails eval otherwise; `check_leakage.py` passes (0 overlap, 0 verbatim, 0 same-thread top-5) |
| 16 | No fake evidence | PASS | 5-set table, held-out misses, failed v2 + failed judge validation published |
| 17 | Tests | PASS | 27 pytest all green (bare `pytest` + `python -m pytest`), offline, error-quarantine regression test |
| 18 | Secrets safety | PASS | `.env` gitignored; `.env.example` documents all vars |
| 19 | Offline/demo mode | PASS | `--offline` verified; Streamlit AppTest boot + live interaction green |
| 20 | Submission-ready link | FAIL | No git repo (owner action, explicitly out of agent scope) |
| 21 | Cross-platform | PASS | Windows cp1252 encoding crash in `check_leakage.py` fixed (Unicode ≥ → ASCII >=) |
| 22 | Metric consistency | PASS | README, REPORT.md, golden_250.csv, and eval_report.json all verified synchronized |

**Overall: 20 PASS / 1 WARN / 1 FAIL (owner-action).**
Residual risks, stated plainly: single reviewer throughout; judge absolute
grades untrusted (human primary); escalation held-outs 0.73/0.36; PRIME/ACCOUNT
pair accuracy 0.527 after a reverted fix attempt.
