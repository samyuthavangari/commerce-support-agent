"""Offline test suite — no API keys, no network, no Qdrant needed.

Covers: preprocessing normalization, taxonomy schema, escalation policy
(incl. golden + heldout expectations that pin the calibration contract),
leakage hygiene on local files, metric sanity, judge safety rules,
URL guardrail, and report-artifact validity.

Run:  python -m pytest tests/ -q
"""

import json
import os
import re
import sys
from pathlib import Path

import pytest

# Dummy creds BEFORE importing src modules (config refuses empty keys).
os.environ.setdefault("GOOGLE_API_KEY", "test-key-not-used-offline")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # run_eval.py lives at repo root
sys.path.insert(0, str(ROOT / "src"))  # src/ modules

import pandas as pd  # noqa: E402


# ── preprocessing: tweet-ID normalization ────────────────────────────────

def _norm_ids(s: pd.Series) -> pd.Series:
    """Mirror of the data_prep.py fix: '272.0' -> '272'."""
    return s.fillna("").str.strip().str.replace(r"\.0$", "", regex=True)


def test_id_normalization_joins_parents():
    tweet_ids = pd.Series(["1", "272", "999"])
    parents = _norm_ids(pd.Series(["272.0", "5.0", None]))
    assert "272" in set(tweet_ids) & set(parents)
    assert "" not in set(tweet_ids)


def test_id_normalization_leaves_integers_alone():
    assert _norm_ids(pd.Series(["12345"])).iloc[0] == "12345"


# ── taxonomy schema ──────────────────────────────────────────────────────

def test_taxonomy_schema():
    from intent_taxonomy import INTENTS, INTENT_NAMES
    assert len(INTENTS) == 7
    assert len(set(INTENT_NAMES)) == 7
    labels = sorted(i.label for i in INTENTS)
    assert labels == list(range(7))
    for intent in INTENTS:
        assert intent.trigger_keywords, intent.name
        assert len(intent.examples) >= 2, intent.name


def test_intent_schema_json_matches_code():
    from intent_taxonomy import INTENT_NAMES
    schema = json.loads((ROOT / "data" / "intent_schema.json").read_text(encoding="utf-8"))
    assert sorted(i["name"] for i in schema["intents"]) == sorted(INTENT_NAMES)
    for item in schema["intents"]:
        for key in ("definition", "include", "exclude", "examples",
                    "confusing_neighbors"):
            assert item[key], (item["name"], key)


# ── escalation policy ────────────────────────────────────────────────────

def _decide(msg: str) -> str:
    from escalation import check_hard_rules
    hit = check_hard_rules(msg, "GENERAL_COMPLAINT", 0.9)
    return hit.triggered_by if hit else "auto"


def test_legal_threat_escalates():
    assert _decide("I will sue you and contact my lawyer") == "rule:legal"


def test_rhetorical_fraud_venting_does_not_escalate():
    # Pure venting with no money context stays auto (documented tradeoff).
    assert _decide("This company is an absolute fraud, worst ever") == "auto"


def test_fraud_charge_escalates():
    assert _decide("DISPUTE THE WHOLE CHARGE PLEASE") == "rule:fraud"


def test_public_order_id_escalates():
    assert _decide("Order 408-1899080-9287553 never arrived") == "rule:public_pii"


def test_repeat_contact_escalates():
    assert _decide("I contacted twice, no one handling it") == "rule:repeat_contact"


def test_human_ask_escalates():
    assert _decide("Is there a real person I can email?") == "rule:human_ask"


def test_routine_query_auto_handles():
    assert _decide("My package is late, where is it?") == "auto"


def test_golden_escalation_contract():
    """Pin the calibration contract: golden recall must stay high; any rule
    change that breaks this must recalibrate openly. (R fell 1.00 -> 0.88
    when 50 harder top-up rows joined golden; floor documents the new bar.)"""
    from escalation import check_hard_rules
    g = pd.read_csv(ROOT / "golden_set" / "golden_250.csv")
    y = g["escalate_yn"].astype(bool).tolist()
    p = [check_hard_rules(str(t), "GENERAL_COMPLAINT", 0.9) is not None
         for t in g["text"].astype(str)]
    tp = sum(1 for a, b in zip(y, p) if a and b)
    fp = sum(1 for a, b in zip(y, p) if not a and b)
    rec = tp / max(sum(y), 1)
    assert rec >= 0.95, f"golden escalation recall regressed to {rec:.3f}"
    assert tp / max(tp + fp, 1) >= 0.850


def test_heldout_generalization_floor():
    """Documents the KNOWN generalization gap: frozen rules score ~0.53
    on fresh phrasings. Fails only if it gets worse."""
    from escalation import check_hard_rules
    h = pd.read_csv(ROOT / "golden_set" / "heldout_50.csv")
    y = h["escalate_yn"].astype(bool).tolist()
    p = [check_hard_rules(str(t), "GENERAL_COMPLAINT", 0.9) is not None
         for t in h["text"].astype(str)]
    tp = sum(1 for a, b in zip(y, p) if a and b)
    rec = tp / max(sum(y), 1)
    assert rec >= 0.50, f"heldout recall regressed to {rec:.3f}"


def test_v4_four_set_contract():
    """Pin the frozen v4 table (docs/escalation_v4.md). Cal drove iteration;
    golden/heldoutA/B are report-only — thresholds below are floors, and any
    rule edit that breaks them must recalibrate openly."""
    from escalation import check_hard_rules

    def recall(csv_name):
        df = pd.read_csv(ROOT / "golden_set" / csv_name)
        y = df["escalate_yn"].astype(bool).tolist()
        p = [check_hard_rules(str(t), "GENERAL_COMPLAINT", 0.9) is not None
             for t in df["text"].astype(str)]
        tp = sum(1 for a, b in zip(y, p) if a and b)
        return tp / max(sum(y), 1)

    assert recall("cal_100.csv") >= 0.80
    assert recall("golden_250.csv") >= 0.85  # 250 rows since rare-class top-up
    assert recall("heldoutB_50.csv") >= 0.30


# ── leakage hygiene (local files only; live index in check_leakage.py) ───

def test_golden_heldout_disjoint():
    g = set(pd.read_csv(ROOT / "golden_set" / "golden_250.csv")["thread_id"].astype(str))
    h = set(pd.read_csv(ROOT / "golden_set" / "heldout_50.csv")["thread_id"].astype(str))
    assert not (g & h)


def test_golden_sampled_from_corpus():
    corpus_path = ROOT / "data" / "processed" / "amazon_threads.jsonl"
    if not corpus_path.exists():
        pytest.skip("amazon_threads.jsonl not present (large corpus is gitignored for CI)")
    import json as J
    tids = set()
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            tids.add(str(J.loads(line)["thread_id"]))
    g = pd.read_csv(ROOT / "golden_set" / "golden_250.csv")
    # Original 200 come from the corpus; the 50 rare-class top-up rows were
    # deliberately sampled OUTSIDE it (no index rebuild needed).
    main = set(g[g["message_id"] <= "GS_0200"]["thread_id"].astype(str))
    topup = set(g[g["message_id"] > "GS_0200"]["thread_id"].astype(str))
    assert len(main - tids) == 0
    assert not (topup & tids), "top-up rows must stay outside the index"


def test_known_dupe_pairs_share_labels():
    g = pd.read_csv(ROOT / "golden_set" / "golden_250.csv").set_index("message_id")
    for a, b in [("GS_0006", "GS_0184"), ("GS_0074", "GS_0129"), ("GS_0109", "GS_0197")]:
        assert g.loc[a, "true_intent"] == g.loc[b, "true_intent"], (a, b)
        assert bool(g.loc[a, "escalate_yn"]) == bool(g.loc[b, "escalate_yn"]), (a, b)


# ── metrics sanity ───────────────────────────────────────────────────────

def test_perfect_predictions_score_one():
    from eval.metrics import compute_intent_metrics, compute_escalation_metrics
    y = ["A", "B", "A", "B"]
    m = compute_intent_metrics(y, y, labels=["A", "B"])
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0
    e = compute_escalation_metrics([True, False], [True, False])
    assert e["f1"] == 1.0 and e["false_auto_handle_rate"] == 0.0


def test_bootstrap_ci_contains_point_estimate():
    from eval.metrics import bootstrap_ci
    y = ["A", "B"] * 50
    p = ["A", "B"] * 48 + ["B", "A"] * 2
    ci = bootstrap_ci(y, p, "accuracy", n_boot=200, seed=0)
    assert ci["lo_95"] <= 0.96 <= ci["hi_95"]


# ── guardrail + judge rules (pure functions) ─────────────────────────────

def test_sanitize_reply_strips_urls():
    from agent import sanitize_reply
    out = sanitize_reply("DM us at https://t.co/abc for help  thanks")
    assert "http" not in out and "  " not in out
    assert sanitize_reply("plain reply, no links.") == "plain reply, no links."


def test_judge_safety_cap_on_url():
    from eval.llm_judge import _apply_safety_rules
    s = {"safety": 5, "weaknesses": ""}
    _apply_safety_rules(s, "see https://t.co/abc for help")
    assert s["safety"] == 2 and "[rule]" in s["weaknesses"]


def test_judge_safety_floor_on_pii_ask():
    from eval.llm_judge import _apply_safety_rules
    s = {"safety": 5, "weaknesses": ""}
    _apply_safety_rules(s, "Reply here with your card number please")
    assert s["safety"] == 1


def test_judge_leaves_clean_reply_alone():
    from eval.llm_judge import _apply_safety_rules
    s = {"safety": 5, "weaknesses": ""}
    _apply_safety_rules(s, "Please send us a DM with your order number")
    assert s["safety"] == 5


# ── Streamlit UI boots ───────────────────────────────────────────────────

def test_app_boots_without_exception():
    """The demo must import and render (button + input present). Slow (~20s
    streamlit import); interaction is covered manually, not here."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not list(at.exception)
    assert len(at.button) == 1 and len(at.text_area) == 1


# ── harness output routing ───────────────────────────────────────────────

def test_fast_offline_writes_elsewhere(tmp_path, monkeypatch):
    """Regression test: --fast/--offline runs must never clobber results/.
    (An offline smoke test once overwrote the 250-row live artifacts.)"""
    import run_eval
    before = pd.read_csv(ROOT / "results" / "agent_replies.csv")
    monkeypatch.setattr(
        sys, "argv",
        ["run_eval.py", "--offline", "--max-examples", "5",
         "--out-dir", str(tmp_path)])
    run_eval.main()
    after = pd.read_csv(ROOT / "results" / "agent_replies.csv")
    assert len(before) == len(after) == 250
    assert (tmp_path / "agent_replies.csv").exists()
    assert (tmp_path / "eval_report.json").exists()


# ── error quarantine ─────────────────────────────────────────────────────

def test_agent_errors_quarantined_not_fabricated(tmp_path, monkeypatch):
    """An agent exception must NEVER become an invented GENERAL_COMPLAINT /
    escalate row inside the metrics. Errors go to agent_errors.csv, metrics
    cover evaluated rows only."""
    import json as J
    import types as _types
    import run_eval

    calls = {"n": 0}

    def flaky_agent(message, qdrant_client=None, verbose=False):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated outage")
        return _types.SimpleNamespace(
            intent="ORDER_STATUS", intent_confidence=0.9,
            intent_reasoning="stub", retrieved_examples=[],
            draft_reply="stub reply", reply_char_count=10,
            escalation_decision="auto", escalation_reason="stub")

    monkeypatch.setattr(run_eval, "run_agent", flaky_agent)
    monkeypatch.setattr(
        sys, "argv",
        ["run_eval.py", "--max-examples", "30", "--no-judge",
         "--out-dir", str(tmp_path)])
    run_eval.main()

    replies = pd.read_csv(tmp_path / "agent_replies.csv")
    errors = pd.read_csv(tmp_path / "agent_errors.csv")
    assert len(errors) == 1 and len(replies) == 29
    assert "[AGENT ERROR]" not in replies["draft_reply"].tolist()
    rep = J.loads((tmp_path / "eval_report.json").read_text())
    assert rep["num_examples"] == 29


# ── report artifacts ─────────────────────────────────────────────────────

def test_eval_report_valid_json_no_nan():
    import math
    def no_nan(o):
        raise ValueError("NaN in eval_report.json")
    text = (ROOT / "results" / "eval_report.json").read_text()
    rep = json.loads(text, parse_constant=no_nan)
    assert rep["num_examples"] == 250
    for key in ("intent_classification", "escalation", "baselines",
                "reply_quality_judge", "human_judge_agreement", "provenance"):
        assert key in rep, key
