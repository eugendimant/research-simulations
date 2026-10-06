"""Robustness of the instructor-analysis reports (Markdown, HTML) and of how the app delivers them.

Everything here builds small DataFrames/metadata directly (no engine run), so it is fast. The statistics
tests run twice: with scipy and with the pure-numpy fallbacks the deployed app uses (scipy is not in
requirements.txt), by switching the module's availability flags.
"""
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_APP_DIR = Path(__file__).resolve().parent.parent / "simulation_app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import utils.instructor_report as ir  # noqa: E402
from utils.instructor_report import ComprehensiveInstructorReport  # noqa: E402

NAN_INF = re.compile(r"(?<![A-Za-z0-9_])(nan|NaN|inf|-inf|Infinity)(?![A-Za-z0-9_])")


@pytest.fixture(params=["scipy", "numpy-fallback"])
def stats_mode(request, monkeypatch):
    """Run the test with scipy and with the numpy fallbacks."""
    if request.param == "numpy-fallback":
        monkeypatch.setattr(ir, "SCIPY_AVAILABLE", False)
        monkeypatch.setattr(ir, "scipy_stats", None)
    elif not ir.SCIPY_AVAILABLE:
        pytest.skip("scipy is not installed")
    return request.param


def _text_of_html(markup: str) -> str:
    """The visible text of a report (tags dropped), enough to scan for stray nan / inf."""
    markup = re.sub(r"<style.*?</style>", " ", markup, flags=re.S)
    return re.sub(r"<[^>]+>", " | ", markup)


def _base_frame(n_per_condition: int = 30, conditions=("Control", "Treatment"), seed: int = 3) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    for cond in conditions:
        for _ in range(n_per_condition):
            rows.append({
                "PARTICIPANT_ID": len(rows) + 1, "CONDITION": cond, "Age": int(rng.randint(18, 70)),
                "Gender": str(rng.choice(["Male", "Female"])), "Attention_Pass_Rate": 1.0,
                "Completion_Time_Seconds": int(rng.randint(300, 900)), "Exclude_Recommended": 0,
            })
    return pd.DataFrame(rows)


def _likert_items(df: pd.DataFrame, name: str, shift_by_condition=None, k: int = 3, seed: int = 5) -> list:
    """Add k correlated 1-7 items named <name>_1..k; returns the column names."""
    rng = np.random.RandomState(seed)
    shift_by_condition = shift_by_condition or {}
    base = rng.normal(0, 1.0, len(df))
    cols = []
    for i in range(1, k + 1):
        col = f"{name}_{i}"
        mu = 4.0 + df["CONDITION"].map(shift_by_condition).fillna(0.0).to_numpy()
        df[col] = np.clip(np.round(mu + base + rng.normal(0, 0.8, len(df))), 1, 7).astype(int)
        cols.append(col)
    return cols


def _meta(df: pd.DataFrame, scales: list, registry: dict = None, **extra) -> dict:
    conditions = list(dict.fromkeys(df["CONDITION"].tolist()))
    meta = {
        "study_title": "Robustness study", "study_description": "Small synthetic study.", "conditions": conditions,
        "sample_size": len(df), "factors": [], "open_ended_questions": [], "scales": scales,
        "run_id": "R1", "generation_timestamp": "2026-10-06T10:00:00", "effect_sizes_observed": [],
        "scale_generation_log": [{"name": n, "columns_generated": cols} for n, cols in (registry or {}).items()],
    }
    meta.update(extra)
    return meta


def _scale(name: str, k: int, kind: str = "likert", lo: int = 1, hi: int = 7) -> dict:
    return {"name": name, "num_items": k, "scale_points": hi - lo + 1, "scale_min": lo, "scale_max": hi, "type": kind}


def _both_reports(df, meta, prereg: str = ""):
    gen = ComprehensiveInstructorReport()
    md = gen.generate_comprehensive_report(df=df, metadata=meta, schema_validation={}, prereg_text=prereg, team_info={})
    html = ComprehensiveInstructorReport().generate_html_report(df=df, metadata=meta, schema_validation={}, prereg_text=prereg, team_info={})
    return md, html


def _assert_clean(md: str, html: str):
    visible = _text_of_html(html)
    for label, text in (("markdown", md), ("html", visible)):
        assert not NAN_INF.search(text), f"{label}: {NAN_INF.search(text).group(0)!r} in {text[max(0, NAN_INF.search(text).start() - 80):NAN_INF.search(text).end() + 40]!r}"
    assert "Report Error" not in html and "Report generation encountered an error" not in md


# ---------------------------------------------------------------------------
# 1. A preregistration that yields a hypothesis used to stub both attachments
# ---------------------------------------------------------------------------
PREREG_TEXTS = [
    "Participants in the gamified group will see a tier badge; participants in the control group will not.",
    "1. Trust (higher = more trust)\n2. Purchase intention",
    "- Loyalty should be measured on a 7-point scale",
    "Trust will be higher in the Treatment condition",
    "Hypotheses\nH1: Participants in the treatment condition will report higher trust than those in the control condition.",
    "Analysis: independent samples t-test; we predict that trust is higher after the treatment.",
]


@pytest.fixture()
def trust_study():
    df = _base_frame(40)
    cols = _likert_items(df, "Trust", {"Treatment": 1.2})
    return df, _meta(df, [_scale("Trust", 3)], {"Trust": cols})


@pytest.mark.parametrize("prereg", PREREG_TEXTS)
def test_prereg_with_hypotheses_does_not_crash_either_report(trust_study, stats_mode, prereg):
    df, meta = trust_study
    assert ComprehensiveInstructorReport()._parse_prereg_hypotheses(prereg)["hypotheses"] or "H1" not in prereg
    md, html = _both_reports(df, meta, prereg)
    _assert_clean(md, html)
    assert "Executive Summary" in html


def test_hypotheses_are_plain_strings_and_the_summary_accepts_strings_and_dicts(trust_study, monkeypatch):
    df, meta = trust_study
    gen = ComprehensiveInstructorReport()
    parsed = gen._parse_prereg_hypotheses("Trust will be higher in the Treatment condition")
    assert parsed["hypotheses"] and all(isinstance(h, str) for h in parsed["hypotheses"])
    # an older caller (or a future parser) may return dicts: both shapes must work
    for shape in (["Trust will be higher in the Treatment condition"],
                  [{"text": "Trust will be higher in the Treatment condition"}],
                  [None, "", {"text": ""}, 42, "Trust will be higher in the Treatment condition"]):
        monkeypatch.setattr(ComprehensiveInstructorReport, "_parse_prereg_hypotheses",
                            lambda self, text, _s=shape: {"hypotheses": list(_s), "control_variables": []})
        html = ComprehensiveInstructorReport().generate_html_report(df=df, metadata=meta, prereg_text="x", team_info={})
        assert "Pre-Registration Hypotheses" in html and "Report Error" not in html


def test_hypothesis_lines_never_claim_support(stats_mode):
    df = _base_frame(60)
    cols = _likert_items(df, "Trust", {"Treatment": 1.5})
    _likert_items(df, "Loyalty", {}, seed=11)
    meta = _meta(df, [_scale("Trust", 3), _scale("Loyalty", 3)], {"Trust": cols, "Loyalty": ["Loyalty_1", "Loyalty_2", "Loyalty_3"]})
    prereg = "Participants in the Treatment condition will show lower trust.\nLoyalty will be higher in the Treatment condition."
    html = ComprehensiveInstructorReport().generate_html_report(df=df, metadata=meta, prereg_text=prereg, team_info={})
    text = _text_of_html(html)
    assert "Supported" not in text and "supported</strong>" not in html and "Not supported" not in text
    # the (opposite-direction) hypothesis about trust is only called "related", with the caveat that direction was not checked
    assert "a significant result on a related measure was found (Trust)" in text
    assert "the direction was not checked" in text
    assert "no significant related result" in text  # the loyalty hypothesis: no significant loyalty effect


# ---------------------------------------------------------------------------
# 3. Open-ended text columns and odd scale names in the markdown tables
# ---------------------------------------------------------------------------
def test_text_column_sharing_the_scale_prefix_is_not_a_scale_column():
    df = _base_frame(10)
    cols = _likert_items(df, "Punitive_Pilot", k=3)
    df["Punitive_Pilot_03"] = "I think the punishment was fair because of the story."
    scale = _scale("Punitive_Pilot", 3)
    assert ir._find_scale_columns(df, scale) == cols  # prefix match: the text column is left out
    assert ir._find_scale_columns(df, scale, {"Punitive_Pilot": cols + ["Punitive_Pilot_03"]}) == cols  # even a bad registry
    # a longer name that merely starts with the prefix is not an item of the shorter scale
    _likert_items(df, "Punitive", k=2, seed=9)
    df["Punitive_Pilot_mean"] = 4.0
    assert ir._find_scale_columns(df, _scale("Punitive", 2)) == ["Punitive_1", "Punitive_2"]


def test_markdown_range_table_survives_text_columns_and_lists_digit_leading_scales(stats_mode):
    df = _base_frame(30)
    cols_a = _likert_items(df, "Punitive_Pilot", {"Treatment": 0.8}, k=3)
    df["Punitive_Pilot_03"] = "free text answer"  # open-ended question named like an item
    cols_b = [f"1_9Q_{i}" for i in (1, 2)]
    for col in cols_b:
        df[col] = np.random.RandomState(2).randint(1, 8, len(df))
    meta = _meta(df, [_scale("Punitive_Pilot", 3), _scale("1.9Q", 2)], {"Punitive_Pilot": cols_a, "1.9Q": cols_b})
    md, html = _both_reports(df, meta)
    _assert_clean(md, html)
    table = md.split("### Automated Quality Checks")[1].split("###")[0]
    assert re.search(r"\| Punitive_Pilot \| 3 \|", table), table
    assert re.search(r"\| 1\.9Q \| 2 \|", table), table  # used to be left out: "1.9Q" never matched "1_9Q_1"
    # the text column is not described as a scale item anywhere in the DV analysis
    assert "| Punitive_Pilot_03 |" not in md


def test_scale_without_any_numeric_column_is_reported_not_dropped_or_crashed():
    df = _base_frame(10)
    df["Q7_1"] = "text"
    md, html = _both_reports(df, _meta(df, [_scale("Q7", 1)], {"Q7": ["Q7_1"]}))
    assert "| Q7 | 0 |" in md and "no columns found" in md
    assert "Report Error" not in html


# ---------------------------------------------------------------------------
# 2. The app delivers each document on its own
# ---------------------------------------------------------------------------
def _load_app():
    spec = importlib.util.spec_from_file_location("_app_report_robustness_test", str(_APP_DIR / "app.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["_app_report_robustness_test"] = module
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass
    return module


@pytest.fixture()
def app_env(monkeypatch, tmp_path):
    import streamlit as st

    monkeypatch.chdir(tmp_path)
    app = _load_app()
    monkeypatch.setattr(st, "session_state", {})
    return app


def _kwargs(df, meta):
    return dict(df=df, metadata=meta, schema_results={}, prereg_text="", team_info={"team_name": "T", "team_members": "A\nB"})


def test_app_builds_all_three_documents_when_nothing_fails(app_env, trust_study):
    df, meta = trust_study
    out = app_env._build_instructor_reports(**_kwargs(df, meta))
    assert out["problems"] == []
    assert "Report Error" not in out["comp_html"] and "encountered an error" not in out["comp_md"]
    assert out["comp_html"].lstrip().lower().startswith("<!doctype html") and "COMPREHENSIVE INSTRUCTOR REPORT" in out["comp_md"]
    assert out["student_md"].strip()


def test_a_markdown_failure_does_not_stub_the_html_report(app_env, trust_study, monkeypatch):
    df, meta = trust_study

    def boom(self, *a, **k):
        raise KeyError("DV_mean")

    monkeypatch.setattr(app_env.ComprehensiveInstructorReport, "generate_comprehensive_report", boom)
    out = app_env._build_instructor_reports(**_kwargs(df, meta))
    assert "encountered an error" in out["comp_md"] and "DV_mean" in out["comp_md"]
    assert "Report Error" not in out["comp_html"] and "Executive Summary" in out["comp_html"]  # the real HTML report
    assert out["student_md"].strip() and "encountered an error" not in out["student_md"]
    assert out["problems"] == ["instructor analysis (Markdown): KeyError: 'DV_mean'"]


def test_an_html_failure_does_not_stub_the_markdown_report(app_env, trust_study, monkeypatch):
    df, meta = trust_study

    def boom(self, *a, **k):
        raise ValueError("bad chart")

    monkeypatch.setattr(app_env.ComprehensiveInstructorReport, "generate_html_report", boom)
    out = app_env._build_instructor_reports(**_kwargs(df, meta))
    assert "<h1>Report Error</h1>" in out["comp_html"] and "bad chart" in out["comp_html"]
    assert "encountered an error" not in out["comp_md"] and "COMPREHENSIVE INSTRUCTOR REPORT" in out["comp_md"]
    assert out["problems"] == ["instructor analysis (HTML): ValueError: bad chart"]


def test_a_study_summary_failure_stubs_only_the_study_summary(app_env, trust_study, monkeypatch):
    df, meta = trust_study

    def boom(self, *a, **k):
        raise RuntimeError("summary down")

    monkeypatch.setattr(app_env.InstructorReportGenerator, "generate_markdown_report", boom)
    out = app_env._build_instructor_reports(**_kwargs(df, meta))
    assert "encountered an error" in out["student_md"] and "summary down" in out["student_md"]
    assert "Report Error" not in out["comp_html"] and "encountered an error" not in out["comp_md"]
    assert out["problems"] == ["study summary: RuntimeError: summary down"]


def test_all_three_can_fail_together_and_each_is_listed(app_env, trust_study, monkeypatch):
    df, meta = trust_study

    def boom(self, *a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(app_env.InstructorReportGenerator, "generate_markdown_report", boom)
    monkeypatch.setattr(app_env.ComprehensiveInstructorReport, "generate_comprehensive_report", boom)
    monkeypatch.setattr(app_env.ComprehensiveInstructorReport, "generate_html_report", boom)
    out = app_env._build_instructor_reports(**_kwargs(df, meta))
    assert [p.split(":")[0] for p in out["problems"]] == ["study summary", "instructor analysis (Markdown)", "instructor analysis (HTML)"]
    assert all("down" in out[k] for k in ("student_md", "comp_md", "comp_html"))


def test_an_empty_or_non_text_result_counts_as_a_failure(app_env, trust_study, monkeypatch):
    df, meta = trust_study
    monkeypatch.setattr(app_env.ComprehensiveInstructorReport, "generate_html_report", lambda self, *a, **k: "   ")
    monkeypatch.setattr(app_env.ComprehensiveInstructorReport, "generate_comprehensive_report", lambda self, *a, **k: None)
    out = app_env._build_instructor_reports(**_kwargs(df, meta))
    assert len(out["problems"]) == 2 and "no text" in out["problems"][0]
    assert "<h1>Report Error</h1>" in out["comp_html"] and "encountered an error" in out["comp_md"]
