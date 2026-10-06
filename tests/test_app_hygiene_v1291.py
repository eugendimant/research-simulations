"""Regression tests for the app-hygiene fixes (v1.2.9.1).

One section per fixed finding. The UI tests drive the real app with Streamlit's AppTest from a
temporary working directory, so no data/ folder lands in the repository.
"""
import contextlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_APP_DIR = Path(__file__).resolve().parent.parent / "simulation_app"
_USAGE_FILE = _APP_DIR / ".usage_counter.json"


@pytest.fixture()
def apptest_env(monkeypatch, tmp_path):
    """Run the app from a temporary cwd and never leave the runtime usage counter behind."""
    existed = _USAGE_FILE.exists()
    monkeypatch.chdir(tmp_path)
    if str(_APP_DIR) not in sys.path:
        sys.path.insert(0, str(_APP_DIR))
    yield tmp_path
    if not existed and _USAGE_FILE.exists():
        _USAGE_FILE.unlink()


def _scale(name: str = "Satisfaction", items: int = 3) -> dict:
    return {"name": name, "variable_name": name, "num_items": items, "scale_points": 7, "scale_min": 1,
            "scale_max": 7, "type": "matrix", "reverse_items": []}


def _generate_page(conds, n: int = 120, advanced: bool = True, extra_state=None):
    """An AppTest on the Generate page for a described (builder-path) study with `conds`."""
    from streamlit.testing.v1 import AppTest

    scales = [_scale()]
    state = {
        "active_page": 3, "study_title": "Hygiene study", "study_description": "A small pilot study of satisfaction.",
        "sample_size": n, "study_input_mode": "describe_study", "conversational_builder_complete": True,
        "confirmed_scales": scales, "scales_confirmed": True, "open_ended_confirmed": True,
        "selected_conditions": list(conds), "confirmed_conditions": list(conds), "team_name": "t",
        "team_members_raw": "a", "advanced_mode": advanced, "generation_method": "abe_v2",
        "allow_template_fallback_once": True, "_use_abe_v2": True, "_use_socsim_experimental": True,
        "inferred_design": {"conditions": list(conds), "factors": [{"name": "Condition", "levels": list(conds)}],
                            "scales": scales, "open_ended_questions": [], "attention_checks": [],
                            "manipulation_checks": [], "randomization_level": "Participant-level",
                            "condition_visibility_map": {}},
    }
    state.update(extra_state or {})
    at = AppTest.from_file(str(_APP_DIR / "app.py"), default_timeout=300)
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def _configure_effect(at, high: str, low: str, d: float = 0.8) -> None:
    at.checkbox(key="add_effect_checkbox").check()
    at.run()
    at.slider(key="effect_cohens_d").set_value(d)
    at.run()
    {s.key: s for s in at.selectbox}["effect_level_high"].select(high)
    at.run()
    {s.key: s for s in at.selectbox}["effect_level_low"].select(low)
    at.run()
    at.checkbox(key="_auto_effects_input").uncheck()  # every other contrast is a true null
    at.run()


def _click_generate(at) -> dict:
    next(b for b in at.button if b.key == "generate_dataset_btn").click()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.session_state["has_generated"]
    return _zip_members(at)


def _zip_members(at) -> dict:
    archive = zipfile.ZipFile(io.BytesIO(at.session_state["last_zip"]))
    return {name: archive.read(name) for name in archive.namelist()}


def _scale_mean_d(csv_bytes: bytes, high: str, low: str, name: str = "Satisfaction") -> float:
    df = pd.read_csv(io.BytesIO(csv_bytes))
    items = [c for c in df.columns if c.startswith(f"{name}_") and c[len(name) + 1:].isdigit()]
    y = df[items].astype(float).mean(axis=1)
    a, b = y[df["CONDITION"] == high], y[df["CONDITION"] == low]
    pooled = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled)


# ---- 1. effect direction: the text the user reads and the data agree ----------------------------
@pytest.mark.parametrize("levels, high, low", [
    (["Alpha", "Bravo"], "Alpha", "Bravo"),
    (["Alpha", "Bravo"], "Bravo", "Alpha"),
    (["Alpha", "Bravo", "Charlie"], "Charlie", "Alpha"),
    (["Alpha", "Bravo", "Charlie"], "Bravo", "Charlie"),
])
def test_effect_ui_text_metadata_and_data_all_say_the_high_condition_scores_higher(apptest_env, levels, high, low):
    at = _generate_page(levels, n=40 * len(levels) + 40)
    _configure_effect(at, high, low, d=0.8)
    assert not [r for r in at.radio if r.key == "effect_direction"], "the redundant Higher/Lower radio must be gone"
    messages = [s.value for s in at.success if "Effect configured" in s.value]
    assert messages and f"higher in '{high}' vs '{low}'" in messages[0], messages
    files = _click_generate(at)
    metadata = json.loads(files["Metadata.json"])
    (configured,) = metadata["effect_sizes_configured"]
    assert (configured["level_high"], configured["level_low"], configured["direction"]) == (high, low, "positive")
    contrast = next(r for r in metadata["effect_sizes_applied"]["contrasts"]
                    if r["source"] == "user" and {r["condition_1"], r["condition_2"]} == {high, low})
    sign = 1 if (contrast["condition_1"], contrast["condition_2"]) == (high, low) else -1
    assert sign * contrast["intended_d"] == pytest.approx(0.8)
    assert sign * contrast["observed_d"] > 0.3, contrast  # the data agree with the text: `high` scores higher
    assert _scale_mean_d(files["Simulated_Data.csv"], high, low) > 0.3
    summary = files["User_Study_Summary.md"].decode("utf-8")
    assert f"| Satisfaction | {high} | {low} | +0.80 |" in summary  # the design summary says the same


def test_engine_still_accepts_a_negative_direction_from_api_users():
    from utils.enhanced_simulation_engine import EffectSizeSpec, EnhancedSimulationEngine

    scale = _scale()
    spec = EffectSizeSpec(variable="Satisfaction", factor="Condition", level_high="Alpha", level_low="Bravo",
                          cohens_d=0.8, direction="negative")
    assert spec.direction == "negative"
    engine = EnhancedSimulationEngine(
        study_title="API", study_description="A pilot study of satisfaction", sample_size=300, conditions=["Alpha", "Bravo"],
        factors=[], scales=[scale], additional_vars=[], demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12},
        open_ended_questions=[], effect_sizes=[spec], seed=5, auto_effects=False)
    engine.llm_generator.disable_permanently("test")
    df, metadata = engine.generate()
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    assert _scale_mean_d(buffer.getvalue().encode(), "Alpha", "Bravo") < -0.3  # the API semantics are unchanged
    assert metadata["effect_sizes_configured"][0]["direction"] == "negative"


# ---- 2. stale results after the design changed ---------------------------------------------------
_STALE_TEXT = "design changed after this dataset was generated"
_COFFEE_QSF = _APP_DIR / "example_files" / "Coffee_Shop_Loyalty_Programs.qsf"


def _qsf_generate_page(n: int = 40):
    """An AppTest on the Generate page for the bundled Coffee QSF (the QSF path, not the builder path)."""
    from streamlit.testing.v1 import AppTest

    import app as appmod
    from utils.qsf_preview import QSFPreviewParser

    raw = _COFFEE_QSF.read_bytes()
    preview = QSFPreviewParser().parse(raw)
    inputs = appmod._preview_to_engine_inputs(preview)
    conds = [c.replace("\xa0", " ").strip() for c in inputs["conditions"]]
    at = AppTest.from_file(str(_APP_DIR / "app.py"), default_timeout=300)
    state = {
        "active_page": 3, "study_title": "Stale check", "study_description": "A pilot study of coffee shop loyalty programs.",
        "sample_size": n, "study_input_mode": "upload_qsf", "qsf_preview": preview, "qsf_raw_content": raw,
        "qsf_file_name": _COFFEE_QSF.name, "confirmed_scales": inputs["scales"], "confirmed_conditions": conds,
        "selected_conditions": list(conds), "team_name": "t", "team_members_raw": "a", "advanced_mode": False,
        "generation_method": "abe_v2", "allow_template_fallback_once": True, "_use_abe_v2": True,
        "_use_socsim_experimental": True, "scales_confirmed": True, "open_ended_confirmed": True,
        "confirmed_open_ended": [],  # the user dropped the detected text questions: numeric data only
        "inferred_design": {"conditions": conds, "factors": inputs["factors"], "scales": inputs["scales"],
                            "open_ended_questions": [], "attention_checks": [], "manipulation_checks": [],
                            "randomization_level": "Participant-level", "condition_visibility_map": {}},
    }
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    # The first visit to the Design page completes the design state (descriptions, visibility map);
    # a real user always passes it before generating.
    at.session_state["active_page"] = 2
    at.run()
    at.session_state["active_page"] = 3
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def _stale_warnings(at) -> list:
    return [w.value for w in at.warning if _STALE_TEXT in w.value]


def _goto(at, page: int) -> None:
    at.session_state["active_page"] = page
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]


def test_changing_n_after_generation_flags_the_old_download_as_stale_and_keeps_it(apptest_env):
    at = _qsf_generate_page(n=40)
    _click_generate(at)
    zip_before = at.session_state["last_zip"]
    assert not _stale_warnings(at)
    _goto(at, 2)  # the Design page: the user changes the sample size
    next(w for w in at.number_input if w.key == "sample_size_step3").set_value(200)
    at.run()
    assert at.session_state["sample_size"] == 200
    _goto(at, 3)
    assert len(_stale_warnings(at)) == 1, [w.value[:60] for w in at.warning]
    assert any("Simulation complete" in m.value for m in at.markdown), "the old results stay on screen"
    assert len(at.get("download_button")) >= 1 and at.session_state["last_zip"] == zip_before
    assert at.session_state["has_generated"], "nothing is cleared automatically"
    # Generating again replaces the dataset and the notice goes away
    next(b for b in at.button if b.key == "reset_after_gen_btn").click()
    at.run()
    assert not at.exception and not _stale_warnings(at)
    _click_generate(at)
    assert not _stale_warnings(at)
    assert json.loads(_zip_members(at)["Metadata.json"])["sample_size"] == 200


def test_visiting_the_design_page_without_changing_anything_does_not_flag_the_dataset(apptest_env):
    at = _qsf_generate_page(n=40)
    _click_generate(at)
    for page in (2, 3, 1, 2, 3):
        _goto(at, page)
    assert at.session_state["has_generated"] and not _stale_warnings(at)


@pytest.mark.parametrize("change", ["conditions", "dv", "effect", "method"])
def test_every_kind_of_design_change_is_detected(apptest_env, change):
    at = _generate_page(["Alpha", "Bravo", "Charlie"], n=90, advanced=False)
    _click_generate(at)
    assert not _stale_warnings(at)
    if change == "conditions":
        design = dict(at.session_state["inferred_design"])
        design["conditions"] = ["Alpha", "Bravo"]
        at.session_state["inferred_design"] = design
    elif change == "dv":
        at.session_state["confirmed_scales"] = [_scale("Satisfaction", items=5)]
    elif change == "effect":
        at.session_state["builder_effect_sizes"] = [{"variable": "Satisfaction", "factor": "condition", "level_high": "Alpha",
                                                     "level_low": "Bravo", "cohens_d": 0.5, "direction": "positive"}]
    else:
        at.session_state["generation_method"] = "template"
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert len(_stale_warnings(at)) == 1


def test_design_signature_ignores_ordering_and_numeric_types_but_sees_real_changes(monkeypatch, tmp_path):
    import importlib.util

    import streamlit as st

    monkeypatch.chdir(tmp_path)
    sys.path.insert(0, str(_APP_DIR))
    spec = importlib.util.spec_from_file_location("_app_hygiene_sig", str(_APP_DIR / "app.py"))
    app = importlib.util.module_from_spec(spec)
    sys.modules["_app_hygiene_sig"] = app
    try:
        spec.loader.exec_module(app)
    except SystemExit:
        pass
    base = {"sample_size": 100, "study_title": "T", "study_description": "D", "generation_method": "abe_v2",
            "confirmed_scales": [{"name": "A", "num_items": 3, "scale_points": 7, "scale_min": 1, "scale_max": 7}],
            "inferred_design": {"conditions": ["X", "Y"], "factors": [], "scales": []}}
    monkeypatch.setattr(st, "session_state", dict(base))
    reference = app._design_signature()
    assert reference and len(reference) == 16
    same = dict(base, sample_size=100.0, confirmed_scales=[{"scale_max": 7.0, "scale_min": 1, "scale_points": 7, "num_items": 3, "name": "A"}])
    monkeypatch.setattr(st, "session_state", same)
    assert app._design_signature() == reference
    for key, value in {"sample_size": 101, "generation_method": "template", "study_title": "T2",
                       "confirmed_scales": [{"name": "A", "num_items": 4, "scale_points": 7, "scale_min": 1, "scale_max": 7}],
                       "inferred_design": {"conditions": ["X", "Z"], "factors": [], "scales": []}}.items():
        monkeypatch.setattr(st, "session_state", dict(base, **{key: value}))
        assert app._design_signature() != reference, key
    monkeypatch.setattr(st, "session_state", dict(base))
    from utils.enhanced_simulation_engine import EffectSizeSpec

    spec_a = EffectSizeSpec(variable="A", factor="c", level_high="X", level_low="Y", cohens_d=0.5)
    spec_b = EffectSizeSpec(variable="A", factor="c", level_high="X", level_low="Y", cohens_d=0.6)
    assert app._design_signature([spec_a]) != reference and app._design_signature([spec_a]) != app._design_signature([spec_b])
    assert app._design_signature([spec_a]) == app._design_signature([EffectSizeSpec(
        variable="A", factor="c", level_high="X", level_low="Y", cohens_d=0.5)])


# ---- 3. admin tab: a stored package is addressed by its folder name, not its list position -------
class _FakeSMTP:
    sent: list = []

    def __init__(self, *args, **kwargs):
        pass

    def ehlo(self):
        pass

    def starttls(self, context=None):
        pass

    def login(self, user, password):
        pass

    def send_message(self, msg, from_addr=None, to_addrs=None, **kwargs):
        _FakeSMTP.sent.append(msg)
        return {}

    def quit(self):
        pass

    def close(self):
        pass


def _write_package(root: Path, folder: str, study: str) -> None:
    path = root / "data" / "simulation_runs" / folder
    path.mkdir(parents=True, exist_ok=True)
    (path / "INSTRUCTOR_Statistical_Report.html").write_text(f"<html><body>{study}</body></html>", encoding="utf-8")
    (path / "INSTRUCTOR_Detailed_Analysis.md").write_text(f"# {study}\n", encoding="utf-8")
    (path / "Metadata.json").write_text(json.dumps({"study_title": study, "run_id": folder}), encoding="utf-8")


def _admin_email_page(monkeypatch):
    import smtplib

    from streamlit.testing.v1 import AppTest

    _FakeSMTP.sent = []
    monkeypatch.setattr(smtplib, "SMTP", _FakeSMTP)
    monkeypatch.setattr(smtplib, "SMTP_SSL", _FakeSMTP)
    at = AppTest.from_file(str(_APP_DIR / "app.py"), default_timeout=120)
    for key, value in {"SMTP_SERVER": "smtp.example.org", "SMTP_USERNAME": "sender@example.org",
                       "SMTP_PASSWORD": "secret-" + "value", "INSTRUCTOR_NOTIFICATION_EMAIL": "owner@example.edu"}.items():
        at.secrets[key] = value
    at.query_params["admin"] = "1"
    at.session_state["_admin_authenticated"] = True
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def test_resend_button_of_a_stored_package_survives_a_new_run_finishing_before_the_click(apptest_env, monkeypatch):
    for folder, study in (("20261006_090000__PILOT_S1", "First study"), ("20261006_100000__PILOT_S2", "Second study")):
        _write_package(apptest_env, folder, study)
    at = _admin_email_page(monkeypatch)
    send_buttons = [b for b in at.button if str(b.key).startswith("_admin_pkg_send_")]
    assert len(send_buttons) == 2
    older = send_buttons[1]  # the second expander: the older package, "First study"
    for prefix in ("_admin_pkg_html_", "_admin_pkg_md_"):  # the keys name the package they belong to
        assert sum("PILOT_S1" in str(d.key) for d in at.get("download_button") if str(d.key).startswith(prefix)) == 1
    assert "PILOT_S1" in older.key
    # a student's run finishes while the admin page is open: the list shifts by one position
    _write_package(apptest_env, "20261006_110000__PILOT_S3", "Third study")
    older.click()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    subjects = [str(m["Subject"]) for m in _FakeSMTP.sent]
    assert subjects and all("[RE-SENT]" in s and "First study" in s for s in subjects), subjects  # not the shifted package


def test_stored_package_keys_are_unique_stable_and_safe():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_app_hygiene_keys", str(_APP_DIR / "app.py"))
    sys.path.insert(0, str(_APP_DIR))
    app = importlib.util.module_from_spec(spec)
    sys.modules["_app_hygiene_keys"] = app
    try:
        spec.loader.exec_module(app)
    except SystemExit:
        pass
    key = app._stored_package_key("_admin_pkg_send_", "20261006_100000__PILOT_S2")
    assert key == app._stored_package_key("_admin_pkg_send_", "20261006_100000__PILOT_S2")
    assert key.startswith("_admin_pkg_send_20261006_100000__PILOT_S2_")
    odd_a, odd_b = (app._stored_package_key("p_", name) for name in ("run one/é", "run one/e"))
    assert odd_a != odd_b and all(ch.isalnum() or ch in "_.-" for ch in odd_a)


# ---- 4. raw-HTML sinks: user, QSF and exception text is escaped -----------------------------------
_PAYLOAD = "<img src=x onerror=alert(1)>"


def test_scale_name_is_escaped_in_the_construct_badges_of_the_design_page(apptest_env):
    from streamlit.testing.v1 import AppTest

    import app as appmod

    raw = _COFFEE_QSF.read_bytes()
    from utils.qsf_preview import QSFPreviewParser

    preview = QSFPreviewParser().parse(raw)
    enhanced = appmod._perform_enhanced_analysis(qsf_content=raw)
    scales = [_scale("Trust " + _PAYLOAD), _scale("Satisfaction")]
    conds = ["Control", "No gamified tier", "Gamified No Tier", "Gamified Tier"]
    at = AppTest.from_file(str(_APP_DIR / "app.py"), default_timeout=300)
    state = {
        "active_page": 2, "study_title": "T", "study_description": "D study", "sample_size": 60,
        "study_input_mode": "upload_qsf", "qsf_preview": preview, "qsf_raw_content": raw, "qsf_file_name": _COFFEE_QSF.name,
        "enhanced_analysis": enhanced, "advanced_mode": True, "confirmed_scales": scales, "scales_confirmed": True,
        "open_ended_confirmed": True, "confirmed_open_ended": [], "selected_conditions": conds,
        "inferred_design": {"conditions": conds, "factors": [], "scales": scales, "open_ended_questions": [],
                            "attention_checks": [], "manipulation_checks": [], "randomization_level": "Participant-level",
                            "condition_visibility_map": {}},
    }
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    badges = [m.value for m in at.markdown if "border-radius:4px" in m.value and "Trust" in m.value]
    assert badges, "the construct badges for the two scales should be on the page"
    assert not [m for m in at.markdown if _PAYLOAD in m.value], "the scale name reached the page as live HTML"
    assert "&lt;img src=x onerror=alert(1)&gt;" in badges[0], badges[0]


def test_engine_failure_banner_shows_the_error_as_text_not_as_html(apptest_env, monkeypatch):
    from utils.enhanced_simulation_engine import EnhancedSimulationEngine

    def _boom(self, *args, **kwargs):
        raise RuntimeError("bad <img src=x onerror=alert(2)> input")

    monkeypatch.setattr(EnhancedSimulationEngine, "__init__", _boom)
    at = _generate_page(["Alpha", "Bravo"], n=40, advanced=False)
    next(b for b in at.button if b.key == "generate_dataset_btn").click()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    banners = [m.value for m in at.markdown if "failed to initialize" in m.value]
    assert banners, [m.value[:80] for m in at.markdown][-6:]
    assert "<img" not in banners[0] and "bad &lt;img src=x onerror=alert(2)&gt; input" in banners[0]


def test_no_raw_html_call_interpolates_exception_text_unescaped():
    """Static guard: an exception caught in app.py never reaches unsafe_allow_html markup unescaped."""
    import ast
    import re

    tree = ast.parse((_APP_DIR / "app.py").read_text(encoding="utf-8"))
    looks_like_exception = re.compile(r"(^|_)(e|exc|err|error|ex)$|_exc\b|_err\b|exception", re.IGNORECASE)

    def _escaped_names(node):
        """Names that appear outside any html_escape(...) call inside `node`."""
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "html_escape":
            return []
        if isinstance(node, ast.Name):
            return [node.id]
        names = []
        for child in ast.iter_child_nodes(node):
            names.extend(_escaped_names(child))
        return names

    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and any(k.arg == "unsafe_allow_html" for k in node.keywords) and node.args):
            continue
        first = node.args[0]
        for part in ast.walk(first):
            if isinstance(part, ast.FormattedValue):
                for name in _escaped_names(part.value):
                    if looks_like_exception.search(name):
                        offenders.append((node.lineno, name))
    assert not offenders, f"unescaped exception text in raw HTML at (line, name): {offenders}"
