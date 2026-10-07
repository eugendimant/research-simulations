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
    # 150 per cell and a pinned seed: with ~50 per cell the observed d for a request of 0.8 has an SD of
    # about 0.25, so a ">0.3" check failed by chance on roughly one run in thirty (seen once on CI).
    at = _generate_page(levels, n=150 * len(levels), extra_state={"_user_seed_value": 7})
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
    app = _load_app_module("_app_hygiene_keys")
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


# ---- 5. the builder suggests a within-subjects design from whole words, not substrings -----------
def _load_app_module(name: str):
    """Execute app.py as a plain module (like the import-safety tests), from a scratch cwd so that
    its data/ folder never lands in the repository."""
    import importlib.util
    import os
    import tempfile

    if str(_APP_DIR) not in sys.path:
        sys.path.insert(0, str(_APP_DIR))
    spec = importlib.util.spec_from_file_location(name, str(_APP_DIR / "app.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = os.getcwd()
    with tempfile.TemporaryDirectory() as scratch:
        os.chdir(scratch)
        try:
            spec.loader.exec_module(module)
        except SystemExit:
            pass
        finally:
            os.chdir(previous)
    return module


@pytest.mark.parametrize("names, expected", [
    (["Control", "Treatment"], "between"),
    (["Premium brand", "Standard brand"], "between"),
    (["Prevention message", "Promotion message"], "between"),
    (["Present", "Absent"], "between"),
    (["High followers", "Low followers"], "between"),
    (["Postal service", "Courier"], "between"),
    (["Afterthought framing", "Control"], "between"),
    (["Pre-test", "Post-test"], "within"),
    (["pre_treatment", "post_treatment"], "within"),
    (["Pre", "Post"], "within"),
    (["Time 1", "Time 2"], "within"),
    (["Wave 1", "Wave 2"], "within"),
    (["Before", "After"], "within"),
    (["Treatment", "Follow-up"], "within"),
    (["Treatment", "Followup"], "within"),
    (["Mixed feelings", "Calm"], "mixed"),
    (["Repeated exposure", "Single exposure"], "mixed"),
    (["Time 12 hours", "Control"], "between"),
    ([], "between"),
])
def test_design_suggestion_matches_whole_words_only(names, expected):
    app = _load_app_module("_app_hygiene_design")
    assert app._detect_design_from_condition_names(names) == expected


@pytest.mark.parametrize("conditions_text, expected", [
    ("Control\nTreatment", "between"),
    ("Premium brand\nStandard brand", "between"),
    ("Present\nAbsent", "between"),
    ("Pre-test\nPost-test", "within"),
])
def test_builder_radio_preselects_the_design_from_the_condition_labels(apptest_env, conditions_text, expected):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(_APP_DIR / "app.py"), default_timeout=300)
    for key, value in {"active_page": 1, "study_input_mode": "describe_study", "study_title": "T", "study_description": "D study",
                       "builder_conditions_text": conditions_text, "builder_scales_text": "Trust (1-7 Likert, 5 items)",
                       "cond_input_mode": "Text / Factorial notation"}.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    radio = next(r for r in at.radio if r.key == "builder_design_type_input")
    assert radio.value == expected
    note = [w for w in at.warning if "one condition per participant" in w.value]
    assert bool(note) == (expected != "between")


# ---- 6. the design-structure note says only what is true -----------------------------------------
def test_design_structure_note_does_not_claim_the_design_type_is_recorded(apptest_env):
    at = _qsf_generate_page(n=40)
    _goto(at, 2)
    next(s for s in at.selectbox if s.key == "design_type_select").select("Within-subjects (each participant sees all conditions)")
    at.run()
    notes = [w.value for w in at.warning if "one condition per participant" in w.value]
    assert len(notes) == 1, notes
    note = notes[0]
    _goto(at, 3)
    metadata = json.loads(_click_generate(at)["Metadata.json"])
    # A note may only promise that the choice is recorded when Metadata.json really holds it.
    if "recorded" in note.lower() or "design summary" in note.lower():
        assert "design_type" in metadata.get("design_review", {}), "the note claims a record that does not exist"
    assert "repeated measures" in note and "does not change" in note
    assert str(metadata["design_review"].get("randomization_level")).startswith("Participant-level")


# ---- 7. QSF collector: duplicates do not spend the upload budget, names stay distinct ------------
_QSF_A = json.dumps({"SurveyEntry": {"SurveyID": "SV_A"}, "SurveyElements": []}).encode()
_QSF_B = json.dumps({"SurveyEntry": {"SurveyID": "SV_B"}, "SurveyElements": []}).encode()


class _Reply:
    def __init__(self, status, payload=None):
        self.status_code, self._payload = status, payload if payload is not None else {}

    def json(self):
        return self._payload


@pytest.fixture()
def collector(monkeypatch):
    """The collector with a recorded fake GitHub: `repo` maps file name -> content already stored."""
    import types

    from utils import github_qsf_collector as coll

    repo: dict = {}
    calls = {"get": 0, "put": []}

    def fake_get(url, headers=None, params=None, timeout=None, **_kw):
        calls["get"] += 1
        return _Reply(200, [{"name": name, "type": "file", "sha": coll._git_blob_sha(body)} for name, body in repo.items()])

    def fake_put(url, headers=None, json=None, timeout=None, **_kw):  # noqa: A002 - mirrors requests.put
        import base64

        name = url.rsplit("/", 1)[-1]
        calls["put"].append(name)
        repo[name] = base64.b64decode(json["content"])
        return _Reply(201)

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(get=fake_get, put=fake_put))
    monkeypatch.setattr(coll, "_get_config", lambda: {"token": "ghp_" + "a" * 36, "repo": "o/r", "path": "p",
                                                       "enabled": True, "branch": "collected"})
    coll._upload_times.clear()
    coll._lookup_times.clear()
    yield coll, repo, calls
    coll._upload_times.clear()
    coll._lookup_times.clear()


def test_duplicate_uploads_do_not_use_up_the_hourly_upload_budget(collector):
    coll, repo, calls = collector
    for i in range(coll.MAX_UPLOADS_PER_HOUR + 5):
        repo[f"2026_10_06_file{i}.qsf"] = _QSF_A
    for i in range(coll.MAX_UPLOADS_PER_HOUR + 5):
        ok, message = coll.collect_qsf_sync(f"2026_10_06_file{i}.qsf", _QSF_A)
        assert not ok and "already exists" in message
    assert len(coll._upload_times) == 0 and calls["put"] == []
    ok, message = coll.collect_qsf_sync("2026_10_06_brand_new.qsf", _QSF_B)  # still gets its upload slot
    assert ok, message
    assert calls["put"] == ["2026_10_06_brand_new.qsf"] and len(coll._upload_times) == 1


def test_a_listing_without_content_ids_is_treated_as_duplicates(collector, monkeypatch):
    import types

    coll, _repo, calls = collector
    names = [{"name": f"file{i}.qsf"} for i in range(30)]  # no "sha": the content cannot be compared
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(
        get=lambda *a, **k: _Reply(200, names), put=lambda *a, **k: pytest.fail("a same-name file must not be re-uploaded")))
    for i in range(coll.MAX_UPLOADS_PER_HOUR + 5):
        assert coll.collect_qsf_sync(f"file{i}.qsf", _QSF_A)[0] is False
    assert len(coll._upload_times) == 0 and calls["put"] == []


def test_an_unreadable_listing_skips_the_file_without_spending_budget(collector, monkeypatch):
    import types

    coll, _repo, _calls = collector
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(
        get=lambda *a, **k: _Reply(500), put=lambda *a, **k: pytest.fail("must not upload when the listing is unknown")))
    ok, message = coll.collect_qsf_sync("x.qsf", _QSF_A)
    assert not ok and "already exists" in message and len(coll._upload_times) == 0


def test_same_name_with_different_content_is_collected_under_a_hashed_name(collector):
    coll, repo, calls = collector
    repo["2026_10_06_Survey.qsf"] = _QSF_A
    ok, message = coll.collect_qsf_sync("2026_10_06_Survey.qsf", _QSF_B)  # another student's different survey
    assert ok, message
    (stored,) = calls["put"]
    assert stored.startswith("2026_10_06_Survey_") and stored.endswith(".qsf") and stored != "2026_10_06_Survey.qsf"
    assert repo[stored] == _QSF_B and repo["2026_10_06_Survey.qsf"] == _QSF_A  # nothing was overwritten
    for body in (_QSF_A, _QSF_B):  # now both versions are known: re-uploading either is a duplicate
        ok, message = coll.collect_qsf_sync("2026_10_06_Survey.qsf", body)
        assert not ok and "already exists" in message
    assert len(calls["put"]) == 1


def test_lookups_have_their_own_hourly_bound(collector):
    coll, repo, calls = collector
    total = coll.MAX_LOOKUPS_PER_HOUR + 3
    for i in range(total):
        repo[f"dup{i}.qsf"] = _QSF_A
    messages = [coll.collect_qsf_sync(f"dup{i}.qsf", _QSF_A)[1] for i in range(total)]
    assert all("already exists" in m for m in messages[:-3])
    assert messages[-3:] == ["Hourly upload limit reached"] * 3  # reads are bounded too
    assert calls["get"] == coll.MAX_LOOKUPS_PER_HOUR and calls["put"] == [] and len(coll._upload_times) == 0


def test_async_path_also_checks_for_duplicates_before_spending_budget(collector, monkeypatch):
    coll, repo, calls = collector

    class _Immediate:
        def __init__(self, target, daemon=None):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(coll.threading, "Thread", _Immediate)
    repo["dup.qsf"] = _QSF_A
    for _ in range(coll.MAX_UPLOADS_PER_HOUR + 3):
        coll.collect_qsf_async("dup.qsf", _QSF_A)
    assert len(coll._upload_times) == 0
    coll.collect_qsf_async("fresh.qsf", _QSF_B)
    assert calls["put"] == ["fresh.qsf"] and len(coll._upload_times) == 1


def test_non_ascii_file_names_stay_distinct_and_ascii_names_are_unchanged():
    from utils import github_qsf_collector as coll

    names = ["調査.qsf", "实验.qsf", "我的调查.qsf", "調査1.qsf", "实验1.qsf", "\U0001f600.qsf", "Ωmega.qsf"]
    cleaned = [coll._sanitize_filename(n) for n in names]
    assert len(set(cleaned)) == len(names), cleaned
    assert all(n.isascii() and n.endswith(".qsf") and len(n) <= coll.MAX_FILENAME_LENGTH for n in cleaned)
    assert coll._sanitize_filename("調査.qsf") == coll._sanitize_filename("調査.qsf")  # stable across calls
    assert coll._sanitize_filename("Étude.qsf") == "Etude.qsf"  # accents fold to their letters
    # names that were already fine keep exactly the name they had before
    for before, after in {"survey.QSF": "survey.QSF", "my survey #1 (final)?.qsf": "my survey _1 _final_.qsf",
                          "../../etc/passwd.qsf": "etc_passwd.qsf", "report.qsf.exe": "report.qsf.exe.qsf",
                          "": "survey.qsf", "2026_10_06___.qsf": "2026_10_06_.qsf", "name.qsf.": "name.qsf"}.items():
        assert coll._sanitize_filename(before) == after, before


# ---- 8. a ZIP upload cannot inflate into hundreds of MB ------------------------------------------
def _zip_bytes(members: dict) -> bytes:
    import zipfile as _zf

    buffer = io.BytesIO()
    with _zf.ZipFile(buffer, "w", _zf.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, body in members.items():
            archive.writestr(name, body)
    return buffer.getvalue()


def test_zip_that_inflates_beyond_the_cap_is_refused_before_anything_is_read(monkeypatch):
    import zipfile as _zf

    app = _load_app_module("_app_hygiene_zip")
    bomb = _zip_bytes({"bomb.qsf": b"0" * (30 * 1024 * 1024)})
    assert len(bomb) < 200 * 1024  # a small upload that claims 30 MB

    def _never(*args, **kwargs):
        raise AssertionError("the member must not be opened once its declared size is over the limit")

    monkeypatch.setattr(_zf.ZipFile, "open", _never)
    monkeypatch.setattr(_zf.ZipFile, "read", _never)
    with pytest.raises(ValueError, match=r"would expand to 30 MB.*limit 25 MB"):
        app._extract_qsf_payload(bomb)


def test_a_header_that_understates_the_size_still_cannot_force_a_large_read():
    import struct
    import zipfile as _zf

    app = _load_app_module("_app_hygiene_zip")
    raw = bytearray(_zip_bytes({"bomb.qsf": b"0" * (30 * 1024 * 1024)}))
    central = raw.index(b"PK\x01\x02")
    raw[central + 24:central + 28] = struct.pack("<I", 100)  # the central directory says: 100 bytes
    with pytest.raises((ValueError, _zf.BadZipFile)):
        app._extract_qsf_payload(bytes(raw))


def test_zip_with_a_huge_member_count_is_refused_and_normal_uploads_still_work():
    import zipfile as _zf

    app = _load_app_module("_app_hygiene_zip")
    many = _zip_bytes({f"f{i}.txt": b"x" for i in range(app.MAX_QSF_ZIP_MEMBERS + 1)})
    with pytest.raises(ValueError, match="should contain a single survey file"):
        app._extract_qsf_payload(many)
    survey = _QSF_A
    assert app._extract_qsf_payload(survey) == (survey, "uploaded.qsf")  # a raw .qsf is passed through
    assert app._extract_qsf_payload(_zip_bytes({"folder/Survey.qsf": survey, "readme.txt": b"x"})) == (survey, "folder/Survey.qsf")
    with pytest.raises(ValueError, match="did not contain a .qsf or .json"):
        app._extract_qsf_payload(_zip_bytes({"readme.txt": b"x"}))
    assert issubclass(_zf.BadZipFile, Exception)


def test_uploading_a_zip_bomb_shows_a_clear_error_and_does_not_start_the_design(apptest_env):
    from streamlit.testing.v1 import AppTest

    bomb = _zip_bytes({"bomb.qsf": b"0" * (30 * 1024 * 1024)})
    at = AppTest.from_file(str(_APP_DIR / "app.py"), default_timeout=300)
    for key, value in {"active_page": 1, "study_input_mode": "upload_qsf", "study_title": "T", "study_description": "D study"}.items():
        at.session_state[key] = value
    at.run()
    at.file_uploader[0].set_value(("bomb.zip", bomb, "application/zip"))
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    errors = [e.value for e in at.error if "QSF parsing failed" in e.value]
    assert errors and "would expand to 30 MB" in errors[0], [e.value[:80] for e in at.error]
    assert not at.session_state["qsf_preview"] if "qsf_preview" in at.session_state else True


# ---- 9. inferred factor names do not depend on PYTHONHASHSEED --------------------------------------
_FACTOR_CASES = [
    # the real conditions of a Qualtrics survey whose factor name used to flip between "Politeness" and "Rudeness"
    ["Condition 1\xa0(High strategic silence/ politeness)", "Condition 2 (High strategic silence/ rudeness)",
     "Condition 3 (low strategic silence/ politeness)", "Condition 4 (low strategic silence/ rudeness)"],
    ["Cond1A", "Cond1B", "Cond2A", "Cond2B"],                                    # numeric / suffix route
    ["AI_Hedonic", "AI_Utilitarian", "NoAI_Hedonic", "NoAI_Utilitarian"],         # underscore route
    ["red apple", "green pear"],                                                  # varying words of equal length
    ["wolf lion a", "wolf lion b"],                                               # common words of equal length
    ["No sugar x Sweet tea", "Sugar x Iced tea", "No sugar x Iced tea", "Sugar x Sweet tea"],
    ["Control", "Treatment"],
]


def _factor_names_under_hash_seed(seed: str, workdir: Path) -> dict:
    import os
    import subprocess

    code = (
        "import contextlib, importlib.util, io, json, sys\n"
        f"sys.path.insert(0, {str(_APP_DIR)!r})\n"
        "buf = io.StringIO()\n"
        "with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):\n"
        f"    spec = importlib.util.spec_from_file_location('_app_seed_probe', {str(_APP_DIR / 'app.py')!r})\n"
        "    app = importlib.util.module_from_spec(spec); sys.modules['_app_seed_probe'] = app\n"
        "    try:\n"
        "        spec.loader.exec_module(app)\n"
        "    except SystemExit:\n"
        "        pass\n"
        f"cases = {_FACTOR_CASES!r}\n"
        "out = [[[f['name'], f['levels']] for f in app._infer_factors_from_conditions(list(c))] for c in cases]\n"
        "print('RESULT:' + json.dumps(out))\n"
    )
    env = {**os.environ, "PYTHONHASHSEED": seed, "STREAMLIT_SERVER_HEADLESS": "true"}
    done = subprocess.run([sys.executable, "-c", code], cwd=workdir, env=env, capture_output=True, text=True, timeout=240, check=True)  # noqa: S603
    line = next(line for line in done.stdout.splitlines() if line.startswith("RESULT:"))
    return json.loads(line[len("RESULT:"):])


def test_factor_names_and_levels_are_identical_under_different_hash_seeds(tmp_path):
    first = _factor_names_under_hash_seed("11", tmp_path)
    second = _factor_names_under_hash_seed("12", tmp_path)
    assert first == second
    names = [[factor[0] for factor in case] for case in first]
    assert names[3] == ["Apple"]        # ties on length are broken alphabetically, not by set order
    assert names[4] == ["Lion Wolf"]
    assert names[0][-1] == "( Strategic Silence/ Politeness)"


# ---- 10. one broken OPTIONAL module must not take the app down -------------------------------------
class _FailingImport:
    """A meta-path finder that makes importing `names` fail with `exc` (a half-copied module file)."""

    def __init__(self, names, exc):
        self.names, self.exc = set(names), exc

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.names:
            raise self.exc
        return None


@pytest.mark.parametrize("module, error, check", [
    ("utils.email_delivery", RuntimeError("half-copied file"), lambda app: app._email_delivery is None),
    ("utils.email_delivery", SyntaxError("invalid syntax"), lambda app: app._email_delivery is None),
    ("utils.html_safety", AttributeError("module has no attribute"), lambda app: app._harden_report_html("<p>x</p>") == "<p>x</p>"),
    ("utils.correlation_matrix", ValueError("bad constant"), lambda app: app._HAS_CORRELATION_MODULE is False),
])
def test_a_broken_optional_module_is_logged_and_the_app_still_loads(monkeypatch, tmp_path, module, error, check):
    import logging

    import utils

    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, module, raising=False)
    monkeypatch.delattr(utils, module.rsplit(".", 1)[-1], raising=False)
    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("simulation_app")
    handler = _Collect(level=logging.WARNING)
    logger.addHandler(handler)
    hook = _FailingImport({module}, error)
    sys.meta_path.insert(0, hook)
    try:
        app = _load_app_module("_app_hygiene_optional")
    finally:
        sys.meta_path.remove(hook)
        logger.removeHandler(handler)
    assert check(app)
    assert any(module in message and type(error).__name__ in message for message in records), records


def test_core_modules_are_still_imported_strictly():
    """Only the optional modules are guarded broadly: the three guards must not swallow a missing core module."""
    source = (_APP_DIR / "app.py").read_text(encoding="utf-8")
    head = source[: source.index("APP_TITLE = ")]
    for core in ("from utils.qsf_preview import", "from utils.enhanced_simulation_engine import (\n    EnhancedSimulationEngine,",
                 "from utils.survey_builder import"):
        index = head.index(core)
        assert not head[max(0, index - 200):index].rstrip().endswith("try:"), core


# ---- 11. documentation says what the code does ----------------------------------------------------
_ROOT = _APP_DIR.parent


def _read(relative: str) -> str:
    return (_ROOT / relative).read_text(encoding="utf-8")


def test_docs_give_the_real_template_engine_limit_instead_of_unlimited():
    import re

    max_n = int(re.search(r"^MAX_SIMULATED_N\s*=\s*(\d+)", _read("simulation_app/app.py"), re.MULTILINE).group(1))
    assert max_n == 10000
    for relative in ("docs/guide/limitations.md", "README.md"):
        text = _read(relative)
        assert "unlimited" not in text.lower(), relative
        assert f"{max_n:,}" in text, relative


def test_limitations_page_does_not_send_users_to_a_reverse_item_control_that_does_not_exist():
    import ast

    assert "mark reverse-keyed items on the Design page" not in _read("docs/guide/limitations.md")
    widgets = {"checkbox", "multiselect", "text_input", "text_area", "number_input", "selectbox", "radio", "toggle", "data_editor"}
    labelled = []
    for node in ast.walk(ast.parse(_read("simulation_app/app.py"))):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") in widgets and node.args:
            first = node.args[0]
            parts = [first] if isinstance(first, ast.Constant) else list(ast.walk(first))
            text = " ".join(str(p.value) for p in parts if isinstance(p, ast.Constant) and isinstance(p.value, str))
            if "reverse" in text.lower():
                labelled.append((node.lineno, text[:60]))
    assert not labelled, f"the app has a reverse-item widget now, update the limitations page: {labelled}"


def test_technical_methods_page_carries_the_real_licence():
    text = _read("docs/internal/technical_methods.md")
    assert "proprietary" not in text.lower() and "all rights reserved" not in text.lower()
    assert text.count("PolyForm Noncommercial License 1.0.0") == 2
    assert "PolyForm Noncommercial License 1.0.0" in _read("LICENSE")


def test_exported_scripts_are_described_as_data_preparation_because_that_is_all_they_do():
    from utils.enhanced_simulation_engine import EnhancedSimulationEngine

    assert "ready-to-run analysis scripts" not in _read("simulation_app/README.md")
    source = _read("simulation_app/app.py")
    for stale in ("analysis scripts in 5", "metadata + analysis scripts)", "metadata, analysis scripts)", "(data files, '\n            'analysis scripts"):
        assert stale not in source, stale
    engine = EnhancedSimulationEngine(
        study_title="Scripts", study_description="A pilot study of satisfaction", sample_size=40, conditions=["Alpha", "Bravo"],
        factors=[], scales=[_scale()], additional_vars=[], demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12},
        open_ended_questions=[], effect_sizes=[], seed=3)
    engine.llm_generator.disable_permanently("test")
    df, _ = engine.generate()
    analysis_calls = ("t.test(", "aov(", "anova(", "lm(", "glm(", "ttest", "f_oneway", "ols(", "mixedlm", "regress ", "ttest ", "oneway ",
                      "T-TEST", "ONEWAY", "UNIANOVA", "MIXED ", "pingouin", "scipy.stats", "statsmodels", "HypothesisTests")
    for name in ("r", "python", "julia", "spss", "stata"):
        script = getattr(engine, f"generate_{name}_export")(df)
        assert "Data Preparation" in script or "prepar" in script.lower(), name
        assert not [call for call in analysis_calls if call in script], (name, [c for c in analysis_calls if c in script])


def test_download_button_says_data_preparation_scripts(apptest_env):
    at = _generate_page(["Alpha", "Bravo"], n=40, advanced=False)
    _click_generate(at)
    labels = [b.label for b in at.get("download_button")]
    assert "Download ZIP (CSV + metadata + data-preparation scripts)" in labels, labels


def test_changelog_numeric_text_box_count_matches_the_corpus():
    from utils.enhanced_simulation_engine import clean_question_text, infer_numeric_answer_spec
    from utils.qsf_preview import QSFPreviewParser

    # The figures in the changelog were counted on the 302 files that existed when they were measured;
    # demo files added to the folder later are not part of that corpus.
    added_later = {"2026_10_06_Demo_study_design.qsf"}
    files = sorted(path for path in (_APP_DIR / "example_files").glob("*.qsf") if path.name not in added_later)
    if len(files) != 302:
        pytest.skip("the changelog counts the 302-file corpus; the example folder has changed")
    total = numeric = 0
    for path in files:
        for question in QSFPreviewParser().parse(path.read_bytes()).open_ended_details:
            total += 1
            text = clean_question_text(question.get("question_text") or "")
            if infer_numeric_answer_spec(text, question["variable_name"], dict(question)) is not None:
                numeric += 1
    assert f"Numeric text boxes ({numeric:,} of the {total:,} open-ended questions in the 302-file corpus)" in _read("docs/CHANGELOG.md")
