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
