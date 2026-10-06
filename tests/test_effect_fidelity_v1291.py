"""Effect fidelity (v1.2.9.1): the d you ask for is the d you get (within sampling error).

Complements tests/test_effect_size_recovery.py with longer scales, the true-null option, game DVs,
binary items and the effects-applied metadata.

Each case simulates one scale with an explicit effect and measures the Cohen's d on the scale
mean. N is 1,200, so one standard error of d is about 0.06. Tolerances are set at roughly
four standard errors plus the calibration's own fit error, so a pass means the effect is
within about 25% of the request.
"""

import numpy as np
import pytest

from utils.enhanced_simulation_engine import EnhancedSimulationEngine, EffectSizeSpec

HI, LO = "Group 1", "Group 2"


def _simulate(points_min, points_max, items, d, n=1200, seed=11, conditions=None, auto_effects=True,
              name="DV", reverse=None, desc="A study of participants"):
    conds = conditions or [HI, LO]
    scale = {"name": name, "variable_name": name, "type": "slider" if points_max - points_min >= 20 else "likert",
             "num_items": items, "scale_points": points_max - points_min + 1, "scale_min": points_min,
             "scale_max": points_max, "reverse_items": reverse or []}
    spec = [EffectSizeSpec(variable=name, factor="CONDITION", level_high=conds[0], level_low=conds[1],
                           cohens_d=d, direction="positive")] if d else []
    e = EnhancedSimulationEngine(
        study_title="Recovery", study_description=desc, sample_size=n, conditions=conds, factors=[],
        scales=[scale], additional_vars=[], demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12},
        open_ended_questions=[], effect_sizes=spec, seed=seed, auto_effects=auto_effects)
    e.llm_generator.disable_permanently("test")
    df, meta = e.generate()
    return df, meta, conds


def _scale_d(df, conds, name="DV"):
    col = f"{name}_mean" if f"{name}_mean" in df.columns else f"{name}_1"
    a = df.loc[df.CONDITION == conds[0], col].astype(float)
    b = df.loc[df.CONDITION == conds[1], col].astype(float)
    pooled = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / pooled)


@pytest.mark.parametrize("lo, hi, items", [
    (1, 5, 1),      # single Likert item
    (1, 7, 4),      # typical multi-item Likert scale
    (0, 10, 8),     # longer 0-10 scale
    (0, 100, 1),    # slider
])
def test_requested_d_is_recovered_on_the_scale_mean(lo, hi, items):
    df, _, conds = _simulate(lo, hi, items, d=0.8)
    observed = _scale_d(df, conds)
    assert 0.8 * 0.72 <= observed <= 0.8 * 1.28, f"requested 0.8, observed {observed:.2f} ({lo}-{hi}, {items} items)"


def test_zero_effect_without_inference_is_a_true_null():
    """With inferred effects off, condition NAMES must not create differences."""
    df, meta, conds = _simulate(1, 7, 4, d=0, conditions=["High anxiety", "Low anxiety"], auto_effects=False,
                                desc="An anxiety study with high and low conditions")
    assert abs(_scale_d(df, conds)) < 0.25
    applied = meta["effect_sizes_applied"]
    assert applied["inferred_effects_enabled"] is False
    assert all(r["source"] == "none" and r["intended_d"] == 0 for r in applied["contrasts"])


def test_explicit_effect_is_not_contaminated_by_condition_names():
    """"High"/"Low" in the labels used to add name-based shifts on top of the requested effect."""
    df, _, conds = _simulate(1, 7, 4, d=0.8, conditions=["High threat", "Low threat"])
    observed = _scale_d(df, conds)
    assert 0.8 * 0.72 <= observed <= 0.8 * 1.28, f"observed {observed:.2f}"


def test_effects_applied_metadata_distinguishes_user_and_inferred():
    df, meta, conds = _simulate(1, 7, 4, d=0.5, n=300)
    row = meta["effect_sizes_applied"]["contrasts"][0]
    assert row["source"] == "user"
    assert row["intended_d"] == pytest.approx(0.5, abs=0.01)
    assert row["observed_d"] is not None
    assert meta["effect_sizes_configured"][0]["level_high"] == HI
    assert meta["effect_sizes_configured"][0]["level_low"] == LO


def test_same_seed_gives_identical_data_and_different_seed_does_not():
    a, _, _ = _simulate(1, 7, 3, d=0.5, n=150, seed=5)
    b, _, _ = _simulate(1, 7, 3, d=0.5, n=150, seed=5)
    c, _, _ = _simulate(1, 7, 3, d=0.5, n=150, seed=6)
    cols = [x for x in a.columns if x.startswith("DV_")]
    assert a[cols].equals(b[cols])
    assert not a[cols].equals(c[cols])


def test_scale_mean_is_reverse_scored():
    df, _, _ = _simulate(1, 7, 4, d=0.5, n=300, reverse=[2, 4])
    items = df[["DV_1", "DV_2", "DV_3", "DV_4"]].astype(float)
    expected = (items["DV_1"] + (8 - items["DV_2"]) + items["DV_3"] + (8 - items["DV_4"])) / 4
    assert np.allclose(df["DV_mean"].astype(float), expected.round(2), atol=0.011)


def test_binary_items_keep_the_requested_effect():
    """Straight-line repair used to randomise binary 3-item scales and erase the effect."""
    df, _, conds = _simulate(0, 1, 3, d=0.8, n=1500)
    observed = _scale_d(df, conds)
    assert observed >= 0.8 * 0.6, f"requested 0.8, observed {observed:.2f} on a binary 3-item scale"


@pytest.mark.parametrize("lo, hi, items", [(0, 100, 1), (1, 7, 3)])
def test_game_dv_keeps_the_requested_effect(lo, hi, items):
    """The behavioral-economics game model rewrites game DVs; the requested effect must survive."""
    name = "Dictator_Giving"
    conds = ["Charity", "Individual"]
    scale = {"name": name, "variable_name": name, "type": "slider" if hi - lo >= 20 else "likert",
             "num_items": items, "scale_points": hi - lo + 1, "scale_min": lo, "scale_max": hi,
             "reverse_items": []}
    spec = [EffectSizeSpec(variable=name, factor="CONDITION", level_high=conds[0], level_low=conds[1],
                           cohens_d=0.5, direction="positive")]
    e = EnhancedSimulationEngine(
        study_title="Dictator game giving", study_description="Participants play a dictator game and decide how much to give.",
        sample_size=1000, conditions=conds, factors=[], scales=[scale], additional_vars=[],
        demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12}, open_ended_questions=[],
        effect_sizes=spec, seed=11, use_socsim_experimental=True)
    e.llm_generator.disable_permanently("test")
    df, meta = e.generate()
    assert (meta.get("socsim") or {}).get("enriched_dvs"), "game model did not run: the test is not testing anything"
    observed = _scale_d(df, conds, name)
    assert 0.5 * 0.7 <= observed <= 0.5 * 1.3, f"requested 0.5, observed {observed:.2f}"
    assert df[[c for c in df.columns if c.startswith(name + "_") and c[len(name) + 1:].isdigit()]].min().min() >= lo
