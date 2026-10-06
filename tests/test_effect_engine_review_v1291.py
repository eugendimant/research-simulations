"""Engine review fixes (v1.2.9.1): the effect you request is the effect you get.

Each test pins one reproduced failure:

* labels that start or end with punctuation ("Norm message (Control)", "80%", "$10") silently lost
  the requested effect because the word-boundary test needed a word character beside the label;
* a game word in a condition LABEL ("Dictator game") moved the baseline by ~1.9 d even with inferred
  effects off or an explicit d;
* an effect that matches no variable or condition was dropped without a word;
* a spec for "Trust" also moved "Distrust"; a blank variable moved every DV;
* snake_case labels ("High_Threat") lost every name-based modifier;
* the main effects of a factorial design were averaged instead of added.

Generation runs are small (N <= 1,000) and fully seeded, so every number below is reproducible; the
tolerances are several sampling standard errors wide (one SE of d is about 0.1 at N = 400).
"""

import json
import math

import numpy as np
import pytest

from utils.enhanced_simulation_engine import (
    EffectSizeSpec,
    EnhancedSimulationEngine,
    _kw_hit,
    _stem_in,
    _word_in,
)

DEMO = {"gender_quota": 50, "age_mean": 35, "age_sd": 12}


def _scale(name="Trust", items=4, lo=1, hi=7, reverse=None):
    return {"name": name, "variable_name": name, "type": "likert", "num_items": items,
            "scale_points": hi - lo + 1, "scale_min": lo, "scale_max": hi, "reverse_items": list(reverse or [])}


def _engine(conditions, scales, specs=(), n=60, seed=1, auto=True, desc="A decision-making study", **kw):
    e = EnhancedSimulationEngine(
        study_title="Review", study_description=desc, sample_size=n, conditions=list(conditions), factors=kw.pop("factors", []),
        scales=scales, additional_vars=[], demographics=dict(DEMO), open_ended_questions=[],
        effect_sizes=list(specs), seed=seed, auto_effects=auto, **kw)
    e.llm_generator.disable_permanently("test")
    return e


def _spec(variable, high, low, d, factor="CONDITION", direction="positive"):
    return EffectSizeSpec(variable=variable, factor=factor, level_high=high, level_low=low, cohens_d=d, direction=direction)


def _d(df, hi, lo, col):
    a = df.loc[df.CONDITION == hi, col].astype(float)
    b = df.loc[df.CONDITION == lo, col].astype(float)
    return float((a.mean() - b.mean()) / math.sqrt((a.var() + b.var()) / 2))


def _mean_d(conds, specs, seeds, n=400, auto=True, col="Trust_mean", desc="A decision-making study", scales=None):
    ds = []
    for seed in seeds:
        e = _engine(conds, scales or [_scale()], specs, n=n, seed=seed, auto=auto, desc=desc)
        df, _ = e.generate()
        ds.append(_d(df, conds[0], conds[1], col))
    return float(np.mean(ds))


# ----------------------------------------------------------------------------------------------
# P1-1: labels with punctuation at either end
# ----------------------------------------------------------------------------------------------
def test_word_boundaries_accept_punctuation_at_the_label_edges():
    assert _word_in("norm message (control)", "norm message (control)")
    assert _word_in("80%", "choose 80% of them") and _word_in("$10", "$10")
    assert _word_in("treatment (high)", "the treatment (high) group")
    assert _stem_in("$10", "pay $10 now") and _stem_in("(high", "x (high)")
    # still whole-word tests
    assert not _word_in("ai", "wait") and not _word_in("low", "follow-up")
    assert not _word_in("$10", "$100") and not _word_in("80%", "180%")
    assert not _word_in("gain", "bargain")


@pytest.mark.parametrize("high, low", [
    ("Norm message with reference (Empirical)", "Norm message (Control)"),
    ("80%", "20%"),
    ("$10", "$0"),
    ("Treatment (high)", "Control (low)"),
])
def test_effect_on_labels_with_punctuation_is_recovered(high, low):
    """Requested d = 0.8; before the fix these labels gave 0.02-0.12."""
    d = _mean_d([high, low], [_spec("Trust", high, low, 0.8)], seeds=(11, 12))
    assert d > 0.8 * 0.6, f"{high!r} vs {low!r}: observed d = {d:.2f}"
    e = _engine([high, low], [_scale()], [_spec("Trust", high, low, 0.8)], n=200, seed=3)
    _, meta = e.generate()
    row = meta["effect_sizes_applied"]["contrasts"][0]
    assert row["source"] == "user" and row["intended_d"] == pytest.approx(0.8, abs=0.01)
    assert meta["effect_sizes_applied"]["specs"][0]["matched"] is True


def test_explicit_condition_detection_handles_punctuation_labels():
    e = _engine(["Treatment (high)", "Control (low)", "Placebo"], [_scale()],
                [_spec("Trust", "Treatment (high)", "Control (low)", 0.5)])
    assert e._is_explicit_condition("Treatment (high)") and e._is_explicit_condition("Control (low)")
    assert not e._is_explicit_condition("Placebo")


# ----------------------------------------------------------------------------------------------
# P1-2: game words in labels
# ----------------------------------------------------------------------------------------------
GAME_DESC = "A study of cooperation"
GAME_CONDS = ["Dictator game", "Trust game"]
GAME_SCALE = [_scale("Cooperation_Score")]


def test_game_word_labels_are_a_true_null_when_inference_is_off():
    d = _mean_d(GAME_CONDS, [], seeds=(8001, 8002), auto=False, col="Cooperation_Score_mean", desc=GAME_DESC, scales=GAME_SCALE)
    assert abs(d) < 0.3, f"inference off, no effect: d = {d:+.2f} (was -1.88)"


def test_explicit_zero_and_explicit_effect_are_not_contaminated_by_game_labels():
    spec0 = [_spec("Cooperation_Score", *GAME_CONDS, 0.0)]
    d0 = _mean_d(GAME_CONDS, spec0, seeds=(8001, 8002), col="Cooperation_Score_mean", desc=GAME_DESC, scales=GAME_SCALE)
    assert abs(d0) < 0.3, f"explicit d=0: d = {d0:+.2f} (was -1.88)"
    spec5 = [_spec("Cooperation_Score", *GAME_CONDS, 0.5)]
    d5 = _mean_d(GAME_CONDS, spec5, seeds=(8001, 8002), col="Cooperation_Score_mean", desc=GAME_DESC, scales=GAME_SCALE)
    assert 0.3 < d5 < 0.9, f"explicit d=0.5: d = {d5:+.2f} (was -1.38)"


def test_label_driven_game_baselines_remain_an_inferred_effect():
    """With inferred effects ON and no user effect the label still picks the game baseline."""
    d = _mean_d(GAME_CONDS, [], seeds=(8001, 8002), auto=True, col="Cooperation_Score_mean", desc=GAME_DESC, scales=GAME_SCALE)
    assert d < -1.0, f"inferred game baselines are expected to separate the games, got d = {d:+.2f}"


# ----------------------------------------------------------------------------------------------
# P1-1: an effect that reaches nothing must say so
# ----------------------------------------------------------------------------------------------
def _diag(conds, specs, scales=None):
    e = _engine(conds, scales or [_scale()], specs, n=60)
    df, meta = e.generate()
    return meta, df


def test_spec_matching_no_condition_is_reported():
    meta, _ = _diag(["Group 1", "Group 2"], [_spec("Trust", "Hot", "Cold", 0.5)])
    row = meta["effect_sizes_applied"]["specs"][0]
    assert row["matched"] is False and row["status"] == "levels_not_found"
    assert any("NOT applied" in w and "Trust" in w for w in meta["generation_warnings"])


def test_spec_for_an_unknown_variable_is_reported():
    meta, _ = _diag(["Group 1", "Group 2"], [_spec("Loyalty", "Group 1", "Group 2", 0.5)])
    row = meta["effect_sizes_applied"]["specs"][0]
    assert row["matched"] is False and row["status"] == "variable_not_found"
    assert any("no generated variable" in w for w in meta["generation_warnings"])


def test_spec_matching_one_level_only_warns_about_half_the_effect():
    meta, _ = _diag(["Group 1", "Group 2"], [_spec("Trust", "Group 1", "Typo level", 0.5)])
    row = meta["effect_sizes_applied"]["specs"][0]
    assert row["matched"] is True and row["status"] == "one_side_only"
    assert any("one side only" in w for w in meta["generation_warnings"])


def test_matched_spec_adds_no_warning_and_is_json_serialisable():
    meta, _ = _diag(["Group 1", "Group 2"], [_spec("Trust", "Group 1", "Group 2", 0.5)])
    assert meta["effect_sizes_applied"]["specs"][0]["status"] == "applied"
    assert not [w for w in meta["generation_warnings"] if "NOT applied" in w or "one side" in w]
    json.dumps(meta["effect_sizes_applied"], allow_nan=False)


# ----------------------------------------------------------------------------------------------
# P2: variable matching on whole words, blank variable, snake_case labels
# ----------------------------------------------------------------------------------------------
def test_spec_for_trust_does_not_leak_onto_distrust():
    conds = ["Group 1", "Group 2"]
    e = _engine(conds, [_scale("Trust"), _scale("Distrust")], [_spec("Trust", *conds, 0.5)], auto=False)
    assert e._get_effect_for_condition("Group 1", "Trust") > 0
    assert e._get_effect_for_condition("Group 1", "Distrust") == 0.0
    assert e._applied_effects[("Group 1", "Distrust")]["source"] == "none"
    assert not e._variable_has_user_spec("Distrust")


def test_variable_names_match_as_whole_words():
    e = _engine(["A", "B"], [_scale("DV1"), _scale("DV10"), _scale("Perceived_Quality"), _scale("Trust in AI")],
                [_spec("DV1", "A", "B", 0.5), _spec("Perceived Quality", "A", "B", 0.5)], auto=False)
    assert e._variable_has_user_spec("DV1") and not e._variable_has_user_spec("DV10")
    assert e._variable_has_user_spec("Perceived_Quality")
    assert not e._variable_has_user_spec("Trust_in_AI")


def test_exact_variable_name_takes_priority_over_a_word_run():
    e = _engine(["A", "B"], [_scale("Trust"), _scale("Trust in AI")], [_spec("Trust", "A", "B", 0.5)], auto=False)
    assert e._variable_has_user_spec("Trust") and not e._variable_has_user_spec("Trust_in_AI")
    e2 = _engine(["A", "B"], [_scale("Trust in AI"), _scale("Satisfaction")], [_spec("Trust", "A", "B", 0.5)], auto=False)
    assert e2._variable_has_user_spec("Trust_in_AI") and not e2._variable_has_user_spec("Satisfaction")


def test_blank_variable_applies_to_nothing_and_is_reported():
    e = _engine(["A", "B"], [_scale("Trust"), _scale("Satisfaction")], [_spec("", "A", "B", 0.5)], auto=False)
    assert not e._variable_has_user_spec("Trust") and not e._variable_has_user_spec("Satisfaction")
    assert e._get_effect_for_condition("A", "Trust") == 0.0
    assert e._effect_spec_diagnostics()["specs"][0]["status"] == "variable_not_found"


def test_non_ascii_display_name_still_finds_its_column():
    e = _engine(["A", "B"], [_scale("Vertrauen für Ärzte")], [_spec("Vertrauen für Ärzte", "A", "B", 0.5)], auto=False)
    col = list(e._variable_alias_map())[0]
    assert e._variable_has_user_spec(col) and e._get_effect_for_condition("A", col) > 0


def test_kw_hit_treats_underscore_as_a_word_separator():
    assert _kw_hit("high", "high_threat") and _kw_hit("no ai", "no_ai") and _kw_hit("loss frame", "loss_frame")
    assert not _kw_hit("ai", "wait") and not _kw_hit("low", "follow-up")


@pytest.mark.parametrize("snake, spaced", [("High_Threat", "High Threat"), ("No_AI", "No AI"), ("Loss_Frame", "Loss Frame")])
def test_snake_case_labels_get_the_same_name_based_modifiers(snake, spaced):
    e = _engine(["a", "b"], [_scale()])
    assert e._get_condition_trait_modifier(snake) == e._get_condition_trait_modifier(spaced)
    assert abs(e._get_automatic_condition_effect(snake, "Trust") - e._get_automatic_condition_effect(spaced, "Trust")) < 0.06


# ----------------------------------------------------------------------------------------------
# Factorial main effects add; effects on one factor average
# ----------------------------------------------------------------------------------------------
CELLS = ["A1 B1", "A1 B2", "A2 B1", "A2 B2"]


def test_main_effects_of_different_factors_add():
    specs = [_spec("DV", "A1", "A2", 0.5, factor="A"), _spec("DV", "B1", "B2", 0.5, factor="B")]
    e = _engine(CELLS, [_scale("DV")], specs, auto=False)
    one = _engine(CELLS, [_scale("DV")], specs[:1], auto=False)
    unit = one._get_effect_for_condition("A1 B1", "DV")          # +d/2 on the high side of one spec
    assert e._get_effect_for_condition("A1 B1", "DV") == pytest.approx(2 * unit)
    assert e._get_effect_for_condition("A1 B2", "DV") == pytest.approx(0.0, abs=1e-12)
    assert e._get_effect_for_condition("A2 B1", "DV") == pytest.approx(0.0, abs=1e-12)
    assert e._get_effect_for_condition("A2 B2", "DV") == pytest.approx(-2 * unit)


def test_effects_that_share_a_control_are_averaged_not_stacked():
    conds = ["Treatment A", "Treatment B", "Control"]
    specs = [_spec("DV", "Treatment A", "Control", 0.5), _spec("DV", "Treatment B", "Control", 0.5)]
    e = _engine(conds, [_scale("DV")], specs, auto=False)
    one = _engine(conds, [_scale("DV")], specs[:1], auto=False)
    assert e._get_effect_for_condition("Control", "DV") == pytest.approx(one._get_effect_for_condition("Control", "DV"))
    assert e._get_effect_for_condition("Treatment A", "DV") == pytest.approx(one._get_effect_for_condition("Treatment A", "DV"))


def test_duplicate_specs_do_not_double_the_effect():
    specs = [_spec("DV", "A1", "A2", 0.5, factor="A"), _spec("DV", "A1", "A2", 0.5, factor="Condition")]
    e = _engine(CELLS, [_scale("DV")], specs, auto=False)
    one = _engine(CELLS, [_scale("DV")], specs[:1], auto=False)
    assert e._get_effect_for_condition("A1 B1", "DV") == pytest.approx(one._get_effect_for_condition("A1 B1", "DV"))


def test_factorial_marginal_d_matches_the_request_when_both_effects_are_given():
    """Marginal d of both factors, requested 0.5 each (the averaged version gave 0.25-0.28)."""
    ds_a, ds_b = [], []
    for seed in (6001, 6002):
        specs = [_spec("DV", "A1", "A2", 0.5, factor="A"), _spec("DV", "B1", "B2", 0.5, factor="B")]
        e = _engine(CELLS, [_scale("DV")], specs, n=1000, seed=seed, factors=[{"name": "A", "levels": ["A1", "A2"]}, {"name": "B", "levels": ["B1", "B2"]}])
        df, _ = e.generate()
        m = df["DV_mean"].astype(float)
        for mask, out in ((df.CONDITION.str.startswith("A1"), ds_a), (df.CONDITION.str.endswith("B1"), ds_b)):
            x, y = m[mask], m[~mask]
            sp = math.sqrt(((len(x) - 1) * x.var() + (len(y) - 1) * y.var()) / (len(x) + len(y) - 2))
            out.append((x.mean() - y.mean()) / sp)
    assert np.mean(ds_a) > 0.36 and np.mean(ds_b) > 0.36, (np.mean(ds_a), np.mean(ds_b))


# ----------------------------------------------------------------------------------------------
# P2: question-text cleaning must not delete prose; Metadata.json must be valid JSON
# ----------------------------------------------------------------------------------------------
def test_clean_question_text_keeps_prose_around_abbreviations_and_piped_text():
    from utils.enhanced_simulation_engine import _clean_question_text as clean

    assert clean("The U.S. government gave ${e://Field/amount} to you") == "The U.S. government gave ${e://Field/amount} to you"
    assert clean("Consider Dr. Smith. He said #1 is best {sic}. OK") == "Consider Dr. Smith. He said #1 is best {sic}. OK"
    assert clean("Visit www.example.com. {x} then answer.") == "Visit www.example.com. {x} then answer."


def test_clean_question_text_still_removes_pasted_css_and_script_elements():
    from utils.enhanced_simulation_engine import _clean_question_text as clean

    assert clean("#QID154-7-label {display: inline-block; width: 5%;} How satisfied are you?") == "How satisfied are you?"
    assert clean(".highlight, .other > .x {font-weight: bold} Q") == "Q"
    assert clean("Rate the product <style>.x{color:red}</style> below <script>var a=1;</script>please") == "Rate the product below please"
    assert clean('<style type="text/css">#QID5 .Choice {margin: 0}</style>How much do you trust the U.S. Navy?') == "How much do you trust the U.S. Navy?"


def test_metadata_is_valid_json_when_missing_data_is_enabled():
    scales = [_scale("Trust"), _scale("Sat", items=1)]
    e = _engine(["A", "B"], scales, [_spec("Trust", "A", "B", 0.5)], n=120, missing_data_rate=0.1, dropout_rate=0.07)
    _, meta = e.generate()
    text = json.dumps(meta, allow_nan=False, default=str)      # raises on a bare NaN / Infinity
    assert "NaN" not in text
    assert all(np.isfinite(r.get("observed_mean", 0.0)) for r in meta["scale_verification"])


def test_json_writer_never_emits_nan_or_infinity():
    from utils.simulation_run_audit import _safe_json

    text = _safe_json({"a": float("nan"), "b": [1.0, float("inf")], "c": np.float64("nan"), "d": {"e": -float("inf")}})
    assert json.loads(text, parse_constant=lambda c: pytest.fail(f"bare {c} written")) == {
        "a": None, "b": [1.0, None], "c": None, "d": {"e": None}}
