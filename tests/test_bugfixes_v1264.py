#!/usr/bin/env python3
"""Regression tests for v1.2.6.4 bug fixes.

Each test pins a specific bug that was found and fixed so it cannot silently
regress:

1. Political identity: a same-valence condition ("supporter and fan") is INGROUP
   (positive effect), not OUTGROUP. Only opposite valences (lover AND hater) are
   outgroup. (enhanced_simulation_engine._get_automatic_condition_effect)
2. QSF bipolar scale-point detection includes NEGATIVE recode values
   (-3..+3 => 7 points, not 4). (qsf_preview._detect_scale_points)
3. hbs_validator tolerates NaN on the dict-of-lists code path without crashing
   in int()-based scale detection. (hbs_validator._find_scale_columns)
4. svg_charts renders valid SVG (no literal "nan") when a condition mean is NaN.
   (svg_charts.create_bar_chart_svg)
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation_app"))


# ── Fix 1: political same-valence is ingroup, opposite-valence is outgroup ──────
def _effect(cond, var="dollars_allocated"):
    from utils.enhanced_simulation_engine import EnhancedSimulationEngine
    eng = EnhancedSimulationEngine(
        study_title="Political Dictator Game",
        study_description="Partisan identity and economic allocation in a dictator game.",
        sample_size=30, conditions=[{"name": "c1"}, {"name": "c2"}], factors=[],
        scales=[{"name": "dollars_allocated", "items": 1, "min": 0, "max": 100}],
        additional_vars=[], demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12},
        seed=1,
    )
    return eng._get_automatic_condition_effect(cond, var)


def test_political_same_valence_positive_words_are_ingroup():
    # Two positive words → ingroup (positive), NOT outgroup. This was the bug.
    assert _effect("Trump supporter and fan") > 0
    assert _effect("Trump lover") > 0


def test_political_negative_attitude_is_outgroup():
    assert _effect("Trump hater") < 0


def test_political_mixed_valence_is_outgroup():
    # Opposite valences in the same condition → outgroup pairing (negative).
    assert _effect("Trump lover vs Biden hater") < 0


# ── Fix 2: bipolar recode values (negatives) counted correctly ──────────────────
def test_qsf_bipolar_recode_scale_points():
    from utils.qsf_preview import QSFPreviewParser
    p = QSFPreviewParser()
    payload = {"RecodeValues": {"1": "-3", "2": "-2", "3": "-1", "4": "0",
                                "5": "1", "6": "2", "7": "3"}}
    pts = p._detect_scale_points(payload, "MC", "SAVR", [], [])
    assert pts == 7, f"expected 7 scale points for -3..+3, got {pts}"


def test_qsf_positive_recode_still_works():
    # No regression for ordinary 1..7 recodes.
    from utils.qsf_preview import QSFPreviewParser
    p = QSFPreviewParser()
    payload = {"RecodeValues": {str(i): str(i) for i in range(1, 8)}}
    pts = p._detect_scale_points(payload, "MC", "SAVR", [], [])
    assert pts == 7, f"expected 7 scale points for 1..7, got {pts}"


# ── Fix 3: hbs_validator NaN robustness on dict-of-lists path ────────────────────
def test_hbs_validator_handles_nan_dict_path():
    from utils.hbs_validator import HBSValidator
    v = HBSValidator()
    # dict-of-lists with a genuine float NaN in a scale-like column.
    data = {
        "Trust_1": [1.0, 2.0, 3.0, 4.0, float("nan"), 6.0, 7.0, 2.0, 3.0, 4.0],
        "Trust_2": [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 5.0, 5.0, 5.0],
        "CONDITION": ["A"] * 5 + ["B"] * 5,
    }
    # Must not raise (previously int(nan) raised ValueError here).
    cols = v._find_scale_columns(data)
    assert isinstance(cols, list)


# ── Fix (v1.2.6.9): Likert matrix in a generic CONTENT block is detected ─────────
def test_matrix_dv_in_questionnaire_block_detected():
    """A Likert matrix in a block named 'Questionnaire'/'Survey'/'Main' must be
    detected as a DV scale — those names are excluded from CONDITION detection but
    routinely hold the real DVs (regression: Coffee_Shop loyalty matrix was dropped)."""
    import os
    from utils.qsf_preview import QSFPreviewParser
    qsf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "simulation_app", "example_files",
                       "Coffee_Shop_Loyalty_Programs.qsf")
    if not os.path.exists(qsf):
        return  # corpus file not present in this checkout
    res = QSFPreviewParser().parse(open(qsf, "rb").read())
    matrices = [s for s in res.detected_scales if s.get("type") == "matrix"]
    assert matrices, f"expected a matrix DV, got {[s.get('type') for s in res.detected_scales]}"
    loyalty = next((s for s in matrices if "loyalty" in str(s.get("variable_name", "")).lower()), None)
    assert loyalty is not None, "LoyaltyQuestions matrix not detected"
    assert loyalty.get("items") == 5 and loyalty.get("scale_points") == 7


def _sim_typed_dv(dv_type, num_items, lo, hi):
    from utils.enhanced_simulation_engine import EnhancedSimulationEngine
    eng = EnhancedSimulationEngine(
        study_title="Typed DV", study_description="typed dv study", sample_size=25,
        conditions=[{"name": "A"}, {"name": "B"}], factors=[],
        scales=[{"name": "DV", "variable_name": "DV", "type": dv_type,
                 "num_items": num_items, "items": num_items,
                 "scale_points": hi - lo + 1, "scale_min": lo, "scale_max": hi}],
        additional_vars=[], demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12},
        seed=3)
    if getattr(eng, "llm_generator", None) is not None:
        eng.llm_generator.disable_permanently("test")
    df, _ = eng.generate()
    return df, [c for c in df.columns if c.startswith("DV_") and not c.endswith("_mean")]


def test_constant_sum_rows_sum_to_total():
    """Constant-sum DV: every participant's allocations must sum EXACTLY to the
    total (was 0% before v1.2.7.0 — independent integers)."""
    df, cols = _sim_typed_dv("constant_sum", 3, 0, 100)
    assert len(cols) == 3
    sums = df[cols].sum(axis=1)
    assert (sums == 100).all(), f"rows must sum to 100; got {sorted(set(sums))[:5]}"
    assert "DV_mean" not in df.columns  # meaningless composite suppressed


def test_rank_order_rows_are_valid_permutations():
    """Rank-order DV: each row must be a permutation of 1..k (was 0% valid before
    v1.2.7.0 — duplicate ranks)."""
    df, cols = _sim_typed_dv("rank_order", 4, 1, 4)
    assert len(cols) == 4
    for _, row in df[cols].iterrows():
        assert sorted(int(x) for x in row) == [1, 2, 3, 4], f"not a permutation: {list(row)}"
    assert "DV_mean" not in df.columns


def test_likert_dv_unaffected_by_typed_dispatch():
    """A normal Likert DV must be unchanged (composite mean present, values in range)."""
    df, cols = _sim_typed_dv("likert", 4, 1, 7)
    assert "DV_mean" in df.columns
    for c in cols:
        assert df[c].between(1, 7).all()


def test_attention_check_instruction_not_detected_as_dv():
    """Attention-check instruction items ("Please select 'Agree' for this question")
    must NOT be detected as DVs even though they use real Likert anchors."""
    import os
    from utils.qsf_preview import QSFPreviewParser
    qsf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "simulation_app", "example_files", "Group #13 - Survey.qsf")
    if not os.path.exists(qsf):
        return
    res = QSFPreviewParser().parse(open(qsf, "rb").read())
    names = [str(s.get("variable_name", "")).lower() for s in res.detected_scales]
    assert "ac2" not in names, f"attention check ac2 wrongly detected as DV: {names}"
    # and no detected DV is an attention-check instruction
    for s in res.detected_scales:
        t = str(s.get("question_text", "")).lower()
        assert "for this question" not in t or "select" not in t, f"attn-check leaked: {t}"


# ── Fix 4: svg chart NaN robustness ─────────────────────────────────────────────
def test_svg_bar_chart_nan_robust():
    from utils.svg_charts import create_bar_chart_svg
    data = {"Control": (3.5, 0.2), "Treatment": (float("nan"), float("nan"))}
    svg = create_bar_chart_svg(data, title="t", ylabel="y")
    assert svg.startswith("<svg") or "<svg" in svg
    assert "nan" not in svg.lower(), "SVG must not contain NaN coordinates"


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    sys.exit(1 if failed else 0)
