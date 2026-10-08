"""Within-subjects and mixed designs: the engine side (v1.3.0.6).

Pins the contract of ``utils/within_design.py`` and the ``design=`` argument of ``EnhancedSimulationEngine``:

* a between-subjects run is untouched (output hashes recorded on the integration head eecf4c7, which already contains the repeated-measures hook),
* wide layout: ``<Measure>_<Condition>_<i>`` / ``..._mean``, one row per participant, one attention check,
* a long file with one row per participant and condition that agrees with the wide data,
* the requested effect is d_av (recovered on average), d_z is reported, inferred effects work, a true null is a null,
* the same measure is correlated across conditions (default 0.5, adjustable),
* counterbalancing (Latin square, full, random, fixed) is recorded in ``Order``/``Position_*``,
* attrition removes the LATER conditions, a careless person is careless in every condition,
* mixed designs: a group effect and a group x time interaction can be requested, factorial within effects add per factor,
* same seed gives the same data.

Runs are small (N <= 400) and use the built-in engine only (no network).
"""
import hashlib
import json
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation_app"))
warnings.filterwarnings("ignore")

from utils import within_design as W  # noqa: E402
from utils.enhanced_simulation_engine import EffectSizeSpec, EnhancedSimulationEngine, ExclusionCriteria  # noqa: E402

DEMO = {"gender_quota": 50, "age_mean": 35, "age_sd": 12}


def _scale(name, items, points=7, reverse=None):
    return {"name": name, "variable_name": name, "num_items": items, "scale_points": points, "scale_min": 1,
            "scale_max": points, "reverse_items": reverse or [], "type": "likert" if items > 1 else "single_item"}


def _within(levels, n=200, seed=7, scales=None, effects=None, design=None, **kw):
    design = dict({"type": "within"}, **(design or {}))
    return EnhancedSimulationEngine(
        study_title="Repeated ratings", study_description="Ratings of a message in several versions.",
        sample_size=n, conditions=list(levels), factors=[{"name": "Version", "levels": list(levels)}],
        scales=scales or [_scale("Attitude", 4), _scale("Intent", 1)], additional_vars=[], demographics=DEMO,
        seed=seed, effect_sizes=effects, design=design, **kw)


def _d_av(df, meta, dv, a, b):
    rows = [o for o in meta["effect_sizes_observed"]
            if o["variable"] == dv and o["condition_1"] == a and o["condition_2"] == b]
    assert rows, (dv, a, b)
    return rows[0]


# ---------------------------------------------------------------------------------------------
# 1. between-subjects designs are untouched
# ---------------------------------------------------------------------------------------------
# md5 (first 12 hex digits) of df.to_csv and of the metadata JSON (without the timestamp), measured on the integration head eecf4c7 (the same configurations gave the pre-integration hashes on 6d97a95)
BASELINE_HASHES = {
    "basic_1": ["f29a321a06b4", "9efd3ec0dcb5"],
    "basic_2": ["0dda120ad65e", "fc684765385f"],
    "basic_3": ["66e61ce23c71", "fcfb79dde944"],
    "effects_4": ["f98161e88664", "0ee5f9cc0f68"],
    "factorial_oe_5": ["12b3a5a6725a", "b953fbe58479"],
}


def _hash(df, md):
    md = dict(md)
    md.pop("generation_timestamp", None)           # wall-clock values are the only run-to-run differences
    md.pop("abe3_processing_time_seconds", None)
    md["abe3_engine"] = {k: v for k, v in (md.get("abe3_engine") or {}).items() if k != "abe3_processing_time_seconds"}
    return [hashlib.md5(df.to_csv(index=False).encode()).hexdigest()[:12],
            hashlib.md5(json.dumps(md, sort_keys=True, default=str).encode()).hexdigest()[:12]]


def _between_runs():
    common = dict(additional_vars=[], demographics=DEMO)
    out = {}
    for seed in (1, 2, 3):
        e = EnhancedSimulationEngine(study_title="A", study_description="Attitudes under message versions.", sample_size=120,
                                     conditions=["Version A", "Version B"], factors=[],
                                     scales=[_scale("Attitude", 4), _scale("Intent", 1)], seed=seed, **common)
        out[f"basic_{seed}"] = _hash(*e.generate())
    e = EnhancedSimulationEngine(
        study_title="B", study_description="Trust in AI advice.", sample_size=150, conditions=["AI", "Human", "Control"],
        factors=[{"name": "Source", "levels": ["AI", "Human", "Control"]}],
        scales=[_scale("Trust", 5, 7, [2]), _scale("Slider", 1, 101)], seed=4, dropout_rate=0.05, missing_data_rate=0.02,
        effect_sizes=[EffectSizeSpec("Trust", "Source", "AI", "Control", 0.5)], **common)
    out["effects_4"] = _hash(*e.generate())
    e = EnhancedSimulationEngine(
        study_title="C", study_description="Gain and loss framing of a charity appeal.", sample_size=100,
        conditions=["Gain x High", "Gain x Low", "Loss x High", "Loss x Low"],
        factors=[{"name": "Frame", "levels": ["Gain", "Loss"]}, {"name": "Level", "levels": ["High", "Low"]}],
        scales=[_scale("Donation", 3)], seed=5,
        open_ended_questions=[{"name": "Why", "question_text": "Why did you decide this way?", "type": "text"}], **common)
    out["factorial_oe_5"] = _hash(*e.generate())
    return out


def test_between_designs_are_bit_identical_to_the_baseline():
    assert _between_runs() == BASELINE_HASHES


def test_default_design_is_between_and_unchanged_in_structure():
    e = EnhancedSimulationEngine(study_title="A", study_description="d", sample_size=40, conditions=["X", "Y"], factors=[],
                                 scales=[_scale("Attitude", 3)], additional_vars=[], demographics=DEMO, seed=1)
    assert e.design_spec is None
    df, md = e.generate()
    assert "design" not in md and "Order" not in df.columns
    assert set(df["CONDITION"]) == {"X", "Y"}


def test_between_string_and_dict_designs_mean_between():
    assert W.normalize_design(None) is None
    assert W.normalize_design("between") is None
    assert W.normalize_design({"type": "between"}) is None


# ---------------------------------------------------------------------------------------------
# 2. design description and counterbalancing
# ---------------------------------------------------------------------------------------------
def test_normalize_within_cells_and_slugs():
    spec = W.normalize_design({"type": "within"}, [{"name": "Time", "levels": ["Week 1", "Week 4"]}], ["Week 1", "Week 4"])
    assert [c["label"] for c in spec.cells] == ["Week 1", "Week 4"]
    assert [c["slug"] for c in spec.cells] == ["Week_1", "Week_4"]
    assert spec.within_correlation == 0.5 and spec.order == "random" and spec.order_effects


def test_normalize_factorial_within_cells_follow_the_condition_labels():
    factors = [{"name": "Frame", "levels": ["Gain", "Loss"]}, {"name": "Size", "levels": ["Small", "Large"]}]
    conds = ["Gain x Small", "Gain x Large", "Loss x Small", "Loss x Large"]
    spec = W.normalize_design("within", factors, conds)
    assert [c["label"] for c in spec.cells] == conds
    assert spec.cells[2]["levels"] == {"Frame": "Loss", "Size": "Small"}


def test_mixed_needs_a_within_factor_and_unknown_types_fail():
    with pytest.raises(ValueError):
        W.normalize_design({"type": "mixed"}, [{"name": "Group", "levels": ["T", "C"]}], ["T", "C"])
    with pytest.raises(ValueError):
        W.normalize_design({"type": "crossover"})
    with pytest.raises(ValueError):
        W.normalize_design({"type": "within"}, [{"name": "A", "levels": ["only"]}], ["only"])


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_latin_square_balances_positions_and_neighbours(k):
    rng = np.random.RandomState(0)
    n = 2 * k * (2 if k % 2 else 1) * 5
    orders, eff = W.build_orders(k, n, "latin_square", rng)
    assert eff == "latin_square" and orders.shape == (n, k)
    for c in range(k):
        counts = [int(np.sum(orders[:, p] == c)) for p in range(k)]
        assert max(counts) - min(counts) <= 1, counts
    if k >= 3:
        pairs = {}
        for row in orders:
            for a, b in zip(row[:-1], row[1:]):
                pairs[(a, b)] = pairs.get((a, b), 0) + 1
        assert max(pairs.values()) - min(pairs.values()) <= 1


def test_full_counterbalancing_uses_every_permutation_and_falls_back_above_five():
    orders, eff = W.build_orders(3, 60, "full", np.random.RandomState(1))
    assert eff == "full" and len({tuple(r) for r in orders}) == 6
    _, eff6 = W.build_orders(6, 60, "full", np.random.RandomState(1))
    assert eff6 == "latin_square"


def test_random_and_fixed_orders():
    a, _ = W.build_orders(4, 50, "random", np.random.RandomState(5))
    b, _ = W.build_orders(4, 50, "random", np.random.RandomState(5))
    assert (a == b).all() and all(sorted(r) == [0, 1, 2, 3] for r in a)
    fixed, eff = W.build_orders(3, 10, "fixed", np.random.RandomState(1))
    assert eff == "fixed" and (fixed == np.arange(3)).all()


# ---------------------------------------------------------------------------------------------
# 3. wide / long layout
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def wide_run():
    e = _within(["Low", "Mid", "High"], n=150, seed=3, design={"order": "latin_square"},
                effects=[EffectSizeSpec("Attitude", "Version", "High", "Low", 0.5)])
    df, md = e.generate()
    return e, df, md


def test_wide_layout_columns_and_rows(wide_run):
    _, df, md = wide_run
    assert len(df) == 150
    for cell in ("Low", "Mid", "High"):
        assert all(f"Attitude_{cell}_{i}" in df.columns for i in (1, 2, 3, 4))
        assert f"Attitude_{cell}_mean" in df.columns
        assert f"Intent_{cell}_1" in df.columns and f"Intent_{cell}_mean" not in df.columns
        assert f"Position_{cell}" in df.columns
    assert "Order" in df.columns and "Conditions_Completed" in df.columns
    assert not any("_occ" in c for c in df.columns)


def test_one_attention_check_per_participant_and_person_level_columns_once(wide_run):
    _, df, _ = wide_run
    assert [c for c in df.columns if c.startswith("Attention_Check")] == ["Attention_Check_1"]
    assert df["PARTICIPANT_ID"].is_unique and list(df["PARTICIPANT_ID"]) == list(range(1, 151))
    assert [c for c in df.columns if c in ("Age", "Gender")] == ["Age", "Gender"]


def test_composites_match_items_and_scale_range(wide_run):
    _, df, _ = wide_run
    for cell in ("Low", "Mid", "High"):
        items = df[[f"Attitude_{cell}_{i}" for i in range(1, 5)]]
        assert items.min().min() >= 1 and items.max().max() <= 7
        assert np.allclose(df[f"Attitude_{cell}_mean"], items.mean(axis=1).round(2), equal_nan=True, atol=0.011)


def test_position_columns_are_a_permutation_and_match_order(wide_run):
    _, df, _ = wide_run
    pos = df[["Position_Low", "Position_Mid", "Position_High"]].to_numpy()
    assert all(sorted(r) == [1, 2, 3] for r in pos)
    for _, row in df.head(30).iterrows():
        order = row["Order"].split(">")
        assert [row[f"Position_{c}"] for c in order] == [1, 2, 3]


def test_metadata_records_the_design(wide_run):
    e, df, md = wide_run
    d = md["design"]
    assert md["design_type"] == "within" and d["type"] == "within"
    assert [c["label"] for c in d["cells"]] == ["Low", "Mid", "High"]
    assert d["wide_columns"]["Attitude"]["High"]["items"] == [f"Attitude_High_{i}" for i in range(1, 5)]
    assert d["wide_columns"]["Attitude"]["High"]["mean"] == "Attitude_High_mean"
    assert d["within_correlation"] == 0.5 and d["order_effective"] == "latin_square"
    assert d["long_format_file"] == "Simulated_Data_Long.csv"
    assert "d_av" in d["effect_definition"]
    assert md["effect_sizes_applied"]["specs"][0]["status"] == "applied"


def test_long_format_agrees_with_the_wide_data(wide_run):
    _, df, md = wide_run
    long = W.build_long_format(df, md)
    assert len(long) == 150 * 3
    assert set(long["Condition"]) == {"Low", "Mid", "High"}
    assert {"Version", "Position", "Order", "Attitude_1", "Attitude_mean", "Intent_1"} <= set(long.columns)
    assert not any(c.startswith(("Flag_", "ABE3_")) for c in long.columns)   # diagnostics stay in the sidecar
    row = long[(long["PARTICIPANT_ID"] == 7) & (long["Condition"] == "Mid")].iloc[0]
    wide = df[df["PARTICIPANT_ID"] == 7].iloc[0]
    assert row["Attitude_mean"] == wide["Attitude_Mid_mean"] or (math.isnan(row["Attitude_mean"]) and math.isnan(wide["Attitude_Mid_mean"]))
    assert row["Position"] == wide["Position_Mid"]
    assert (long.groupby("PARTICIPANT_ID").size() == 3).all()


def test_same_seed_same_data_and_other_seed_differs():
    a = _within(["A", "B"], n=60, seed=11).generate()[0]
    b = _within(["A", "B"], n=60, seed=11).generate()[0]
    c = _within(["A", "B"], n=60, seed=12).generate()[0]
    assert a.equals(b)
    assert not a.equals(c)


def test_exports_know_the_wide_layout(wide_run):
    e, df, md = wide_run
    r = e.generate_r_export(df)
    assert "Attitude_Low_composite" in r and "Simulated_Data_Long.csv" in r and "does not run any analysis" in r
    py = e.generate_python_export(df)
    assert "Attitude_High_composite" in py and "levels=" not in py
    assert "Attitude_Mid_composite" in e.generate_stata_export(df).lower().replace("attitude_mid_composite", "Attitude_Mid_composite")
    assert "COMPUTE Attitude_Low_composite = MEAN(" in e.generate_spss_export(df)
    assert "Attitude_Low_composite" in e.generate_julia_export(df)


# ---------------------------------------------------------------------------------------------
# 4. correlation across conditions
# ---------------------------------------------------------------------------------------------
def _mean_r(df, dv, cells):
    cols = [f"{dv}_{c}_mean" if f"{dv}_{c}_mean" in df.columns else f"{dv}_{c}_1" for c in cells]
    m = df[cols].corr().to_numpy()
    return float(np.mean(m[np.triu_indices(len(cols), 1)]))


@pytest.mark.parametrize("r", [0.2, 0.5, 0.7])
def test_within_person_correlation_is_the_requested_one(r):
    e = _within(["A", "B", "C"], n=400, seed=21, design={"within_correlation": r, "order_effects": False})
    df, md = e.generate()
    assert abs(_mean_r(df, "Attitude", ["A", "B", "C"]) - r) < 0.08
    assert abs(_mean_r(df, "Intent", ["A", "B", "C"]) - r) < 0.10     # single item: top-up closes the gap


def test_ar1_correlation_decays_with_lag():
    e = _within(["T1", "T2", "T3"], n=400, seed=5, design={"within_correlation": 0.6, "correlation_structure": "ar1"})
    df, _ = e.generate()
    c = df[[f"Attitude_{t}_mean" for t in ("T1", "T2", "T3")]].corr().to_numpy()
    assert c[0, 1] > c[0, 2] + 0.08


# ---------------------------------------------------------------------------------------------
# 5. effects: d_av is the requested d
# ---------------------------------------------------------------------------------------------
def test_recovery_of_d_av_for_composite_and_single_item():
    got = {"Attitude": [], "Intent": []}
    for seed in (101, 102, 103, 104, 105):
        e = _within(["L1", "L2"], n=300, seed=seed, effects=[EffectSizeSpec(v, "Version", "L2", "L1", 0.5) for v in got],
                    design={"order": "latin_square"})
        df, md = e.generate()
        for v in got:
            got[v].append(-_d_av(df, md, v, "L1", "L2")["d_av"])
    for v, vals in got.items():
        ratio = float(np.mean(vals)) / 0.5
        assert 0.85 <= ratio <= 1.15, (v, ratio)


def test_d_z_is_reported_and_related_to_d_av():
    e = _within(["L1", "L2"], n=400, seed=9, effects=[EffectSizeSpec("Attitude", "Version", "L2", "L1", 0.6)])
    df, md = e.generate()
    o = _d_av(df, md, "Attitude", "L1", "L2")
    assert o["d_av"] < -0.35 and o["d_z"] < -0.35
    r = o["r"]
    assert o["d_z"] == pytest.approx(o["d_av"] / math.sqrt(2 * (1 - r)), rel=0.15)


def test_negative_direction_flips_the_sign():
    e = _within(["L1", "L2"], n=300, seed=12, effects=[EffectSizeSpec("Attitude", "Version", "L2", "L1", 0.6, "negative")])
    df, md = e.generate()
    assert _d_av(df, md, "Attitude", "L1", "L2")["d_av"] > 0.3


def test_true_null_has_no_condition_effect():
    e = _within(["A", "B", "C"], n=400, seed=31, auto_effects=False, design={"order_effects": False, "order": "random"})
    df, md = e.generate()
    for a, b in (("A", "B"), ("A", "C"), ("B", "C")):
        for dv in ("Attitude", "Intent"):
            assert abs(_d_av(df, md, dv, a, b)["d_av"]) < 0.22, (dv, a, b)
    assert all(row["source"] == "none" for row in md["effect_sizes_applied"]["cell_offsets_d"])


def test_inferred_effects_work_and_the_reference_level_is_the_zero_point():
    levels = ["Control", "Positive message"]
    e = _within(levels, n=300, seed=15, scales=[_scale("Attitude", 4)])
    df, md = e.generate()
    offs = {r["cell"]: r["offset_d"] for r in md["effect_sizes_applied"]["cell_offsets_d"]}
    assert offs["Control"] == 0.0
    assert offs["Positive message"] != 0.0
    assert {r["source"] for r in md["effect_sizes_applied"]["cell_offsets_d"]} == {"inferred"}
    off_run = _within(levels, n=300, seed=15, scales=[_scale("Attitude", 4)], auto_effects=False).generate()[1]
    assert {r["source"] for r in off_run["effect_sizes_applied"]["cell_offsets_d"]} == {"none"}


def test_a_spec_that_reaches_nothing_is_reported():
    e = _within(["A", "B"], n=60, seed=2, effects=[EffectSizeSpec("Attitude", "Version", "Nope", "Never", 0.5)])
    df, md = e.generate()
    assert md["effect_sizes_applied"]["specs"][0]["status"] == "levels_not_found"


def test_chained_specs_combine_by_least_squares():
    effects = [EffectSizeSpec("Attitude", "Version", "B", "A", 0.4), EffectSizeSpec("Attitude", "Version", "C", "B", 0.4)]
    e = _within(["A", "B", "C"], n=400, seed=41, effects=effects, scales=[_scale("Attitude", 4)],
                design={"order_effects": False, "order": "latin_square"})
    df, md = e.generate()
    offs = {r["cell"]: r["offset_d"] for r in md["effect_sizes_applied"]["cell_offsets_d"]}
    assert offs["C"] - offs["A"] == pytest.approx(0.8, abs=1e-6)
    assert -_d_av(df, md, "Attitude", "A", "C")["d_av"] == pytest.approx(0.8, abs=0.25)


def test_factorial_within_effects_add_per_factor():
    conds = ["Gain x Small", "Gain x Large", "Loss x Small", "Loss x Large"]
    factors = [{"name": "Frame", "levels": ["Gain", "Loss"]}, {"name": "Size", "levels": ["Small", "Large"]}]
    effects = [EffectSizeSpec("Attitude", "Frame", "Gain", "Loss", 0.5), EffectSizeSpec("Attitude", "Size", "Large", "Small", 0.5)]
    e = EnhancedSimulationEngine(
        study_title="F", study_description="Framing of a donation appeal.", sample_size=300, conditions=conds, factors=factors,
        scales=[_scale("Attitude", 4)], additional_vars=[], demographics=DEMO, seed=51, effect_sizes=effects,
        design={"type": "within", "order": "latin_square", "order_effects": False})
    df, md = e.generate()
    offs = {r["cell"]: r["offset_d"] for r in md["effect_sizes_applied"]["cell_offsets_d"]}
    assert offs["Gain x Large"] - offs["Loss x Small"] == pytest.approx(1.0, abs=1e-6)
    assert offs["Gain x Small"] - offs["Loss x Small"] == pytest.approx(0.5, abs=1e-6)
    gain = df[["Attitude_Gain_x_Small_mean", "Attitude_Gain_x_Large_mean"]].mean(axis=1)
    loss = df[["Attitude_Loss_x_Small_mean", "Attitude_Loss_x_Large_mean"]].mean(axis=1)
    assert 0.2 < (gain - loss).mean() / ((gain.std() + loss.std()) / 2) < 0.9


# ---------------------------------------------------------------------------------------------
# 6. order, attrition, careless responding
# ---------------------------------------------------------------------------------------------
def test_order_drift_is_switchable_and_small():
    on = _within(["A", "B", "C"], n=400, seed=61, auto_effects=False,
                 design={"order": "fixed", "order_effects": True, "order_effect_d": -0.2})
    df_on, _ = on.generate()
    off = _within(["A", "B", "C"], n=400, seed=61, auto_effects=False, design={"order": "fixed", "order_effects": False})
    df_off, _ = off.generate()
    drift_on = df_on["Attitude_A_mean"].mean() - df_on["Attitude_C_mean"].mean()
    drift_off = df_off["Attitude_A_mean"].mean() - df_off["Attitude_C_mean"].mean()
    assert drift_on - drift_off > 0.25          # earlier conditions score higher when the drift is -0.2 SD per position


def test_attrition_removes_the_later_conditions():
    e = _within(["A", "B", "C"], n=300, seed=71, dropout_rate=0.2, missing_data_rate=0.0, design={"order": "random"})
    df, md = e.generate()
    assert (df["Conditions_Completed"] < 3).sum() == 60
    for cell in ("A", "B", "C"):
        missing = df[f"Attitude_{cell}_1"].isna()
        expected = df[f"Position_{cell}"] > df["Conditions_Completed"]
        assert (missing == expected).all(), cell
        assert df.loc[missing, f"Attitude_{cell}_mean"].isna().all()
    assert md["design"]["attrition"]["dropped"] == 60


def test_no_dropout_means_everyone_finishes():
    df, _ = _within(["A", "B"], n=80, seed=72, missing_data_rate=0.0).generate()
    assert (df["Conditions_Completed"] == 2).all()
    assert df[[c for c in df.columns if c.startswith("Attitude_") and c[-1].isdigit()]].notna().all().all()


def test_a_careless_person_is_careless_in_every_condition():
    e = _within(["A", "B", "C"], n=300, seed=81, random_responder_rate=0.25, attention_rate=0.8, scales=[_scale("Attitude", 5)],
                missing_data_rate=0.0, exclusion_criteria=ExclusionCriteria(straight_line_threshold=5))
    df, md = e.generate()
    const = np.column_stack([(df[[f"Attitude_{c}_{i}" for i in range(1, 6)]].nunique(axis=1) == 1).to_numpy() for c in "ABC"])
    n_const = const.sum(1)
    assert (n_const == 3).sum() >= 10                  # straight-liners exist, and they do it in all three conditions
    # almost nobody straight-lines in "most but not all" conditions (an attentive person who ends up on the scale
    # maximum in two blocks by chance is the only way to get there)
    assert (n_const == 2).sum() <= 3
    flagged = df["Flag_StraightLine"].to_numpy() == 1
    assert (n_const[flagged] != 2).all()
    assert md["design"]["careless"]["n_flagged_straight_line"] >= int((n_const == 3).sum())


def test_scale_reliability_is_kept_in_every_block():
    df, _ = _within(["A", "B"], n=400, seed=91, scales=[_scale("Attitude", 5)], design={"order_effects": False}).generate()
    for cell in "AB":
        m = df[[f"Attitude_{cell}_{i}" for i in range(1, 6)]].to_numpy(dtype=float)
        k = m.shape[1]
        alpha = k / (k - 1) * (1 - m.var(axis=0, ddof=1).sum() / m.sum(axis=1).var(ddof=1))
        assert 0.6 < alpha < 0.95, (cell, alpha)


def test_reverse_keyed_items_stay_raw_in_the_columns_and_recoded_in_the_composite():
    df, _ = _within(["A", "B"], n=80, seed=92, scales=[_scale("Attitude", 4, reverse=[2])]).generate()
    raw = df[[f"Attitude_A_{i}" for i in range(1, 5)]].to_numpy(dtype=float)
    expect = np.nanmean(np.column_stack([raw[:, 0], 8 - raw[:, 1], raw[:, 2], raw[:, 3]]), axis=1)
    assert np.allclose(expect, df["Attitude_A_mean"], atol=0.011, equal_nan=True)


# ---------------------------------------------------------------------------------------------
# 7. mixed designs
# ---------------------------------------------------------------------------------------------
def _mixed(effects=None, simple=None, n=300, seed=111, **kw):
    design = {"type": "mixed", "within_factors": [{"name": "Time", "levels": ["Pre", "Post"]}], "order_effects": False,
              "simple_effects": simple or []}
    return EnhancedSimulationEngine(
        study_title="Training", study_description="A training programme and later performance ratings.", sample_size=n,
        conditions=["Training", "Waitlist"], factors=[{"name": "Group", "levels": ["Training", "Waitlist"]}],
        scales=[_scale("Skill", 4)], additional_vars=[], demographics=DEMO, seed=seed, effect_sizes=effects, design=design,
        auto_effects=False, **kw)


def test_mixed_design_layout_and_groups():
    df, md = _mixed(effects=[EffectSizeSpec("Skill", "Group", "Training", "Waitlist", 0.6)]).generate()
    assert set(df["CONDITION"]) == {"Training", "Waitlist"}
    assert {"Skill_Pre_mean", "Skill_Post_mean"} <= set(df.columns)
    assert md["design"]["type"] == "mixed" and md["design"]["group_labels"] == ["Training", "Waitlist"]
    long = W.build_long_format(df, md)
    assert len(long) == 2 * len(df) and set(long["Condition"]) == {"Pre", "Post"}


def test_group_effect_appears_after_the_baseline_level_only():
    df, md = _mixed(effects=[EffectSizeSpec("Skill", "Group", "Training", "Waitlist", 0.8)], n=600).generate()
    g = df["CONDITION"]
    d_pre = (df.loc[g == "Training", "Skill_Pre_mean"].mean() - df.loc[g == "Waitlist", "Skill_Pre_mean"].mean()) / df["Skill_Pre_mean"].std()
    d_post = (df.loc[g == "Training", "Skill_Post_mean"].mean() - df.loc[g == "Waitlist", "Skill_Post_mean"].mean()) / df["Skill_Post_mean"].std()
    assert abs(d_pre) < 0.25
    assert 0.5 < d_post < 1.15


def test_interaction_can_be_requested_at_one_within_level():
    simple = [{"variable": "Skill", "factor": "Group", "level_high": "Training", "level_low": "Waitlist", "cohens_d": 0.8,
               "direction": "positive", "at": {"Time": "Post"}}]
    df, md = _mixed(simple=simple, n=600).generate()
    g = df["CONDITION"]
    d_post = (df.loc[g == "Training", "Skill_Post_mean"].mean() - df.loc[g == "Waitlist", "Skill_Post_mean"].mean()) / df["Skill_Post_mean"].std()
    assert 0.5 < d_post < 1.15
    assert md["effect_sizes_applied"]["specs"][0]["scope"] == {"Time": "Post"}


def test_within_effect_inside_a_mixed_design_moves_both_groups():
    effects = [EffectSizeSpec("Skill", "Time", "Post", "Pre", 0.5)]
    df, md = _mixed(effects=effects, n=500).generate()
    change = (df["Skill_Post_mean"] - df["Skill_Pre_mean"])
    assert change.mean() > 0.3
    assert abs(change[df["CONDITION"] == "Training"].mean() - change[df["CONDITION"] == "Waitlist"].mean()) < 0.5


# ---------------------------------------------------------------------------------------------
# 8. suggestions from a survey's block structure
# ---------------------------------------------------------------------------------------------
def test_repeated_blocks_that_everyone_sees_are_suggested_as_within():
    blocks = [{"name": "Consent", "questions": ["I agree"]},
              {"name": "Pre-test", "questions": ["How anxious do you feel?", "How calm do you feel?"]},
              {"name": "Post-test", "questions": ["How anxious do you feel?", "How calm do you feel?"]}]
    out = W.suggest_design_from_blocks(blocks)
    assert out["suggest"] == "within" and out["confidence"] == "high" and out["levels"] == ["Pre-test", "Post-test"]


def test_randomized_or_one_off_blocks_are_not_suggested():
    blocks = [{"name": "Cond A", "questions": ["Rate the ad"]}, {"name": "Cond B", "questions": ["Rate the ad"]}]
    assert W.suggest_design_from_blocks(blocks, has_randomizer=True)["suggest"] is None
    assert W.suggest_design_from_blocks([{"name": "Q", "questions": ["Rate the ad"]}])["suggest"] is None


def test_wave_numbers_and_digits_are_ignored_when_matching_questions():
    blocks = [{"name": "Wave 1", "questions": ["Wave 1: how satisfied are you?"]},
              {"name": "Wave 2", "questions": ["Wave 2: how satisfied are you?"]},
              {"name": "Wave 3", "questions": ["Wave 3: how satisfied are you?"]}]
    out = W.suggest_design_from_blocks(blocks)
    assert out["suggest"] == "within" and len(out["blocks"]) == 3


# ---------------------------------------------------------------------------------------------
# fixed order: no drift by default, and the confound is stated
# ---------------------------------------------------------------------------------------------
def test_fixed_order_adds_no_drift_by_default_and_says_why():
    e = _within(["A", "B", "C"], n=60, seed=4, design={"order": "fixed"})
    assert e.design_spec.order_effects is False
    df, md = e.generate()
    assert md["design"]["order_effect"]["enabled"] is False
    assert any("confounded" in w for w in md["generation_warnings"])
    assert any("confounded" in n for n in md["design"]["notes"])
    with_drift = _within(["A", "B", "C"], n=60, seed=4, design={"order": "fixed", "order_effects": True})
    assert with_drift.design_spec.order_effects is True
    assert any("confounded with the contrast" in n for n in with_drift.design_spec.notes)


def test_counterbalanced_orders_keep_the_default_drift():
    for order in ("random", "latin_square", "full"):
        assert _within(["A", "B", "C"], n=30, seed=4, design={"order": order}).design_spec.order_effects is True


# ---------------------------------------------------------------------------------------------
# open-ended questions follow the within-conditions
# ---------------------------------------------------------------------------------------------
def _oe_run(**kw):
    lv = ["Gain frame", "Loss frame"]
    oe = [{"name": "Why", "question_text": "Why did you rate it this way?", "block_name": "Gain frame"},
          {"name": "Why2", "question_text": "Why did you rate it this way?", "block_name": "Loss frame survey"},
          {"name": "Comments", "question_text": "Any final comments?", "block_name": "Wrap-up"},
          {"name": "Both", "question_text": "Describe your reaction.", "per_condition": True},
          {"name": "Ambiguous", "question_text": "Anything else?", "block_name": "Gain frame and Loss frame"}]
    e = EnhancedSimulationEngine(
        study_title="Framing donation appeals", study_description="Gain and loss framing of a charity appeal.", sample_size=40,
        conditions=lv, factors=[{"name": "Frame", "levels": lv}], scales=[_scale("Donation", 3)], additional_vars=[],
        demographics=DEMO, seed=2, open_ended_questions=oe, design={"type": "within"}, **kw)
    e.llm_generator and e.llm_generator.disable_permanently("test")
    return e, *e.generate()


def test_open_ended_questions_are_bound_to_their_condition_block_or_asked_once():
    _, df, md = _oe_run()
    d = md["design"]
    assert d["open_ended_columns"]["Why"] == {"Gain frame": "Why_Gain_frame"}
    assert d["open_ended_columns"]["Why2"] == {"Loss frame": "Why2_Loss_frame"}
    assert set(d["open_ended_columns"]["Both"]) == {"Gain frame", "Loss frame"}
    assert set(d["open_ended_once"]) == {"Comments", "Ambiguous"}
    for col in ("Why_Gain_frame", "Why2_Loss_frame", "Both_Gain_frame", "Both_Loss_frame", "Comments", "Ambiguous"):
        assert col in df.columns and (df[col].astype(str).str.len() > 0).all()
    assert not any("_occ" in c for c in df.columns)
    assert df["Both_Gain_frame"].tolist() != df["Both_Loss_frame"].tolist()
    names = {q.get("name") for q in md["open_ended_questions"]}
    assert {"Why_Gain_frame", "Both_Loss_frame", "Comments"} <= names


def test_a_condition_bound_question_is_blank_for_people_who_never_reached_it():
    _, df, md = _oe_run(dropout_rate=0.5)
    gone = df["Conditions_Completed"] < df["Position_Gain_frame"]
    assert gone.any()
    assert (df.loc[gone, "Both_Gain_frame"].isna() | (df.loc[gone, "Both_Gain_frame"] == "")).all()


def test_long_format_carries_condition_bound_text_in_its_own_rows():
    _, df, md = _oe_run()
    long = W.build_long_format(df, md)
    gain = long[long["Condition"] == "Gain frame"].set_index("PARTICIPANT_ID")
    assert gain["Both"].tolist() == df.set_index("PARTICIPANT_ID")["Both_Gain_frame"].tolist()
    assert (long[long["Condition"] == "Loss frame"]["Why"].astype(str) == "").all()
    assert "Comments" in long.columns


def test_open_ended_text_in_a_mixed_design_names_the_group_and_the_condition():
    lv = ["Treatment", "Control"]
    e = EnhancedSimulationEngine(
        study_title="Training", study_description="A training programme.", sample_size=30, conditions=lv,
        factors=[{"name": "Group", "levels": lv}], scales=[_scale("Skill", 3)], additional_vars=[], demographics=DEMO, seed=3,
        open_ended_questions=[{"name": "Reflect", "question_text": "Reflect on the session.", "per_condition": True}],
        design={"type": "mixed", "within_factors": [{"name": "Time", "levels": ["Pre", "Post"]}]})
    e.llm_generator and e.llm_generator.disable_permanently("test")
    df, md = e.generate()
    assert {"Reflect_Pre", "Reflect_Post"} <= set(df.columns)


def test_whole_sample_correlation_lands_within_005_of_the_request_with_careless_responders():
    """r is what the analyst sees in the whole sample (careless responders who repeat an answer included)."""
    for target in (0.3, 0.6):
        vals = []
        for seed in (1, 2, 3):
            df, _ = _within(["A", "B", "C"], n=300, seed=seed, design={"within_correlation": target, "order": "latin_square"}).generate()
            vals.append(_mean_r(df, "Attitude", ["A", "B", "C"]))
        assert abs(float(np.mean(vals)) - target) < 0.05, (target, vals)
