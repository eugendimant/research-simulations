"""Comprehension, manipulation and named attention checks (v1.3.1.4).

These guard the fixes for the defect cluster found on 2026-11-14, when a student
reported that her manipulation check came back in a fixed 3-7 range that ignored
the scale she had entered, with no way to customise it.

Root cause: all three check families were stamped onto the finished dataframe by
``app.py`` *after* ``generate()`` returned, using fixed literal seeds, a
hard-coded 1-7 scale, and a flat +1.5 applied to every arm whose label did not
contain the substring "control".
"""

import numpy as np
import pytest

from utils.check_simulation import (
    calibrate_probability_scale,
    condition_intensity_order,
    normalize_check_name,
    normalize_comprehension_specs,
    normalize_manipulation_specs,
    resolve_existing_column,
)
from utils.enhanced_simulation_engine import EnhancedSimulationEngine, ExclusionCriteria


def _engine(**overrides):
    kwargs = dict(
        study_title="Scarcity study",
        study_description="Does perceived scarcity raise purchase intention?",
        sample_size=300,
        conditions=["Low scarcity", "High scarcity"],
        factors=[],
        scales=[{"name": "Purchase", "num_items": 3, "scale_points": 7, "type": "matrix"}],
        additional_vars=[],
        demographics={},
        seed=101,
    )
    kwargs.update(overrides)
    return EnhancedSimulationEngine(**kwargs)


# ---------------------------------------------------------------------------
# Name handling (finding 9: two different normalisations)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Comp Check 1", "Comp_Check_1"),
    ("Q1.Comp", "Q1_Comp"),
    ("MC-scarce", "MC_scarce"),
    ("  ", "Check"),
    ("3rd check", "V_3rd_check"),
])
def test_one_name_rule_for_writing_and_reading(raw, expected):
    assert normalize_check_name(raw) == expected


def test_resolve_existing_column_finds_the_engines_own_spelling():
    # A single-item DV named MC_Scarce is written as MC_Scarce_1; the old guard
    # tested the bare name, missed, and added a second fabricated column.
    assert resolve_existing_column("MC_Scarce", ["MC_Scarce_1", "Purchase_1"]) == "MC_Scarce_1"
    assert resolve_existing_column("Trust", ["Trust_1", "Trust_2", "Trust_mean"]) == "Trust_mean"
    assert resolve_existing_column("Absent", ["Purchase_1"]) is None


# ---------------------------------------------------------------------------
# Direction (findings 1 and 4)
# ---------------------------------------------------------------------------

def test_direction_from_ordinal_labels_without_a_control_arm():
    ranks, basis = condition_intensity_order(["Low scarcity", "Medium scarcity", "High scarcity"])
    assert basis == "ordinal labels"
    assert ranks[0] < ranks[1] < ranks[2]


def test_direction_from_numeric_labels():
    ranks, basis = condition_intensity_order(["10% off", "50% off", "80% off"])
    assert basis == "numeric labels"
    assert ranks == sorted(ranks)


def test_control_arm_is_pinned_low_but_not_by_substring():
    ranks, basis = condition_intensity_order(["Control", "Treatment"])
    assert basis == "reference arm"
    assert ranks[0] < ranks[1]
    # "Self-control training" is not a control arm in the experimental sense, but
    # a substring test made it one.
    _, basis2 = condition_intensity_order(["Self-control training", "Mindfulness training"])
    assert basis2 == "declared order"


def test_nominal_arms_still_get_a_direction_and_say_so():
    specs, warnings = normalize_manipulation_specs(
        [{"name": "MC"}], ["Ingroup partner", "Outgroup partner"],
    )
    assert specs[0].order_basis == "declared order"
    assert any("listed" in w for w in warnings)
    assert len(set(specs[0].targets.values())) == 2


def test_analyst_specified_direction_wins():
    specs, warnings = normalize_manipulation_specs(
        [{"name": "MC", "high_conditions": ["Low scarcity"]}],
        ["Low scarcity", "High scarcity"],
    )
    assert specs[0].order_basis == "analyst-specified"
    assert specs[0].targets["Low scarcity"] > specs[0].targets["High scarcity"]
    assert warnings == []


def test_unknown_condition_in_high_conditions_is_reported():
    specs, warnings = normalize_manipulation_specs(
        [{"name": "MC", "high_conditions": ["Typo arm"]}], ["A", "B"],
    )
    assert any("Typo arm" in w for w in warnings)
    assert specs[0].high_conditions == []


# ---------------------------------------------------------------------------
# Range (finding 2)
# ---------------------------------------------------------------------------

def test_configured_range_is_honoured_end_to_end():
    eng = _engine(manipulation_checks=[{"name": "MC", "scale_min": 1, "scale_max": 5}])
    df, meta = eng.generate()
    values = df["MC"].dropna()
    assert values.min() >= 1 and values.max() <= 5
    assert meta["checks"]["manipulation_checks"][0]["scale_max"] == 5


def test_a_binary_manipulation_check_stays_binary():
    eng = _engine(manipulation_checks=[{"name": "MC", "scale_min": 0, "scale_max": 1}])
    df, _ = eng.generate()
    assert set(df["MC"].dropna().unique()) <= {0, 1}


def test_inconsistent_range_is_repaired_and_reported():
    specs, warnings = normalize_manipulation_specs(
        [{"name": "MC", "scale_min": 7, "scale_max": 3}], ["A", "B"],
    )
    assert specs[0].scale_min < specs[0].scale_max
    assert any("inconsistent range" in w for w in warnings)


# ---------------------------------------------------------------------------
# Condition effect (findings 1 and 4) — the student's actual bug
# ---------------------------------------------------------------------------

def test_manipulation_check_differs_by_condition_without_a_control_label():
    eng = _engine(
        sample_size=600,
        conditions=["Low scarcity", "High scarcity"],
        manipulation_checks=[{"name": "MC"}],
    )
    df, _ = eng.generate()
    means = df.groupby("CONDITION")["MC"].mean()
    assert means["High scarcity"] - means["Low scarcity"] > 1.0


def test_every_arm_gets_its_own_mean_in_a_three_arm_design():
    eng = _engine(
        sample_size=900,
        conditions=["Low scarcity", "Medium scarcity", "High scarcity"],
        manipulation_checks=[{"name": "MC"}],
    )
    df, _ = eng.generate()
    means = df.groupby("CONDITION")["MC"].mean()
    assert means["Low scarcity"] < means["Medium scarcity"] < means["High scarcity"]
    # The old code gave every non-control arm the identical +1.5.
    assert abs(means["High scarcity"] - means["Medium scarcity"]) > 0.2


def test_requested_separation_is_recovered():
    eng = _engine(
        sample_size=2000,
        manipulation_checks=[{"name": "MC", "scale_min": 1, "scale_max": 7, "effect_size": 2.0}],
    )
    df, _ = eng.generate()
    low = df.loc[df["CONDITION"] == "Low scarcity", "MC"].dropna()
    high = df.loc[df["CONDITION"] == "High scarcity", "MC"].dropna()
    pooled = np.sqrt((low.var(ddof=1) + high.var(ddof=1)) / 2.0)
    d = (high.mean() - low.mean()) / pooled
    assert 1.4 <= d <= 2.6


# ---------------------------------------------------------------------------
# Duplicate columns (finding 3)
# ---------------------------------------------------------------------------

def test_a_manipulation_check_that_is_also_a_dv_does_not_get_a_second_column():
    eng = _engine(
        scales=[
            {"name": "Purchase", "num_items": 3, "scale_points": 7, "type": "matrix"},
            {"name": "MC_Scarce", "num_items": 1, "scale_points": 5, "type": "single_item"},
        ],
        manipulation_checks=[{"name": "MC_Scarce"}],
    )
    df, meta = eng.generate()
    assert "MC_Scarce_1" in df.columns
    assert "MC_Scarce" not in df.columns  # the fabricated twin
    assert "MC_Scarce_1_Pass" in df.columns
    assert meta["checks"]["manipulation_checks"][0]["bound_to_dv"] is True
    # The DV keeps the range the analyst configured for it.
    assert df["MC_Scarce_1"].dropna().max() <= 5


# ---------------------------------------------------------------------------
# Pass / fail and exclusion (finding 5)
# ---------------------------------------------------------------------------

def test_manipulation_check_has_a_pass_column():
    eng = _engine(sample_size=400, manipulation_checks=[{"name": "MC"}])
    df, meta = eng.generate()
    assert set(df["MC_Pass"].unique()) <= {0, 1}
    assert 0.6 < df["MC_Pass"].mean() <= 1.0
    assert "Flag_Manipulation" in df.columns


def test_failed_checks_only_drive_exclusion_when_asked():
    base = dict(sample_size=400, manipulation_checks=[{"name": "MC"}],
                comprehension_checks=[{"name": "Comp", "correct_answer": "A",
                                       "answer_options": "A; B", "pass_rate": 0.7}])
    off, _ = _engine(**base).generate()
    on, _ = _engine(
        exclusion_criteria=ExclusionCriteria(
            exclude_failed_comprehension=True, exclude_failed_manipulation=True
        ),
        **base,
    ).generate()
    assert off["Flag_Comprehension"].sum() > 0
    assert on["Exclude_Recommended"].sum() > off["Exclude_Recommended"].sum()


def test_no_checks_means_no_flag_columns():
    df, _ = _engine().generate()
    assert "Flag_Comprehension" not in df.columns
    assert "Flag_Manipulation" not in df.columns


# ---------------------------------------------------------------------------
# The person behind the answers (finding 6)
# ---------------------------------------------------------------------------

def test_the_run_seed_changes_the_check_columns():
    a, _ = _engine(seed=1, manipulation_checks=[{"name": "MC"}]).generate()
    b, _ = _engine(seed=2, manipulation_checks=[{"name": "MC"}]).generate()
    assert not a["MC"].equals(b["MC"])


def test_the_same_seed_reproduces_the_check_columns():
    a, _ = _engine(seed=5, manipulation_checks=[{"name": "MC"}]).generate()
    b, _ = _engine(seed=5, manipulation_checks=[{"name": "MC"}]).generate()
    assert a["MC"].equals(b["MC"])


def test_inattentive_participants_pass_the_manipulation_check_less_often():
    eng = _engine(sample_size=1200, attention_rate=0.75, random_responder_rate=0.15,
                  manipulation_checks=[{"name": "MC"}])
    df, _ = eng.generate()
    attentive = df.loc[df["Attention_Pass_Rate"] >= 1.0, "MC_Pass"].mean()
    careless = df.loc[df["Attention_Pass_Rate"] < 1.0, "MC_Pass"].mean()
    assert attentive > careless


def test_manipulation_check_correlates_with_the_outcome_within_condition():
    eng = _engine(sample_size=1500, manipulation_checks=[{"name": "MC"}])
    df, _ = eng.generate()
    rs = [
        g["MC"].corr(g["Purchase_mean"])
        for _, g in df.groupby("CONDITION")
    ]
    assert float(np.mean(rs)) > 0.05


# ---------------------------------------------------------------------------
# Comprehension checks (findings 8, 9, 10)
# ---------------------------------------------------------------------------

def test_realised_pass_rate_matches_the_configured_one():
    eng = _engine(
        sample_size=3000,
        comprehension_checks=[{"name": "Comp", "correct_answer": "Limited supply",
                               "answer_options": "Limited supply; High demand; Neither",
                               "pass_rate": 0.80}],
    )
    df, meta = eng.generate()
    observed = df["Comp_Pass"].mean()
    assert abs(observed - 0.80) < 0.03
    reported = meta["checks"]["comprehension_checks"][0]
    assert abs(reported["pass_rate_observed"] - observed) < 1e-3  # reported rounded to 4 dp


def test_reported_pass_rate_survives_punctuation_in_the_name():
    # The old metadata looked the column up with .replace(" ", "_") while it was
    # created with re.sub(r'[^\w]', '_', ...), so this reported 0.0.
    eng = _engine(
        sample_size=200,
        comprehension_checks=[{"name": "Q1.Comp Check", "correct_answer": "A",
                               "answer_options": "A; B", "pass_rate": 0.9}],
    )
    df, meta = eng.generate()
    assert "Q1_Comp_Check" in df.columns
    assert meta["checks"]["comprehension_checks"][0]["pass_rate_observed"] > 0.5


def test_wrong_answers_are_real_options_never_a_placeholder():
    eng = _engine(
        sample_size=400,
        comprehension_checks=[{"name": "Comp", "correct_answer": "Limited supply",
                               "answer_options": "Limited supply; High demand; Neither",
                               "pass_rate": 0.7}],
    )
    df, _ = eng.generate()
    assert set(df["Comp"].dropna().unique()) <= {"Limited supply", "High demand", "Neither"}
    assert not any(str(v).startswith("Wrong_") for v in df["Comp"].dropna())


def test_a_check_without_options_becomes_a_coded_column_with_a_warning():
    eng = _engine(
        sample_size=200,
        comprehension_checks=[{"name": "Comp", "correct_answer": "2", "pass_rate": 0.9}],
    )
    df, meta = eng.generate()
    assert set(df["Comp"].dropna().unique()) <= {0, 1}
    assert any("1/0" in w for w in meta["generation_warnings"])


def test_a_check_without_a_correct_answer_warns_instead_of_failing_silently():
    specs, warnings = normalize_comprehension_specs([{"name": "Comp", "correct_answer": ""}])
    assert specs == []
    assert any("no correct answer" in w for w in warnings)


def test_per_condition_correct_answers():
    eng = _engine(
        sample_size=400,
        comprehension_checks=[{
            "name": "Comp",
            "correct_by_condition": {"Low scarcity": "Plentiful", "High scarcity": "Scarce"},
            "answer_options": "Plentiful; Scarce",
            "pass_rate": 1.0,
        }],
    )
    df, _ = eng.generate()
    low = df.loc[df["CONDITION"] == "Low scarcity", "Comp"].dropna()
    high = df.loc[df["CONDITION"] == "High scarcity", "Comp"].dropna()
    assert set(low.unique()) == {"Plentiful"}
    assert set(high.unique()) == {"Scarce"}


def test_calibration_solves_for_the_sample_mean():
    weights = [0.45, 0.6, 0.8, 1.0]
    scale = calibrate_probability_scale(weights, 0.75)
    realised = float(np.mean(np.minimum(1.0, np.array(weights) * scale)))
    assert abs(realised - 0.75) < 1e-4


# ---------------------------------------------------------------------------
# Codebook and missing data (finding 11)
# ---------------------------------------------------------------------------

def test_check_columns_are_documented_in_the_codebook():
    eng = _engine(
        manipulation_checks=[{"name": "MC"}],
        comprehension_checks=[{"name": "Comp", "correct_answer": "A", "answer_options": "A; B"}],
        named_attention_checks=["AttnQ1"],
    )
    df, meta = eng.generate()
    descriptions = meta["column_descriptions"]
    for col in ("MC", "MC_Pass", "Comp", "Comp_Pass", "AttnQ1",
                "Flag_Comprehension", "Flag_Manipulation"):
        assert col in descriptions, f"{col} missing from the codebook"
        assert descriptions[col].strip()


def test_check_answers_receive_missing_data_but_pass_flags_do_not():
    eng = _engine(
        sample_size=600,
        missing_data_rate=0.08,
        manipulation_checks=[{"name": "MC"}],
    )
    df, _ = eng.generate()
    assert df["MC"].isna().sum() > 0
    assert df["MC_Pass"].isna().sum() == 0
    assert df["Flag_Manipulation"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Named attention checks (finding 6)
# ---------------------------------------------------------------------------

def test_a_named_attention_check_agrees_with_the_engines_own_pass_rate():
    eng = _engine(sample_size=1200, attention_rate=0.80, named_attention_checks=["AttnQ1"])
    df, _ = eng.generate()
    assert abs(df["AttnQ1"].mean() - df["Attention_Pass_Rate"].mean()) < 0.08


# ---------------------------------------------------------------------------
# Silent skips (finding 13)
# ---------------------------------------------------------------------------

def test_duplicate_and_empty_entries_are_reported():
    eng = _engine(manipulation_checks=[{"name": "MC"}, {"name": "MC"}, {"name": ""}])
    _, meta = eng.generate()
    warnings = " ".join(meta["generation_warnings"])
    assert "duplicates an earlier entry" in warnings
    assert "no name" in warnings


# ---------------------------------------------------------------------------
# The between-subjects path is untouched when no checks are configured
# ---------------------------------------------------------------------------

def test_configuring_no_checks_changes_nothing():
    import pandas as pd

    a, _ = _engine(seed=77).generate()
    b, _ = _engine(seed=77, comprehension_checks=[], manipulation_checks=[],
                   named_attention_checks=[]).generate()
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# The Design page (finding 14: a manipulation check had no editor at all)
# ---------------------------------------------------------------------------

def test_manipulation_check_editor_exists_on_the_design_page(monkeypatch, tmp_path):
    """The real app, driven as a student drives it.

    Until v1.3.1.4 the only control was a name box, which is why the range the
    analyst entered elsewhere was ignored by the generator.
    """
    import json
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app_dir = Path(__file__).resolve().parent.parent / "simulation_app"
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(app_dir / "app.py"), default_timeout=300)

    def _click(key):
        found = [b for b in at.button if b.key == key]
        assert found, f"no button {key!r}"
        found[0].click()

    at.run()
    _click("landing_cta")
    at.run()
    at.text_input(key="study_title").input("Scarcity study")
    at.text_area(key="study_description").input(
        "A pilot study of whether perceived scarcity raises purchase intentions."
    )
    at.run()
    _click("nav_next_0")
    at.run()

    qsf = {
        "SurveyEntry": {"SurveyID": "SV_test", "SurveyName": "Scarcity"},
        "SurveyElements": [
            {"Element": "BL", "Payload": [{
                "ID": "BL_1", "Description": "Outcomes",
                "BlockElements": [{"Type": "Question", "QuestionID": "QID1"}],
            }]},
            {"Element": "SQ", "PrimaryAttribute": "QID1", "Payload": {
                "QuestionID": "QID1", "DataExportTag": "Purchase",
                "QuestionText": "How likely are you to buy this product?",
                "QuestionType": "Matrix", "Selector": "Likert",
                "Choices": {str(i): {"Display": str(i)} for i in range(1, 8)},
                "Answers": {str(i): {"Display": f"Item {i}"} for i in range(1, 4)},
            }},
        ],
    }
    at.file_uploader[0].set_value(("scarcity.qsf", json.dumps(qsf).encode(), "application/octet-stream"))
    at.run()
    _click("nav_next_1")
    at.run()

    at.session_state["inferred_design"] = dict(
        at.session_state.get("inferred_design", {}) or {},
        conditions=["Low scarcity", "High scarcity"],
    )
    at.session_state["confirmed_manipulation_checks"] = [{"name": "MC_Scarce"}]
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]

    keys = {str(w.key) for w in at.number_input} | {str(w.key) for w in at.multiselect}
    for prefix in ("manip_min_v", "manip_max_v", "manip_d_v", "manip_high_v"):
        assert any(k.startswith(prefix) for k in keys), f"no {prefix} control on the Design page"

    # The stored spec carries the range, the direction and the separation.
    stored = at.session_state["confirmed_manipulation_checks"][0]
    for field_name in ("scale_min", "scale_max", "effect_size", "high_conditions"):
        assert field_name in stored
