#!/usr/bin/env python3
"""v1.3.0.6: participant-quality flags match their configuration and the recorded columns."""
import contextlib
import io
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation_app"))
warnings.filterwarnings("ignore")

from utils.enhanced_simulation_engine import EnhancedSimulationEngine  # noqa: E402

LIKERT = {"name": "Att", "variable_name": "Att", "type": "likert", "num_items": 4,
          "scale_points": 7, "scale_min": 1, "scale_max": 7, "reverse_items": []}


def _run(seed=3, n=1200, scales=None, **kw):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        eng = EnhancedSimulationEngine(
            study_title="T", study_description="A survey of attitudes", sample_size=n,
            conditions=["A", "B"], factors=[], scales=scales or [dict(LIKERT)], additional_vars=[],
            demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12},
            open_ended_questions=[], seed=seed, **kw)
        eng.llm_generator.disable_permanently("test")
        df, meta = eng.generate()
    return eng, df, meta


@pytest.mark.parametrize("rate", [0.95, 0.85])
def test_attention_failure_rate_matches_configured_rate(rate):
    _, df, _ = _run(attention_rate=rate)
    fail = float((df["Attention_Pass_Rate"].astype(float) < 1).mean())
    assert abs(fail - (1 - rate)) < 0.025, fail


def test_perfect_attention_rate_means_no_failures_and_no_flags():
    _, df, _ = _run(attention_rate=1.0, n=600)
    assert (df["Attention_Pass_Rate"].astype(float) == 1).all()
    assert df["Flag_Attention"].sum() == 0


def test_careless_persona_share_follows_random_responder_rate():
    for rate in (0.05, 0.10):
        eng, _, _ = _run(random_responder_rate=rate, n=800)
        share = np.mean([eng._assign_persona(i)[0] == "careless_responder" for i in range(4000)])
        assert abs(share - rate) < 0.02, (rate, share)


def test_careless_participants_fail_more_often():
    eng, df, _ = _run(n=2000)
    careless = np.array([eng._assign_persona(i)[0] == "careless_responder" for i in range(2000)])
    fails = (df["Attention_Pass_Rate"].astype(float) < 1).to_numpy()
    assert fails[careless].mean() > 3 * fails[~careless].mean()


def test_flag_attention_fires_and_feeds_exclusion():
    _, df, meta = _run()
    failed = df["Attention_Pass_Rate"].astype(float) < 1
    assert failed.sum() > 0
    assert (df.loc[failed, "Flag_Attention"] == 1).all()
    assert (df.loc[~failed, "Flag_Attention"] == 0).all()
    assert (df.loc[failed, "Exclude_Recommended"] == 1).all()
    assert meta["exclusion_summary"]["flagged_attention"] == int(df["Flag_Attention"].sum())
    assert meta["exclusion_summary"]["total_excluded"] == int(df["Exclude_Recommended"].sum())


def test_flags_are_derivable_from_final_columns():
    eng, df, _ = _run()
    crit = eng.exclusion_criteria
    t = df["Completion_Time_Seconds"].astype(float)
    speed = ((t < crit.completion_time_min_seconds) | (t > crit.completion_time_max_seconds)).astype(int)
    assert (df["Flag_Speed"] == speed).all()
    expected = ((df["Flag_Speed"] > 0) | (df["Flag_Attention"] > 0) | (df["Flag_StraightLine"] > 0)).astype(int)
    assert (df["Exclude_Recommended"] == expected).all()


def test_straightline_gate_is_per_block_not_global():
    """A binary item in the survey must not switch the audit off for the Likert block."""
    binary = {"name": "Yes", "variable_name": "Yes", "type": "single_item", "num_items": 1,
              "scale_points": 2, "scale_min": 0, "scale_max": 1, "reverse_items": []}
    likert = dict(LIKERT, num_items=6)
    _, _, meta = _run(n=1500, scales=[likert, binary])
    _, _, meta_l = _run(n=1500, scales=[likert])
    rep = meta.get("consistency_audit") or {}
    rep_l = meta_l.get("consistency_audit") or {}
    assert rep.get("repairs_performed", 0) > 0
    assert rep_l.get("repairs_performed", 0) > 0


def test_speeders_are_kept_and_flag_speed_tracks_careless_personas():
    """The validator must not rewrite genuinely fast times: Flag_Speed fires for a realistic
    minority (3-8% with defaults) that is concentrated in the careless persona."""
    n = 2000
    eng, df, _ = _run(n=n)
    careless = np.array([eng._assign_persona(i)[0] == "careless_responder" for i in range(n)])
    speed = (df["Flag_Speed"] == 1).to_numpy()
    assert 0.03 <= speed.mean() <= 0.08, speed.mean()
    assert speed[careless].mean() > 0.4
    assert speed[~careless].mean() < 0.02
    assert (df["Completion_Time_Seconds"].astype(float) > 0).all()


def test_validator_only_corrects_impossible_durations():
    from utils.hbs_validator import HBSValidator
    import pandas as pd
    df = pd.DataFrame({"Completion_Time_Seconds": [25.0, 55.0, 300.0, 0.0, -5.0, 90000.0]})
    out = HBSValidator(seed=1)._correct_completion_time(df.copy())
    t = out["Completion_Time_Seconds"].astype(float).tolist()
    assert t[:3] == [25.0, 55.0, 300.0]
    assert all(10 <= v <= 14400 for v in t[3:])


def test_straight_line_judged_per_likert_block_not_across_unrelated_columns():
    """Many identical single-item/binary DVs must not produce straight-line flags; a real
    constant answer across a 6-item 7-point block must."""
    singles = [{"name": f"Dec{k}", "variable_name": f"Dec{k}", "type": "single_item", "num_items": 1,
                "scale_points": 2, "scale_min": 0, "scale_max": 1, "reverse_items": []}
               for k in range(12)]
    likert = dict(LIKERT, num_items=6)
    eng, df, _ = _run(n=1500, scales=[likert] + singles)
    assert df["Flag_StraightLine"].mean() < 0.06, df["Flag_StraightLine"].mean()
    assert df["Exclude_Recommended"].mean() < 0.2
    cols = [c for c in df.columns if c.startswith("Att_") and c[4:].isdigit()]
    msl = df["Max_Straight_Line"].to_numpy()
    assert msl.max() <= len(cols)
    full = (df[cols].nunique(axis=1) == 1).to_numpy()
    assert (df.loc[full, "Flag_StraightLine"] == 1).all()
    assert (df.loc[~full, "Flag_StraightLine"] == 0).all()
