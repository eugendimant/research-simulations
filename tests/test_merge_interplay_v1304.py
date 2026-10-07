"""Straight-line handling after the v1.3.0.3 / v1.2.9.1 merge (v1.3.0.4).

Two independent changes touch the same quantity, the share of respondents who give one
answer to every item of a block:

* the effect-fidelity line gated the consistency audit's straight-line repair (CHECK 3) so it
  stays away from short and narrow scales, where honest respondents agree by chance;
* the empirical-realism line added an identical-answer pass that runs last and pulls every
  block of three or more items toward a share measured on 5- to 9-point instruments.

Merged naively, that pass broke most constant rows of a binary, 3- or 4-point block (77% of a
3-item binary block) to reach a Likert-measured 19%, and a requested d of 0.8 came back as 0.61.
"""
import contextlib
import io
import random

import numpy as np
import pytest

from utils.enhanced_simulation_engine import EffectSizeSpec, EnhancedSimulationEngine

HI, LO = "Group 1", "Group 2"


def _scale(name, items, points, lo=1):
    return {"name": name, "variable_name": name, "type": "likert", "num_items": items,
            "scale_points": points, "scale_min": lo, "scale_max": lo + points - 1, "reverse_items": []}


def _run(scales, n, seed, d=0.0, effect_on=None):
    """Generate once; return (engine, frame, number of audit repairs)."""
    specs = []
    if effect_on is not None:
        specs = [EffectSizeSpec(variable=effect_on, factor="condition", level_high=HI, level_low=LO,
                                cohens_d=d, direction="positive")]
    eng = EnhancedSimulationEngine(
        study_title="Message Framing and Attitudes",
        study_description="A survey study of attitudes under different message versions.",
        sample_size=n, conditions=[HI, LO], factors=[], scales=scales, additional_vars=[],
        demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12}, effect_sizes=specs, seed=seed)
    eng.llm_generator.disable_permanently("test")
    seen = {}
    audit = eng._audit_individual_consistency

    def counting_audit(*args, **kwargs):
        report = audit(*args, **kwargs)
        seen["repairs"] = report.get("repairs_performed", 0)
        return report

    eng._audit_individual_consistency = counting_audit
    with contextlib.redirect_stdout(io.StringIO()):
        df, _ = eng.generate()
    return eng, df, seen.get("repairs", 0)


def _stage(eng, stage):
    return [r for r in eng._item_realism_log if r.get("stage") == stage]


def _identical_rows(df, name, items):
    return float((df[[f"{name}_{i}" for i in range(1, items + 1)]].nunique(axis=1) == 1).mean())


def _d(df, col):
    a = df.loc[df.CONDITION == HI, col].astype(float)
    b = df.loc[df.CONDITION == LO, col].astype(float)
    return float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2))


# ---------------------------------------------------------------------------
# 2. Narrow scales are left to chance agreement, in the audit and in the pass.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("points, lo", [(2, 0), (3, 1), (4, 1)])
def test_narrow_scales_get_neither_straightline_pass(points, lo):
    eng, _, repairs = _run([_scale("Attitude", 4, points, lo)], n=600, seed=3)
    assert repairs == 0
    assert not _stage(eng, "identical_answers"), (
        f"the registry's straight-lining share was applied to a {points}-point block")


def test_binary_scale_keeps_the_requested_effect_through_the_realism_layer():
    """3 binary items, d = 0.8: the branch recovers 0.77 on average (seeds 11-13, N = 1,500);
    the realism pass alone took it to 0.61 by flipping an item in 59% of respondents."""
    ds, shares = [], []
    for seed in (11, 12, 13):
        eng, df, _ = _run([_scale("DV", 3, 2, 0)], n=1500, seed=seed, d=0.8, effect_on="DV")
        assert not _stage(eng, "identical_answers")
        ds.append(_d(df, "DV_mean"))
        shares.append(_identical_rows(df, "DV", 3))
    assert np.mean(shares) > 0.6, f"identical rows were pulled down to {np.mean(shares):.2f}"
    assert np.mean(ds) >= 0.8 * 0.85, f"requested 0.8, observed {np.mean(ds):.2f} ({ds})"


# ---------------------------------------------------------------------------
# The pass itself, without the rest of the pipeline.
# ---------------------------------------------------------------------------
def _pass_on(points, lo, items=3, n=400, seed=2):
    """Run only the identical-answer pass on independent columns; return (changed, before, after)."""
    eng = EnhancedSimulationEngine(
        study_title="Pass", study_description="Unit test of one pass", sample_size=50,
        conditions=[HI, LO], factors=[], scales=[_scale("S", items, points, lo)], additional_vars=[],
        demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12}, seed=1)
    rng = random.Random(seed)
    cols = [f"S_{i}" for i in range(1, items + 1)]
    data = {c: [lo + rng.randrange(points) for _ in range(n)] for c in cols}
    before = {c: list(v) for c, v in data.items()}
    log = [{"name": "S", "columns_generated": cols, "scale_min": lo, "scale_max": lo + points - 1,
            "reverse_items": []}]
    changed = eng._apply_identical_answer_realism(data, log)
    return changed, before, data


@pytest.mark.parametrize("points, lo", [(2, 0), (3, 1), (4, 1)])
def test_identical_answer_pass_declines_below_five_options(points, lo):
    changed, before, after = _pass_on(points, lo)
    assert changed == []
    assert after == before


@pytest.mark.parametrize("points, lo", [(5, 1), (7, 1)])
def test_identical_answer_pass_still_runs_from_five_options(points, lo):
    changed, before, after = _pass_on(points, lo)
    assert changed, f"{points}-point block was left alone"
    assert after != before
