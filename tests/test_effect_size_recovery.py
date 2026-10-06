#!/usr/bin/env python3
"""Recovered-effect-size regression test.

The simulator must produce between-condition effects close to the Cohen's d the
user configures. Historically it inflated them ~4x (target d=0.5 -> observed ~2.1)
because the d->scale-shift conversion was applied to BOTH levels with a constant
that ignored the real within-condition SD, and because the persona interaction
multiplier averaged ~1.12 instead of 1.0.

Configured d refers to what the user analyses: the scale mean (multi-item) or the
single item. The engine scales the shift by item count so composites are not
inflated by averaging. Bounds tolerate bounded-scale effects (ceiling compression,
ERS) and sampling noise (SE of d ~0.06 at n=1000) but are tight enough that the
old 2-4x inflation fails by a large margin.
"""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation_app"))
warnings.filterwarnings("ignore")

from utils.enhanced_simulation_engine import EnhancedSimulationEngine, EffectSizeSpec  # noqa: E402

N_PER_RUN = 1000

# (label, scale_min, scale_max, num_items)
DESIGNS = [
    ("7pt_4items", 1, 7, 4),
    ("7pt_8items", 1, 7, 8),
    ("7pt_single", 1, 7, 1),
    ("5pt_3items", 1, 5, 3),
    ("slider_0_100", 0, 100, 1),
]


def _recovered_d(target_d, lo, hi, items, seed=7, n=N_PER_RUN):
    scales = [{
        "name": "Attitude", "variable_name": "Attitude", "num_items": items,
        "scale_points": hi - lo + 1, "scale_min": lo, "scale_max": hi,
        "reverse_items": [], "type": "likert" if items > 1 else "single_item",
    }]
    eng = EnhancedSimulationEngine(
        study_title="Message Framing and Attitudes",
        study_description="A survey study of attitudes under two message versions.",
        sample_size=n, conditions=["Version A", "Version B"], factors=[],
        scales=scales, additional_vars=[],
        demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12},
        effect_sizes=[EffectSizeSpec(
            variable="Attitude", factor="condition",
            level_high="Version A", level_low="Version B",
            cohens_d=target_d, direction="positive")],
        seed=seed,
    )
    df, _ = eng.generate()
    col = "Attitude_mean" if "Attitude_mean" in df.columns else "Attitude_1"
    a = df.loc[df["CONDITION"] == "Version A", col].astype(float)
    b = df.loc[df["CONDITION"] == "Version B", col].astype(float)
    pooled = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled)


@pytest.mark.parametrize("label,lo,hi,items", DESIGNS)
@pytest.mark.parametrize("target", [0.5, 0.8])
def test_recovered_d_tracks_target(label, lo, hi, items, target):
    """Observed d lies within [0.70x, 1.30x] of the target (old code: 2-4x)."""
    d = _recovered_d(target, lo, hi, items)
    assert 0.70 * target <= d <= 1.30 * target, (
        f"{label}: target d={target}, recovered d={d:.2f} "
        f"(allowed {0.70 * target:.2f}-{1.30 * target:.2f})")


@pytest.mark.parametrize("label,lo,hi,items", DESIGNS[:2])
def test_null_effect_stays_null(label, lo, hi, items):
    """With d=0 there must be no spurious treatment effect."""
    d = _recovered_d(0.0, lo, hi, items)
    assert abs(d) < 0.20, f"{label}: d=0 produced recovered d={d:.2f}"


def test_effect_is_monotonic_in_target():
    """Larger configured d must give larger recovered d, with no runaway inflation."""
    ds = [_recovered_d(t, 1, 7, 4) for t in (0.2, 0.5, 0.8)]
    assert ds[0] < ds[1] < ds[2], f"recovered d not monotonic: {ds}"
    assert ds[2] < 1.3 * 0.8, f"recovered d at target 0.8 inflated: {ds[2]:.2f}"


# ---------------------------------------------------------------------------
# Reverse-keyed scales: the composite must be a SCORED scale (reverse items
# recoded), otherwise opposite-keyed items cancel and effects are under-recovered.
# ---------------------------------------------------------------------------
def _reverse_engine(target_d=0.5, n=1000, seed=7, **kw):
    scales = [{
        "name": "Trust", "variable_name": "Trust", "num_items": 5,
        "scale_points": 7, "scale_min": 1, "scale_max": 7,
        "reverse_items": [2, 4], "type": "likert",
    }]
    return EnhancedSimulationEngine(
        study_title="AI vs Human Advisor and Trust",
        study_description="Participants rate trust in an AI or human advisor.",
        sample_size=n, conditions=["AI advisor", "Human advisor"], factors=[],
        scales=scales, additional_vars=[],
        demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12},
        effect_sizes=[EffectSizeSpec(
            variable="Trust", factor="condition",
            level_high="Human advisor", level_low="AI advisor",
            cohens_d=target_d, direction="positive")],
        seed=seed, **kw)


def test_composite_recodes_reverse_items():
    df, _ = _reverse_engine(n=300).generate()
    items = df[[f"Trust_{i}" for i in range(1, 6)]].astype(float).copy()
    for i in (2, 4):
        items[f"Trust_{i}"] = 8 - items[f"Trust_{i}"]
    assert np.allclose(df["Trust_mean"].astype(float), items.mean(axis=1), atol=0.011)


def test_composite_recodes_reverse_items_with_missing_data():
    df, _ = _reverse_engine(n=400, missing_data_rate=0.05, missing_data_mechanism="mcar").generate()
    items = df[[f"Trust_{i}" for i in range(1, 6)]].astype(float).copy()
    for i in (2, 4):
        items[f"Trust_{i}"] = 8 - items[f"Trust_{i}"]
    expected = items.mean(axis=1, skipna=True)
    ok = df["Trust_mean"].astype(float).notna() & expected.notna()
    assert ok.sum() > 300
    assert np.allclose(df.loc[ok, "Trust_mean"].astype(float), expected[ok], atol=0.011)


def test_reverse_keyed_composite_recovers_effect():
    """Composite d on a scale with reverse items tracks the target (was ~0.4x)."""
    df, _ = _reverse_engine(target_d=0.5).generate()
    a = df.loc[df["CONDITION"] == "Human advisor", "Trust_mean"].astype(float)
    b = df.loc[df["CONDITION"] == "AI advisor", "Trust_mean"].astype(float)
    pooled = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    d = float((a.mean() - b.mean()) / pooled)
    assert 0.6 * 0.5 <= d <= 1.3 * 0.5, f"reverse-keyed composite d={d:.2f} vs target 0.5"
