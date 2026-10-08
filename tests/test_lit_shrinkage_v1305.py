"""Replication shrinkage of INFERRED effects (v1.3.0.5).

Published effects are inflated by selective reporting, so a simulator that reproduces
published d is too clean. Effects the tool infers (the paradigm anchor and the
literature fallback) are now multiplied by a recalled replication factor and given a
between-study draw. The factor is recalled, not source-verified, and the tests hold the
tier honest. What must NOT change: effects the user specifies, the ``auto_effects=False``
null, economic-game baselines, and seeded reproducibility.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulation_app"))

from utils import empirical_registry as R                      # noqa: E402
from utils import literature_effects as L                      # noqa: E402
from utils.enhanced_simulation_engine import (                 # noqa: E402
    EffectSizeSpec, EnhancedSimulationEngine, _match_meta_entry)

POLICY_KEY = "policy:publication_bias"
AS_PUBLISHED = R.EffectPolicy(mode="as_published", heterogeneity_draw=False)
SHRINK_ONLY = R.EffectPolicy(heterogeneity_draw=False)


def _engine(title="Anchoring effect on price estimates", conds=("High anchor", "Low anchor"),
            n=300, seed=5, explicit=False, auto=True, policy=None, dv="Price"):
    scales = [{"name": dv, "variable_name": dv, "num_items": 4, "scale_points": 7,
               "scale_min": 1, "scale_max": 7, "reverse_items": [], "type": "likert"}]
    specs = [EffectSizeSpec(variable=dv, factor="condition", level_high=conds[0],
                            level_low=conds[1], cohens_d=0.5, direction="positive")] if explicit else None
    eng = EnhancedSimulationEngine(
        study_title=title, study_description=title, sample_size=n, conditions=list(conds),
        factors=[], scales=scales, additional_vars=[],
        demographics={"gender_quota": 50, "age_mean": 30, "age_sd": 8},
        effect_sizes=specs, seed=seed, auto_effects=auto)
    if policy is not None:
        eng._INFERRED_EFFECT_POLICY = policy
    return eng


def _dissonance(seed=11, policy=None):
    scales = [{"name": "attitude", "variable_name": "attitude", "num_items": 4, "scale_points": 7,
               "_validated": True, "scale_min": 1, "scale_max": 7,
               "question_text": "How favourable is your attitude toward the essay topic?",
               "dv_description": "attitude"}]
    eng = EnhancedSimulationEngine(
        study_title="Dissonance", study_description="", sample_size=20,
        conditions=["cognitive_dissonance_control", "cognitive_dissonance_induced"],
        factors=[], scales=scales, additional_vars=[], demographics={}, seed=seed)
    if policy is not None:
        eng._INFERRED_EFFECT_POLICY = policy
    return eng


# ── the policy itself: recalled, never verified ───────────────────────────────

def test_policy_is_active_and_labelled_recalled():
    rec = R.PROVENANCE[POLICY_KEY]
    assert rec.status in R.RECALL_TIERS and rec.status != R.VERIFIED
    assert rec.note.startswith("RECALL, NOT SOURCE-VERIFIED")
    assert rec.doi == "" and rec.url == "" and rec.quote == ""
    assert R.shrinkage_factor() == pytest.approx(0.60)
    assert R.default_tau() == pytest.approx(0.15)
    assert R.shrinkage_verified() is False
    s = R.coverage_summary()
    assert s["shrinkage_verified"] is False and s["shrinkage_active"] is True
    assert s["shrinkage_tier"] in R.RECALL_TIERS
    assert "NOT been checked" in R.honesty_notice()


def test_a_source_verified_gate_still_refuses_recall_and_recall_cannot_overwrite_a_check():
    with pytest.raises(R.ProvenanceTooWeak):
        R.set_shrinkage(0.6, 0.15, R.Provenance(status=R.RECALL_CONSISTENT, quote="x"))
    saved_prov, saved_state = dict(R.PROVENANCE), dict(R._SHRINKAGE)
    try:
        R.PROVENANCE[POLICY_KEY] = R.Provenance(status=R.VERIFIED, quote="q", doi="10.x/y")
        assert R.set_recalled_shrinkage(0.3, 0.1, "later recall") is False
        assert R.PROVENANCE[POLICY_KEY].status == R.VERIFIED
    finally:
        R.PROVENANCE.clear(); R.PROVENANCE.update(saved_prov)
        R._SHRINKAGE.clear(); R._SHRINKAGE.update(saved_state)


def test_no_registry_or_match_ever_reports_the_policy_as_verified():
    m = L.lookup("cognitive_dissonance_induced", "attitude")
    assert m is not None and m.status != R.VERIFIED
    eng = _engine()
    eng._get_effect_for_condition("High anchor", "Price")
    for row in eng._inferred_effect_log:
        assert row["verification"] in R.RECALL_TIERS
    assert eng._inferred_policy_summary()["source_verified"] is False


def test_adjust_effect_applies_the_factor_and_respects_the_floor():
    assert R.adjust_effect(0.8, key="", policy=SHRINK_ONLY) == pytest.approx(0.48)
    assert R.adjust_effect(-0.8, key="", policy=SHRINK_ONLY) == pytest.approx(-0.48)
    assert R.adjust_effect(0.8, key="", policy=AS_PUBLISHED) == pytest.approx(0.8)
    rng = random.Random(0)
    for pub in (0.1, 0.3, 0.8):
        for tau in (0.0, 0.15, 0.4):
            for _ in range(300):
                out = R.adjust_effect(pub, key="unknown_key", rng=rng, tau=tau)
                assert out >= R.EffectPolicy().min_retained * pub - 1e-12   # never erased
                assert out > 0                                              # never flipped
    assert R.adjust_effect(0.0) == 0.0


# ── both inferred paths are shrunk ────────────────────────────────────────────

def test_literature_fallback_is_shrunk():
    kw = dict(condition="cognitive_dissonance_induced", variable="attitude")
    raw = L.lookup(policy=AS_PUBLISHED, **kw)
    shr = L.lookup(policy=SHRINK_ONLY, **kw)
    assert raw.published_d == shr.published_d
    assert shr.shrinkage == pytest.approx(0.60) and raw.shrinkage == 1.0
    # v1.3.0.6: one discount -- the stronger of the entry's tier weight (already in `raw`) and the 0.60
    weight = R.confidence_weight("meta", raw.key)
    assert shr.effect_d == pytest.approx(shr.published_d * min(weight, 0.60))
    assert abs(shr.effect_d) <= abs(raw.effect_d) + 1e-12
    assert abs(shr.effect_d) <= abs(shr.published_d) * 0.60 + 1e-12      # never harder than a verified entry
    assert shr.as_dict()["shrinkage_factor"] == pytest.approx(0.60)


def test_fallback_default_draw_uses_a_default_tau_when_the_entry_has_none():
    # an entry without tau must still get the policy's between-study draw
    rng_a, rng_b = random.Random(1), random.Random(2)
    a = R.adjust_effect(0.5, key="", rng=rng_a, tau=None)
    b = R.adjust_effect(0.5, key="", rng=rng_b, tau=None)
    assert a != b


def test_engine_fallback_logs_published_factor_and_applied():
    eng = _dissonance()
    v = eng._get_effect_for_condition("cognitive_dissonance_induced", "attitude")
    # since the paradigm vocabulary grew, this label is anchored; either inferred path must log the policy
    row = [r for r in eng._inferred_effect_log if r["path"] in ("literature_fallback", "paradigm_anchor")][0]
    assert row["shrinkage_factor"] == pytest.approx(0.60)
    assert row["applied_d"] < row["published_d"]
    assert row["applied_d"] >= 0.35 * row["published_d"] - 1e-9
    assert v > 0


def test_paradigm_anchor_is_shrunk_and_matches_the_logged_d():
    pub = _match_meta_entry("anchoring effect on price estimates")[1]
    raw = _engine(policy=AS_PUBLISHED)
    shr = _engine(policy=SHRINK_ONLY)
    v_raw = raw._get_effect_for_condition("High anchor", "Price")
    v_shr = shr._get_effect_for_condition("High anchor", "Price")
    assert v_shr == pytest.approx(v_raw * 0.60, rel=1e-6)
    row = shr._inferred_effect_log[0]
    assert row["path"] == "paradigm_anchor"
    assert row["published_d"] == pytest.approx(pub, abs=1e-4)
    assert row["applied_d"] == pytest.approx(pub * 0.60, abs=1e-4)
    # default policy: shrunk, with a between-study draw, never under the floor
    dflt = _engine()
    dflt._get_effect_for_condition("High anchor", "Price")
    ap = dflt._inferred_effect_log[0]["applied_d"]
    assert ap >= 0.35 * pub - 1e-9 and ap > 0


def test_all_arms_of_one_contrast_share_the_anchor_draw():
    eng = _engine()
    hi = eng._get_effect_for_condition("High anchor", "Price")
    lo = eng._get_effect_for_condition("Low anchor", "Price")
    assert hi > 0 > lo and len(eng._inferred_effect_log) == 1


# ── what must not change ──────────────────────────────────────────────────────

def _frame(eng):
    df, _ = eng.generate()
    return df


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_explicit_effects_do_not_depend_on_the_shrinkage_policy(seed, monkeypatch):
    """User-specified d never goes through adjust_effect: with the factor removed and
    with the default policy the generated data are bit-identical. (The same runs were
    also compared against the tree at 40af6c5, the commit before this change.)"""
    with_policy = _frame(_engine(seed=seed, explicit=True))
    monkeypatch.setitem(R._SHRINKAGE, "factor", None)
    monkeypatch.setitem(R._SHRINKAGE, "default_tau", None)
    without = _frame(_engine(seed=seed, explicit=True, policy=AS_PUBLISHED))
    pd.testing.assert_frame_equal(with_policy, without)


def test_explicit_effect_leaves_the_inferred_log_empty():
    eng = _engine(explicit=True)
    _frame(eng)
    assert not getattr(eng, "_inferred_effect_log", [])


def test_true_null_stays_null():
    eng = _engine(auto=False)
    assert eng._get_effect_for_condition("High anchor", "Price") == 0.0
    assert eng._get_effect_for_condition("Low anchor", "Price") == 0.0
    df = _frame(_engine(auto=False, n=600))
    rows = _engine(auto=False)
    meta = rows.generate()[1]["effect_sizes_applied"]
    assert not getattr(rows, "_inferred_effect_log", [])
    assert all(c["source"] == "none" for c in meta["contrasts"])
    assert len(df) == 600


def test_economic_game_titles_are_not_anchored_or_shrunk():
    eng = _engine(title="Dictator game giving to a partner", conds=("Ingroup partner", "Outgroup partner"),
                  dv="Giving")
    eng._get_effect_for_condition("Ingroup partner", "Giving")
    assert not [r for r in getattr(eng, "_inferred_effect_log", []) if r["path"] == "paradigm_anchor"]


def test_seeded_runs_reproduce_and_seeds_differ():
    def applied(seed):
        e = _engine(seed=seed)
        e._get_effect_for_condition("High anchor", "Price")
        return e._inferred_effect_log[0]["applied_d"]
    assert applied(7) == applied(7)
    assert len({applied(s) for s in range(1, 8)}) > 1          # two labs differ
    d1, d2 = _dissonance(3), _dissonance(3)
    assert (d1._get_effect_for_condition("cognitive_dissonance_induced", "attitude")
            == d2._get_effect_for_condition("cognitive_dissonance_induced", "attitude"))


# ── metadata ──────────────────────────────────────────────────────────────────

def test_metadata_reports_published_factor_and_applied():
    eng = _engine(n=200)
    meta = eng.generate()[1]["effect_sizes_applied"]
    pol = meta["inferred_effect_policy"]
    assert pol["mode"] == "replication_adjusted" and pol["shrinkage_factor"] == pytest.approx(0.60)
    assert pol["source_verified"] is False and pol["evidence_tier"] in R.RECALL_TIERS
    row = [c for c in meta["contrasts"] if c["source"] == "inferred"][0]
    assert {"published_d", "shrinkage_factor", "applied_d"} <= set(row)
    assert row["shrinkage_factor"] == pytest.approx(0.60)
    assert meta["inferred_effect_sources"]
    json.dumps(meta, allow_nan=False)                           # no NaN / inf anywhere
    assert np.isfinite(row["applied_d"])


# ── the shrunk d is what the data show ────────────────────────────────────────

def test_named_paradigm_lands_near_the_shrunk_d_not_the_published_d():
    pub = _match_meta_entry("anchoring effect on price estimates")[1]
    ds = []
    for seed in (11, 12, 13):
        eng = _engine(n=1200, seed=seed, conds=("High anchor", "Low anchor"))
        df = _frame(eng)
        a = df.loc[df["CONDITION"] == "High anchor", "Price_mean"].astype(float)
        b = df.loc[df["CONDITION"] == "Low anchor", "Price_mean"].astype(float)
        ds.append((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2))
    assert 0.35 * pub - 0.1 < np.mean(ds) < 0.85 * pub
