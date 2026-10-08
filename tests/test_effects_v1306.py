"""v1.3.0.6 effect calibration: alpha-aware lone-scale route, consistent inferred effects, one discount.

* A lone scale of three or more items has its effect built into the finished items (after the
  reliability steps), so the 4-item 5-point block no longer loses ~12% to alpha attenuation.
* Inferred effects (keyword rules, curated paradigm, literature fallback) are built in the same way
  and in the same currency, so the applied d is what the composite shows.
* Tier weight and replication shrinkage are one discount (the stronger), not a product.
* Generic keyword effects take the recalled replication shrinkage; intergroup effects do not.
"""

import numpy as np
import pytest

from utils import empirical_registry as reg
from utils.enhanced_simulation_engine import EnhancedSimulationEngine, EffectSizeSpec

HI, LO = "Group 1", "Group 2"


def _engine(conds, items=4, smax=5, spec_d=None, n=1200, seed=501, title="A study of participants",
            name="DV", auto_effects=True, extra_scales=0):
    scales = [{"name": name, "variable_name": name, "type": "likert", "num_items": items, "scale_points": smax,
               "scale_min": 1, "scale_max": smax, "reverse_items": []}]
    for j in range(extra_scales):
        scales.append({"name": f"Other{j}", "variable_name": f"Other{j}", "type": "likert", "num_items": 3,
                       "scale_points": 7, "scale_min": 1, "scale_max": 7, "reverse_items": []})
    spec = [EffectSizeSpec(variable=name, factor="CONDITION", level_high=conds[0], level_low=conds[1],
                           cohens_d=spec_d, direction="positive")] if spec_d else []
    e = EnhancedSimulationEngine(
        study_title=title, study_description=title, sample_size=n, conditions=list(conds), factors=[],
        scales=scales, additional_vars=[], demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12},
        open_ended_questions=[], effect_sizes=spec, seed=seed, auto_effects=auto_effects)
    e.llm_generator.disable_permanently("test")
    return e


def _d(df, c1, c2, name="DV", items=4):
    col = f"{name}_mean" if f"{name}_mean" in df.columns else f"{name}_1"
    a = df.loc[df.CONDITION == c1, col].astype(float)
    b = df.loc[df.CONDITION == c2, col].astype(float)
    return float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2))


def _nominal(e, c1, c2):
    ap = e._applied_effects
    i1 = next(v for (c, _), v in ap.items() if c == c1)
    i2 = next(v for (c, _), v in ap.items() if c == c2)
    return (i1["offset"] - i2["offset"]) / (2.0 * i1["unit"])


# ---------------------------------------------------------------- lone-scale route

def test_lone_block_of_three_or_more_items_is_built_in_after_the_reliability_steps():
    e = _engine([HI, LO], items=4, spec_d=0.5, n=200)
    e.generate()
    assert "DV" in e._deferred_effect_vars and e._deferred_effect_log
    e1 = _engine([HI, LO], items=1, smax=7, spec_d=0.5, n=200)
    e1.generate()
    assert not e1._deferred_effect_vars        # one- and two-item scales keep the in-generator calibration


def test_four_item_five_point_block_recovers_the_request():
    ratios = []
    for seed in (601, 602, 603, 604, 605, 606):
        e = _engine([HI, LO], items=4, smax=5, spec_d=0.5, seed=seed)
        df, _ = e.generate()
        ratios.append(_d(df, HI, LO) / 0.5)
    assert 0.92 <= float(np.mean(ratios)) <= 1.12, np.round(ratios, 2)    # was 0.88 before the change


def test_auto_effects_false_is_still_a_true_null_on_a_lone_block():
    e = _engine(["Gain frame", "Loss frame"], items=4, spec_d=None, auto_effects=False, n=800)
    df, meta = e.generate()
    assert not e._deferred_effect_vars
    assert abs(_d(df, "Gain frame", "Loss frame")) < 0.25


def test_same_seed_same_data_on_the_deferred_route():
    a, _ = _engine([HI, LO], spec_d=0.5, n=300, seed=77).generate()
    b, _ = _engine([HI, LO], spec_d=0.5, n=300, seed=77).generate()
    assert a.equals(b)


# ---------------------------------------------------------------- inferred effects

def test_inferred_keyword_effect_is_delivered_on_the_composite():
    obs, nom = [], []
    for seed in (611, 612, 613, 614, 615, 616):
        e = _engine(["Positive review", "Negative review"], items=4, smax=7, seed=seed,
                    title="Online reviews and brand attitude", name="Attitude")
        df, _ = e.generate()
        assert "Attitude" in e._deferred_effect_vars
        obs.append(_d(df, "Positive review", "Negative review", name="Attitude"))
        nom.append(_nominal(e, "Positive review", "Negative review"))
    assert float(np.mean(nom)) > 0.2
    assert 0.8 <= float(np.mean(obs)) / float(np.mean(nom)) <= 1.2, (np.mean(obs), np.mean(nom))


def test_inferred_effect_next_to_other_scales_is_delivered_too():
    obs, nom = [], []
    for seed in (621, 622, 623, 624):
        e = _engine(["Positive review", "Negative review"], items=4, smax=7, seed=seed, extra_scales=1,
                    title="Online reviews and brand attitude", name="Attitude")
        df, _ = e.generate()
        obs.append(_d(df, "Positive review", "Negative review", name="Attitude"))
        nom.append(_nominal(e, "Positive review", "Negative review"))
    assert 0.75 <= float(np.mean(obs)) / float(np.mean(nom)) <= 1.25, (np.mean(obs), np.mean(nom))


def test_literature_routes_apply_the_gap_factor_to_every_match(monkeypatch):
    """The content-matched fallback used to shift one arm by 0.109 x d (about d/2 realised)."""
    e = _engine(["Some paradigm", "Control"], items=4, smax=7, n=50)
    e._scale_effect_meta = {"DV": (1, 1, 7)}

    class Hit:
        effect_d, published_d, key, source, status, shrinkage, rule, polarity_aware = 0.30, 0.5, "k", "s", "recall", 0.6, "", False

        def as_dict(self):
            return {}

    got = e._literature_match_to_shift("Some paradigm", "DV", Hit(), "test")
    assert got == pytest.approx(0.30 * 0.109 * 2.0 * e._explicit_effect_scale("DV"))


# ---------------------------------------------------------------- one discount

def test_tier_weight_and_shrinkage_are_not_multiplied():
    pol = reg.EffectPolicy(heterogeneity_draw=False)
    f = reg.policy_factor(pol)
    w = reg.confidence_weight("meta", "no_such_entry_for_this_test")
    assert w < 1.0
    got = reg.adjust_effect(0.5, kind="meta", key="no_such_entry_for_this_test", policy=pol)
    assert got == pytest.approx(0.5 * min(w, f))
    assert got > 0.5 * w * f - 1e-9 and got >= 0.5 * pol.min_retained - 1e-9


def test_an_unverified_number_never_pushes_harder_than_a_verified_one():
    pol = reg.EffectPolicy(heterogeneity_draw=False)
    unverified = reg.adjust_effect(0.5, kind="meta", key="no_such_entry_for_this_test", policy=pol)
    verified = reg.adjust_effect(0.5, policy=pol)          # no key: shrinkage alone
    assert unverified <= verified + 1e-12
    assert verified == pytest.approx(0.5 * reg.policy_factor(pol))


def test_as_published_keeps_the_tier_weight_alone():
    pol = reg.EffectPolicy(mode="as_published", heterogeneity_draw=False)
    w = reg.confidence_weight("meta", "no_such_entry_for_this_test")
    assert reg.adjust_effect(0.5, kind="meta", key="no_such_entry_for_this_test", policy=pol) == pytest.approx(0.5 * w)


# ---------------------------------------------------------------- keyword shrinkage

def _auto(e, cond, dv="Attitude"):
    return e._get_automatic_condition_effect(cond, dv)


def test_generic_keyword_effects_take_the_replication_shrinkage():
    shrunk = _engine(["Positive review", "Negative review"], items=1, smax=7, title="Reviews", name="Attitude", n=50)
    raw = _engine(["Positive review", "Negative review"], items=1, smax=7, title="Reviews", name="Attitude", n=50)
    raw._INFERRED_EFFECT_POLICY = reg.EffectPolicy(mode="as_published", heterogeneity_draw=False)
    f = reg.policy_factor(None)
    assert f < 1.0
    for cond in ("Positive review", "Negative review"):
        assert _auto(shrunk, cond) == pytest.approx(_auto(raw, cond) * f, rel=1e-9)
    assert abs(_auto(shrunk, "Positive review")) > 0


def test_intergroup_effects_are_not_shrunk():
    kw = dict(items=1, smax=7, title="Political partisan dictator game", name="Attitude", n=50)
    a = _engine(["Trump lover", "Trump hater"], **kw)
    b = _engine(["Trump lover", "Trump hater"], **kw)
    b._INFERRED_EFFECT_POLICY = reg.EffectPolicy(mode="as_published", heterogeneity_draw=False)
    for cond, sign in (("Trump lover", 1), ("Trump hater", -1)):
        assert sign * _auto(a, cond) > 0
        assert _auto(a, cond) == pytest.approx(_auto(b, cond))
