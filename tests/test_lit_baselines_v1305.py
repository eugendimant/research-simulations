"""Literature baselines that reach generated data (v1.3.0.5).

A second, narrower pass over the knowledge-base entries that actually drive output:

* the public goods and prisoner's dilemma calibrations were unreachable (the lookup
  keys are underscored, study text says "public goods game" / "prisoner's dilemma"),
  so a one-shot public goods game came out at a 64% mean contribution and a
  prisoner's dilemma at 70% cooperation;
* the ultimatum entry had no shares, so a Beta around 0.40 put 3% of offers on exactly
  half (the modal offer) and 14% above it;
* the prisoner's dilemma entry (SD 0.5, Bernoulli) cannot be fitted by a Beta, so on a
  slider it fell back to a generic tendency;
* 31 zero-based instruments (PSS, CD-RISC, MBI, IRI, MFQ, PCL-type 0-4 scales, the CRT,
  ...) carried their native 0-k mean with the default scale_min of 1.0.
"""
import contextlib
import io

import numpy as np
import pytest

from utils import scientific_knowledge_base as skb
from utils.enhanced_simulation_engine import EnhancedSimulationEngine

SEEDS = (1, 2, 3)


def _slider(col):
    return [{"name": col, "variable_name": col, "num_items": 1, "scale_points": 101,
             "scale_min": 0, "scale_max": 100, "reverse_items": [], "type": "slider"}]


def _draw(title, desc, col, seed, n=800):
    eng = EnhancedSimulationEngine(
        study_title=title, study_description=desc, sample_size=n,
        conditions=["Control", "Control B"], factors=[], scales=_slider(col),
        additional_vars=[], demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12},
        effect_sizes=[], seed=seed)
    eng.llm_generator.disable_permanently("test")
    with contextlib.redirect_stdout(io.StringIO()):
        df, _ = eng.generate()
    return df[col + "_1"].astype(float).to_numpy()


def _pooled(title, desc, col):
    return np.concatenate([_draw(title, desc, col, s) for s in SEEDS])


@pytest.fixture(scope="module")
def ultimatum():
    return _pooled("Ultimatum game offers",
                   "Proposers decide how much of a $100 pie to offer the responder in an ultimatum game.",
                   "Ultimatum_Offer")


@pytest.fixture(scope="module")
def public_goods():
    return _pooled("Public goods game",
                   "Participants decide how much of a $100 endowment to contribute to a group account in a public goods game.",
                   "PGG_Contribution")


@pytest.fixture(scope="module")
def prisoners_dilemma():
    return _pooled("Prisoner's dilemma",
                   "Participants choose to cooperate or defect in a one-shot prisoner's dilemma.",
                   "PD_Cooperation")


@pytest.fixture(scope="module")
def dictator():
    return _pooled("Dictator game giving",
                   "Participants decide how much of a $100 endowment to give to an anonymous partner in a dictator game.",
                   "Dictator_Giving")


# ---------------------------------------------------------------------------
# Generated game outcomes
# ---------------------------------------------------------------------------
def test_dictator_engel_2011_point_masses(dictator):
    """Engel (2011): mean 28%, ~36% give nothing, ~17% give half."""
    assert 23 <= dictator.mean() <= 33
    assert 0.30 <= (dictator == 0).mean() <= 0.41
    assert 0.13 <= (dictator == 50).mean() <= 0.21


def test_ultimatum_modal_offer_is_half_and_hyper_fair_offers_are_rare(ultimatum):
    """Oosterbeek et al. (2004): mean offer ~40%; the modal offer is exactly half
    (a quarter to a third of proposers); offers above half and below 20% are rare."""
    assert 37 <= ultimatum.mean() <= 44
    assert 0.20 <= (ultimatum == 50).mean() <= 0.38, (ultimatum == 50).mean()
    assert (ultimatum > 50).mean() <= 0.08
    assert (ultimatum < 20).mean() <= 0.10
    assert np.median(ultimatum) >= 38


def test_public_goods_game_is_routed_to_its_calibration(public_goods):
    """Zelmer (2003) pooled mean ~38%; one-shot first-round ~40-50%. Unrouted, the
    generic branch gave a 64% mean."""
    assert 33 <= public_goods.mean() <= 47, public_goods.mean()


def test_public_goods_contributions_spread_over_the_whole_endowment(public_goods):
    """VCM contributions are widely dispersed (SD about 30% of the endowment), not a
    tight cluster around the mean."""
    assert public_goods.std() >= 24, public_goods.std()
    assert (public_goods < 20).mean() >= 0.20
    assert (public_goods > 60).mean() >= 0.15


def test_prisoners_dilemma_is_routed_and_bimodal(prisoners_dilemma):
    """Sally (1995): ~47% cooperate. On a 0-100 slider the choice is two point masses
    (defect = 0, cooperate = 100); the generic branch gave a 63-70% mean."""
    assert 38 <= prisoners_dilemma.mean() <= 52, prisoners_dilemma.mean()
    ends = ((prisoners_dilemma == 0) | (prisoners_dilemma == 100)).mean()
    assert ends >= 0.95, ends


def test_binary_prisoners_dilemma_moves_toward_the_published_rate():
    """The 0/1 route only receives the mean shift (the binary path does not use the
    outcome distribution), so it is not yet at 47%, but it is well below the 70% the
    generic branch produced."""
    rates = []
    for seed in SEEDS:
        sc = [{"name": "Cooperate", "variable_name": "Cooperate", "num_items": 1,
               "scale_points": 2, "scale_min": 0, "scale_max": 1, "reverse_items": [],
               "type": "binary"}]
        eng = EnhancedSimulationEngine(
            study_title="Prisoner's dilemma",
            study_description="Participants choose to cooperate or defect in a one-shot prisoner's dilemma.",
            sample_size=800, conditions=["Control", "Control B"], factors=[], scales=sc,
            additional_vars=[], demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12},
            effect_sizes=[], seed=seed)
        eng.llm_generator.disable_permanently("test")
        with contextlib.redirect_stdout(io.StringIO()):
            df, _ = eng.generate()
        rates.append(float(df["Cooperate_1"].astype(float).mean()))
    assert float(np.mean(rates)) < 0.64, rates


def test_cooperation_likert_scale_is_not_taken_for_a_game():
    """'cooperat' alone must not route a rating scale to the game calibration: the
    study text has to name the game."""
    sc = [{"name": "Team_Cooperation", "variable_name": "Team_Cooperation", "num_items": 4,
           "scale_points": 7, "scale_min": 1, "scale_max": 7, "reverse_items": [], "type": "likert"}]
    eng = EnhancedSimulationEngine(
        study_title="Teamwork climate", study_description="Employees rate cooperation in their team.",
        sample_size=300, conditions=["A", "B"], factors=[], scales=sc, additional_vars=[],
        demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12}, effect_sizes=[], seed=1)
    eng.llm_generator.disable_permanently("test")
    with contextlib.redirect_stdout(io.StringIO()):
        df, _ = eng.generate()
    vals = df[[f"Team_Cooperation_{i}" for i in range(1, 5)]].to_numpy()
    assert vals.min() >= 1 and vals.max() <= 7
    assert 2.5 <= vals.mean() <= 6.0


# ---------------------------------------------------------------------------
# Internal consistency of the stored game entries
# ---------------------------------------------------------------------------
def _implied_mean(cal):
    from utils.enhanced_simulation_engine import _parse_subpop_band
    total = sum(cal.subpopulations.values())
    acc = 0.0
    for name, w in cal.subpopulations.items():
        lo, hi = _parse_subpop_band(name)
        acc += (w / total) * 0.5 * (lo + hi)
    return acc


@pytest.mark.parametrize("key", ["dictator_standard", "ultimatum_standard",
                                 "prisoners_dilemma_standard"])
def test_stored_shares_sum_to_one_and_imply_the_stated_mean(key):
    cal = skb.GAME_CALIBRATIONS[key]
    assert sum(cal.subpopulations.values()) == pytest.approx(1.0, abs=1e-9)
    # Within-band mass is tilted to the mean at run time, so shares may only imply a
    # mean at or below the stated one by a few points, never above it.
    assert _implied_mean(cal) <= cal.mean_proportion + 0.02
    assert _implied_mean(cal) >= cal.mean_proportion - 0.08


def test_prisoners_dilemma_shares_reproduce_a_bernoulli_sd():
    cal = skb.GAME_CALIBRATIONS["prisoners_dilemma_standard"]
    p = cal.subpopulations["cooperate_100"]
    assert p == pytest.approx(cal.mean_proportion)
    assert np.sqrt(p * (1 - p)) == pytest.approx(cal.sd_proportion, abs=0.01)


def test_public_goods_sd_is_compatible_with_a_wide_distribution():
    cal = skb.GAME_CALIBRATIONS["public_goods_standard"]
    assert 0.25 <= cal.sd_proportion <= np.sqrt(cal.mean_proportion * (1 - cal.mean_proportion))


def test_every_game_calibration_is_inside_its_own_bounds():
    for key, cal in skb.GAME_CALIBRATIONS.items():
        assert cal.sd_proportion > 0, key
        if cal.game_type in ("dictator", "trust", "ultimatum", "public_goods",
                             "prisoners_dilemma") and cal.variant != "taking":
            assert 0.0 <= cal.mean_proportion <= 1.0, key


# ---------------------------------------------------------------------------
# Construct norms: means inside instrument ranges, zero-based instruments declared
# ---------------------------------------------------------------------------
def test_every_norm_mean_lies_inside_its_instrument_range():
    for key, n in skb.CONSTRUCT_NORMS.items():
        lo, hi = n.scale_min, n.scale_min + n.scale_points - 1
        assert lo <= n.mean <= hi, (key, n.mean, lo, hi)


def test_no_norm_sits_on_the_floor_of_a_range_that_starts_below_it():
    """A mean in the bottom 5% of a 1-based range for a symptom-free instrument is the
    signature of a zero-based mean stored against scale_min 1.0."""
    for key, n in skb.CONSTRUCT_NORMS.items():
        span = n.scale_points - 1
        if n.scale_min == 1.0 and span >= 3:
            assert (n.mean - 1.0) / span >= 0.06, (key, n.mean, n.scale_points)


ZERO_BASED = {
    "perceived_stress_pss": 5, "resilience_cd_risc": 5, "pain_catastrophizing_pcs": 5,
    "burnout_emotional_exhaustion": 7, "burnout_depersonalization_mbi": 7,
    "burnout_personal_accomplishment_mbi": 7, "post_traumatic_growth_ptgi": 6,
    "moral_foundations_care": 6, "moral_foundations_fairness": 6,
    "moral_foundations_loyalty": 6, "moral_foundations_authority": 6,
    "moral_foundations_purity": 6, "empathic_concern_iri": 5, "perspective_taking_iri": 5,
    "personal_distress_iri": 5, "fantasy_iri": 5, "empathy_teq": 5,
    "eating_attitudes_edeq": 7, "cognitive_reflection_crt": 4, "flourishing_perma": 11,
    "illness_perception_bipq": 11, "work_engagement_uwes": 7, "academic_engagement_uwes_s": 7,
}


@pytest.mark.parametrize("key,points", sorted(ZERO_BASED.items()))
def test_zero_based_instruments_declare_their_origin(key, points):
    n = skb.CONSTRUCT_NORMS[key]
    assert n.scale_min == 0.0 and n.scale_points == points


def test_zero_based_norms_rescale_to_the_right_position():
    # Cohen et al. (1983): PSS-10 is 0-4, community mean ~1.4/item -> 35% of the range.
    pss = skb.get_construct_norm("perceived_stress_pss", target_scale_points=5)
    assert pss["mean"] == pytest.approx(1.0 + 4 * 1.4 / 4.0)
    # Connor & Davidson (2003): CD-RISC total 80.4 over 25 items = 3.2 on 0-4 (80%).
    cd = skb.get_construct_norm("resilience_cd_risc", target_scale_points=5)
    assert cd["mean"] == pytest.approx(1.0 + 4 * 3.2 / 4.0)
    # Frederick (2005): CRT mean 1.24 of 3 -> 41% of the range, not the floor.
    crt = skb.get_construct_norm("cognitive_reflection_crt", target_scale_points=4)
    assert crt["mean"] == pytest.approx(1.0 + 3 * 1.24 / 3.0)
    # Maslach & Jackson (1981): MBI Emotional Exhaustion ~2.3/item on 0-6.
    ee = skb.get_construct_norm("burnout_emotional_exhaustion", target_scale_points=7)
    assert ee["mean"] == pytest.approx(1.0 + 2.3)


def test_second_pass_corrections_keep_the_value_they_replaced():
    import json
    import os

    from utils import empirical_registry as R

    path = os.path.join(os.path.dirname(os.path.abspath(R.__file__)), "registry",
                        R.RECALL_AUDIT_FILE)
    recs = json.load(open(path, encoding="utf-8"))["records"]
    for key in ZERO_BASED:
        rec = recs["norm:" + key]
        if rec.get("audited_on") != "2026-10-08":
            continue
        assert rec["tier"] == "recall_corrected", key
        assert rec["corrected"]["scale_min"] == 0.0 and rec["corrected"]["scale_min_was"] == 1.0, key
    for key in ("ultimatum_standard", "public_goods_standard", "prisoners_dilemma_standard"):
        rec = recs["game:" + key]
        assert rec["audited_on"] == "2026-10-08" and rec["tier"] == "recall_corrected"
        assert any(f.endswith("_was") for f in rec["corrected"])
