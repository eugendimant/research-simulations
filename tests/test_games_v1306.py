"""Economic games: reachability, binary outcomes, restored-effect variability, baselines (v1.3.0.6).

* every GAME_CALIBRATIONS key is reachable from realistic study text (whole-phrase game
  names; "trust" or "game" alone never makes a game outcome);
* a two-option prisoner's dilemma column is drawn at the published choice rate (47%), with
  a requested Cohen's d recovered and ``auto_effects=False`` a true null;
* an effect restored after the game model varies from seed to seed like a real sample;
* ultimatum SD, beauty-contest mean, DS-R and FSQ baselines and their recall records.
"""
import contextlib
import io
import json
import os

import numpy as np
import pytest

from utils import empirical_registry as R
from utils import scientific_knowledge_base as skb
from utils.enhanced_simulation_engine import EnhancedSimulationEngine, _game_quantile_fn

# key -> (study title, description, outcome variable)
TXT = {
"dictator_standard":("Dictator game","Participants split $10 as dictators","Allocation"),
"dictator_taking":("Dictator game with taking","Dictators may give or take money from the recipient","Allocation"),
"dictator_third_party_punishment":("Dictator game","A third party observes the dictator and may punish","Allocation"),
"dictator_earned_money":("Dictator game","Dictators allocate earned money after a real effort task","Allocation"),
"dictator_deserving_receiver":("Dictator game","The recipient is a deserving poor person","Allocation"),
"dictator_ingroup":("Dictator game","Allocation to others","Ingroup_Allocation"),
"dictator_outgroup":("Dictator game","Allocation to others","Outgroup_Allocation"),
"dictator_social_distance":("Dictator game","Recipients differ in social distance","Allocation"),
"dictator_charity":("Dictator game","Dictators can give to a charity","Allocation"),
"dictator_multiple_recipients":("Dictator game","Dictators split money among multiple recipients","Allocation"),
"dictator_uncertainty":("Dictator game","Dictators face uncertain payoffs","Allocation"),
"trust_standard":("Trust game","Investors send money to a trustee","Amount_Sent"),
"trust_with_communication":("Trust game","Investors can chat with the trustee before sending","Amount_Sent"),
"trust_with_reputation":("Trust game","The trustee has a reputation score","Amount_Sent"),
"trust_with_punishment":("Trust game","The investor can punish the trustee","Amount_Sent"),
"trust_cross_cultural":("Trust game","A cross-cultural sample plays the trust game","Amount_Sent"),
"trust_binary":("Trust game","Investors send all or nothing","Sent_All"),
"trust_with_risk_info":("Trust game","Investors see risk information about the trustee","Amount_Sent"),
"trust_repeated":("Trust game","A repeated trust game over ten rounds","Amount_Sent"),
"trust_with_inequality":("Trust game","Players start with unequal inequality endowments","Amount_Sent"),
"ultimatum_standard":("Ultimatum game","Proposers offer a share of $10","Offer"),
"ultimatum_alternative_offers":("Ultimatum game","Responders have an outside option","Offer"),
"ultimatum_costly_rejection":("Ultimatum game","Rejection is costly: costly rejection","Offer"),
"ultimatum_third_party":("Ultimatum game","A third party receives the leftover","Offer"),
"mini_ultimatum":("Mini ultimatum game","Proposers choose between two offers","Offer"),
"ultimatum_with_delay":("Ultimatum game","Responders face a delay before answering","Offer"),
"ultimatum_information_asymmetry":("Ultimatum game","Proposers have private information about the pie","Offer"),
"ultimatum_multi_round":("Ultimatum game","Played repeated across rounds","Offer"),
"ultimatum_with_communication":("Ultimatum game","Players communicate before the offer","Offer"),
"public_goods_standard":("Public goods game","Contribute to a group account","Contribution"),
"public_goods_with_punishment":("Public goods game","With punishment of free riders","Contribution"),
"public_goods_peer_punishment":("Public goods game","Peer punishment after contributions","Contribution"),
"public_goods_with_reward":("Public goods game","Players may reward others","Contribution"),
"public_goods_threshold":("Public goods game","Provision point threshold","Contribution"),
"public_goods_step_level":("Public goods game","A step level public good","Contribution"),
"public_goods_with_communication":("Public goods game","Group communication is allowed","Contribution"),
"public_goods_with_leadership":("Public goods game","A leader contributes first","Contribution"),
"public_goods_repeated_decay":("Public goods game","Repeated for ten rounds","Contribution"),
"public_goods_with_inequality":("Public goods game","Heterogeneous endowments create inequality","Contribution"),
"prisoners_dilemma_standard":("Prisoner's dilemma","One-shot","Cooperate"),
"prisoners_dilemma_with_punishment":("Prisoner's dilemma","With punishment","Cooperate"),
"prisoners_dilemma_iterated_axelrod":("Iterated prisoner's dilemma","Axelrod tournament","Cooperate"),
"prisoners_dilemma_with_reputation":("Prisoner's dilemma","Players see reputation","Cooperate"),
"prisoners_dilemma_exit_option":("Prisoner's dilemma","Players have an exit option","Cooperate"),
"prisoners_dilemma_multiplayer":("Multiplayer prisoner's dilemma","Groups of five","Cooperate"),
"prisoners_dilemma_asymmetric":("Prisoner's dilemma","Asymmetric payoffs","Cooperate"),
"prisoners_dilemma_costly_signaling":("Prisoner's dilemma","Costly signaling before play","Cooperate"),
"first_price_auction_standard":("First-price auction","Sealed bid","Bid"),
"second_price_auction_standard":("Second-price auction","Vickrey","Bid"),
"all_pay_auction_standard":("All-pay auction","Everyone pays","Bid"),
"nash_bargaining_standard":("Bargaining game","Nash bargaining over a pie","Demand"),
"gift_exchange_standard":("Gift exchange game","Workers choose effort after a wage","Effort"),
"stag_hunt_standard":("Stag hunt","Coordination","Choice"),
"stag_hunt_with_communication":("Stag hunt","Players use cheap talk communication","Choice"),
"common_pool_resource_standard":("Common pool resource game","Harvesting","Extraction"),
"common_pool_resource_with_communication":("Common pool resource game","Communication allowed","Extraction"),
"tragedy_of_commons_standard":("Tragedy of the commons game","Herders","Extraction"),
"holt_laury_standard":("Holt-Laury risk task","Multiple price list","Safe_Choices"),
"beauty_contest_standard":("Beauty contest","Guess two thirds of the average","Guess"),
"beauty_contest_iterated":("Beauty contest","Repeated guessing game over rounds","Guess"),
"die_roll_honesty":("Die roll honesty task","Participants report a die roll","Reported"),
"battle_of_sexes_standard":("Battle of the sexes","Coordination","Choice"),
"chicken_hawk_dove_standard":("Hawk-dove game","Conflict","Choice"),
"bertrand_competition_standard":("Bertrand competition","Price setting","Price"),
"cournot_competition_standard":("Cournot competition","Quantity setting","Quantity"),
"centipede_standard":("Centipede game","Pass or take","Pass_Decision"),
"market_entry_standard":("Market entry game","Enter or stay out","Entry_Decision"),
"volunteer_dilemma_standard":("Volunteer's dilemma","Someone must volunteer","Volunteer_Decision"),
}


def _engine(title, desc, var, n=10, scale_min=0, scale_max=100, points=101, effects=None,
            seed=1, conds=("Group A", "Group B"), **kw):
    sc = [{"name": var, "variable_name": var, "num_items": 1, "scale_points": points,
           "scale_min": scale_min, "scale_max": scale_max, "reverse_items": [], "type": "slider"}]
    eng = EnhancedSimulationEngine(
        study_title=title, study_description=desc, sample_size=n, conditions=list(conds),
        factors=[], scales=sc, additional_vars=[],
        demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12},
        effect_sizes=effects or [], seed=seed, **kw)
    eng.llm_generator.disable_permanently("test")
    return eng


def _run(eng):
    with contextlib.redirect_stdout(io.StringIO()):
        return eng.generate()


def test_every_game_key_has_a_reachability_case():
    assert set(TXT) == set(skb.GAME_CALIBRATIONS)


@pytest.mark.parametrize("key", sorted(TXT))
def test_every_game_calibration_is_reachable_from_study_text(key):
    title, desc, var = TXT[key]
    cal = skb.GAME_CALIBRATIONS[key]
    assert skb.resolve_game_calibration_key(var, title, desc) == key
    got = _engine(title, desc, var)._get_domain_response_calibration(var, "Group A")
    assert got.get("_game_variant") == f"{cal.game_type}_{cal.variant}"
    assert got["_kb_dist"]["game"] == cal.game_type


@pytest.mark.parametrize("title,desc,var", [
    ("Trust in institutions", "A survey about trust in the government and a game of thrones fan club.", "Trust_in_Government"),
    ("Trust game", "Participants rate how trustworthy the partner seemed afterwards.", "Trust_Rating_Scale"),
    ("Gaming habits", "How many hours of video games do students play, and do they trust game studios?", "Hours_Played"),
    ("Consent and attitudes", "A short attitudes survey.", "Consent_Check"),
    ("Community survey", "People report their trust in neighbours; some enjoy a neighbourhood game night.", "Neighbour_Trust"),
])
def test_a_survey_that_merely_mentions_trust_or_game_is_not_a_game_outcome(title, desc, var):
    eng = _engine(title, desc, var)
    cal = eng._get_domain_response_calibration(var, "Group A")
    assert "_kb_dist" not in cal and "_game_variant" not in cal
    assert cal["mean_adjustment"] == pytest.approx(0.0, abs=0.06)


def test_phrases_alone_do_not_name_a_game():
    for text in ("trust", "game", "a trust fall exercise", "prisoners of war memoir", "online shopping auction house",
                 "bargaining power of labour unions", "chicken farming", "beauty and personal care"):
        assert skb.detect_game_type("x", text, text)[0] is None, text


def test_variant_comes_from_study_text_not_condition_labels_and_negation_is_respected():
    key = skb.resolve_game_calibration_key
    assert key("Contribution", "Public goods game", "no punishment stage") == "public_goods_standard"
    assert key("Contribution", "Public goods game", "with punishment of free riders") == "public_goods_with_punishment"
    # a condition called "Punishment" does not move the baseline of the whole design
    assert key("Contribution", "Public goods game", "", "Punishment, Control") == "public_goods_standard"
    # one-shot designs never select a repeated variant
    assert key("Contribution", "Public goods game", "one-shot, not repeated") == "public_goods_standard"


# ---------------------------------------------------------------------------
# Two-option prisoner's dilemma
# ---------------------------------------------------------------------------
PD = ("Prisoner's dilemma game",
      "Participants play a one-shot prisoner's dilemma and choose to cooperate (1) or defect (0).", "Cooperate")


def _pd(seed, d=None, n=600, **kw):
    eff = [{"variable": "Cooperate", "factor": "Cond", "level_high": "Group B", "level_low": "Group A",
            "cohens_d": d, "direction": "positive"}] if d else []
    eng = _engine(*PD, n=n, scale_min=0, scale_max=1, points=2, effects=eff, seed=seed, **kw)
    df, _ = _run(eng)
    return df


def test_binary_prisoners_dilemma_uses_the_published_cooperation_rate():
    rates = [_pd(s)["Cooperate_1"].mean() for s in (1, 2, 3, 4)]
    assert set(np.unique(_pd(1)["Cooperate_1"])) <= {0, 1}
    # Sally (1995): 47%; Dal Bo & Frechette (2018) one-shot/stranger first rounds 40-50%. It was 57-60%.
    assert 0.42 <= float(np.mean(rates)) <= 0.52


def test_binary_prisoners_dilemma_recovers_a_requested_effect():
    ds = []
    for s in range(1, 9):
        df = _pd(s, d=0.5)
        a = df[df.CONDITION == "Group A"]["Cooperate_1"].to_numpy(float)
        b = df[df.CONDITION == "Group B"]["Cooperate_1"].to_numpy(float)
        ds.append((b.mean() - a.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
    assert 0.9 * 0.5 <= float(np.mean(ds)) <= 1.12 * 0.5


def test_binary_prisoners_dilemma_is_a_true_null_without_inferred_effects():
    gaps = []
    for s in (1, 2, 3, 4, 5, 6):
        df = _pd(s, auto_effects=False)
        gaps.append(df[df.CONDITION == "Group B"]["Cooperate_1"].mean()
                    - df[df.CONDITION == "Group A"]["Cooperate_1"].mean())
    assert abs(float(np.mean(gaps))) < 0.04


def test_binary_stag_hunt_uses_its_stored_rate():
    eng = _engine("Stag hunt", "Players choose stag or hare in a stag hunt coordination game.", "Stag_Choice",
                  n=1200, scale_min=0, scale_max=1, points=2)
    df, _ = _run(eng)
    assert 0.54 <= df["Stag_Choice_1"].mean() <= 0.66   # Battalio et al. 2001 entry: 0.60


# ---------------------------------------------------------------------------
# Restored game effects keep their sampling variation
# ---------------------------------------------------------------------------
def _restored_d(seed, d, n=600):
    eff = [{"variable": "Give", "factor": "Cond", "level_high": "Group B", "level_low": "Group A",
            "cohens_d": d, "direction": "positive"}]
    eng = _engine("Dictator game", "Dictators split $100 between themselves and a recipient", "Give",
                  n=n, effects=eff, seed=seed, use_socsim_experimental=True)
    df, meta = _run(eng)
    assert meta["socsim"]["user_effects_reapplied"]
    a = df[df.CONDITION == "Group A"]["Give_1"].to_numpy(float)
    b = df[df.CONDITION == "Group B"]["Give_1"].to_numpy(float)
    return (b.mean() - a.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)


def test_restored_game_effect_varies_between_seeds_like_a_real_sample():
    ds = np.array([_restored_d(s, 0.5) for s in range(1, 13)])
    assert 0.9 * 0.5 <= ds.mean() <= 1.1 * 0.5            # still honours the request on average
    assert 0.06 <= ds.std(ddof=1) <= 0.13                   # was ~0.01; sqrt(1/300+1/300) = 0.082


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def _realised(game, variant):
    cal = skb.GAME_CALIBRATIONS[f"{game}_{variant}"]
    _game_quantile_fn.__globals__["_GAME_QFN_CACHE"].pop((cal.game_type, cal.variant), None)
    fn = _game_quantile_fn({"game": cal.game_type, "variant": cal.variant, "mean": cal.mean_proportion,
                            "sd": cal.sd_proportion, "subpops": dict(cal.subpopulations or {})})
    return np.array([fn(q) for q in np.linspace(0.0005, 0.9995, 4000)])


def test_ultimatum_offers_disperse_as_recalled():
    x = _realised("ultimatum", "standard")
    assert skb.GAME_CALIBRATIONS["ultimatum_standard"].sd_proportion == pytest.approx(0.13)
    assert 0.39 <= x.mean() <= 0.41
    assert 0.12 <= x.std() <= 0.15                      # was 0.115
    assert 0.24 <= (x == 0.5).mean() <= 0.32            # the modal offer stays exactly half


def test_beauty_contest_mean_is_an_observed_first_round_mean_not_the_level_one_point():
    cal = skb.GAME_CALIBRATIONS["beauty_contest_standard"]
    assert cal.mean_proportion == pytest.approx(0.36)   # Nagel (1995), p = 2/3: ~36 of 100
    assert cal.mean_proportion > 50 * (2 / 3) / 100     # above the level-1 point 33.3
    assert cal.sd_proportion == pytest.approx(0.20)
    assert cal.moderators["p_target"]["1/2"] < cal.mean_proportion < cal.moderators["p_target"]["4/3"]
    x = _realised("beauty_contest", "standard")
    assert 0.34 <= x.mean() <= 0.38


def test_disgust_and_phobia_norms_are_zero_based():
    dsr = skb.CONSTRUCT_NORMS["disgust_sensitivity_dsr"]
    assert dsr.scale_min == 0.0 and dsr.scale_points == 5 and 1.4 <= dsr.mean <= 2.0
    fsq = skb.CONSTRUCT_NORMS["specific_phobia_fsq"]
    assert fsq.scale_min == 0.0 and fsq.scale_points == 8
    # on a 1-7 item the FSQ mean now sits at 26% of the range, not at 10% near the floor
    assert skb.get_construct_norm("specific_phobia_fsq", 7)["mean"] == pytest.approx(1 + 6 * 1.8 / 7)
    assert skb.get_construct_norm("disgust_sensitivity_dsr", 5)["mean"] == pytest.approx(1 + 4 * dsr.mean / 4)


def test_corrected_values_keep_the_old_value_and_are_never_verified():
    path = os.path.join(os.path.dirname(os.path.abspath(R.__file__)), "registry", R.RECALL_AUDIT_FILE)
    recs = json.load(open(path, encoding="utf-8"))["records"]
    expect = {
        "game:ultimatum_standard": ("sd_proportion", 0.1),
        "game:beauty_contest_standard": ("mean_proportion", 0.33),
        "norm:disgust_sensitivity_dsr": ("mean", 2.6),
        "norm:specific_phobia_fsq": ("scale_min", 1.0),
    }
    for key, (fld, was) in expect.items():
        rec = recs[key]
        assert rec["tier"] == "recall_corrected" and rec["audited_on"] == "2026-10-08"
        assert rec["corrected"][fld + "_was"] == was
        assert R.status_of(*key.split(":")) == "recall_corrected"
        assert R.status_of(*key.split(":")) not in ("verified", "measured")
