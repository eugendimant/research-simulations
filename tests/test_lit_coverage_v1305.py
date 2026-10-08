"""Paradigm coverage (v1.3.0.5): more published paradigms reachable by default, none claimed verified.

What is pinned here:

* every paradigm added in `utils/paradigm_coverage.py` has a recall-band provenance record, no entry
  claims a source was read, and no count / heterogeneity figure is invented;
* realistic labels and study titles resolve to the intended entry with the expected sign and a size
  inside the entry's own interval;
* unrelated, negated and direction-reversed labels resolve to nothing (a wrong effect is worse than none);
* the reference arm never takes a literature effect, an explicit user effect is untouched, and
  economic-game designs are left to the game models.

Values are read from the knowledge base, not frozen: recalibrating an entry is not a matching bug.
"""
import io
import contextlib

import pytest

from utils import empirical_registry as reg
from utils import literature_effects as LE
from utils import paradigm_coverage as PC
from utils import scientific_knowledge_base as skb
from utils.enhanced_simulation_engine import (
    EffectSizeSpec,
    EnhancedSimulationEngine,
    _match_meta_effect,
)

DEMO = {"gender_quota": 50, "age_mean": 35, "age_sd": 12}
NEW = sorted(PC.PARADIGM_ENTRIES)


def _engine(conds, dv="Attitude", title="A study", specs=(), desc="", scale_points=7):
    sc = [{"name": dv, "variable_name": dv.replace(" ", "_"), "type": "likert", "num_items": 1,
           "scale_points": scale_points, "scale_min": 1, "scale_max": scale_points, "reverse_items": []}]
    with contextlib.redirect_stdout(io.StringIO()):
        e = EnhancedSimulationEngine(
            study_title=title, study_description=desc, sample_size=60, conditions=list(conds), factors=[],
            scales=sc, additional_vars=[], demographics=dict(DEMO), open_ended_questions=[],
            effect_sizes=list(specs), seed=1, auto_effects=True)
        e.llm_generator.disable_permanently("test")
    return e


def _shift_in_d(engine, label, dv="Attitude"):
    """Mean shift of one arm, in units of the engine's own Cohen's-d currency."""
    var = dv.replace(" ", "_")
    unit = engine._EFFECT_D_TO_NORMALIZED * engine._explicit_effect_scale(var)
    return engine._get_effect_for_condition(label, var) / unit


# --------------------------------------------------------------------------- provenance

def test_every_new_entry_is_in_the_knowledge_base_and_the_recall_band():
    assert len(NEW) >= 15
    for key in NEW:
        assert key in skb.META_ANALYTIC_DB, key
        status = reg.status_of("meta", key)
        assert status in reg.RECALL_TIERS, (key, status)
        rec = reg.provenance_of("meta", key)
        assert rec is not None
        assert rec.note.startswith("RECALL, NOT SOURCE-VERIFIED"), key
        assert not rec.doi and not rec.url and not rec.quote and not rec.verified_on, key
        assert reg.confidence_weight("meta", key) < reg.TIER_WEIGHT[reg.CITED_UNCHECKED], key


def test_no_new_entry_claims_a_verified_tier_or_moves_the_sourced_count():
    sourced_tiers = {reg.VERIFIED, reg.CORRECTED, reg.PARTIAL, reg.MEASURED}
    for key in NEW:
        assert reg.status_of("meta", key) not in sourced_tiers, key
    expected = sum(1 for k, p in reg.PROVENANCE.items()
                   if p.status in {reg.VERIFIED, reg.CORRECTED, reg.PARTIAL} and k.split(":", 1)[1] not in NEW)
    assert reg.coverage_summary()["sourced_entries"] == expected


def test_the_recall_file_covers_exactly_the_new_entries():
    import json
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(reg.__file__)), "registry", reg.RECALL_COVERAGE_FILE)
    blob = json.load(open(path, encoding="utf-8"))
    assert {k.split(":", 1)[1] for k in blob["records"]} == set(NEW)
    for key, rec in blob["records"].items():
        assert key.startswith("meta:")
        assert rec["tier"] in reg.RECALL_TIERS and rec["note"].strip()


def test_nothing_is_invented_for_heterogeneity_or_counts():
    for key in NEW:
        e = skb.META_ANALYTIC_DB[key]
        assert e.heterogeneity_tau == 0.0 and e.i_squared == 0.0, key      # not recorded, not guessed
        assert e.n_participants == 0, key                                  # no pooled N is claimed
        lo, hi = e.ci_95
        assert lo <= e.effect_d <= hi, key
        assert hi - lo >= 0.10, f"{key}: interval too narrow for a recalled value"
        assert abs(e.effect_d) <= 0.5, f"{key}: a recalled value should be conservative"
        assert e.source and e.construct and e.paradigm and e.notes


def test_every_new_entry_is_reachable_and_every_rule_points_at_a_real_entry():
    assert PC.entries_without_rules() == []
    for rule in PC.LABEL_RULES:
        assert rule.key in skb.META_ANALYTIC_DB, rule.key
    for key in PC.ENTRY_ALIASES:
        assert key in skb.META_ANALYTIC_DB, key
    for key in NEW:
        assert any(r.key == key for r in PC.LABEL_RULES), key


def test_new_entries_do_not_join_the_open_ended_content_ranking():
    idx = LE._index()
    assert idx is not None
    assert not (set(NEW) & set(idx.doc_tokens))


# --------------------------------------------------------------------------- labels -> entries

LABEL_CASES = [
    # label, entry, sign of the applied effect on a positive-type DV
    ("Nudge condition", "nudge_general_meta", +1),
    ("Descriptive norm", "social_norm_message_meta", +1),
    ("injunctive_norm_message", "social_norm_message_meta", +1),
    ("Identifiable victim", "identifiable_victim_effect", +1),
    ("Anthropomorphic robot", "anthropomorphism_meta", +1),
    ("Mortality salience", "mortality_salience_meta", +1),
    ("Credible source", "source_credibility_meta", +1),
    ("Sponsorship disclosure", "sponsorship_disclosure_meta", -1),
    ("Ostracized", "social_exclusion_ostracism_meta", -1),
    ("Social exclusion", "social_exclusion_ostracism_meta", -1),
    ("Expressive writing", "expressive_writing_meta", +1),
    ("Gratitude journal", "gratitude_intervention_meta", +1),
    ("Three good things", "gratitude_intervention_meta", +1),
    ("Positive psychology intervention", "positive_psychology_intervention_meta", +1),
    ("Cognitive reappraisal", "cognitive_reappraisal_meta", +1),
    ("Active learning", "active_learning_stem_meta", +1),
    ("Performance feedback", "feedback_intervention_meta", +1),
    ("Tailored health message", "tailored_health_messages_meta", +1),
    ("Graphic warning label", "graphic_warning_labels_meta", +1),
    ("Illusory truth: repeated statements", "illusory_truth_meta", +1),
    ("Gamified", "gamification_meta", +1),
    ("imagined_contact", "imagined_contact_meta", +1),
    # entries that already existed but could not be reached from a label
    ("Limited time offer", "scarcity_effect_meta", +1),
    ("Free shipping", "zero_price_effect", +1),
    ("Personalized message", "personalization_meta", +1),
    ("Foot-in-the-door", "foot_in_door_meta", +1),
    ("Stereotype threat", "stereotype_threat", -1),
    ("Bestseller badge", "social_proof_marketing_meta", +1),
    ("sustainable_default", "green_defaults_meta", +1),
    ("Mindfulness training", "mindfulness_intervention_meta", +1),
]


@pytest.mark.parametrize("label,key,sign", LABEL_CASES)
def test_label_resolves_to_the_entry_with_the_expected_sign_and_a_size_inside_its_interval(label, key, sign):
    rule = PC.match_label_rule(label)
    assert rule is not None and rule.key == key, (label, rule)
    import random
    hit = LE._rule_lookup(label, rng=random.Random(0))
    assert hit is not None and hit.key == key
    entry = skb.META_ANALYTIC_DB[key]
    assert (hit.published_d > 0) == (sign > 0), (label, hit.published_d)
    lo, hi = sorted((entry.ci_95[0] * rule.sign, entry.ci_95[1] * rule.sign))
    assert lo - 1e-9 <= hit.published_d <= hi + 1e-9
    # the tier damping never grows an effect and never flips its sign
    assert (hit.effect_d > 0) == (hit.published_d > 0)
    if entry.heterogeneity_tau == 0.0:      # an older entry with a recorded tau draws its own effect
        assert 0 < abs(hit.effect_d) <= abs(hit.published_d) + 1e-12


@pytest.mark.parametrize("label", [
    "Group A", "Treatment 1", "Condition 2", "Block SS", "T1", "c3", "Immigration", "Female High Performance",
    "Hedonic - Care Focused", "Receptive (I) x Unreceptive (O)", "Dictator game", "who loves Trump",
    "Red Bull Prosocial", "Original Website", "Brand+Nickname", "Message 2 Emoji",
    # negated / reference / direction-reversed forms of a recognised paradigm
    "No gamified tier", "Without nudge", "Non-credible source", "Low credibility source", "Anti-Social Norm",
    "Dishonest norm", "Unsustainable default", "No social exclusion", "Zero gratitude",
    # a recognised word inside a longer, different word
    "Nudgeless", "Gratitudes survey", "Meditations in ethics class",
])
def test_unrelated_negated_or_reversed_labels_resolve_to_nothing(label):
    assert PC.match_label_rule(label) is None
    assert LE.lookup_curated(label) is None


def test_two_different_entries_with_the_same_phrase_length_are_ambiguous():
    # "social norm" (norm entry) and "social exclusion" (ostracism entry) share a first word only
    assert PC.match_label_rule("Social norm") is not None
    assert PC.match_label_rule("Social exclusion").key == "social_exclusion_ostracism_meta"


# --------------------------------------------------------------------------- study text -> entries

def _expected_anchor(key):
    e = skb.META_ANALYTIC_DB[key]
    d = abs(e.effect_d)
    if key in PC.RULE_ONLY_KEYS:
        d *= reg.confidence_weight("meta", key)
    return d


TEXT_CASES = [
    ("Nudging healthy food choices", "nudge_general_meta"),
    ("Descriptive norms and recycling", "social_norm_message_meta"),
    ("Mortality salience and worldview defense", "mortality_salience_meta"),
    ("Gratitude intervention for students", "gratitude_intervention_meta"),
    ("Active learning versus lecture", "active_learning_stem_meta"),
    ("Gamification in a coffee shop loyalty app", "gamification_meta"),
    ("Illusory truth and repeated claims", "illusory_truth_meta"),
    ("Imagined contact and prejudice", "imagined_contact_meta"),
    ("Cognitive reappraisal and negative emotion", "cognitive_reappraisal_meta"),
    ("Limited time scarcity message", "scarcity_effect_meta"),
    ("loss_frame vs gain_frame", "framing_general_meta"),
    ("Foot-in-the-door and compliance", "foot_in_door_meta"),
    ("Personalized messages and response rates", "personalization_meta"),
    ("Halo effect of physical attractiveness", "halo_effect_meta"),
    ("Green default electricity tariff", "green_defaults_meta"),
]


@pytest.mark.parametrize("text,key", TEXT_CASES)
def test_study_text_anchors_to_the_intended_entry(text, key):
    assert _match_meta_effect(text) == pytest.approx(_expected_anchor(key), abs=1e-9), text


def test_snake_case_and_hyphenated_text_read_as_phrases():
    assert _match_meta_effect("loss_frame") == _match_meta_effect("loss frame")
    assert _match_meta_effect("foot-in-the-door") == _match_meta_effect("foot in the door")
    assert _match_meta_effect("opt_out_default") == _match_meta_effect("opt-out default")


def test_social_proof_is_the_marketing_effect_unless_the_text_is_about_conformity():
    marketing = skb.META_ANALYTIC_DB["social_proof_marketing_meta"].effect_d
    assert _match_meta_effect("Social proof in online shops") == pytest.approx(marketing)
    assert _match_meta_effect("Social proof marketing study") == pytest.approx(marketing)
    # the Asch vocabulary keeps the older conformity reading
    assert _match_meta_effect("Social proof and conformity in the Asch paradigm") == pytest.approx(
        skb.META_ANALYTIC_DB["social_proof_meta"].effect_d)


@pytest.mark.parametrize("text", [
    "a survey about everyday things option a option b",
    "participants choose between free will and determinism",
    "the company spokesperson read a statement about the merger",
    "the most popular baby names of the decade",
    "a diary of daily mood and sleep",
    "meditations on ethics, a reading task",
    "students rated how much feedback they wanted",
    "personal medicine and family history",
])
def test_unrelated_study_text_matches_nothing(text):
    assert _match_meta_effect(text) is None


# --------------------------------------------------------------------------- engine behaviour

def test_the_reference_arm_never_takes_a_literature_effect():
    e = _engine(["Gamified", "Control"])
    assert _shift_in_d(e, "Control") == 0.0
    assert _shift_in_d(e, "Gamified") > 0.0
    e2 = _engine(["Mortality salience", "Baseline"], title="Study")
    assert _shift_in_d(e2, "Baseline") == 0.0


def test_polarity_flips_the_sign_for_symptom_type_outcomes():
    pos = _engine(["Gratitude journal", "Neutral journal"], dv="Well-being")
    neg = _engine(["Gratitude journal", "Neutral journal"], dv="Depressive symptoms")
    assert _shift_in_d(pos, "Gratitude journal", "Well-being") > 0
    assert _shift_in_d(neg, "Gratitude journal", "Depressive symptoms") < 0
    # a harmful manipulation lowers a positive construct and raises a symptom
    ost_pos = _engine(["Ostracized", "Included"], dv="Sense of belonging")
    ost_neg = _engine(["Ostracized", "Included"], dv="Anxiety")
    assert _shift_in_d(ost_pos, "Ostracized", "Sense of belonging") < 0
    assert _shift_in_d(ost_neg, "Ostracized", "Anxiety") > 0


def test_the_applied_size_is_the_damped_recalled_value():
    e = _engine(["Expressive writing", "Neutral writing"], dv="Well-being")
    d = _shift_in_d(e, "Expressive writing", "Well-being")
    published = skb.META_ANALYTIC_DB["expressive_writing_meta"].effect_d
    # v1.3.0.6: the recall tier and the replication shrinkage are ONE discount (the stronger), not a product
    damped = (published * min(reg.confidence_weight("meta", "expressive_writing_meta"),
                              reg.policy_factor(reg.EffectPolicy())))
    # explicit currency: the reference arm is the zero point and the gap is 2 x 0.109 x d
    assert d == pytest.approx(damped * PC.CURATED_GAP_FACTOR, rel=1e-6)
    assert 0 < d / PC.CURATED_GAP_FACTOR < published


def test_an_explicit_user_effect_is_untouched_by_the_literature_layer():
    spec = EffectSizeSpec(variable="Attitude", factor="condition", level_high="Mortality salience",
                          level_low="Control", cohens_d=0.6, direction="positive")
    e = _engine(["Mortality salience", "Control"], specs=[spec])
    hi = e._get_effect_for_condition("Mortality salience", "Attitude")
    lo = e._get_effect_for_condition("Control", "Attitude")
    unit = e._EFFECT_D_TO_NORMALIZED * e._explicit_effect_scale("Attitude")
    assert (hi - lo) / unit == pytest.approx(2 * 0.6, rel=1e-6)      # the explicit currency: +d/2 / -d/2 per side
    assert e._applied_effects[("Mortality salience", "Attitude")]["source"] == "user"
    assert not getattr(e, "_literature_effect_log", [])


def test_inference_switched_off_is_a_true_null_even_for_a_curated_label():
    with contextlib.redirect_stdout(io.StringIO()):
        e = EnhancedSimulationEngine(
            study_title="Study", study_description="", sample_size=60, conditions=["Gamified", "Control"], factors=[],
            scales=[{"name": "Attitude", "variable_name": "Attitude", "type": "likert", "num_items": 1,
                     "scale_points": 7, "scale_min": 1, "scale_max": 7, "reverse_items": []}],
            additional_vars=[], demographics=dict(DEMO), open_ended_questions=[], effect_sizes=[], seed=1,
            auto_effects=False)
    assert e._get_effect_for_condition("Gamified", "Attitude") == 0.0


def test_economic_game_designs_are_left_to_the_game_models():
    e = _engine(["Social norm", "Control"], dv="Dictator game amount given", title="Dictator game with norm")
    e._get_effect_for_condition("Social norm", "Dictator_game_amount_given")
    assert not [r for r in getattr(e, "_literature_effect_log", []) if r["key"] == "social_norm_message_meta"]
    assert e._is_economic_game_context("Dictator game amount given")


def test_the_literature_log_records_the_curated_match():
    e = _engine(["Gamified", "Control"])
    _shift_in_d(e, "Gamified")
    log = getattr(e, "_literature_effect_log", [])
    assert log and log[-1]["key"] == "gamification_meta" and log[-1]["rule"] == "label_phrase"
    assert log[-1]["verification"] in reg.RECALL_TIERS


def test_older_entries_keep_the_keyword_first_order():
    # "Mindfulness training" is an older entry: the keyword rules still run first, so the
    # curated-label route is NOT taken for it (it only fills in when they find nothing).
    assert LE.lookup_curated("Mindfulness training") is None
    assert LE.lookup_curated("Gamified") is not None
