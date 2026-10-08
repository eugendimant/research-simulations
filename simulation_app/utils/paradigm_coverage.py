"""
Paradigm coverage -- more of the published literature reachable by default.
==========================================================================

When a user gives no effect size, the engine falls back on published magnitudes
(`META_ANALYTIC_DB`). Two things limited how often that happened:

  * a paradigm with no entry (nudges, social-norm appeals, ostracism, expressive
    writing, ...) could only ever receive a generic keyword guess, and
  * an entry with no phrase alias (a bare "personalized message", "limited time",
    "foot in the door") could not be found from a study title or a condition label.

This module holds the data that closes both gaps. It is DATA ONLY, imports nothing
from the rest of the package, and is read by three consumers:

  * `scientific_knowledge_base` merges `PARADIGM_ENTRIES` into `META_ANALYTIC_DB`;
  * `enhanced_simulation_engine` merges `ENTRY_ALIASES` into its phrase aliases and
    damps the entries in `PARADIGM_ENTRIES` by their verification tier;
  * `literature_effects` resolves a condition LABEL through `LABEL_RULES`.

Honesty rules (docs/REGISTRY.md, "The recall band"):

  * Every value in `PARADIGM_ENTRIES` is recalled, not read. None is tiered above
    the recall band: the verdict for each lives in
    `registry/recall_coverage_v1305.json` and is installed through
    `empirical_registry.register_recall`, which refuses anything stronger.
  * `n_studies` / `n_participants` are filled only where the count is known with
    confidence. A zero means "not recorded", never "none". `heterogeneity_tau`
    and `i_squared` are 0.0 everywhere for the same reason: nothing here is
    guessed, so no between-study draw is simulated for these entries.
  * Where memory is unsure the smaller effect and a wide interval were chosen.
  * Matching is strict. A label rule needs a whole-word phrase, and is vetoed by a
    negator or a valence-reversing word, because a wrong effect is worse than none.
"""
from __future__ import annotations

__version__ = "1.0.0"

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# New knowledge-base entries (keyword arguments for MetaAnalyticEffect)
# ---------------------------------------------------------------------------
# `effect_d` carries the NATURAL sign of the published contrast on its construct
# (harm-type manipulations are negative). The recall tier of each entry is in
# registry/recall_coverage_v1305.json, not here.

PARADIGM_ENTRIES: Dict[str, Dict[str, Any]] = {
    "nudge_general_meta": dict(
        source="Mertens et al. (2022, PNAS); Maier et al. (2022, PNAS)",
        effect_d=0.15, ci_95=(0.0, 0.45), domain="behavioral_economics",
        construct="nudge_behavior_change", paradigm="choice_architecture_nudge",
        replication_status="contested",
        notes="Mertens et al. reported a raw pooled d of about 0.43 across nudge interventions; "
              "Maier et al. argued that after correcting for publication bias there is no clear "
              "evidence of an average effect. 0.15 is a deliberately small compromise, not a "
              "published figure."),
    "social_norm_message_meta": dict(
        source="Rhodes, Shulman & McClaran (2020, Human Communication Research)",
        effect_d=0.20, ci_95=(0.05, 0.35), domain="social_influence",
        construct="norm_message_compliance", paradigm="descriptive_injunctive_norm_appeal",
        replication_status="replicated",
        notes="Social-norm appeals shift intentions and behaviour by a small amount; intentions "
              "move more than observed behaviour. Magnitude recalled loosely."),
    "identifiable_victim_effect": dict(
        source="Lee & Feeley (2016, Social Influence)",
        effect_d=0.10, ci_95=(0.0, 0.20), n_studies=41, domain="prosocial_behavior",
        construct="donation_to_identified_victim", paradigm="identifiable_vs_statistical_victim",
        replication_status="contested",
        notes="Meta-analytic correlation about .05: a reliable but very small effect, far from "
              "the large effect the textbook story implies."),
    "anthropomorphism_meta": dict(
        source="Blut, Wang, Wunderlich & Brock (2021, Journal of the Academy of Marketing Science)",
        effect_d=0.20, ci_95=(0.05, 0.40), domain="human_computer_interaction",
        construct="anthropomorphic_design_response", paradigm="humanlike_vs_machinelike_agent",
        replication_status="contested",
        notes="Human-like cues raise liking, trust and intentions modestly; recalled loosely, "
              "interval kept wide."),
    "mortality_salience_meta": dict(
        source="Burke, Martens & Faucher (2010, Personality and Social Psychology Review); "
               "Klein et al. (2022, multi-site replication)",
        effect_d=0.15, ci_95=(0.0, 0.40), domain="social_psychology",
        construct="worldview_defense", paradigm="mortality_salience_manipulation",
        replication_status="contested",
        notes="The original meta-analysis reported a moderate effect; large preregistered "
              "replications found little. 0.15 sits near the replication end."),
    "source_credibility_meta": dict(
        source="Wilson & Sherrell (1993, Journal of the Academy of Marketing Science)",
        effect_d=0.25, ci_95=(0.05, 0.45), domain="persuasion",
        construct="persuasion_by_credible_source", paradigm="source_credibility_manipulation",
        replication_status="replicated",
        notes="Credible or expert sources persuade somewhat more; the effect shrinks when the "
              "message is strong or the audience is highly involved. Recalled loosely."),
    "sponsorship_disclosure_meta": dict(
        source="Eisend, van Reijmersdal, Boerman & Tarrahi (2020, Journal of the Academy of Marketing Science)",
        effect_d=-0.10, ci_95=(-0.25, 0.02), domain="marketing",
        construct="attitude_after_sponsorship_disclosure", paradigm="sponsored_content_disclosure",
        replication_status="replicated",
        notes="Disclosing sponsorship raises recognition of persuasive intent and lowers brand "
              "attitudes only slightly. Sign is that of the effect on attitudes."),
    "social_exclusion_ostracism_meta": dict(
        source="Gerber & Wheeler (2009, Perspectives on Psychological Science); "
               "Hartgerink et al. (2015, PLOS ONE)",
        effect_d=-0.45, ci_95=(-0.75, -0.20), domain="social_psychology",
        construct="belonging_and_mood_after_exclusion", paradigm="ostracism_cyberball",
        replication_status="replicated",
        notes="Exclusion lowers belonging, mood and need satisfaction; the immediate effect on "
              "need satisfaction is larger than on downstream outcomes. Magnitude recalled "
              "loosely and set well below the largest reported figures."),
    "expressive_writing_meta": dict(
        source="Frattaroli (2006, Psychological Bulletin)",
        effect_d=0.15, ci_95=(0.05, 0.25), n_studies=146, domain="health_psychology",
        construct="health_after_expressive_writing", paradigm="expressive_writing",
        replication_status="replicated",
        notes="Pooled correlation about .075 (d about 0.15) on psychological and physical "
              "health outcomes."),
    "gratitude_intervention_meta": dict(
        source="Davis et al. (2016, Journal of Counseling Psychology)",
        effect_d=0.25, ci_95=(0.10, 0.40), n_studies=27, domain="positive_psychology",
        construct="wellbeing_after_gratitude_exercise", paradigm="gratitude_intervention",
        replication_status="replicated",
        notes="Gratitude exercises beat neutral or measurement-only controls modestly and are "
              "not clearly better than other positive exercises."),
    "positive_psychology_intervention_meta": dict(
        source="Bolier et al. (2013, BMC Public Health)",
        effect_d=0.25, ci_95=(0.10, 0.40), n_studies=39, domain="positive_psychology",
        construct="wellbeing_after_positive_intervention", paradigm="positive_psychology_intervention",
        replication_status="replicated",
        notes="Small effects on subjective and psychological well-being and on depression. "
              "Point estimate set conservatively inside the reported range."),
    "cognitive_reappraisal_meta": dict(
        source="Webb, Miles & Sheeran (2012, Psychological Bulletin)",
        effect_d=0.30, ci_95=(0.15, 0.45), domain="emotion_regulation",
        construct="emotion_after_reappraisal", paradigm="cognitive_reappraisal_instruction",
        replication_status="replicated",
        notes="Reappraisal changes emotional experience by a small-to-medium amount; recalled "
              "loosely."),
    "active_learning_stem_meta": dict(
        source="Freeman et al. (2014, PNAS)",
        effect_d=0.47, ci_95=(0.35, 0.60), n_studies=225, domain="education",
        construct="exam_performance_active_learning", paradigm="active_learning_vs_lecture",
        replication_status="replicated",
        notes="Active learning raised STEM exam scores by about half a standard deviation "
              "relative to traditional lecturing (quasi-experimental sections included)."),
    "feedback_intervention_meta": dict(
        source="Kluger & DeNisi (1996, Psychological Bulletin)",
        effect_d=0.41, ci_95=(0.30, 0.50), domain="organizational_psychology",
        construct="performance_after_feedback", paradigm="feedback_intervention",
        replication_status="replicated",
        notes="Average effect about 0.4 SD, but roughly a third of interventions lowered "
              "performance; the mean hides large heterogeneity."),
    "tailored_health_messages_meta": dict(
        source="Noar, Benac & Harris (2007, Psychological Bulletin)",
        effect_d=0.15, ci_95=(0.05, 0.25), n_studies=57, domain="health_communication",
        construct="health_behavior_after_tailored_message", paradigm="tailored_vs_generic_message",
        replication_status="replicated",
        notes="Mean correlation about .074 (d about 0.15) for tailored versus non-tailored "
              "health communication."),
    "graphic_warning_labels_meta": dict(
        source="Noar et al. (2016, Tobacco Control)",
        effect_d=0.20, ci_95=(0.05, 0.40), domain="health_communication",
        construct="risk_appraisal_after_graphic_warning", paradigm="pictorial_warning_label",
        replication_status="replicated",
        notes="Pictorial warnings raise fear, attention and risk appraisal more than behavioural "
              "intentions. A single blended figure is recalled loosely."),
    "illusory_truth_meta": dict(
        source="Dechene, Stahl, Hansen & Wanke (2010, Personality and Social Psychology Review)",
        effect_d=0.39, ci_95=(0.25, 0.55), n_studies=51, domain="cognition",
        construct="truth_rating_of_repeated_statement", paradigm="repetition_truth_effect",
        replication_status="replicated",
        notes="Repeated statements are rated as more true than new ones."),
    "gamification_meta": dict(
        source="Sailer & Homner (2020, Educational Psychology Review); Bai, Hew & Huang (2020)",
        effect_d=0.30, ci_95=(0.10, 0.50), domain="education",
        construct="engagement_after_gamification", paradigm="gamified_vs_plain_design",
        replication_status="contested",
        notes="Effects are positive and small to medium, larger for learning outcomes than for "
              "behaviour, and sensitive to study quality. Point estimate set conservatively."),
    "imagined_contact_meta": dict(
        source="Miles & Crisp (2014, Group Processes & Intergroup Relations)",
        effect_d=0.25, ci_95=(0.10, 0.40), n_studies=71, domain="intergroup_relations",
        construct="intergroup_attitude_after_imagined_contact", paradigm="imagined_intergroup_contact",
        replication_status="contested",
        notes="Imagined contact improves intergroup attitudes a little; recalled loosely."),
}

#: Entries that a free-text content match must never reach (they are found only
#: through `ENTRY_ALIASES` / `LABEL_RULES`, which are curated).
RULE_ONLY_KEYS = frozenset(PARADIGM_ENTRIES)

# ---------------------------------------------------------------------------
# Phrase aliases for study text (title + condition names). Normalised text has
# "_" and "-" turned into spaces, so write phrases with spaces only.
# ---------------------------------------------------------------------------
ENTRY_ALIASES: Dict[str, Tuple[str, ...]] = {
    # --- new entries (positive-signed only: the study-level anchor cannot know a sign) ---
    "nudge_general_meta": ("nudge", "nudges", "nudging"),
    "social_norm_message_meta": ("descriptive norm", "descriptive norms", "injunctive norm",
                                 "injunctive norms", "social norm message", "social norm messages",
                                 "social norm appeal", "social norm appeals", "norm message",
                                 "norm messages", "peer norm", "peer norms"),
    "identifiable_victim_effect": ("identifiable victim", "identified victim", "identifiable victims"),
    "anthropomorphism_meta": ("anthropomorphism", "anthropomorphic", "anthropomorphized"),
    "mortality_salience_meta": ("mortality salience", "terror management"),
    "source_credibility_meta": ("source credibility", "credible source", "expert source"),
    "expressive_writing_meta": ("expressive writing",),
    "gratitude_intervention_meta": ("gratitude intervention", "gratitude exercise",
                                    "gratitude journal", "gratitude letter", "gratitude condition"),
    "positive_psychology_intervention_meta": ("positive psychology intervention",
                                              "positive psychology interventions",
                                              "best possible self", "acts of kindness"),
    "cognitive_reappraisal_meta": ("cognitive reappraisal", "reappraisal"),
    "active_learning_stem_meta": ("active learning",),
    "feedback_intervention_meta": ("feedback intervention", "performance feedback"),
    "tailored_health_messages_meta": ("tailored message", "tailored messages", "tailored health message",
                                      "tailored health messages", "message tailoring"),
    "graphic_warning_labels_meta": ("graphic warning", "graphic warnings", "pictorial warning",
                                    "pictorial warnings", "graphic warning label",
                                    "graphic warning labels"),
    "illusory_truth_meta": ("illusory truth", "truth effect"),
    "gamification_meta": ("gamification", "gamified"),
    "imagined_contact_meta": ("imagined contact",),
    # --- existing entries that had no phrase a title or label could reach ---
    "default_effect": ("default setting", "default settings", "preselected",
                       "pre selected", "automatic enrollment", "auto enrollment",
                       "automatic enrolment"),
    "green_defaults_meta": ("green default", "green defaults", "sustainable default",
                            "sustainable defaults", "eco default", "renewable default"),
    "framing_general_meta": ("attribute framing", "risky choice framing", "goal framing",
                             "positive framing", "negative framing", "asian disease",
                             "framing manipulation"),
    "social_proof_marketing_meta": ("social proof", "bandwagon", "bestseller", "best seller",
                                    "best selling", "popularity cue",
                                    "others bought", "customers also bought"),
    "scarcity_effect_meta": ("limited time", "limited quantity", "limited supply", "limited stock",
                             "limited edition", "limited availability", "only a few left",
                             "few left", "low stock"),
    "zero_price_effect": ("zero price", "free gift", "free shipping", "free sample", "free samples",
                          "free product", "free item"),
    "user_review_meta": ("online review", "online reviews", "customer review", "customer reviews",
                         "consumer review", "consumer reviews", "product review", "product reviews",
                         "star rating", "star ratings", "review valence"),
    "celebrity_endorsement_meta": ("influencer endorsement", "influencer marketing",
                                   "celebrity endorser", "celebrity endorsement"),
    "personalization_meta": ("personalization", "personalisation", "personalized message",
                             "personalised message", "personalized messages", "personalised messages",
                             "personalized email", "personalised email", "personalized recommendation",
                             "personalized recommendations", "personalized ad", "personalized ads"),
    "humor_persuasion_meta": ("humor appeal", "humorous appeal", "humorous ad", "humorous message",
                              "humorous advertisement", "humour appeal", "humorous ads"),
    "fear_appeals_meta": ("fear appeal", "fear appeals", "high fear", "threat appeal", "threat appeals"),
    "cause_marketing_meta": ("corporate social responsibility", "cause related marketing",
                             "cause related", "csr"),
    "eco_labeling_meta": ("eco label", "eco labels", "ecolabel", "ecolabels", "sustainability label",
                          "carbon label"),
    "loyalty_program_meta": ("loyalty reward", "loyalty rewards", "rewards program", "reward program",
                             "loyalty card", "loyalty tier"),
    "decoy_effect_meta": ("asymmetric dominance", "attraction effect", "decoy option"),
    "hyperbolic_discounting": ("present bias", "delay discounting", "temporal discounting",
                               "intertemporal choice", "time preference"),
    "loss_aversion": ("loss averse",),
    "optimistic_bias": ("unrealistic optimism",),
    "bystander_effect": ("diffusion of responsibility",),
    "ingroup_cooperation_meta": ("minimal group", "minimal groups", "ingroup favoritism",
                                 "ingroup favouritism", "ingroup bias", "in group favoritism",
                                 "in group bias"),
    "mindfulness_intervention_meta": ("mindfulness training", "mindfulness meditation",
                                      "mindfulness intervention", "mindfulness exercise",
                                      "meditation", "mbsr", "mbct"),
    "self_affirmation_meta": ("values affirmation", "value affirmation"),
    "prebunking_misinformation": ("prebunking", "prebunk"),
    "debunking_misinformation_meta": ("fact check", "fact checking", "fact checks"),
    "spaced_practice_meta": ("distributed practice", "spacing effect"),
    "testing_effect_meta": ("practice testing", "practice test"),
    "foot_in_door_meta": ("foot in the door", "foot in door"),
    "door_in_face_meta": ("door in the face", "door in face"),
    "halo_effect_meta": ("halo effect", "attractiveness halo", "physical attractiveness",
                         "what is beautiful is good"),
    "peak_end_rule_meta": ("peak end rule", "peak end effect"),
    "ikea_effect_meta": ("ikea effect",),
    "intergroup_contact_extended": ("intergroup contact", "cross group contact"),
}

#: When two distinct keys match the study text with the same length and disagree
#: by more than the ambiguity band, the first key wins unless the guard pattern
#: also occurs. "social proof" in a consumer study is the marketing effect, not
#: the Asch conformity paradigm that the older `social_proof_meta` entry (0.9)
#: describes.
TIE_PREFER: Tuple[Tuple[str, str, str], ...] = (
    ("social_proof_marketing_meta", "social_proof_meta",
     r"\b(?:asch|conform\w*|confederates?|line judg\w*|unanimous\w*)\b"),
)

# ---------------------------------------------------------------------------
# Condition-label rules (used by literature_effects, only when the engine's own
# keyword rules found nothing). Patterns run on the NORMALISED label: lower case,
# camelCase split, every non-alphanumeric character a space.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelRule:
    key: str
    patterns: Tuple[str, ...]
    sign: int = 1                 # multiplies the entry's own signed effect
    polarity_aware: bool = False  # the engine flips the sign for symptom-type DVs


def _r(key: str, *patterns: str, sign: int = 1, polar: bool = False) -> LabelRule:
    return LabelRule(key, tuple(patterns), sign, polar)


LABEL_RULES: Tuple[LabelRule, ...] = (
    _r("nudge_general_meta", r"nudges?"),
    _r("social_norm_message_meta",
       r"(?:descriptive|injunctive|social|peer|empirical|normative) norms?",
       r"norm (?:message|appeal|prime|nudge|information|feedback)s?", polar=True),
    _r("identifiable_victim_effect", r"(?:identifiable|identified) (?:victim|beneficiary|child|recipient)s?",
       r"single victim", r"individual victim"),
    _r("anthropomorphism_meta", r"anthropomorph\w*", r"humanlike",
       r"human like (?:robot|agent|chatbot|avatar|ai|assistant)s?"),
    _r("mortality_salience_meta", r"mortality (?:salience|reminder|prime|priming)",
       r"death (?:reminder|prime|priming|thoughts?|salience)"),
    _r("source_credibility_meta", r"(?:high|strong) (?:source )?credibility", r"credible source",
       r"expert source", r"source credibility", r"expert (?:endorsement|endorser|spokesperson)"),
    _r("sponsorship_disclosure_meta",
       r"(?:sponsorship|sponsored|advertising|ad|paid partnership|influencer) disclosures?",
       r"disclosed (?:sponsorship|ad|advertising)"),
    _r("social_exclusion_ostracism_meta", r"ostrac\w+", r"social(?:ly)? exclu\w+",
       r"social rejection", r"cyberball", polar=True),
    _r("expressive_writing_meta", r"expressive writing", r"emotional disclosure",
       r"writing about (?:an? )?(?:emotional|traumatic|stressful)", polar=True),
    _r("gratitude_intervention_meta", r"gratitude", r"counting blessings", r"three good things", polar=True),
    _r("positive_psychology_intervention_meta", r"positive psychology interventions?",
       r"best possible self", r"acts? of kindness",
       r"kindness (?:intervention|exercise|condition)", polar=True),
    _r("cognitive_reappraisal_meta", r"(?:cognitive )?reappraisal", r"reappraise\w*", polar=True),
    _r("active_learning_stem_meta", r"active learning", r"peer instruction"),
    _r("feedback_intervention_meta",
       r"(?:performance|task|formative|corrective|elaborated|individual) feedback",
       r"feedback (?:intervention|condition|group)"),
    _r("tailored_health_messages_meta",
       r"tailored (?:health )?(?:messages?|interventions?|communications?|advice)",
       r"message tailoring", r"personally tailored"),
    _r("graphic_warning_labels_meta", r"(?:graphic|pictorial) (?:health )?warnings?(?: labels?)?",
       r"warning labels? with (?:images?|pictures?|graphics?)", polar=True),
    _r("illusory_truth_meta", r"illusory truth", r"truth effect", r"repeated (?:claims?|statements?|headlines?)"),
    _r("gamification_meta", r"gamif\w+", r"leaderboards?", r"points and badges"),
    _r("imagined_contact_meta", r"imagined (?:intergroup )?contact", polar=True),
    # existing entries, reached by condition label
    _r("scarcity_effect_meta", r"scarcity (?:message|appeal|cue|framing|manipulation)s?",
       r"limited (?:time|quantity|supply|stock|edition|availability)", r"only (?:a )?few left", r"low stock"),
    _r("zero_price_effect", r"free (?:gift|shipping|sample|product|item)s?", r"zero price"),
    _r("user_review_meta", r"(?:online|customer|consumer|product|user) reviews?", r"star ratings?"),
    _r("loyalty_program_meta", r"loyalty (?:program|programme|reward|card|tier)s?", r"rewards? program"),
    _r("personalization_meta", r"personali[sz]ed (?:message|email|ad|offer|recommendation|content|greeting)s?",
       r"personali[sz]ation"),
    _r("default_effect", r"opt out(?: default)?", r"default (?:nudge|option|setting|enrollment|enrolment|choice)s?",
       r"pre ?selected"),
    _r("green_defaults_meta", r"(?:green|sustainable|eco|renewable) defaults?"),
    _r("social_proof_marketing_meta", r"social proof", r"bandwagon", r"best ?sellers?", r"most popular",
       r"others? (?:also )?bought"),
    _r("celebrity_endorsement_meta", r"celebrity (?:endorse\w*|spokesperson)", r"influencer (?:endorse\w*|marketing)"),
    _r("fear_appeals_meta", r"fear appeals?", r"high fear", r"threat appeals?"),
    _r("humor_persuasion_meta", r"humou?r(?:ous)? (?:appeal|ad|advert\w*|message)s?"),
    _r("cause_marketing_meta", r"cause related(?: marketing)?", r"corporate social responsibility", r"csr"),
    _r("moral_licensing_meta", r"moral licen[cs]\w*"),
    _r("mere_exposure_meta", r"mere exposure", r"repeated exposure", r"exposure frequency"),
    _r("mindfulness_intervention_meta",
       r"mindfulness (?:training|meditation|intervention|exercise|condition|session)s?",
       r"meditation", r"mbsr", r"mbct", polar=True),
    _r("self_affirmation_meta", r"self affirm\w*", r"values? affirmation", polar=True),
    _r("foot_in_door_meta", r"foot in (?:the )?door"),
    _r("door_in_face_meta", r"door in (?:the )?face"),
    _r("intergroup_contact_extended", r"intergroup contact", r"contact hypothesis", polar=True),
    _r("stereotype_threat", r"stereotype threat", sign=-1),
    _r("sunk_cost_escalation", r"sunk costs?"),
    _r("growth_mindset_meta", r"growth mindset (?:intervention|condition|message)s?"),
    _r("implementation_intentions_meta", r"implementation intentions?", r"if then plans?"),
    _r("inoculation_meta", r"inoculat\w+", r"prebunk\w*", polar=True),
    _r("testing_effect_meta", r"retrieval practice", r"practice test(?:ing)?", r"testing effect"),
    _r("spaced_practice_meta", r"spaced (?:practice|learning|repetition)", r"distributed practice"),
    _r("goal_setting_meta", r"goal setting"),
    _r("perspective_taking_meta", r"perspective taking", polar=True),
    _r("narrative_transportation_meta", r"narrative (?:message|story|format|condition|transportation)s?"),
)

#: Gap multiplier for a curated label match. The engine's convention for a contrast of d is a gap of
#: 2 x 0.109 x d between the two arms, with the reference arm as the zero point; a single arm
#: shifted by 0.109 x d delivers only d/2 (measured: a recalled 0.17 came back as 0.10, a 0.26 as 0.13).
CURATED_GAP_FACTOR = 2.0

#: A label carrying any of these is not the plain, positive form of the paradigm:
#: either it is a negated or reference arm ("No gamified tier", "without a nudge"),
#: or it reverses the direction ("dishonest norm", "unsustainable default"). Both
#: fall back to the engine's own rules rather than receive a signed guess.
_VETO_RE = re.compile(
    r"(?<![a-z0-9])(?:no|not|non|without|absent|absence|zero|lack|lacking|never|anti|antisocial|"
    r"dis\w*honest\w*|unsustainable|negative|low|reduced|minimal|weak|false|fake|bad|reverse\w*)(?![a-z0-9])"
)

#: Symptom-type DV words beyond the engine's own list; a beneficial treatment
#: lowers them. Used only to flip the sign of rules marked `polarity_aware`.
_NEGATIVE_DV_EXTRA_RE = re.compile(
    r"(?<![a-z0-9])(?:negative affect|sadness|sad|anger|angry|disgust|tension|rejection|"
    r"negative mood|dysphoria|prejudice|stereotyp\w*|smoking intention|intention to smoke)"
)

_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalise_label(text: str) -> str:
    """Lower-case, split camelCase, turn every non-alphanumeric run into one space."""
    spaced = _CAMEL_RE.sub(" ", str(text or ""))
    return _NON_ALNUM_RE.sub(" ", spaced.lower()).strip()


def normalise_text(text: str) -> str:
    """Normalisation for study-level text: only "_" and "-" become spaces, so the
    punctuation the engine's own patterns rely on is left alone."""
    return re.sub(r"[_\-]+", " ", str(text or "").lower())


def dv_is_negative(text: str) -> bool:
    """Whether a DV description names a symptom-type construct a benefit lowers
    (beyond the engine's own list, which the engine checks itself)."""
    return bool(_NEGATIVE_DV_EXTRA_RE.search(normalise_label(text)))


@lru_cache(maxsize=1)
def _compiled_rules() -> Tuple[Tuple[LabelRule, Tuple["re.Pattern[str]", ...]], ...]:
    out = []
    for rule in LABEL_RULES:
        pats = tuple(re.compile(r"(?<![a-z0-9])(?:" + p + r")(?![a-z0-9])") for p in rule.patterns)
        out.append((rule, pats))
    return tuple(out)


def match_label_rule(label: str) -> Optional[LabelRule]:
    """The rule a condition label names, or None.

    The longest matching phrase wins. Two different entries matching with the same
    length are ambiguous and give None. A vetoed label (negator or direction
    reversal) gives None.
    """
    norm = normalise_label(label)
    if not norm or _VETO_RE.search(norm):
        return None
    best_len = 0
    best: List[LabelRule] = []
    for rule, pats in _compiled_rules():
        n = 0
        for pat in pats:
            m = pat.search(norm)
            if m:
                n = max(n, len(m.group(0)))
        if not n:
            continue
        if n > best_len:
            best_len, best = n, [rule]
        elif n == best_len:
            best.append(rule)
    if not best:
        return None
    if len({r.key for r in best}) > 1:
        return None
    return best[0]


#: Words that mark an economic-game DV or design. Game calibrations belong to the game models, so
#: neither the curated label route nor the label phrases of `literature_effects` act on them.
_ECON_GAME_RE = re.compile(
    r"(?<![a-z0-9])(?:dictator\w*|ultimatum|public goods?|trust game|prisoner\w* dilemma|economic game|"
    r"pgg|give|gives|giving|gave|given|allocat\w*|endowment|contribut\w*|amount (?:sent|shared|given)|"
    r"tokens?|punisher\w*|third party punishment)(?![a-z0-9])"
)


def is_economic_game_text(text: str) -> bool:
    """Whether text names an economic game or an allocation DV (whole words, deliberately narrower
    than the engine's own substring list: this gate decides only whether a curated phrase may act)."""
    return bool(_ECON_GAME_RE.search(normalise_label(text)))


def entries_without_rules() -> List[str]:
    """New entries that no alias or label rule can reach (a coverage self-check)."""
    ruled = {r.key for r in LABEL_RULES}
    return sorted(k for k in PARADIGM_ENTRIES if k not in ruled and k not in ENTRY_ALIASES)
