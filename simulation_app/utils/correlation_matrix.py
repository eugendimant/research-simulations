"""
Correlation Matrix Module for Behavioral Experiment Simulation.

Provides a scientifically-grounded domain correlation library based on
published meta-analytic findings. Supports automatic construct detection
from scale/DV names, empirical correlation lookup, and generation of
multivariate correlated latent scores.

Version: 1.8.8.0

Key capabilities:
    - Detect construct types from scale names and question text
    - Look up empirically-informed correlations between construct pairs
    - Build positive-definite correlation matrices
    - Generate multivariate normal latent scores with realistic correlation structure

Meta-analytic sources informing default correlations:
    - Palmatier et al. (2006) - Relationship marketing
    - Szymanski & Henard (2001) - Customer satisfaction
    - Stajkovic & Luthans (1998) - Self-efficacy and performance
    - Judge et al. (2001) - Job satisfaction
    - Venkatesh et al. (2003) - Technology acceptance (UTAUT)
    - Colquitt et al. (2001) - Organizational justice
    - Armitage & Conner (2001) - Theory of planned behavior
    - Meyer et al. (2002) - Organizational commitment
    - Maslach et al. (2001) - Burnout
    - Deci & Ryan (2000) - Self-determination theory
    - Watson & Clark (1984) - Negative affectivity
    - Diener et al. (1999) - Subjective well-being
    - Bandura (1997) - Self-efficacy theory
    - Mount & Barrick (1995) - Big Five and job performance
    - Mitchell (1999) - Perceived risk
    - Ajzen (1991) - Theory of planned behavior
    - Morgan & Hunt (1994) - Commitment-trust theory
"""

__version__ = "1.8.8.0"

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# 1. CONSTRUCT_KEYWORDS
#    Maps keywords found in scale/DV names to construct types.
#    Multi-word phrases are listed first for higher specificity matching.
# ---------------------------------------------------------------------------

CONSTRUCT_KEYWORDS: Dict[str, List[str]] = {
    # --- Attitudes (9 constructs) ---
    "trust": [
        "trust", "trustworthiness", "trustworthy", "reliability", "reliable",
        "dependability", "dependable", "faith", "confidence_in",
    ],
    "satisfaction": [
        "satisfaction", "satisfied", "content", "pleased", "contentment",
        "fulfillment", "gratification",
    ],
    "loyalty": [
        "loyalty", "loyal", "retention", "repeat", "repurchase",
        "switching_cost", "allegiance",
    ],
    "commitment": [
        "commitment", "committed", "dedication", "devoted", "devotion",
        "pledge", "obligation",
    ],
    "engagement": [
        "engagement", "engaged", "involvement", "involved", "immersion",
        "participation", "interact",
    ],
    "attitude": [
        "attitude", "favorability", "favorable", "unfavorable", "evaluation",
        "disposition", "valence", "sentiment",
    ],
    "preference": [
        "preference", "prefer", "choice", "selected", "chosen", "option",
        "alternative",
    ],
    "liking": [
        "liking", "like", "appeal", "attractive", "attractiveness",
        "desirability", "desirable", "pleasant",
    ],
    "brand_attitude": [
        "brand_attitude", "brand_perception", "brand_image", "brand_equity",
        "brand_evaluation", "brand_affect", "brand_trust", "brand_love",
    ],

    # --- Behavioral Intentions (5 constructs) ---
    "purchase_intent": [
        "purchase_intent", "purchase_intention", "purchase", "buy", "buying",
        "WTP", "willingness_to_pay", "willing_to_pay", "spending",
        "acquisition", "checkout",
    ],
    "use_intent": [
        "use_intent", "usage_intent", "usage_intention", "adoption", "adopt",
        "trial", "try", "uptake", "download", "install", "sign_up",
    ],
    "recommendation": [
        "recommendation", "recommend", "word_of_mouth", "WOM", "referral",
        "NPS", "net_promoter", "tell_friend", "share", "endorse",
        "endorsement", "advocate",
    ],
    "compliance": [
        "compliance", "comply", "adherence", "adhere", "follow", "obey",
        "obedience", "conforming", "rule_following",
    ],
    "behavioral_intent": [
        "behavioral_intent", "behavioural_intent", "intention", "plan",
        "likelihood", "propensity", "inclination", "tendency",
    ],

    # --- Emotions (10 constructs) ---
    "anxiety": [
        "anxiety", "anxious", "worry", "worried", "nervous", "nervousness",
        "apprehension", "apprehensive", "unease", "uneasy", "dread",
    ],
    "stress": [
        "stress", "stressor", "strain", "pressure", "tension", "distress",
        "overwhelm", "overload", "hassle",
    ],
    "happiness": [
        "happiness", "happy", "joy", "joyful", "cheerful", "delight",
        "delighted", "elation", "bliss", "pleased",
    ],
    "well_being": [
        "well_being", "wellbeing", "welfare", "flourishing",
        "life_satisfaction", "quality_of_life", "QOL", "eudaimonic",
        "hedonic", "thriving",
    ],
    "mood": [
        "mood", "affect_state", "feeling", "emotional_state", "temperament",
    ],
    "positive_affect": [
        "positive_affect", "PA", "enthusiasm", "excited", "excitement",
        "inspired", "inspiration", "alert", "attentive", "active",
        "energetic", "vigor", "vitality",
    ],
    "negative_affect": [
        "negative_affect", "NA", "upset", "hostile", "hostility",
        "irritable", "irritability", "ashamed", "guilty", "distressed",
        "jittery",
    ],
    "anger": [
        "anger", "angry", "rage", "fury", "furious", "hostility",
        "aggression", "aggressive", "outrage", "outraged", "irritation",
        "annoyed", "annoyance", "resentment",
    ],
    "fear": [
        "fear", "afraid", "frightened", "scared", "terror", "terrified",
        "panic", "phobia", "fright", "alarm",
    ],
    "disgust": [
        "disgust", "disgusted", "revulsion", "repulsion", "repulsed",
        "aversion", "nauseated", "nausea", "gross",
    ],

    # --- Cognition (8 constructs) ---
    "perceived_risk": [
        "perceived_risk", "risk", "danger", "hazard", "threat",
        "vulnerability", "risk_perception", "riskiness", "unsafe",
        "uncertain", "uncertainty",
    ],
    "perceived_quality": [
        "perceived_quality", "quality", "excellence", "excellent", "superior",
        "premium", "craftsmanship", "workmanship", "high_quality",
    ],
    "perceived_fairness": [
        "perceived_fairness", "fairness", "justice", "equity", "equitable",
        "fair", "unfair", "procedural_justice", "distributive_justice",
        "interactional_justice", "organizational_justice",
    ],
    "credibility": [
        "credibility", "credible", "believability", "believable",
        "trustworthy_source", "source_credibility", "expertise",
        "authoritativeness", "authoritative",
    ],
    "competence": [
        "competence", "competent", "ability", "capable", "capability",
        "skill", "skilled", "proficiency", "proficient", "aptitude",
        "talent",
    ],
    "knowledge": [
        "knowledge", "awareness", "familiarity", "understanding", "literacy",
        "informed", "comprehension", "learning", "expertise_level",
    ],
    "perceived_value": [
        "perceived_value", "value", "worth", "benefit", "utility",
        "cost_benefit", "value_for_money", "bang_for_buck",
    ],
    "cognitive_load": [
        "cognitive_load", "mental_effort", "processing", "difficulty",
        "mental_demand", "information_overload", "complexity",
        "effortful", "taxing",
    ],

    # --- Self-Concept (5 constructs) ---
    "self_efficacy": [
        "self_efficacy", "efficacy", "confidence", "self_confidence",
        "capable", "mastery", "self_belief", "can_do",
    ],
    "self_esteem": [
        "self_esteem", "self_worth", "self_regard", "self_respect",
        "self_acceptance", "self_value",
    ],
    "self_control": [
        "self_control", "self_regulation", "willpower", "impulse",
        "impulse_control", "self_discipline", "restraint", "inhibition",
        "delay_of_gratification",
    ],
    "identity": [
        "identity", "self_concept", "self_identity", "self_image",
        "self_perception", "personal_identity", "role_identity",
    ],
    "autonomy": [
        "autonomy", "autonomous", "independence", "independent",
        "self_determination", "self_direction", "agency", "volition",
        "free_will", "choice_freedom",
    ],

    # --- Social (6 constructs) ---
    "social_norm": [
        "social_norm", "norm", "normative", "social_pressure",
        "peer_pressure", "subjective_norm", "descriptive_norm",
        "injunctive_norm", "social_expectation",
    ],
    "conformity": [
        "conformity", "conform", "compliance_social", "obedience_social",
        "groupthink", "herd", "bandwagon", "social_influence",
    ],
    "empathy": [
        "empathy", "empathic", "empathetic", "perspective_taking",
        "compassion", "compassionate", "sympathy", "sympathetic",
        "emotional_intelligence", "other_regard",
    ],
    "prosocial": [
        "prosocial", "altruism", "altruistic", "helping", "charitable",
        "donation", "generosity", "generous", "volunteering", "volunteer",
        "philanthropy", "benevolence",
    ],
    "cooperation": [
        "cooperation", "cooperate", "cooperative", "collaborative",
        "collaboration", "teamwork", "team_work", "coordination",
        "collective_action", "joint_effort",
    ],
    "social_identity": [
        "social_identity", "group_identity", "ingroup", "in_group",
        "belonging", "belongingness", "group_membership", "collective_self",
        "we_feeling", "collective_identity",
    ],

    # --- Organizational (5 constructs) ---
    "job_satisfaction": [
        "job_satisfaction", "work_satisfaction", "career_satisfaction",
        "pay_satisfaction", "supervisor_satisfaction", "coworker_satisfaction",
        "job_content", "job_fulfillment",
    ],
    "organizational_commitment": [
        "organizational_commitment", "org_commitment",
        "affective_commitment", "continuance_commitment",
        "normative_commitment", "company_loyalty", "organizational_loyalty",
    ],
    "work_engagement": [
        "work_engagement", "job_engagement", "employee_engagement",
        "vigor_work", "absorption_work", "dedication_work",
        "work_enthusiasm", "job_involvement",
    ],
    "burnout": [
        "burnout", "burn_out", "exhaustion", "emotional_exhaustion",
        "cynicism", "depersonalization", "compassion_fatigue",
        "occupational_stress",
    ],
    "leadership": [
        "leadership", "leader", "transformational", "transactional",
        "servant_leadership", "authentic_leadership", "charismatic",
        "leader_effectiveness", "LMX", "leader_member_exchange",
    ],

    # --- Health (4 constructs) ---
    "health_behavior": [
        "health_behavior", "health_behaviour", "exercise", "diet",
        "physical_activity", "health_action", "preventive_behavior",
        "health_compliance", "medication_adherence", "screening",
        "vaccination", "sleep_quality",
    ],
    "health_anxiety": [
        "health_anxiety", "health_worry", "hypochondria",
        "illness_anxiety", "cyberchondria", "health_concern",
        "medical_anxiety", "disease_fear",
    ],
    "body_image": [
        "body_image", "body_satisfaction", "body_dissatisfaction",
        "appearance", "body_esteem", "body_perception", "body_shame",
        "weight_concern", "appearance_evaluation",
    ],
    "coping": [
        "coping", "cope", "resilience", "resilient", "adaptation",
        "recovery", "coping_strategy", "coping_style",
        "emotion_focused_coping", "problem_focused_coping",
        "stress_management",
    ],

    # --- Technology (3 constructs) ---
    "tech_acceptance": [
        "tech_acceptance", "technology_acceptance", "perceived_usefulness",
        "ease_of_use", "perceived_ease", "TAM", "technology",
        "UTAUT", "system_usefulness", "user_acceptance",
    ],
    "privacy_concern": [
        "privacy_concern", "privacy", "surveillance", "data_concern",
        "information_privacy", "data_privacy", "tracking",
        "data_collection", "personal_data", "GDPR",
    ],
    "algorithm_aversion": [
        "algorithm_aversion", "algorithm", "AI_aversion", "automation",
        "automation_bias", "machine_distrust", "robo", "chatbot_aversion",
        "AI_distrust", "algorithmic_bias",
    ],

    # --- Motivation (3 constructs) ---
    "intrinsic_motivation": [
        "intrinsic_motivation", "intrinsic", "interest", "curiosity",
        "curious", "enjoyment", "inherent_satisfaction", "flow",
        "autotelic", "mastery_motivation",
    ],
    "extrinsic_motivation": [
        "extrinsic_motivation", "extrinsic", "reward", "incentive",
        "compensation", "bonus", "prize", "external_regulation",
        "contingent_reward",
    ],
    "need_for_cognition": [
        "need_for_cognition", "NFC", "thinking", "deliberation",
        "deliberate", "analytical", "reflective_thinking",
        "cognitive_engagement", "intellectual_curiosity",
    ],
}


# ---------------------------------------------------------------------------
# CONSTRUCT_CATEGORIES
# Groups construct types into broader categories for default-correlation logic.
# ---------------------------------------------------------------------------

CONSTRUCT_CATEGORIES: Dict[str, List[str]] = {
    "attitudes": [
        "trust", "satisfaction", "loyalty", "commitment", "engagement",
        "attitude", "preference", "liking", "brand_attitude",
    ],
    "behavioral_intentions": [
        "purchase_intent", "use_intent", "recommendation", "compliance",
        "behavioral_intent",
    ],
    "emotions": [
        "anxiety", "stress", "happiness", "well_being", "mood",
        "positive_affect", "negative_affect", "anger", "fear", "disgust",
    ],
    "cognition": [
        "perceived_risk", "perceived_quality", "perceived_fairness",
        "credibility", "competence", "knowledge", "perceived_value",
        "cognitive_load",
    ],
    "self_concept": [
        "self_efficacy", "self_esteem", "self_control", "identity",
        "autonomy",
    ],
    "social": [
        "social_norm", "conformity", "empathy", "prosocial", "cooperation",
        "social_identity",
    ],
    "organizational": [
        "job_satisfaction", "organizational_commitment", "work_engagement",
        "burnout", "leadership",
    ],
    "health": [
        "health_behavior", "health_anxiety", "body_image", "coping",
    ],
    "technology": [
        "tech_acceptance", "privacy_concern", "algorithm_aversion",
    ],
    "motivation": [
        "intrinsic_motivation", "extrinsic_motivation", "need_for_cognition",
    ],
}

# Reverse lookup: construct_type -> category
_CONSTRUCT_TO_CATEGORY: Dict[str, str] = {}
for _cat, _constructs in CONSTRUCT_CATEGORIES.items():
    for _c in _constructs:
        _CONSTRUCT_TO_CATEGORY[_c] = _cat


# ---------------------------------------------------------------------------
# 2. EMPIRICAL_CORRELATIONS
#    Empirically-informed default correlations between construct pairs.
#    Based on published meta-analytic findings. Each entry includes a brief
#    citation comment. Values are Pearson r correlations.
# ---------------------------------------------------------------------------

EMPIRICAL_CORRELATIONS: Dict[Tuple[str, str], float] = {
    # ===================================================================
    # TRUST CLUSTER
    # Primary sources: Morgan & Hunt (1994); Palmatier et al. (2006)
    # ===================================================================
    ("trust", "satisfaction"): 0.52,          # Palmatier et al. 2006
    ("trust", "commitment"): 0.50,            # Morgan & Hunt 1994
    ("trust", "loyalty"): 0.44,               # Palmatier et al. 2006
    ("trust", "cooperation"): 0.42,           # Morgan & Hunt 1994
    ("trust", "recommendation"): 0.38,        # Palmatier et al. 2006
    ("trust", "purchase_intent"): 0.40,       # Gefen et al. 2003
    ("trust", "perceived_risk"): -0.38,       # Mitchell 1999; Pavlou 2003
    ("trust", "perceived_quality"): 0.45,     # Garbarino & Johnson 1999
    ("trust", "credibility"): 0.63,           # Ohanian 1990; Hovland 1953
    ("trust", "perceived_fairness"): 0.48,    # Colquitt et al. 2001
    ("trust", "attitude"): 0.42,              # Palmatier et al. 2006
    ("trust", "engagement"): 0.40,            # Brodie et al. 2013
    ("trust", "behavioral_intent"): 0.38,     # Gefen et al. 2003
    ("trust", "perceived_value"): 0.42,       # Sirdeshmukh et al. 2002
    ("trust", "competence"): 0.52,            # Mayer et al. 1995
    ("trust", "empathy"): 0.28,               # Colquitt et al. 2007
    ("trust", "privacy_concern"): -0.35,      # Malhotra et al. 2004
    ("trust", "use_intent"): 0.38,            # Gefen et al. 2003

    # ===================================================================
    # SATISFACTION CLUSTER
    # Primary sources: Szymanski & Henard (2001); Oliver (1999)
    # ===================================================================
    ("satisfaction", "loyalty"): 0.58,         # Szymanski & Henard 2001
    ("satisfaction", "recommendation"): 0.55,  # Szymanski & Henard 2001
    ("satisfaction", "purchase_intent"): 0.48, # Oliver 1999
    ("satisfaction", "commitment"): 0.52,      # Gustafsson et al. 2005
    ("satisfaction", "perceived_quality"): 0.55,  # Cronin et al. 2000
    ("satisfaction", "perceived_value"): 0.58,    # Cronin et al. 2000
    ("satisfaction", "positive_affect"): 0.42,    # Oliver 1993
    ("satisfaction", "well_being"): 0.45,         # Diener et al. 1999
    ("satisfaction", "engagement"): 0.50,         # Brodie et al. 2013
    ("satisfaction", "attitude"): 0.50,           # Oliver 1999
    ("satisfaction", "behavioral_intent"): 0.48,  # Szymanski & Henard 2001
    ("satisfaction", "liking"): 0.52,             # Oliver 1999
    ("satisfaction", "preference"): 0.45,         # Fornell 1992
    ("satisfaction", "negative_affect"): -0.35,   # Oliver 1993
    ("satisfaction", "use_intent"): 0.45,         # Bhattacherjee 2001

    # ===================================================================
    # SELF-EFFICACY CLUSTER
    # Primary sources: Stajkovic & Luthans (1998); Bandura (1997)
    # ===================================================================
    ("self_efficacy", "anxiety"): -0.35,          # Bandura 1997; Pajares 1996
    ("self_efficacy", "stress"): -0.30,           # Schwarzer & Jerusalem 1995
    ("self_efficacy", "self_esteem"): 0.49,       # Judge & Bono 2001
    ("self_efficacy", "intrinsic_motivation"): 0.42,  # Zimmerman 2000
    ("self_efficacy", "self_control"): 0.38,      # Bandura 1997
    ("self_efficacy", "competence"): 0.55,        # Stajkovic & Luthans 1998
    ("self_efficacy", "autonomy"): 0.40,          # Deci & Ryan 2000
    ("self_efficacy", "well_being"): 0.38,        # Bandura 1997
    ("self_efficacy", "coping"): 0.42,            # Schwarzer & Jerusalem 1995
    ("self_efficacy", "perceived_risk"): -0.28,   # Witte 1996
    ("self_efficacy", "knowledge"): 0.35,         # Stajkovic & Luthans 1998
    ("self_efficacy", "behavioral_intent"): 0.38, # Ajzen 1991
    ("self_efficacy", "health_behavior"): 0.38,   # Schwarzer 2008
    ("self_efficacy", "burnout"): -0.38,          # Alarcon et al. 2009
    ("self_efficacy", "work_engagement"): 0.42,   # Xanthopoulou et al. 2007
    ("self_efficacy", "fear"): -0.30,             # Bandura 1997
    ("self_efficacy", "positive_affect"): 0.32,   # Judge & Bono 2001
    ("self_efficacy", "tech_acceptance"): 0.35,   # Venkatesh et al. 2003

    # ===================================================================
    # JOB SATISFACTION CLUSTER
    # Primary sources: Judge et al. (2001); Meyer et al. (2002)
    # ===================================================================
    ("job_satisfaction", "organizational_commitment"): 0.53,  # Meyer et al. 2002
    ("job_satisfaction", "work_engagement"): 0.52,   # Christian et al. 2011
    ("job_satisfaction", "burnout"): -0.48,          # Lee & Ashforth 1996
    ("job_satisfaction", "well_being"): 0.44,        # Bowling et al. 2010
    ("job_satisfaction", "leadership"): 0.35,        # Judge et al. 2004
    ("job_satisfaction", "positive_affect"): 0.40,   # Thoresen et al. 2003
    ("job_satisfaction", "stress"): -0.35,           # Podsakoff et al. 2007
    ("job_satisfaction", "autonomy"): 0.38,          # Humphrey et al. 2007
    ("job_satisfaction", "perceived_fairness"): 0.45,  # Colquitt et al. 2001
    ("job_satisfaction", "trust"): 0.40,             # Dirks & Ferrin 2002
    ("job_satisfaction", "self_efficacy"): 0.38,     # Judge & Bono 2001
    ("job_satisfaction", "negative_affect"): -0.35,  # Thoresen et al. 2003
    ("job_satisfaction", "cooperation"): 0.30,       # LePine et al. 2002
    ("job_satisfaction", "self_esteem"): 0.35,       # Judge & Bono 2001
    ("job_satisfaction", "anxiety"): -0.30,          # Spector 1997
    ("job_satisfaction", "satisfaction"): 0.55,      # Judge et al. 2001 (domain overlap)
    ("job_satisfaction", "commitment"): 0.50,        # Meyer et al. 2002

    # ===================================================================
    # ORGANIZATIONAL COMMITMENT CLUSTER
    # Primary sources: Meyer et al. (2002); Allen & Meyer (1990)
    # ===================================================================
    ("organizational_commitment", "work_engagement"): 0.48,  # Saks 2006
    ("organizational_commitment", "burnout"): -0.42,    # Lee & Ashforth 1996
    ("organizational_commitment", "leadership"): 0.38,  # Avolio et al. 2004
    ("organizational_commitment", "trust"): 0.45,       # Dirks & Ferrin 2002
    ("organizational_commitment", "perceived_fairness"): 0.40,  # Colquitt et al. 2001
    ("organizational_commitment", "satisfaction"): 0.48,  # Meyer et al. 2002
    ("organizational_commitment", "social_identity"): 0.42,  # Ashforth & Mael 1989
    ("organizational_commitment", "autonomy"): 0.32,    # Meyer et al. 2002
    ("organizational_commitment", "cooperation"): 0.35, # Meyer et al. 2002
    ("organizational_commitment", "stress"): -0.28,     # Meyer et al. 2002

    # ===================================================================
    # BURNOUT CLUSTER
    # Primary sources: Maslach et al. (2001); Lee & Ashforth (1996)
    # ===================================================================
    ("burnout", "work_engagement"): -0.55,        # Schaufeli & Bakker 2004
    ("burnout", "stress"): 0.52,                  # Maslach et al. 2001
    ("burnout", "anxiety"): 0.45,                 # Alarcon 2011
    ("burnout", "well_being"): -0.48,             # Hakanen & Schaufeli 2012
    ("burnout", "negative_affect"): 0.48,         # Thoresen et al. 2003
    ("burnout", "positive_affect"): -0.40,        # Alarcon et al. 2009
    ("burnout", "leadership"): -0.30,             # Halbesleben & Buckley 2004
    ("burnout", "autonomy"): -0.32,               # Humphrey et al. 2007
    ("burnout", "self_esteem"): -0.35,            # Alarcon et al. 2009
    ("burnout", "cooperation"): -0.25,            # Maslach et al. 2001
    ("burnout", "coping"): -0.35,                 # Lee & Ashforth 1996
    ("burnout", "health_behavior"): -0.22,        # Maslach et al. 2001

    # ===================================================================
    # TECHNOLOGY ACCEPTANCE CLUSTER
    # Primary sources: Venkatesh et al. (2003); Davis (1989)
    # ===================================================================
    ("tech_acceptance", "use_intent"): 0.55,      # Venkatesh et al. 2003
    ("tech_acceptance", "attitude"): 0.52,        # Davis 1989
    ("tech_acceptance", "perceived_quality"): 0.42,  # Venkatesh et al. 2003
    ("tech_acceptance", "anxiety"): -0.28,        # Venkatesh 2000
    ("tech_acceptance", "behavioral_intent"): 0.50,  # Venkatesh et al. 2003
    ("tech_acceptance", "knowledge"): 0.30,       # Venkatesh et al. 2003
    ("tech_acceptance", "satisfaction"): 0.42,    # Bhattacherjee 2001
    ("tech_acceptance", "trust"): 0.38,           # Gefen et al. 2003
    ("tech_acceptance", "perceived_value"): 0.40, # Kim et al. 2007

    # ===================================================================
    # PRIVACY / ALGORITHM AVERSION CLUSTER
    # Primary sources: Malhotra et al. (2004); Dietvorst et al. (2015)
    # ===================================================================
    ("privacy_concern", "use_intent"): -0.30,     # Malhotra et al. 2004
    ("privacy_concern", "perceived_risk"): 0.42,  # Malhotra et al. 2004
    ("privacy_concern", "anxiety"): 0.28,         # Smith et al. 2011
    ("privacy_concern", "purchase_intent"): -0.25,  # Dinev & Hart 2006
    ("privacy_concern", "satisfaction"): -0.22,   # Dinev & Hart 2006
    ("privacy_concern", "algorithm_aversion"): 0.38,  # Castelo et al. 2019
    ("privacy_concern", "fear"): 0.25,            # Smith et al. 2011
    ("algorithm_aversion", "trust"): -0.32,       # Dietvorst et al. 2015
    ("algorithm_aversion", "tech_acceptance"): -0.38,  # Dietvorst et al. 2015
    ("algorithm_aversion", "use_intent"): -0.35,  # Dietvorst et al. 2015
    ("algorithm_aversion", "competence"): -0.22,  # Logg et al. 2019
    ("algorithm_aversion", "anxiety"): 0.22,      # Castelo et al. 2019
    ("algorithm_aversion", "perceived_risk"): 0.28,  # Dietvorst et al. 2015

    # ===================================================================
    # EMOTION CORRELATIONS
    # Primary sources: Watson & Clark (1984); Diener et al. (1999)
    # ===================================================================
    ("positive_affect", "negative_affect"): -0.20,  # Watson et al. 1988
    ("positive_affect", "happiness"): 0.65,          # Diener et al. 1999
    ("positive_affect", "well_being"): 0.55,         # Diener et al. 1999
    ("positive_affect", "satisfaction"): 0.42,       # (duplicate removal - kept above)
    ("positive_affect", "mood"): 0.55,               # Watson & Clark 1984
    ("positive_affect", "self_esteem"): 0.42,        # Watson et al. 1988
    ("positive_affect", "engagement"): 0.42,         # Fredrickson 2001
    ("positive_affect", "intrinsic_motivation"): 0.38,  # Isen 2001
    ("positive_affect", "cooperation"): 0.28,        # Isen 2001
    ("positive_affect", "prosocial"): 0.30,          # Isen 2001
    ("negative_affect", "anxiety"): 0.58,            # Watson & Clark 1984
    ("negative_affect", "stress"): 0.55,             # Watson et al. 1988
    ("negative_affect", "anger"): 0.52,              # Watson & Clark 1984
    ("negative_affect", "fear"): 0.50,               # Watson & Clark 1984
    ("negative_affect", "disgust"): 0.42,            # Watson & Clark 1984
    ("negative_affect", "well_being"): -0.45,        # Diener et al. 1999
    ("negative_affect", "self_esteem"): -0.38,       # Watson et al. 1988
    ("negative_affect", "mood"): -0.45,              # Watson & Clark 1984
    ("negative_affect", "perceived_risk"): 0.35,     # Lerner & Keltner 2001
    ("happiness", "well_being"): 0.62,               # Diener et al. 1999
    ("happiness", "satisfaction"): 0.55,             # Diener et al. 1999
    ("happiness", "self_esteem"): 0.45,              # Diener & Diener 1995
    ("happiness", "intrinsic_motivation"): 0.38,     # Lyubomirsky et al. 2005
    ("happiness", "mood"): 0.58,                     # Diener et al. 1999
    ("happiness", "body_image"): 0.30,               # Stice 2002
    ("happiness", "social_identity"): 0.32,          # Haslam et al. 2009
    ("happiness", "cooperation"): 0.25,              # Lyubomirsky et al. 2005
    ("anxiety", "stress"): 0.65,                     # Clark & Watson 1991
    ("anxiety", "fear"): 0.58,                       # Barlow 2002
    ("anxiety", "anger"): 0.35,                      # Deffenbacher et al. 1996
    ("anxiety", "well_being"): -0.42,                # Headey & Wearing 1989
    ("anxiety", "self_esteem"): -0.40,               # Sowislo & Orth 2013
    ("anxiety", "competence"): -0.30,                # Eysenck et al. 2007
    ("anxiety", "cognitive_load"): 0.30,             # Eysenck et al. 2007
    ("anger", "stress"): 0.42,                       # Spielberger 1999
    ("anger", "fear"): 0.32,                         # Lerner & Keltner 2001
    ("anger", "perceived_fairness"): -0.35,          # Folger & Cropanzano 1998
    ("anger", "disgust"): 0.35,                      # Rozin et al. 1999
    ("anger", "self_control"): -0.32,                # Denson et al. 2012
    ("fear", "perceived_risk"): 0.42,                # Lerner & Keltner 2001
    ("fear", "compliance"): 0.30,                    # Witte & Allen 2000
    ("fear", "self_efficacy"): -0.30,                # (duplicate - kept above)
    ("fear", "stress"): 0.45,                        # Barlow 2002
    ("fear", "disgust"): 0.30,                       # Rozin et al. 1999
    ("disgust", "perceived_risk"): 0.28,             # Rozin et al. 1999
    ("mood", "well_being"): 0.48,                    # Diener et al. 1999
    ("mood", "satisfaction"): 0.40,                  # Schwarz & Clore 1983
    ("mood", "self_esteem"): 0.32,                   # Watson et al. 1988

    # ===================================================================
    # PERCEIVED RISK CLUSTER
    # Primary sources: Mitchell (1999); Pavlou (2003)
    # ===================================================================
    ("perceived_risk", "purchase_intent"): -0.35,   # Mitchell 1999
    ("perceived_risk", "anxiety"): 0.38,            # Mitchell 1999
    ("perceived_risk", "perceived_quality"): -0.25, # Mitchell 1999
    ("perceived_risk", "perceived_value"): -0.30,   # Sweeney et al. 1999
    ("perceived_risk", "fear"): 0.42,               # (duplicate - kept above)
    ("perceived_risk", "satisfaction"): -0.30,      # Sweeney et al. 1999
    ("perceived_risk", "use_intent"): -0.32,        # Pavlou 2003
    ("perceived_risk", "behavioral_intent"): -0.30, # Mitchell 1999
    ("perceived_risk", "knowledge"): -0.22,         # Mitchell 1999

    # ===================================================================
    # PERCEIVED QUALITY / VALUE CLUSTER
    # Primary sources: Cronin et al. (2000); Zeithaml (1988)
    # ===================================================================
    ("perceived_quality", "perceived_value"): 0.55,    # Cronin et al. 2000
    ("perceived_quality", "purchase_intent"): 0.42,    # Dodds et al. 1991
    ("perceived_quality", "loyalty"): 0.38,            # Zeithaml et al. 1996
    ("perceived_quality", "credibility"): 0.45,        # Erdem & Swait 2004
    ("perceived_quality", "brand_attitude"): 0.45,     # Aaker 1991
    ("perceived_quality", "recommendation"): 0.38,     # Zeithaml et al. 1996
    ("perceived_value", "purchase_intent"): 0.48,      # Cronin et al. 2000
    ("perceived_value", "loyalty"): 0.42,              # Sirdeshmukh et al. 2002
    ("perceived_value", "recommendation"): 0.40,       # Cronin et al. 2000
    ("perceived_value", "behavioral_intent"): 0.45,    # Cronin et al. 2000
    ("perceived_value", "brand_attitude"): 0.42,       # Aaker 1991

    # ===================================================================
    # SOCIAL PSYCHOLOGY CORRELATIONS
    # Primary sources: Cialdini & Goldstein (2004); Batson (2011)
    # ===================================================================
    ("social_norm", "conformity"): 0.52,           # Cialdini & Goldstein 2004
    ("social_norm", "compliance"): 0.45,           # Cialdini 2001
    ("social_norm", "behavioral_intent"): 0.38,   # Ajzen 1991
    ("social_norm", "attitude"): 0.30,             # Armitage & Conner 2001
    ("social_norm", "social_identity"): 0.40,      # Terry & Hogg 1996
    ("social_norm", "cooperation"): 0.32,          # Ostrom 2000
    ("social_norm", "purchase_intent"): 0.28,      # Ajzen 1991
    ("empathy", "prosocial"): 0.48,                # Eisenberg & Miller 1987
    ("empathy", "cooperation"): 0.38,              # Batson 2011
    ("empathy", "well_being"): 0.28,               # Wei et al. 2011
    ("empathy", "leadership"): 0.32,               # Kellett et al. 2002
    ("empathy", "trust"): 0.28,                    # (duplicate - kept above)
    ("empathy", "social_identity"): 0.30,          # Stuermer et al. 2006
    ("prosocial", "cooperation"): 0.42,            # Penner et al. 2005
    ("prosocial", "social_identity"): 0.35,        # Levine et al. 2005
    ("prosocial", "positive_affect"): 0.30,        # (duplicate - kept above)
    ("prosocial", "well_being"): 0.30,             # Dunn et al. 2008
    ("prosocial", "self_esteem"): 0.25,            # Penner et al. 2005
    ("prosocial", "perceived_fairness"): 0.30,     # Van Lange et al. 1997
    ("social_identity", "conformity"): 0.42,       # Terry & Hogg 1996
    ("social_identity", "commitment"): 0.38,       # Ellemers et al. 2002
    ("social_identity", "cooperation"): 0.35,      # Tyler & Blader 2001
    ("social_identity", "well_being"): 0.30,       # Haslam et al. 2009
    ("social_identity", "self_esteem"): 0.32,      # Luhtanen & Crocker 1992
    ("conformity", "compliance"): 0.42,            # Cialdini & Goldstein 2004
    ("conformity", "social_norm"): 0.52,           # (duplicate - kept above)

    # ===================================================================
    # SELF-CONCEPT CORRELATIONS
    # Primary sources: Judge & Bono (2001); Deci & Ryan (2000)
    # ===================================================================
    ("self_esteem", "well_being"): 0.50,           # Diener & Diener 1995
    ("self_esteem", "body_image"): 0.45,           # Harter 1999
    ("self_esteem", "social_identity"): 0.32,      # (duplicate - kept above)
    ("self_esteem", "identity"): 0.42,             # Swann et al. 2007
    ("self_esteem", "autonomy"): 0.35,             # Ryan & Deci 2000
    ("self_esteem", "coping"): 0.35,               # Zeigler-Hill 2011
    ("self_esteem", "competence"): 0.45,           # Judge & Bono 2001
    ("self_control", "compliance"): 0.25,          # Tangney et al. 2004
    ("self_control", "health_behavior"): 0.35,     # Tangney et al. 2004
    ("self_control", "well_being"): 0.32,          # Tangney et al. 2004
    ("self_control", "stress"): -0.30,             # Tangney et al. 2004
    ("self_control", "negative_affect"): -0.28,    # Tangney et al. 2004
    ("self_control", "prosocial"): 0.20,           # DeWall et al. 2008
    ("identity", "social_identity"): 0.48,         # Brewer & Gardner 1996
    ("identity", "autonomy"): 0.35,                # Ryan & Deci 2000
    ("identity", "commitment"): 0.38,              # Burke & Reitzes 1991
    ("identity", "engagement"): 0.32,              # Kahn 1990
    ("autonomy", "well_being"): 0.42,              # Deci & Ryan 2000
    ("autonomy", "intrinsic_motivation"): 0.52,    # Deci & Ryan 2000
    ("autonomy", "positive_affect"): 0.35,         # Sheldon et al. 1996
    ("autonomy", "engagement"): 0.38,              # Bakker & Demerouti 2007
    ("autonomy", "satisfaction"): 0.38,            # Humphrey et al. 2007

    # ===================================================================
    # MOTIVATION CORRELATIONS
    # Primary sources: Deci & Ryan (2000); Cacioppo et al. (1996)
    # ===================================================================
    ("intrinsic_motivation", "engagement"): 0.52,        # Deci & Ryan 2000
    ("intrinsic_motivation", "well_being"): 0.38,        # Ryan & Deci 2000
    ("intrinsic_motivation", "need_for_cognition"): 0.40,  # Cacioppo et al. 1996
    ("intrinsic_motivation", "extrinsic_motivation"): -0.15,  # Deci et al. 1999
    ("intrinsic_motivation", "satisfaction"): 0.40,      # Vallerand 1997
    ("intrinsic_motivation", "positive_affect"): 0.38,   # (duplicate - kept above)
    ("intrinsic_motivation", "competence"): 0.42,        # Deci & Ryan 2000
    ("intrinsic_motivation", "knowledge"): 0.32,         # Pintrich 2003
    ("intrinsic_motivation", "self_esteem"): 0.30,       # Vallerand 1997
    ("extrinsic_motivation", "compliance"): 0.30,        # Deci et al. 1999
    ("extrinsic_motivation", "purchase_intent"): 0.28,   # Chandon et al. 2000
    ("extrinsic_motivation", "engagement"): 0.22,        # Deci & Ryan 2000
    ("extrinsic_motivation", "behavioral_intent"): 0.30, # Vallerand 1997
    ("extrinsic_motivation", "stress"): 0.18,            # Deci & Ryan 2000
    ("need_for_cognition", "knowledge"): 0.35,           # Cacioppo et al. 1996
    ("need_for_cognition", "cognitive_load"): -0.20,     # Cacioppo et al. 1996
    ("need_for_cognition", "competence"): 0.30,          # Cacioppo et al. 1996
    ("need_for_cognition", "attitude"): 0.22,            # Haugtvedt & Petty 1992
    ("need_for_cognition", "perceived_quality"): 0.18,   # Cacioppo et al. 1996

    # ===================================================================
    # HEALTH CORRELATIONS
    # Primary sources: Schwarzer (2008); Lazarus & Folkman (1984)
    # ===================================================================
    ("health_behavior", "well_being"): 0.35,       # Penedo & Dahn 2005
    ("health_behavior", "coping"): 0.30,           # Schwarzer 2008
    ("health_behavior", "self_control"): 0.35,     # (duplicate - kept above)
    ("health_behavior", "positive_affect"): 0.25,  # Pressman & Cohen 2005
    ("health_behavior", "autonomy"): 0.28,         # Ryan et al. 2008
    ("health_behavior", "intrinsic_motivation"): 0.32,  # Ryan et al. 2008
    ("health_behavior", "stress"): -0.25,          # Penedo & Dahn 2005
    ("health_behavior", "knowledge"): 0.25,        # Conner & Norman 2005
    ("health_anxiety", "anxiety"): 0.55,           # Salkovskis et al. 2002
    ("health_anxiety", "perceived_risk"): 0.42,    # Salkovskis et al. 2002
    ("health_anxiety", "stress"): 0.40,            # Abramowitz et al. 2007
    ("health_anxiety", "health_behavior"): 0.28,   # Asmundson et al. 2010
    ("health_anxiety", "negative_affect"): 0.42,   # Fergus 2014
    ("health_anxiety", "fear"): 0.48,              # Barlow 2002
    ("health_anxiety", "well_being"): -0.35,       # Salkovskis et al. 2002
    ("health_anxiety", "self_efficacy"): -0.25,    # Abramowitz et al. 2007
    ("body_image", "self_esteem"): 0.45,           # (duplicate - kept above)
    ("body_image", "well_being"): 0.35,            # Stice 2002
    ("body_image", "anxiety"): -0.32,              # Cash & Pruzinsky 2002
    ("body_image", "negative_affect"): -0.35,      # Stice 2002
    ("body_image", "social_identity"): 0.22,       # Stice 2002
    ("coping", "well_being"): 0.40,                # Folkman & Moskowitz 2004
    ("coping", "stress"): -0.35,                   # Lazarus & Folkman 1984
    ("coping", "positive_affect"): 0.30,           # Folkman & Moskowitz 2004
    ("coping", "self_esteem"): 0.35,               # (duplicate - kept above)
    ("coping", "social_identity"): 0.22,           # Haslam et al. 2005

    # ===================================================================
    # CREDIBILITY / COMPETENCE CLUSTER
    # Primary sources: Ohanian (1990); Hovland et al. (1953)
    # ===================================================================
    ("credibility", "competence"): 0.55,           # Ohanian 1990
    ("credibility", "attitude"): 0.42,             # Petty & Cacioppo 1986
    ("credibility", "purchase_intent"): 0.35,      # Ohanian 1990
    ("credibility", "recommendation"): 0.38,       # Ohanian 1990
    ("credibility", "brand_attitude"): 0.42,       # Erdem & Swait 2004
    ("credibility", "perceived_value"): 0.35,      # Erdem & Swait 2004
    ("credibility", "satisfaction"): 0.35,         # Erdem & Swait 2004
    ("competence", "trust"): 0.52,                 # (duplicate - kept above)
    ("competence", "leadership"): 0.42,            # Judge et al. 2002
    ("competence", "self_esteem"): 0.45,           # (duplicate - kept above)
    ("competence", "knowledge"): 0.48,             # Ericsson et al. 1993
    ("competence", "intrinsic_motivation"): 0.42,  # (duplicate - kept above)

    # ===================================================================
    # ENGAGEMENT CLUSTER
    # Primary sources: Brodie et al. (2013); Saks (2006)
    # ===================================================================
    ("engagement", "loyalty"): 0.48,               # Brodie et al. 2013
    ("engagement", "commitment"): 0.52,            # Saks 2006
    ("engagement", "behavioral_intent"): 0.45,     # Brodie et al. 2013
    ("engagement", "recommendation"): 0.42,        # Brodie et al. 2013
    ("engagement", "liking"): 0.40,                # Brodie et al. 2013
    ("engagement", "well_being"): 0.38,            # Saks 2006
    ("engagement", "knowledge"): 0.28,             # Brodie et al. 2013

    # ===================================================================
    # ATTITUDE-BEHAVIOR LINK
    # Primary sources: Ajzen (1991); Armitage & Conner (2001)
    # ===================================================================
    ("attitude", "behavioral_intent"): 0.52,       # Armitage & Conner 2001
    ("attitude", "purchase_intent"): 0.48,         # Ajzen 1991
    ("attitude", "preference"): 0.55,              # Ajzen 1991
    ("attitude", "liking"): 0.60,                  # Ajzen 1991
    ("attitude", "loyalty"): 0.42,                 # Dick & Basu 1994
    ("attitude", "recommendation"): 0.40,          # Brown et al. 2005
    ("attitude", "use_intent"): 0.48,              # Davis 1989
    ("attitude", "compliance"): 0.35,              # Ajzen 1991
    ("preference", "purchase_intent"): 0.52,       # Ajzen 1991
    ("preference", "liking"): 0.58,                # Zajonc 1968
    ("preference", "loyalty"): 0.42,               # Dick & Basu 1994
    ("preference", "recommendation"): 0.38,        # Brown et al. 2005
    ("preference", "behavioral_intent"): 0.48,     # Ajzen 1991
    ("liking", "loyalty"): 0.42,                   # Dick & Basu 1994
    ("liking", "recommendation"): 0.40,            # Brown et al. 2005
    ("liking", "purchase_intent"): 0.45,           # Zajonc 1968
    ("liking", "trust"): 0.38,                     # Swan et al. 1999
    ("liking", "behavioral_intent"): 0.42,         # Ajzen 1991

    # ===================================================================
    # COMMITMENT / LOYALTY CLUSTER
    # Primary sources: Morgan & Hunt (1994); Oliver (1999)
    # ===================================================================
    ("commitment", "loyalty"): 0.55,               # Morgan & Hunt 1994
    ("commitment", "behavioral_intent"): 0.48,     # Morgan & Hunt 1994
    ("commitment", "recommendation"): 0.42,        # Fullerton 2003
    ("commitment", "engagement"): 0.52,            # (duplicate - kept above)
    ("commitment", "compliance"): 0.35,            # Meyer & Herscovitch 2001
    ("loyalty", "recommendation"): 0.55,           # Reichheld 2003
    ("loyalty", "purchase_intent"): 0.52,          # Oliver 1999
    ("loyalty", "behavioral_intent"): 0.50,        # Oliver 1999
    ("loyalty", "use_intent"): 0.45,               # Bhattacherjee 2001
    ("compliance", "behavioral_intent"): 0.45,     # Cialdini 2001

    # ===================================================================
    # CROSS-DOMAIN STRESS / WELL-BEING
    # Primary sources: Cohen et al. (1983); Diener et al. (1999)
    # ===================================================================
    ("stress", "well_being"): -0.45,               # Cohen et al. 1983
    ("stress", "self_esteem"): -0.32,              # Lazarus & Folkman 1984
    ("stress", "self_control"): -0.30,             # (duplicate - kept above)
    ("stress", "health_behavior"): -0.25,          # (duplicate - kept above)
    ("stress", "satisfaction"): -0.35,             # Podsakoff et al. 2007
    ("stress", "cognitive_load"): 0.35,            # Karasek 1979
    ("stress", "mood"): -0.40,                     # Cohen et al. 1983

    # ===================================================================
    # LEADERSHIP CLUSTER
    # Primary sources: Judge et al. (2004); Avolio et al. (2004)
    # ===================================================================
    ("leadership", "trust"): 0.42,                 # Dirks & Ferrin 2002
    ("leadership", "work_engagement"): 0.40,       # Bakker & Demerouti 2007
    ("leadership", "satisfaction"): 0.35,          # Judge et al. 2004
    ("leadership", "cooperation"): 0.32,           # Yukl 2012
    ("leadership", "self_efficacy"): 0.30,         # Avolio et al. 2004
    ("leadership", "positive_affect"): 0.28,       # Bono & Judge 2003

    # ===================================================================
    # BRAND ATTITUDE CLUSTER
    # Primary sources: Aaker (1991); Keller (1993)
    # ===================================================================
    ("brand_attitude", "loyalty"): 0.48,           # Chaudhuri & Holbrook 2001
    ("brand_attitude", "purchase_intent"): 0.50,   # Mitchell & Olson 1981
    ("brand_attitude", "trust"): 0.52,             # Chaudhuri & Holbrook 2001
    ("brand_attitude", "recommendation"): 0.42,    # Brown et al. 2005
    ("brand_attitude", "satisfaction"): 0.48,      # Oliver 1999
    ("brand_attitude", "attitude"): 0.62,          # Keller 1993
    ("brand_attitude", "liking"): 0.55,            # Mitchell & Olson 1981
    ("brand_attitude", "engagement"): 0.42,        # Hollebeek et al. 2014
    ("brand_attitude", "behavioral_intent"): 0.48, # Keller 1993
    ("brand_attitude", "preference"): 0.52,        # Keller 1993

    # ===================================================================
    # ADDITIONAL CROSS-DOMAIN CORRELATIONS
    # ===================================================================
    ("use_intent", "behavioral_intent"): 0.62,     # Venkatesh et al. 2003
    ("use_intent", "perceived_quality"): 0.38,     # Venkatesh et al. 2003
    ("use_intent", "recommendation"): 0.42,        # Bhattacherjee 2001
    ("use_intent", "satisfaction"): 0.45,          # (duplicate - kept above)
    ("cognitive_load", "anxiety"): 0.30,           # (duplicate - kept above)
    ("cognitive_load", "satisfaction"): -0.25,     # Sweller 1988
    ("cognitive_load", "perceived_quality"): -0.20,  # Paas et al. 2003
    ("cognitive_load", "knowledge"): -0.18,        # Sweller 1988
    ("cognitive_load", "self_efficacy"): -0.22,    # Sweller 1988
    ("knowledge", "perceived_risk"): -0.22,        # (duplicate - kept above)
    ("knowledge", "attitude"): 0.28,               # Fabrigar et al. 2006
    ("knowledge", "behavioral_intent"): 0.25,      # Ajzen 1991
    ("well_being", "autonomy"): 0.42,              # (duplicate - kept above)
    ("well_being", "competence"): 0.38,            # Deci & Ryan 2000
    ("well_being", "self_control"): 0.32,          # (duplicate - kept above)
    ("well_being", "cooperation"): 0.25,           # Lyubomirsky et al. 2005
    ("perceived_fairness", "satisfaction"): 0.45,  # Colquitt et al. 2001
    ("perceived_fairness", "commitment"): 0.42,    # Colquitt et al. 2001
    ("perceived_fairness", "cooperation"): 0.35,   # Colquitt et al. 2001
    ("perceived_fairness", "prosocial"): 0.30,     # Van Lange et al. 1997
    ("perceived_fairness", "loyalty"): 0.35,       # Tax et al. 1998
    ("perceived_fairness", "behavioral_intent"): 0.38,  # Colquitt et al. 2001

    # ===================================================================
    # WORK ENGAGEMENT EXTRA
    # Primary sources: Bakker & Demerouti (2007); Schaufeli et al. (2002)
    # ===================================================================
    ("work_engagement", "intrinsic_motivation"): 0.48,  # Bakker & Demerouti 2007
    ("work_engagement", "autonomy"): 0.42,         # Bakker & Demerouti 2007
    ("work_engagement", "positive_affect"): 0.45,  # Schaufeli et al. 2002
    ("work_engagement", "well_being"): 0.45,       # Bakker & Demerouti 2007
    ("work_engagement", "self_esteem"): 0.32,      # Xanthopoulou et al. 2007
    ("work_engagement", "cooperation"): 0.30,      # Bakker & Demerouti 2007
    ("work_engagement", "negative_affect"): -0.30, # Schaufeli et al. 2002
    ("work_engagement", "stress"): -0.28,          # Bakker & Demerouti 2007
}


# ---------------------------------------------------------------------------
# 3. FUNCTIONS
# ---------------------------------------------------------------------------

def detect_construct_types(scales: List[Dict[str, Any]]) -> Dict[str, str]:
    """Detect the construct type for each scale based on its name and question text.

    Uses a two-pass matching strategy:
        1. Multi-word phrases (highest specificity) are checked first.
        2. Single keywords are checked second.
    Falls back to ``"general"`` if no match is found.

    Args:
        scales: A list of scale dictionaries. Each must have a ``"name"`` key
            and may optionally have a ``"question_text"`` key.

    Returns:
        A mapping of ``{scale_name: construct_type}``.
    """
    result: Dict[str, str] = {}

    # Pre-sort keywords by length (longest first) for each construct so that
    # multi-word phrases get priority over single keywords.
    sorted_keywords: Dict[str, List[str]] = {}
    for construct, keywords in CONSTRUCT_KEYWORDS.items():
        sorted_keywords[construct] = sorted(keywords, key=len, reverse=True)

    for scale in scales:
        name: str = scale.get("name", "")
        question_text: str = scale.get("question_text", "")

        # Build combined searchable strings (lower-cased) in two forms:
        #   - underscored: spaces and hyphens become underscores
        #   - spaced: underscores and hyphens become spaces
        # This ensures "perceived_risk" matches "Perceived Risk" and vice versa.
        raw = f"{name} {question_text}".lower()
        search_underscored = raw.replace("-", "_").replace(" ", "_")
        search_spaced = raw.replace("-", " ").replace("_", " ")

        matched_construct: Optional[str] = None
        best_keyword_len: int = 0

        for construct, keywords in sorted_keywords.items():
            for kw in keywords:
                kw_lower = kw.lower()
                kw_spaced = kw_lower.replace("_", " ")
                # Check against both normalised forms
                if (
                    kw_lower in search_underscored
                    or kw_spaced in search_spaced
                ) and len(kw_lower) > best_keyword_len:
                    matched_construct = construct
                    best_keyword_len = len(kw_lower)

        result[name] = matched_construct if matched_construct else "general"

    return result


def nearest_positive_definite(A: np.ndarray) -> np.ndarray:
    """Find the nearest positive-definite matrix to *A* using the method of
    Higham (2002) — iterative projection onto the cone of symmetric
    positive-definite matrices.

    Reference:
        Higham, N. J. (2002). Computing the nearest correlation matrix —
        a problem from finance. *IMA Journal of Numerical Analysis*,
        22(3), 329--343.

    Args:
        A: A symmetric matrix (n x n) that may not be positive-definite.

    Returns:
        The nearest symmetric positive-definite matrix of the same shape.
    """
    n = A.shape[0]
    if n == 0:
        return A.copy()

    # Symmetrise
    B = (A + A.T) / 2.0

    # Compute the symmetric polar factor H of B
    _, S, Vt = np.linalg.svd(B)
    H = Vt.T @ np.diag(S) @ Vt

    A_hat = (B + H) / 2.0

    # Ensure symmetry
    A_hat = (A_hat + A_hat.T) / 2.0

    # Iteratively adjust until positive-definite
    max_iterations = 100
    spacing = np.spacing(np.linalg.norm(A_hat))

    I = np.eye(n)  # noqa: E741
    k = 0
    while True:
        try:
            np.linalg.cholesky(A_hat)
            break  # Already positive-definite
        except np.linalg.LinAlgError:
            pass

        k += 1
        if k > max_iterations:
            # Last resort: add a small diagonal perturbation
            min_eig = np.min(np.real(np.linalg.eigvalsh(A_hat)))
            A_hat += I * (-min_eig * (k ** 2) + spacing)
            break

        min_eig = np.min(np.real(np.linalg.eigvalsh(A_hat)))
        A_hat += I * (-min_eig * (k ** 2) + spacing)

    # Force diagonal to 1 (correlation matrix)
    d = np.sqrt(np.diag(A_hat))
    d[d == 0] = 1.0
    A_hat = A_hat / np.outer(d, d)
    np.fill_diagonal(A_hat, 1.0)

    # Final symmetry enforcement
    A_hat = (A_hat + A_hat.T) / 2.0

    # Clip to [-1, 1]
    np.clip(A_hat, -1.0, 1.0, out=A_hat)

    return A_hat


def _lookup_correlation(construct_a: str, construct_b: str) -> Optional[float]:
    """Look up the empirical correlation between two construct types.

    Checks both orderings ``(a, b)`` and ``(b, a)``.

    Args:
        construct_a: First construct type identifier.
        construct_b: Second construct type identifier.

    Returns:
        The correlation value if found, otherwise ``None``.
    """
    r = EMPIRICAL_CORRELATIONS.get((construct_a, construct_b))
    if r is not None:
        return r
    return EMPIRICAL_CORRELATIONS.get((construct_b, construct_a))


def _get_category(construct_type: str) -> Optional[str]:
    """Return the broad category for a given construct type.

    Args:
        construct_type: A construct type identifier (e.g., ``"trust"``).

    Returns:
        The category name (e.g., ``"attitudes"``) or ``None``.
    """
    return _CONSTRUCT_TO_CATEGORY.get(construct_type)


def infer_correlation_matrix(
    scales: List[Dict[str, Any]],
    construct_types: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Build a positive-definite correlation matrix for the given scales.

    Workflow:
        1. Detect construct types if not provided.
        2. For each scale pair, look up the empirical correlation.
        3. For unmatched pairs, apply intelligent defaults:
            - Same construct type on both scales: ``r = 0.65``
            - Same broad category, different constructs: ``r = 0.35``
            - Different categories: ``r = 0.15``
        4. Ensure the resulting matrix is positive-definite via
           :func:`nearest_positive_definite` (Higham 2002).

    Args:
        scales: A list of scale dictionaries with at least a ``"name"`` key.
        construct_types: Optional pre-computed mapping of
            ``{scale_name: construct_type}``. If ``None``, construct types
            are detected automatically.

    Returns:
        A tuple of ``(correlation_matrix, scale_names)`` where
        *correlation_matrix* is a positive-definite ``np.ndarray`` of shape
        ``(n, n)`` and *scale_names* is a list of the scale name strings.
    """
    if not scales:
        return np.array([]).reshape(0, 0), []

    if construct_types is None:
        construct_types = detect_construct_types(scales)

    scale_names = [s.get("name", f"scale_{i}") for i, s in enumerate(scales)]
    n = len(scale_names)

    if n == 1:
        return np.array([[1.0]]), scale_names

    # Initialise with identity
    corr = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            ct_i = construct_types.get(scale_names[i], "general")
            ct_j = construct_types.get(scale_names[j], "general")

            # 1. Try empirical lookup
            r = _lookup_correlation(ct_i, ct_j)

            if r is not None:
                corr[i, j] = r
                corr[j, i] = r
                continue

            # 2. Intelligent defaults
            if ct_i == ct_j and ct_i != "general":
                # Same construct type (e.g., two trust scales)
                r_default = 0.65
            else:
                cat_i = _get_category(ct_i)
                cat_j = _get_category(ct_j)
                if cat_i and cat_j and cat_i == cat_j:
                    # Same broad category
                    r_default = 0.35
                else:
                    # Different categories or unknown
                    r_default = 0.15

            corr[i, j] = r_default
            corr[j, i] = r_default

    # Ensure positive-definiteness
    try:
        np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        corr = nearest_positive_definite(corr)

    return corr, scale_names


def generate_latent_scores(
    n_participants: int,
    correlation_matrix: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Generate multivariate normal latent scores with the specified
    correlation structure using Cholesky decomposition.

    Args:
        n_participants: Number of participants (rows).
        correlation_matrix: A positive-definite correlation matrix of shape
            ``(k, k)`` where *k* is the number of scales/DVs.
        seed: Random seed for reproducibility.

    Returns:
        An array of shape ``(n_participants, k)`` containing standard-normal
        z-scores with the desired inter-scale correlations.

    Raises:
        ValueError: If *n_participants* < 1 or *correlation_matrix* is empty.
        np.linalg.LinAlgError: If the correlation matrix is not
            positive-definite (call :func:`nearest_positive_definite` first).
    """
    if n_participants < 1:
        raise ValueError("n_participants must be >= 1")

    k = correlation_matrix.shape[0]
    if k == 0:
        raise ValueError("correlation_matrix must not be empty")

    rng = np.random.default_rng(seed)

    # Cholesky decomposition: C = L @ L.T
    L = np.linalg.cholesky(correlation_matrix)

    # Generate independent standard normals
    Z = rng.standard_normal(size=(n_participants, k))

    # Correlate via Cholesky factor
    X = Z @ L.T

    return X


def get_correlation_summary(
    correlation_matrix: np.ndarray,
    scale_names: List[str],
    construct_types: Dict[str, str],
) -> str:
    """Return a human-readable summary of the inferred correlations.

    Lists each unique scale pair with its inferred correlation, construct
    types, and whether the correlation was from an empirical source or a
    default value.

    Args:
        correlation_matrix: The correlation matrix (n x n).
        scale_names: List of scale names corresponding to matrix rows/columns.
        construct_types: Mapping of ``{scale_name: construct_type}``.

    Returns:
        A formatted multi-line string summarising all pairwise correlations.
    """
    n = len(scale_names)
    if n == 0:
        return "No scales provided."

    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("CORRELATION MATRIX SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    # Scale listing
    lines.append("Scales and detected construct types:")
    lines.append("-" * 40)
    for name in scale_names:
        ct = construct_types.get(name, "general")
        cat = _get_category(ct) or "unknown"
        lines.append(f"  {name}: {ct} ({cat})")
    lines.append("")

    # Pairwise correlations
    lines.append("Pairwise correlations:")
    lines.append("-" * 40)

    empirical_count = 0
    default_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            r = correlation_matrix[i, j]
            ct_i = construct_types.get(scale_names[i], "general")
            ct_j = construct_types.get(scale_names[j], "general")

            # Check if this was from empirical data
            empirical_r = _lookup_correlation(ct_i, ct_j)
            if empirical_r is not None:
                source = "empirical"
                empirical_count += 1
            else:
                source = "default"
                default_count += 1

            lines.append(
                f"  {scale_names[i]} <-> {scale_names[j]}: "
                f"r = {r:+.3f}  [{ct_i} <-> {ct_j}]  ({source})"
            )

    lines.append("")
    lines.append(f"Total pairs: {empirical_count + default_count}")
    lines.append(f"  Empirically-informed: {empirical_count}")
    lines.append(f"  Default values: {default_count}")
    lines.append("")

    # Matrix positive-definiteness check
    try:
        np.linalg.cholesky(correlation_matrix)
        lines.append("Matrix status: positive-definite (valid for simulation)")
    except np.linalg.LinAlgError:
        lines.append("Matrix status: NOT positive-definite (needs projection)")

    lines.append("=" * 70)

    return "\n".join(lines)
