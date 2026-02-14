"""
Scientific Knowledge Base — Structured empirical calibrations for behavioral simulation.
=========================================================================================

v1.0.8.7: NEW MODULE — Externalizes all scientific calibrations from hardcoded values
into a queryable, auditable database structure.

This module provides:
1. META_ANALYTIC_DB: Effect sizes with confidence intervals and moderators
2. GAME_CALIBRATIONS: Economic game baselines from published meta-analyses
3. CONSTRUCT_NORMS: Published means/SDs for common psychological constructs
4. CULTURAL_ADJUSTMENTS: Cross-cultural calibration factors (WEIRD vs non-WEIRD)
5. RESPONSE_TIME_NORMS: Empirical response time distributions by engagement level
6. ORDER_EFFECT_MODELS: Fatigue, learning, and carryover effect parameters
7. IMPLICIT_MEASURE_PARAMS: IAT D-score and implicit measure calibrations

Every entry includes:
- source: Author (Year) citation
- n_studies or n_participants: Sample size (for meta-analyses vs single studies)
- effect_d or mean/sd: The empirical calibration value
- ci_95: 95% confidence interval [lower, upper]
- moderators: Dict of moderator-specific adjustments
- domain: Research domain tag
- replication_status: "replicated", "contested", "original_only"

Architecture note:
- This module is IMPORTED by enhanced_simulation_engine.py
- The engine uses get_effect_size() and get_calibration() to look up values
- When a study matches multiple entries, the most specific match wins
- Confidence intervals enable uncertainty-aware simulation (sample from CI)

References are organized by research area with full APA-style citations in comments.
"""
from __future__ import annotations

__version__ = "1.1.0.3"

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import random
import math


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MetaAnalyticEffect:
    """A single meta-analytic effect size with full metadata."""
    source: str                        # "Engel (2011)" — primary citation
    effect_d: float                    # Cohen's d point estimate
    ci_95: Tuple[float, float]         # 95% CI [lower, upper]
    n_studies: int = 0                 # k studies in meta-analysis
    n_participants: int = 0            # Total N across studies
    domain: str = ""                   # Research domain tag
    construct: str = ""                # Psychological construct measured
    paradigm: str = ""                 # Experimental paradigm
    heterogeneity_tau: float = 0.0     # Between-study heterogeneity (τ)
    i_squared: float = 0.0            # I² heterogeneity percentage
    replication_status: str = "replicated"  # replicated/contested/original_only
    moderators: Dict[str, Dict[str, float]] = field(default_factory=dict)
    notes: str = ""

    def sample_effect(self, rng: Optional[random.Random] = None) -> float:
        """Sample an effect size from the meta-analytic distribution.

        Uses the heterogeneity estimate (τ) to model between-study variation.
        If τ is available, samples from N(d, τ²). Otherwise, samples uniformly
        from the 95% CI.
        """
        _rng = rng or random.Random()
        if self.heterogeneity_tau > 0:
            # Sample from the random-effects distribution
            return _rng.gauss(self.effect_d, self.heterogeneity_tau)
        else:
            # Sample uniformly from the CI
            return _rng.uniform(self.ci_95[0], self.ci_95[1])

    def get_moderated_effect(self, moderator_key: str, moderator_value: str) -> float:
        """Get effect size adjusted for a specific moderator level."""
        if moderator_key in self.moderators:
            mod_dict = self.moderators[moderator_key]
            if moderator_value in mod_dict:
                return mod_dict[moderator_value]
        return self.effect_d


@dataclass
class ConstructNorm:
    """Published norms for a psychological construct on a standard scale."""
    source: str
    construct: str
    scale_name: str               # e.g., "UCLA Loneliness Scale", "SWLS"
    scale_points: int             # e.g., 7 for 1-7 Likert
    mean: float                   # Published mean
    sd: float                     # Published SD
    ci_95_mean: Tuple[float, float] = (0.0, 0.0)
    skewness: float = 0.0        # Distribution skew
    kurtosis: float = 0.0        # Distribution kurtosis
    sample_type: str = "general"  # "student", "clinical", "general", "mturk"
    culture: str = "western"      # "western", "east_asian", "global", etc.
    n_participants: int = 0
    moderators: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class GameCalibration:
    """Empirical calibration for an economic game paradigm."""
    source: str
    game_type: str               # "dictator", "trust", "ultimatum", etc.
    variant: str = "standard"    # "standard", "taking", "third_party", etc.
    mean_proportion: float = 0.0  # Mean allocation as proportion of endowment
    sd_proportion: float = 0.0
    ci_95: Tuple[float, float] = (0.0, 0.0)
    distribution_shape: str = "bimodal"  # "bimodal", "normal", "right_skew"
    modes: List[float] = field(default_factory=list)  # Distribution modes
    n_studies: int = 0
    n_participants: int = 0
    subpopulations: Dict[str, float] = field(default_factory=dict)  # type → proportion
    moderators: Dict[str, Dict[str, float]] = field(default_factory=dict)
    notes: str = ""


@dataclass
class CulturalAdjustment:
    """Cross-cultural calibration factors."""
    source: str
    culture: str                  # "western", "east_asian", "latin_american", etc.
    construct: str                # What's being adjusted
    adjustment_factor: float      # Multiplicative adjustment to Western baseline
    adjustment_additive: float = 0.0  # Additive adjustment (on raw scale)
    n_cultures: int = 0
    n_participants: int = 0
    notes: str = ""


@dataclass
class ResponseTimeNorm:
    """Empirical response time parameters."""
    source: str
    item_type: str                # "likert", "slider", "open_ended", "iat"
    engagement_level: str         # "engaged", "satisficing", "careless"
    mean_ms: float                # Mean response time in milliseconds
    sd_ms: float                  # SD of response time
    distribution: str = "lognormal"  # "lognormal", "ex_gaussian", "normal"
    # Ex-Gaussian parameters (mu, sigma, tau)
    ex_gaussian_mu: float = 0.0
    ex_gaussian_sigma: float = 0.0
    ex_gaussian_tau: float = 0.0
    notes: str = ""


@dataclass
class OrderEffect:
    """Parameters for longitudinal/order effects within a survey."""
    source: str
    effect_type: str              # "fatigue", "learning", "carryover", "primacy", "recency"
    magnitude_per_item: float     # Effect size change per additional item
    onset_item: int = 0           # Item number where effect begins
    plateau_item: int = 0         # Item number where effect plateaus
    affects: str = ""             # "variance", "mean", "response_time", "all"
    notes: str = ""


# =============================================================================
# META-ANALYTIC EFFECT SIZE DATABASE
# =============================================================================
# Each entry represents a published meta-analysis or well-replicated finding.
# Effect sizes are Cohen's d unless otherwise noted.

META_ANALYTIC_DB: Dict[str, MetaAnalyticEffect] = {

    # ── ECONOMIC GAMES ──────────────────────────────────────────────────────

    "dictator_giving": MetaAnalyticEffect(
        source="Engel (2011)",
        effect_d=0.0,  # Not a comparison — this is a baseline distribution
        ci_95=(-0.05, 0.05),
        n_studies=616,
        n_participants=20813,
        domain="behavioral_economics",
        construct="dictator_game_giving",
        paradigm="dictator_game",
        heterogeneity_tau=0.15,
        i_squared=78.0,
        moderators={
            "stakes": {"low": 0.32, "medium": 0.28, "high": 0.24},
            "sample": {"students": 0.26, "non_students": 0.31, "mturk": 0.25},
            "anonymity": {"double_blind": 0.22, "single_blind": 0.30, "no_blind": 0.35},
            "endowment": {"small_1_10": 0.30, "medium_10_50": 0.28, "large_50_plus": 0.22},
        },
        notes="Mean giving ≈ 28% of endowment. Bimodal: modes at 0% and 50%."
    ),

    "dictator_taking": MetaAnalyticEffect(
        source="List (2007); Bardsley (2008)",
        effect_d=-0.45,  # Taking option reduces giving by d ≈ 0.45
        ci_95=(-0.60, -0.30),
        n_studies=12,
        n_participants=1847,
        domain="behavioral_economics",
        construct="dictator_game_taking",
        paradigm="dictator_game_taking",
        heterogeneity_tau=0.12,
        i_squared=65.0,
        moderators={
            "taking_frame": {"take": -0.50, "destroy": -0.35, "steal": -0.55},
        },
        notes="15-25% of participants take. Mean giving drops to 10-15%."
    ),

    "trust_game_sent": MetaAnalyticEffect(
        source="Johnson & Mislin (2011)",
        effect_d=0.0,
        ci_95=(-0.05, 0.05),
        n_studies=162,
        n_participants=23000,
        domain="behavioral_economics",
        construct="trust_game_amount_sent",
        paradigm="trust_game",
        heterogeneity_tau=0.10,
        i_squared=72.0,
        moderators={
            "stakes": {"low": 0.52, "medium": 0.50, "high": 0.47},
            "multiplier": {"2x": 0.45, "3x": 0.50, "4x": 0.55},
            "sample": {"students": 0.50, "non_students": 0.52, "mturk": 0.48},
        },
        notes="Mean sent ≈ 50% of endowment (Berg et al. 1995 baseline)."
    ),

    "trust_game_return": MetaAnalyticEffect(
        source="Johnson & Mislin (2011)",
        effect_d=0.0,
        ci_95=(-0.05, 0.05),
        n_studies=162,
        n_participants=23000,
        domain="behavioral_economics",
        construct="trust_game_return_ratio",
        paradigm="trust_game",
        heterogeneity_tau=0.08,
        i_squared=68.0,
        notes="Mean return ≈ 33% of received amount (less than sent)."
    ),

    "ultimatum_offer": MetaAnalyticEffect(
        source="Oosterbeek et al. (2004)",
        effect_d=0.0,
        ci_95=(-0.03, 0.03),
        n_studies=37,
        n_participants=7500,
        domain="behavioral_economics",
        construct="ultimatum_game_offer",
        paradigm="ultimatum_game",
        heterogeneity_tau=0.08,
        i_squared=60.0,
        moderators={
            "stakes": {"low": 0.45, "medium": 0.42, "high": 0.38},
            "culture": {"western": 0.42, "east_asian": 0.40, "indigenous": 0.48},
        },
        notes="Modal offer 40-50%. Offers below 20% rejected ~50% of the time."
    ),

    "public_goods_contribution": MetaAnalyticEffect(
        source="Zelmer (2003)",
        effect_d=0.0,
        ci_95=(-0.04, 0.04),
        n_studies=27,
        n_participants=5400,
        domain="behavioral_economics",
        construct="public_goods_contribution",
        paradigm="public_goods_game",
        heterogeneity_tau=0.12,
        i_squared=75.0,
        moderators={
            "mpcr": {"low_0.3": 0.35, "medium_0.5": 0.45, "high_0.75": 0.60},
            "group_size": {"2": 0.50, "4": 0.45, "10": 0.38},
            "punishment": {"no_punishment": 0.45, "peer_punishment": 0.75},
        },
        notes="Mean contribution ≈ 40-60% of endowment. Decays over rounds."
    ),

    "prisoners_dilemma_cooperation": MetaAnalyticEffect(
        source="Sally (1995)",
        effect_d=0.0,
        ci_95=(-0.04, 0.04),
        n_studies=130,
        n_participants=26000,
        domain="behavioral_economics",
        construct="cooperation_rate",
        paradigm="prisoners_dilemma",
        heterogeneity_tau=0.10,
        i_squared=70.0,
        moderators={
            "communication": {"no_comm": 0.40, "pre_play_comm": 0.65},
            "repetition": {"one_shot": 0.47, "repeated": 0.55},
        },
        notes="Mean cooperation rate ≈ 47%. Higher with communication."
    ),

    # ── NEW GAME PARADIGMS (v1.0.8.7) ─────────────────────────────────────

    "first_price_auction": MetaAnalyticEffect(
        source="Kagel & Levin (2015); Cox et al. (1988)",
        effect_d=0.0,
        ci_95=(-0.03, 0.03),
        n_studies=45,
        n_participants=3200,
        domain="behavioral_economics",
        construct="auction_bid",
        paradigm="first_price_sealed_bid",
        heterogeneity_tau=0.08,
        moderators={
            "n_bidders": {"2": 0.72, "5": 0.80, "10": 0.88},
            "experience": {"novice": 0.85, "experienced": 0.78},
        },
        notes="Overbidding relative to RNNE. Mean bid/value ≈ 0.72-0.88."
    ),

    "second_price_auction": MetaAnalyticEffect(
        source="Kagel & Levin (1993)",
        effect_d=0.0,
        ci_95=(-0.05, 0.05),
        n_studies=15,
        n_participants=1200,
        domain="behavioral_economics",
        construct="auction_bid",
        paradigm="second_price_auction",
        heterogeneity_tau=0.10,
        notes="Overbidding common despite dominant strategy to bid value."
    ),

    "all_pay_auction": MetaAnalyticEffect(
        source="Dechenaux et al. (2015)",
        effect_d=0.0,
        ci_95=(-0.05, 0.05),
        n_studies=20,
        n_participants=2000,
        domain="behavioral_economics",
        construct="auction_bid",
        paradigm="all_pay_auction",
        heterogeneity_tau=0.15,
        notes="Aggregate revenue close to Nash prediction but individual bids highly variable."
    ),

    "nash_bargaining": MetaAnalyticEffect(
        source="Roth (1995); Babcock & Loewenstein (1997)",
        effect_d=0.0,
        ci_95=(-0.05, 0.05),
        n_studies=30,
        n_participants=3000,
        domain="behavioral_economics",
        construct="bargaining_outcome",
        paradigm="nash_bargaining",
        heterogeneity_tau=0.12,
        moderators={
            "information": {"complete": 0.50, "asymmetric": 0.55},
            "deadline": {"no_deadline": 0.48, "deadline": 0.52},
        },
        notes="Outcomes cluster near 50-50 split. Self-serving bias in fairness."
    ),

    "gift_exchange": MetaAnalyticEffect(
        source="Fehr et al. (1993); Charness (2004)",
        effect_d=0.35,
        ci_95=(0.20, 0.50),
        n_studies=25,
        n_participants=2500,
        domain="behavioral_economics",
        construct="reciprocity",
        paradigm="gift_exchange",
        heterogeneity_tau=0.10,
        notes="Higher wages → higher effort. Reciprocity in labor markets."
    ),

    "stag_hunt_coordination": MetaAnalyticEffect(
        source="Skyrms (2004); Battalio et al. (2001)",
        effect_d=0.0,
        ci_95=(-0.05, 0.05),
        n_studies=12,
        n_participants=960,
        domain="behavioral_economics",
        construct="coordination",
        paradigm="stag_hunt",
        heterogeneity_tau=0.10,
        moderators={
            "risk_dominance": {"aligned": 0.70, "conflicting": 0.45},
        },
        notes="Coordination rate depends on risk-dominance vs payoff-dominance."
    ),

    "common_pool_resource": MetaAnalyticEffect(
        source="Ostrom et al. (1994); Janssen et al. (2010)",
        effect_d=-0.30,
        ci_95=(-0.45, -0.15),
        n_studies=18,
        n_participants=1800,
        domain="behavioral_economics",
        construct="resource_extraction",
        paradigm="common_pool_resource",
        heterogeneity_tau=0.12,
        moderators={
            "communication": {"no_comm": -0.45, "comm": -0.15},
            "monitoring": {"no_monitor": -0.40, "monitor": -0.20},
        },
        notes="Overextraction relative to optimal. Communication reduces tragedy."
    ),

    "repeated_pd_cooperation_decay": MetaAnalyticEffect(
        source="Dal Bó & Fréchette (2018)",
        effect_d=-0.20,
        ci_95=(-0.30, -0.10),
        n_studies=44,
        n_participants=8800,
        domain="behavioral_economics",
        construct="cooperation_decay",
        paradigm="repeated_prisoners_dilemma",
        heterogeneity_tau=0.08,
        moderators={
            "continuation_prob": {"low_0.5": -0.35, "medium_0.75": -0.20, "high_0.9": -0.10},
        },
        notes="Cooperation decays in finite games. Shadow of future sustains cooperation."
    ),

    "holt_laury_risk": MetaAnalyticEffect(
        source="Holt & Laury (2002); Filippin & Crosetto (2016)",
        effect_d=0.0,
        ci_95=(-0.05, 0.05),
        n_studies=50,
        n_participants=8000,
        domain="behavioral_economics",
        construct="risk_preference",
        paradigm="holt_laury",
        heterogeneity_tau=0.10,
        i_squared=65.0,
        moderators={
            "stakes": {"hypothetical": 0.55, "real_low": 0.58, "real_high": 0.65},
            "gender": {"male": 0.54, "female": 0.60},
        },
        notes="Mean safe choices ≈ 5.8/10 (risk averse). Stakes ↑ → risk aversion ↑."
    ),

    "beauty_contest": MetaAnalyticEffect(
        source="Nagel (1995); Bosch-Domènech et al. (2002)",
        effect_d=0.0,
        ci_95=(-0.05, 0.05),
        n_studies=20,
        n_participants=4000,
        domain="behavioral_economics",
        construct="strategic_depth",
        paradigm="beauty_contest",
        heterogeneity_tau=0.12,
        notes="Mean guess ≈ 33 (level-1 reasoning). Converges to 0 with experience."
    ),

    # ── INTERGROUP & POLITICAL ────────────────────────────────────────────

    "ingroup_cooperation_meta": MetaAnalyticEffect(
        source="Balliet et al. (2014)",
        effect_d=0.32,
        ci_95=(0.24, 0.40),
        n_studies=212,
        n_participants=42000,
        domain="social_psychology",
        construct="ingroup_favoritism",
        paradigm="economic_games",
        heterogeneity_tau=0.15,
        i_squared=80.0,
        replication_status="replicated",
        moderators={
            "game_type": {"dictator": 0.35, "trust": 0.30, "pd": 0.28, "public_goods": 0.25},
            "group_type": {"minimal": 0.22, "natural": 0.38, "political": 0.45},
            "stakes": {"hypothetical": 0.25, "real": 0.35},
        },
        notes="Robust ingroup favoritism. Stronger for natural vs minimal groups."
    ),

    "political_affective_polarization": MetaAnalyticEffect(
        source="Iyengar & Westwood (2015); Dimant (2024)",
        effect_d=0.75,
        ci_95=(0.55, 0.95),
        n_studies=45,
        n_participants=15000,
        domain="political_psychology",
        construct="partisan_discrimination",
        paradigm="intergroup_economic_games",
        heterogeneity_tau=0.18,
        i_squared=82.0,
        moderators={
            "game": {"dictator": 0.80, "trust": 0.70, "hiring": 0.65},
            "identity_strength": {"weak": 0.45, "moderate": 0.70, "strong": 0.90},
            "context": {"economic": 0.80, "social": 0.65, "political": 0.85},
        },
        notes="Political discrimination exceeds racial in economic games. d ≈ 0.6-0.9."
    ),

    "ethnic_discrimination_games": MetaAnalyticEffect(
        source="Fershtman & Gneezy (2001); Lane (2016 meta)",
        effect_d=0.35,
        ci_95=(0.20, 0.50),
        n_studies=40,
        n_participants=8000,
        domain="social_psychology",
        construct="ethnic_discrimination",
        paradigm="trust_dictator_games",
        heterogeneity_tau=0.12,
        moderators={
            "game": {"dictator": 0.30, "trust": 0.40, "ultimatum": 0.25},
            "visibility": {"anonymous": 0.25, "visible": 0.40},
        },
        notes="20-30% less generous toward ethnic outgroups."
    ),

    "contact_hypothesis": MetaAnalyticEffect(
        source="Pettigrew & Tropp (2006)",
        effect_d=0.42,  # r = -0.21 → d ≈ 0.42
        ci_95=(0.35, 0.49),
        n_studies=515,
        n_participants=250000,
        domain="social_psychology",
        construct="prejudice_reduction",
        paradigm="intergroup_contact",
        heterogeneity_tau=0.15,
        i_squared=75.0,
        replication_status="replicated",
        moderators={
            "contact_quality": {"optimal": 0.55, "moderate": 0.40, "minimal": 0.25},
            "target_group": {"racial": 0.45, "elderly": 0.40, "disability": 0.50, "sexual_orientation": 0.48},
        },
        notes="r = -0.21 across 515 studies. Optimal contact conditions amplify."
    ),

    # ── BEHAVIORAL ECONOMICS / DECISION-MAKING ────────────────────────────

    "loss_aversion": MetaAnalyticEffect(
        source="Tversky & Kahneman (1992); Walasek & Stewart (2015 meta)",
        effect_d=0.50,
        ci_95=(0.35, 0.65),
        n_studies=150,
        n_participants=30000,
        domain="behavioral_economics",
        construct="loss_aversion_lambda",
        paradigm="prospect_theory",
        heterogeneity_tau=0.15,
        i_squared=78.0,
        moderators={
            "domain": {"financial": 0.55, "health": 0.45, "consumer": 0.50},
            "stakes": {"hypothetical": 0.40, "real_low": 0.50, "real_high": 0.60},
        },
        notes="Lambda (λ) ≈ 2.0-2.5. Losses loom larger than gains."
    ),

    "anchoring_effect": MetaAnalyticEffect(
        source="Tversky & Kahneman (1974); Furnham & Boo (2011 meta)",
        effect_d=0.80,
        ci_95=(0.60, 1.00),
        n_studies=95,
        n_participants=19000,
        domain="behavioral_economics",
        construct="anchoring",
        paradigm="anchor_and_adjust",
        heterogeneity_tau=0.20,
        i_squared=82.0,
        moderators={
            "anchor_type": {"random": 0.65, "plausible": 0.85, "extreme": 0.95},
            "expertise": {"novice": 0.85, "expert": 0.50},
        },
        notes="Large, robust effect. Even irrelevant anchors shift estimates."
    ),

    "default_effect": MetaAnalyticEffect(
        source="Johnson & Goldstein (2003); Jachimowicz et al. (2019 meta)",
        effect_d=0.68,
        ci_95=(0.52, 0.84),
        n_studies=58,
        n_participants=70000,
        domain="behavioral_economics",
        construct="default_bias",
        paradigm="opt_in_vs_opt_out",
        heterogeneity_tau=0.18,
        i_squared=80.0,
        moderators={
            "domain": {"organ_donation": 0.85, "retirement": 0.65, "consumer": 0.55},
            "cost": {"low_cost": 0.75, "high_cost": 0.50},
        },
        notes="60-80pp difference between opt-in and opt-out. Massive effect."
    ),

    "sunk_cost_escalation": MetaAnalyticEffect(
        source="Sleesman et al. (2012)",
        effect_d=0.37,
        ci_95=(0.21, 0.53),
        n_studies=79,
        n_participants=12000,
        domain="behavioral_economics",
        construct="escalation_of_commitment",
        paradigm="sunk_cost",
        heterogeneity_tau=0.12,
        i_squared=70.0,
        moderators={
            "investment_type": {"financial": 0.42, "time": 0.35, "effort": 0.30},
            "responsibility": {"high": 0.45, "low": 0.28},
        },
        notes="People continue investing after sunk costs. Moderated by responsibility."
    ),

    "hyperbolic_discounting": MetaAnalyticEffect(
        source="Amlung et al. (2017 meta)",
        effect_d=0.40,
        ci_95=(0.30, 0.50),
        n_studies=63,
        n_participants=18000,
        domain="behavioral_economics",
        construct="temporal_discounting",
        paradigm="delay_discounting",
        heterogeneity_tau=0.12,
        i_squared=68.0,
        moderators={
            "substance_use": {"non_user": 0.35, "substance_user": 0.55},
            "reward_type": {"money": 0.40, "health": 0.35, "food": 0.50},
        },
        notes="Present bias: immediate rewards overvalued. Clinical link to addiction."
    ),

    # ── SOCIAL PSYCHOLOGY / COMPLIANCE ─────────────────────────────────────

    "stereotype_threat": MetaAnalyticEffect(
        source="Nguyen & Ryan (2008)",
        effect_d=0.26,
        ci_95=(0.15, 0.37),
        n_studies=92,
        n_participants=12500,
        domain="social_psychology",
        construct="stereotype_threat_performance",
        paradigm="diagnostic_test",
        heterogeneity_tau=0.10,
        i_squared=65.0,
        replication_status="contested",
        moderators={
            "target_group": {"race": 0.30, "gender_math": 0.22, "age": 0.20},
            "test_diagnosticity": {"high": 0.32, "low": 0.18},
        },
        notes="d = 0.26 overall. Contested after replication failures. Moderated by diagnosticity."
    ),

    "bystander_effect": MetaAnalyticEffect(
        source="Fischer et al. (2011)",
        effect_d=0.50,
        ci_95=(0.38, 0.62),
        n_studies=105,
        n_participants=7700,
        domain="social_psychology",
        construct="helping_behavior",
        paradigm="bystander",
        heterogeneity_tau=0.12,
        i_squared=72.0,
        replication_status="replicated",
        moderators={
            "danger": {"low": 0.55, "high": 0.35},
            "group_size": {"2": 0.40, "4": 0.55, "6_plus": 0.60},
        },
        notes="85% help alone vs 31% with 4 others. Danger reverses effect."
    ),

    "obedience_authority": MetaAnalyticEffect(
        source="Milgram (1963); Haslam et al. (2014 meta review)",
        effect_d=0.60,
        ci_95=(0.45, 0.75),
        n_studies=20,
        n_participants=2000,
        domain="social_psychology",
        construct="obedience",
        paradigm="authority_compliance",
        heterogeneity_tau=0.15,
        moderators={
            "proximity": {"remote": 0.70, "adjacent": 0.55, "touching": 0.35},
            "institutional": {"high": 0.65, "low": 0.45},
        },
        notes="43.6% full obedience in original. Robust across replications."
    ),

    "mere_exposure_meta": MetaAnalyticEffect(
        source="Bornstein (1989)",
        effect_d=0.52,  # r = 0.26 → d ≈ 0.52
        ci_95=(0.42, 0.62),
        n_studies=134,
        n_participants=20000,
        domain="social_psychology",
        construct="familiarity_preference",
        paradigm="mere_exposure",
        heterogeneity_tau=0.10,
        i_squared=65.0,
        replication_status="replicated",
        notes="r = 0.26. Robust. Strongest for novel stimuli and subliminal exposure."
    ),

    "social_proof_meta": MetaAnalyticEffect(
        source="Bond & Smith (1996)",
        effect_d=0.42,
        ci_95=(0.30, 0.54),
        n_studies=133,
        n_participants=25000,
        domain="social_psychology",
        construct="conformity",
        paradigm="asch_conformity",
        heterogeneity_tau=0.15,
        i_squared=78.0,
        moderators={
            "culture": {"individualist": 0.35, "collectivist": 0.55},
            "group_size": {"1": 0.15, "3": 0.35, "5_plus": 0.42},
        },
        notes="Conformity rates declining over time. Culture moderates strongly."
    ),

    # ── HEALTH / CLINICAL ─────────────────────────────────────────────────

    "fear_appeals_meta": MetaAnalyticEffect(
        source="Witte & Allen (2000); Tannenbaum et al. (2015)",
        effect_d=0.45,
        ci_95=(0.30, 0.60),
        n_studies=248,
        n_participants=50000,
        domain="health_psychology",
        construct="fear_appeal_effectiveness",
        paradigm="fear_appeal",
        heterogeneity_tau=0.15,
        i_squared=75.0,
        moderators={
            "efficacy": {"high_efficacy": 0.60, "low_efficacy": 0.20},
            "fear_level": {"low": 0.25, "moderate": 0.45, "high": 0.55},
            "outcome": {"attitude": 0.50, "intention": 0.40, "behavior": 0.25},
        },
        notes="d = 0.3-0.8. High fear + high efficacy = optimal. Low efficacy backfires."
    ),

    "psychotherapy_depression": MetaAnalyticEffect(
        source="Cuijpers et al. (2019)",
        effect_d=0.72,
        ci_95=(0.60, 0.84),
        n_studies=385,
        n_participants=45000,
        domain="clinical_psychology",
        construct="depression_reduction",
        paradigm="psychotherapy_rct",
        heterogeneity_tau=0.20,
        i_squared=82.0,
        moderators={
            "therapy_type": {"cbt": 0.75, "behavioral": 0.70, "psychodynamic": 0.68, "interpersonal": 0.72},
            "control": {"waitlist": 0.82, "treatment_as_usual": 0.55, "pill_placebo": 0.35},
        },
        notes="Large effect vs waitlist. Smaller vs active control."
    ),

    "prebunking_misinformation": MetaAnalyticEffect(
        source="Roozenbeek et al. (2022)",
        effect_d=0.40,
        ci_95=(0.25, 0.55),
        n_studies=30,
        n_participants=20000,
        domain="health_psychology",
        construct="misinformation_resistance",
        paradigm="inoculation_prebunking",
        heterogeneity_tau=0.12,
        notes="d = 0.3-0.5 for prebunking interventions against misinformation."
    ),

    "optimistic_bias": MetaAnalyticEffect(
        source="Shepperd et al. (2013 meta)",
        effect_d=0.45,
        ci_95=(0.35, 0.55),
        n_studies=124,
        n_participants=50000,
        domain="health_psychology",
        construct="unrealistic_optimism",
        paradigm="comparative_risk",
        heterogeneity_tau=0.12,
        i_squared=70.0,
        notes="People believe they are less likely to experience negative events."
    ),

    # ── ORGANIZATIONAL ────────────────────────────────────────────────────

    "procedural_justice_meta": MetaAnalyticEffect(
        source="Colquitt et al. (2001)",
        effect_d=0.80,  # ρ = .40-.50 → d ≈ 0.80
        ci_95=(0.65, 0.95),
        n_studies=183,
        n_participants=65000,
        domain="organizational_behavior",
        construct="justice_satisfaction",
        paradigm="organizational_justice",
        heterogeneity_tau=0.15,
        i_squared=76.0,
        notes="ρ ≈ .40-.50 with satisfaction, commitment, OCB."
    ),

    "transformational_leadership_meta": MetaAnalyticEffect(
        source="Judge & Piccolo (2004)",
        effect_d=0.88,  # ρ = .44 → d ≈ 0.88
        ci_95=(0.72, 1.04),
        n_studies=87,
        n_participants=25000,
        domain="organizational_behavior",
        construct="leadership_effectiveness",
        paradigm="transformational_leadership",
        heterogeneity_tau=0.18,
        i_squared=80.0,
        notes="ρ ≈ .44 with satisfaction. Large corrected effect."
    ),

    "psychological_safety_meta": MetaAnalyticEffect(
        source="Frazier et al. (2017)",
        effect_d=0.55,
        ci_95=(0.42, 0.68),
        n_studies=136,
        n_participants=22000,
        domain="organizational_behavior",
        construct="psychological_safety",
        paradigm="team_climate",
        heterogeneity_tau=0.12,
        notes="Psychological safety predicts learning, performance, engagement."
    ),

    # ── COMMUNICATION / PERSUASION ────────────────────────────────────────

    "narrative_transportation_meta": MetaAnalyticEffect(
        source="van Laer et al. (2014); Green & Brock (2000)",
        effect_d=0.70,  # r = 0.35 → d ≈ 0.70
        ci_95=(0.55, 0.85),
        n_studies=76,
        n_participants=15000,
        domain="communication",
        construct="narrative_persuasion",
        paradigm="narrative_transportation",
        heterogeneity_tau=0.15,
        i_squared=75.0,
        notes="r = 0.35 transportation → persuasion. Robust."
    ),

    "inoculation_meta": MetaAnalyticEffect(
        source="Banas & Rains (2010)",
        effect_d=0.29,
        ci_95=(0.18, 0.40),
        n_studies=52,
        n_participants=10000,
        domain="communication",
        construct="attitude_resistance",
        paradigm="inoculation_theory",
        heterogeneity_tau=0.10,
        notes="d = 0.29 for inoculation vs no inoculation."
    ),

    # ── EMBODIMENT / PRIMING ──────────────────────────────────────────────

    "power_posing": MetaAnalyticEffect(
        source="Cuddy et al. (2018 reanalysis); Credé & Phillips (2017 critique)",
        effect_d=0.15,
        ci_95=(0.02, 0.28),
        n_studies=55,
        n_participants=6000,
        domain="embodied_cognition",
        construct="felt_power",
        paradigm="power_pose",
        heterogeneity_tau=0.10,
        i_squared=60.0,
        replication_status="contested",
        notes="Small effect on felt power. No hormonal effects. Contested."
    ),

    "warmth_priming": MetaAnalyticEffect(
        source="Williams & Bargh (2008); Lynott et al. (2014 replication)",
        effect_d=0.12,
        ci_95=(-0.05, 0.29),
        n_studies=8,
        n_participants=1200,
        domain="embodied_cognition",
        construct="interpersonal_warmth",
        paradigm="temperature_priming",
        heterogeneity_tau=0.10,
        replication_status="contested",
        notes="d ≈ 0.12. CI includes zero. Failed direct replications."
    ),

    "facial_feedback": MetaAnalyticEffect(
        source="Coles et al. (2019 many-labs)",
        effect_d=0.06,
        ci_95=(-0.02, 0.14),
        n_studies=17,
        n_participants=1900,
        domain="embodied_cognition",
        construct="emotional_experience",
        paradigm="facial_feedback",
        heterogeneity_tau=0.05,
        replication_status="contested",
        notes="r = 0.03 in many-labs. Very small, possibly null."
    ),

    # ── LEARNING / MEMORY ─────────────────────────────────────────────────

    "testing_effect_meta": MetaAnalyticEffect(
        source="Rowland (2014)",
        effect_d=0.50,
        ci_95=(0.42, 0.58),
        n_studies=159,
        n_participants=30000,
        domain="educational_psychology",
        construct="retrieval_practice",
        paradigm="testing_effect",
        heterogeneity_tau=0.12,
        i_squared=70.0,
        replication_status="replicated",
        moderators={
            "material": {"verbal": 0.55, "visual": 0.40, "procedural": 0.45},
            "delay": {"immediate": 0.35, "1_day": 0.50, "1_week": 0.55},
        },
        notes="d = 0.50. Robust. Strongest at longer delays."
    ),

    "implementation_intentions_meta": MetaAnalyticEffect(
        source="Gollwitzer & Sheeran (2006)",
        effect_d=0.65,
        ci_95=(0.52, 0.78),
        n_studies=94,
        n_participants=15000,
        domain="motivation",
        construct="goal_attainment",
        paradigm="implementation_intentions",
        heterogeneity_tau=0.15,
        i_squared=75.0,
        replication_status="replicated",
        notes="d = 0.65. Very robust for if-then planning."
    ),

    "growth_mindset_meta": MetaAnalyticEffect(
        source="Sisk et al. (2018)",
        effect_d=0.10,
        ci_95=(0.04, 0.16),
        n_studies=273,
        n_participants=365000,
        domain="educational_psychology",
        construct="academic_achievement",
        paradigm="growth_mindset",
        heterogeneity_tau=0.08,
        i_squared=55.0,
        replication_status="contested",
        moderators={
            "risk_status": {"low_risk": 0.08, "high_risk": 0.18},
        },
        notes="d = 0.10 overall. Small. Larger for at-risk students."
    ),

    # ── MORAL / DISHONESTY ────────────────────────────────────────────────

    "moral_reminders_honesty": MetaAnalyticEffect(
        source="Mazar et al. (2008); Verschuere et al. (2018 many-labs)",
        effect_d=0.18,
        ci_95=(0.05, 0.31),
        n_studies=25,
        n_participants=5000,
        domain="moral_psychology",
        construct="honesty",
        paradigm="moral_reminders",
        heterogeneity_tau=0.10,
        replication_status="contested",
        notes="Original d ≈ 0.30. Many-labs reduced estimate. Still positive."
    ),

    "moral_licensing_meta": MetaAnalyticEffect(
        source="Blanken et al. (2015 meta)",
        effect_d=0.31,
        ci_95=(0.20, 0.42),
        n_studies=91,
        n_participants=7700,
        domain="moral_psychology",
        construct="moral_licensing",
        paradigm="licensing_paradigm",
        heterogeneity_tau=0.12,
        i_squared=68.0,
        notes="Past moral behavior licenses subsequent immoral behavior."
    ),

    # ── CONSUMER / MARKETING ──────────────────────────────────────────────

    "scarcity_effect_meta": MetaAnalyticEffect(
        source="Barton et al. (2022)",
        effect_d=0.56,  # r = 0.28 → d ≈ 0.56
        ci_95=(0.42, 0.70),
        n_studies=131,
        n_participants=30000,
        domain="consumer_psychology",
        construct="purchase_intention",
        paradigm="scarcity",
        heterogeneity_tau=0.15,
        moderators={
            "scarcity_type": {"supply": 0.60, "time": 0.55, "demand": 0.50},
        },
        notes="r = 0.28. Robust. Supply scarcity > demand scarcity."
    ),

    "choice_overload_meta": MetaAnalyticEffect(
        source="Scheibehenne et al. (2010)",
        effect_d=0.02,
        ci_95=(-0.10, 0.14),
        n_studies=50,
        n_participants=10000,
        domain="consumer_psychology",
        construct="choice_satisfaction",
        paradigm="choice_overload",
        heterogeneity_tau=0.12,
        i_squared=70.0,
        replication_status="contested",
        notes="Original jam study d = 0.77 but meta near 0. Highly moderated."
    ),

    "endowment_effect": MetaAnalyticEffect(
        source="Kahneman et al. (1990); Morewedge & Giblin (2015)",
        effect_d=0.55,
        ci_95=(0.40, 0.70),
        n_studies=80,
        n_participants=12000,
        domain="behavioral_economics",
        construct="wta_wtp_ratio",
        paradigm="endowment_effect",
        heterogeneity_tau=0.15,
        notes="WTA/WTP ratio ≈ 2:1. Robust but moderated by experience."
    ),

    # ── ENVIRONMENTAL / CONTEXTUAL ────────────────────────────────────────

    "temperature_aggression": MetaAnalyticEffect(
        source="Anderson et al. (2000 meta)",
        effect_d=0.30,
        ci_95=(0.18, 0.42),
        n_studies=50,
        n_participants=15000,
        domain="environmental",
        construct="aggression",
        paradigm="heat_hypothesis",
        heterogeneity_tau=0.10,
        replication_status="replicated",
        notes="Heat increases aggression. Linear up to extreme temperatures."
    ),

    "nature_stress_reduction": MetaAnalyticEffect(
        source="Bratman et al. (2019 meta); Bowler et al. (2010)",
        effect_d=0.35,
        ci_95=(0.22, 0.48),
        n_studies=103,
        n_participants=20000,
        domain="environmental",
        construct="stress_reduction",
        paradigm="nature_exposure",
        heterogeneity_tau=0.12,
        notes="Nature exposure reduces stress, improves mood. Robust."
    ),

    # ── IMPLICIT MEASURES ────────────────────────────────────────────────

    "iat_race_meta": MetaAnalyticEffect(
        source="Greenwald et al. (2009); Oswald et al. (2013 meta)",
        effect_d=0.40,
        ci_95=(0.30, 0.50),
        n_studies=122,
        n_participants=14900,
        domain="social_psychology",
        construct="implicit_racial_bias",
        paradigm="iat",
        heterogeneity_tau=0.15,
        i_squared=75.0,
        moderators={
            "target_race": {"black_white": 0.42, "arab_european": 0.38, "asian_white": 0.30},
            "criterion": {"attitude": 0.40, "behavior": 0.15},
        },
        notes="IAT-behavior r ≈ 0.15 (Oswald 2013). Attitudes d ≈ 0.40."
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # EXPANDED META-ANALYTIC DATABASE (v1.0.9.3)
    # ~150 new entries across 10 research domains
    # ═══════════════════════════════════════════════════════════════════════

    # ── SOCIAL PSYCHOLOGY (expanded) ────────────────────────────────────────

    "conformity_asch_meta": MetaAnalyticEffect(
        source="Bond & Smith (1996)",
        effect_d=0.42,
        ci_95=(0.30, 0.54),
        n_studies=133,
        n_participants=25143,
        domain="social_psychology",
        construct="conformity",
        paradigm="asch_line_judgment",
        heterogeneity_tau=0.15,
        i_squared=78.0,
        replication_status="replicated",
        moderators={
            "culture": {"individualist": 0.35, "collectivist": 0.55},
            "group_size": {"1_confederate": 0.15, "3_confederates": 0.35, "5_plus": 0.42},
            "era": {"pre_1970": 0.50, "post_1990": 0.32},
        },
        notes="Conformity declining over decades. Collectivist cultures show ~57% higher rates."
    ),

    "cognitive_dissonance_meta": MetaAnalyticEffect(
        source="Kenworthy et al. (2011); Hinojosa et al. (2017)",
        effect_d=0.54,
        ci_95=(0.42, 0.66),
        n_studies=78,
        n_participants=8500,
        domain="social_psychology",
        construct="attitude_change",
        paradigm="induced_compliance",
        heterogeneity_tau=0.14,
        i_squared=72.0,
        replication_status="replicated",
        moderators={
            "justification": {"insufficient": 0.65, "sufficient": 0.20},
            "choice": {"high_choice": 0.60, "low_choice": 0.30},
            "culture": {"western": 0.54, "east_asian": 0.38},
        },
        notes="Robust effect for induced compliance paradigm. Weaker in collectivist cultures."
    ),

    "self_affirmation_meta": MetaAnalyticEffect(
        source="Epton et al. (2015)",
        effect_d=0.32,
        ci_95=(0.24, 0.40),
        n_studies=144,
        n_participants=28000,
        domain="social_psychology",
        construct="health_behavior_intention",
        paradigm="self_affirmation",
        heterogeneity_tau=0.12,
        i_squared=68.0,
        replication_status="replicated",
        moderators={
            "outcome": {"intention": 0.35, "behavior": 0.28},
            "threat_severity": {"low": 0.22, "high": 0.42},
        },
        notes="d = 0.32 for message acceptance. Larger when threat is high."
    ),

    "fundamental_attribution_error": MetaAnalyticEffect(
        source="Malle (2006); Jones & Harris (1967)",
        effect_d=0.55,
        ci_95=(0.40, 0.70),
        n_studies=85,
        n_participants=12000,
        domain="social_psychology",
        construct="dispositional_attribution",
        paradigm="attitude_attribution",
        heterogeneity_tau=0.15,
        i_squared=74.0,
        replication_status="replicated",
        moderators={
            "culture": {"western": 0.60, "east_asian": 0.35},
            "cognitive_load": {"low": 0.50, "high": 0.70},
        },
        notes="Tendency to attribute behavior to dispositions over situations. Weaker in East Asian samples."
    ),

    "actor_observer_asymmetry": MetaAnalyticEffect(
        source="Malle (2006)",
        effect_d=0.09,
        ci_95=(-0.02, 0.20),
        n_studies=173,
        n_participants=20000,
        domain="social_psychology",
        construct="actor_observer_difference",
        paradigm="attribution",
        heterogeneity_tau=0.10,
        i_squared=60.0,
        replication_status="contested",
        notes="Original claim overstated. d = 0.09, near zero. Malle (2006) challenged the classic finding."
    ),

    "self_serving_bias_meta": MetaAnalyticEffect(
        source="Mezulis et al. (2004)",
        effect_d=0.65,
        ci_95=(0.55, 0.75),
        n_studies=266,
        n_participants=50000,
        domain="social_psychology",
        construct="self_serving_attribution",
        paradigm="success_failure_attribution",
        heterogeneity_tau=0.15,
        i_squared=76.0,
        replication_status="replicated",
        moderators={
            "age": {"children": 0.55, "adults": 0.65, "elderly": 0.45},
            "culture": {"western": 0.70, "east_asian": 0.40},
            "depression": {"non_depressed": 0.65, "depressed": 0.15},
        },
        notes="Robust self-serving attributional bias. Absent in depression."
    ),

    "halo_effect_meta": MetaAnalyticEffect(
        source="Landy & Sigall (1974); Eagly et al. (1991 meta)",
        effect_d=0.56,
        ci_95=(0.42, 0.70),
        n_studies=76,
        n_participants=14000,
        domain="social_psychology",
        construct="attractiveness_halo",
        paradigm="person_perception",
        heterogeneity_tau=0.14,
        i_squared=72.0,
        replication_status="replicated",
        moderators={
            "judgment_type": {"competence": 0.50, "sociability": 0.65, "morality": 0.35},
        },
        notes="Physical attractiveness creates positive halo. Strongest for sociability judgments."
    ),

    "social_loafing_meta": MetaAnalyticEffect(
        source="Karau & Williams (1993)",
        effect_d=0.44,
        ci_95=(0.35, 0.53),
        n_studies=78,
        n_participants=8000,
        domain="social_psychology",
        construct="effort_reduction_in_groups",
        paradigm="social_loafing",
        heterogeneity_tau=0.12,
        i_squared=68.0,
        replication_status="replicated",
        moderators={
            "task_type": {"physical": 0.50, "cognitive": 0.38, "perceptual": 0.42},
            "group_size": {"2": 0.30, "4": 0.42, "8_plus": 0.52},
            "culture": {"individualist": 0.50, "collectivist": 0.28},
        },
        notes="Robust. Larger groups = more loafing. Collectivist cultures loaf less."
    ),

    "deindividuation_meta": MetaAnalyticEffect(
        source="Postmes & Spears (1998)",
        effect_d=0.25,
        ci_95=(0.12, 0.38),
        n_studies=60,
        n_participants=5000,
        domain="social_psychology",
        construct="antinormative_behavior",
        paradigm="deindividuation",
        heterogeneity_tau=0.12,
        i_squared=65.0,
        replication_status="replicated",
        moderators={
            "anonymity": {"anonymous": 0.35, "identifiable": 0.12},
            "group_norms": {"anti_social_norm": 0.40, "pro_social_norm": -0.10},
        },
        notes="Effect depends on group norms. Anonymity amplifies norm-consistent behavior, not just aggression."
    ),

    "social_facilitation_meta": MetaAnalyticEffect(
        source="Bond & Titus (1983)",
        effect_d=0.35,
        ci_95=(0.25, 0.45),
        n_studies=241,
        n_participants=24000,
        domain="social_psychology",
        construct="social_facilitation_inhibition",
        paradigm="audience_effects",
        heterogeneity_tau=0.14,
        i_squared=72.0,
        replication_status="replicated",
        moderators={
            "task_complexity": {"simple": 0.45, "complex": -0.30},
        },
        notes="Audience improves simple tasks (d = 0.45) but impairs complex tasks (d = -0.30)."
    ),

    "spotlight_effect_meta": MetaAnalyticEffect(
        source="Gilovich et al. (2000); Gilovich & Savitsky (1999)",
        effect_d=0.48,
        ci_95=(0.32, 0.64),
        n_studies=18,
        n_participants=2400,
        domain="social_psychology",
        construct="egocentric_bias",
        paradigm="spotlight_effect",
        heterogeneity_tau=0.10,
        i_squared=55.0,
        replication_status="replicated",
        notes="People overestimate how much others notice their appearance/behavior by ~2x."
    ),

    "false_consensus_meta": MetaAnalyticEffect(
        source="Mullen et al. (1985); Ross et al. (1977)",
        effect_d=0.40,
        ci_95=(0.30, 0.50),
        n_studies=115,
        n_participants=15000,
        domain="social_psychology",
        construct="false_consensus",
        paradigm="consensus_estimation",
        heterogeneity_tau=0.12,
        i_squared=68.0,
        replication_status="replicated",
        notes="People overestimate the prevalence of their own opinions and behaviors."
    ),

    "just_world_belief_meta": MetaAnalyticEffect(
        source="Hafer & Begue (2005); Lerner (1980)",
        effect_d=0.35,
        ci_95=(0.22, 0.48),
        n_studies=50,
        n_participants=8000,
        domain="social_psychology",
        construct="victim_derogation",
        paradigm="just_world",
        heterogeneity_tau=0.12,
        i_squared=65.0,
        replication_status="replicated",
        moderators={
            "victim_innocence": {"clearly_innocent": 0.42, "ambiguous": 0.28},
        },
        notes="Innocent victims derogated more when threat to belief in a just world is high."
    ),

    "foot_in_door_meta": MetaAnalyticEffect(
        source="Burger (1999); Cialdini (1993)",
        effect_d=0.35,
        ci_95=(0.24, 0.46),
        n_studies=69,
        n_participants=8200,
        domain="social_psychology",
        construct="compliance",
        paradigm="foot_in_door",
        heterogeneity_tau=0.12,
        i_squared=66.0,
        replication_status="replicated",
        moderators={
            "delay": {"immediate": 0.40, "delayed": 0.28},
            "request_type": {"prosocial": 0.42, "consumer": 0.28},
        },
        notes="Small initial compliance increases later large-request compliance."
    ),

    "door_in_face_meta": MetaAnalyticEffect(
        source="Cialdini et al. (1975); O'Keefe & Hale (2001 meta)",
        effect_d=0.36,
        ci_95=(0.24, 0.48),
        n_studies=43,
        n_participants=5500,
        domain="social_psychology",
        construct="compliance",
        paradigm="door_in_face",
        heterogeneity_tau=0.10,
        i_squared=62.0,
        replication_status="replicated",
        moderators={
            "requester": {"same_person": 0.40, "different_person": 0.15},
            "prosocial": {"prosocial": 0.42, "self_serving": 0.22},
        },
        notes="Rejection of large request increases compliance with moderate request. Same-requester key."
    ),

    "lowball_technique_meta": MetaAnalyticEffect(
        source="Burger & Petty (1981); Cialdini et al. (1978)",
        effect_d=0.42,
        ci_95=(0.28, 0.56),
        n_studies=22,
        n_participants=3500,
        domain="social_psychology",
        construct="compliance",
        paradigm="lowball",
        heterogeneity_tau=0.10,
        i_squared=58.0,
        replication_status="replicated",
        notes="Commitment to initial deal maintained even after cost increase. d = 0.42."
    ),

    "scarcity_psychological": MetaAnalyticEffect(
        source="Cialdini (2001); Lynn (1991 meta)",
        effect_d=0.52,
        ci_95=(0.38, 0.66),
        n_studies=50,
        n_participants=9000,
        domain="social_psychology",
        construct="desirability",
        paradigm="scarcity",
        heterogeneity_tau=0.14,
        i_squared=70.0,
        replication_status="replicated",
        moderators={
            "scarcity_type": {"supply_limited": 0.58, "time_limited": 0.48, "demand_driven": 0.45},
        },
        notes="Scarce items rated as more desirable. Supply scarcity > demand scarcity."
    ),

    "ego_depletion_meta": MetaAnalyticEffect(
        source="Hagger et al. (2010); Carter & McCullough (2014 p-curve)",
        effect_d=0.17,
        ci_95=(-0.02, 0.36),
        n_studies=198,
        n_participants=22000,
        domain="social_psychology",
        construct="self_control_depletion",
        paradigm="sequential_task",
        heterogeneity_tau=0.18,
        i_squared=82.0,
        replication_status="contested",
        notes="Original d = 0.62 (Hagger 2010). After p-curve correction d = 0.17. Many-labs RRR d = 0.04. Highly contested."
    ),

    "social_desirability_responding": MetaAnalyticEffect(
        source="Nederhof (1985); Paulhus (2002)",
        effect_d=0.40,
        ci_95=(0.28, 0.52),
        n_studies=80,
        n_participants=20000,
        domain="social_psychology",
        construct="social_desirability_bias",
        paradigm="self_report_vs_behavioral",
        heterogeneity_tau=0.15,
        i_squared=75.0,
        replication_status="replicated",
        moderators={
            "topic_sensitivity": {"prejudice": 0.65, "substance_use": 0.55, "prosocial": 0.40, "factual": 0.15},
            "anonymity": {"anonymous": 0.28, "identified": 0.52},
        },
        notes="Self-report inflated for socially desirable traits. Strongest for sensitive topics."
    ),

    "primacy_recency_impression": MetaAnalyticEffect(
        source="Asch (1946); Jones et al. (1968); Haugtvedt & Wegener (1994)",
        effect_d=0.38,
        ci_95=(0.25, 0.51),
        n_studies=40,
        n_participants=5000,
        domain="social_psychology",
        construct="impression_formation",
        paradigm="primacy_recency",
        heterogeneity_tau=0.12,
        i_squared=65.0,
        replication_status="replicated",
        moderators={
            "modality": {"written": 0.42, "oral": 0.30},
            "attention": {"high_attention": 0.45, "low_attention": 0.25},
        },
        notes="Primacy dominates in written/high-attention. Recency in oral/low-attention."
    ),

    # ── HEALTH & CLINICAL ────────────────────────────────────────────────

    "cbt_depression_meta": MetaAnalyticEffect(
        source="Cuijpers et al. (2013); Hofmann et al. (2012)",
        effect_d=0.71,
        ci_95=(0.62, 0.80),
        n_studies=269,
        n_participants=21000,
        domain="clinical_psychology",
        construct="depression_symptom_reduction",
        paradigm="cbt",
        heterogeneity_tau=0.18,
        i_squared=80.0,
        replication_status="replicated",
        moderators={
            "control": {"waitlist": 0.82, "treatment_as_usual": 0.53, "active_control": 0.38},
            "format": {"individual": 0.75, "group": 0.62, "guided_self_help": 0.55},
            "severity": {"mild": 0.55, "moderate": 0.72, "severe": 0.85},
        },
        notes="CBT is well-established for depression. Larger effects vs passive controls."
    ),

    "cbt_anxiety_meta": MetaAnalyticEffect(
        source="Hofmann & Smits (2008); Carpenter et al. (2018)",
        effect_d=0.73,
        ci_95=(0.62, 0.84),
        n_studies=160,
        n_participants=12000,
        domain="clinical_psychology",
        construct="anxiety_symptom_reduction",
        paradigm="cbt",
        heterogeneity_tau=0.16,
        i_squared=78.0,
        replication_status="replicated",
        moderators={
            "disorder": {"gad": 0.70, "social_anxiety": 0.75, "panic": 0.80, "ptsd": 0.65},
            "control": {"waitlist": 0.85, "pill_placebo": 0.45},
        },
        notes="CBT robust across anxiety disorders. Largest for panic disorder."
    ),

    "mindfulness_intervention_meta": MetaAnalyticEffect(
        source="Khoury et al. (2013); Goldberg et al. (2018)",
        effect_d=0.55,
        ci_95=(0.47, 0.63),
        n_studies=209,
        n_participants=12000,
        domain="clinical_psychology",
        construct="psychological_distress",
        paradigm="mindfulness_based_intervention",
        heterogeneity_tau=0.15,
        i_squared=74.0,
        replication_status="replicated",
        moderators={
            "program": {"mbsr": 0.55, "mbct": 0.58, "brief_mindfulness": 0.35},
            "outcome": {"anxiety": 0.60, "depression": 0.52, "stress": 0.50, "wellbeing": 0.40},
            "control": {"waitlist": 0.65, "active_control": 0.30},
        },
        notes="Moderate effect. MBCT particularly effective for recurrent depression prevention."
    ),

    "exercise_depression_meta": MetaAnalyticEffect(
        source="Schuch et al. (2016); Cooney et al. (2013)",
        effect_d=0.56,
        ci_95=(0.41, 0.71),
        n_studies=25,
        n_participants=1487,
        domain="health_psychology",
        construct="depression_reduction",
        paradigm="exercise_intervention",
        heterogeneity_tau=0.14,
        i_squared=72.0,
        replication_status="replicated",
        moderators={
            "exercise_type": {"aerobic": 0.60, "resistance": 0.50, "mixed": 0.55},
            "supervision": {"supervised": 0.62, "unsupervised": 0.38},
            "severity": {"mild_moderate": 0.48, "severe": 0.72},
        },
        notes="Exercise as effective as psychotherapy for mild-moderate depression."
    ),

    "placebo_effect_meta": MetaAnalyticEffect(
        source="Hrobjartsson & Gotzsche (2010); Wager & Atlas (2015)",
        effect_d=0.30,
        ci_95=(0.20, 0.40),
        n_studies=202,
        n_participants=40000,
        domain="health_psychology",
        construct="symptom_improvement",
        paradigm="placebo_controlled_trial",
        heterogeneity_tau=0.15,
        i_squared=76.0,
        replication_status="replicated",
        moderators={
            "outcome_type": {"pain": 0.55, "nausea": 0.40, "depression": 0.35, "objective_measures": 0.08},
            "mechanism": {"open_label": 0.28, "deceptive": 0.35},
        },
        notes="Placebos strongest for subjective outcomes (pain, nausea). Negligible for objective measures."
    ),

    "medication_adherence_meta": MetaAnalyticEffect(
        source="Conn et al. (2016); Kripalani et al. (2007)",
        effect_d=0.34,
        ci_95=(0.25, 0.43),
        n_studies=146,
        n_participants=30000,
        domain="health_psychology",
        construct="medication_adherence",
        paradigm="adherence_intervention",
        heterogeneity_tau=0.12,
        i_squared=68.0,
        replication_status="replicated",
        moderators={
            "intervention_type": {"behavioral": 0.40, "educational": 0.25, "combined": 0.45},
            "disease": {"chronic": 0.30, "acute": 0.42},
        },
        notes="Behavioral interventions outperform educational-only. Combined approaches best."
    ),

    "smoking_cessation_meta": MetaAnalyticEffect(
        source="Lancaster & Stead (2017); Hartmann-Boyce et al. (2018)",
        effect_d=0.40,
        ci_95=(0.32, 0.48),
        n_studies=312,
        n_participants=250000,
        domain="health_psychology",
        construct="smoking_abstinence",
        paradigm="cessation_intervention",
        heterogeneity_tau=0.12,
        i_squared=65.0,
        replication_status="replicated",
        moderators={
            "intervention": {"nrt": 0.38, "varenicline": 0.52, "counseling": 0.30, "combined": 0.55},
            "followup": {"6_month": 0.45, "12_month": 0.35},
        },
        notes="Pharmacotherapy + counseling most effective. Long-term quit rates 15-25%."
    ),

    "vaccination_intention_meta": MetaAnalyticEffect(
        source="Brewer et al. (2017); Sheeran et al. (2017)",
        effect_d=0.48,
        ci_95=(0.38, 0.58),
        n_studies=85,
        n_participants=45000,
        domain="health_psychology",
        construct="vaccination_uptake",
        paradigm="vaccination_promotion",
        heterogeneity_tau=0.14,
        i_squared=72.0,
        replication_status="replicated",
        moderators={
            "strategy": {"implementation_intentions": 0.55, "risk_communication": 0.40, "social_norms": 0.45},
            "vaccine_type": {"flu": 0.42, "hpv": 0.52, "covid": 0.38},
        },
        notes="Behavioral nudges (defaults, reminders, implementation intentions) most effective."
    ),

    "health_literacy_meta": MetaAnalyticEffect(
        source="Berkman et al. (2011); Dewalt et al. (2004)",
        effect_d=0.42,
        ci_95=(0.30, 0.54),
        n_studies=96,
        n_participants=18000,
        domain="health_psychology",
        construct="health_outcomes",
        paradigm="health_literacy_intervention",
        heterogeneity_tau=0.14,
        i_squared=72.0,
        replication_status="replicated",
        notes="Low health literacy associated with worse outcomes. Interventions show d = 0.42."
    ),

    "exposure_therapy_meta": MetaAnalyticEffect(
        source="Powers & Emmelkamp (2008); Olatunji et al. (2009)",
        effect_d=1.13,
        ci_95=(0.95, 1.31),
        n_studies=83,
        n_participants=3600,
        domain="clinical_psychology",
        construct="anxiety_fear_reduction",
        paradigm="exposure_therapy",
        heterogeneity_tau=0.20,
        i_squared=78.0,
        replication_status="replicated",
        moderators={
            "exposure_type": {"in_vivo": 1.20, "imaginal": 0.95, "virtual_reality": 1.05},
            "disorder": {"specific_phobia": 1.30, "social_anxiety": 0.90, "ptsd": 0.85, "ocd": 1.10},
        },
        notes="Large effect. In vivo exposure most effective. Gold standard for specific phobias."
    ),

    "motivational_interviewing_meta": MetaAnalyticEffect(
        source="Lundahl et al. (2010); Hettema et al. (2005)",
        effect_d=0.28,
        ci_95=(0.20, 0.36),
        n_studies=119,
        n_participants=18000,
        domain="clinical_psychology",
        construct="behavior_change",
        paradigm="motivational_interviewing",
        heterogeneity_tau=0.12,
        i_squared=68.0,
        replication_status="replicated",
        moderators={
            "target": {"substance_use": 0.32, "diet_exercise": 0.25, "treatment_engagement": 0.35},
            "comparison": {"no_treatment": 0.38, "active_treatment": 0.15},
        },
        notes="Small-moderate effect. Best as add-on. Equivalent to other treatments for substance use."
    ),

    "behavioral_activation_meta": MetaAnalyticEffect(
        source="Ekers et al. (2014); Cuijpers et al. (2007)",
        effect_d=0.74,
        ci_95=(0.56, 0.92),
        n_studies=26,
        n_participants=1524,
        domain="clinical_psychology",
        construct="depression_reduction",
        paradigm="behavioral_activation",
        heterogeneity_tau=0.15,
        i_squared=70.0,
        replication_status="replicated",
        moderators={
            "control": {"waitlist": 0.87, "treatment_as_usual": 0.56},
            "vs_cbt": {"vs_full_cbt": 0.02},
        },
        notes="BA as effective as full CBT for depression. Simpler to train and deliver."
    ),

    "sleep_hygiene_meta": MetaAnalyticEffect(
        source="Irwin et al. (2006); Mitchell et al. (2012)",
        effect_d=0.45,
        ci_95=(0.32, 0.58),
        n_studies=49,
        n_participants=5200,
        domain="health_psychology",
        construct="sleep_quality",
        paradigm="sleep_hygiene_intervention",
        heterogeneity_tau=0.12,
        i_squared=65.0,
        replication_status="replicated",
        moderators={
            "intervention": {"cbti": 0.70, "sleep_hygiene_only": 0.30, "relaxation": 0.40},
            "population": {"general": 0.40, "insomnia": 0.65},
        },
        notes="CBT-I (d = 0.70) far superior to sleep hygiene education alone (d = 0.30)."
    ),

    "psychological_pain_meta": MetaAnalyticEffect(
        source="Williams et al. (2012); Veehof et al. (2016)",
        effect_d=0.37,
        ci_95=(0.25, 0.49),
        n_studies=64,
        n_participants=6900,
        domain="health_psychology",
        construct="pain_reduction",
        paradigm="psychological_pain_management",
        heterogeneity_tau=0.14,
        i_squared=72.0,
        replication_status="replicated",
        moderators={
            "approach": {"cbt": 0.42, "act": 0.35, "mindfulness": 0.30},
            "pain_type": {"chronic_back": 0.38, "fibromyalgia": 0.40, "headache": 0.45},
        },
        notes="Psychological interventions moderately effective for chronic pain. CBT has most evidence."
    ),

    "psychotherapy_general_meta": MetaAnalyticEffect(
        source="Smith & Glass (1977); Wampold (2001)",
        effect_d=0.80,
        ci_95=(0.70, 0.90),
        n_studies=475,
        n_participants=50000,
        domain="clinical_psychology",
        construct="symptom_improvement",
        paradigm="psychotherapy_general",
        heterogeneity_tau=0.20,
        i_squared=82.0,
        replication_status="replicated",
        moderators={
            "control": {"no_treatment": 0.80, "waitlist": 0.70, "active_placebo": 0.35},
            "modality_difference": {"cbt_vs_psychodynamic": 0.05},
        },
        notes="Large overall effect. Small differences between bona fide therapies (dodo bird effect)."
    ),

    # ── EDUCATIONAL PSYCHOLOGY ───────────────────────────────────────────

    "interleaving_meta": MetaAnalyticEffect(
        source="Brunmair & Richter (2019); Rohrer (2012)",
        effect_d=0.42,
        ci_95=(0.30, 0.54),
        n_studies=57,
        n_participants=5800,
        domain="educational_psychology",
        construct="learning_transfer",
        paradigm="interleaved_practice",
        heterogeneity_tau=0.12,
        i_squared=66.0,
        replication_status="replicated",
        moderators={
            "material": {"math": 0.55, "motor_skills": 0.40, "verbal": 0.30},
            "similarity": {"high_similarity": 0.50, "low_similarity": 0.28},
        },
        notes="Interleaving improves discrimination learning. Largest for math and similar categories."
    ),

    "elaborative_interrogation_meta": MetaAnalyticEffect(
        source="Dunlosky et al. (2013); Ozgungor & Guthrie (2004)",
        effect_d=0.42,
        ci_95=(0.28, 0.56),
        n_studies=31,
        n_participants=3500,
        domain="educational_psychology",
        construct="fact_learning",
        paradigm="elaborative_interrogation",
        heterogeneity_tau=0.12,
        i_squared=64.0,
        replication_status="replicated",
        notes="Asking 'why is this true?' during study improves retention. Moderate utility."
    ),

    "self_explanation_meta": MetaAnalyticEffect(
        source="Rittle-Johnson (2006); Dunlosky et al. (2013)",
        effect_d=0.55,
        ci_95=(0.40, 0.70),
        n_studies=35,
        n_participants=4000,
        domain="educational_psychology",
        construct="conceptual_understanding",
        paradigm="self_explanation",
        heterogeneity_tau=0.14,
        i_squared=68.0,
        replication_status="replicated",
        moderators={
            "domain": {"science": 0.60, "math": 0.55, "reading": 0.40},
        },
        notes="Self-explanation promotes deeper understanding. Effective across STEM domains."
    ),

    "cooperative_learning_meta": MetaAnalyticEffect(
        source="Johnson & Johnson (2009); Roseth et al. (2008)",
        effect_d=0.54,
        ci_95=(0.45, 0.63),
        n_studies=148,
        n_participants=17000,
        domain="educational_psychology",
        construct="academic_achievement",
        paradigm="cooperative_learning",
        heterogeneity_tau=0.14,
        i_squared=72.0,
        replication_status="replicated",
        moderators={
            "structure": {"jigsaw": 0.58, "stad": 0.52, "unstructured_group": 0.30},
            "comparison": {"competitive": 0.62, "individualistic": 0.48},
        },
        notes="Structured cooperative learning outperforms both competitive and individual learning."
    ),

    "class_size_meta": MetaAnalyticEffect(
        source="Hattie (2009); Finn & Achilles (1999)",
        effect_d=0.21,
        ci_95=(0.12, 0.30),
        n_studies=96,
        n_participants=500000,
        domain="educational_psychology",
        construct="academic_achievement",
        paradigm="class_size_reduction",
        heterogeneity_tau=0.10,
        i_squared=62.0,
        replication_status="replicated",
        moderators={
            "grade_level": {"k_3": 0.28, "4_8": 0.18, "9_12": 0.12},
            "reduction_size": {"to_15": 0.28, "to_20": 0.18, "to_25": 0.10},
        },
        notes="Small effect overall. Largest benefit in early grades (K-3) with reduction to 15 or fewer."
    ),

    "homework_meta": MetaAnalyticEffect(
        source="Cooper et al. (2006); Fan et al. (2017)",
        effect_d=0.29,
        ci_95=(0.18, 0.40),
        n_studies=120,
        n_participants=100000,
        domain="educational_psychology",
        construct="academic_achievement",
        paradigm="homework",
        heterogeneity_tau=0.12,
        i_squared=70.0,
        replication_status="replicated",
        moderators={
            "grade_level": {"elementary": 0.10, "middle": 0.25, "high_school": 0.42},
            "subject": {"math": 0.35, "reading": 0.22, "science": 0.28},
        },
        notes="Homework benefit increases with age. Minimal benefit in elementary school."
    ),

    "tutoring_meta": MetaAnalyticEffect(
        source="Bloom (1984); VanLehn (2011); Nickow et al. (2020)",
        effect_d=0.56,
        ci_95=(0.42, 0.70),
        n_studies=96,
        n_participants=15000,
        domain="educational_psychology",
        construct="academic_achievement",
        paradigm="tutoring",
        heterogeneity_tau=0.15,
        i_squared=74.0,
        replication_status="replicated",
        moderators={
            "tutor_type": {"expert": 0.65, "peer": 0.40, "intelligent_tutoring_system": 0.45},
            "dosage": {"high_dose": 0.62, "low_dose": 0.38},
        },
        notes="Expert human tutoring approaches Bloom's 2-sigma. High-dosage tutoring programs effective."
    ),

    "formative_assessment_meta": MetaAnalyticEffect(
        source="Black & Wiliam (1998); Kingston & Nash (2011)",
        effect_d=0.40,
        ci_95=(0.28, 0.52),
        n_studies=250,
        n_participants=50000,
        domain="educational_psychology",
        construct="academic_achievement",
        paradigm="formative_assessment",
        heterogeneity_tau=0.15,
        i_squared=76.0,
        replication_status="replicated",
        moderators={
            "feedback_type": {"elaborated": 0.52, "corrective_only": 0.28},
            "frequency": {"frequent": 0.48, "occasional": 0.30},
        },
        notes="Formative assessment with elaborated feedback most effective."
    ),

    "spaced_practice_meta": MetaAnalyticEffect(
        source="Cepeda et al. (2006); Dunlosky et al. (2013)",
        effect_d=0.60,
        ci_95=(0.48, 0.72),
        n_studies=254,
        n_participants=14000,
        domain="educational_psychology",
        construct="long_term_retention",
        paradigm="spaced_practice",
        heterogeneity_tau=0.14,
        i_squared=72.0,
        replication_status="replicated",
        moderators={
            "retention_interval": {"1_day": 0.45, "1_week": 0.58, "1_month": 0.70},
            "material": {"verbal": 0.62, "motor": 0.50, "conceptual": 0.55},
        },
        notes="Spacing effect is one of most robust findings in learning. Stronger at longer retention intervals."
    ),

    "multimedia_learning_meta": MetaAnalyticEffect(
        source="Mayer (2009); Butcher (2014)",
        effect_d=0.52,
        ci_95=(0.38, 0.66),
        n_studies=86,
        n_participants=9000,
        domain="educational_psychology",
        construct="learning_outcomes",
        paradigm="multimedia_learning",
        heterogeneity_tau=0.14,
        i_squared=70.0,
        replication_status="replicated",
        moderators={
            "principle": {"modality": 0.58, "redundancy": 0.45, "coherence": 0.50, "signaling": 0.42},
        },
        notes="Words + pictures > words alone. Modality principle (narration > text with images) strongest."
    ),

    "worked_examples_meta": MetaAnalyticEffect(
        source="Atkinson et al. (2000); Renkl (2014)",
        effect_d=0.57,
        ci_95=(0.42, 0.72),
        n_studies=42,
        n_participants=4500,
        domain="educational_psychology",
        construct="problem_solving_transfer",
        paradigm="worked_examples",
        heterogeneity_tau=0.12,
        i_squared=66.0,
        replication_status="replicated",
        moderators={
            "expertise": {"novice": 0.65, "intermediate": 0.40, "expert": 0.10},
        },
        notes="Worked examples most effective for novices (expertise reversal effect). Reduces cognitive load."
    ),

    "problem_based_learning_meta": MetaAnalyticEffect(
        source="Dochy et al. (2003); Strobel & van Barneveld (2009)",
        effect_d=0.20,
        ci_95=(0.05, 0.35),
        n_studies=43,
        n_participants=8000,
        domain="educational_psychology",
        construct="knowledge_application",
        paradigm="problem_based_learning",
        heterogeneity_tau=0.12,
        i_squared=68.0,
        replication_status="replicated",
        moderators={
            "outcome": {"knowledge_recall": -0.10, "knowledge_application": 0.35, "clinical_skills": 0.42},
        },
        notes="PBL improves application/skills but slightly reduces factual knowledge retention."
    ),

    "flipped_classroom_meta": MetaAnalyticEffect(
        source="Lo & Hew (2017); Shi et al. (2020)",
        effect_d=0.35,
        ci_95=(0.22, 0.48),
        n_studies=55,
        n_participants=8500,
        domain="educational_psychology",
        construct="academic_achievement",
        paradigm="flipped_classroom",
        heterogeneity_tau=0.12,
        i_squared=66.0,
        replication_status="replicated",
        moderators={
            "discipline": {"stem": 0.40, "humanities": 0.25, "health_sciences": 0.38},
        },
        notes="Small-moderate benefit. Effectiveness depends on quality of in-class activities."
    ),

    "growth_mindset_intervention_meta": MetaAnalyticEffect(
        source="Sisk et al. (2018); Yeager et al. (2019)",
        effect_d=0.10,
        ci_95=(0.04, 0.16),
        n_studies=43,
        n_participants=57000,
        domain="educational_psychology",
        construct="academic_achievement",
        paradigm="growth_mindset_intervention",
        heterogeneity_tau=0.08,
        i_squared=55.0,
        replication_status="contested",
        moderators={
            "student_risk": {"low_risk": 0.06, "high_risk": 0.19},
            "school_norms": {"supportive": 0.14, "unsupportive": 0.04},
        },
        notes="Small overall. Larger for at-risk students in supportive school contexts."
    ),

    # ── ORGANIZATIONAL BEHAVIOR ──────────────────────────────────────────

    "goal_setting_meta": MetaAnalyticEffect(
        source="Locke & Latham (2002); Klein et al. (1999)",
        effect_d=0.58,
        ci_95=(0.48, 0.68),
        n_studies=400,
        n_participants=40000,
        domain="organizational_behavior",
        construct="task_performance",
        paradigm="goal_setting",
        heterogeneity_tau=0.14,
        i_squared=72.0,
        replication_status="replicated",
        moderators={
            "goal_type": {"specific_difficult": 0.62, "do_your_best": 0.15, "easy": 0.22},
            "feedback": {"with_feedback": 0.65, "without_feedback": 0.42},
            "commitment": {"high": 0.68, "low": 0.35},
        },
        notes="Specific, difficult goals with feedback = best performance. Among most robust findings in OB."
    ),

    "diversity_training_meta": MetaAnalyticEffect(
        source="Bezrukova et al. (2016); Paluck et al. (2021)",
        effect_d=0.23,
        ci_95=(0.12, 0.34),
        n_studies=260,
        n_participants=30000,
        domain="organizational_behavior",
        construct="attitude_behavior_change",
        paradigm="diversity_training",
        heterogeneity_tau=0.14,
        i_squared=72.0,
        replication_status="replicated",
        moderators={
            "outcome": {"attitudes": 0.30, "knowledge": 0.35, "behavior": 0.10},
            "mandatory_voluntary": {"mandatory": 0.15, "voluntary": 0.32},
            "duration": {"short_4h": 0.18, "long_8h_plus": 0.30},
        },
        notes="Small effects, especially on behavior. Mandatory training can backfire."
    ),

    "team_diversity_meta": MetaAnalyticEffect(
        source="van Dijk et al. (2012); Bell et al. (2011)",
        effect_d=0.05,
        ci_95=(-0.08, 0.18),
        n_studies=108,
        n_participants=10800,
        domain="organizational_behavior",
        construct="team_performance",
        paradigm="team_diversity",
        heterogeneity_tau=0.15,
        i_squared=75.0,
        replication_status="replicated",
        moderators={
            "diversity_type": {"demographic": -0.05, "functional": 0.15, "cognitive": 0.22},
            "task_type": {"routine": -0.08, "creative": 0.18},
        },
        notes="Demographic diversity near-zero or negative. Functional/cognitive diversity positive for creative tasks."
    ),

    "remote_work_meta": MetaAnalyticEffect(
        source="Gajendran & Harrison (2007); Allen et al. (2015)",
        effect_d=0.12,
        ci_95=(0.02, 0.22),
        n_studies=46,
        n_participants=12000,
        domain="organizational_behavior",
        construct="job_performance",
        paradigm="telecommuting",
        heterogeneity_tau=0.08,
        i_squared=55.0,
        replication_status="replicated",
        moderators={
            "intensity": {"partial_1_2_days": 0.18, "full_remote": 0.05},
            "outcome": {"productivity": 0.15, "satisfaction": 0.22, "turnover_intent": 0.25},
        },
        notes="Small positive effect on performance. Partial remote optimal."
    ),

    "job_autonomy_meta": MetaAnalyticEffect(
        source="Humphrey et al. (2007); Spector (1986)",
        effect_d=0.52,
        ci_95=(0.42, 0.62),
        n_studies=259,
        n_participants=220000,
        domain="organizational_behavior",
        construct="job_satisfaction",
        paradigm="job_characteristics",
        heterogeneity_tau=0.14,
        i_squared=72.0,
        replication_status="replicated",
        moderators={
            "autonomy_facet": {"work_scheduling": 0.50, "work_methods": 0.55, "decision_making": 0.58},
        },
        notes="Autonomy consistently among strongest predictors of satisfaction and performance."
    ),

    "organizational_justice_expanded_meta": MetaAnalyticEffect(
        source="Colquitt et al. (2013); Cohen-Charash & Spector (2001)",
        effect_d=0.65,
        ci_95=(0.55, 0.75),
        n_studies=493,
        n_participants=200000,
        domain="organizational_behavior",
        construct="work_outcomes",
        paradigm="organizational_justice",
        heterogeneity_tau=0.15,
        i_squared=76.0,
        replication_status="replicated",
        moderators={
            "justice_type": {"procedural": 0.58, "distributive": 0.52, "interpersonal": 0.65, "informational": 0.55},
            "outcome": {"satisfaction": 0.62, "commitment": 0.55, "ocb": 0.42, "cwb": -0.38},
        },
        notes="All justice types predict outcomes. Interpersonal justice strongest for supervisor satisfaction."
    ),

    "emotional_labor_meta": MetaAnalyticEffect(
        source="Hulsheger & Schewe (2011); Kammeyer-Mueller et al. (2013)",
        effect_d=0.40,
        ci_95=(0.30, 0.50),
        n_studies=95,
        n_participants=22000,
        domain="organizational_behavior",
        construct="emotional_exhaustion",
        paradigm="emotional_labor",
        heterogeneity_tau=0.12,
        i_squared=68.0,
        replication_status="replicated",
        moderators={
            "strategy": {"surface_acting": 0.48, "deep_acting": -0.10},
            "outcome": {"burnout": 0.44, "satisfaction": -0.35, "performance": -0.15},
        },
        notes="Surface acting (faking emotions) harmful. Deep acting (reappraising) protective."
    ),

    "workplace_bullying_meta": MetaAnalyticEffect(
        source="Nielsen & Einarsen (2012); Hershcovis (2011)",
        effect_d=0.65,
        ci_95=(0.52, 0.78),
        n_studies=66,
        n_participants=75000,
        domain="organizational_behavior",
        construct="mental_health_outcomes",
        paradigm="workplace_bullying",
        heterogeneity_tau=0.15,
        i_squared=76.0,
        replication_status="replicated",
        moderators={
            "outcome": {"mental_health": 0.68, "physical_health": 0.35, "job_satisfaction": -0.55, "turnover": 0.50},
        },
        notes="Bullying strongly predicts mental health problems and turnover intention."
    ),

    "employee_voice_meta": MetaAnalyticEffect(
        source="Chamberlin et al. (2017); Morrison (2011)",
        effect_d=0.30,
        ci_95=(0.20, 0.40),
        n_studies=42,
        n_participants=15000,
        domain="organizational_behavior",
        construct="work_outcomes",
        paradigm="employee_voice",
        heterogeneity_tau=0.10,
        i_squared=60.0,
        replication_status="replicated",
        moderators={
            "voice_type": {"promotive": 0.35, "prohibitive": 0.22},
            "outcome": {"performance": 0.28, "satisfaction": 0.32, "supervisor_rating": 0.25},
        },
        notes="Voice behavior positively related to performance. Promotive voice more valued than prohibitive."
    ),

    "training_transfer_meta": MetaAnalyticEffect(
        source="Blume et al. (2010); Burke & Hutchins (2007)",
        effect_d=0.35,
        ci_95=(0.24, 0.46),
        n_studies=89,
        n_participants=12000,
        domain="organizational_behavior",
        construct="training_transfer",
        paradigm="workplace_training",
        heterogeneity_tau=0.12,
        i_squared=68.0,
        replication_status="replicated",
        moderators={
            "transfer_climate": {"supportive": 0.48, "unsupportive": 0.15},
            "training_type": {"open_skills": 0.25, "closed_skills": 0.45},
        },
        notes="Only ~20-30% of training transfers to workplace. Transfer climate critical moderator."
    ),

    "mentoring_meta": MetaAnalyticEffect(
        source="Allen et al. (2004); Eby et al. (2008)",
        effect_d=0.32,
        ci_95=(0.22, 0.42),
        n_studies=43,
        n_participants=12000,
        domain="organizational_behavior",
        construct="career_outcomes",
        paradigm="formal_mentoring",
        heterogeneity_tau=0.10,
        i_squared=62.0,
        replication_status="replicated",
        moderators={
            "outcome": {"career_satisfaction": 0.38, "compensation": 0.18, "promotion": 0.22},
            "mentor_type": {"formal": 0.28, "informal": 0.42},
        },
        notes="Informal mentoring produces larger effects. Career-related mentoring > psychosocial."
    ),

    "personality_job_fit_meta": MetaAnalyticEffect(
        source="Barrick & Mount (1991); Kristof-Brown et al. (2005)",
        effect_d=0.44,
        ci_95=(0.34, 0.54),
        n_studies=162,
        n_participants=40000,
        domain="organizational_behavior",
        construct="job_performance",
        paradigm="personality_assessment",
        heterogeneity_tau=0.14,
        i_squared=72.0,
        replication_status="replicated",
        moderators={
            "trait": {"conscientiousness": 0.46, "emotional_stability": 0.30, "agreeableness": 0.18, "extraversion": 0.25, "openness": 0.12},
            "job_type": {"sales": 0.35, "managerial": 0.28, "professional": 0.22},
        },
        notes="Conscientiousness most valid predictor across jobs. Extraversion for sales/management."
    ),

    "recruitment_source_meta": MetaAnalyticEffect(
        source="Zottoli & Wanous (2000); Breaugh (2008)",
        effect_d=0.18,
        ci_95=(0.08, 0.28),
        n_studies=50,
        n_participants=25000,
        domain="organizational_behavior",
        construct="turnover_performance",
        paradigm="recruitment",
        heterogeneity_tau=0.08,
        i_squared=55.0,
        replication_status="replicated",
        moderators={
            "source": {"referral": 0.22, "job_board": 0.10, "campus": 0.12},
        },
        notes="Employee referrals predict slightly better outcomes. Small overall effect of source."
    ),

    "performance_appraisal_meta": MetaAnalyticEffect(
        source="Jawahar & Williams (1997); Adler et al. (2016)",
        effect_d=0.35,
        ci_95=(0.22, 0.48),
        n_studies=54,
        n_participants=18000,
        domain="organizational_behavior",
        construct="rater_accuracy",
        paradigm="performance_appraisal",
        heterogeneity_tau=0.12,
        i_squared=68.0,
        replication_status="replicated",
        moderators={
            "format": {"behavioral": 0.42, "trait_based": 0.25, "outcome_based": 0.38},
            "training": {"trained_raters": 0.45, "untrained": 0.22},
        },
        notes="Rater training and behavioral anchors improve accuracy. Still substantial rater bias."
    ),

    # ── CONSUMER & MARKETING ────────────────────────────────────────────

    "price_anchoring_meta": MetaAnalyticEffect(
        source="Furnham & Boo (2011); Mussweiler et al. (2000)",
        effect_d=0.80,
        ci_95=(0.62, 0.98),
        n_studies=85,
        n_participants=16000,
        domain="consumer_psychology",
        construct="willingness_to_pay",
        paradigm="price_anchoring",
        heterogeneity_tau=0.18,
        i_squared=80.0,
        replication_status="replicated",
        moderators={
            "anchor_plausibility": {"plausible": 0.85, "implausible": 0.60},
            "expertise": {"novice": 0.90, "expert": 0.50},
        },
        notes="Large robust effect. Even experts anchored. Higher anchors inflate WTP."
    ),

    "social_proof_marketing_meta": MetaAnalyticEffect(
        source="Cialdini (2001); Luo et al. (2014)",
        effect_d=0.38,
        ci_95=(0.26, 0.50),
        n_studies=67,
        n_participants=18000,
        domain="consumer_psychology",
        construct="purchase_behavior",
        paradigm="social_proof",
        heterogeneity_tau=0.12,
        i_squared=68.0,
        replication_status="replicated",
        moderators={
            "source": {"peers": 0.45, "experts": 0.40, "crowd": 0.32},
            "uncertainty": {"high_uncertainty": 0.50, "low_uncertainty": 0.22},
        },
        notes="Social proof most effective under uncertainty. Peer recommendations strongest."
    ),

    "celebrity_endorsement_meta": MetaAnalyticEffect(
        source="Amos et al. (2008); Knoll & Matthes (2017)",
        effect_d=0.25,
        ci_95=(0.15, 0.35),
        n_studies=46,
        n_participants=11000,
        domain="consumer_psychology",
        construct="brand_attitude",
        paradigm="celebrity_endorsement",
        heterogeneity_tau=0.10,
        i_squared=62.0,
        replication_status="replicated",
        moderators={
            "congruence": {"high_fit": 0.38, "low_fit": 0.10},
            "attractiveness": {"attractive": 0.30, "expert": 0.28},
            "credibility": {"high": 0.35, "low": 0.12},
        },
        notes="Small-moderate effect. Celebrity-brand fit is key moderator."
    ),

    "country_of_origin_meta": MetaAnalyticEffect(
        source="Verlegh & Steenkamp (1999); Magnusson et al. (2011)",
        effect_d=0.38,
        ci_95=(0.28, 0.48),
        n_studies=52,
        n_participants=15000,
        domain="consumer_psychology",
        construct="product_evaluation",
        paradigm="country_of_origin",
        heterogeneity_tau=0.12,
        i_squared=68.0,
        replication_status="replicated",
        moderators={
            "product_type": {"luxury": 0.48, "utilitarian": 0.28, "technology": 0.42},
            "cue_type": {"single_cue": 0.55, "multi_cue": 0.28},
        },
        notes="COO effect weakens in multi-cue settings. Strongest for luxury/hedonic products."
    ),

    "brand_extension_meta": MetaAnalyticEffect(
        source="Volckner & Sattler (2006); Bottomley & Holden (2001)",
        effect_d=0.42,
        ci_95=(0.30, 0.54),
        n_studies=48,
        n_participants=12000,
        domain="consumer_psychology",
        construct="extension_evaluation",
        paradigm="brand_extension",
        heterogeneity_tau=0.12,
        i_squared=66.0,
        replication_status="replicated",
        moderators={
            "fit": {"high_fit": 0.55, "moderate_fit": 0.35, "low_fit": 0.12},
            "parent_quality": {"high": 0.52, "average": 0.35},
        },
        notes="Perceived fit between parent brand and extension is dominant success factor."
    ),

    "product_placement_meta": MetaAnalyticEffect(
        source="van Reijmersdal et al. (2009); Dens et al. (2012)",
        effect_d=0.18,
        ci_95=(0.08, 0.28),
        n_studies=38,
        n_participants=9000,
        domain="consumer_psychology",
        construct="brand_memory_attitude",
        paradigm="product_placement",
        heterogeneity_tau=0.08,
        i_squared=55.0,
        replication_status="replicated",
        moderators={
            "prominence": {"prominent": 0.25, "subtle": 0.10},
            "outcome": {"recall": 0.28, "attitude": 0.12, "purchase_intent": 0.08},
        },
        notes="Small effect. Prominent placements increase recall but may trigger reactance."
    ),

    "retargeting_meta": MetaAnalyticEffect(
        source="Lambrecht & Tucker (2013); Bleier & Eisenbeiss (2015)",
        effect_d=0.22,
        ci_95=(0.10, 0.34),
        n_studies=18,
        n_participants=500000,
        domain="consumer_psychology",
        construct="conversion_rate",
        paradigm="behavioral_retargeting",
        heterogeneity_tau=0.08,
        i_squared=58.0,
        replication_status="replicated",
        moderators={
            "specificity": {"generic": 0.15, "dynamic_specific": 0.30},
            "timing": {"immediate": 0.28, "delayed": 0.15},
        },
        notes="Retargeting increases conversion. Dynamic/specific ads outperform generic."
    ),

    "loyalty_program_meta": MetaAnalyticEffect(
        source="Dorotic et al. (2012); Breugelmans et al. (2015)",
        effect_d=0.20,
        ci_95=(0.10, 0.30),
        n_studies=35,
        n_participants=100000,
        domain="consumer_psychology",
        construct="purchase_frequency",
        paradigm="loyalty_program",
        heterogeneity_tau=0.08,
        i_squared=58.0,
        replication_status="replicated",
        moderators={
            "program_type": {"points": 0.22, "tiered": 0.28, "cashback": 0.15},
            "industry": {"airline": 0.25, "retail": 0.18, "hospitality": 0.22},
        },
        notes="Modest effects on purchase behavior. Tiered programs most effective."
    ),

    "personalization_meta": MetaAnalyticEffect(
        source="Aguirre et al. (2015); Tam & Ho (2006)",
        effect_d=0.32,
        ci_95=(0.20, 0.44),
        n_studies=40,
        n_participants=35000,
        domain="consumer_psychology",
        construct="response_rate",
        paradigm="personalization",
        heterogeneity_tau=0.10,
        i_squared=64.0,
        replication_status="replicated",
        moderators={
            "type": {"content": 0.38, "name_only": 0.15, "recommendation": 0.35},
            "privacy_concern": {"low": 0.40, "high": 0.15},
        },
        notes="Content-based personalization effective but privacy concerns moderate effects."
    ),

    "user_review_meta": MetaAnalyticEffect(
        source="Floyd et al. (2014); Babic Rosario et al. (2016)",
        effect_d=0.35,
        ci_95=(0.25, 0.45),
        n_studies=96,
        n_participants=50000,
        domain="consumer_psychology",
        construct="sales_purchase_intention",
        paradigm="online_reviews",
        heterogeneity_tau=0.12,
        i_squared=68.0,
        replication_status="replicated",
        moderators={
            "valence": {"positive": 0.40, "negative": -0.50},
            "platform": {"independent": 0.38, "retailer": 0.28},
            "product_type": {"experience": 0.42, "search": 0.28},
        },
        notes="Negative reviews have larger absolute impact than positive. Experience goods most affected."
    ),

    "nostalgia_marketing_meta": MetaAnalyticEffect(
        source="Sedikides et al. (2015); Muehling & Pascal (2011)",
        effect_d=0.38,
        ci_95=(0.25, 0.51),
        n_studies=30,
        n_participants=6000,
        domain="consumer_psychology",
        construct="brand_evaluation",
        paradigm="nostalgia_appeal",
        heterogeneity_tau=0.10,
        i_squared=62.0,
        replication_status="replicated",
        moderators={
            "nostalgia_type": {"personal": 0.42, "historical": 0.28},
            "product_type": {"hedonic": 0.45, "utilitarian": 0.22},
        },
        notes="Nostalgia evokes warmth and social connectedness. Most effective for hedonic products."
    ),

    "cause_marketing_meta": MetaAnalyticEffect(
        source="Winterich & Barone (2011); Lafferty et al. (2016)",
        effect_d=0.28,
        ci_95=(0.16, 0.40),
        n_studies=42,
        n_participants=12000,
        domain="consumer_psychology",
        construct="purchase_intention",
        paradigm="cause_related_marketing",
        heterogeneity_tau=0.10,
        i_squared=62.0,
        replication_status="replicated",
        moderators={
            "brand_cause_fit": {"high_fit": 0.38, "low_fit": 0.12},
            "donation_size": {"small": 0.20, "large": 0.35},
        },
        notes="Cause-brand fit is key. Skepticism about motives can undermine effectiveness."
    ),

    "sensory_marketing_meta": MetaAnalyticEffect(
        source="Krishna (2012); Spence et al. (2014)",
        effect_d=0.35,
        ci_95=(0.22, 0.48),
        n_studies=45,
        n_participants=8000,
        domain="consumer_psychology",
        construct="product_evaluation",
        paradigm="sensory_marketing",
        heterogeneity_tau=0.12,
        i_squared=66.0,
        replication_status="replicated",
        moderators={
            "sense": {"touch": 0.42, "smell": 0.38, "sound": 0.30, "taste": 0.35},
        },
        notes="Multi-sensory experiences enhance product evaluation. Touch (haptics) particularly impactful."
    ),

    "green_marketing_meta": MetaAnalyticEffect(
        source="Trivedi et al. (2018); Dangelico & Vocalelli (2017)",
        effect_d=0.30,
        ci_95=(0.18, 0.42),
        n_studies=55,
        n_participants=15000,
        domain="consumer_psychology",
        construct="purchase_intention",
        paradigm="green_marketing",
        heterogeneity_tau=0.12,
        i_squared=68.0,
        replication_status="replicated",
        moderators={
            "greenwashing_risk": {"credible": 0.38, "suspected_greenwashing": 0.08},
            "product_type": {"everyday": 0.28, "high_involvement": 0.38},
        },
        notes="Green claims boost intention but greenwashing perception eliminates the effect."
    ),

    # ── COGNITIVE & DECISION-MAKING ──────────────────────────────────────

    "dunning_kruger_meta": MetaAnalyticEffect(
        source="Kruger & Dunning (1999); Gignac & Zajenkowski (2020)",
        effect_d=0.45,
        ci_95=(0.30, 0.60),
        n_studies=30,
        n_participants=8000,
        domain="cognitive_psychology",
        construct="metacognitive_miscalibration",
        paradigm="self_assessment_accuracy",
        heterogeneity_tau=0.14,
        i_squared=70.0,
        replication_status="contested",
        moderators={
            "domain": {"logical_reasoning": 0.55, "humor": 0.40, "grammar": 0.42},
        },
        notes="Bottom quartile overestimates by ~50 percentile points. Debate over statistical artifact."
    ),

    "hindsight_bias_meta": MetaAnalyticEffect(
        source="Guilbault et al. (2004); Roese & Vohs (2012)",
        effect_d=0.39,
        ci_95=(0.30, 0.48),
        n_studies=122,
        n_participants=15000,
        domain="cognitive_psychology",
        construct="memory_distortion",
        paradigm="hindsight_bias",
        heterogeneity_tau=0.12,
        i_squared=68.0,
        replication_status="replicated",
        moderators={
            "judgment_type": {"memory": 0.45, "inevitability": 0.35, "foreseeability": 0.38},
            "outcome_valence": {"negative": 0.45, "positive": 0.32},
        },
        notes="'Knew-it-all-along' effect. Robust. Stronger for negative outcomes."
    ),

    "planning_fallacy_meta": MetaAnalyticEffect(
        source="Buehler et al. (2010); Halkjelsvik & Jorgensen (2012 meta)",
        effect_d=0.52,
        ci_95=(0.38, 0.66),
        n_studies=60,
        n_participants=8000,
        domain="cognitive_psychology",
        construct="time_estimation_bias",
        paradigm="planning_fallacy",
        heterogeneity_tau=0.14,
        i_squared=72.0,
        replication_status="replicated",
        moderators={
            "task_type": {"academic": 0.55, "home_projects": 0.50, "software": 0.60},
            "experience": {"novice": 0.58, "experienced": 0.35},
        },
        notes="People underestimate completion time by 30-50%. Even experienced planners show bias."
    ),

    "gamblers_fallacy_meta": MetaAnalyticEffect(
        source="Ayton & Fischer (2004); Suetens et al. (2016)",
        effect_d=0.35,
        ci_95=(0.22, 0.48),
        n_studies=32,
        n_participants=6000,
        domain="cognitive_psychology",
        construct="sequence_expectation",
        paradigm="gamblers_fallacy",
        heterogeneity_tau=0.10,
        i_squared=60.0,
        replication_status="replicated",
        notes="Expecting reversal after runs in random sequences. Robust in lab and field."
    ),

    "hot_hand_belief": MetaAnalyticEffect(
        source="Gilovich et al. (1985); Miller & Sanjurjo (2018)",
        effect_d=0.15,
        ci_95=(0.02, 0.28),
        n_studies=22,
        n_participants=5000,
        domain="cognitive_psychology",
        construct="streak_perception",
        paradigm="hot_hand",
        heterogeneity_tau=0.08,
        i_squared=55.0,
        replication_status="contested",
        notes="People believe in hot streaks. Miller & Sanjurjo (2018) showed small real hot hand exists."
    ),

    "availability_heuristic_meta": MetaAnalyticEffect(
        source="Tversky & Kahneman (1973); Pachur et al. (2012)",
        effect_d=0.48,
        ci_95=(0.35, 0.61),
        n_studies=55,
        n_participants=8000,
        domain="cognitive_psychology",
        construct="frequency_probability_judgment",
        paradigm="availability_heuristic",
        heterogeneity_tau=0.14,
        i_squared=70.0,
        replication_status="replicated",
        moderators={
            "ease_vs_content": {"ease_of_retrieval": 0.55, "number_retrieved": 0.38},
        },
        notes="Ease of retrieval often more impactful than number of instances retrieved."
    ),

    "representativeness_meta": MetaAnalyticEffect(
        source="Tversky & Kahneman (1974); Kahneman & Frederick (2002)",
        effect_d=0.52,
        ci_95=(0.38, 0.66),
        n_studies=45,
        n_participants=7500,
        domain="cognitive_psychology",
        construct="probability_judgment",
        paradigm="representativeness_heuristic",
        heterogeneity_tau=0.14,
        i_squared=70.0,
        replication_status="replicated",
        notes="Judging probability by similarity to prototype. Leads to base rate neglect."
    ),

    "base_rate_neglect_meta": MetaAnalyticEffect(
        source="Bar-Hillel (1980); Koehler (1996)",
        effect_d=0.55,
        ci_95=(0.40, 0.70),
        n_studies=40,
        n_participants=6500,
        domain="cognitive_psychology",
        construct="bayesian_updating",
        paradigm="base_rate_neglect",
        heterogeneity_tau=0.14,
        i_squared=70.0,
        replication_status="replicated",
        moderators={
            "salience": {"causal_base_rate": 0.30, "statistical_base_rate": 0.65},
        },
        notes="People underweight base rates when individuating info available. Causal base rates used more."
    ),

    "conjunction_fallacy_meta": MetaAnalyticEffect(
        source="Tversky & Kahneman (1983); Mellers et al. (2001)",
        effect_d=0.60,
        ci_95=(0.45, 0.75),
        n_studies=35,
        n_participants=5000,
        domain="cognitive_psychology",
        construct="probability_judgment",
        paradigm="conjunction_fallacy",
        heterogeneity_tau=0.12,
        i_squared=65.0,
        replication_status="replicated",
        moderators={
            "scenario_type": {"classic_linda": 0.70, "medical": 0.48, "sports": 0.50},
        },
        notes="60-80% of participants commit the fallacy in classic Linda problem. Robust."
    ),

    "peak_end_rule_meta": MetaAnalyticEffect(
        source="Kahneman et al. (1993); Do et al. (2008 meta)",
        effect_d=0.42,
        ci_95=(0.28, 0.56),
        n_studies=35,
        n_participants=4500,
        domain="cognitive_psychology",
        construct="retrospective_evaluation",
        paradigm="peak_end_rule",
        heterogeneity_tau=0.12,
        i_squared=66.0,
        replication_status="replicated",
        moderators={
            "valence": {"pain": 0.50, "pleasure": 0.35},
            "duration_neglect": {"short": 0.35, "long": 0.50},
        },
        notes="Retrospective evaluation dominated by peak intensity and ending. Duration largely neglected."
    ),

    "ikea_effect_meta": MetaAnalyticEffect(
        source="Norton et al. (2012); Marsh et al. (2018)",
        effect_d=0.48,
        ci_95=(0.32, 0.64),
        n_studies=15,
        n_participants=3000,
        domain="cognitive_psychology",
        construct="effort_valuation",
        paradigm="ikea_effect",
        heterogeneity_tau=0.10,
        i_squared=58.0,
        replication_status="replicated",
        moderators={
            "completion": {"completed": 0.55, "incomplete": 0.15},
        },
        notes="Self-made products valued ~63% more than identical pre-made. Requires successful completion."
    ),

    "decoy_effect_meta": MetaAnalyticEffect(
        source="Huber et al. (1982); Frederick et al. (2014)",
        effect_d=0.42,
        ci_95=(0.28, 0.56),
        n_studies=45,
        n_participants=8000,
        domain="cognitive_psychology",
        construct="choice_shift",
        paradigm="attraction_effect",
        heterogeneity_tau=0.14,
        i_squared=70.0,
        replication_status="replicated",
        moderators={
            "decoy_type": {"asymmetric_dominance": 0.48, "compromise": 0.38},
            "product_category": {"consumer": 0.45, "gambles": 0.35},
        },
        notes="Adding dominated decoy shifts choice toward target. Robust but context-dependent."
    ),

    "status_quo_bias_meta": MetaAnalyticEffect(
        source="Samuelson & Zeckhauser (1988); Dean (2008)",
        effect_d=0.50,
        ci_95=(0.35, 0.65),
        n_studies=52,
        n_participants=10000,
        domain="cognitive_psychology",
        construct="preference_for_default",
        paradigm="status_quo_bias",
        heterogeneity_tau=0.14,
        i_squared=72.0,
        replication_status="replicated",
        moderators={
            "switching_cost": {"low": 0.35, "high": 0.65},
            "options": {"few_2_3": 0.40, "many_5_plus": 0.58},
        },
        notes="Preference for current state. Increases with number of alternatives and switching costs."
    ),

    "zero_price_effect": MetaAnalyticEffect(
        source="Shampanier et al. (2007); Nicolau & Sellers (2012)",
        effect_d=0.62,
        ci_95=(0.45, 0.79),
        n_studies=18,
        n_participants=4500,
        domain="cognitive_psychology",
        construct="preference_shift",
        paradigm="zero_price",
        heterogeneity_tau=0.12,
        i_squared=62.0,
        replication_status="replicated",
        notes="Free items disproportionately preferred. Qualitative shift at zero price."
    ),

    "denomination_effect": MetaAnalyticEffect(
        source="Raghubir & Srivastava (2009); Mishra et al. (2006)",
        effect_d=0.38,
        ci_95=(0.22, 0.54),
        n_studies=12,
        n_participants=3000,
        domain="cognitive_psychology",
        construct="spending_behavior",
        paradigm="denomination_effect",
        heterogeneity_tau=0.10,
        i_squared=58.0,
        replication_status="replicated",
        notes="People spend less when money is in large denominations vs equivalent small denominations."
    ),

    "framing_general_meta": MetaAnalyticEffect(
        source="Kuhberger (1998); Steiger & Kuhberger (2018)",
        effect_d=0.31,
        ci_95=(0.22, 0.40),
        n_studies=230,
        n_participants=30000,
        domain="cognitive_psychology",
        construct="choice_reversal",
        paradigm="risky_choice_framing",
        heterogeneity_tau=0.15,
        i_squared=76.0,
        replication_status="replicated",
        moderators={
            "framing_type": {"gain_loss": 0.35, "attribute": 0.28, "goal": 0.30},
            "material": {"asian_disease": 0.42, "medical": 0.32, "financial": 0.25},
        },
        notes="Gain frame = risk averse; loss frame = risk seeking. Robust but highly moderated."
    ),

    # ── DEVELOPMENTAL & PARENTING ────────────────────────────────────────

    "attachment_security_meta": MetaAnalyticEffect(
        source="Groh et al. (2017); Fearon et al. (2010)",
        effect_d=0.40,
        ci_95=(0.30, 0.50),
        n_studies=127,
        n_participants=22000,
        domain="developmental_psychology",
        construct="socioemotional_competence",
        paradigm="strange_situation",
        heterogeneity_tau=0.12,
        i_squared=68.0,
        replication_status="replicated",
        moderators={
            "outcome": {"social_competence": 0.42, "externalizing": -0.38, "internalizing": -0.30},
            "attachment": {"secure_vs_avoidant": 0.35, "secure_vs_disorganized": 0.55},
        },
        notes="Secure attachment predicts better social outcomes. Disorganized attachment strongest risk factor."
    ),

    "authoritative_parenting_meta": MetaAnalyticEffect(
        source="Pinquart (2017); Steinberg et al. (1994)",
        effect_d=0.35,
        ci_95=(0.25, 0.45),
        n_studies=428,
        n_participants=200000,
        domain="developmental_psychology",
        construct="child_adjustment",
        paradigm="parenting_style",
        heterogeneity_tau=0.14,
        i_squared=72.0,
        replication_status="replicated",
        moderators={
            "outcome": {"academic": 0.30, "social_competence": 0.40, "psychopathology": -0.35},
            "culture": {"western": 0.38, "east_asian": 0.22, "african_american": 0.25},
        },
        notes="Authoritative parenting (warm + firm) consistently best outcomes. Culture moderates."
    ),

    "early_intervention_meta": MetaAnalyticEffect(
        source="Camilli et al. (2010); Duncan & Magnuson (2013)",
        effect_d=0.35,
        ci_95=(0.24, 0.46),
        n_studies=123,
        n_participants=50000,
        domain="developmental_psychology",
        construct="cognitive_development",
        paradigm="early_childhood_intervention",
        heterogeneity_tau=0.14,
        i_squared=72.0,
        replication_status="replicated",
        moderators={
            "program_quality": {"high_quality": 0.48, "average_quality": 0.28},
            "outcome_timing": {"immediate": 0.50, "age_5": 0.35, "age_10_plus": 0.20},
        },
        notes="Effects fade over time but high-quality programs show lasting benefits."
    ),

    "adverse_childhood_experiences": MetaAnalyticEffect(
        source="Hughes et al. (2017); Bellis et al. (2019)",
        effect_d=0.55,
        ci_95=(0.42, 0.68),
        n_studies=96,
        n_participants=250000,
        domain="developmental_psychology",
        construct="adult_health_outcomes",
        paradigm="ace_study",
        heterogeneity_tau=0.15,
        i_squared=76.0,
        replication_status="replicated",
        moderators={
            "ace_count": {"1_3": 0.35, "4_plus": 0.75},
            "outcome": {"mental_health": 0.60, "substance_use": 0.52, "physical_health": 0.38},
        },
        notes="Dose-response: 4+ ACEs double risk of depression. Robust across populations."
    ),

    "bullying_effects_meta": MetaAnalyticEffect(
        source="Reijntjes et al. (2010); Ttofi et al. (2011)",
        effect_d=0.42,
        ci_95=(0.30, 0.54),
        n_studies=80,
        n_participants=45000,
        domain="developmental_psychology",
        construct="adjustment_problems",
        paradigm="peer_victimization",
        heterogeneity_tau=0.14,
        i_squared=72.0,
        replication_status="replicated",
        moderators={
            "outcome": {"depression": 0.45, "anxiety": 0.40, "self_esteem": -0.38, "academic": -0.30},
            "type": {"traditional": 0.40, "cyber": 0.45},
        },
        notes="Victimization prospectively predicts internalizing problems. Cyberbullying similar magnitude."
    ),

    "screen_time_meta": MetaAnalyticEffect(
        source="Orben & Przybylski (2019); Hancock et al. (2022)",
        effect_d=0.08,
        ci_95=(0.02, 0.14),
        n_studies=80,
        n_participants=300000,
        domain="developmental_psychology",
        construct="wellbeing",
        paradigm="screen_time",
        heterogeneity_tau=0.06,
        i_squared=50.0,
        replication_status="contested",
        moderators={
            "content_type": {"social_media": 0.12, "educational": -0.05, "passive_video": 0.10},
            "age": {"adolescent": 0.10, "adult": 0.05},
        },
        notes="Very small association with wellbeing. r = -0.04. Comparable to wearing glasses. Debated."
    ),

    "parental_monitoring_meta": MetaAnalyticEffect(
        source="Stattin & Kerr (2000); Li et al. (2013 meta)",
        effect_d=0.40,
        ci_95=(0.30, 0.50),
        n_studies=52,
        n_participants=30000,
        domain="developmental_psychology",
        construct="adolescent_risk_behavior",
        paradigm="parental_monitoring",
        heterogeneity_tau=0.12,
        i_squared=68.0,
        replication_status="replicated",
        moderators={
            "source": {"child_disclosure": 0.48, "parental_solicitation": 0.25, "parental_control": 0.20},
        },
        notes="Child voluntary disclosure stronger predictor than parental solicitation/control."
    ),

    "helicopter_parenting_meta": MetaAnalyticEffect(
        source="Schiffrin et al. (2014); Luebbe et al. (2018)",
        effect_d=0.28,
        ci_95=(0.15, 0.41),
        n_studies=30,
        n_participants=8000,
        domain="developmental_psychology",
        construct="emerging_adult_adjustment",
        paradigm="overparenting",
        heterogeneity_tau=0.10,
        i_squared=60.0,
        replication_status="replicated",
        moderators={
            "outcome": {"anxiety": 0.32, "depression": 0.28, "self_efficacy": -0.25},
        },
        notes="Overparenting associated with poorer adjustment in emerging adults."
    ),

    "resilience_intervention_meta": MetaAnalyticEffect(
        source="Leppin et al. (2014); Joyce et al. (2018)",
        effect_d=0.37,
        ci_95=(0.24, 0.50),
        n_studies=25,
        n_participants=3500,
        domain="developmental_psychology",
        construct="resilience",
        paradigm="resilience_training",
        heterogeneity_tau=0.10,
        i_squared=62.0,
        replication_status="replicated",
        moderators={
            "approach": {"cbt_based": 0.42, "mindfulness_based": 0.35, "mixed": 0.30},
            "population": {"military": 0.35, "students": 0.40, "healthcare_workers": 0.38},
        },
        notes="Resilience trainable. CBT-based programs most evidence. Effects moderate."
    ),

    "sel_programs_meta": MetaAnalyticEffect(
        source="Durlak et al. (2011); Taylor et al. (2017)",
        effect_d=0.30,
        ci_95=(0.22, 0.38),
        n_studies=213,
        n_participants=270000,
        domain="developmental_psychology",
        construct="social_emotional_outcomes",
        paradigm="social_emotional_learning",
        heterogeneity_tau=0.12,
        i_squared=68.0,
        replication_status="replicated",
        moderators={
            "outcome": {"social_skills": 0.35, "academic": 0.28, "conduct_problems": -0.25, "emotional_distress": -0.22},
            "implementation": {"high_fidelity": 0.38, "low_fidelity": 0.15},
        },
        notes="SEL programs improve social skills and academics. Implementation quality critical."
    ),

    # ── ENVIRONMENTAL PSYCHOLOGY ─────────────────────────────────────────

    "climate_communication_meta": MetaAnalyticEffect(
        source="van der Linden et al. (2015); Goldberg et al. (2021)",
        effect_d=0.22,
        ci_95=(0.10, 0.34),
        n_studies=48,
        n_participants=25000,
        domain="environmental_psychology",
        construct="climate_concern",
        paradigm="climate_communication",
        heterogeneity_tau=0.10,
        i_squared=64.0,
        replication_status="replicated",
        moderators={
            "message_type": {"scientific_consensus": 0.28, "local_impacts": 0.25, "fear_appeal": 0.15, "solution_focused": 0.30},
            "audience": {"concerned": 0.30, "doubtful": 0.12, "dismissive": 0.05},
        },
        notes="Small effects. Solution-focused and consensus messaging most effective."
    ),

    "social_norms_conservation_meta": MetaAnalyticEffect(
        source="Abrahamse & Steg (2013); Allcott (2011)",
        effect_d=0.35,
        ci_95=(0.24, 0.46),
        n_studies=64,
        n_participants=50000,
        domain="environmental_psychology",
        construct="energy_conservation",
        paradigm="social_norm_feedback",
        heterogeneity_tau=0.12,
        i_squared=68.0,
        replication_status="replicated",
        moderators={
            "norm_type": {"descriptive": 0.35, "injunctive": 0.28, "combined": 0.42},
            "domain": {"energy": 0.35, "water": 0.30, "recycling": 0.38},
        },
        notes="Social norm feedback reduces energy use 2-5%. Opower studies show persistent effects."
    ),

    "green_defaults_meta": MetaAnalyticEffect(
        source="Ebeling & Lotz (2015); Jachimowicz et al. (2019)",
        effect_d=0.68,
        ci_95=(0.50, 0.86),
        n_studies=25,
        n_participants=30000,
        domain="environmental_psychology",
        construct="green_choice",
        paradigm="default_option",
        heterogeneity_tau=0.15,
        i_squared=74.0,
        replication_status="replicated",
        moderators={
            "domain": {"green_energy": 0.75, "double_sided_printing": 0.62, "carbon_offset": 0.55},
            "cost": {"no_cost": 0.78, "small_cost": 0.52, "substantial_cost": 0.30},
        },
        notes="Green defaults very effective. 90%+ stick with green default when no cost difference."
    ),

    "eco_labeling_meta": MetaAnalyticEffect(
        source="Delmas & Grant (2014); Potter et al. (2021)",
        effect_d=0.25,
        ci_95=(0.14, 0.36),
        n_studies=42,
        n_participants=18000,
        domain="environmental_psychology",
        construct="sustainable_purchase",
        paradigm="eco_labeling",
        heterogeneity_tau=0.10,
        i_squared=62.0,
        replication_status="replicated",
        moderators={
            "label_type": {"certified_official": 0.32, "self_declared": 0.15},
            "product_type": {"food": 0.30, "electronics": 0.22, "clothing": 0.20},
        },
        notes="Official certification labels more trusted. Self-declared green claims less effective."
    ),

    "carbon_footprint_info_meta": MetaAnalyticEffect(
        source="Camilleri et al. (2019); Zhao & Zhong (2022)",
        effect_d=0.18,
        ci_95=(0.06, 0.30),
        n_studies=25,
        n_participants=8000,
        domain="environmental_psychology",
        construct="consumption_choice",
        paradigm="carbon_labeling",
        heterogeneity_tau=0.08,
        i_squared=55.0,
        replication_status="replicated",
        notes="Carbon footprint labels have small effects. Comparison formats more effective."
    ),

    "environmental_education_meta": MetaAnalyticEffect(
        source="Ardoin et al. (2018); Stern et al. (2014)",
        effect_d=0.45,
        ci_95=(0.32, 0.58),
        n_studies=66,
        n_participants=15000,
        domain="environmental_psychology",
        construct="environmental_knowledge_attitudes",
        paradigm="environmental_education",
        heterogeneity_tau=0.14,
        i_squared=72.0,
        replication_status="replicated",
        moderators={
            "outcome": {"knowledge": 0.55, "attitudes": 0.40, "behavior": 0.25},
            "setting": {"outdoor": 0.52, "classroom": 0.35},
        },
        notes="Strong on knowledge, moderate on attitudes, small on behavior. Outdoor programs better."
    ),

    "nature_exposure_wellbeing": MetaAnalyticEffect(
        source="McMahan & Estes (2015); Capaldi et al. (2014)",
        effect_d=0.40,
        ci_95=(0.28, 0.52),
        n_studies=70,
        n_participants=12000,
        domain="environmental_psychology",
        construct="subjective_wellbeing",
        paradigm="nature_exposure",
        heterogeneity_tau=0.12,
        i_squared=66.0,
        replication_status="replicated",
        moderators={
            "exposure_type": {"immersive_walk": 0.50, "window_view": 0.30, "virtual": 0.22},
            "duration": {"short_15min": 0.32, "long_60min_plus": 0.50},
        },
        notes="Nature exposure improves mood, reduces rumination. Real exposure > virtual."
    ),

    "values_behavior_gap_meta": MetaAnalyticEffect(
        source="Kollmuss & Agyeman (2002); Morren & Grinstein (2016)",
        effect_d=0.25,
        ci_95=(0.15, 0.35),
        n_studies=56,
        n_participants=20000,
        domain="environmental_psychology",
        construct="attitude_behavior_consistency",
        paradigm="environmental_values_behavior",
        heterogeneity_tau=0.10,
        i_squared=62.0,
        replication_status="replicated",
        moderators={
            "behavior_type": {"private": 0.30, "public": 0.35, "high_cost": 0.12},
            "culture": {"western": 0.25, "collectivist": 0.32},
        },
        notes="Environmental attitudes weakly predict behavior. Gap largest for high-cost behaviors."
    ),

    "plastic_reduction_meta": MetaAnalyticEffect(
        source="Xanthos & Walker (2017); Schnurr et al. (2018)",
        effect_d=0.42,
        ci_95=(0.28, 0.56),
        n_studies=22,
        n_participants=12000,
        domain="environmental_psychology",
        construct="plastic_use_reduction",
        paradigm="anti_plastic_intervention",
        heterogeneity_tau=0.12,
        i_squared=64.0,
        replication_status="replicated",
        moderators={
            "intervention": {"bag_charge": 0.60, "education": 0.25, "social_norms": 0.35},
        },
        notes="Bag charges most effective (70-90% reduction). Education alone smaller effects."
    ),

    "energy_conservation_meta": MetaAnalyticEffect(
        source="Abrahamse et al. (2005); Delmas et al. (2013)",
        effect_d=0.30,
        ci_95=(0.18, 0.42),
        n_studies=58,
        n_participants=25000,
        domain="environmental_psychology",
        construct="energy_use_reduction",
        paradigm="conservation_intervention",
        heterogeneity_tau=0.12,
        i_squared=68.0,
        replication_status="replicated",
        moderators={
            "strategy": {"feedback": 0.35, "goal_setting": 0.32, "commitment": 0.28, "information_only": 0.12},
            "energy_type": {"electricity": 0.32, "gas": 0.25, "transport": 0.20},
        },
        notes="Real-time feedback most effective. Information alone insufficient. Average savings 5-15%."
    ),

    # ── INTERGROUP RELATIONS & PREJUDICE ─────────────────────────────────

    "intergroup_contact_extended": MetaAnalyticEffect(
        source="Pettigrew & Tropp (2006); Lemmer & Wagner (2015)",
        effect_d=0.42,
        ci_95=(0.36, 0.48),
        n_studies=515,
        n_participants=250000,
        domain="social_psychology",
        construct="prejudice_reduction",
        paradigm="intergroup_contact",
        heterogeneity_tau=0.15,
        i_squared=76.0,
        replication_status="replicated",
        moderators={
            "contact_conditions": {"allport_conditions_met": 0.55, "not_met": 0.32},
            "target_outgroup": {"racial_ethnic": 0.42, "lgbtq": 0.48, "disability": 0.50, "mental_illness": 0.40},
            "design": {"experimental": 0.52, "cross_sectional": 0.38},
        },
        notes="Contact reduces prejudice across groups. Effect generalizes beyond immediate contact partner."
    ),

    "implicit_bias_training_meta": MetaAnalyticEffect(
        source="Forscher et al. (2019); Lai et al. (2016)",
        effect_d=0.12,
        ci_95=(0.02, 0.22),
        n_studies=492,
        n_participants=87000,
        domain="social_psychology",
        construct="implicit_bias_change",
        paradigm="implicit_bias_intervention",
        heterogeneity_tau=0.08,
        i_squared=55.0,
        replication_status="contested",
        moderators={
            "intervention": {"vivid_counterstereotype": 0.20, "perspective_taking": 0.15, "awareness_raising": 0.05},
            "durability": {"immediate": 0.18, "delayed_24h": 0.05},
        },
        notes="Small immediate effects, near-zero durability. No evidence of behavior change."
    ),

    "perspective_taking_meta": MetaAnalyticEffect(
        source="Teding van Berkhout & Malouff (2016); Todd et al. (2011)",
        effect_d=0.33,
        ci_95=(0.22, 0.44),
        n_studies=51,
        n_participants=7500,
        domain="social_psychology",
        construct="empathy_prejudice_reduction",
        paradigm="perspective_taking",
        heterogeneity_tau=0.12,
        i_squared=66.0,
        replication_status="replicated",
        moderators={
            "target": {"racial_outgroup": 0.35, "elderly": 0.30, "disabled": 0.38},
            "method": {"essay_writing": 0.38, "vr_embodiment": 0.42, "instruction_only": 0.22},
        },
        notes="Perspective-taking increases empathy and reduces prejudice. VR embodiment promising."
    ),

    "common_ingroup_identity_meta": MetaAnalyticEffect(
        source="Gaertner & Dovidio (2000); Gonzalez & Brown (2006)",
        effect_d=0.38,
        ci_95=(0.24, 0.52),
        n_studies=32,
        n_participants=5000,
        domain="social_psychology",
        construct="intergroup_attitudes",
        paradigm="recategorization",
        heterogeneity_tau=0.12,
        i_squared=66.0,
        replication_status="replicated",
        moderators={
            "strategy": {"one_group": 0.42, "dual_identity": 0.45, "decategorization": 0.28},
        },
        notes="Recategorizing outgroup as ingroup reduces bias. Dual identity may be optimal."
    ),

    "cross_group_friendship_meta": MetaAnalyticEffect(
        source="Davies et al. (2011); Turner et al. (2007)",
        effect_d=0.45,
        ci_95=(0.32, 0.58),
        n_studies=60,
        n_participants=15000,
        domain="social_psychology",
        construct="prejudice_reduction",
        paradigm="cross_group_friendship",
        heterogeneity_tau=0.14,
        i_squared=72.0,
        replication_status="replicated",
        moderators={
            "friendship_quality": {"close": 0.55, "acquaintance": 0.28},
        },
        notes="Close cross-group friendships reduce prejudice more than casual contact."
    ),

    "diversity_exposure_meta": MetaAnalyticEffect(
        source="Denson (2009); Bowman (2010)",
        effect_d=0.22,
        ci_95=(0.12, 0.32),
        n_studies=42,
        n_participants=25000,
        domain="social_psychology",
        construct="openness_attitudes",
        paradigm="diversity_exposure",
        heterogeneity_tau=0.10,
        i_squared=62.0,
        replication_status="replicated",
        moderators={
            "exposure_type": {"structural_diversity": 0.18, "informal_interaction": 0.28, "curricular": 0.22},
        },
        notes="Campus diversity exposure improves intergroup attitudes. Informal interaction most effective."
    ),

    "prejudice_reduction_meta": MetaAnalyticEffect(
        source="Paluck & Green (2009); Paluck et al. (2021)",
        effect_d=0.22,
        ci_95=(0.12, 0.32),
        n_studies=418,
        n_participants=60000,
        domain="social_psychology",
        construct="prejudice_behavior_change",
        paradigm="prejudice_reduction",
        heterogeneity_tau=0.12,
        i_squared=68.0,
        replication_status="replicated",
        moderators={
            "approach": {"contact": 0.42, "education": 0.18, "media": 0.15, "diversity_training": 0.10},
            "outcome": {"attitudes": 0.28, "behavior": 0.15},
        },
        notes="Contact-based approaches most effective. Few interventions show durable behavioral change."
    ),

    "empathy_training_meta": MetaAnalyticEffect(
        source="Teding van Berkhout & Malouff (2016); Weisz & Zaki (2018)",
        effect_d=0.48,
        ci_95=(0.35, 0.61),
        n_studies=18,
        n_participants=3500,
        domain="social_psychology",
        construct="empathic_responding",
        paradigm="empathy_training",
        heterogeneity_tau=0.12,
        i_squared=65.0,
        replication_status="replicated",
        moderators={
            "population": {"healthcare": 0.52, "students": 0.45, "general": 0.38},
        },
        notes="Empathy trainable. Larger effects for healthcare professionals."
    ),

    "media_representation_meta": MetaAnalyticEffect(
        source="Mastro & Tukachinsky (2011); Ramasubramanian (2015)",
        effect_d=0.28,
        ci_95=(0.15, 0.41),
        n_studies=28,
        n_participants=6000,
        domain="social_psychology",
        construct="intergroup_attitudes",
        paradigm="media_exposure",
        heterogeneity_tau=0.10,
        i_squared=60.0,
        replication_status="replicated",
        moderators={
            "portrayal": {"positive_counterstereotypic": 0.35, "negative_stereotypic": -0.30},
            "medium": {"television": 0.25, "film": 0.30, "news": 0.22},
        },
        notes="Positive media portrayal improves attitudes. Stereotypic portrayal reinforces prejudice."
    ),

    "multicultural_education_meta": MetaAnalyticEffect(
        source="Beelmann & Heinemann (2014); Aboud et al. (2012)",
        effect_d=0.30,
        ci_95=(0.18, 0.42),
        n_studies=122,
        n_participants=15000,
        domain="social_psychology",
        construct="intergroup_attitudes",
        paradigm="multicultural_education",
        heterogeneity_tau=0.12,
        i_squared=68.0,
        replication_status="replicated",
        moderators={
            "age": {"children_5_10": 0.38, "adolescents": 0.28, "adults": 0.22},
            "program_type": {"direct_contact": 0.40, "indirect_media": 0.25, "cognitive_training": 0.30},
        },
        notes="More effective for younger children. Direct contact programs outperform indirect."
    ),

    # ── PERSUASION & COMMUNICATION ───────────────────────────────────────

    "elaboration_likelihood_meta": MetaAnalyticEffect(
        source="Carpenter (2015); Petty & Cacioppo (1986)",
        effect_d=0.44,
        ci_95=(0.32, 0.56),
        n_studies=120,
        n_participants=20000,
        domain="communication",
        construct="attitude_change",
        paradigm="elaboration_likelihood",
        heterogeneity_tau=0.14,
        i_squared=72.0,
        replication_status="replicated",
        moderators={
            "route": {"central_strong_args": 0.55, "peripheral_cues": 0.30},
            "motivation": {"high_involvement": 0.58, "low_involvement": 0.28},
            "argument_quality": {"strong": 0.55, "weak": 0.15},
        },
        notes="Central route persuasion produces more durable attitude change. Involvement moderates route."
    ),

    "narrative_persuasion_expanded": MetaAnalyticEffect(
        source="Braddock & Dillard (2016); de Graaf et al. (2012)",
        effect_d=0.48,
        ci_95=(0.36, 0.60),
        n_studies=74,
        n_participants=14000,
        domain="communication",
        construct="narrative_persuasion",
        paradigm="narrative_vs_statistical",
        heterogeneity_tau=0.14,
        i_squared=72.0,
        replication_status="replicated",
        moderators={
            "outcome": {"attitude": 0.50, "intention": 0.42, "behavior": 0.30},
            "identification": {"high": 0.58, "low": 0.30},
            "vs_statistical": {"narrative_advantage": 0.15},
        },
        notes="Narratives persuade through transportation and identification."
    ),

    "humor_persuasion_meta": MetaAnalyticEffect(
        source="Eisend (2009); Walter et al. (2018)",
        effect_d=0.28,
        ci_95=(0.18, 0.38),
        n_studies=56,
        n_participants=12000,
        domain="communication",
        construct="attitude_toward_ad",
        paradigm="humor_in_advertising",
        heterogeneity_tau=0.10,
        i_squared=62.0,
        replication_status="replicated",
        moderators={
            "humor_relatedness": {"related_to_product": 0.35, "unrelated": 0.15},
            "outcome": {"attention": 0.42, "liking": 0.35, "recall": 0.10, "persuasion": 0.22},
        },
        notes="Humor increases attention and liking but may reduce message recall."
    ),

    "emotional_appeal_meta": MetaAnalyticEffect(
        source="Hornik et al. (2016); Nabi (2002)",
        effect_d=0.40,
        ci_95=(0.28, 0.52),
        n_studies=62,
        n_participants=15000,
        domain="communication",
        construct="persuasion",
        paradigm="emotional_appeal",
        heterogeneity_tau=0.14,
        i_squared=72.0,
        replication_status="replicated",
        moderators={
            "emotion": {"fear": 0.45, "guilt": 0.38, "hope": 0.42, "anger": 0.35, "sadness": 0.30},
            "topic": {"health": 0.45, "environment": 0.35, "social_cause": 0.40},
        },
        notes="Discrete emotions differentially effective. Fear and hope most persuasive."
    ),

    "two_sided_message_meta": MetaAnalyticEffect(
        source="Allen (1991); Eisend (2006)",
        effect_d=0.20,
        ci_95=(0.10, 0.30),
        n_studies=60,
        n_participants=10000,
        domain="communication",
        construct="credibility_persuasion",
        paradigm="two_sided_message",
        heterogeneity_tau=0.08,
        i_squared=55.0,
        replication_status="replicated",
        moderators={
            "refutation": {"refutational": 0.28, "non_refutational": 0.08},
            "audience_awareness": {"aware_of_counterargs": 0.30, "unaware": 0.12},
        },
        notes="Two-sided messages more persuasive when they refute counterarguments."
    ),

    "inoculation_expanded_meta": MetaAnalyticEffect(
        source="Banas & Rains (2010); Compton (2013)",
        effect_d=0.30,
        ci_95=(0.20, 0.40),
        n_studies=54,
        n_participants=12000,
        domain="communication",
        construct="resistance_to_persuasion",
        paradigm="inoculation_theory",
        heterogeneity_tau=0.10,
        i_squared=62.0,
        replication_status="replicated",
        moderators={
            "inoculation_type": {"active": 0.35, "passive": 0.25},
            "delay": {"immediate": 0.32, "1_week": 0.28, "2_weeks": 0.20},
        },
        notes="Inoculation produces resistance that persists but decays. Active generation stronger."
    ),

    "debunking_misinformation_meta": MetaAnalyticEffect(
        source="Chan et al. (2017); Walter & Murphy (2018)",
        effect_d=0.35,
        ci_95=(0.22, 0.48),
        n_studies=52,
        n_participants=10000,
        domain="communication",
        construct="belief_correction",
        paradigm="debunking",
        heterogeneity_tau=0.14,
        i_squared=72.0,
        replication_status="replicated",
        moderators={
            "strategy": {"detailed_alternative": 0.45, "simple_negation": 0.15, "source_discrediting": 0.32},
            "worldview": {"worldview_consistent": 0.42, "worldview_threatening": 0.18},
        },
        notes="Alternative explanation more effective than simple negation. Worldview-consistent easier."
    ),

    "gain_loss_framing_health_meta": MetaAnalyticEffect(
        source="Gallagher & Updegraff (2012); O'Keefe & Jensen (2009)",
        effect_d=0.15,
        ci_95=(0.05, 0.25),
        n_studies=94,
        n_participants=25000,
        domain="communication",
        construct="health_behavior",
        paradigm="gain_loss_framing",
        heterogeneity_tau=0.10,
        i_squared=62.0,
        replication_status="replicated",
        moderators={
            "behavior_type": {"prevention_detection": 0.08, "prevention_only": 0.18, "detection_only": -0.05},
            "frame": {"gain_advantage_prevention": 0.18, "loss_advantage_detection": 0.10},
        },
        notes="Small effect. Gain frames slightly better for prevention, loss for detection."
    ),

    "metaphor_persuasion_meta": MetaAnalyticEffect(
        source="Sopory & Dillard (2002); van Stee (2018)",
        effect_d=0.27,
        ci_95=(0.16, 0.38),
        n_studies=35,
        n_participants=6000,
        domain="communication",
        construct="persuasion",
        paradigm="metaphor_in_communication",
        heterogeneity_tau=0.08,
        i_squared=58.0,
        replication_status="replicated",
        moderators={
            "novelty": {"novel": 0.35, "conventional": 0.18},
            "topic_familiarity": {"unfamiliar": 0.32, "familiar": 0.18},
        },
        notes="Novel metaphors more persuasive. Metaphors help most when topic is unfamiliar."
    ),

    "misinformation_correction_meta": MetaAnalyticEffect(
        source="Walter & Tukachinsky (2020); Lewandowsky et al. (2012)",
        effect_d=0.30,
        ci_95=(0.20, 0.40),
        n_studies=65,
        n_participants=15000,
        domain="communication",
        construct="continued_influence_reduction",
        paradigm="correction",
        heterogeneity_tau=0.12,
        i_squared=68.0,
        replication_status="replicated",
        moderators={
            "repetition": {"single": 0.25, "repeated": 0.38},
            "medium": {"text": 0.28, "video": 0.35, "infographic": 0.32},
        },
        notes="Corrections reduce but don't eliminate continued influence. Repetition helps."
    ),
}


# =============================================================================
# GAME CALIBRATIONS DATABASE
# =============================================================================

GAME_CALIBRATIONS: Dict[str, GameCalibration] = {

    "dictator_standard": GameCalibration(
        source="Engel (2011)",
        game_type="dictator",
        variant="standard",
        mean_proportion=0.28,
        sd_proportion=0.18,
        ci_95=(0.25, 0.31),
        distribution_shape="bimodal",
        modes=[0.0, 0.50],
        n_studies=616,
        n_participants=20813,
        subpopulations={
            "pure_selfish_zero": 0.36,  # Give nothing
            "fair_split_50": 0.17,       # Give exactly half
            "low_giver_1_20": 0.20,      # Give 1-20%
            "moderate_giver_21_49": 0.16, # Give 21-49%
            "generous_51_plus": 0.11,     # Give >50%
        },
        moderators={
            "stakes": {"small": 0.32, "medium": 0.28, "large": 0.22},
            "anonymity": {"double_blind": 0.22, "single_blind": 0.30},
            "gender": {"male": 0.25, "female": 0.32},
        },
    ),

    "dictator_taking": GameCalibration(
        source="List (2007); Bardsley (2008)",
        game_type="dictator",
        variant="taking",
        mean_proportion=0.12,
        sd_proportion=0.25,
        ci_95=(0.08, 0.16),
        distribution_shape="trimodal",
        modes=[-0.20, 0.0, 0.50],
        n_studies=12,
        n_participants=1847,
        subpopulations={
            "taker_negative": 0.20,      # Take from other
            "pure_selfish_zero": 0.25,    # Keep everything
            "fair_divider": 0.35,         # Give around 50%
            "moderate_giver": 0.20,       # Give 10-40%
        },
    ),

    "dictator_third_party_punishment": GameCalibration(
        source="Fehr & Fischbacher (2004)",
        game_type="dictator",
        variant="third_party_punishment",
        mean_proportion=0.25,
        sd_proportion=0.20,
        ci_95=(0.20, 0.30),
        distribution_shape="bimodal",
        modes=[0.0, 0.50],
        n_studies=8,
        n_participants=800,
        subpopulations={
            "no_punishment": 0.40,
            "moderate_punishment": 0.35,
            "strong_punishment": 0.25,
        },
        notes="~60% of observers punish unfair allocations."
    ),

    "trust_standard": GameCalibration(
        source="Berg et al. (1995); Johnson & Mislin (2011)",
        game_type="trust",
        variant="standard",
        mean_proportion=0.50,
        sd_proportion=0.15,
        ci_95=(0.47, 0.53),
        distribution_shape="normal",
        modes=[0.50],
        n_studies=162,
        n_participants=23000,
        moderators={
            "multiplier": {"2x": 0.45, "3x": 0.50, "4x": 0.55},
        },
    ),

    "ultimatum_standard": GameCalibration(
        source="Oosterbeek et al. (2004)",
        game_type="ultimatum",
        variant="standard",
        mean_proportion=0.42,
        sd_proportion=0.10,
        ci_95=(0.40, 0.44),
        distribution_shape="left_skew",
        modes=[0.50],
        n_studies=37,
        n_participants=7500,
        notes="Modal offer 40-50%. Below 20% rejected ~50%."
    ),

    "public_goods_standard": GameCalibration(
        source="Zelmer (2003)",
        game_type="public_goods",
        variant="standard",
        mean_proportion=0.47,
        sd_proportion=0.18,
        ci_95=(0.42, 0.52),
        distribution_shape="right_skew",
        modes=[0.50],
        n_studies=27,
        n_participants=5400,
        moderators={
            "mpcr": {"0.3": 0.35, "0.5": 0.47, "0.75": 0.60},
            "punishment": {"no": 0.47, "peer": 0.75},
        },
    ),

    "public_goods_with_punishment": GameCalibration(
        source="Fehr & Gächter (2000)",
        game_type="public_goods",
        variant="punishment",
        mean_proportion=0.80,
        sd_proportion=0.12,
        ci_95=(0.72, 0.88),
        distribution_shape="left_skew",
        modes=[0.90],
        n_studies=15,
        n_participants=1500,
        notes="Punishment increases contributions from ~47% to ~80%."
    ),

    "prisoners_dilemma_standard": GameCalibration(
        source="Sally (1995)",
        game_type="prisoners_dilemma",
        variant="standard",
        mean_proportion=0.47,
        sd_proportion=0.20,
        ci_95=(0.43, 0.51),
        distribution_shape="bimodal",
        modes=[0.0, 1.0],
        n_studies=130,
        n_participants=26000,
    ),

    # ── NEW GAME CALIBRATIONS (v1.0.8.7) ──

    "first_price_auction_standard": GameCalibration(
        source="Kagel & Levin (2015)",
        game_type="first_price_auction",
        variant="standard",
        mean_proportion=0.82,
        sd_proportion=0.12,
        ci_95=(0.78, 0.86),
        distribution_shape="left_skew",
        modes=[0.85],
        n_studies=45,
        n_participants=3200,
        notes="Bid/value ratio ≈ 0.82. Persistent overbidding vs RNNE."
    ),

    "second_price_auction_standard": GameCalibration(
        source="Kagel & Levin (1993)",
        game_type="second_price_auction",
        variant="standard",
        mean_proportion=1.05,
        sd_proportion=0.20,
        ci_95=(0.95, 1.15),
        distribution_shape="right_skew",
        modes=[1.0],
        n_studies=15,
        n_participants=1200,
        notes="Overbidding despite dominant strategy to bid true value."
    ),

    "all_pay_auction_standard": GameCalibration(
        source="Dechenaux et al. (2015)",
        game_type="all_pay_auction",
        variant="standard",
        mean_proportion=0.50,
        sd_proportion=0.30,
        ci_95=(0.40, 0.60),
        distribution_shape="bimodal",
        modes=[0.0, 0.80],
        n_studies=20,
        n_participants=2000,
        notes="Bimodal: dropout (bid 0) vs overbidding. Aggregate near Nash."
    ),

    "nash_bargaining_standard": GameCalibration(
        source="Roth (1995)",
        game_type="nash_bargaining",
        variant="standard",
        mean_proportion=0.50,
        sd_proportion=0.12,
        ci_95=(0.46, 0.54),
        distribution_shape="normal",
        modes=[0.50],
        n_studies=30,
        n_participants=3000,
    ),

    "gift_exchange_standard": GameCalibration(
        source="Fehr et al. (1993)",
        game_type="gift_exchange",
        variant="standard",
        mean_proportion=0.45,
        sd_proportion=0.18,
        ci_95=(0.40, 0.50),
        distribution_shape="normal",
        modes=[0.45],
        n_studies=25,
        n_participants=2500,
        notes="Effort ∝ wage. Reciprocity in labor markets."
    ),

    "stag_hunt_standard": GameCalibration(
        source="Battalio et al. (2001)",
        game_type="stag_hunt",
        variant="standard",
        mean_proportion=0.60,
        sd_proportion=0.25,
        ci_95=(0.50, 0.70),
        distribution_shape="bimodal",
        modes=[0.0, 1.0],
        n_studies=12,
        n_participants=960,
        notes="Coordination game. Risk vs payoff dominance."
    ),

    "common_pool_resource_standard": GameCalibration(
        source="Ostrom et al. (1994)",
        game_type="common_pool_resource",
        variant="standard",
        mean_proportion=0.65,
        sd_proportion=0.18,
        ci_95=(0.58, 0.72),
        distribution_shape="right_skew",
        modes=[0.70],
        n_studies=18,
        n_participants=1800,
        notes="Extraction ≈ 65% of endowment. Overextraction relative to optimal."
    ),

    "holt_laury_standard": GameCalibration(
        source="Holt & Laury (2002)",
        game_type="holt_laury",
        variant="standard",
        mean_proportion=0.58,
        sd_proportion=0.12,
        ci_95=(0.55, 0.61),
        distribution_shape="normal",
        modes=[0.58],
        n_studies=50,
        n_participants=8000,
        notes="Mean safe choices 5.8/10 → risk averse."
    ),

    "beauty_contest_standard": GameCalibration(
        source="Nagel (1995)",
        game_type="beauty_contest",
        variant="standard",
        mean_proportion=0.33,
        sd_proportion=0.15,
        ci_95=(0.28, 0.38),
        distribution_shape="right_skew",
        modes=[0.33],
        n_studies=20,
        n_participants=4000,
        notes="Mean guess ≈ 33 (of 0-100 with target 2/3 of average)."
    ),

    "die_roll_honesty": GameCalibration(
        source="Abeler et al. (2019)",
        game_type="die_roll",
        variant="standard",
        mean_proportion=0.58,
        sd_proportion=0.20,
        ci_95=(0.54, 0.62),
        distribution_shape="right_skew",
        modes=[0.83],
        n_studies=90,
        n_participants=44000,
        subpopulations={
            "honest_reporter": 0.55,
            "partial_liar": 0.25,
            "full_liar": 0.20,
        },
        notes="Mean reported ≈ 58% of maximum (vs 50% expected). Not full lying."
    ),

    # ── DICTATOR GAME VARIANTS (v1.0.9.4) ─────────────────────────────────

    "dictator_earned_money": GameCalibration(
        source="Cherry et al. (2002); Oxoby & Spraggon (2008)",
        game_type="dictator",
        variant="earned_money",
        mean_proportion=0.15,
        sd_proportion=0.18,
        ci_95=(0.11, 0.19),
        distribution_shape="right_skew",
        modes=[0.0],
        n_studies=18,
        n_participants=1600,
        subpopulations={
            "pure_selfish_zero": 0.55,
            "low_giver_1_20": 0.22,
            "moderate_giver_21_49": 0.13,
            "fair_split_50": 0.07,
            "generous_51_plus": 0.03,
        },
        moderators={
            "earner": {"both_earned": 0.10, "dictator_earned": 0.12, "receiver_earned": 0.22},
        },
        notes="Earning reduces giving by ~45% vs windfall. Property rights effect."
    ),

    "dictator_deserving_receiver": GameCalibration(
        source="Ruffle (1998); Eckel & Grossman (1996)",
        game_type="dictator",
        variant="deserving_receiver",
        mean_proportion=0.38,
        sd_proportion=0.20,
        ci_95=(0.33, 0.43),
        distribution_shape="bimodal",
        modes=[0.0, 0.50],
        n_studies=14,
        n_participants=1200,
        subpopulations={
            "pure_selfish_zero": 0.18,
            "low_giver_1_20": 0.12,
            "moderate_giver_21_49": 0.22,
            "fair_split_50": 0.30,
            "generous_51_plus": 0.18,
        },
        moderators={
            "charity_type": {"red_cross": 0.42, "unknown_charity": 0.35, "individual_need": 0.38},
        },
        notes="Deserving receivers increase giving ~35% above standard DG."
    ),

    "dictator_ingroup": GameCalibration(
        source="Whitt & Wilson (2007); Bernhard et al. (2006)",
        game_type="dictator",
        variant="ingroup_receiver",
        mean_proportion=0.35,
        sd_proportion=0.18,
        ci_95=(0.31, 0.39),
        distribution_shape="bimodal",
        modes=[0.0, 0.50],
        n_studies=22,
        n_participants=2800,
        subpopulations={
            "pure_selfish_zero": 0.22,
            "low_giver_1_20": 0.14,
            "moderate_giver_21_49": 0.22,
            "fair_split_50": 0.28,
            "generous_51_plus": 0.14,
        },
        moderators={
            "group_type": {"ethnic": 0.37, "political": 0.38, "minimal": 0.32, "national": 0.36},
            "conflict_salience": {"low": 0.33, "high": 0.40},
        },
        notes="Ingroup favoritism: ~25% more giving than to strangers. Stronger with real groups."
    ),

    "dictator_outgroup": GameCalibration(
        source="Iyengar & Westwood (2015); Dimant (2024); Fershtman & Gneezy (2001)",
        game_type="dictator",
        variant="outgroup_receiver",
        mean_proportion=0.20,
        sd_proportion=0.18,
        ci_95=(0.16, 0.24),
        distribution_shape="right_skew",
        modes=[0.0],
        n_studies=28,
        n_participants=3400,
        subpopulations={
            "pure_selfish_zero": 0.42,
            "low_giver_1_20": 0.24,
            "moderate_giver_21_49": 0.18,
            "fair_split_50": 0.12,
            "generous_51_plus": 0.04,
        },
        moderators={
            "group_type": {"political": 0.17, "ethnic": 0.19, "racial": 0.18, "minimal": 0.24},
            "polarization_era": {"pre_2016": 0.22, "post_2016": 0.16},
        },
        notes="Outgroup discrimination d ≈ 0.6-0.9 (Dimant 2024). Political strongest."
    ),

    "dictator_social_distance": GameCalibration(
        source="Hoffman et al. (1996); Bohnet & Frey (1999)",
        game_type="dictator",
        variant="social_distance",
        mean_proportion=0.33,
        sd_proportion=0.20,
        ci_95=(0.28, 0.38),
        distribution_shape="bimodal",
        modes=[0.0, 0.50],
        n_studies=16,
        n_participants=1800,
        moderators={
            "distance_level": {"face_to_face": 0.45, "one_way_mirror": 0.35,
                               "double_blind": 0.22, "internet": 0.18},
        },
        notes="Social distance inversely predicts giving. Face-to-face: ~45% vs online: ~18%."
    ),

    "dictator_charity": GameCalibration(
        source="Eckel & Grossman (1996); Engel (2011) subset",
        game_type="dictator",
        variant="charity_option",
        mean_proportion=0.42,
        sd_proportion=0.22,
        ci_95=(0.37, 0.47),
        distribution_shape="bimodal",
        modes=[0.0, 0.50],
        n_studies=20,
        n_participants=2200,
        subpopulations={
            "pure_selfish_zero": 0.15,
            "low_giver_1_20": 0.12,
            "moderate_giver_21_49": 0.18,
            "fair_split_50": 0.25,
            "generous_51_plus": 0.30,
        },
        moderators={
            "charity_salience": {"named_charity": 0.46, "generic_charity": 0.38},
        },
        notes="Charity as receiver increases mean giving ~50% above standard DG."
    ),

    "dictator_multiple_recipients": GameCalibration(
        source="Andreoni & Bernheim (2009); Bolton & Ockenfels (2000)",
        game_type="dictator",
        variant="multiple_recipients",
        mean_proportion=0.32,
        sd_proportion=0.15,
        ci_95=(0.27, 0.37),
        distribution_shape="normal",
        modes=[0.30],
        n_studies=10,
        n_participants=900,
        moderators={
            "n_recipients": {"2": 0.34, "3": 0.31, "5": 0.28, "10": 0.25},
        },
        notes="Per-recipient giving decreases with more recipients. Total giving increases slightly."
    ),

    "dictator_uncertainty": GameCalibration(
        source="Dana et al. (2007); Larson & Capra (2009)",
        game_type="dictator",
        variant="uncertainty",
        mean_proportion=0.18,
        sd_proportion=0.20,
        ci_95=(0.13, 0.23),
        distribution_shape="right_skew",
        modes=[0.0],
        n_studies=12,
        n_participants=1100,
        subpopulations={
            "pure_selfish_zero": 0.48,
            "low_giver_1_20": 0.22,
            "moderate_giver_21_49": 0.16,
            "fair_split_50": 0.10,
            "generous_51_plus": 0.04,
        },
        notes="Moral wiggle room: uncertainty about receiver outcomes reduces giving ~35%."
    ),

    # ── TRUST GAME VARIANTS (v1.0.9.4) ────────────────────────────────────

    "trust_with_communication": GameCalibration(
        source="Ben-Ner & Putterman (2009); Charness & Dufwenberg (2006)",
        game_type="trust",
        variant="communication",
        mean_proportion=0.62,
        sd_proportion=0.18,
        ci_95=(0.57, 0.67),
        distribution_shape="normal",
        modes=[0.60],
        n_studies=20,
        n_participants=2400,
        moderators={
            "communication_type": {"free_form": 0.65, "structured_promise": 0.60,
                                   "cheap_talk": 0.55},
        },
        notes="Communication increases trust ~24%. Promises especially effective (guilt aversion)."
    ),

    "trust_with_reputation": GameCalibration(
        source="Bohnet & Huck (2004); Bolton et al. (2004)",
        game_type="trust",
        variant="reputation",
        mean_proportion=0.60,
        sd_proportion=0.16,
        ci_95=(0.55, 0.65),
        distribution_shape="normal",
        modes=[0.60],
        n_studies=15,
        n_participants=1800,
        moderators={
            "reputation_info": {"full_history": 0.65, "summary_score": 0.58,
                                "single_encounter": 0.50},
        },
        notes="Reputation information increases trust ~20%. History more effective than summary."
    ),

    "trust_with_punishment": GameCalibration(
        source="Fehr & Rockenbach (2003); Houser et al. (2008)",
        game_type="trust",
        variant="punishment",
        mean_proportion=0.45,
        sd_proportion=0.18,
        ci_95=(0.40, 0.50),
        distribution_shape="bimodal",
        modes=[0.20, 0.60],
        n_studies=12,
        n_participants=1200,
        moderators={
            "punishment_frame": {"sanction_available": 0.42, "sanction_imposed": 0.35,
                                 "no_sanction": 0.55},
        },
        notes="Punishment REDUCES trust (crowding out). Imposed sanctions signal distrust."
    ),

    "trust_cross_cultural": GameCalibration(
        source="Johnson & Mislin (2011); Bohnet et al. (2008)",
        game_type="trust",
        variant="cross_cultural",
        mean_proportion=0.44,
        sd_proportion=0.18,
        ci_95=(0.39, 0.49),
        distribution_shape="normal",
        modes=[0.45],
        n_studies=35,
        n_participants=5200,
        moderators={
            "culture_pair": {"western_western": 0.52, "western_east_asian": 0.42,
                             "east_asian_east_asian": 0.40, "arab_western": 0.38,
                             "within_developing": 0.45},
        },
        notes="Cross-cultural trust ~15% lower than within-culture. Strongest deficit with high cultural distance."
    ),

    "trust_binary": GameCalibration(
        source="Ermisch et al. (2009); Glaeser et al. (2000)",
        game_type="trust",
        variant="binary",
        mean_proportion=0.55,
        sd_proportion=0.20,
        ci_95=(0.50, 0.60),
        distribution_shape="bimodal",
        modes=[0.0, 1.0],
        n_studies=18,
        n_participants=2600,
        subpopulations={
            "trusters": 0.55,
            "non_trusters": 0.45,
        },
        notes="Binary trust decision: ~55% choose to trust. Simpler than continuous."
    ),

    "trust_with_risk_info": GameCalibration(
        source="Eckel & Wilson (2004); Houser et al. (2010)",
        game_type="trust",
        variant="risk_information",
        mean_proportion=0.48,
        sd_proportion=0.17,
        ci_95=(0.43, 0.53),
        distribution_shape="normal",
        modes=[0.50],
        n_studies=14,
        n_participants=1500,
        moderators={
            "risk_frame": {"probability_displayed": 0.45, "no_risk_info": 0.50,
                           "return_rate_shown": 0.52},
        },
        notes="Explicit risk info slightly reduces trust. Trust ≠ risk preferences (Eckel & Wilson)."
    ),

    "trust_repeated": GameCalibration(
        source="Engle-Warnick & Slonim (2004); King-Casas et al. (2005)",
        game_type="trust",
        variant="repeated",
        mean_proportion=0.55,
        sd_proportion=0.15,
        ci_95=(0.50, 0.60),
        distribution_shape="normal",
        modes=[0.55],
        n_studies=16,
        n_participants=1400,
        moderators={
            "round": {"early_1_3": 0.48, "middle_4_7": 0.55, "late_8_10": 0.60},
            "reciprocity_experienced": {"high_return": 0.65, "low_return": 0.35},
        },
        notes="Trust grows with repeated interaction. Reciprocity builds trust over rounds."
    ),

    "trust_with_inequality": GameCalibration(
        source="Anderson et al. (2006); Xiao & Bicchieri (2010)",
        game_type="trust",
        variant="inequality",
        mean_proportion=0.42,
        sd_proportion=0.20,
        ci_95=(0.36, 0.48),
        distribution_shape="bimodal",
        modes=[0.20, 0.60],
        n_studies=10,
        n_participants=1100,
        moderators={
            "inequality_direction": {"trustor_richer": 0.48, "trustor_poorer": 0.35,
                                     "equal_endowment": 0.50},
        },
        notes="Inequality reduces trust, especially when trustor is disadvantaged."
    ),

    # ── ULTIMATUM GAME VARIANTS (v1.0.9.4) ────────────────────────────────

    "ultimatum_alternative_offers": GameCalibration(
        source="Fischbacher et al. (2009); Bolton & Zwick (1995)",
        game_type="ultimatum",
        variant="alternative_offers",
        mean_proportion=0.38,
        sd_proportion=0.12,
        ci_95=(0.34, 0.42),
        distribution_shape="left_skew",
        modes=[0.40],
        n_studies=10,
        n_participants=1000,
        moderators={
            "alternative_size": {"attractive_alt": 0.35, "unattractive_alt": 0.42},
        },
        notes="Outside options reduce offers. Proposers extract more with weak responder alternatives."
    ),

    "ultimatum_costly_rejection": GameCalibration(
        source="Yamagishi et al. (2012); Henrich et al. (2006)",
        game_type="ultimatum",
        variant="costly_rejection",
        mean_proportion=0.40,
        sd_proportion=0.12,
        ci_95=(0.36, 0.44),
        distribution_shape="left_skew",
        modes=[0.40],
        n_studies=14,
        n_participants=1600,
        subpopulations={
            "fair_offer_40_50": 0.50,
            "moderate_offer_25_39": 0.30,
            "low_offer_below_25": 0.15,
            "hyper_fair_above_50": 0.05,
        },
        moderators={
            "rejection_cost": {"free_rejection": 0.42, "costly_rejection": 0.38,
                               "very_costly": 0.34},
        },
        notes="Costly rejection reduces rejection rates but barely changes offers."
    ),

    "ultimatum_third_party": GameCalibration(
        source="Güth & van Damme (1998); Bereby-Meyer & Niederle (2005)",
        game_type="ultimatum",
        variant="third_party_allocation",
        mean_proportion=0.35,
        sd_proportion=0.14,
        ci_95=(0.30, 0.40),
        distribution_shape="normal",
        modes=[0.35],
        n_studies=8,
        n_participants=800,
        notes="Third-party allocators offer less than self-interested proposers. Strategic fairness reduced."
    ),

    "mini_ultimatum": GameCalibration(
        source="Falk et al. (2003); Güth et al. (2001)",
        game_type="ultimatum",
        variant="mini",
        mean_proportion=0.40,
        sd_proportion=0.10,
        ci_95=(0.36, 0.44),
        distribution_shape="bimodal",
        modes=[0.20, 0.50],
        n_studies=12,
        n_participants=1400,
        moderators={
            "choice_set": {"fair_vs_hyper_fair": 0.50, "fair_vs_unfair": 0.45,
                           "unfair_vs_very_unfair": 0.30},
        },
        notes="Restricted choice set reveals intention-based fairness preferences (Falk et al. 2003)."
    ),

    "ultimatum_with_delay": GameCalibration(
        source="Grimm & Mengel (2011); Neo et al. (2013)",
        game_type="ultimatum",
        variant="delay",
        mean_proportion=0.38,
        sd_proportion=0.14,
        ci_95=(0.33, 0.43),
        distribution_shape="normal",
        modes=[0.40],
        n_studies=8,
        n_participants=700,
        moderators={
            "delay_length": {"immediate": 0.42, "10_minutes": 0.38, "24_hours": 0.35},
        },
        notes="Delay reduces rejection of unfair offers. Hot vs cold emotion effect."
    ),

    "ultimatum_information_asymmetry": GameCalibration(
        source="Kagel et al. (1996); Mitzkewitz & Nagel (1993)",
        game_type="ultimatum",
        variant="information_asymmetry",
        mean_proportion=0.35,
        sd_proportion=0.15,
        ci_95=(0.30, 0.40),
        distribution_shape="right_skew",
        modes=[0.30],
        n_studies=10,
        n_participants=1000,
        moderators={
            "info_condition": {"full_info": 0.42, "proposer_knows_more": 0.30,
                               "responder_knows_more": 0.40},
        },
        notes="Informed proposers exploit info advantage. Offers ~28% lower with private info."
    ),

    "ultimatum_multi_round": GameCalibration(
        source="Slembeck (1999); Cooper et al. (2003)",
        game_type="ultimatum",
        variant="multi_round",
        mean_proportion=0.40,
        sd_proportion=0.10,
        ci_95=(0.37, 0.43),
        distribution_shape="left_skew",
        modes=[0.40],
        n_studies=12,
        n_participants=1200,
        moderators={
            "round": {"round_1": 0.42, "round_5": 0.40, "round_10": 0.38},
            "role_rotation": {"fixed_roles": 0.38, "rotating_roles": 0.42},
        },
        notes="Minimal learning effect. Offers slightly decline over rounds with fixed roles."
    ),

    "ultimatum_with_communication": GameCalibration(
        source="Rankin (2003); Zultan (2012)",
        game_type="ultimatum",
        variant="communication",
        mean_proportion=0.45,
        sd_proportion=0.10,
        ci_95=(0.41, 0.49),
        distribution_shape="normal",
        modes=[0.45],
        n_studies=10,
        n_participants=1100,
        moderators={
            "communication_type": {"pre_play_chat": 0.47, "written_message": 0.44,
                                   "demand_option": 0.42},
        },
        notes="Communication increases offers ~7%. Reduces rejection rates more than it changes offers."
    ),

    # ── PUBLIC GOODS VARIANTS (v1.0.9.4) ──────────────────────────────────

    "public_goods_peer_punishment": GameCalibration(
        source="Fehr & Gächter (2002); Nikiforakis & Normann (2008)",
        game_type="public_goods",
        variant="peer_punishment",
        mean_proportion=0.78,
        sd_proportion=0.14,
        ci_95=(0.72, 0.84),
        distribution_shape="left_skew",
        modes=[0.90],
        n_studies=25,
        n_participants=2800,
        subpopulations={
            "full_contributor": 0.40,
            "high_contributor": 0.30,
            "moderate_contributor": 0.18,
            "free_rider": 0.12,
        },
        moderators={
            "punishment_cost_ratio": {"1_to_3": 0.82, "1_to_1": 0.72, "3_to_1": 0.65},
            "rounds": {"round_1": 0.55, "round_5": 0.75, "round_10": 0.85},
        },
        notes="Peer punishment sustains cooperation at ~78%. Cost ratio matters. Antisocial punishment ~15%."
    ),

    "public_goods_with_reward": GameCalibration(
        source="Sefton et al. (2007); Rand et al. (2009)",
        game_type="public_goods",
        variant="reward",
        mean_proportion=0.60,
        sd_proportion=0.16,
        ci_95=(0.55, 0.65),
        distribution_shape="normal",
        modes=[0.60],
        n_studies=12,
        n_participants=1400,
        moderators={
            "reward_type": {"monetary_reward": 0.62, "social_approval": 0.58,
                            "status_reward": 0.56},
        },
        notes="Rewards less effective than punishment for sustaining cooperation. +13% above baseline."
    ),

    "public_goods_threshold": GameCalibration(
        source="Croson & Marks (2000); Cadsby & Maynes (1999)",
        game_type="public_goods",
        variant="threshold",
        mean_proportion=0.65,
        sd_proportion=0.18,
        ci_95=(0.59, 0.71),
        distribution_shape="bimodal",
        modes=[0.0, 0.80],
        n_studies=18,
        n_participants=1800,
        subpopulations={
            "full_contributor": 0.35,
            "threshold_matcher": 0.30,
            "free_rider": 0.20,
            "partial_contributor": 0.15,
        },
        moderators={
            "threshold_level": {"low_25pct": 0.72, "medium_50pct": 0.65, "high_75pct": 0.55},
            "refund_rule": {"no_refund": 0.58, "full_refund": 0.70},
        },
        notes="Threshold provision point increases cooperation. Refund rule matters. Bimodal: contribute or not."
    ),

    "public_goods_step_level": GameCalibration(
        source="Van de Kragt et al. (1986); Rapoport & Eshed-Levy (1989)",
        game_type="public_goods",
        variant="step_level",
        mean_proportion=0.55,
        sd_proportion=0.22,
        ci_95=(0.48, 0.62),
        distribution_shape="bimodal",
        modes=[0.0, 1.0],
        n_studies=14,
        n_participants=1200,
        moderators={
            "group_size": {"small_3_4": 0.65, "medium_5_7": 0.55, "large_8_plus": 0.42},
        },
        notes="Step-level: all-or-nothing provision. Coordination failure common in large groups."
    ),

    "public_goods_with_communication": GameCalibration(
        source="Isaac & Walker (1988); Bochet et al. (2006)",
        game_type="public_goods",
        variant="communication",
        mean_proportion=0.72,
        sd_proportion=0.15,
        ci_95=(0.67, 0.77),
        distribution_shape="left_skew",
        modes=[0.80],
        n_studies=22,
        n_participants=2500,
        moderators={
            "communication_type": {"face_to_face": 0.82, "chat": 0.70,
                                   "pre_play_only": 0.65, "numerical_signal": 0.55},
        },
        notes="Communication is most effective institution for cooperation. Face-to-face strongest."
    ),

    "public_goods_with_leadership": GameCalibration(
        source="Güth et al. (2007); Levati et al. (2007)",
        game_type="public_goods",
        variant="leadership",
        mean_proportion=0.62,
        sd_proportion=0.16,
        ci_95=(0.56, 0.68),
        distribution_shape="normal",
        modes=[0.60],
        n_studies=10,
        n_participants=1000,
        moderators={
            "leader_type": {"elected": 0.68, "appointed": 0.58,
                            "leading_by_example": 0.65, "sanctioning_leader": 0.72},
        },
        notes="Leaders contribute more and pull followers up. Elected leaders most effective."
    ),

    "public_goods_repeated_decay": GameCalibration(
        source="Isaac et al. (1994); Ledyard (1995)",
        game_type="public_goods",
        variant="repeated_decay",
        mean_proportion=0.35,
        sd_proportion=0.18,
        ci_95=(0.30, 0.40),
        distribution_shape="right_skew",
        modes=[0.20],
        n_studies=30,
        n_participants=3600,
        moderators={
            "round_block": {"round_1_3": 0.50, "round_4_6": 0.38,
                            "round_7_9": 0.28, "round_10": 0.18},
            "group_size": {"small_4": 0.40, "medium_8": 0.33, "large_40": 0.25},
        },
        notes="Classic decay: contributions fall from ~50% to ~18% over 10 rounds without punishment."
    ),

    "public_goods_with_inequality": GameCalibration(
        source="Cherry et al. (2005); Chan et al. (1999)",
        game_type="public_goods",
        variant="inequality",
        mean_proportion=0.40,
        sd_proportion=0.20,
        ci_95=(0.34, 0.46),
        distribution_shape="normal",
        modes=[0.40],
        n_studies=12,
        n_participants=1200,
        moderators={
            "inequality_type": {"unequal_endowment": 0.38, "unequal_mpcr": 0.42,
                                "both_unequal": 0.35},
            "position": {"rich_player": 0.35, "poor_player": 0.48},
        },
        notes="Inequality reduces total contributions. Rich contribute less proportionally. Poor contribute more."
    ),

    # ── COORDINATION & OTHER GAMES (v1.0.9.4) ─────────────────────────────

    "stag_hunt_with_communication": GameCalibration(
        source="Charness (2000); Clark & Sefton (2001)",
        game_type="stag_hunt",
        variant="communication",
        mean_proportion=0.78,
        sd_proportion=0.18,
        ci_95=(0.72, 0.84),
        distribution_shape="left_skew",
        modes=[1.0],
        n_studies=10,
        n_participants=800,
        moderators={
            "communication_type": {"two_way": 0.82, "one_way": 0.72, "none": 0.60},
        },
        notes="Communication dramatically increases payoff-dominant (Stag) coordination."
    ),

    "battle_of_sexes_standard": GameCalibration(
        source="Cooper et al. (1989); Straub (1995)",
        game_type="battle_of_sexes",
        variant="standard",
        mean_proportion=0.50,
        sd_proportion=0.25,
        ci_95=(0.43, 0.57),
        distribution_shape="bimodal",
        modes=[0.0, 1.0],
        n_studies=12,
        n_participants=1000,
        subpopulations={
            "own_preferred": 0.55,
            "other_preferred": 0.30,
            "miscoordination": 0.15,
        },
        moderators={
            "payoff_asymmetry": {"symmetric": 0.50, "asymmetric": 0.55},
        },
        notes="~55% choose own preferred equilibrium. Miscoordination ~15%. Focal points help."
    ),

    "chicken_hawk_dove_standard": GameCalibration(
        source="Rapoport & Chammah (1966); Neugebauer et al. (2008)",
        game_type="chicken",
        variant="standard",
        mean_proportion=0.52,
        sd_proportion=0.22,
        ci_95=(0.45, 0.59),
        distribution_shape="bimodal",
        modes=[0.0, 1.0],
        n_studies=10,
        n_participants=900,
        subpopulations={
            "hawk_aggressive": 0.48,
            "dove_yield": 0.52,
        },
        moderators={
            "framing": {"abstract": 0.50, "conflict_framing": 0.55, "cooperation_framing": 0.45},
        },
        notes="Hawk-Dove/Chicken: ~52% yield (Dove). Anti-coordination game."
    ),

    "common_pool_resource_with_communication": GameCalibration(
        source="Ostrom et al. (1992); Hackett et al. (1994)",
        game_type="common_pool_resource",
        variant="communication",
        mean_proportion=0.45,
        sd_proportion=0.15,
        ci_95=(0.40, 0.50),
        distribution_shape="normal",
        modes=[0.45],
        n_studies=14,
        n_participants=1200,
        moderators={
            "communication_type": {"face_to_face": 0.38, "chat": 0.45,
                                   "one_shot_announcement": 0.52, "none": 0.65},
        },
        notes="Communication reduces extraction ~30%. Face-to-face most effective for resource conservation."
    ),

    "bertrand_competition_standard": GameCalibration(
        source="Dufwenberg & Gneezy (2000); Argenton & Müller (2012)",
        game_type="bertrand_competition",
        variant="standard",
        mean_proportion=0.45,
        sd_proportion=0.20,
        ci_95=(0.38, 0.52),
        distribution_shape="right_skew",
        modes=[0.10],
        n_studies=15,
        n_participants=1400,
        moderators={
            "n_firms": {"duopoly": 0.55, "triopoly": 0.35, "4_plus_firms": 0.15},
        },
        notes="Price/max_price ratio. Converges to MC with more firms. Duopoly: supracompetitive."
    ),

    "cournot_competition_standard": GameCalibration(
        source="Huck et al. (2004); Fouraker & Siegel (1963)",
        game_type="cournot_competition",
        variant="standard",
        mean_proportion=0.58,
        sd_proportion=0.15,
        ci_95=(0.52, 0.64),
        distribution_shape="normal",
        modes=[0.55],
        n_studies=18,
        n_participants=1600,
        moderators={
            "n_firms": {"duopoly": 0.55, "triopoly": 0.60, "4_plus_firms": 0.65},
            "information": {"full": 0.58, "incomplete": 0.55},
        },
        notes="Quantity/Nash equilibrium quantity ratio. Slight overproduction (above Nash). More competitive than theory."
    ),

    "beauty_contest_iterated": GameCalibration(
        source="Ho et al. (1998); Bosch-Domènech et al. (2002)",
        game_type="beauty_contest",
        variant="iterated",
        mean_proportion=0.15,
        sd_proportion=0.12,
        ci_95=(0.10, 0.20),
        distribution_shape="right_skew",
        modes=[0.05],
        n_studies=12,
        n_participants=2000,
        moderators={
            "round": {"round_1": 0.33, "round_3": 0.20, "round_5": 0.12, "round_10": 0.05},
            "population": {"students": 0.33, "game_theorists": 0.15, "ceos": 0.30},
        },
        notes="Learning drives guesses toward 0 (Nash). CEOs no better than students initially."
    ),

    "centipede_standard": GameCalibration(
        source="McKelvey & Palfrey (1992); Levitt et al. (2011)",
        game_type="centipede",
        variant="standard",
        mean_proportion=0.65,
        sd_proportion=0.22,
        ci_95=(0.58, 0.72),
        distribution_shape="bimodal",
        modes=[0.40, 0.90],
        n_studies=15,
        n_participants=1400,
        subpopulations={
            "early_stopper_1_2": 0.25,
            "mid_stopper_3_4": 0.35,
            "late_stopper_5_6": 0.30,
            "full_cooperator": 0.10,
        },
        moderators={
            "game_length": {"4_moves": 0.55, "6_moves": 0.65, "10_moves": 0.75},
            "stakes": {"low": 0.70, "high": 0.55},
        },
        notes="Proportion of game played before stopping. Most deviate from backward induction (stop at move 1)."
    ),

    "market_entry_standard": GameCalibration(
        source="Sundali et al. (1995); Rapoport et al. (2002)",
        game_type="market_entry",
        variant="standard",
        mean_proportion=0.55,
        sd_proportion=0.15,
        ci_95=(0.50, 0.60),
        distribution_shape="normal",
        modes=[0.55],
        n_studies=14,
        n_participants=1600,
        moderators={
            "market_capacity": {"low_25pct": 0.35, "medium_50pct": 0.55, "high_75pct": 0.72},
            "information": {"full": 0.52, "partial": 0.58},
        },
        notes="Proportion choosing to enter. Remarkable aggregate convergence to Nash prediction (magic of markets)."
    ),

    "volunteer_dilemma_standard": GameCalibration(
        source="Diekmann (1985); Goeree et al. (2017)",
        game_type="volunteer_dilemma",
        variant="standard",
        mean_proportion=0.55,
        sd_proportion=0.20,
        ci_95=(0.48, 0.62),
        distribution_shape="bimodal",
        modes=[0.0, 1.0],
        n_studies=12,
        n_participants=1100,
        subpopulations={
            "volunteer": 0.55,
            "free_rider": 0.45,
        },
        moderators={
            "group_size": {"2_person": 0.65, "4_person": 0.50, "8_person": 0.38},
            "cost_of_volunteering": {"low": 0.65, "medium": 0.50, "high": 0.35},
        },
        notes="Volunteering rate decreases with group size (diffusion of responsibility)."
    ),

    # ── SOCIAL DILEMMA VARIANTS (v1.0.9.4) ────────────────────────────────

    "tragedy_of_commons_standard": GameCalibration(
        source="Ostrom (1990); Walker & Gardner (1992)",
        game_type="tragedy_of_commons",
        variant="standard",
        mean_proportion=0.62,
        sd_proportion=0.18,
        ci_95=(0.56, 0.68),
        distribution_shape="right_skew",
        modes=[0.70],
        n_studies=20,
        n_participants=2000,
        moderators={
            "resource_visibility": {"visible_depletion": 0.55, "hidden_stock": 0.68},
            "group_size": {"small_4": 0.55, "medium_8": 0.62, "large_20_plus": 0.72},
        },
        notes="Extraction/optimal ratio. Groups overextract by ~24%. Larger groups overextract more."
    ),

    "prisoners_dilemma_with_punishment": GameCalibration(
        source="Dreber et al. (2008); Rand et al. (2009)",
        game_type="prisoners_dilemma",
        variant="punishment",
        mean_proportion=0.58,
        sd_proportion=0.20,
        ci_95=(0.52, 0.64),
        distribution_shape="bimodal",
        modes=[0.0, 1.0],
        n_studies=16,
        n_participants=2400,
        subpopulations={
            "cooperator": 0.58,
            "defector": 0.42,
        },
        moderators={
            "punishment_type": {"costly_punishment": 0.62, "free_punishment": 0.70,
                                "antisocial_possible": 0.50},
        },
        notes="Punishment increases cooperation ~23%. But costly punishment can reduce group payoffs."
    ),

    "prisoners_dilemma_iterated_axelrod": GameCalibration(
        source="Axelrod (1984); Dal Bó & Fréchette (2011)",
        game_type="prisoners_dilemma",
        variant="iterated",
        mean_proportion=0.60,
        sd_proportion=0.20,
        ci_95=(0.55, 0.65),
        distribution_shape="bimodal",
        modes=[0.0, 1.0],
        n_studies=22,
        n_participants=3200,
        subpopulations={
            "always_cooperate": 0.15,
            "conditional_cooperator": 0.45,
            "tit_for_tat_like": 0.20,
            "always_defect": 0.20,
        },
        moderators={
            "continuation_probability": {"low_0.5": 0.45, "medium_0.75": 0.58, "high_0.9": 0.70},
            "round_block": {"early": 0.55, "middle": 0.60, "late": 0.62},
        },
        notes="Cooperation rate ~60%. Higher with high continuation probability (shadow of future)."
    ),

    "prisoners_dilemma_with_reputation": GameCalibration(
        source="Milinski et al. (2002); Nowak & Sigmund (2005)",
        game_type="prisoners_dilemma",
        variant="reputation",
        mean_proportion=0.65,
        sd_proportion=0.18,
        ci_95=(0.59, 0.71),
        distribution_shape="left_skew",
        modes=[1.0],
        n_studies=14,
        n_participants=1800,
        moderators={
            "reputation_visibility": {"public_history": 0.70, "score_only": 0.62,
                                      "private": 0.47},
            "observation_probability": {"always_observed": 0.70, "sometimes_0.5": 0.58,
                                        "rarely_0.1": 0.50},
        },
        notes="Reputation sustains indirect reciprocity. Public history strongest. Image scoring works."
    ),

    "prisoners_dilemma_exit_option": GameCalibration(
        source="Orbell & Dawes (1993); Batali & Kitcher (1995)",
        game_type="prisoners_dilemma",
        variant="exit_option",
        mean_proportion=0.62,
        sd_proportion=0.18,
        ci_95=(0.56, 0.68),
        distribution_shape="trimodal",
        modes=[0.0, 0.50, 1.0],
        n_studies=10,
        n_participants=1200,
        subpopulations={
            "cooperator_stays": 0.45,
            "exits_to_safety": 0.25,
            "defector_stays": 0.30,
        },
        moderators={
            "exit_payoff": {"low_exit": 0.55, "medium_exit": 0.62, "high_exit": 0.70},
        },
        notes="Exit option among cooperators: cooperation rate. Exit enables assortment of cooperators."
    ),

    "prisoners_dilemma_multiplayer": GameCalibration(
        source="Barcelo & Capraro (2015); Grujić et al. (2010)",
        game_type="prisoners_dilemma",
        variant="multiplayer",
        mean_proportion=0.42,
        sd_proportion=0.22,
        ci_95=(0.36, 0.48),
        distribution_shape="bimodal",
        modes=[0.0, 1.0],
        n_studies=14,
        n_participants=2000,
        moderators={
            "group_size": {"3_players": 0.50, "5_players": 0.42, "10_players": 0.35,
                           "20_plus_players": 0.28},
        },
        notes="Cooperation declines with group size. N-player PD converges to public goods structure."
    ),

    "prisoners_dilemma_asymmetric": GameCalibration(
        source="Beckenkamp et al. (2007); Ahn et al. (2007)",
        game_type="prisoners_dilemma",
        variant="asymmetric_payoffs",
        mean_proportion=0.40,
        sd_proportion=0.22,
        ci_95=(0.34, 0.46),
        distribution_shape="bimodal",
        modes=[0.0, 1.0],
        n_studies=10,
        n_participants=1000,
        moderators={
            "asymmetry_level": {"slight": 0.45, "moderate": 0.40, "large": 0.32},
            "position": {"advantaged_player": 0.35, "disadvantaged_player": 0.45},
        },
        notes="Asymmetric payoffs reduce cooperation. Advantaged players defect more. Inequity aversion matters."
    ),

    "prisoners_dilemma_costly_signaling": GameCalibration(
        source="Gintis et al. (2001); Smith & Bliege Bird (2000)",
        game_type="prisoners_dilemma",
        variant="costly_signaling",
        mean_proportion=0.58,
        sd_proportion=0.20,
        ci_95=(0.51, 0.65),
        distribution_shape="bimodal",
        modes=[0.0, 1.0],
        n_studies=8,
        n_participants=800,
        subpopulations={
            "signaler_cooperator": 0.50,
            "non_signaler_cooperator": 0.20,
            "signaler_defector": 0.10,
            "non_signaler_defector": 0.20,
        },
        moderators={
            "signal_cost": {"cheap": 0.52, "moderate": 0.58, "expensive": 0.65},
        },
        notes="Costly signals of cooperative intent increase cooperation. Higher cost = more credible."
    ),
}


# =============================================================================
# CONSTRUCT NORMS DATABASE
# =============================================================================

CONSTRUCT_NORMS: Dict[str, ConstructNorm] = {

    "loneliness_ucla": ConstructNorm(
        source="Russell (1996)",
        construct="loneliness",
        scale_name="UCLA Loneliness Scale",
        scale_points=4,
        mean=2.2,
        sd=0.65,
        skewness=0.4,
        sample_type="student",
        n_participants=4000,
    ),

    "life_satisfaction_swls": ConstructNorm(
        source="Diener et al. (1985); Pavot & Diener (1993)",
        construct="life_satisfaction",
        scale_name="SWLS",
        scale_points=7,
        mean=4.8,
        sd=1.35,
        skewness=-0.3,
        sample_type="general",
        n_participants=12000,
        moderators={
            "culture": {"western": 5.0, "east_asian": 4.2, "latin_american": 5.3},
            "age": {"18_25": 4.6, "26_40": 4.8, "41_60": 5.0, "61_plus": 5.2},
        },
    ),

    "self_esteem_rse": ConstructNorm(
        source="Schmitt & Allik (2005) — 53-nation study",
        construct="self_esteem",
        scale_name="Rosenberg Self-Esteem Scale",
        scale_points=4,
        mean=3.0,
        sd=0.55,
        skewness=-0.3,
        sample_type="general",
        culture="global",
        n_participants=16998,
        moderators={
            "culture": {"western": 3.2, "east_asian": 2.7, "african": 2.9, "latin_american": 3.1},
            "gender": {"male": 3.1, "female": 2.9},
        },
    ),

    "big_five_agreeableness": ConstructNorm(
        source="Costa & McCrae (1992); Schmitt et al. (2007)",
        construct="agreeableness",
        scale_name="NEO-PI-R / BFI",
        scale_points=5,
        mean=3.7,
        sd=0.60,
        skewness=-0.2,
        sample_type="general",
        n_participants=50000,
        moderators={
            "gender": {"male": 3.5, "female": 3.9},
            "culture": {"western": 3.7, "east_asian": 3.4, "african": 3.8},
        },
    ),

    "big_five_conscientiousness": ConstructNorm(
        source="Costa & McCrae (1992)",
        construct="conscientiousness",
        scale_name="NEO-PI-R",
        scale_points=5,
        mean=3.5,
        sd=0.65,
        sample_type="general",
        n_participants=50000,
    ),

    "big_five_extraversion": ConstructNorm(
        source="Costa & McCrae (1992)",
        construct="extraversion",
        scale_name="NEO-PI-R",
        scale_points=5,
        mean=3.3,
        sd=0.70,
        sample_type="general",
        n_participants=50000,
        moderators={
            "culture": {"western": 3.4, "east_asian": 3.0},
        },
    ),

    "big_five_neuroticism": ConstructNorm(
        source="Costa & McCrae (1992)",
        construct="neuroticism",
        scale_name="NEO-PI-R",
        scale_points=5,
        mean=2.9,
        sd=0.75,
        skewness=0.2,
        sample_type="general",
        n_participants=50000,
        moderators={
            "gender": {"male": 2.7, "female": 3.1},
        },
    ),

    "big_five_openness": ConstructNorm(
        source="Costa & McCrae (1992)",
        construct="openness",
        scale_name="NEO-PI-R",
        scale_points=5,
        mean=3.4,
        sd=0.60,
        sample_type="general",
        n_participants=50000,
    ),

    "burnout_emotional_exhaustion": ConstructNorm(
        source="Maslach & Jackson (1981)",
        construct="emotional_exhaustion",
        scale_name="MBI",
        scale_points=7,
        mean=3.2,
        sd=1.30,
        skewness=0.3,
        sample_type="general",
        n_participants=11000,
    ),

    "gratitude_gq6": ConstructNorm(
        source="McCullough et al. (2002)",
        construct="gratitude",
        scale_name="GQ-6",
        scale_points=7,
        mean=5.8,
        sd=0.85,
        skewness=-0.5,
        sample_type="general",
        n_participants=2000,
    ),

    "resilience_cd_risc": ConstructNorm(
        source="Connor & Davidson (2003)",
        construct="resilience",
        scale_name="CD-RISC",
        scale_points=5,
        mean=3.2,
        sd=0.62,
        sample_type="general",
        n_participants=577,
    ),

    "moral_identity_aquino": ConstructNorm(
        source="Aquino & Reed (2002)",
        construct="moral_identity",
        scale_name="Moral Identity Scale",
        scale_points=7,
        mean=5.7,
        sd=0.85,
        skewness=-0.4,
        sample_type="student",
        n_participants=600,
    ),

    "narcissism_npi": ConstructNorm(
        source="Raskin & Terry (1988); Foster et al. (2003 meta)",
        construct="narcissism",
        scale_name="NPI-40",
        scale_points=40,
        mean=15.5,
        sd=6.8,
        skewness=0.3,
        sample_type="student",
        n_participants=10000,
        moderators={
            "gender": {"male": 16.5, "female": 14.5},
        },
    ),

    "attachment_anxiety_ecr": ConstructNorm(
        source="Brennan et al. (1998); Fraley et al. (2000)",
        construct="attachment_anxiety",
        scale_name="ECR",
        scale_points=7,
        mean=3.2,
        sd=1.20,
        sample_type="general",
        n_participants=5000,
    ),

    "attachment_avoidance_ecr": ConstructNorm(
        source="Brennan et al. (1998)",
        construct="attachment_avoidance",
        scale_name="ECR",
        scale_points=7,
        mean=3.0,
        sd=1.15,
        sample_type="general",
        n_participants=5000,
    ),

    "need_for_cognition": ConstructNorm(
        source="Cacioppo et al. (1984)",
        construct="need_for_cognition",
        scale_name="NFC-18",
        scale_points=5,
        mean=3.4,
        sd=0.65,
        sample_type="student",
        n_participants=3000,
    ),

    "conspiracy_beliefs_gcbs": ConstructNorm(
        source="Brotherton et al. (2013)",
        construct="conspiracy_beliefs",
        scale_name="GCBS",
        scale_points=5,
        mean=2.6,
        sd=0.75,
        skewness=0.4,
        sample_type="general",
        n_participants=2000,
    ),

    "disgust_sensitivity_dsr": ConstructNorm(
        source="Olatunji et al. (2007)",
        construct="disgust_sensitivity",
        scale_name="DS-R",
        scale_points=5,
        mean=2.6,
        sd=0.68,
        sample_type="general",
        n_participants=3000,
        moderators={
            "gender": {"male": 2.3, "female": 2.9},
        },
    ),

    "state_anxiety_stai": ConstructNorm(
        source="Spielberger et al. (1983)",
        construct="state_anxiety",
        scale_name="STAI-S",
        scale_points=4,
        mean=2.0,
        sd=0.55,
        skewness=0.3,
        sample_type="general",
        n_participants=5000,
        moderators={
            "sample": {"non_clinical": 2.0, "clinical": 2.8},
        },
    ),

    "depression_phq9": ConstructNorm(
        source="Kroenke et al. (2001)",
        construct="depression",
        scale_name="PHQ-9",
        scale_points=4,
        mean=1.0,
        sd=0.65,
        skewness=1.2,
        sample_type="general",
        n_participants=6000,
        moderators={
            "sample": {"non_clinical": 0.8, "clinical": 2.2},
        },
    ),

    "perceived_stress_pss": ConstructNorm(
        source="Cohen et al. (1983); Lee (2012 meta norms)",
        construct="perceived_stress",
        scale_name="PSS-10",
        scale_points=5,
        mean=2.5,
        sd=0.65,
        sample_type="general",
        n_participants=15000,
    ),

    "impulsivity_bis": ConstructNorm(
        source="Stanford et al. (2009)",
        construct="impulsivity",
        scale_name="BIS-11",
        scale_points=4,
        mean=2.1,
        sd=0.40,
        sample_type="general",
        n_participants=1577,
    ),

    # =========================================================================
    # CLINICAL PSYCHOLOGY (~20 entries)
    # =========================================================================

    "anxiety_gad7": ConstructNorm(
        source="Spitzer et al. (2006)",
        construct="generalized_anxiety",
        scale_name="GAD-7",
        scale_points=4,  # 0-3 per item
        mean=0.97,
        sd=0.78,
        skewness=1.0,
        sample_type="general",
        n_participants=5030,
        moderators={
            "sample": {"non_clinical": 0.8, "primary_care": 1.0, "clinical_anxiety": 2.4},
            "gender": {"male": 0.85, "female": 1.10},
        },
    ),

    "depression_bdi2": ConstructNorm(
        source="Beck et al. (1996); Dozois et al. (1998)",
        construct="depression",
        scale_name="BDI-II",
        scale_points=4,  # 0-3 per item
        mean=0.53,
        sd=0.48,
        skewness=1.3,
        sample_type="general",
        n_participants=8000,
        moderators={
            "sample": {"non_clinical": 0.40, "student": 0.53, "clinical_depression": 2.10},
            "gender": {"male": 0.45, "female": 0.60},
        },
    ),

    "ptsd_pcl5": ConstructNorm(
        source="Blevins et al. (2015); Bovin et al. (2016)",
        construct="ptsd_symptoms",
        scale_name="PCL-5",
        scale_points=5,  # 0-4 per item
        mean=0.75,
        sd=0.80,
        skewness=1.4,
        sample_type="general",
        n_participants=3500,
        moderators={
            "sample": {"non_clinical": 0.50, "trauma_exposed": 1.20, "clinical_ptsd": 2.80},
        },
    ),

    "social_anxiety_lsas": ConstructNorm(
        source="Liebowitz (1987); Heimberg et al. (1999)",
        construct="social_anxiety",
        scale_name="LSAS",
        scale_points=4,  # 0-3 per item (fear + avoidance)
        mean=1.10,
        sd=0.72,
        skewness=0.8,
        sample_type="general",
        n_participants=3200,
        moderators={
            "sample": {"non_clinical": 0.80, "clinical_social_anxiety": 2.30},
        },
    ),

    "ocd_ybocs": ConstructNorm(
        source="Goodman et al. (1989); mean from Farris et al. (2013)",
        construct="obsessive_compulsive",
        scale_name="Y-BOCS",
        scale_points=5,  # 0-4 per item
        mean=0.55,
        sd=0.70,
        skewness=1.6,
        sample_type="general",
        n_participants=2500,
        moderators={
            "sample": {"non_clinical": 0.30, "clinical_ocd": 2.80},
        },
    ),

    "eating_attitudes_eat26": ConstructNorm(
        source="Garner et al. (1982); Garfinkel & Newman (2001)",
        construct="eating_disorder_risk",
        scale_name="EAT-26",
        scale_points=6,  # 0-3 scored (6-point response)
        mean=0.60,
        sd=0.55,
        skewness=1.5,
        sample_type="general",
        n_participants=4200,
        moderators={
            "gender": {"male": 0.40, "female": 0.75},
            "sample": {"non_clinical": 0.50, "clinical_eating_disorder": 2.50},
        },
    ),

    "panic_pdss": ConstructNorm(
        source="Shear et al. (1997); Houck et al. (2002)",
        construct="panic_severity",
        scale_name="PDSS",
        scale_points=5,  # 0-4 per item
        mean=0.45,
        sd=0.65,
        skewness=1.8,
        sample_type="general",
        n_participants=1800,
        moderators={
            "sample": {"non_clinical": 0.25, "clinical_panic": 2.70},
        },
    ),

    "alcohol_use_audit": ConstructNorm(
        source="Saunders et al. (1993); Reinert & Allen (2007 review)",
        construct="alcohol_use",
        scale_name="AUDIT",
        scale_points=5,  # 0-4 per item
        mean=0.90,
        sd=0.72,
        skewness=1.2,
        sample_type="general",
        n_participants=8000,
        moderators={
            "gender": {"male": 1.10, "female": 0.65},
            "age": {"18_25": 1.15, "26_40": 0.95, "41_60": 0.80, "61_plus": 0.55},
        },
    ),

    "anger_staxi": ConstructNorm(
        source="Spielberger (1999); Lievaart et al. (2016)",
        construct="state_anger",
        scale_name="STAXI-2",
        scale_points=4,  # 1-4
        mean=1.45,
        sd=0.55,
        skewness=1.5,
        sample_type="general",
        n_participants=3000,
        moderators={
            "gender": {"male": 1.55, "female": 1.35},
        },
    ),

    "death_anxiety_das": ConstructNorm(
        source="Templer (1970); Lester & Abdel-Khalek (2003 cross-cultural)",
        construct="death_anxiety",
        scale_name="DAS",
        scale_points=2,  # True/False (0-1)
        mean=0.44,
        sd=0.22,
        sample_type="general",
        n_participants=4500,
        moderators={
            "age": {"18_25": 0.50, "26_40": 0.45, "41_60": 0.40, "61_plus": 0.38},
        },
    ),

    "sleep_quality_psqi": ConstructNorm(
        source="Buysse et al. (1989); Mollayeva et al. (2016 review)",
        construct="sleep_quality",
        scale_name="PSQI",
        scale_points=4,  # 0-3 per component
        mean=1.20,
        sd=0.65,
        skewness=0.7,
        sample_type="general",
        n_participants=6000,
        moderators={
            "age": {"18_25": 1.05, "26_40": 1.15, "41_60": 1.30, "61_plus": 1.45},
            "sample": {"non_clinical": 1.05, "clinical_insomnia": 2.50},
        },
    ),

    "body_image_bss": ConstructNorm(
        source="Slade et al. (1990); Cash & Pruzinsky (2002 review)",
        construct="body_satisfaction",
        scale_name="BSS",
        scale_points=7,
        mean=4.2,
        sd=1.30,
        skewness=-0.3,
        sample_type="general",
        n_participants=3000,
        moderators={
            "gender": {"male": 4.5, "female": 3.9},
        },
    ),

    "health_anxiety_hai": ConstructNorm(
        source="Salkovskis et al. (2002)",
        construct="health_anxiety",
        scale_name="HAI",
        scale_points=4,  # 0-3 per item
        mean=0.70,
        sd=0.50,
        skewness=1.0,
        sample_type="general",
        n_participants=2500,
        moderators={
            "sample": {"non_clinical": 0.55, "clinical_health_anxiety": 2.20},
        },
    ),

    "somatization_phq15": ConstructNorm(
        source="Kroenke et al. (2002)",
        construct="somatic_symptoms",
        scale_name="PHQ-15",
        scale_points=3,  # 0-2 per item
        mean=0.80,
        sd=0.52,
        skewness=0.8,
        sample_type="general",
        n_participants=6000,
        moderators={
            "gender": {"male": 0.65, "female": 0.95},
            "sample": {"non_clinical": 0.70, "primary_care": 1.05},
        },
    ),

    "interpersonal_sensitivity_scl": ConstructNorm(
        source="Derogatis (1994); Schmitz et al. (2000)",
        construct="interpersonal_sensitivity",
        scale_name="SCL-90-R IS",
        scale_points=5,  # 0-4
        mean=0.85,
        sd=0.70,
        skewness=0.9,
        sample_type="general",
        n_participants=4000,
        moderators={
            "sample": {"non_clinical": 0.70, "clinical": 1.80},
            "gender": {"male": 0.75, "female": 0.95},
        },
    ),

    "social_phobia_spin": ConstructNorm(
        source="Connor et al. (2000)",
        construct="social_phobia",
        scale_name="SPIN",
        scale_points=5,  # 0-4 per item
        mean=0.85,
        sd=0.68,
        skewness=0.9,
        sample_type="general",
        n_participants=2600,
        moderators={
            "sample": {"non_clinical": 0.65, "clinical_social_phobia": 2.60},
        },
    ),

    "agoraphobia_mi": ConstructNorm(
        source="Chambless et al. (1985; 2011 norms)",
        construct="agoraphobia",
        scale_name="MI",
        scale_points=5,  # 1-5
        mean=1.55,
        sd=0.65,
        skewness=1.2,
        sample_type="general",
        n_participants=1800,
        moderators={
            "sample": {"non_clinical": 1.30, "clinical_agoraphobia": 3.20},
        },
    ),

    "specific_phobia_fsq": ConstructNorm(
        source="Szymanski & O'Donohue (1995); Muris & Merckelbach (1996)",
        construct="specific_phobia",
        scale_name="FSQ",
        scale_points=8,  # 0-7 per item
        mean=1.80,
        sd=1.60,
        skewness=1.4,
        sample_type="general",
        n_participants=2000,
        moderators={
            "gender": {"male": 1.40, "female": 2.20},
        },
    ),

    "hypochondriasis_whi": ConstructNorm(
        source="Pilowsky (1967); Speckens et al. (1996)",
        construct="hypochondriasis",
        scale_name="WI",
        scale_points=2,  # 0-1 (yes/no)
        mean=0.28,
        sd=0.20,
        skewness=0.8,
        sample_type="general",
        n_participants=3000,
        moderators={
            "sample": {"non_clinical": 0.22, "clinical": 0.60},
        },
    ),

    "chronic_fatigue_cfs": ConstructNorm(
        source="Chalder et al. (1993); Cella & Chalder (2010)",
        construct="fatigue",
        scale_name="CFQ",
        scale_points=4,  # 0-3 per item
        mean=1.15,
        sd=0.65,
        skewness=0.5,
        sample_type="general",
        n_participants=3500,
        moderators={
            "sample": {"non_clinical": 1.00, "clinical_cfs": 2.50},
            "gender": {"male": 1.00, "female": 1.25},
        },
    ),

    # =========================================================================
    # WELLBEING & POSITIVE PSYCHOLOGY (~15 entries)
    # =========================================================================

    "flourishing_perma": ConstructNorm(
        source="Butler & Kern (2016); Seligman (2011)",
        construct="flourishing",
        scale_name="PERMA-Profiler",
        scale_points=11,  # 0-10
        mean=7.2,
        sd=1.50,
        skewness=-0.5,
        sample_type="general",
        n_participants=4500,
        moderators={
            "culture": {"western": 7.3, "east_asian": 6.5, "latin_american": 7.5},
        },
    ),

    "positive_affect_panas": ConstructNorm(
        source="Watson et al. (1988); Crawford & Henry (2004 UK norms)",
        construct="positive_affect",
        scale_name="PANAS-PA",
        scale_points=5,  # 1-5
        mean=3.5,
        sd=0.70,
        skewness=-0.2,
        sample_type="general",
        n_participants=5000,
        moderators={
            "age": {"18_25": 3.4, "26_40": 3.5, "41_60": 3.6, "61_plus": 3.5},
            "gender": {"male": 3.5, "female": 3.4},
        },
    ),

    "negative_affect_panas": ConstructNorm(
        source="Watson et al. (1988); Crawford & Henry (2004)",
        construct="negative_affect",
        scale_name="PANAS-NA",
        scale_points=5,  # 1-5
        mean=2.0,
        sd=0.70,
        skewness=0.7,
        sample_type="general",
        n_participants=5000,
        moderators={
            "gender": {"male": 1.9, "female": 2.1},
            "age": {"18_25": 2.2, "26_40": 2.0, "41_60": 1.9, "61_plus": 1.8},
        },
    ),

    "psychological_wellbeing_pwb": ConstructNorm(
        source="Ryff & Keyes (1995); Ryff (1989)",
        construct="psychological_wellbeing",
        scale_name="PWB",
        scale_points=6,  # 1-6
        mean=4.3,
        sd=0.80,
        skewness=-0.2,
        sample_type="general",
        n_participants=3500,
        moderators={
            "age": {"18_25": 4.1, "26_40": 4.3, "41_60": 4.4, "61_plus": 4.5},
        },
    ),

    "meaning_presence_mlq": ConstructNorm(
        source="Steger et al. (2006)",
        construct="meaning_in_life_presence",
        scale_name="MLQ-Presence",
        scale_points=7,  # 1-7
        mean=4.7,
        sd=1.30,
        skewness=-0.3,
        sample_type="general",
        n_participants=3000,
        moderators={
            "age": {"18_25": 4.3, "26_40": 4.7, "41_60": 5.0, "61_plus": 5.3},
        },
    ),

    "meaning_search_mlq": ConstructNorm(
        source="Steger et al. (2006)",
        construct="meaning_in_life_search",
        scale_name="MLQ-Search",
        scale_points=7,  # 1-7
        mean=4.5,
        sd=1.40,
        sample_type="general",
        n_participants=3000,
        moderators={
            "age": {"18_25": 5.0, "26_40": 4.5, "41_60": 4.0, "61_plus": 3.5},
        },
    ),

    "hope_ahs": ConstructNorm(
        source="Snyder et al. (1991)",
        construct="hope",
        scale_name="AHS",
        scale_points=8,  # 1-8
        mean=5.7,
        sd=1.10,
        skewness=-0.3,
        sample_type="general",
        n_participants=3500,
    ),

    "optimism_lotr": ConstructNorm(
        source="Scheier et al. (1994); Herzberg et al. (2006 norms)",
        construct="optimism",
        scale_name="LOT-R",
        scale_points=5,  # 0-4
        mean=2.7,
        sd=0.70,
        skewness=-0.2,
        sample_type="general",
        n_participants=6000,
        moderators={
            "culture": {"western": 2.8, "east_asian": 2.4},
            "gender": {"male": 2.8, "female": 2.6},
        },
    ),

    "happiness_shs": ConstructNorm(
        source="Lyubomirsky & Lepper (1999)",
        construct="subjective_happiness",
        scale_name="SHS",
        scale_points=7,  # 1-7
        mean=4.9,
        sd=1.10,
        skewness=-0.4,
        sample_type="general",
        n_participants=5000,
        moderators={
            "culture": {"western": 5.0, "east_asian": 4.4, "latin_american": 5.3},
        },
    ),

    "vitality_svs": ConstructNorm(
        source="Ryan & Frederick (1997)",
        construct="subjective_vitality",
        scale_name="SVS",
        scale_points=7,  # 1-7
        mean=4.6,
        sd=1.15,
        sample_type="general",
        n_participants=2000,
    ),

    "life_purpose_pil": ConstructNorm(
        source="Crumbaugh & Maholick (1964); Schulenberg et al. (2011 meta)",
        construct="purpose_in_life",
        scale_name="PIL",
        scale_points=7,  # 1-7
        mean=4.8,
        sd=1.15,
        sample_type="general",
        n_participants=4000,
        moderators={
            "age": {"18_25": 4.5, "26_40": 4.8, "41_60": 5.0, "61_plus": 5.1},
        },
    ),

    "self_compassion_scs": ConstructNorm(
        source="Neff (2003); Neff et al. (2019 cross-cultural)",
        construct="self_compassion",
        scale_name="SCS",
        scale_points=5,  # 1-5
        mean=3.0,
        sd=0.68,
        sample_type="general",
        n_participants=5000,
        moderators={
            "gender": {"male": 3.1, "female": 2.9},
            "culture": {"western": 3.0, "east_asian": 2.8, "southeast_asian": 3.2},
        },
    ),

    "positive_relations_pwb": ConstructNorm(
        source="Ryff (1989); Ryff & Keyes (1995)",
        construct="positive_relations",
        scale_name="PWB-PR",
        scale_points=6,  # 1-6
        mean=4.4,
        sd=0.85,
        skewness=-0.2,
        sample_type="general",
        n_participants=3500,
        moderators={
            "gender": {"male": 4.2, "female": 4.6},
        },
    ),

    "environmental_mastery_pwb": ConstructNorm(
        source="Ryff (1989); Ryff & Keyes (1995)",
        construct="environmental_mastery",
        scale_name="PWB-EM",
        scale_points=6,  # 1-6
        mean=4.1,
        sd=0.85,
        sample_type="general",
        n_participants=3500,
        moderators={
            "age": {"18_25": 3.8, "26_40": 4.1, "41_60": 4.3, "61_plus": 4.4},
        },
    ),

    "personal_growth_pwb": ConstructNorm(
        source="Ryff (1989); Ryff & Keyes (1995)",
        construct="personal_growth",
        scale_name="PWB-PG",
        scale_points=6,  # 1-6
        mean=4.5,
        sd=0.80,
        skewness=-0.2,
        sample_type="general",
        n_participants=3500,
        moderators={
            "age": {"18_25": 4.6, "26_40": 4.5, "41_60": 4.4, "61_plus": 4.2},
        },
    ),

    # =========================================================================
    # VALUES & IDEOLOGY (~15 entries)
    # =========================================================================

    "schwartz_universalism": ConstructNorm(
        source="Schwartz (1992); Schwartz (2012 refined theory)",
        construct="universalism_values",
        scale_name="PVQ-R",
        scale_points=6,  # 1-6
        mean=4.5,
        sd=0.80,
        sample_type="general",
        n_participants=12000,
        moderators={
            "gender": {"male": 4.3, "female": 4.7},
            "culture": {"western": 4.5, "east_asian": 4.3},
        },
    ),

    "schwartz_benevolence": ConstructNorm(
        source="Schwartz (1992); Schwartz (2012 refined theory)",
        construct="benevolence_values",
        scale_name="PVQ-R",
        scale_points=6,  # 1-6
        mean=4.8,
        sd=0.72,
        skewness=-0.3,
        sample_type="general",
        n_participants=12000,
        moderators={
            "gender": {"male": 4.6, "female": 5.0},
        },
    ),

    "schwartz_power": ConstructNorm(
        source="Schwartz (1992); Schwartz (2012 refined theory)",
        construct="power_values",
        scale_name="PVQ-R",
        scale_points=6,  # 1-6
        mean=3.0,
        sd=1.00,
        skewness=0.3,
        sample_type="general",
        n_participants=12000,
        moderators={
            "gender": {"male": 3.3, "female": 2.7},
        },
    ),

    "sdo_scale": ConstructNorm(
        source="Pratto et al. (1994); Ho et al. (2015 SDO7)",
        construct="social_dominance_orientation",
        scale_name="SDO-7",
        scale_points=7,  # 1-7
        mean=2.8,
        sd=1.10,
        skewness=0.5,
        sample_type="general",
        n_participants=8000,
        moderators={
            "gender": {"male": 3.2, "female": 2.4},
            "culture": {"western": 2.8, "east_asian": 2.6},
        },
    ),

    "rwa_scale": ConstructNorm(
        source="Altemeyer (1981; 1996); Duckitt & Bizumic (2013 ACT)",
        construct="right_wing_authoritarianism",
        scale_name="RWA / ACT",
        scale_points=7,  # 1-7 (varies by version)
        mean=3.5,
        sd=1.15,
        sample_type="general",
        n_participants=6000,
        moderators={
            "age": {"18_25": 3.0, "26_40": 3.4, "41_60": 3.8, "61_plus": 4.0},
        },
    ),

    "political_ideology": ConstructNorm(
        source="Jost et al. (2009); compiled from multiple surveys",
        construct="political_ideology",
        scale_name="Liberal-Conservative",
        scale_points=7,  # 1=very liberal to 7=very conservative
        mean=4.0,
        sd=1.50,
        sample_type="general",
        n_participants=20000,
        moderators={
            "age": {"18_25": 3.5, "26_40": 3.8, "41_60": 4.2, "61_plus": 4.5},
        },
    ),

    "system_justification": ConstructNorm(
        source="Kay & Jost (2003); Jost & Hunyady (2005)",
        construct="system_justification",
        scale_name="SJT",
        scale_points=9,  # 1-9
        mean=4.8,
        sd=1.55,
        sample_type="general",
        n_participants=3000,
        moderators={
            "culture": {"western": 4.8, "east_asian": 5.2},
        },
    ),

    "just_world_belief_bjw": ConstructNorm(
        source="Dalbert (1999); Lipkus (1991)",
        construct="just_world_belief",
        scale_name="BJW",
        scale_points=6,  # 1-6
        mean=3.5,
        sd=0.90,
        sample_type="general",
        n_participants=4000,
        moderators={
            "culture": {"western": 3.3, "east_asian": 3.8},
        },
    ),

    "protestant_work_ethic": ConstructNorm(
        source="Mirels & Garrett (1971); Christopher & Jones (2014 revision)",
        construct="protestant_work_ethic",
        scale_name="PWE",
        scale_points=7,  # 1-7
        mean=4.2,
        sd=0.90,
        sample_type="general",
        n_participants=2500,
    ),

    "materialism_mvs": ConstructNorm(
        source="Richins & Dawson (1992); Richins (2004 MVS-15)",
        construct="materialism",
        scale_name="MVS",
        scale_points=5,  # 1-5
        mean=2.8,
        sd=0.70,
        skewness=0.3,
        sample_type="general",
        n_participants=4000,
        moderators={
            "age": {"18_25": 3.1, "26_40": 2.9, "41_60": 2.6, "61_plus": 2.3},
        },
    ),

    "moral_foundations_care": ConstructNorm(
        source="Graham et al. (2011); yourmorals.org norms",
        construct="moral_foundations_care",
        scale_name="MFQ-30 Care",
        scale_points=6,  # 0-5
        mean=3.7,
        sd=0.82,
        sample_type="general",
        n_participants=12000,
        moderators={
            "political": {"liberal": 4.0, "moderate": 3.6, "conservative": 3.3},
            "gender": {"male": 3.5, "female": 3.9},
        },
    ),

    "moral_foundations_fairness": ConstructNorm(
        source="Graham et al. (2011); yourmorals.org norms",
        construct="moral_foundations_fairness",
        scale_name="MFQ-30 Fairness",
        scale_points=6,  # 0-5
        mean=3.6,
        sd=0.80,
        sample_type="general",
        n_participants=12000,
        moderators={
            "political": {"liberal": 3.9, "moderate": 3.5, "conservative": 3.2},
        },
    ),

    "moral_foundations_loyalty": ConstructNorm(
        source="Graham et al. (2011); yourmorals.org norms",
        construct="moral_foundations_loyalty",
        scale_name="MFQ-30 Loyalty",
        scale_points=6,  # 0-5
        mean=2.9,
        sd=0.90,
        sample_type="general",
        n_participants=12000,
        moderators={
            "political": {"liberal": 2.4, "moderate": 2.9, "conservative": 3.5},
        },
    ),

    "moral_foundations_authority": ConstructNorm(
        source="Graham et al. (2011); yourmorals.org norms",
        construct="moral_foundations_authority",
        scale_name="MFQ-30 Authority",
        scale_points=6,  # 0-5
        mean=2.8,
        sd=0.92,
        sample_type="general",
        n_participants=12000,
        moderators={
            "political": {"liberal": 2.2, "moderate": 2.8, "conservative": 3.5},
        },
    ),

    "moral_foundations_purity": ConstructNorm(
        source="Graham et al. (2011); yourmorals.org norms",
        construct="moral_foundations_purity",
        scale_name="MFQ-30 Purity",
        scale_points=6,  # 0-5
        mean=2.5,
        sd=1.05,
        skewness=0.3,
        sample_type="general",
        n_participants=12000,
        moderators={
            "political": {"liberal": 1.8, "moderate": 2.5, "conservative": 3.3},
        },
    ),

    # =========================================================================
    # SOCIAL PSYCHOLOGY (~15 entries)
    # =========================================================================

    "social_support_mspss": ConstructNorm(
        source="Zimet et al. (1988); Cecil et al. (1995 norms)",
        construct="perceived_social_support",
        scale_name="MSPSS",
        scale_points=7,  # 1-7
        mean=5.5,
        sd=1.15,
        skewness=-0.6,
        sample_type="general",
        n_participants=5000,
        moderators={
            "gender": {"male": 5.3, "female": 5.7},
            "age": {"18_25": 5.6, "26_40": 5.4, "41_60": 5.3, "61_plus": 5.5},
        },
    ),

    "belongingness_gnbs": ConstructNorm(
        source="Leary et al. (2013); Malone et al. (2012)",
        construct="need_to_belong",
        scale_name="NTB",
        scale_points=5,  # 1-5
        mean=3.5,
        sd=0.65,
        sample_type="general",
        n_participants=3000,
        moderators={
            "gender": {"male": 3.3, "female": 3.7},
        },
    ),

    "social_comparison_sco": ConstructNorm(
        source="Gibbons & Buunk (1999); INCOM scale",
        construct="social_comparison_orientation",
        scale_name="INCOM",
        scale_points=5,  # 1-5
        mean=3.3,
        sd=0.72,
        sample_type="general",
        n_participants=2500,
        moderators={
            "gender": {"male": 3.1, "female": 3.5},
        },
    ),

    "social_identity_sis": ConstructNorm(
        source="Cameron (2004); Leach et al. (2008)",
        construct="social_identity",
        scale_name="SIS",
        scale_points=7,  # 1-7
        mean=4.5,
        sd=1.15,
        sample_type="general",
        n_participants=3000,
    ),

    "collective_self_esteem_cse": ConstructNorm(
        source="Luhtanen & Crocker (1992)",
        construct="collective_self_esteem",
        scale_name="CSE",
        scale_points=7,  # 1-7
        mean=5.0,
        sd=0.95,
        skewness=-0.3,
        sample_type="student",
        n_participants=2000,
    ),

    "empathic_concern_iri": ConstructNorm(
        source="Davis (1983); De Corte et al. (2007 meta norms)",
        construct="empathic_concern",
        scale_name="IRI-EC",
        scale_points=5,  # 0-4
        mean=2.8,
        sd=0.68,
        sample_type="general",
        n_participants=8000,
        moderators={
            "gender": {"male": 2.5, "female": 3.1},
        },
    ),

    "perspective_taking_iri": ConstructNorm(
        source="Davis (1983); De Corte et al. (2007 meta norms)",
        construct="perspective_taking",
        scale_name="IRI-PT",
        scale_points=5,  # 0-4
        mean=2.6,
        sd=0.65,
        sample_type="general",
        n_participants=8000,
        moderators={
            "gender": {"male": 2.5, "female": 2.7},
        },
    ),

    "personal_distress_iri": ConstructNorm(
        source="Davis (1983)",
        construct="personal_distress",
        scale_name="IRI-PD",
        scale_points=5,  # 0-4
        mean=2.0,
        sd=0.70,
        sample_type="general",
        n_participants=8000,
        moderators={
            "gender": {"male": 1.8, "female": 2.2},
        },
    ),

    "fantasy_iri": ConstructNorm(
        source="Davis (1983)",
        construct="fantasy",
        scale_name="IRI-FS",
        scale_points=5,  # 0-4
        mean=2.4,
        sd=0.75,
        sample_type="general",
        n_participants=8000,
        moderators={
            "gender": {"male": 2.2, "female": 2.6},
        },
    ),

    "interpersonal_trust_its": ConstructNorm(
        source="Rotter (1967); Wrightsman (1991)",
        construct="interpersonal_trust",
        scale_name="ITS",
        scale_points=5,  # 1-5
        mean=3.2,
        sd=0.60,
        sample_type="general",
        n_participants=4000,
    ),

    "social_distance_bogardus": ConstructNorm(
        source="Bogardus (1933); Wark & Galliher (2007 update)",
        construct="social_distance",
        scale_name="Bogardus",
        scale_points=7,  # 1-7
        mean=2.8,
        sd=1.40,
        skewness=0.7,
        sample_type="general",
        n_participants=5000,
    ),

    "contact_frequency_scale": ConstructNorm(
        source="Islam & Hewstone (1993); Pettigrew & Tropp (2006 meta)",
        construct="intergroup_contact",
        scale_name="Contact Scale",
        scale_points=7,  # 1-7
        mean=3.8,
        sd=1.50,
        sample_type="general",
        n_participants=4000,
    ),

    "social_exclusion_nts": ConstructNorm(
        source="Williams (2009); van Beest & Williams (2006)",
        construct="need_threat_ostracism",
        scale_name="NTS",
        scale_points=5,  # 1-5
        mean=3.8,
        sd=0.75,
        skewness=-0.3,
        sample_type="student",
        n_participants=2000,
        moderators={
            "condition": {"included": 4.2, "excluded": 2.1},
        },
    ),

    "ostracism_cyberball": ConstructNorm(
        source="Williams et al. (2000); Hartgerink et al. (2015 meta)",
        construct="ostracism_distress",
        scale_name="NTS-Cyberball",
        scale_points=5,  # 1-5
        mean=3.5,
        sd=0.90,
        sample_type="student",
        n_participants=3000,
        moderators={
            "condition": {"included": 4.0, "excluded": 2.0},
        },
    ),

    "prejudice_feeling_thermometer": ConstructNorm(
        source="Haddock et al. (1993); ANES compiled norms",
        construct="prejudice",
        scale_name="Feeling Thermometer",
        scale_points=101,  # 0-100
        mean=58.0,
        sd=22.0,
        sample_type="general",
        n_participants=10000,
    ),

    # =========================================================================
    # COGNITIVE & SELF-REGULATION (~15 entries)
    # =========================================================================

    "mindfulness_maas": ConstructNorm(
        source="Brown & Ryan (2003)",
        construct="mindful_attention",
        scale_name="MAAS",
        scale_points=6,  # 1-6
        mean=3.8,
        sd=0.70,
        sample_type="general",
        n_participants=3500,
        moderators={
            "sample": {"student": 3.7, "community": 3.9, "meditators": 4.3},
        },
    ),

    "cognitive_flexibility_cfs": ConstructNorm(
        source="Martin & Rubin (1995); Dennis & Vander Wal (2010 CFI)",
        construct="cognitive_flexibility",
        scale_name="CFI",
        scale_points=7,  # 1-7
        mean=4.7,
        sd=0.82,
        skewness=-0.2,
        sample_type="general",
        n_participants=2500,
    ),

    "tolerance_of_ambiguity": ConstructNorm(
        source="Budner (1962); McLain (2009 MSTAT-II)",
        construct="tolerance_of_ambiguity",
        scale_name="MSTAT-II",
        scale_points=5,  # 1-5
        mean=3.3,
        sd=0.60,
        sample_type="general",
        n_participants=2500,
    ),

    "locus_of_control_rotter": ConstructNorm(
        source="Rotter (1966); compiled norms Lefcourt (1991)",
        construct="internal_locus_of_control",
        scale_name="LOC",
        scale_points=2,  # 0/1 forced choice (29 items)
        mean=0.42,
        sd=0.15,
        sample_type="general",
        n_participants=6000,
    ),

    "self_regulation_srs": ConstructNorm(
        source="Carey et al. (2004); Neal & Carey (2005)",
        construct="self_regulation",
        scale_name="SSRQ",
        scale_points=5,  # 1-5
        mean=3.3,
        sd=0.52,
        sample_type="student",
        n_participants=2000,
    ),

    "cognitive_reflection_crt": ConstructNorm(
        source="Frederick (2005); Campitelli & Gerrans (2014 expanded)",
        construct="cognitive_reflection",
        scale_name="CRT",
        scale_points=3,  # 0-3 (proportion correct)
        mean=1.24,
        sd=1.08,
        skewness=0.4,
        sample_type="general",
        n_participants=5000,
        moderators={
            "gender": {"male": 1.47, "female": 1.03},
            "sample": {"student": 1.30, "mturk": 1.50},
        },
    ),

    "growth_mindset_itis": ConstructNorm(
        source="Dweck (1999); De Castella & Byrne (2015 ITIS)",
        construct="growth_mindset",
        scale_name="ITIS",
        scale_points=6,  # 1-6
        mean=4.0,
        sd=0.95,
        sample_type="general",
        n_participants=3000,
        moderators={
            "age": {"18_25": 3.9, "26_40": 4.0, "41_60": 4.1},
        },
    ),

    "grit_scale": ConstructNorm(
        source="Duckworth et al. (2007); Duckworth & Quinn (2009 Short Grit)",
        construct="grit",
        scale_name="Grit-S",
        scale_points=5,  # 1-5
        mean=3.5,
        sd=0.60,
        sample_type="general",
        n_participants=4000,
        moderators={
            "age": {"18_25": 3.3, "26_40": 3.5, "41_60": 3.7, "61_plus": 3.8},
        },
    ),

    "delay_of_gratification": ConstructNorm(
        source="Bembenutty & Karabenick (1998); Hoerger et al. (2011 DGI)",
        construct="delay_of_gratification",
        scale_name="DGI",
        scale_points=5,  # 1-5
        mean=3.2,
        sd=0.65,
        sample_type="general",
        n_participants=2000,
    ),

    "self_monitoring_sm": ConstructNorm(
        source="Snyder (1974); Lennox & Wolfe (1984 revised)",
        construct="self_monitoring",
        scale_name="SMS-R",
        scale_points=6,  # 0-5 (some versions 1-6)
        mean=3.2,
        sd=0.75,
        sample_type="general",
        n_participants=3000,
    ),

    "rumination_rrs": ConstructNorm(
        source="Treynor et al. (2003); Nolen-Hoeksema & Morrow (1991)",
        construct="rumination",
        scale_name="RRS",
        scale_points=4,  # 1-4
        mean=2.1,
        sd=0.55,
        skewness=0.5,
        sample_type="general",
        n_participants=3500,
        moderators={
            "gender": {"male": 1.9, "female": 2.3},
            "sample": {"non_clinical": 2.0, "clinical_depression": 3.0},
        },
    ),

    "worry_pswq": ConstructNorm(
        source="Meyer et al. (1990); Startup & Erickson (2006 norms)",
        construct="worry",
        scale_name="PSWQ",
        scale_points=5,  # 1-5
        mean=2.9,
        sd=0.80,
        skewness=0.3,
        sample_type="general",
        n_participants=4000,
        moderators={
            "gender": {"male": 2.7, "female": 3.1},
            "sample": {"non_clinical": 2.7, "clinical_gad": 4.0},
        },
    ),

    "emotion_regulation_reappraisal_erq": ConstructNorm(
        source="Gross & John (2003)",
        construct="cognitive_reappraisal",
        scale_name="ERQ-CR",
        scale_points=7,  # 1-7
        mean=4.6,
        sd=1.00,
        sample_type="general",
        n_participants=4000,
    ),

    "emotion_regulation_suppression_erq": ConstructNorm(
        source="Gross & John (2003)",
        construct="expressive_suppression",
        scale_name="ERQ-ES",
        scale_points=7,  # 1-7
        mean=3.6,
        sd=1.15,
        sample_type="general",
        n_participants=4000,
        moderators={
            "gender": {"male": 3.9, "female": 3.3},
            "culture": {"western": 3.6, "east_asian": 4.2},
        },
    ),

    "executive_function_brief": ConstructNorm(
        source="Roth et al. (2005); BRIEF-A",
        construct="executive_function_deficit",
        scale_name="BRIEF-A",
        scale_points=3,  # 1-3 (never/sometimes/often)
        mean=1.45,
        sd=0.32,
        skewness=0.8,
        sample_type="general",
        n_participants=2000,
    ),

    # =========================================================================
    # MOTIVATION & ACHIEVEMENT (~10 entries)
    # =========================================================================

    "intrinsic_motivation_imi": ConstructNorm(
        source="Ryan (1982); McAuley et al. (1989)",
        construct="intrinsic_motivation",
        scale_name="IMI",
        scale_points=7,  # 1-7
        mean=4.8,
        sd=1.10,
        sample_type="student",
        n_participants=3000,
    ),

    "extrinsic_motivation_ams": ConstructNorm(
        source="Vallerand et al. (1992; 1993 AMS)",
        construct="extrinsic_motivation",
        scale_name="AMS-Ext",
        scale_points=7,  # 1-7
        mean=4.3,
        sd=1.20,
        sample_type="student",
        n_participants=3000,
    ),

    "academic_motivation_ams_intrinsic": ConstructNorm(
        source="Vallerand et al. (1992; 1993 AMS)",
        construct="academic_intrinsic_motivation",
        scale_name="AMS-IM",
        scale_points=7,  # 1-7
        mean=4.5,
        sd=1.15,
        sample_type="student",
        n_participants=4000,
    ),

    "achievement_goal_mastery": ConstructNorm(
        source="Elliot & Church (1997); Elliot & Murayama (2008 AGQ-R)",
        construct="mastery_approach_goals",
        scale_name="AGQ-R Mastery",
        scale_points=5,  # 1-5
        mean=4.0,
        sd=0.70,
        skewness=-0.3,
        sample_type="student",
        n_participants=3500,
    ),

    "achievement_goal_performance": ConstructNorm(
        source="Elliot & Church (1997); Elliot & Murayama (2008 AGQ-R)",
        construct="performance_approach_goals",
        scale_name="AGQ-R Performance",
        scale_points=5,  # 1-5
        mean=3.5,
        sd=0.90,
        sample_type="student",
        n_participants=3500,
    ),

    "self_efficacy_gse": ConstructNorm(
        source="Schwarzer & Jerusalem (1995); Scholz et al. (2002 cross-cultural)",
        construct="general_self_efficacy",
        scale_name="GSE",
        scale_points=4,  # 1-4
        mean=3.0,
        sd=0.45,
        skewness=-0.2,
        sample_type="general",
        culture="global",
        n_participants=18000,
        moderators={
            "culture": {"western": 3.1, "east_asian": 2.8, "latin_american": 3.0},
            "gender": {"male": 3.1, "female": 2.9},
        },
    ),

    "work_engagement_uwes": ConstructNorm(
        source="Schaufeli et al. (2006); Schaufeli & Bakker (2003 manual)",
        construct="work_engagement",
        scale_name="UWES-9",
        scale_points=7,  # 0-6 (never to always)
        mean=3.8,
        sd=1.10,
        sample_type="general",
        n_participants=10000,
        moderators={
            "age": {"18_25": 3.5, "26_40": 3.8, "41_60": 4.0, "61_plus": 4.2},
        },
    ),

    "flow_dfs": ConstructNorm(
        source="Jackson & Eklund (2002); Engeser & Rheinberg (2008 FKS)",
        construct="flow_experience",
        scale_name="DFS-2",
        scale_points=5,  # 1-5
        mean=3.5,
        sd=0.65,
        sample_type="general",
        n_participants=2500,
    ),

    "procrastination_gps": ConstructNorm(
        source="Lay (1986); Steel (2010 meta-analysis norms)",
        construct="procrastination",
        scale_name="GPS",
        scale_points=5,  # 1-5
        mean=3.0,
        sd=0.72,
        skewness=0.2,
        sample_type="general",
        n_participants=5000,
        moderators={
            "age": {"18_25": 3.3, "26_40": 3.0, "41_60": 2.7, "61_plus": 2.5},
        },
    ),

    "test_anxiety_tai": ConstructNorm(
        source="Spielberger (1980); Chapell et al. (2005 meta norms)",
        construct="test_anxiety",
        scale_name="TAI",
        scale_points=4,  # 1-4
        mean=2.2,
        sd=0.60,
        skewness=0.4,
        sample_type="student",
        n_participants=5000,
        moderators={
            "gender": {"male": 2.0, "female": 2.4},
        },
    ),

    # =========================================================================
    # INTERPERSONAL & RELATIONSHIPS (~15 entries)
    # =========================================================================

    "forgiveness_tfs": ConstructNorm(
        source="Thompson et al. (2005); Berry et al. (2005 meta)",
        construct="trait_forgiveness",
        scale_name="HFS",
        scale_points=7,  # 1-7
        mean=4.5,
        sd=0.90,
        sample_type="general",
        n_participants=3000,
        moderators={
            "gender": {"male": 4.4, "female": 4.6},
            "age": {"18_25": 4.2, "26_40": 4.4, "41_60": 4.7, "61_plus": 4.9},
        },
    ),

    "relationship_satisfaction_ras": ConstructNorm(
        source="Hendrick (1988); Vaughn & Baier (1999 norms)",
        construct="relationship_satisfaction",
        scale_name="RAS",
        scale_points=5,  # 1-5
        mean=3.9,
        sd=0.70,
        skewness=-0.5,
        sample_type="general",
        n_participants=4500,
    ),

    "trust_in_others_gts": ConstructNorm(
        source="Yamagishi & Yamagishi (1994); Naef & Schupp (2009)",
        construct="generalized_trust",
        scale_name="GTS",
        scale_points=5,  # 1-5
        mean=3.0,
        sd=0.70,
        sample_type="general",
        n_participants=5000,
        moderators={
            "culture": {"western": 3.1, "east_asian": 2.7, "scandinavian": 3.5},
        },
    ),

    "conflict_resolution_crq": ConstructNorm(
        source="Kurdek (1994); Heavey et al. (1993)",
        construct="constructive_conflict_resolution",
        scale_name="CRQ",
        scale_points=5,  # 1-5
        mean=3.3,
        sd=0.75,
        sample_type="general",
        n_participants=2000,
    ),

    "communication_quality_csi": ConstructNorm(
        source="Christensen & Sullaway (1984); Funk & Rogge (2007 CSI)",
        construct="relationship_communication",
        scale_name="CSI-4",
        scale_points=7,  # varies 0-6 composite
        mean=4.5,
        sd=1.30,
        skewness=-0.4,
        sample_type="general",
        n_participants=3000,
    ),

    "jealousy_mrs": ConstructNorm(
        source="Pfeiffer & Wong (1989)",
        construct="romantic_jealousy",
        scale_name="MJS",
        scale_points=7,  # 1-7
        mean=3.2,
        sd=1.10,
        skewness=0.4,
        sample_type="general",
        n_participants=2500,
        moderators={
            "gender": {"male": 3.0, "female": 3.4},
        },
    ),

    "romantic_love_pls": ConstructNorm(
        source="Hatfield & Sprecher (1986); Sprecher & Fehr (2005)",
        construct="passionate_love",
        scale_name="PLS",
        scale_points=9,  # 1-9
        mean=6.5,
        sd=1.60,
        skewness=-0.5,
        sample_type="general",
        n_participants=3000,
    ),

    "compassion_scale": ConstructNorm(
        source="Pommier et al. (2020); Sprecher & Fehr (2005 compassionate love)",
        construct="compassion",
        scale_name="CS",
        scale_points=5,  # 1-5
        mean=3.8,
        sd=0.60,
        skewness=-0.2,
        sample_type="general",
        n_participants=2500,
        moderators={
            "gender": {"male": 3.6, "female": 4.0},
        },
    ),

    "assertiveness_ras_rathus": ConstructNorm(
        source="Rathus (1973); Thompson & Berenbaum (2011 norms)",
        construct="assertiveness",
        scale_name="RAS",
        scale_points=6,  # -3 to +3 (rescaled 1-6)
        mean=3.3,
        sd=0.80,
        sample_type="general",
        n_participants=3000,
        moderators={
            "gender": {"male": 3.5, "female": 3.1},
        },
    ),

    "social_skills_ssrs": ConstructNorm(
        source="Gresham & Elliott (1990; 2008 SSIS)",
        construct="social_skills",
        scale_name="SSIS",
        scale_points=4,  # 0-3
        mean=2.1,
        sd=0.50,
        sample_type="general",
        n_participants=4000,
        moderators={
            "gender": {"male": 2.0, "female": 2.2},
        },
    ),

    "emotional_intelligence_teiq": ConstructNorm(
        source="Petrides & Furnham (2003); Petrides (2009 TEIQue)",
        construct="trait_emotional_intelligence",
        scale_name="TEIQue",
        scale_points=7,  # 1-7
        mean=4.8,
        sd=0.80,
        sample_type="general",
        n_participants=5000,
        moderators={
            "gender": {"male": 4.7, "female": 4.9},
            "age": {"18_25": 4.5, "26_40": 4.8, "41_60": 5.0, "61_plus": 5.1},
        },
    ),

    "empathy_teq": ConstructNorm(
        source="Spreng et al. (2009)",
        construct="empathy",
        scale_name="TEQ",
        scale_points=5,  # 0-4
        mean=2.9,
        sd=0.55,
        sample_type="general",
        n_participants=2000,
        moderators={
            "gender": {"male": 2.7, "female": 3.1},
        },
    ),

    "cooperation_svo": ConstructNorm(
        source="Murphy et al. (2011 SVO Slider); Van Lange et al. (1997)",
        construct="prosocial_orientation",
        scale_name="SVO Slider",
        scale_points=9,  # Angle: -16.26 to 61.39 degrees
        mean=3.5,  # rescaled: ~57% prosocial, ~30% individualist, ~13% competitive
        sd=1.80,
        sample_type="general",
        n_participants=4000,
    ),

    "prosocial_tendency_ptm": ConstructNorm(
        source="Carlo & Randall (2002); PTM-R",
        construct="prosocial_tendencies",
        scale_name="PTM-R",
        scale_points=5,  # 1-5
        mean=3.5,
        sd=0.60,
        sample_type="general",
        n_participants=3000,
        moderators={
            "gender": {"male": 3.3, "female": 3.7},
        },
    ),

    "helping_behavior_hbs": ConstructNorm(
        source="Amato (1990); Levine et al. (2005 cross-cultural)",
        construct="helping_intention",
        scale_name="HBS",
        scale_points=7,  # 1-7
        mean=5.0,
        sd=1.10,
        skewness=-0.3,
        sample_type="general",
        n_participants=3000,
    ),

    # =========================================================================
    # WORK & ORGANIZATIONAL (~10 entries)
    # =========================================================================

    "job_satisfaction_msq": ConstructNorm(
        source="Weiss et al. (1967 MSQ); Judge et al. (2001 meta)",
        construct="job_satisfaction",
        scale_name="MSQ-Short",
        scale_points=5,  # 1-5
        mean=3.5,
        sd=0.75,
        skewness=-0.3,
        sample_type="general",
        n_participants=8000,
        moderators={
            "age": {"18_25": 3.3, "26_40": 3.4, "41_60": 3.6, "61_plus": 3.8},
        },
    ),

    "organizational_commitment_ocq": ConstructNorm(
        source="Meyer & Allen (1991); Meyer et al. (2002 meta)",
        construct="affective_organizational_commitment",
        scale_name="ACS",
        scale_points=7,  # 1-7
        mean=4.1,
        sd=1.30,
        sample_type="general",
        n_participants=10000,
    ),

    "leader_member_exchange_lmx": ConstructNorm(
        source="Graen & Uhl-Bien (1995); LMX-7",
        construct="leader_member_exchange",
        scale_name="LMX-7",
        scale_points=5,  # 1-5
        mean=3.5,
        sd=0.80,
        sample_type="general",
        n_participants=5000,
    ),

    "psychological_capital_psycap": ConstructNorm(
        source="Luthans et al. (2007); PCQ-24",
        construct="psychological_capital",
        scale_name="PCQ-24",
        scale_points=6,  # 1-6
        mean=4.2,
        sd=0.72,
        sample_type="general",
        n_participants=4000,
    ),

    "workplace_incivility_wis": ConstructNorm(
        source="Cortina et al. (2001); WIS-12",
        construct="workplace_incivility",
        scale_name="WIS",
        scale_points=5,  # 0-4 (never to many times)
        mean=0.75,
        sd=0.70,
        skewness=1.2,
        sample_type="general",
        n_participants=3500,
        moderators={
            "gender": {"male": 0.65, "female": 0.85},
        },
    ),

    "work_family_conflict_wfc": ConstructNorm(
        source="Netemeyer et al. (1996); WFC Scale",
        construct="work_family_conflict",
        scale_name="WFC",
        scale_points=7,  # 1-7
        mean=3.5,
        sd=1.30,
        sample_type="general",
        n_participants=4500,
        moderators={
            "gender": {"male": 3.3, "female": 3.7},
        },
    ),

    "role_ambiguity_ra": ConstructNorm(
        source="Rizzo et al. (1970); Tubre & Collins (2000 meta)",
        construct="role_ambiguity",
        scale_name="RA Scale",
        scale_points=7,  # 1-7
        mean=3.0,
        sd=1.20,
        sample_type="general",
        n_participants=5000,
    ),

    "role_conflict_rc": ConstructNorm(
        source="Rizzo et al. (1970); Tubre & Collins (2000 meta)",
        construct="role_conflict",
        scale_name="RC Scale",
        scale_points=7,  # 1-7
        mean=3.4,
        sd=1.25,
        sample_type="general",
        n_participants=5000,
    ),

    "turnover_intention_tis": ConstructNorm(
        source="Bothma & Roodt (2013); TIS-6",
        construct="turnover_intention",
        scale_name="TIS-6",
        scale_points=5,  # 1-5
        mean=2.6,
        sd=0.90,
        skewness=0.4,
        sample_type="general",
        n_participants=3500,
    ),

    "job_burnout_olbi": ConstructNorm(
        source="Demerouti et al. (2010); OLBI",
        construct="job_burnout",
        scale_name="OLBI",
        scale_points=4,  # 1-4
        mean=2.3,
        sd=0.50,
        skewness=0.3,
        sample_type="general",
        n_participants=3000,
    ),

    # =========================================================================
    # HEALTH & BODY (~10 entries)
    # =========================================================================

    "body_satisfaction_bass": ConstructNorm(
        source="Cash (2000); BASS / BAS-2",
        construct="body_appreciation",
        scale_name="BAS-2",
        scale_points=5,  # 1-5
        mean=3.5,
        sd=0.80,
        sample_type="general",
        n_participants=3000,
        moderators={
            "gender": {"male": 3.7, "female": 3.3},
        },
    ),

    "eating_attitudes_edeq": ConstructNorm(
        source="Fairburn & Beglin (1994; 2008 EDE-Q)",
        construct="eating_pathology",
        scale_name="EDE-Q",
        scale_points=7,  # 0-6
        mean=1.55,
        sd=1.30,
        skewness=1.0,
        sample_type="general",
        n_participants=5000,
        moderators={
            "gender": {"male": 1.10, "female": 1.90},
            "sample": {"non_clinical": 1.30, "clinical_eating_disorder": 4.00},
        },
    ),

    "exercise_self_efficacy_eses": ConstructNorm(
        source="Bandura (2006); Resnick & Jenkins (2000 ESES)",
        construct="exercise_self_efficacy",
        scale_name="ESES",
        scale_points=10,  # 0-10 (confidence percentage / 10)
        mean=5.5,
        sd=2.20,
        sample_type="general",
        n_participants=2500,
    ),

    "health_locus_of_control_mhlc": ConstructNorm(
        source="Wallston et al. (1978); MHLC",
        construct="health_locus_of_control_internal",
        scale_name="MHLC-I",
        scale_points=6,  # 1-6
        mean=4.2,
        sd=0.80,
        sample_type="general",
        n_participants=4000,
    ),

    "illness_perception_bipq": ConstructNorm(
        source="Broadbent et al. (2006); B-IPQ",
        construct="illness_threat_perception",
        scale_name="B-IPQ",
        scale_points=11,  # 0-10
        mean=4.5,
        sd=2.20,
        sample_type="general",
        n_participants=3500,
    ),

    "pain_catastrophizing_pcs": ConstructNorm(
        source="Sullivan et al. (1995)",
        construct="pain_catastrophizing",
        scale_name="PCS",
        scale_points=5,  # 0-4
        mean=1.30,
        sd=0.85,
        skewness=0.6,
        sample_type="general",
        n_participants=4000,
        moderators={
            "gender": {"male": 1.15, "female": 1.45},
            "sample": {"non_clinical": 1.10, "chronic_pain": 2.50},
        },
    ),

    "medication_adherence_mars": ConstructNorm(
        source="Horne & Weinman (2002); MARS-5",
        construct="medication_adherence",
        scale_name="MARS-5",
        scale_points=5,  # 1-5
        mean=4.2,
        sd=0.70,
        skewness=-1.0,
        sample_type="general",
        n_participants=3000,
    ),

    "health_literacy_hlq": ConstructNorm(
        source="Osborne et al. (2013); HLQ",
        construct="health_literacy",
        scale_name="HLQ",
        scale_points=5,  # 1-4 or 1-5 depending on scale
        mean=3.2,
        sd=0.55,
        sample_type="general",
        n_participants=3500,
    ),

    "physical_activity_ipaq": ConstructNorm(
        source="Craig et al. (2003); IPAQ",
        construct="physical_activity",
        scale_name="IPAQ-SF",
        scale_points=4,  # categorical: low/moderate/high + MET-minutes
        mean=2.1,
        sd=0.75,
        sample_type="general",
        n_participants=8000,
        moderators={
            "gender": {"male": 2.3, "female": 1.9},
            "age": {"18_25": 2.4, "26_40": 2.2, "41_60": 2.0, "61_plus": 1.7},
        },
    ),

    "body_mass_satisfaction_bms": ConstructNorm(
        source="Stunkard et al. (1983 FRS); Thompson & Gray (1995)",
        construct="body_mass_satisfaction",
        scale_name="FRS Discrepancy",
        scale_points=9,  # 1-9 figure rating (discrepancy ideal-current)
        mean=4.5,
        sd=1.20,
        sample_type="general",
        n_participants=5000,
        moderators={
            "gender": {"male": 4.3, "female": 4.7},
        },
    ),

    # =========================================================================
    # TECHNOLOGY & MEDIA (~11 entries)
    # =========================================================================

    "technology_acceptance_tam_pu": ConstructNorm(
        source="Davis (1989); Venkatesh & Davis (2000 TAM2)",
        construct="perceived_usefulness",
        scale_name="TAM-PU",
        scale_points=7,  # 1-7
        mean=4.8,
        sd=1.20,
        sample_type="general",
        n_participants=5000,
    ),

    "technology_acceptance_tam_peou": ConstructNorm(
        source="Davis (1989); Venkatesh & Davis (2000 TAM2)",
        construct="perceived_ease_of_use",
        scale_name="TAM-PEOU",
        scale_points=7,  # 1-7
        mean=4.9,
        sd=1.15,
        sample_type="general",
        n_participants=5000,
    ),

    "computer_self_efficacy_cse_tech": ConstructNorm(
        source="Compeau & Higgins (1995); Marakas et al. (1998)",
        construct="computer_self_efficacy",
        scale_name="CSE",
        scale_points=10,  # 1-10
        mean=6.8,
        sd=2.00,
        sample_type="general",
        n_participants=3000,
        moderators={
            "age": {"18_25": 7.5, "26_40": 7.0, "41_60": 6.2, "61_plus": 5.5},
        },
    ),

    "internet_addiction_iat": ConstructNorm(
        source="Young (1998); Widyanto & McMurran (2004 IAT norms)",
        construct="internet_addiction",
        scale_name="IAT",
        scale_points=5,  # 1-5 (0-5 in some versions)
        mean=2.0,
        sd=0.70,
        skewness=0.7,
        sample_type="general",
        n_participants=5000,
        moderators={
            "age": {"18_25": 2.4, "26_40": 2.0, "41_60": 1.7, "61_plus": 1.4},
        },
    ),

    "social_media_intensity_smi": ConstructNorm(
        source="Ellison et al. (2007 Facebook); adapted general SMI",
        construct="social_media_use_intensity",
        scale_name="SMI",
        scale_points=5,  # 1-5
        mean=3.2,
        sd=0.90,
        sample_type="general",
        n_participants=4000,
        moderators={
            "age": {"18_25": 3.8, "26_40": 3.2, "41_60": 2.7, "61_plus": 2.0},
        },
    ),

    "cyberbullying_cbv": ConstructNorm(
        source="Hinduja & Patchin (2008); Menesini et al. (2012 ECVS)",
        construct="cyberbullying_victimization",
        scale_name="ECVS",
        scale_points=5,  # 0-4 (never to always)
        mean=0.55,
        sd=0.60,
        skewness=1.8,
        sample_type="student",
        n_participants=4000,
        moderators={
            "age": {"12_15": 0.70, "16_18": 0.60, "18_25": 0.45},
        },
    ),

    "digital_literacy_dlq": ConstructNorm(
        source="Hargittai (2005); Ng (2012 DLQ)",
        construct="digital_literacy",
        scale_name="DLQ",
        scale_points=5,  # 1-5
        mean=3.6,
        sd=0.75,
        sample_type="general",
        n_participants=3000,
        moderators={
            "age": {"18_25": 4.0, "26_40": 3.7, "41_60": 3.3, "61_plus": 2.8},
        },
    ),

    "privacy_concern_iuipc": ConstructNorm(
        source="Malhotra et al. (2004); IUIPC",
        construct="information_privacy_concern",
        scale_name="IUIPC",
        scale_points=7,  # 1-7
        mean=5.0,
        sd=1.10,
        sample_type="general",
        n_participants=3500,
        moderators={
            "age": {"18_25": 4.6, "26_40": 5.0, "41_60": 5.3, "61_plus": 5.5},
        },
    ),

    "technology_anxiety_ta": ConstructNorm(
        source="Meuter et al. (2003); TRI (Parasuraman 2000)",
        construct="technology_anxiety",
        scale_name="TRI-TA",
        scale_points=5,  # 1-5
        mean=2.3,
        sd=0.85,
        sample_type="general",
        n_participants=3000,
        moderators={
            "age": {"18_25": 1.8, "26_40": 2.1, "41_60": 2.6, "61_plus": 3.1},
        },
    ),

    "ai_attitudes_gaais": ConstructNorm(
        source="Schepman & Rodway (2020; 2023 GAAIS)",
        construct="ai_attitudes",
        scale_name="GAAIS",
        scale_points=5,  # 1-5
        mean=3.2,
        sd=0.68,
        sample_type="general",
        n_participants=2500,
        moderators={
            "age": {"18_25": 3.4, "26_40": 3.3, "41_60": 3.0, "61_plus": 2.8},
        },
    ),

    "phone_addiction_spai": ConstructNorm(
        source="Lin et al. (2014 SPAI); Kwon et al. (2013 SAS)",
        construct="smartphone_addiction",
        scale_name="SAS-SV",
        scale_points=6,  # 1-6
        mean=2.8,
        sd=0.90,
        skewness=0.4,
        sample_type="general",
        n_participants=4000,
        moderators={
            "age": {"18_25": 3.3, "26_40": 2.8, "41_60": 2.3, "61_plus": 1.8},
        },
    ),

    # =========================================================================
    # CONSUMER & MARKETING (~10 entries)
    # =========================================================================

    "brand_loyalty_bl": ConstructNorm(
        source="Yoo & Donthu (2001); CBBE scale",
        construct="brand_loyalty",
        scale_name="BL Scale",
        scale_points=5,  # 1-5
        mean=3.2,
        sd=0.90,
        sample_type="general",
        n_participants=3000,
    ),

    "purchase_intention_pi": ConstructNorm(
        source="Dodds et al. (1991); Baker & Churchill (1977 compiled)",
        construct="purchase_intention",
        scale_name="PI Scale",
        scale_points=7,  # 1-7
        mean=3.8,
        sd=1.50,
        sample_type="general",
        n_participants=5000,
    ),

    "customer_satisfaction_acsi": ConstructNorm(
        source="Fornell et al. (1996); ACSI national norms",
        construct="customer_satisfaction",
        scale_name="ACSI",
        scale_points=10,  # 1-10
        mean=7.4,
        sd=1.70,
        skewness=-0.5,
        sample_type="general",
        n_participants=15000,
    ),

    "perceived_value_pv": ConstructNorm(
        source="Zeithaml (1988); Sweeney & Soutar (2001 PERVAL)",
        construct="perceived_value",
        scale_name="PERVAL",
        scale_points=7,  # 1-7
        mean=4.5,
        sd=1.15,
        sample_type="general",
        n_participants=3500,
    ),

    "brand_trust_bt": ConstructNorm(
        source="Delgado-Ballester (2004); BTS",
        construct="brand_trust",
        scale_name="BTS",
        scale_points=5,  # 1-5
        mean=3.4,
        sd=0.80,
        sample_type="general",
        n_participants=3000,
    ),

    "price_sensitivity_psp": ConstructNorm(
        source="Goldsmith et al. (2005); Lichtenstein et al. (1993)",
        construct="price_sensitivity",
        scale_name="PSP",
        scale_points=5,  # 1-5
        mean=3.3,
        sd=0.80,
        sample_type="general",
        n_participants=3000,
    ),

    "impulse_buying_ibs": ConstructNorm(
        source="Rook & Fisher (1995); Verplanken & Herabadi (2001)",
        construct="impulse_buying_tendency",
        scale_name="IBS",
        scale_points=5,  # 1-5
        mean=2.7,
        sd=0.75,
        skewness=0.3,
        sample_type="general",
        n_participants=3000,
        moderators={
            "gender": {"male": 2.5, "female": 2.9},
            "age": {"18_25": 3.0, "26_40": 2.7, "41_60": 2.4, "61_plus": 2.2},
        },
    ),

    "store_atmosphere_sa": ConstructNorm(
        source="Baker et al. (2002); Turley & Milliman (2000 review)",
        construct="store_atmosphere_perception",
        scale_name="SA Scale",
        scale_points=7,  # 1-7
        mean=4.5,
        sd=1.20,
        sample_type="general",
        n_participants=2500,
    ),

    "ad_attitude_aad": ConstructNorm(
        source="MacKenzie et al. (1986); Mitchell & Olson (1981)",
        construct="attitude_toward_ad",
        scale_name="Aad",
        scale_points=7,  # 1-7 (semantic differential)
        mean=4.2,
        sd=1.30,
        sample_type="general",
        n_participants=4000,
    ),

    "wom_intention_wom": ConstructNorm(
        source="Zeithaml et al. (1996); Goyette et al. (2010 eWOM)",
        construct="word_of_mouth_intention",
        scale_name="WOM",
        scale_points=7,  # 1-7
        mean=4.3,
        sd=1.40,
        sample_type="general",
        n_participants=3500,
    ),

    # =========================================================================
    # ACADEMIC & EDUCATION (~10 entries)
    # =========================================================================

    "academic_self_efficacy_ase": ConstructNorm(
        source="Bandura (1997); Chemers et al. (2001); Honicke & Broadbent (2016 meta)",
        construct="academic_self_efficacy",
        scale_name="ASE",
        scale_points=7,  # 1-7
        mean=4.8,
        sd=1.05,
        sample_type="student",
        n_participants=5000,
    ),

    "test_anxiety_rta": ConstructNorm(
        source="Benson & El-Zahhar (1994); RTT norms",
        construct="test_anxiety_worry",
        scale_name="RTT",
        scale_points=4,  # 1-4
        mean=2.3,
        sd=0.65,
        skewness=0.3,
        sample_type="student",
        n_participants=4000,
        moderators={
            "gender": {"male": 2.1, "female": 2.5},
        },
    ),

    "deep_learning_approach_rspq": ConstructNorm(
        source="Biggs et al. (2001); R-SPQ-2F",
        construct="deep_learning_approach",
        scale_name="R-SPQ-2F Deep",
        scale_points=5,  # 1-5
        mean=3.2,
        sd=0.62,
        sample_type="student",
        n_participants=3500,
    ),

    "surface_learning_approach_rspq": ConstructNorm(
        source="Biggs et al. (2001); R-SPQ-2F",
        construct="surface_learning_approach",
        scale_name="R-SPQ-2F Surface",
        scale_points=5,  # 1-5
        mean=2.7,
        sd=0.60,
        sample_type="student",
        n_participants=3500,
    ),

    "school_belonging_pssm": ConstructNorm(
        source="Goodenow (1993); PSSM",
        construct="school_belonging",
        scale_name="PSSM",
        scale_points=5,  # 1-5
        mean=3.6,
        sd=0.70,
        skewness=-0.3,
        sample_type="student",
        n_participants=4000,
    ),

    "academic_engagement_uwes_s": ConstructNorm(
        source="Schaufeli et al. (2002); UWES-S",
        construct="academic_engagement",
        scale_name="UWES-S",
        scale_points=7,  # 0-6
        mean=3.4,
        sd=1.10,
        sample_type="student",
        n_participants=5000,
    ),

    "teacher_self_efficacy_tses": ConstructNorm(
        source="Tschannen-Moran & Hoy (2001); TSES",
        construct="teacher_self_efficacy",
        scale_name="TSES",
        scale_points=9,  # 1-9
        mean=6.5,
        sd=1.20,
        sample_type="general",
        n_participants=3000,
    ),

    "educational_aspirations_ea": ConstructNorm(
        source="Sewell & Shah (1968); compiled norms",
        construct="educational_aspirations",
        scale_name="EA Scale",
        scale_points=7,  # 1-7
        mean=5.2,
        sd=1.30,
        sample_type="student",
        n_participants=4000,
    ),

    "reading_motivation_mrq": ConstructNorm(
        source="Wigfield & Guthrie (1997); MRQ",
        construct="reading_motivation",
        scale_name="MRQ",
        scale_points=4,  # 1-4
        mean=2.8,
        sd=0.55,
        sample_type="student",
        n_participants=3000,
        moderators={
            "gender": {"male": 2.6, "female": 3.0},
        },
    ),

    "math_anxiety_mars": ConstructNorm(
        source="Richardson & Suinn (1972); Hopko et al. (2003 AMAS)",
        construct="math_anxiety",
        scale_name="AMAS",
        scale_points=5,  # 1-5
        mean=2.5,
        sd=0.85,
        skewness=0.4,
        sample_type="student",
        n_participants=4000,
        moderators={
            "gender": {"male": 2.3, "female": 2.7},
        },
    ),

    # =========================================================================
    # ADDITIONAL PERSONALITY & INDIVIDUAL DIFFERENCES (~15 entries)
    # =========================================================================

    "dark_triad_machiavellianism": ConstructNorm(
        source="Jones & Paulhus (2014); SD3",
        construct="machiavellianism",
        scale_name="SD3-Mach",
        scale_points=5,  # 1-5
        mean=2.9,
        sd=0.70,
        sample_type="general",
        n_participants=4000,
        moderators={
            "gender": {"male": 3.1, "female": 2.7},
        },
    ),

    "dark_triad_psychopathy": ConstructNorm(
        source="Jones & Paulhus (2014); SD3",
        construct="psychopathy",
        scale_name="SD3-Psych",
        scale_points=5,  # 1-5
        mean=2.1,
        sd=0.65,
        skewness=0.5,
        sample_type="general",
        n_participants=4000,
        moderators={
            "gender": {"male": 2.3, "female": 1.9},
        },
    ),

    "perfectionism_fmps": ConstructNorm(
        source="Frost et al. (1990); FMPS",
        construct="perfectionism",
        scale_name="FMPS",
        scale_points=5,  # 1-5
        mean=3.0,
        sd=0.65,
        sample_type="general",
        n_participants=3500,
        moderators={
            "gender": {"male": 2.9, "female": 3.1},
        },
    ),

    "sensation_seeking_bsss": ConstructNorm(
        source="Hoyle et al. (2002); BSSS-8",
        construct="sensation_seeking",
        scale_name="BSSS",
        scale_points=5,  # 1-5
        mean=3.0,
        sd=0.80,
        sample_type="general",
        n_participants=4000,
        moderators={
            "gender": {"male": 3.2, "female": 2.8},
            "age": {"18_25": 3.4, "26_40": 3.0, "41_60": 2.7, "61_plus": 2.3},
        },
    ),

    "alexithymia_tas": ConstructNorm(
        source="Bagby et al. (1994); TAS-20",
        construct="alexithymia",
        scale_name="TAS-20",
        scale_points=5,  # 1-5
        mean=2.6,
        sd=0.62,
        skewness=0.3,
        sample_type="general",
        n_participants=5000,
        moderators={
            "gender": {"male": 2.7, "female": 2.5},
        },
    ),

    "intolerance_of_uncertainty_ius": ConstructNorm(
        source="Freeston et al. (1994); Carleton et al. (2007 IUS-12)",
        construct="intolerance_of_uncertainty",
        scale_name="IUS-12",
        scale_points=5,  # 1-5
        mean=2.8,
        sd=0.80,
        skewness=0.3,
        sample_type="general",
        n_participants=3500,
        moderators={
            "gender": {"male": 2.6, "female": 3.0},
        },
    ),

    "trait_curiosity_cei": ConstructNorm(
        source="Kashdan et al. (2009); CEI-II",
        construct="trait_curiosity",
        scale_name="CEI-II",
        scale_points=5,  # 1-5
        mean=3.5,
        sd=0.65,
        sample_type="general",
        n_participants=2500,
    ),

    "authenticity_ais": ConstructNorm(
        source="Wood et al. (2008); AIS",
        construct="authenticity",
        scale_name="AIS",
        scale_points=7,  # 1-7
        mean=5.0,
        sd=0.85,
        sample_type="general",
        n_participants=2000,
    ),

    "emotional_expressivity_bes": ConstructNorm(
        source="Gross & John (1997); BEQ",
        construct="emotional_expressivity",
        scale_name="BEQ",
        scale_points=7,  # 1-7
        mean=4.5,
        sd=0.90,
        sample_type="general",
        n_participants=3000,
        moderators={
            "gender": {"male": 4.1, "female": 4.9},
        },
    ),

    "boredom_proneness_bps": ConstructNorm(
        source="Farmer & Sundberg (1986); Struk et al. (2017 SBPS)",
        construct="boredom_proneness",
        scale_name="SBPS",
        scale_points=7,  # 1-7
        mean=3.4,
        sd=1.10,
        sample_type="general",
        n_participants=3000,
        moderators={
            "age": {"18_25": 3.8, "26_40": 3.4, "41_60": 3.0, "61_plus": 2.7},
        },
    ),

    "ambivalence_over_emotional_expression": ConstructNorm(
        source="King & Emmons (1990); AEQ",
        construct="emotional_ambivalence",
        scale_name="AEQ",
        scale_points=5,  # 1-5
        mean=2.8,
        sd=0.72,
        sample_type="general",
        n_participants=2500,
    ),

    "trait_anger_staxi2": ConstructNorm(
        source="Spielberger (1999); STAXI-2 Trait Anger",
        construct="trait_anger",
        scale_name="STAXI-2 TA",
        scale_points=4,  # 1-4
        mean=1.75,
        sd=0.52,
        skewness=1.0,
        sample_type="general",
        n_participants=4000,
        moderators={
            "gender": {"male": 1.80, "female": 1.70},
        },
    ),

    "emotion_dysregulation_ders": ConstructNorm(
        source="Gratz & Roemer (2004); DERS",
        construct="emotion_dysregulation",
        scale_name="DERS",
        scale_points=5,  # 1-5
        mean=2.2,
        sd=0.60,
        skewness=0.5,
        sample_type="general",
        n_participants=4000,
        moderators={
            "gender": {"male": 2.1, "female": 2.3},
            "sample": {"non_clinical": 2.1, "clinical": 3.0},
        },
    ),

    "social_desirability_mcsds": ConstructNorm(
        source="Crowne & Marlowe (1960); Reynolds (1982 MC-SDS-13)",
        construct="social_desirability",
        scale_name="MC-SDS",
        scale_points=2,  # 0/1 (true/false)
        mean=0.55,
        sd=0.20,
        sample_type="general",
        n_participants=6000,
        moderators={
            "age": {"18_25": 0.48, "26_40": 0.53, "41_60": 0.58, "61_plus": 0.65},
        },
    ),

    "nationalism_np": ConstructNorm(
        source="Kosterman & Feshbach (1989); NP Scale",
        construct="nationalism",
        scale_name="NP",
        scale_points=5,  # 1-5
        mean=2.9,
        sd=0.85,
        sample_type="general",
        n_participants=3000,
    ),

    # =========================================================================
    # STRESS, COPING & RESILIENCE (~8 entries)
    # =========================================================================

    "coping_brief_cope_active": ConstructNorm(
        source="Carver (1997); Brief COPE",
        construct="active_coping",
        scale_name="Brief COPE Active",
        scale_points=4,  # 1-4
        mean=2.8,
        sd=0.70,
        sample_type="general",
        n_participants=5000,
    ),

    "coping_brief_cope_avoidant": ConstructNorm(
        source="Carver (1997); Brief COPE",
        construct="avoidant_coping",
        scale_name="Brief COPE Avoidant",
        scale_points=4,  # 1-4
        mean=1.7,
        sd=0.65,
        skewness=0.8,
        sample_type="general",
        n_participants=5000,
    ),

    "post_traumatic_growth_ptgi": ConstructNorm(
        source="Tedeschi & Calhoun (1996); PTGI",
        construct="post_traumatic_growth",
        scale_name="PTGI",
        scale_points=6,  # 0-5
        mean=2.8,
        sd=1.20,
        sample_type="general",
        n_participants=3000,
    ),

    "hardiness_drs": ConstructNorm(
        source="Bartone (2007); DRS-15",
        construct="psychological_hardiness",
        scale_name="DRS-15",
        scale_points=4,  # 0-3
        mean=2.0,
        sd=0.50,
        sample_type="general",
        n_participants=2500,
    ),

    "emotion_focused_coping": ConstructNorm(
        source="Folkman & Lazarus (1988); WCQ",
        construct="emotion_focused_coping",
        scale_name="WCQ-EF",
        scale_points=4,  # 0-3
        mean=1.5,
        sd=0.55,
        sample_type="general",
        n_participants=3000,
        moderators={
            "gender": {"male": 1.3, "female": 1.7},
        },
    ),

    "social_support_coping": ConstructNorm(
        source="Folkman & Lazarus (1988); WCQ",
        construct="social_support_seeking",
        scale_name="WCQ-SS",
        scale_points=4,  # 0-3
        mean=1.6,
        sd=0.60,
        sample_type="general",
        n_participants=3000,
        moderators={
            "gender": {"male": 1.4, "female": 1.8},
        },
    ),

    "burnout_depersonalization_mbi": ConstructNorm(
        source="Maslach & Jackson (1981); MBI Depersonalization",
        construct="depersonalization",
        scale_name="MBI-DP",
        scale_points=7,  # 0-6
        mean=2.0,
        sd=1.40,
        skewness=0.6,
        sample_type="general",
        n_participants=8000,
        moderators={
            "gender": {"male": 2.2, "female": 1.8},
        },
    ),

    "burnout_personal_accomplishment_mbi": ConstructNorm(
        source="Maslach & Jackson (1981); MBI Personal Accomplishment",
        construct="personal_accomplishment",
        scale_name="MBI-PA",
        scale_points=7,  # 0-6
        mean=4.4,
        sd=1.10,
        skewness=-0.3,
        sample_type="general",
        n_participants=8000,
    ),

}


# =============================================================================
# CROSS-CULTURAL CALIBRATIONS
# =============================================================================
# Based on Henrich et al. (2005, 2010): "The weirdest people in the world?"
# Most behavioral science uses WEIRD samples. These factors adjust.

CULTURAL_ADJUSTMENTS: Dict[str, CulturalAdjustment] = {

    "dictator_giving_cross_cultural": CulturalAdjustment(
        source="Henrich et al. (2005, 2010) — 15 small-scale societies",
        culture="cross_cultural",
        construct="dictator_giving",
        adjustment_factor=1.0,  # Base
        n_cultures=15,
        n_participants=2000,
        notes="Offers range from 15% (Machiguenga) to 58% (Lamelara). "
              "Market integration correlates with higher offers (r = 0.50)."
    ),

    "western_baseline": CulturalAdjustment(
        source="Henrich et al. (2010)",
        culture="western",
        construct="all",
        adjustment_factor=1.0,
        notes="Baseline. All calibrations use Western norms by default."
    ),

    "east_asian_response_style": CulturalAdjustment(
        source="Chen et al. (1995); Heine et al. (2002)",
        culture="east_asian",
        construct="response_style",
        adjustment_factor=0.85,  # Midpoint preference (less extreme)
        adjustment_additive=-0.3,  # Lower self-enhancement
        n_cultures=5,
        n_participants=8000,
        notes="Lower extremity, higher midpoint use, less self-enhancement."
    ),

    "east_asian_self_esteem": CulturalAdjustment(
        source="Schmitt & Allik (2005)",
        culture="east_asian",
        construct="self_esteem",
        adjustment_factor=0.85,
        adjustment_additive=-0.5,
        notes="Self-esteem 0.5 points lower on 4-pt scale in East Asian samples."
    ),

    "latin_american_response_style": CulturalAdjustment(
        source="Marín et al. (1992); Johnson et al. (2005)",
        culture="latin_american",
        construct="response_style",
        adjustment_factor=1.15,  # Higher acquiescence and extreme responding
        n_participants=5000,
        notes="Higher acquiescence bias, more extreme response style."
    ),

    "collectivist_conformity": CulturalAdjustment(
        source="Bond & Smith (1996)",
        culture="collectivist",
        construct="conformity",
        adjustment_factor=1.30,
        n_cultures=17,
        n_participants=25000,
        notes="Conformity ≈ 30% higher in collectivist cultures."
    ),

    "collectivist_cooperation": CulturalAdjustment(
        source="Balliet et al. (2014)",
        culture="collectivist",
        construct="ingroup_cooperation",
        adjustment_factor=1.20,
        notes="Ingroup cooperation higher, outgroup discrimination also higher."
    ),

    "individualist_social_desirability": CulturalAdjustment(
        source="Lalwani et al. (2006)",
        culture="western",
        construct="social_desirability",
        adjustment_factor=1.0,
        notes="Baseline. Self-deception component similar across cultures."
    ),

    "east_asian_social_desirability": CulturalAdjustment(
        source="Lalwani et al. (2006)",
        culture="east_asian",
        construct="social_desirability",
        adjustment_factor=1.10,
        notes="Impression management slightly higher in collectivist contexts."
    ),

    "dictator_giving_small_scale": CulturalAdjustment(
        source="Henrich et al. (2005)",
        culture="small_scale_society",
        construct="dictator_giving",
        adjustment_factor=1.50,
        notes="Mean giving 50%+ in market-integrated small-scale societies."
    ),

    "mturk_response_quality": CulturalAdjustment(
        source="Hauser & Schwarz (2016); Chmielewski & Kucker (2020)",
        culture="mturk",
        construct="data_quality",
        adjustment_factor=0.95,
        notes="MTurk data quality declining over time. More satisficing."
    ),

    "student_vs_nonstudent": CulturalAdjustment(
        source="Peterson (2001 meta); Hanel & Vione (2016)",
        culture="student",
        construct="effect_magnitude",
        adjustment_factor=0.90,
        notes="Student samples show ~10% smaller effects than non-student in consumer studies."
    ),
}


# =============================================================================
# RESPONSE TIME NORMS
# =============================================================================
# Based on Yan & Tourangeau (2008), Callegaro et al. (2015),
# Malhotra (2008), Zhang & Conrad (2014).

RESPONSE_TIME_NORMS: Dict[str, ResponseTimeNorm] = {

    "likert_engaged": ResponseTimeNorm(
        source="Yan & Tourangeau (2008)",
        item_type="likert",
        engagement_level="engaged",
        mean_ms=4000,
        sd_ms=1200,
        distribution="ex_gaussian",
        ex_gaussian_mu=3200,
        ex_gaussian_sigma=600,
        ex_gaussian_tau=800,
        notes="Engaged: 3-5s per item. Ex-Gaussian captures right-skew (occasional long pauses)."
    ),

    "likert_satisficing": ResponseTimeNorm(
        source="Yan & Tourangeau (2008)",
        item_type="likert",
        engagement_level="satisficing",
        mean_ms=1500,
        sd_ms=600,
        distribution="ex_gaussian",
        ex_gaussian_mu=1200,
        ex_gaussian_sigma=300,
        ex_gaussian_tau=300,
    ),

    "likert_careless": ResponseTimeNorm(
        source="Meade & Craig (2012)",
        item_type="likert",
        engagement_level="careless",
        mean_ms=600,
        sd_ms=300,
        distribution="ex_gaussian",
        ex_gaussian_mu=450,
        ex_gaussian_sigma=150,
        ex_gaussian_tau=150,
    ),

    "slider_engaged": ResponseTimeNorm(
        source="Funke et al. (2011)",
        item_type="slider",
        engagement_level="engaged",
        mean_ms=5500,
        sd_ms=1800,
        distribution="ex_gaussian",
        ex_gaussian_mu=4200,
        ex_gaussian_sigma=800,
        ex_gaussian_tau=1300,
        notes="Sliders take longer than Likert due to motor precision demands."
    ),

    "slider_satisficing": ResponseTimeNorm(
        source="Funke et al. (2011)",
        item_type="slider",
        engagement_level="satisficing",
        mean_ms=2500,
        sd_ms=800,
        distribution="lognormal",
    ),

    "open_ended_engaged": ResponseTimeNorm(
        source="Zhang & Conrad (2014)",
        item_type="open_ended",
        engagement_level="engaged",
        mean_ms=35000,
        sd_ms=15000,
        distribution="lognormal",
        notes="20-45s per OE question for engaged respondents."
    ),

    "open_ended_satisficing": ResponseTimeNorm(
        source="Zhang & Conrad (2014)",
        item_type="open_ended",
        engagement_level="satisficing",
        mean_ms=6000,
        sd_ms=3000,
        distribution="lognormal",
    ),

    "open_ended_careless": ResponseTimeNorm(
        source="Meade & Craig (2012)",
        item_type="open_ended",
        engagement_level="careless",
        mean_ms=2000,
        sd_ms=1000,
        distribution="lognormal",
    ),

    "iat_trial_congruent": ResponseTimeNorm(
        source="Greenwald et al. (2003)",
        item_type="iat",
        engagement_level="congruent",
        mean_ms=650,
        sd_ms=150,
        distribution="ex_gaussian",
        ex_gaussian_mu=550,
        ex_gaussian_sigma=80,
        ex_gaussian_tau=100,
        notes="IAT congruent trials: ~650ms. D-score = (incongruent - congruent) / pooled SD."
    ),

    "iat_trial_incongruent": ResponseTimeNorm(
        source="Greenwald et al. (2003)",
        item_type="iat",
        engagement_level="incongruent",
        mean_ms=800,
        sd_ms=200,
        distribution="ex_gaussian",
        ex_gaussian_mu=680,
        ex_gaussian_sigma=100,
        ex_gaussian_tau=120,
        notes="IAT incongruent trials: ~800ms."
    ),
}


# =============================================================================
# ORDER / FATIGUE / LEARNING EFFECTS
# =============================================================================

ORDER_EFFECTS: Dict[str, OrderEffect] = {

    "survey_fatigue": OrderEffect(
        source="Galesic & Bosnjak (2009); Hoerger (2010)",
        effect_type="fatigue",
        magnitude_per_item=-0.003,  # -0.003 SD per item after onset
        onset_item=20,              # Fatigue begins after ~20 items
        plateau_item=80,            # Plateaus around 80 items
        affects="variance_and_mean",
        notes="Items after #20: mean shifts toward midpoint, variance decreases. "
              "Response time also decreases (speeding). 0.3% SD degradation per item."
    ),

    "survey_learning": OrderEffect(
        source="Tourangeau et al. (2000); Knowles (1988)",
        effect_type="learning",
        magnitude_per_item=0.002,   # +0.002 SD improvement in consistency
        onset_item=1,
        plateau_item=15,            # Learned the scale format by item 15
        affects="variance",
        notes="Variance decreases as respondents learn scale format. "
              "Consistency improves for first ~15 items."
    ),

    "primacy_effect": OrderEffect(
        source="Krosnick & Alwin (1987)",
        effect_type="primacy",
        magnitude_per_item=0.05,    # First options chosen 5% more
        onset_item=1,
        plateau_item=1,
        affects="mean",
        notes="In categorical scales, first listed options chosen disproportionately."
    ),

    "recency_effect": OrderEffect(
        source="Krosnick & Alwin (1987)",
        effect_type="recency",
        magnitude_per_item=0.04,    # Last options chosen 4% more in oral surveys
        onset_item=1,
        plateau_item=1,
        affects="mean",
        notes="In oral/long lists, last options favored. Written → primacy dominant."
    ),

    "carryover_effect": OrderEffect(
        source="Schuman & Presser (1981); Tourangeau et al. (2000)",
        effect_type="carryover",
        magnitude_per_item=0.10,    # r ≈ 0.10 between adjacent items beyond trait correlation
        onset_item=1,
        plateau_item=3,
        affects="correlation",
        notes="Adjacent items more correlated than distant items, beyond trait correlation. "
              "Priming from prior question context."
    ),

    "straight_lining_acceleration": OrderEffect(
        source="Zhang & Conrad (2014); Kim et al. (2019)",
        effect_type="straight_lining",
        magnitude_per_item=0.005,   # 0.5% increase in straight-line probability per item
        onset_item=30,
        plateau_item=100,
        affects="consistency",
        notes="Straight-lining probability increases as survey length grows. "
              "After item 30, each additional item adds ~0.5% straight-line risk."
    ),
}


# =============================================================================
# LOOKUP / QUERY FUNCTIONS
# =============================================================================

def get_meta_analytic_effect(
    key: str,
    moderator_key: str = "",
    moderator_value: str = "",
    rng: Optional[random.Random] = None,
    sample_from_ci: bool = False,
) -> float:
    """Look up a meta-analytic effect size, optionally moderated.

    Args:
        key: Database key (e.g., "dictator_giving", "loss_aversion")
        moderator_key: Optional moderator dimension (e.g., "stakes", "culture")
        moderator_value: Moderator level (e.g., "high", "student")
        rng: Random number generator for sampling
        sample_from_ci: If True, sample from CI/heterogeneity distribution

    Returns:
        Cohen's d effect size (point estimate or sampled)
    """
    entry = META_ANALYTIC_DB.get(key)
    if entry is None:
        return 0.0

    if moderator_key and moderator_value:
        base = entry.get_moderated_effect(moderator_key, moderator_value)
    else:
        base = entry.effect_d

    if sample_from_ci and rng:
        # Sample around the base effect using heterogeneity
        tau = entry.heterogeneity_tau or (entry.ci_95[1] - entry.ci_95[0]) / 4
        return rng.gauss(base, tau)

    return base


def get_game_calibration(
    game_type: str,
    variant: str = "standard",
    moderator_key: str = "",
    moderator_value: str = "",
) -> Optional[GameCalibration]:
    """Look up game calibration by type and variant."""
    key = f"{game_type}_{variant}"
    cal = GAME_CALIBRATIONS.get(key)
    if cal is None:
        # Try just the game type with standard
        cal = GAME_CALIBRATIONS.get(f"{game_type}_standard")
    return cal


def get_construct_norm(
    construct: str,
    target_scale_points: int = 7,
    culture: str = "western",
    sample_type: str = "general",
) -> Optional[Dict[str, float]]:
    """Look up construct norms and rescale to target scale.

    Returns dict with 'mean', 'sd', 'skewness' rescaled to target_scale_points.
    """
    norm = CONSTRUCT_NORMS.get(construct)
    if norm is None:
        # Try prefix match
        for k, v in CONSTRUCT_NORMS.items():
            if construct in k or construct in v.construct:
                norm = v
                break
    if norm is None:
        return None

    # Rescale from published scale to target scale
    published_max = float(norm.scale_points)
    target_max = float(target_scale_points)
    scale_factor = target_max / published_max

    mean = norm.mean * scale_factor
    sd = norm.sd * scale_factor

    # Apply cultural moderator if available
    if culture != "western" and "culture" in norm.moderators:
        cult_mod = norm.moderators["culture"]
        if culture in cult_mod:
            mean = cult_mod[culture] * scale_factor

    # Apply sample type moderator
    if sample_type != "general" and "sample" in norm.moderators:
        samp_mod = norm.moderators["sample"]
        if sample_type in samp_mod:
            mean = samp_mod[sample_type] * scale_factor

    return {
        "mean": mean,
        "sd": sd,
        "skewness": norm.skewness,
        "kurtosis": norm.kurtosis,
    }


def get_cultural_adjustment(
    culture: str,
    construct: str = "all",
) -> Tuple[float, float]:
    """Get multiplicative and additive cultural adjustments.

    Returns (multiplicative_factor, additive_adjustment).
    """
    # Try exact match first
    for key, adj in CULTURAL_ADJUSTMENTS.items():
        if adj.culture == culture and (adj.construct == construct or adj.construct == "all"):
            return adj.adjustment_factor, adj.adjustment_additive

    # Try culture-only match
    for key, adj in CULTURAL_ADJUSTMENTS.items():
        if adj.culture == culture and adj.construct == "response_style":
            return adj.adjustment_factor, adj.adjustment_additive

    return 1.0, 0.0  # No adjustment


def get_response_time_norm(
    item_type: str = "likert",
    engagement_level: str = "engaged",
) -> Optional[ResponseTimeNorm]:
    """Look up response time norms for a given item type and engagement level."""
    key = f"{item_type}_{engagement_level}"
    return RESPONSE_TIME_NORMS.get(key)


def get_order_effect(effect_type: str) -> Optional[OrderEffect]:
    """Look up order effect parameters."""
    for key, eff in ORDER_EFFECTS.items():
        if eff.effect_type == effect_type:
            return eff
    return None


def compute_fatigue_adjustment(
    item_number: int,
    total_items: int,
) -> Dict[str, float]:
    """Compute fatigue-based adjustments for a given item position.

    Returns dict with 'mean_shift', 'variance_multiplier', 'straight_line_boost'.
    """
    fatigue = get_order_effect("fatigue")
    learning = get_order_effect("learning")
    straight = get_order_effect("straight_lining")

    result = {
        "mean_shift": 0.0,
        "variance_multiplier": 1.0,
        "straight_line_boost": 0.0,
    }

    if fatigue and item_number > fatigue.onset_item:
        items_past = min(item_number - fatigue.onset_item,
                        fatigue.plateau_item - fatigue.onset_item)
        result["mean_shift"] = items_past * fatigue.magnitude_per_item
        result["variance_multiplier"] = max(0.85, 1.0 + items_past * fatigue.magnitude_per_item * 0.5)

    if learning and item_number <= learning.plateau_item:
        items_in = min(item_number, learning.plateau_item)
        result["variance_multiplier"] *= max(0.90, 1.0 - items_in * learning.magnitude_per_item)

    if straight and item_number > straight.onset_item:
        items_past = min(item_number - straight.onset_item,
                        straight.plateau_item - straight.onset_item)
        result["straight_line_boost"] = items_past * straight.magnitude_per_item

    return result


def get_knowledge_base_summary() -> Dict[str, Any]:
    """Return a summary of the knowledge base contents for audit/reporting."""
    return {
        "meta_analytic_effects": len(META_ANALYTIC_DB),
        "game_calibrations": len(GAME_CALIBRATIONS),
        "construct_norms": len(CONSTRUCT_NORMS),
        "cultural_adjustments": len(CULTURAL_ADJUSTMENTS),
        "response_time_norms": len(RESPONSE_TIME_NORMS),
        "order_effects": len(ORDER_EFFECTS),
        "total_entries": (
            len(META_ANALYTIC_DB) + len(GAME_CALIBRATIONS) +
            len(CONSTRUCT_NORMS) + len(CULTURAL_ADJUSTMENTS) +
            len(RESPONSE_TIME_NORMS) + len(ORDER_EFFECTS)
        ),
        "unique_citations": len(set(
            [e.source for e in META_ANALYTIC_DB.values()] +
            [g.source for g in GAME_CALIBRATIONS.values()] +
            [n.source for n in CONSTRUCT_NORMS.values()] +
            [c.source for c in CULTURAL_ADJUSTMENTS.values()] +
            [r.source for r in RESPONSE_TIME_NORMS.values()] +
            [o.source for o in ORDER_EFFECTS.values()]
        )),
        "domains_covered": list(set(
            e.domain for e in META_ANALYTIC_DB.values() if e.domain
        )),
        "game_types": list(set(
            g.game_type for g in GAME_CALIBRATIONS.values()
        )),
        "constructs_normed": list(set(
            n.construct for n in CONSTRUCT_NORMS.values()
        )),
    }
