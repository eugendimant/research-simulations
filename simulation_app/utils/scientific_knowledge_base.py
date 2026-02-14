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

__version__ = "1.0.9.1"

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
