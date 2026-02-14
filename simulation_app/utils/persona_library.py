"""
Comprehensive Persona Library for Behavioral Science Simulations
================================================================
A THEORY-GROUNDED library of behavioral personas for generating realistic
synthetic data. All trait parameters are calibrated based on published
empirical research in survey methodology, response styles, and individual
differences psychology.

VERSION 2.2.8 - Full Domain Scientific Calibration

=============================================================================
THEORETICAL FOUNDATIONS & KEY CITATIONS
=============================================================================

SATISFICING THEORY (Krosnick, 1991, 1999)
-----------------------------------------
Reference: Krosnick, J. A. (1991). "Response strategies for coping with the
           cognitive demands of attitude measures in surveys." Applied
           Cognitive Psychology, 5(3), 213-236.

Key findings used for calibration:
- Satisficing occurs when task difficulty exceeds motivation/ability
- ~20-30% of online respondents show satisficing behaviors
- Satisficing manifests as: non-differentiation, acquiescence, primacy effects
- Engaged responders show ~25-35% more variance in scale use

Trait calibrations from Krosnick:
- Satisficers: scale_use_breadth = 0.30-0.40 (restricted range)
- Engaged: scale_use_breadth = 0.65-0.80 (full range use)
- Attention difference: ~0.25-0.35 between satisficers and engaged

EXTREME RESPONSE STYLE (Greenleaf, 1992; Hamilton, 1968)
--------------------------------------------------------
Reference: Greenleaf, E. A. (1992). "Measuring extreme response style."
           Public Opinion Quarterly, 56(3), 328-351.

Key findings:
- ~8-15% of respondents show consistent extreme responding (ERS)
- ERS is trait-like (r = .60-.75 across contexts)
- ERS associated with lower education, certain cultural backgrounds
- Extreme responders use endpoints 2-3x more than modal responders

Trait calibrations from Greenleaf:
- Extreme responders: extremity = 0.85-0.95, scale_use_breadth = 0.90+
- Modal responders: extremity = 0.15-0.25
- Population prevalence: weight = 0.08-0.12

ACQUIESCENCE BIAS (Billiet & McClendon, 2000; Podsakoff et al., 2003)
---------------------------------------------------------------------
Reference: Billiet, J. B., & McClendon, M. J. (2000). "Modeling acquiescence
           in measurement models for two balanced sets of items."
           Structural Equation Modeling, 7(4), 608-628.

Key findings:
- ~5-10% of respondents show strong acquiescence (agree regardless of content)
- Acquiescence correlates r = .30-.40 with lower education
- Acquiescence produces ~0.5-1.0 point inflation on 7-point scales
- Reverse-coded items detect acquiescence (r < .20 expected vs actual r > .60)

Trait calibrations:
- High acquiescers: acquiescence = 0.75-0.85
- Low acquiescers: acquiescence = 0.40-0.50
- Effect on reversed items: ~1.5 point discrepancy vs non-reversed

SOCIAL DESIRABILITY (Paulhus, 1984, 1991, 2002)
-----------------------------------------------
Reference: Paulhus, D. L. (1991). "Measurement and control of response bias."
           In J. P. Robinson et al. (Eds.), Measures of personality and
           social psychological attitudes (pp. 17-59). Academic Press.

Key findings from BIDR (Balanced Inventory of Desirable Responding):
- Two components: Self-Deceptive Enhancement (SDE) and Impression Management (IM)
- SDE: Unconscious positive bias, M = 6.0, SD = 3.2 (0-20 scale)
- IM: Deliberate faking, M = 4.5, SD = 3.5 (0-20 scale)
- High SD responders inflate positive items by ~0.8-1.2 points

Trait calibrations (converted to 0-1):
- High impression management: social_desirability = 0.70-0.85
- High self-deception: self_deception = 0.65-0.75
- Low SD concern: social_desirability = 0.30-0.45

CARELESS RESPONDING (Meade & Craig, 2012; Curran, 2016)
-------------------------------------------------------
Reference: Meade, A. W., & Craig, S. B. (2012). "Identifying careless responses
           in survey data." Psychological Methods, 17(3), 437-455.

Key findings:
- ~3-9% of MTurk/online samples are careless responders
- Careless responding detected by: response time, attention checks, consistency
- Careless responders complete surveys 40-60% faster than attentive ones
- Intra-individual response variability (IRV) is bimodally distributed

Trait calibrations from Meade & Craig:
- Careless: attention_level = 0.30-0.45, response_consistency = 0.25-0.40
- Reading speed: 0.90-0.98 (very fast)
- Attention check failure rate: 40-70%

RESPONSE TIME RESEARCH (Ratcliff, 1978; Yan & Tourangeau, 2008)
---------------------------------------------------------------
Reference: Yan, T., & Tourangeau, R. (2008). "Fast times and easy questions."
           Public Opinion Quarterly, 72(2), 196-212.

Key findings:
- Mean response time per item: 2-5 seconds for Likert scales
- Engaged responders: 3-5 seconds/item
- Satisficers: 1-2 seconds/item
- Careless: <1 second/item

CONDITION EFFECTS ON RESPONSES
------------------------------
Based on experimental literature on how manipulations affect DV scales:

- Treatment effects typically d = 0.2-0.8 (small to large)
- AI disclosure studies: d = 0.3-0.5 (Longoni et al., 2019; Dietvorst et al., 2015)
- Hedonic vs utilitarian framing: d = 0.4-0.6 (Dhar & Wertenbroch, 2000)
- Prosocial priming effects: d = 0.2-0.4 (meta-analyses)

=============================================================================
RESPONSE PATTERN IMPLEMENTATION
=============================================================================

The simulation generates responses using this scientifically-grounded process:

1. BASE RESPONSE = trait.response_tendency × scale_range
   - response_tendency derived from persona traits
   - Calibrated to produce realistic mean responses (M ≈ 4.0-5.0 on 7-point)

2. CONDITION EFFECT = Cohen's d × pooled_SD × direction
   - Effect sizes based on experimental literature
   - Auto-generated if not specified (d = 0.4-0.6 default)

3. INDIVIDUAL VARIANCE = N(0, SD) where SD = scale_range × trait.variance
   - Within-person SD ≈ 1.2-1.8 on 7-point scales (published norms)
   - Variance trait modulates individual differences

4. RESPONSE STYLE EFFECTS:
   - Extremity: P(endpoint) = extremity_trait × 0.4
   - Acquiescence: response += (acquiescence - 0.5) × scale_range × 0.2
   - Social desirability: response += SD_trait × 0.15 (for positive items)

=============================================================================
VALIDATION AGAINST PUBLISHED NORMS
=============================================================================

The simulation should produce data matching these empirical benchmarks:
- Mean scale responses: 4.0-5.2 on 7-point scales
- Within-condition SD: 1.2-1.8 on 7-point scales
- Between-condition d: 0.3-0.7 for typical experimental manipulations
- Attention check pass rate: 85-95% (after exclusions)
- Cronbach's alpha for multi-item scales: 0.70-0.90

Based on recent LLM simulation research:
- Argyle et al. (2023) - "Out of One, Many" Political Analysis
- Horton (2023) - "Homo Silicus" NBER Working Paper
- Aher, Arriaga & Kalai (2023) - LLM human subject replication, ICML
- Binz & Schulz (2023) - LLM cognitive patterns, PNAS
- Park et al. (2023) - Generative Agents, Stanford/Google
- Westwood (2025) - "Existential threat of LLMs to survey research" PNAS
"""

# Version identifier to help track deployed code
__version__ = "1.0.9.0"  # v1.0.9.0: SocSim 10-iteration improvement — 12 strategies, calibration, game families

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


def _stable_int_hash(s: str) -> int:
    """Generate a stable integer hash that is reproducible across Python sessions."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


@dataclass
class PersonaTrait:
    """Represents a single trait dimension with mean and variance.

    Each trait dimension captures a specific aspect of response behavior
    that influences how participants interact with survey items.
    """
    name: str
    base_mean: float  # 0-1 scale
    base_sd: float    # Variance around mean
    description: str
    # Optional additional parameters
    min_value: float = 0.0  # Minimum possible value
    max_value: float = 1.0  # Maximum possible value


# Comprehensive list of trait dimensions (25 dimensions)
TRAIT_DIMENSIONS = {
    # ========== Attention & Engagement Traits ==========
    'attention_level': 'How carefully the persona attends to stimuli',
    'engagement': 'Overall engagement with the survey task',
    'reading_speed': 'Speed of processing survey content',
    'comprehension': 'Understanding of complex questions',
    'effort': 'Cognitive effort invested in responding',

    # ========== Response Style Traits ==========
    'response_consistency': 'Within-person consistency across similar items',
    'scale_use_breadth': 'Tendency to use full range vs. restricted range',
    'acquiescence': 'Tendency to agree regardless of content',
    'extremity': 'Tendency to use scale endpoints',
    'midpoint_preference': 'Tendency to use scale midpoints',

    # ========== Bias Traits ==========
    'social_desirability': 'Tendency to respond in socially desirable ways',
    'impression_management': 'Deliberate positive self-presentation',
    'self_deception': 'Unconscious positive self-perception',
    'demand_characteristics': 'Susceptibility to perceived study expectations',

    # ========== Cognitive Traits ==========
    'need_for_cognition': 'Enjoyment of effortful cognitive tasks',
    'cognitive_reflection': 'Tendency for reflective vs. intuitive thinking',
    'analytical_thinking': 'Preference for systematic analysis',
    'risk_tolerance': 'Willingness to accept uncertainty',

    # ========== Personality-Based Traits ==========
    'openness': 'Openness to new experiences and ideas',
    'conscientiousness': 'Thoroughness and dependability',
    'extraversion': 'Outgoing, energetic orientation',
    'agreeableness': 'Cooperative, trusting disposition',
    'emotional_stability': 'Calm, emotionally resilient disposition',

    # ========== Response Quality Traits ==========
    'honesty': 'Truthfulness in self-reporting',
    'elaboration': 'Tendency to provide detailed responses',
}


@dataclass
class Persona:
    """
    A behavioral persona with trait parameters that influence response patterns.

    Each persona represents a coherent response style grounded in psychological
    literature on survey responding and individual differences.
    """
    name: str
    category: str  # Domain category
    description: str
    weight: float  # Default population weight
    traits: Dict[str, PersonaTrait]
    text_style: Dict[str, Any]  # For open-ended response generation
    applicable_domains: List[str]  # Research domains where this persona is relevant


class PersonaLibrary:
    """
    Comprehensive library of behavioral personas organized by research domain.

    This library provides 50+ behavioral archetypes organized across 15 research
    domains, with 25 trait dimensions per persona for realistic response simulation.

    Domains covered:
    - Consumer Behavior & Marketing
    - AI & Technology Attitudes
    - Behavioral Economics & Decision Making
    - Organizational Behavior & Management
    - Social Psychology
    - Health Psychology
    - Environmental Psychology
    - Political Psychology
    - Educational Psychology
    - Clinical Psychology
    - Cognitive Psychology
    - Legal Psychology
    - Sports Psychology
    - Cross-Cultural Psychology
    - Positive Psychology

    Each persona includes:
    - Trait parameters (25 dimensions)
    - Text style characteristics
    - Applicable research domains
    - Population weights for realistic sampling
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize the persona library with optional seed for reproducibility."""
        self.seed = seed
        if seed:
            np.random.seed(seed)
            random.seed(seed)

        self.personas = self._build_persona_library()
        self.domain_keywords = self._build_domain_keywords()

    def _build_persona_library(self) -> Dict[str, Persona]:
        """Build the complete persona library with all behavioral archetypes."""

        personas = {}

        # ================================================================
        # CORE RESPONSE STYLE PERSONAS (Universal - Apply to all studies)
        # Based on Krosnick (1991), Greenleaf (1992), Paulhus (1991)
        # ================================================================

        # ================================================================
        # ENGAGED RESPONDER - Krosnick (1991) "Optimizing" respondent
        # ================================================================
        # Based on Krosnick's satisficing theory: ~30-40% of respondents
        # engage in careful, effortful responding ("optimizing").
        # Calibrations from Krosnick (1991, 1999) and Tourangeau et al. (2000).
        personas['engaged_responder'] = Persona(
            name="Engaged Responder",
            category="response_style",
            description="Attentive participant who reads carefully and provides thoughtful responses. "
                       "Krosnick (1991): 'Optimizers' who engage fully with survey task. "
                       "Shows natural variance reflecting genuine opinion differences across items.",
            weight=0.35,  # Krosnick estimates 30-40% of respondents optimize
            traits={
                # Attention: High for optimizers (Meade & Craig, 2012: top quartile)
                'attention_level': PersonaTrait('attention_level', 0.92, 0.05, 'High attention - Meade & Craig top quartile'),
                # Consistency: r = .70-.85 for engaged respondents (Curran, 2016)
                'response_consistency': PersonaTrait('response_consistency', 0.78, 0.08, 'Consistent - within published reliability norms'),
                # Scale breadth: 65-80% of range used by engaged (Greenleaf, 1992)
                'scale_use_breadth': PersonaTrait('scale_use_breadth', 0.72, 0.10, 'Full range - Greenleaf benchmark'),
                # Acquiescence: ~0.50 = neutral (Billiet & McClendon, 2000)
                'acquiescence': PersonaTrait('acquiescence', 0.48, 0.08, 'Neutral - no systematic bias'),
                # Social desirability: Low IM (Paulhus, 1991: M=4.5/20, so ~0.40)
                'social_desirability': PersonaTrait('social_desirability', 0.42, 0.10, 'Low - honest responding'),
                # Reading speed: 3-5 sec/item (Yan & Tourangeau, 2008)
                'reading_speed': PersonaTrait('reading_speed', 0.55, 0.12, 'Moderate - thorough reading'),
                # Response tendency: Centered for genuine opinion (produces M≈4.0-4.5)
                'response_tendency': PersonaTrait('response_tendency', 0.58, 0.12, 'Slightly above midpoint - positivity bias'),
                # Extremity: Low for thoughtful responders (Greenleaf, 1992)
                'extremity': PersonaTrait('extremity', 0.18, 0.08, 'Low endpoint use'),
            },
            text_style={
                'verbosity': 'moderate',
                'detail_level': 'specific',
                'coherence': 'high',
                'sentiment_alignment': 'consistent'
            },
            applicable_domains=['all']
        )

        # ================================================================
        # SATISFICER - Krosnick (1991) weak/strong satisficing
        # ================================================================
        # Based on Krosnick's satisficing theory: ~20-30% show satisficing.
        # Key markers: non-differentiation, acquiescence, primacy effects.
        personas['satisficer'] = Persona(
            name="Satisficer",
            category="response_style",
            description="Participant who puts in minimal cognitive effort. "
                       "Krosnick (1991): Satisficers minimize cognitive effort. "
                       "Uses scale midpoints, straight-lines on matrices, short text responses.",
            weight=0.22,  # Krosnick: 20-30% of online samples
            traits={
                # Attention: Lower by ~0.25 (Krosnick, 1991 differential)
                'attention_level': PersonaTrait('attention_level', 0.68, 0.10, 'Moderate-low - satisficing threshold'),
                # Consistency: Lower due to non-differentiation (r ~ .50-.65)
                'response_consistency': PersonaTrait('response_consistency', 0.55, 0.12, 'Reduced differentiation'),
                # Scale breadth: Restricted range, midpoint preference (Greenleaf, 1992)
                'scale_use_breadth': PersonaTrait('scale_use_breadth', 0.35, 0.08, 'Restricted - midpoint clustering'),
                # Acquiescence: Elevated (Krosnick, 1991: "weak satisficing")
                'acquiescence': PersonaTrait('acquiescence', 0.62, 0.08, 'Elevated - path of least resistance'),
                # Social desirability: Neutral (not effortful enough to fake)
                'social_desirability': PersonaTrait('social_desirability', 0.50, 0.10, 'Neutral'),
                # Reading speed: Fast - 1-2 sec/item (Yan & Tourangeau, 2008)
                'reading_speed': PersonaTrait('reading_speed', 0.82, 0.08, 'Fast - minimal processing'),
                # Response tendency: Midpoint anchored (produces M≈4.0)
                'response_tendency': PersonaTrait('response_tendency', 0.50, 0.08, 'Midpoint anchored'),
                # Extremity: Very low (Greenleaf: midpoint = low extremity)
                'extremity': PersonaTrait('extremity', 0.12, 0.05, 'Very low - avoids endpoints'),
            },
            text_style={
                'verbosity': 'minimal',
                'detail_level': 'vague',
                'coherence': 'low',
                'sentiment_alignment': 'neutral'
            },
            applicable_domains=['all']
        )

        # ================================================================
        # EXTREME RESPONDER - Greenleaf (1992) ERS
        # ================================================================
        # Based on Greenleaf (1992): 8-15% show consistent ERS.
        # ERS is trait-like (r = .60-.75 stability).
        personas['extreme_responder'] = Persona(
            name="Extreme Responder",
            category="response_style",
            description="Participant who consistently uses scale endpoints (1s and 7s). "
                       "Greenleaf (1992): ERS is trait-like, ~10% prevalence. "
                       "Uses endpoints 2-3x more than modal responders.",
            weight=0.10,  # Greenleaf: 8-15% prevalence
            traits={
                # Attention: Adequate (ERS not related to carelessness)
                'attention_level': PersonaTrait('attention_level', 0.80, 0.08, 'Good - ERS orthogonal to attention'),
                # Consistency: High within ERS pattern (r = .60-.75)
                'response_consistency': PersonaTrait('response_consistency', 0.72, 0.10, 'Consistent within style'),
                # Scale breadth: Maximum - uses full range (all endpoints)
                'scale_use_breadth': PersonaTrait('scale_use_breadth', 0.92, 0.05, 'Maximum - endpoint focus'),
                # Acquiescence: Variable (ERS can be positive or negative)
                'acquiescence': PersonaTrait('acquiescence', 0.52, 0.15, 'Variable direction'),
                # Social desirability: Lower (less impression management)
                'social_desirability': PersonaTrait('social_desirability', 0.38, 0.12, 'Lower - expressive style'),
                # Reading speed: Moderate
                'reading_speed': PersonaTrait('reading_speed', 0.65, 0.12, 'Moderate'),
                # Response tendency: More positive on average (Greenleaf)
                'response_tendency': PersonaTrait('response_tendency', 0.62, 0.18, 'Variable but often positive'),
                # Extremity: Very high (defining characteristic)
                'extremity': PersonaTrait('extremity', 0.88, 0.05, 'Very high - defining trait'),
            },
            text_style={
                'verbosity': 'moderate',
                'detail_level': 'emphatic',
                'coherence': 'moderate',
                'sentiment_alignment': 'strong'
            },
            applicable_domains=['all']
        )

        # ================================================================
        # ACQUIESCENT RESPONDER - Billiet & McClendon (2000)
        # ================================================================
        # Based on acquiescence research: 5-10% show strong agreement bias.
        # Creates systematic inflation on positively-worded items.
        personas['acquiescent_responder'] = Persona(
            name="Acquiescent Responder",
            category="response_style",
            description="Participant who tends to agree with statements regardless of content. "
                       "Billiet & McClendon (2000): 5-10% prevalence. "
                       "Creates ~0.8 point inflation on positive items vs reverse-coded.",
            weight=0.08,  # Billiet & McClendon: 5-10%
            traits={
                # Attention: Moderate-low (acquiescence linked to low effort)
                'attention_level': PersonaTrait('attention_level', 0.72, 0.10, 'Moderate-low'),
                # Consistency: Poor for reverse-coded (creates inconsistency)
                'response_consistency': PersonaTrait('response_consistency', 0.48, 0.12, 'Poor reverse item consistency'),
                # Scale breadth: Restricted to upper half
                'scale_use_breadth': PersonaTrait('scale_use_breadth', 0.55, 0.10, 'Upper half restricted'),
                # Acquiescence: Very high (defining characteristic)
                'acquiescence': PersonaTrait('acquiescence', 0.82, 0.06, 'Very high - defining trait'),
                # Social desirability: Elevated (agreeing seen as polite)
                'social_desirability': PersonaTrait('social_desirability', 0.62, 0.10, 'Elevated - agreeableness'),
                # Reading speed: Faster (low cognitive engagement)
                'reading_speed': PersonaTrait('reading_speed', 0.75, 0.10, 'Faster'),
                # Response tendency: Elevated (agreement = higher scores)
                'response_tendency': PersonaTrait('response_tendency', 0.68, 0.08, 'Elevated - agreement bias'),
                # Extremity: Moderate-high on agree side
                'extremity': PersonaTrait('extremity', 0.35, 0.12, 'Moderate - agrees strongly'),
            },
            text_style={
                'verbosity': 'moderate',
                'detail_level': 'agreeable',
                'coherence': 'moderate',
                'sentiment_alignment': 'positive'
            },
            applicable_domains=['all']
        )

        # ================================================================
        # CARELESS RESPONDER - Meade & Craig (2012)
        # ================================================================
        # Based on Meade & Craig (2012): 3-9% in online samples.
        # Detected by: response time, attention checks, consistency.
        personas['careless_responder'] = Persona(
            name="Careless Responder",
            category="response_style",
            description="Participant showing clear inattention - fails attention checks, "
                       "inconsistent reversed items, implausible response patterns. "
                       "Meade & Craig (2012): 3-9% prevalence in online samples.",
            weight=0.05,  # Meade & Craig: 3-9% (lower bound for quality samples)
            traits={
                # Attention: Very low (defining characteristic)
                'attention_level': PersonaTrait('attention_level', 0.35, 0.12, 'Very low - Meade & Craig bottom decile'),
                # Consistency: Very poor (IRV in top 10%)
                'response_consistency': PersonaTrait('response_consistency', 0.28, 0.12, 'Very poor - high IRV'),
                # Scale breadth: Random pattern
                'scale_use_breadth': PersonaTrait('scale_use_breadth', 0.50, 0.22, 'Random - no pattern'),
                # Acquiescence: Random (no systematic pattern)
                'acquiescence': PersonaTrait('acquiescence', 0.50, 0.18, 'Random'),
                # Social desirability: Unconcerned
                'social_desirability': PersonaTrait('social_desirability', 0.50, 0.18, 'Unconcerned'),
                # Reading speed: Very fast (Meade & Craig: 40-60% faster)
                'reading_speed': PersonaTrait('reading_speed', 0.94, 0.04, 'Very fast - minimal reading'),
                # Response tendency: Random around midpoint
                'response_tendency': PersonaTrait('response_tendency', 0.50, 0.20, 'Random'),
                # Extremity: Random
                'extremity': PersonaTrait('extremity', 0.35, 0.20, 'Random'),
            },
            text_style={
                'verbosity': 'minimal',
                'detail_level': 'irrelevant',
                'coherence': 'very_low',
                'sentiment_alignment': 'random'
            },
            applicable_domains=['all']
        )

        # ================================================================
        # SOCIALLY DESIRABLE RESPONDER - Paulhus (1991)
        # ================================================================
        # Based on Paulhus BIDR: High Impression Management scores.
        # Inflates positive items, deflates negative admissions.
        personas['socially_desirable_responder'] = Persona(
            name="Socially Desirable Responder",
            category="response_style",
            description="Participant with high impression management tendency. "
                       "Paulhus (1991): High IM scores inflate positive self-report by ~1 point. "
                       "Particularly affects sensitive topics and self-evaluations.",
            weight=0.12,  # ~10-15% show elevated IM
            traits={
                # Attention: Good (requires effort to present well)
                'attention_level': PersonaTrait('attention_level', 0.82, 0.08, 'Good - impression requires attention'),
                # Consistency: High (maintains positive image)
                'response_consistency': PersonaTrait('response_consistency', 0.75, 0.08, 'Consistent positive'),
                # Scale breadth: Restricted to positive end
                'scale_use_breadth': PersonaTrait('scale_use_breadth', 0.60, 0.10, 'Positive end bias'),
                # Acquiescence: Elevated for positive items
                'acquiescence': PersonaTrait('acquiescence', 0.58, 0.10, 'Elevated for positive'),
                # Social desirability: Very high (defining characteristic)
                'social_desirability': PersonaTrait('social_desirability', 0.82, 0.06, 'Very high - defining trait'),
                # Reading speed: Moderate (careful responding)
                'reading_speed': PersonaTrait('reading_speed', 0.60, 0.12, 'Moderate - thoughtful'),
                # Response tendency: Elevated (positive self-presentation)
                'response_tendency': PersonaTrait('response_tendency', 0.70, 0.08, 'Elevated - positive bias'),
                # Extremity: Moderate-high on positive end
                'extremity': PersonaTrait('extremity', 0.40, 0.12, 'Moderate - strong positives'),
            },
            text_style={
                'verbosity': 'moderate',
                'detail_level': 'positive',
                'coherence': 'high',
                'sentiment_alignment': 'favorable'
            },
            applicable_domains=['all']
        )

        # ================================================================
        # CONSUMER BEHAVIOR & MARKETING PERSONAS
        # Scientific basis: Consumer research literature on brand loyalty,
        # price sensitivity, and hedonic/utilitarian consumption.
        # References: Thomson et al. (2005), Babin et al. (1994), Rook (1987)
        # Response tendency calibrated to domain-typical means (M=4.5-5.5)
        # Extremity based on emotional involvement with products/brands
        # ================================================================

        personas['brand_loyalist'] = Persona(
            name="Brand Loyalist",
            category="consumer",
            description="Consumer with strong brand attachments, high brand trust, "
                       "resistant to switching. Values consistency and familiarity. "
                       "Thomson et al. (2005): Brand attachment M=4.8/7 for loyal customers.",
            weight=0.12,
            traits={
                'brand_attachment': PersonaTrait('brand_attachment', 0.82, 0.08, 'High attachment'),
                'price_sensitivity': PersonaTrait('price_sensitivity', 0.35, 0.12, 'Less price sensitive'),
                'novelty_seeking': PersonaTrait('novelty_seeking', 0.30, 0.10, 'Low novelty seeking'),
                'risk_aversion': PersonaTrait('risk_aversion', 0.70, 0.10, 'Risk averse'),
                'wtp_baseline': PersonaTrait('wtp_baseline', 0.65, 0.12, 'Higher WTP'),
                'purchase_involvement': PersonaTrait('purchase_involvement', 0.70, 0.10, 'High involvement'),
                # Scientific calibration for response generation
                'response_tendency': PersonaTrait('response_tendency', 0.68, 0.10, 'Positive toward familiar brands'),
                'extremity': PersonaTrait('extremity', 0.35, 0.10, 'Moderate-high - strong brand opinions'),
            },
            text_style={
                'verbosity': 'moderate',
                'detail_level': 'brand_focused',
                'coherence': 'high',
                'sentiment_alignment': 'positive_toward_familiar'
            },
            applicable_domains=['consumer_behavior', 'marketing', 'branding']
        )

        personas['deal_seeker'] = Persona(
            name="Deal Seeker",
            category="consumer",
            description="Price-conscious consumer who actively seeks discounts and promotions. "
                       "High price sensitivity, enjoys the thrill of finding bargains. "
                       "Lichtenstein et al. (1990): Deal proneness M=5.2/7.",
            weight=0.15,
            traits={
                'brand_attachment': PersonaTrait('brand_attachment', 0.35, 0.12, 'Low attachment'),
                'price_sensitivity': PersonaTrait('price_sensitivity', 0.85, 0.08, 'Very price sensitive'),
                'novelty_seeking': PersonaTrait('novelty_seeking', 0.55, 0.12, 'Moderate'),
                'risk_aversion': PersonaTrait('risk_aversion', 0.50, 0.12, 'Moderate'),
                'wtp_baseline': PersonaTrait('wtp_baseline', 0.35, 0.10, 'Lower WTP'),
                'purchase_involvement': PersonaTrait('purchase_involvement', 0.75, 0.10, 'High for deals'),
                # Scientific calibration
                'response_tendency': PersonaTrait('response_tendency', 0.52, 0.12, 'More critical of full prices'),
                'extremity': PersonaTrait('extremity', 0.25, 0.10, 'Moderate - value-focused analysis'),
            },
            text_style={
                'verbosity': 'moderate',
                'detail_level': 'price_focused',
                'coherence': 'high',
                'sentiment_alignment': 'value_oriented'
            },
            applicable_domains=['consumer_behavior', 'marketing', 'pricing', 'promotions']
        )

        personas['impulse_buyer'] = Persona(
            name="Impulse Buyer",
            category="consumer",
            description="Consumer prone to spontaneous purchases, driven by emotions and "
                       "immediate gratification. Lower self-control in buying contexts. "
                       "Rook (1987): Impulse buying involves heightened emotional states.",
            weight=0.10,
            traits={
                'brand_attachment': PersonaTrait('brand_attachment', 0.45, 0.15, 'Variable'),
                'price_sensitivity': PersonaTrait('price_sensitivity', 0.40, 0.15, 'Lower when excited'),
                'novelty_seeking': PersonaTrait('novelty_seeking', 0.75, 0.10, 'High novelty seeking'),
                'risk_aversion': PersonaTrait('risk_aversion', 0.35, 0.12, 'Lower risk aversion'),
                'wtp_baseline': PersonaTrait('wtp_baseline', 0.60, 0.15, 'Variable, often higher'),
                'purchase_involvement': PersonaTrait('purchase_involvement', 0.45, 0.15, 'Quick decisions'),
                'self_control': PersonaTrait('self_control', 0.35, 0.10, 'Lower self-control'),
                # Scientific calibration - emotional, reactive
                'response_tendency': PersonaTrait('response_tendency', 0.65, 0.15, 'Emotionally positive toward stimuli'),
                'extremity': PersonaTrait('extremity', 0.45, 0.12, 'Higher - emotional reactions'),
            },
            text_style={
                'verbosity': 'enthusiastic',
                'detail_level': 'emotion_focused',
                'coherence': 'moderate',
                'sentiment_alignment': 'excitement_driven'
            },
            applicable_domains=['consumer_behavior', 'marketing', 'impulse_buying', 'self_control']
        )

        personas['conscious_consumer'] = Persona(
            name="Conscious Consumer",
            category="consumer",
            description="Ethically-minded consumer who considers sustainability, social impact, "
                       "and corporate responsibility in purchase decisions. "
                       "Shaw & Shiu (2002): Ethical consumer intentions M=5.4/7.",
            weight=0.12,
            traits={
                'ethical_concern': PersonaTrait('ethical_concern', 0.85, 0.08, 'High ethical concern'),
                'price_sensitivity': PersonaTrait('price_sensitivity', 0.45, 0.12, 'Willing to pay premium'),
                'brand_attachment': PersonaTrait('brand_attachment', 0.55, 0.12, 'Values-based loyalty'),
                'information_seeking': PersonaTrait('information_seeking', 0.80, 0.08, 'Researches thoroughly'),
                'wtp_baseline': PersonaTrait('wtp_baseline', 0.65, 0.10, 'Higher for ethical'),
                'purchase_involvement': PersonaTrait('purchase_involvement', 0.80, 0.08, 'Very involved'),
                # Scientific calibration - values-driven, deliberate
                'response_tendency': PersonaTrait('response_tendency', 0.62, 0.10, 'Positive for ethical options'),
                'extremity': PersonaTrait('extremity', 0.30, 0.10, 'Moderate - thoughtful evaluation'),
            },
            text_style={
                'verbosity': 'detailed',
                'detail_level': 'values_focused',
                'coherence': 'high',
                'sentiment_alignment': 'ethics_driven'
            },
            applicable_domains=['consumer_behavior', 'sustainability', 'csr', 'ethical_consumption']
        )

        personas['hedonic_consumer'] = Persona(
            name="Hedonic Consumer",
            category="consumer",
            description="Pleasure-oriented consumer who values enjoyment, fun, and emotional "
                       "experiences from products. Prioritizes hedonic over utilitarian benefits. "
                       "Babin et al. (1994): Hedonic shopping value M=5.1/7 for experience-seekers.",
            weight=0.12,
            traits={
                'hedonic_motivation': PersonaTrait('hedonic_motivation', 0.85, 0.08, 'High hedonic drive'),
                'utilitarian_motivation': PersonaTrait('utilitarian_motivation', 0.40, 0.12, 'Lower utilitarian'),
                'emotional_intensity': PersonaTrait('emotional_intensity', 0.75, 0.10, 'Strong emotions'),
                'novelty_seeking': PersonaTrait('novelty_seeking', 0.70, 0.10, 'Seeks new experiences'),
                'wtp_baseline': PersonaTrait('wtp_baseline', 0.60, 0.12, 'Pays for pleasure'),
                'aesthetic_sensitivity': PersonaTrait('aesthetic_sensitivity', 0.75, 0.10, 'Values aesthetics'),
                # Scientific calibration - emotional, experience-focused
                'response_tendency': PersonaTrait('response_tendency', 0.70, 0.12, 'Positive for pleasurable stimuli'),
                'extremity': PersonaTrait('extremity', 0.48, 0.10, 'Higher - emotional intensity'),
            },
            text_style={
                'verbosity': 'expressive',
                'detail_level': 'experience_focused',
                'coherence': 'high',
                'sentiment_alignment': 'pleasure_oriented'
            },
            applicable_domains=['consumer_behavior', 'hedonic_consumption', 'experiential_marketing']
        )

        personas['utilitarian_consumer'] = Persona(
            name="Utilitarian Consumer",
            category="consumer",
            description="Function-oriented consumer who prioritizes practical benefits, efficiency, "
                       "and value for money. Focuses on product performance over experience. "
                       "Babin et al. (1994): Utilitarian value emphasizes task completion, efficiency.",
            weight=0.12,
            traits={
                'hedonic_motivation': PersonaTrait('hedonic_motivation', 0.35, 0.10, 'Lower hedonic drive'),
                'utilitarian_motivation': PersonaTrait('utilitarian_motivation', 0.85, 0.08, 'High utilitarian'),
                'emotional_intensity': PersonaTrait('emotional_intensity', 0.40, 0.12, 'More rational'),
                'novelty_seeking': PersonaTrait('novelty_seeking', 0.40, 0.12, 'Prefers proven'),
                'wtp_baseline': PersonaTrait('wtp_baseline', 0.50, 0.10, 'Value-focused'),
                'information_seeking': PersonaTrait('information_seeking', 0.75, 0.10, 'Researches specs'),
                # Scientific calibration - rational, deliberate
                'response_tendency': PersonaTrait('response_tendency', 0.55, 0.08, 'Neutral-analytical'),
                'extremity': PersonaTrait('extremity', 0.18, 0.08, 'Low - measured, analytical'),
            },
            text_style={
                'verbosity': 'concise',
                'detail_level': 'function_focused',
                'coherence': 'high',
                'sentiment_alignment': 'practical'
            },
            applicable_domains=['consumer_behavior', 'utilitarian_consumption', 'product_evaluation']
        )

        # ================================================================
        # AI & TECHNOLOGY PERSONAS
        # Scientific basis: Research on algorithm aversion/appreciation,
        # technology acceptance, and AI attitudes.
        # References: Dietvorst et al. (2015), Longoni et al. (2019), Logg et al. (2019)
        # Response calibrated to AI/tech attitude research norms
        # ================================================================

        personas['tech_enthusiast'] = Persona(
            name="Tech Enthusiast",
            category="technology",
            description="Early adopter with positive attitudes toward new technology and AI. "
                       "High tech self-efficacy, sees benefits over risks. "
                       "Logg et al. (2019): Algorithm appreciation shows M=5.8/7 for tech-positive.",
            weight=0.15,
            traits={
                'tech_affinity': PersonaTrait('tech_affinity', 0.88, 0.07, 'Very high'),
                'ai_attitude': PersonaTrait('ai_attitude', 0.82, 0.08, 'Very positive'),
                'ai_anxiety': PersonaTrait('ai_anxiety', 0.25, 0.10, 'Low anxiety'),
                'perceived_ai_competence': PersonaTrait('perceived_ai_competence', 0.80, 0.08, 'High trust'),
                'human_uniqueness_belief': PersonaTrait('human_uniqueness_belief', 0.40, 0.12, 'Open to AI'),
                'tech_self_efficacy': PersonaTrait('tech_self_efficacy', 0.85, 0.08, 'High confidence'),
                'algorithm_appreciation': PersonaTrait('algorithm_appreciation', 0.80, 0.08, 'Appreciates algorithms'),
                # Scientific calibration - optimistic about technology
                'response_tendency': PersonaTrait('response_tendency', 0.75, 0.10, 'Positive toward AI/tech'),
                'extremity': PersonaTrait('extremity', 0.40, 0.10, 'Moderate-high - enthusiastic'),
            },
            text_style={
                'verbosity': 'detailed',
                'detail_level': 'tech_savvy',
                'coherence': 'high',
                'sentiment_alignment': 'optimistic_tech'
            },
            applicable_domains=['ai', 'technology', 'automation', 'algorithms']
        )

        personas['tech_skeptic'] = Persona(
            name="Tech Skeptic",
            category="technology",
            description="Cautious toward new technology and AI. Concerns about privacy, job loss, "
                       "and loss of human control. Prefers human alternatives. "
                       "Dietvorst et al. (2015): Algorithm aversion shows M=3.2/7 for AI competence.",
            weight=0.15,
            traits={
                'tech_affinity': PersonaTrait('tech_affinity', 0.35, 0.10, 'Low'),
                'ai_attitude': PersonaTrait('ai_attitude', 0.30, 0.10, 'Negative'),
                'ai_anxiety': PersonaTrait('ai_anxiety', 0.75, 0.10, 'High anxiety'),
                'perceived_ai_competence': PersonaTrait('perceived_ai_competence', 0.35, 0.12, 'Low trust'),
                'human_uniqueness_belief': PersonaTrait('human_uniqueness_belief', 0.80, 0.08, 'Strong belief'),
                'tech_self_efficacy': PersonaTrait('tech_self_efficacy', 0.40, 0.12, 'Lower confidence'),
                'algorithm_aversion': PersonaTrait('algorithm_aversion', 0.75, 0.10, 'Prefers humans'),
                # Scientific calibration - cautious, concerned
                'response_tendency': PersonaTrait('response_tendency', 0.38, 0.10, 'Negative toward AI/tech'),
                'extremity': PersonaTrait('extremity', 0.35, 0.10, 'Moderate - concerned opinions'),
            },
            text_style={
                'verbosity': 'moderate',
                'detail_level': 'concern_focused',
                'coherence': 'high',
                'sentiment_alignment': 'cautious'
            },
            applicable_domains=['ai', 'technology', 'automation', 'algorithms', 'ai_aversion']
        )

        personas['ai_pragmatist'] = Persona(
            name="AI Pragmatist",
            category="technology",
            description="Balanced view of AI - sees both benefits and risks. Accepts AI for "
                       "appropriate tasks but values human judgment for important decisions. "
                       "Represents modal response pattern in AI attitude research.",
            weight=0.20,
            traits={
                'tech_affinity': PersonaTrait('tech_affinity', 0.60, 0.12, 'Moderate'),
                'ai_attitude': PersonaTrait('ai_attitude', 0.55, 0.12, 'Balanced'),
                'ai_anxiety': PersonaTrait('ai_anxiety', 0.50, 0.12, 'Moderate concern'),
                'perceived_ai_competence': PersonaTrait('perceived_ai_competence', 0.55, 0.12, 'Task-dependent'),
                'human_uniqueness_belief': PersonaTrait('human_uniqueness_belief', 0.60, 0.10, 'Moderate'),
                'tech_self_efficacy': PersonaTrait('tech_self_efficacy', 0.60, 0.12, 'Adequate'),
                'algorithm_appreciation': PersonaTrait('algorithm_appreciation', 0.55, 0.12, 'Context-dependent'),
                # Scientific calibration - neutral, balanced
                'response_tendency': PersonaTrait('response_tendency', 0.55, 0.10, 'Balanced - context-dependent'),
                'extremity': PersonaTrait('extremity', 0.20, 0.08, 'Low - nuanced views'),
            },
            text_style={
                'verbosity': 'moderate',
                'detail_level': 'balanced',
                'coherence': 'high',
                'sentiment_alignment': 'nuanced'
            },
            applicable_domains=['ai', 'technology', 'automation', 'algorithms']
        )

        personas['privacy_concerned'] = Persona(
            name="Privacy-Concerned User",
            category="technology",
            description="Highly concerned about data privacy and surveillance. Reluctant to share "
                       "personal information, skeptical of data collection practices. "
                       "Westin (1991) typology: Privacy fundamentalists score high on concern.",
            weight=0.12,
            traits={
                'privacy_concern': PersonaTrait('privacy_concern', 0.88, 0.06, 'Very high'),
                'data_sharing_willingness': PersonaTrait('data_sharing_willingness', 0.25, 0.10, 'Very low'),
                'institutional_trust': PersonaTrait('institutional_trust', 0.30, 0.10, 'Low trust'),
                'tech_affinity': PersonaTrait('tech_affinity', 0.50, 0.15, 'Variable'),
                'ai_attitude': PersonaTrait('ai_attitude', 0.40, 0.12, 'Cautious'),
                'personalization_preference': PersonaTrait('personalization_preference', 0.30, 0.12, 'Prefers generic'),
                # Scientific calibration - protective, wary
                'response_tendency': PersonaTrait('response_tendency', 0.42, 0.10, 'Negative toward data collection'),
                'extremity': PersonaTrait('extremity', 0.38, 0.10, 'Moderate-high - strong privacy stance'),
            },
            text_style={
                'verbosity': 'moderate',
                'detail_level': 'privacy_aware',
                'coherence': 'high',
                'sentiment_alignment': 'protective'
            },
            applicable_domains=['technology', 'privacy', 'data', 'personalization', 'surveillance',
                                'consumer_behavior', 'health_psychology', 'political_psychology']
        )

        # ================================================================
        # BEHAVIORAL ECONOMICS PERSONAS
        # Scientific basis: Prospect theory, temporal discounting, social
        # comparison research.
        # References: Kahneman & Tversky (1979), Frederick et al. (2002)
        # Response calibrated to decision-making literature norms
        # ================================================================

        personas['loss_averse'] = Persona(
            name="Loss-Averse Decision Maker",
            category="behavioral_economics",
            description="Individual who weighs losses more heavily than equivalent gains. "
                       "Risk-seeking in loss domain, risk-averse in gain domain. "
                       "Kahneman & Tversky (1979): λ≈2.25 loss aversion coefficient.",
            weight=0.18,
            traits={
                'loss_aversion': PersonaTrait('loss_aversion', 0.80, 0.10, 'High loss aversion'),
                'risk_tolerance': PersonaTrait('risk_tolerance', 0.35, 0.12, 'Low in gains'),
                'status_quo_bias': PersonaTrait('status_quo_bias', 0.70, 0.10, 'Prefers status quo'),
                'endowment_effect': PersonaTrait('endowment_effect', 0.72, 0.10, 'Strong ownership effect'),
                'regret_anticipation': PersonaTrait('regret_anticipation', 0.75, 0.10, 'High regret concern'),
                'time_preference': PersonaTrait('time_preference', 0.55, 0.12, 'Moderate patience'),
                # Scientific calibration - risk-averse, cautious
                'response_tendency': PersonaTrait('response_tendency', 0.48, 0.10, 'Conservative evaluations'),
                'extremity': PersonaTrait('extremity', 0.28, 0.10, 'Moderate - avoids risk'),
            },
            text_style={
                'verbosity': 'moderate',
                'detail_level': 'risk_focused',
                'coherence': 'high',
                'sentiment_alignment': 'cautious'
            },
            applicable_domains=['behavioral_economics', 'decision_making', 'risk', 'finance',
                                'health_psychology', 'consumer_behavior', 'insurance']
        )

        personas['present_biased'] = Persona(
            name="Present-Biased Individual",
            category="behavioral_economics",
            description="Person who heavily discounts future outcomes, preferring immediate "
                       "rewards. Struggles with self-control and long-term planning. "
                       "Frederick et al. (2002): β≈0.7 present bias parameter.",
            weight=0.12,
            traits={
                'time_preference': PersonaTrait('time_preference', 0.25, 0.10, 'Very impatient'),
                'self_control': PersonaTrait('self_control', 0.35, 0.12, 'Low self-control'),
                'planning_tendency': PersonaTrait('planning_tendency', 0.35, 0.12, 'Poor planning'),
                'impulsivity': PersonaTrait('impulsivity', 0.75, 0.10, 'High impulsivity'),
                'future_orientation': PersonaTrait('future_orientation', 0.30, 0.12, 'Low'),
                'commitment_device_use': PersonaTrait('commitment_device_use', 0.40, 0.15, 'Variable'),
                # Scientific calibration - impulsive, immediate-focused
                'response_tendency': PersonaTrait('response_tendency', 0.62, 0.12, 'Positive toward immediate rewards'),
                'extremity': PersonaTrait('extremity', 0.40, 0.12, 'Higher - impulsive reactions'),
            },
            text_style={
                'verbosity': 'moderate',
                'detail_level': 'present_focused',
                'coherence': 'moderate',
                'sentiment_alignment': 'immediate_gratification'
            },
            applicable_domains=['behavioral_economics', 'self_control', 'intertemporal_choice', 'savings']
        )

        personas['rational_deliberator'] = Persona(
            name="Rational Deliberator",
            category="behavioral_economics",
            description="Individual who carefully weighs options, seeks information, and makes "
                       "decisions based on expected utility. Less susceptible to biases. "
                       "Cacioppo & Petty (1982): High Need for Cognition M=4.5/5.",
            weight=0.10,
            traits={
                'need_for_cognition': PersonaTrait('need_for_cognition', 0.85, 0.08, 'High'),
                'information_seeking': PersonaTrait('information_seeking', 0.82, 0.08, 'Thorough'),
                'susceptibility_to_bias': PersonaTrait('susceptibility_to_bias', 0.30, 0.10, 'Lower'),
                'decision_time': PersonaTrait('decision_time', 0.75, 0.10, 'Takes time'),
                'regret_anticipation': PersonaTrait('regret_anticipation', 0.60, 0.12, 'Moderate'),
                'consistency_seeking': PersonaTrait('consistency_seeking', 0.80, 0.08, 'High'),
                # Scientific calibration - analytical, measured
                'response_tendency': PersonaTrait('response_tendency', 0.55, 0.08, 'Neutral - evidence-based'),
                'extremity': PersonaTrait('extremity', 0.15, 0.06, 'Very low - careful differentiation'),
            },
            text_style={
                'verbosity': 'detailed',
                'detail_level': 'analytical',
                'coherence': 'very_high',
                'sentiment_alignment': 'reasoned'
            },
            applicable_domains=['behavioral_economics', 'decision_making', 'judgment',
                                'consumer_behavior', 'ai', 'political_psychology', 'legal',
                                'communication']
        )

        personas['social_comparer'] = Persona(
            name="Social Comparer",
            category="behavioral_economics",
            description="Individual highly influenced by social comparisons and relative standing. "
                       "Strong reactions to inequality and social reference points. "
                       "Fehr & Schmidt (1999): Inequality aversion α≈0.85, β≈0.315.",
            weight=0.12,
            traits={
                'social_comparison_orientation': PersonaTrait('social_comparison_orientation', 0.82, 0.08, 'High'),
                'inequality_aversion': PersonaTrait('inequality_aversion', 0.75, 0.10, 'Strong'),
                'status_concern': PersonaTrait('status_concern', 0.78, 0.10, 'High status concern'),
                'envy_proneness': PersonaTrait('envy_proneness', 0.65, 0.12, 'Moderate-high'),
                'fairness_concern': PersonaTrait('fairness_concern', 0.75, 0.10, 'High'),
                'conformity': PersonaTrait('conformity', 0.65, 0.12, 'Moderate-high'),
                # Scientific calibration - relative, comparison-driven
                'response_tendency': PersonaTrait('response_tendency', 0.52, 0.12, 'Context-dependent on reference'),
                'extremity': PersonaTrait('extremity', 0.35, 0.12, 'Moderate - strong fairness reactions'),
            },
            text_style={
                'verbosity': 'moderate',
                'detail_level': 'comparison_focused',
                'coherence': 'high',
                'sentiment_alignment': 'relative'
            },
            applicable_domains=['behavioral_economics', 'social_comparison', 'inequality', 'fairness',
                                'consumer_behavior', 'marketing', 'political_psychology', 'intergroup_relations',
                                'economic_games']
        )

        # ================================================================
        # ORGANIZATIONAL BEHAVIOR PERSONAS
        # Scientific basis: Work engagement, job satisfaction, and
        # leadership research.
        # References: Schaufeli et al. (2002), Bass & Avolio (1995), Meyer & Allen (1991)
        # Response calibrated to organizational survey norms
        # ================================================================

        personas['high_performer'] = Persona(
            name="High Performer",
            category="organizational",
            description="Highly engaged employee with strong work ethic, high job satisfaction, "
                       "and organizational commitment. Proactive and achievement-oriented. "
                       "Schaufeli et al. (2002): UWES engagement M=5.2/7 for high performers.",
            weight=0.15,
            traits={
                'work_engagement': PersonaTrait('work_engagement', 0.85, 0.08, 'Very engaged'),
                'job_satisfaction': PersonaTrait('job_satisfaction', 0.80, 0.10, 'High satisfaction'),
                'organizational_commitment': PersonaTrait('organizational_commitment', 0.78, 0.10, 'Committed'),
                'proactive_personality': PersonaTrait('proactive_personality', 0.82, 0.08, 'Very proactive'),
                'achievement_motivation': PersonaTrait('achievement_motivation', 0.85, 0.08, 'High'),
                'ocb_tendency': PersonaTrait('ocb_tendency', 0.75, 0.10, 'High citizenship'),
                # Scientific calibration - positive, engaged
                'response_tendency': PersonaTrait('response_tendency', 0.72, 0.10, 'Positive about work'),
                'extremity': PersonaTrait('extremity', 0.32, 0.10, 'Moderate - genuine enthusiasm'),
            },
            text_style={
                'verbosity': 'moderate',
                'detail_level': 'constructive',
                'coherence': 'high',
                'sentiment_alignment': 'positive_professional'
            },
            applicable_domains=['organizational_behavior', 'management', 'hr', 'employee_attitudes']
        )

        personas['disengaged_employee'] = Persona(
            name="Disengaged Employee",
            category="organizational",
            description="Employee with low engagement, considering turnover, minimal extra effort. "
                       "May be experiencing burnout or job dissatisfaction. "
                       "Maslach & Leiter (2016): Burnout-disengagement M=2.8/7 on engagement.",
            weight=0.12,
            traits={
                'work_engagement': PersonaTrait('work_engagement', 0.30, 0.12, 'Low engagement'),
                'job_satisfaction': PersonaTrait('job_satisfaction', 0.35, 0.12, 'Low satisfaction'),
                'organizational_commitment': PersonaTrait('organizational_commitment', 0.30, 0.10, 'Low'),
                'turnover_intention': PersonaTrait('turnover_intention', 0.70, 0.12, 'High'),
                'burnout': PersonaTrait('burnout', 0.70, 0.12, 'High burnout'),
                'ocb_tendency': PersonaTrait('ocb_tendency', 0.30, 0.12, 'Low citizenship'),
                # Scientific calibration - negative, withdrawn
                'response_tendency': PersonaTrait('response_tendency', 0.35, 0.12, 'Negative about work'),
                'extremity': PersonaTrait('extremity', 0.25, 0.12, 'Lower - disengaged, less invested'),
            },
            text_style={
                'verbosity': 'minimal',
                'detail_level': 'negative',
                'coherence': 'moderate',
                'sentiment_alignment': 'critical'
            },
            applicable_domains=['organizational_behavior', 'management', 'hr', 'turnover', 'burnout']
        )

        personas['transformational_leader'] = Persona(
            name="Transformational Leader",
            category="organizational",
            description="Leader who inspires and motivates through vision, intellectual stimulation, "
                       "and individualized consideration. High emotional intelligence. "
                       "Bass & Avolio (1995): MLQ transformational M=3.2/4 for effective leaders.",
            weight=0.08,
            traits={
                'leadership_self_efficacy': PersonaTrait('leadership_self_efficacy', 0.82, 0.08, 'High'),
                'emotional_intelligence': PersonaTrait('emotional_intelligence', 0.80, 0.08, 'High EI'),
                'vision_articulation': PersonaTrait('vision_articulation', 0.78, 0.10, 'Strong vision'),
                'empowerment_orientation': PersonaTrait('empowerment_orientation', 0.80, 0.08, 'Empowering'),
                'ethical_leadership': PersonaTrait('ethical_leadership', 0.78, 0.10, 'High ethics'),
                'change_orientation': PersonaTrait('change_orientation', 0.75, 0.10, 'Change-positive'),
                # Scientific calibration - confident, articulate
                'response_tendency': PersonaTrait('response_tendency', 0.70, 0.08, 'Positive - visionary'),
                'extremity': PersonaTrait('extremity', 0.30, 0.08, 'Moderate - inspiring but measured'),
            },
            text_style={
                'verbosity': 'articulate',
                'detail_level': 'visionary',
                'coherence': 'very_high',
                'sentiment_alignment': 'inspiring'
            },
            applicable_domains=['organizational_behavior', 'leadership', 'management']
        )

        # ================================================================
        # SOCIAL PSYCHOLOGY PERSONAS
        # Scientific basis: Prosocial behavior, social value orientation,
        # conformity and social influence research.
        # References: Batson (1991), Van Lange et al. (1997), Asch (1956)
        # Response calibrated to social psychology norms
        # ================================================================

        personas['prosocial_individual'] = Persona(
            name="Prosocial Individual",
            category="social",
            description="Person with strong prosocial orientation, high empathy, and concern for "
                       "others' welfare. Likely to cooperate and help. "
                       "Van Lange et al. (1997): Prosocials cooperate ~80% in social dilemmas.",
            weight=0.15,
            traits={
                'empathy': PersonaTrait('empathy', 0.82, 0.08, 'High empathy'),
                'altruism': PersonaTrait('altruism', 0.78, 0.10, 'High altruism'),
                'cooperation_tendency': PersonaTrait('cooperation_tendency', 0.80, 0.08, 'Cooperative'),
                'trust_propensity': PersonaTrait('trust_propensity', 0.70, 0.10, 'Trusting'),
                'social_responsibility': PersonaTrait('social_responsibility', 0.78, 0.10, 'Responsible'),
                'moral_identity': PersonaTrait('moral_identity', 0.75, 0.10, 'Strong moral identity'),
                # Scientific calibration - warm, other-focused
                'response_tendency': PersonaTrait('response_tendency', 0.68, 0.10, 'Positive toward cooperation'),
                'extremity': PersonaTrait('extremity', 0.28, 0.10, 'Moderate - genuine caring'),
            },
            text_style={
                'verbosity': 'warm',
                'detail_level': 'other_focused',
                'coherence': 'high',
                'sentiment_alignment': 'caring'
            },
            applicable_domains=['social_psychology', 'prosocial_behavior', 'cooperation', 'altruism',
                                'behavioral_economics', 'economic_games', 'trust', 'intergroup_relations']
        )

        personas['individualist'] = Persona(
            name="Individualist",
            category="social",
            description="Person focused on personal goals and self-interest. Lower concern for "
                       "collective outcomes, competitive orientation. "
                       "Van Lange et al. (1997): Individualists maximize own outcomes.",
            weight=0.12,
            traits={
                'individualism': PersonaTrait('individualism', 0.82, 0.08, 'High'),
                'competition_orientation': PersonaTrait('competition_orientation', 0.75, 0.10, 'Competitive'),
                'self_interest': PersonaTrait('self_interest', 0.78, 0.10, 'Self-focused'),
                'trust_propensity': PersonaTrait('trust_propensity', 0.45, 0.12, 'Lower trust'),
                'cooperation_tendency': PersonaTrait('cooperation_tendency', 0.35, 0.12, 'Lower'),
                'status_concern': PersonaTrait('status_concern', 0.70, 0.10, 'Status-seeking'),
                # Scientific calibration - self-focused, competitive
                'response_tendency': PersonaTrait('response_tendency', 0.58, 0.10, 'Self-interest focused'),
                'extremity': PersonaTrait('extremity', 0.35, 0.10, 'Moderate - strong personal preferences'),
            },
            text_style={
                'verbosity': 'direct',
                'detail_level': 'self_focused',
                'coherence': 'high',
                'sentiment_alignment': 'achievement_oriented'
            },
            applicable_domains=['social_psychology', 'cooperation', 'competition', 'social_dilemmas',
                                'behavioral_economics', 'economic_games', 'consumer_behavior',
                                'organizational_behavior', 'political_psychology']
        )

        personas['conformist'] = Persona(
            name="Conformist",
            category="social",
            description="Person who tends to follow social norms and group opinions. High need for "
                       "belonging, influenced by majority views. "
                       "Asch (1956): ~37% conform to incorrect majority in line judgment.",
            weight=0.12,
            traits={
                'conformity': PersonaTrait('conformity', 0.80, 0.08, 'High conformity'),
                'need_for_belonging': PersonaTrait('need_for_belonging', 0.78, 0.10, 'High'),
                'social_influence_susceptibility': PersonaTrait('social_influence_susceptibility', 0.75, 0.10, 'High'),
                'uniqueness_seeking': PersonaTrait('uniqueness_seeking', 0.30, 0.10, 'Low'),
                'opinion_leadership': PersonaTrait('opinion_leadership', 0.35, 0.12, 'Low'),
                'social_anxiety': PersonaTrait('social_anxiety', 0.55, 0.12, 'Moderate'),
                # Scientific calibration - agreeable, norm-following
                'response_tendency': PersonaTrait('response_tendency', 0.60, 0.08, 'Positive - agreeable'),
                'extremity': PersonaTrait('extremity', 0.15, 0.08, 'Low - avoids standing out'),
            },
            text_style={
                'verbosity': 'moderate',
                'detail_level': 'norm_following',
                'coherence': 'high',
                'sentiment_alignment': 'agreeable'
            },
            applicable_domains=['social_psychology', 'conformity', 'social_influence', 'norms',
                                'political_psychology', 'organizational_behavior', 'consumer_behavior']
        )

        # ================================================================
        # POLITICAL / PARTISAN PERSONAS (v1.0.4.6)
        # Scientific basis: Iyengar & Westwood (2015), Druckman & Levendusky (2019)
        # Political identity is a primary predictor of behavior in political studies
        # ================================================================

        personas['partisan_ideologue'] = Persona(
            name="Partisan Ideologue",
            category="political",
            description="Strongly identified partisan with ideological consistency. High affective "
                       "polarization, strong ingroup loyalty. "
                       "Iyengar & Westwood (2015): Affective polarization d ≈ 0.8-1.0.",
            weight=0.14,
            traits={
                'political_engagement': PersonaTrait('political_engagement', 0.88, 0.06, 'Very high'),
                'ideological_consistency': PersonaTrait('ideological_consistency', 0.85, 0.08, 'High'),
                'affective_polarization': PersonaTrait('affective_polarization', 0.82, 0.08, 'Strong'),
                'ingroup_loyalty': PersonaTrait('ingroup_loyalty', 0.85, 0.08, 'Very strong'),
                'outgroup_hostility': PersonaTrait('outgroup_hostility', 0.72, 0.10, 'High'),
                'moral_conviction': PersonaTrait('moral_conviction', 0.78, 0.10, 'Strong'),
                'response_tendency': PersonaTrait('response_tendency', 0.55, 0.15, 'Condition-dependent extreme'),
                'extremity': PersonaTrait('extremity', 0.72, 0.10, 'High - strong partisan reactions'),
            },
            text_style={
                'verbosity': 'passionate',
                'detail_level': 'ideology_driven',
                'coherence': 'high',
                'sentiment_alignment': 'partisan'
            },
            applicable_domains=['political_psychology', 'political_science', 'intergroup_relations',
                                'moral_psychology', 'social_psychology', 'economic_games']
        )

        personas['pragmatic_moderate'] = Persona(
            name="Pragmatic Moderate",
            category="political",
            description="Centrist/moderate with low partisanship, open to both sides. "
                       "Values compromise and evidence over ideology. "
                       "Fiorina et al. (2005): ~40% of Americans are genuine moderates.",
            weight=0.12,
            traits={
                'political_engagement': PersonaTrait('political_engagement', 0.55, 0.12, 'Moderate'),
                'ideological_consistency': PersonaTrait('ideological_consistency', 0.35, 0.12, 'Low'),
                'affective_polarization': PersonaTrait('affective_polarization', 0.30, 0.12, 'Low'),
                'openness_to_other_side': PersonaTrait('openness_to_other_side', 0.75, 0.10, 'High'),
                'compromise_seeking': PersonaTrait('compromise_seeking', 0.78, 0.10, 'Strong'),
                'evidence_orientation': PersonaTrait('evidence_orientation', 0.72, 0.10, 'High'),
                'response_tendency': PersonaTrait('response_tendency', 0.50, 0.08, 'Centrist/balanced'),
                'extremity': PersonaTrait('extremity', 0.18, 0.08, 'Low - measured responses'),
            },
            text_style={
                'verbosity': 'measured',
                'detail_level': 'balanced',
                'coherence': 'high',
                'sentiment_alignment': 'nuanced'
            },
            applicable_domains=['political_psychology', 'political_science', 'intergroup_relations',
                                'moral_psychology', 'social_psychology']
        )

        # ================================================================
        # ECONOMIC GAME PERSONAS (v1.0.4.6)
        # Scientific basis: Fehr & Schmidt (1999), Fischbacher et al. (2001)
        # Economic game behavior clusters into distinct types
        # ================================================================

        personas['reciprocal_cooperator'] = Persona(
            name="Reciprocal Cooperator",
            category="behavioral_economics",
            description="Conditional cooperator who matches others' contributions. "
                       "Cooperates when others cooperate, retaliates against free-riders. "
                       "Fischbacher et al. (2001): ~50% of participants are conditional cooperators.",
            weight=0.15,
            traits={
                'reciprocity': PersonaTrait('reciprocity', 0.85, 0.08, 'Very high'),
                'fairness_concern': PersonaTrait('fairness_concern', 0.78, 0.10, 'High'),
                'cooperation_tendency': PersonaTrait('cooperation_tendency', 0.70, 0.10, 'Conditionally high'),
                'punishment_willingness': PersonaTrait('punishment_willingness', 0.65, 0.12, 'Moderate-high'),
                'trust_propensity': PersonaTrait('trust_propensity', 0.62, 0.12, 'Moderate'),
                'strategic_thinking': PersonaTrait('strategic_thinking', 0.68, 0.10, 'Moderate-high'),
                'response_tendency': PersonaTrait('response_tendency', 0.55, 0.12, 'Context-dependent'),
                'extremity': PersonaTrait('extremity', 0.30, 0.10, 'Moderate - depends on partner'),
            },
            text_style={
                'verbosity': 'moderate',
                'detail_level': 'strategic',
                'coherence': 'high',
                'sentiment_alignment': 'conditional'
            },
            applicable_domains=['behavioral_economics', 'economic_games', 'cooperation',
                                'social_psychology', 'trust', 'intergroup_relations']
        )

        personas['free_rider'] = Persona(
            name="Free Rider",
            category="behavioral_economics",
            description="Selfish maximizer who contributes minimally in collective action problems. "
                       "Exploits others' cooperation while minimizing own costs. "
                       "Fischbacher et al. (2001): ~30% are free-riders in public goods games.",
            weight=0.10,
            traits={
                'self_interest': PersonaTrait('self_interest', 0.85, 0.08, 'Very high'),
                'cooperation_tendency': PersonaTrait('cooperation_tendency', 0.20, 0.10, 'Very low'),
                'strategic_thinking': PersonaTrait('strategic_thinking', 0.75, 0.10, 'High - strategic exploitation'),
                'fairness_concern': PersonaTrait('fairness_concern', 0.30, 0.12, 'Low'),
                'guilt_proneness': PersonaTrait('guilt_proneness', 0.28, 0.12, 'Low'),
                'trust_propensity': PersonaTrait('trust_propensity', 0.40, 0.12, 'Low-moderate'),
                'response_tendency': PersonaTrait('response_tendency', 0.30, 0.12, 'Favors low contributions'),
                'extremity': PersonaTrait('extremity', 0.42, 0.10, 'Moderate-high - consistent selfishness'),
            },
            text_style={
                'verbosity': 'terse',
                'detail_level': 'self_focused',
                'coherence': 'moderate',
                'sentiment_alignment': 'self_interested'
            },
            applicable_domains=['behavioral_economics', 'economic_games', 'cooperation',
                                'social_dilemmas', 'social_psychology']
        )

        personas['fairness_enforcer'] = Persona(
            name="Fairness Enforcer",
            category="behavioral_economics",
            description="Strong fairness norms, willing to punish norm violators even at personal cost. "
                       "Altruistic punishment and third-party enforcement. "
                       "Fehr & Gächter (2002): ~60% engage in costly punishment of free-riders.",
            weight=0.12,
            traits={
                'fairness_concern': PersonaTrait('fairness_concern', 0.88, 0.06, 'Very high'),
                'punishment_willingness': PersonaTrait('punishment_willingness', 0.82, 0.08, 'Very high'),
                'moral_conviction': PersonaTrait('moral_conviction', 0.80, 0.08, 'Strong'),
                'inequality_aversion': PersonaTrait('inequality_aversion', 0.82, 0.08, 'Strong'),
                'cooperation_tendency': PersonaTrait('cooperation_tendency', 0.72, 0.10, 'High'),
                'anger_at_injustice': PersonaTrait('anger_at_injustice', 0.78, 0.10, 'High'),
                'response_tendency': PersonaTrait('response_tendency', 0.60, 0.12, 'Fair-share promoting'),
                'extremity': PersonaTrait('extremity', 0.45, 0.10, 'Moderate-high on fairness issues'),
            },
            text_style={
                'verbosity': 'principled',
                'detail_level': 'norm_focused',
                'coherence': 'high',
                'sentiment_alignment': 'justice_oriented'
            },
            applicable_domains=['behavioral_economics', 'economic_games', 'punishment',
                                'moral_psychology', 'fairness', 'social_psychology', 'legal']
        )

        # ================================================================
        # INTERGROUP PERSONAS (v1.0.4.6)
        # Scientific basis: Tajfel & Turner (1979), Allport (1954)
        # Intergroup attitudes vary from strong ingroup bias to egalitarian
        # ================================================================

        personas['ingroup_favorer'] = Persona(
            name="Ingroup Favorer",
            category="intergroup",
            description="Strong ingroup identification and favoritism. Minimal Group Paradigm shows "
                       "even arbitrary groups elicit discrimination. "
                       "Balliet et al. (2014): d = 0.32 for ingroup favoritism in cooperation.",
            weight=0.14,
            traits={
                'ingroup_identification': PersonaTrait('ingroup_identification', 0.85, 0.08, 'Very strong'),
                'outgroup_derogation': PersonaTrait('outgroup_derogation', 0.62, 0.12, 'Moderate-high'),
                'social_identity_salience': PersonaTrait('social_identity_salience', 0.80, 0.08, 'High'),
                'ethnocentrism': PersonaTrait('ethnocentrism', 0.68, 0.12, 'Moderate-high'),
                'cooperation_tendency': PersonaTrait('cooperation_tendency', 0.75, 0.12, 'High ingroup, low outgroup'),
                'trust_propensity': PersonaTrait('trust_propensity', 0.65, 0.15, 'High ingroup, low outgroup'),
                'response_tendency': PersonaTrait('response_tendency', 0.60, 0.15, 'Ingroup-biased'),
                'extremity': PersonaTrait('extremity', 0.42, 0.10, 'Moderate-high on group issues'),
            },
            text_style={
                'verbosity': 'group_focused',
                'detail_level': 'identity_aware',
                'coherence': 'high',
                'sentiment_alignment': 'ingroup_positive'
            },
            applicable_domains=['intergroup_relations', 'social_psychology', 'political_psychology',
                                'economic_games', 'prejudice', 'social_identity', 'trust']
        )

        personas['egalitarian'] = Persona(
            name="Egalitarian",
            category="intergroup",
            description="Committed to fairness across groups. Low prejudice, high need for "
                       "equality. Values diversity and cross-group contact. "
                       "Pratto et al. (1994): Low SDO predicts egalitarian attitudes.",
            weight=0.12,
            traits={
                'egalitarianism': PersonaTrait('egalitarianism', 0.85, 0.08, 'Very high'),
                'social_dominance_orientation': PersonaTrait('social_dominance_orientation', 0.18, 0.08, 'Very low'),
                'outgroup_empathy': PersonaTrait('outgroup_empathy', 0.78, 0.10, 'High'),
                'contact_seeking': PersonaTrait('contact_seeking', 0.72, 0.10, 'High'),
                'fairness_concern': PersonaTrait('fairness_concern', 0.80, 0.08, 'Strong'),
                'cooperation_tendency': PersonaTrait('cooperation_tendency', 0.75, 0.10, 'High across groups'),
                'response_tendency': PersonaTrait('response_tendency', 0.55, 0.10, 'Equal across groups'),
                'extremity': PersonaTrait('extremity', 0.32, 0.10, 'Moderate on equality issues'),
            },
            text_style={
                'verbosity': 'thoughtful',
                'detail_level': 'fairness_focused',
                'coherence': 'high',
                'sentiment_alignment': 'equality_oriented'
            },
            applicable_domains=['intergroup_relations', 'social_psychology', 'political_psychology',
                                'prejudice', 'social_identity', 'moral_psychology', 'economic_games']
        )

        # ================================================================
        # PSYCHOLOGICAL OWNERSHIP PERSONAS
        # Scientific basis: Psychological ownership theory, endowment effect.
        # References: Pierce et al. (2003), Kahneman et al. (1990)
        # Response calibrated to ownership and possession research
        # ================================================================

        personas['high_ownership'] = Persona(
            name="High Psychological Ownership",
            category="ownership",
            description="Person who readily develops psychological ownership feelings. Strong "
                       "sense of 'mine', territorial, invests self in objects/ideas. "
                       "Pierce et al. (2003): Strong control, intimate knowledge, self-investment.",
            weight=0.15,
            traits={
                'ownership_tendency': PersonaTrait('ownership_tendency', 0.85, 0.08, 'Very high'),
                'territorial_behavior': PersonaTrait('territorial_behavior', 0.78, 0.10, 'High'),
                'self_extension': PersonaTrait('self_extension', 0.80, 0.08, 'Strong'),
                'control_need': PersonaTrait('control_need', 0.75, 0.10, 'High control need'),
                'attachment_style': PersonaTrait('attachment_style', 0.72, 0.10, 'Strong attachment'),
                'endowment_effect': PersonaTrait('endowment_effect', 0.78, 0.10, 'Strong'),
                # Scientific calibration - possessive, attached
                'response_tendency': PersonaTrait('response_tendency', 0.72, 0.10, 'Positive toward owned items'),
                'extremity': PersonaTrait('extremity', 0.38, 0.10, 'Moderate-high - strong ownership feelings'),
            },
            text_style={
                'verbosity': 'possessive',
                'detail_level': 'ownership_focused',
                'coherence': 'high',
                'sentiment_alignment': 'attached'
            },
            applicable_domains=['psychological_ownership', 'consumer_behavior', 'branding',
                                'political_psychology', 'sports', 'intergroup_relations', 'social_psychology']
        )

        personas['low_ownership'] = Persona(
            name="Low Psychological Ownership",
            category="ownership",
            description="Person who rarely develops ownership feelings. Detached, sharing-oriented, "
                       "less affected by endowment effects. "
                       "Belk (2010): Sharing-oriented individuals show reduced possession attachment.",
            weight=0.10,
            traits={
                'ownership_tendency': PersonaTrait('ownership_tendency', 0.30, 0.10, 'Low'),
                'territorial_behavior': PersonaTrait('territorial_behavior', 0.32, 0.12, 'Low'),
                'self_extension': PersonaTrait('self_extension', 0.35, 0.12, 'Weak'),
                'control_need': PersonaTrait('control_need', 0.40, 0.12, 'Lower control need'),
                'sharing_orientation': PersonaTrait('sharing_orientation', 0.75, 0.10, 'Sharing-oriented'),
                'endowment_effect': PersonaTrait('endowment_effect', 0.35, 0.12, 'Weak'),
                # Scientific calibration - detached, functional
                'response_tendency': PersonaTrait('response_tendency', 0.50, 0.10, 'Neutral toward possessions'),
                'extremity': PersonaTrait('extremity', 0.18, 0.08, 'Low - less invested'),
            },
            text_style={
                'verbosity': 'detached',
                'detail_level': 'functional',
                'coherence': 'high',
                'sentiment_alignment': 'neutral'
            },
            applicable_domains=['psychological_ownership', 'sharing_economy', 'access_based']
        )

        # ================================================================
        # HEALTH PSYCHOLOGY PERSONAS
        # Scientific basis: Health belief model, health locus of control.
        # References: Rosenstock (1974), Wallston et al. (1978)
        # Response calibrated to health psychology research norms
        # ================================================================

        personas['health_conscious'] = Persona(
            name="Health-Conscious Individual",
            category="health",
            description="Person highly attentive to health behaviors, nutrition, and wellness. "
                       "Proactive about prevention, health-motivated decisions. "
                       "Rosenstock (1974): High perceived susceptibility and self-efficacy.",
            weight=0.15,
            traits={
                'health_consciousness': PersonaTrait('health_consciousness', 0.85, 0.08, 'Very high'),
                'prevention_focus': PersonaTrait('prevention_focus', 0.80, 0.08, 'Prevention-oriented'),
                'self_efficacy_health': PersonaTrait('self_efficacy_health', 0.78, 0.10, 'High'),
                'health_locus_control': PersonaTrait('health_locus_control', 0.75, 0.10, 'Internal'),
                'risk_perception': PersonaTrait('risk_perception', 0.70, 0.10, 'Moderate-high'),
                'information_seeking_health': PersonaTrait('information_seeking_health', 0.80, 0.08, 'Seeks info'),
                # Scientific calibration - proactive, health-focused
                'response_tendency': PersonaTrait('response_tendency', 0.68, 0.10, 'Positive toward health behaviors'),
                'extremity': PersonaTrait('extremity', 0.30, 0.10, 'Moderate - strong health opinions'),
            },
            text_style={
                'verbosity': 'detailed',
                'detail_level': 'health_focused',
                'coherence': 'high',
                'sentiment_alignment': 'wellness_oriented'
            },
            applicable_domains=['health_psychology', 'health_behavior', 'prevention', 'nutrition']
        )

        personas['health_fatalist'] = Persona(
            name="Health Fatalist",
            category="health",
            description="Person who believes health outcomes are largely outside personal control. "
                       "Lower engagement with health behaviors, external locus of control. "
                       "Wallston et al. (1978): High chance/powerful others on MHLC.",
            weight=0.10,
            traits={
                'health_consciousness': PersonaTrait('health_consciousness', 0.35, 0.12, 'Low'),
                'health_locus_control': PersonaTrait('health_locus_control', 0.30, 0.10, 'External'),
                'self_efficacy_health': PersonaTrait('self_efficacy_health', 0.35, 0.12, 'Low'),
                'prevention_focus': PersonaTrait('prevention_focus', 0.35, 0.12, 'Low'),
                'risk_perception': PersonaTrait('risk_perception', 0.50, 0.15, 'Variable'),
                'optimistic_bias': PersonaTrait('optimistic_bias', 0.65, 0.12, 'Some denial'),
                # Scientific calibration - resigned, external
                'response_tendency': PersonaTrait('response_tendency', 0.45, 0.12, 'Lower engagement'),
                'extremity': PersonaTrait('extremity', 0.22, 0.10, 'Lower - less invested in health'),
            },
            text_style={
                'verbosity': 'brief',
                'detail_level': 'fatalistic',
                'coherence': 'moderate',
                'sentiment_alignment': 'resigned'
            },
            applicable_domains=['health_psychology', 'health_behavior', 'health_communication']
        )

        # ================================================================
        # ENVIRONMENTAL PSYCHOLOGY PERSONAS
        # Scientific basis: New Ecological Paradigm, value-belief-norm theory.
        # References: Dunlap et al. (2000), Stern et al. (1999)
        # Response calibrated to environmental psychology norms
        # ================================================================

        personas['eco_warrior'] = Persona(
            name="Environmental Activist",
            category="environmental",
            description="Strongly pro-environmental individual. High environmental concern, "
                       "engages in sustainable behaviors, willing to sacrifice for environment. "
                       "Dunlap et al. (2000): NEP scores M=5.8/7 for environmentalists.",
            weight=0.10,
            traits={
                'environmental_concern': PersonaTrait('environmental_concern', 0.90, 0.06, 'Very high'),
                'biospheric_values': PersonaTrait('biospheric_values', 0.88, 0.07, 'Strong'),
                'sustainable_behavior': PersonaTrait('sustainable_behavior', 0.85, 0.08, 'Consistent'),
                'sacrifice_willingness': PersonaTrait('sacrifice_willingness', 0.78, 0.10, 'High'),
                'climate_anxiety': PersonaTrait('climate_anxiety', 0.70, 0.12, 'Moderate-high'),
                'environmental_identity': PersonaTrait('environmental_identity', 0.85, 0.08, 'Core identity'),
                # Scientific calibration - passionate, committed
                'response_tendency': PersonaTrait('response_tendency', 0.78, 0.08, 'Strongly pro-environmental'),
                'extremity': PersonaTrait('extremity', 0.48, 0.10, 'Higher - passionate advocacy'),
            },
            text_style={
                'verbosity': 'passionate',
                'detail_level': 'environmental_focused',
                'coherence': 'high',
                'sentiment_alignment': 'urgency'
            },
            applicable_domains=['environmental_psychology', 'sustainability', 'climate', 'green_consumption']
        )

        personas['environmental_skeptic'] = Persona(
            name="Environmental Skeptic",
            category="environmental",
            description="Skeptical of environmental claims and climate urgency. Lower environmental "
                       "concern, prioritizes economic considerations. "
                       "McCright & Dunlap (2011): Climate skeptics show NEP M=3.2/7.",
            weight=0.08,
            traits={
                'environmental_concern': PersonaTrait('environmental_concern', 0.30, 0.12, 'Low'),
                'climate_skepticism': PersonaTrait('climate_skepticism', 0.75, 0.10, 'Skeptical'),
                'sustainable_behavior': PersonaTrait('sustainable_behavior', 0.35, 0.12, 'Low'),
                'sacrifice_willingness': PersonaTrait('sacrifice_willingness', 0.25, 0.10, 'Low'),
                'economic_priority': PersonaTrait('economic_priority', 0.80, 0.08, 'Economy first'),
                'biospheric_values': PersonaTrait('biospheric_values', 0.30, 0.12, 'Lower'),
                # Scientific calibration - skeptical, economy-focused
                'response_tendency': PersonaTrait('response_tendency', 0.35, 0.10, 'Negative toward environmental claims'),
                'extremity': PersonaTrait('extremity', 0.38, 0.12, 'Moderate-high - strong skepticism'),
            },
            text_style={
                'verbosity': 'direct',
                'detail_level': 'economic_focused',
                'coherence': 'high',
                'sentiment_alignment': 'skeptical'
            },
            applicable_domains=['environmental_psychology', 'sustainability', 'climate', 'skepticism']
        )

        # ================================================================
        # CULTURAL RESPONSE STYLE PERSONAS (v2.4.5: NEW)
        # Scientific basis: Cross-cultural psychology, response style research.
        # References: Chen et al. (1995), Harzing (2006), Hui & Triandis (1989)
        # Response styles differ systematically across cultures
        # ================================================================

        personas['east_asian_response_style'] = Persona(
            name="East Asian Response Style",
            category="cultural",
            description="Response pattern typical of East Asian cultures (China, Japan, Korea). "
                       "Moderate/midpoint tendency, avoids extreme responses, socially harmonious. "
                       "Chen et al. (1995): East Asians show reduced ERS and higher midpoint use.",
            weight=0.08,
            traits={
                'midpoint_preference': PersonaTrait('midpoint_preference', 0.72, 0.10, 'High midpoint use'),
                'extremity': PersonaTrait('extremity', 0.15, 0.08, 'Very low - avoids endpoints'),
                'social_desirability': PersonaTrait('social_desirability', 0.68, 0.10, 'Moderate-high'),
                'acquiescence': PersonaTrait('acquiescence', 0.50, 0.10, 'Neutral'),
                'collectivism': PersonaTrait('collectivism', 0.78, 0.10, 'High'),
                'harmony_seeking': PersonaTrait('harmony_seeking', 0.80, 0.08, 'High'),
                'response_tendency': PersonaTrait('response_tendency', 0.52, 0.08, 'Centered - moderation'),
                'scale_use_breadth': PersonaTrait('scale_use_breadth', 0.45, 0.10, 'Restricted - center bias'),
            },
            text_style={
                'verbosity': 'moderate',
                'detail_level': 'nuanced',
                'coherence': 'high',
                'sentiment_alignment': 'balanced'
            },
            applicable_domains=['all', 'cross_cultural']
        )

        personas['latin_response_style'] = Persona(
            name="Latin Response Style",
            category="cultural",
            description="Response pattern typical of Latin American and Mediterranean cultures. "
                       "Higher extreme responding, expressive, acquiescence tendency. "
                       "Harzing (2006): Latin cultures show elevated ERS and acquiescence.",
            weight=0.08,
            traits={
                'midpoint_preference': PersonaTrait('midpoint_preference', 0.30, 0.12, 'Low midpoint'),
                'extremity': PersonaTrait('extremity', 0.70, 0.10, 'Higher - expressive'),
                'social_desirability': PersonaTrait('social_desirability', 0.62, 0.10, 'Moderate-high'),
                'acquiescence': PersonaTrait('acquiescence', 0.68, 0.10, 'Elevated'),
                'expressiveness': PersonaTrait('expressiveness', 0.78, 0.10, 'High'),
                'warmth': PersonaTrait('warmth', 0.75, 0.10, 'Warm communication'),
                'response_tendency': PersonaTrait('response_tendency', 0.62, 0.12, 'Positive tendency'),
                'scale_use_breadth': PersonaTrait('scale_use_breadth', 0.75, 0.10, 'Wide range use'),
            },
            text_style={
                'verbosity': 'expressive',
                'detail_level': 'emotional',
                'coherence': 'high',
                'sentiment_alignment': 'warm'
            },
            applicable_domains=['all', 'cross_cultural']
        )

        personas['nordic_response_style'] = Persona(
            name="Nordic Response Style",
            category="cultural",
            description="Response pattern typical of Scandinavian cultures (Sweden, Norway, Denmark). "
                       "Direct, low social desirability, balanced scale use. "
                       "Hofstede (1980): Nordic cultures show low power distance, direct communication.",
            weight=0.06,
            traits={
                'midpoint_preference': PersonaTrait('midpoint_preference', 0.45, 0.10, 'Moderate'),
                'extremity': PersonaTrait('extremity', 0.32, 0.10, 'Moderate-low'),
                'social_desirability': PersonaTrait('social_desirability', 0.35, 0.10, 'Low - direct'),
                'acquiescence': PersonaTrait('acquiescence', 0.42, 0.10, 'Low'),
                'egalitarianism': PersonaTrait('egalitarianism', 0.85, 0.08, 'High'),
                'directness': PersonaTrait('directness', 0.78, 0.10, 'Direct communication'),
                'response_tendency': PersonaTrait('response_tendency', 0.55, 0.10, 'Balanced'),
                'scale_use_breadth': PersonaTrait('scale_use_breadth', 0.68, 0.10, 'Thoughtful range'),
            },
            text_style={
                'verbosity': 'concise',
                'detail_level': 'factual',
                'coherence': 'very_high',
                'sentiment_alignment': 'direct'
            },
            applicable_domains=['all', 'cross_cultural']
        )

        personas['middle_eastern_response_style'] = Persona(
            name="Middle Eastern Response Style",
            category="cultural",
            description="Response pattern typical of Middle Eastern cultures. "
                       "Higher acquiescence, social desirability, relationship-focused. "
                       "Hui & Triandis (1989): High collectivism cultures show elevated ARS.",
            weight=0.06,
            traits={
                'midpoint_preference': PersonaTrait('midpoint_preference', 0.35, 0.12, 'Lower'),
                'extremity': PersonaTrait('extremity', 0.55, 0.12, 'Moderate-high'),
                'social_desirability': PersonaTrait('social_desirability', 0.72, 0.10, 'High'),
                'acquiescence': PersonaTrait('acquiescence', 0.70, 0.10, 'Elevated'),
                'collectivism': PersonaTrait('collectivism', 0.80, 0.08, 'High'),
                'hospitality': PersonaTrait('hospitality', 0.82, 0.08, 'Very high'),
                'response_tendency': PersonaTrait('response_tendency', 0.65, 0.10, 'Positive tendency'),
                'scale_use_breadth': PersonaTrait('scale_use_breadth', 0.65, 0.12, 'Moderately wide'),
            },
            text_style={
                'verbosity': 'elaborate',
                'detail_level': 'relationship_focused',
                'coherence': 'high',
                'sentiment_alignment': 'hospitable'
            },
            applicable_domains=['all', 'cross_cultural']
        )

        # ================================================================
        # GENERATIONAL PERSONAS (v2.4.5: NEW)
        # Scientific basis: Generational cohort research, Pew Research Center.
        # References: Twenge (2017), Howe & Strauss (2000)
        # ================================================================

        personas['gen_z_digital_native'] = Persona(
            name="Gen Z Digital Native",
            category="generational",
            description="Younger respondent (born 1997-2012) with digital-first mindset. "
                       "Quick responses, comfortable with technology, values authenticity. "
                       "Twenge (2017): iGen shows different communication patterns.",
            weight=0.10,
            traits={
                'digital_fluency': PersonaTrait('digital_fluency', 0.95, 0.04, 'Very high'),
                'attention_span': PersonaTrait('attention_span', 0.55, 0.15, 'Variable'),
                'authenticity_value': PersonaTrait('authenticity_value', 0.82, 0.08, 'High'),
                'social_media_use': PersonaTrait('social_media_use', 0.88, 0.08, 'Heavy'),
                'environmental_concern': PersonaTrait('environmental_concern', 0.72, 0.12, 'Higher'),
                'diversity_acceptance': PersonaTrait('diversity_acceptance', 0.85, 0.08, 'High'),
                'response_tendency': PersonaTrait('response_tendency', 0.58, 0.12, 'Varied'),
                'reading_speed': PersonaTrait('reading_speed', 0.78, 0.10, 'Fast'),
            },
            text_style={
                'verbosity': 'brief',
                'detail_level': 'informal',
                'coherence': 'moderate',
                'sentiment_alignment': 'authentic'
            },
            applicable_domains=['all', 'technology', 'social_media', 'generational']
        )

        personas['boomer_traditional'] = Persona(
            name="Baby Boomer Traditional",
            category="generational",
            description="Older respondent (born 1946-1964) with more traditional communication. "
                       "Thorough responses, values quality, may be less tech-comfortable. "
                       "Pew Research: Boomers show distinct survey response patterns.",
            weight=0.08,
            traits={
                'digital_fluency': PersonaTrait('digital_fluency', 0.55, 0.15, 'Variable'),
                'thoroughness': PersonaTrait('thoroughness', 0.78, 0.10, 'High'),
                'traditional_values': PersonaTrait('traditional_values', 0.72, 0.12, 'Higher'),
                'brand_loyalty': PersonaTrait('brand_loyalty', 0.75, 0.10, 'Strong'),
                'quality_focus': PersonaTrait('quality_focus', 0.80, 0.08, 'High'),
                'patience': PersonaTrait('patience', 0.72, 0.10, 'Higher'),
                'response_tendency': PersonaTrait('response_tendency', 0.60, 0.10, 'Thoughtful'),
                'reading_speed': PersonaTrait('reading_speed', 0.55, 0.12, 'Careful'),
            },
            text_style={
                'verbosity': 'detailed',
                'detail_level': 'thorough',
                'coherence': 'high',
                'sentiment_alignment': 'considered'
            },
            applicable_domains=['all', 'consumer_behavior', 'generational']
        )

        # ================================================================
        # CLINICAL/ANXIETY PERSONAS (v1.0.4.4)
        # Scientific basis: Clark & Watson (1991) tripartite model,
        # Barlow (2002) hierarchical model of anxiety.
        # Calibrations from published norms on BAI, STAI, BDI-II.
        # ================================================================

        personas['anxious_individual'] = Persona(
            name="Anxious Individual",
            category="clinical",
            description="Person with elevated trait anxiety. Hypervigilant to threat, "
                       "risk-averse, tendency toward negative interpretations. "
                       "Clark & Watson (1991): High negative affect, low positive affect. "
                       "Spielberger (1983): Trait anxiety M=5.2/7 for clinical-adjacent.",
            weight=0.08,
            traits={
                'anxiety_level': PersonaTrait('anxiety_level', 0.78, 0.10, 'Elevated trait anxiety'),
                'negative_affect': PersonaTrait('negative_affect', 0.75, 0.10, 'High NA'),
                'risk_tolerance': PersonaTrait('risk_tolerance', 0.25, 0.10, 'Very risk-averse'),
                'uncertainty_intolerance': PersonaTrait('uncertainty_intolerance', 0.80, 0.08, 'High IU'),
                'attention_to_threat': PersonaTrait('attention_to_threat', 0.82, 0.08, 'Threat-vigilant'),
                'avoidance_tendency': PersonaTrait('avoidance_tendency', 0.72, 0.10, 'Avoidant coping'),
                # Anxiety increases engagement (hypervigilance) but reduces confidence
                'response_tendency': PersonaTrait('response_tendency', 0.40, 0.12, 'Negative-skewed'),
                'extremity': PersonaTrait('extremity', 0.35, 0.12, 'Moderate — worry amplifies'),
            },
            text_style={
                'verbosity': 'moderate',
                'detail_level': 'worry_focused',
                'coherence': 'high',
                'sentiment_alignment': 'apprehensive'
            },
            applicable_domains=['clinical', 'anxiety', 'mental_health', 'health_psychology', 'stress']
        )

        personas['resilient_individual'] = Persona(
            name="Resilient Individual",
            category="clinical",
            description="Psychologically resilient person who adapts well to adversity. "
                       "High self-efficacy, positive reappraisal, effective coping. "
                       "Connor & Davidson (2003): CD-RISC M=5.5/7 for resilient adults.",
            weight=0.10,
            traits={
                'resilience': PersonaTrait('resilience', 0.82, 0.08, 'High resilience'),
                'self_efficacy': PersonaTrait('self_efficacy', 0.80, 0.08, 'Strong self-efficacy'),
                'positive_reappraisal': PersonaTrait('positive_reappraisal', 0.78, 0.10, 'Active coper'),
                'emotional_stability': PersonaTrait('emotional_stability', 0.75, 0.10, 'Stable'),
                'growth_mindset': PersonaTrait('growth_mindset', 0.72, 0.10, 'Growth-oriented'),
                'social_support': PersonaTrait('social_support', 0.75, 0.10, 'Well-connected'),
                'response_tendency': PersonaTrait('response_tendency', 0.65, 0.10, 'Positive-adaptive'),
                'extremity': PersonaTrait('extremity', 0.22, 0.08, 'Low — balanced perspective'),
            },
            text_style={
                'verbosity': 'moderate',
                'detail_level': 'solution_focused',
                'coherence': 'high',
                'sentiment_alignment': 'adaptive'
            },
            applicable_domains=['clinical', 'resilience', 'mental_health', 'positive_psychology', 'stress']
        )

        # ================================================================
        # LEGAL/FORENSIC PERSONA (v1.0.4.4)
        # Scientific basis: Tyler (2006) procedural justice model,
        # Pennington & Hastie (1986) story model of juror decision-making.
        # ================================================================

        personas['justice_oriented'] = Persona(
            name="Justice-Oriented Individual",
            category="legal",
            description="Person with strong concern for procedural and distributive justice. "
                       "Authority-sensitive, rule-following, literal interpretation. "
                       "Tyler (2006): Procedural justice M=5.0/7 for authority-compliant.",
            weight=0.08,
            traits={
                'justice_concern': PersonaTrait('justice_concern', 0.85, 0.08, 'Very high'),
                'authority_respect': PersonaTrait('authority_respect', 0.75, 0.10, 'High'),
                'rule_following': PersonaTrait('rule_following', 0.80, 0.08, 'Strict'),
                'fairness_concern': PersonaTrait('fairness_concern', 0.82, 0.08, 'Strong'),
                'punitive_tendency': PersonaTrait('punitive_tendency', 0.65, 0.12, 'Moderate-high'),
                'due_process_value': PersonaTrait('due_process_value', 0.78, 0.10, 'Important'),
                'response_tendency': PersonaTrait('response_tendency', 0.55, 0.10, 'Balanced-principled'),
                'extremity': PersonaTrait('extremity', 0.38, 0.10, 'Moderate — strong on principles'),
            },
            text_style={
                'verbosity': 'formal',
                'detail_level': 'principle_focused',
                'coherence': 'very_high',
                'sentiment_alignment': 'justice_driven'
            },
            applicable_domains=['legal', 'forensic', 'justice', 'punishment', 'fairness']
        )

        # ================================================================
        # SPORTS/COMPETITION PERSONA (v1.0.4.4)
        # Scientific basis: Vealey (1986) sport confidence model,
        # Duda (2001) achievement goal theory in sport.
        # ================================================================

        personas['competitive_achiever'] = Persona(
            name="Competitive Achiever",
            category="sports",
            description="Performance-oriented individual driven by achievement and competition. "
                       "High self-confidence, goal-directed, risk-tolerant in performance contexts. "
                       "Vealey (1986): Sport confidence M=5.5/7 for competitive athletes.",
            weight=0.08,
            traits={
                'achievement_motivation': PersonaTrait('achievement_motivation', 0.88, 0.07, 'Very high'),
                'competition_orientation': PersonaTrait('competition_orientation', 0.82, 0.08, 'Competitive'),
                'self_confidence': PersonaTrait('self_confidence', 0.78, 0.10, 'High'),
                'risk_tolerance': PersonaTrait('risk_tolerance', 0.70, 0.10, 'Higher'),
                'persistence': PersonaTrait('persistence', 0.82, 0.08, 'Very persistent'),
                'performance_focus': PersonaTrait('performance_focus', 0.85, 0.08, 'Results-driven'),
                'response_tendency': PersonaTrait('response_tendency', 0.68, 0.10, 'Positive — confident'),
                'extremity': PersonaTrait('extremity', 0.42, 0.10, 'Higher — strong convictions'),
            },
            text_style={
                'verbosity': 'direct',
                'detail_level': 'achievement_focused',
                'coherence': 'high',
                'sentiment_alignment': 'competitive'
            },
            applicable_domains=['sports', 'competition', 'achievement', 'performance', 'gaming',
                                'motivation', 'exercise', 'organizational_behavior']
        )

        # ================================================================
        # RELATIONSHIP/ATTACHMENT PERSONAS (v1.0.4.4)
        # Scientific basis: Brennan et al. (1998) ECR dimensions,
        # Fraley et al. (2000) adult attachment theory.
        # ================================================================

        personas['anxious_attachment'] = Persona(
            name="Anxiously Attached Individual",
            category="relationships",
            description="Person with attachment anxiety — fear of abandonment, need for closeness, "
                       "emotional volatility in relationships. "
                       "Brennan et al. (1998): ECR anxiety M=3.8/7 for anxious-preoccupied.",
            weight=0.08,
            traits={
                'attachment_anxiety': PersonaTrait('attachment_anxiety', 0.80, 0.10, 'High'),
                'relationship_worry': PersonaTrait('relationship_worry', 0.78, 0.10, 'Frequent'),
                'closeness_need': PersonaTrait('closeness_need', 0.85, 0.08, 'Very high'),
                'emotional_volatility': PersonaTrait('emotional_volatility', 0.72, 0.10, 'Higher'),
                'reassurance_seeking': PersonaTrait('reassurance_seeking', 0.78, 0.10, 'Frequent'),
                'rejection_sensitivity': PersonaTrait('rejection_sensitivity', 0.80, 0.08, 'High'),
                'response_tendency': PersonaTrait('response_tendency', 0.55, 0.15, 'Variable — mood-dependent'),
                'extremity': PersonaTrait('extremity', 0.45, 0.12, 'Higher — emotional intensity'),
            },
            text_style={
                'verbosity': 'emotional',
                'detail_level': 'relationship_focused',
                'coherence': 'moderate',
                'sentiment_alignment': 'ambivalent'
            },
            applicable_domains=['relationships', 'attachment', 'interpersonal', 'romantic']
        )

        personas['avoidant_attachment'] = Persona(
            name="Avoidantly Attached Individual",
            category="relationships",
            description="Person with attachment avoidance — discomfort with closeness, "
                       "self-reliance, emotional suppression. "
                       "Brennan et al. (1998): ECR avoidance M=3.5/7 for dismissive-avoidant.",
            weight=0.08,
            traits={
                'attachment_avoidance': PersonaTrait('attachment_avoidance', 0.78, 0.10, 'High'),
                'self_reliance': PersonaTrait('self_reliance', 0.82, 0.08, 'Very high'),
                'emotional_suppression': PersonaTrait('emotional_suppression', 0.75, 0.10, 'Suppresses'),
                'closeness_discomfort': PersonaTrait('closeness_discomfort', 0.78, 0.10, 'High'),
                'trust_propensity': PersonaTrait('trust_propensity', 0.35, 0.12, 'Low'),
                'vulnerability_avoidance': PersonaTrait('vulnerability_avoidance', 0.80, 0.08, 'Avoids'),
                'response_tendency': PersonaTrait('response_tendency', 0.48, 0.08, 'Neutral — emotionally flat'),
                'extremity': PersonaTrait('extremity', 0.15, 0.08, 'Low — suppressed responses'),
            },
            text_style={
                'verbosity': 'minimal',
                'detail_level': 'detached',
                'coherence': 'high',
                'sentiment_alignment': 'neutral_distant'
            },
            applicable_domains=['relationships', 'attachment', 'interpersonal', 'romantic']
        )

        # ================================================================
        # FINANCIAL DECISION PERSONA (v1.0.4.4)
        # Scientific basis: Barber & Odean (2001) overconfidence in markets,
        # Shefrin & Statman (1985) disposition effect.
        # ================================================================

        personas['overconfident_investor'] = Persona(
            name="Overconfident Decision Maker",
            category="financial",
            description="Person exhibiting overconfidence in judgments and decisions. "
                       "Excessive certainty, high trading frequency, underestimates risk. "
                       "Barber & Odean (2001): Overconfident investors trade 45% more.",
            weight=0.08,
            traits={
                'overconfidence': PersonaTrait('overconfidence', 0.82, 0.08, 'High overconfidence'),
                'risk_tolerance': PersonaTrait('risk_tolerance', 0.75, 0.10, 'Higher — underestimates risk'),
                'certainty_seeking': PersonaTrait('certainty_seeking', 0.35, 0.12, 'Low — comfortable with bets'),
                'information_seeking': PersonaTrait('information_seeking', 0.55, 0.12, 'Moderate — relies on self'),
                'anchoring_susceptibility': PersonaTrait('anchoring_susceptibility', 0.40, 0.12, 'Lower'),
                'optimism_bias': PersonaTrait('optimism_bias', 0.78, 0.10, 'Optimistic'),
                'response_tendency': PersonaTrait('response_tendency', 0.68, 0.10, 'Confident-positive'),
                'extremity': PersonaTrait('extremity', 0.45, 0.10, 'Higher — strong opinions'),
            },
            text_style={
                'verbosity': 'confident',
                'detail_level': 'assertive',
                'coherence': 'high',
                'sentiment_alignment': 'certain'
            },
            applicable_domains=['financial', 'investment', 'risk', 'behavioral_economics', 'judgment']
        )

        # ================================================================
        # MEDIA/COMMUNICATION PERSONA (v1.0.4.4)
        # Scientific basis: Friestad & Wright (1994) Persuasion Knowledge Model,
        # Petty et al. (2009) elaboration likelihood.
        # ================================================================

        personas['media_literate'] = Persona(
            name="Media-Literate Individual",
            category="communication",
            description="Person with high persuasion knowledge and media literacy. "
                       "Source-critical, recognizes persuasive tactics, systematic processor. "
                       "Friestad & Wright (1994): High PKM individuals resist persuasion attempts.",
            weight=0.08,
            traits={
                'persuasion_knowledge': PersonaTrait('persuasion_knowledge', 0.85, 0.08, 'High PKM'),
                'source_skepticism': PersonaTrait('source_skepticism', 0.78, 0.10, 'Critical'),
                'information_seeking': PersonaTrait('information_seeking', 0.82, 0.08, 'Active seeker'),
                'media_literacy': PersonaTrait('media_literacy', 0.85, 0.08, 'High'),
                'critical_thinking': PersonaTrait('critical_thinking', 0.80, 0.08, 'Strong'),
                'need_for_cognition': PersonaTrait('need_for_cognition', 0.75, 0.10, 'High'),
                'response_tendency': PersonaTrait('response_tendency', 0.50, 0.10, 'Neutral — skeptical'),
                'extremity': PersonaTrait('extremity', 0.22, 0.08, 'Low — nuanced analysis'),
            },
            text_style={
                'verbosity': 'analytical',
                'detail_level': 'critical',
                'coherence': 'very_high',
                'sentiment_alignment': 'evidence_based'
            },
            applicable_domains=['communication', 'media', 'persuasion', 'misinformation', 'advertising',
                                'health_psychology', 'political_psychology', 'consumer_behavior']
        )

        # ================================================================
        # v1.0.4.5: NEW DOMAIN-SPECIFIC PERSONAS
        # 6 additional personas for underserved research domains
        # ================================================================

        # ================================================================
        # CLINICAL/ANXIETY PERSONA
        # Scientific basis: Clark & Watson (1991) Tripartite Model,
        # Barlow (2002) anxiety as hypervigilance + avoidance.
        # ================================================================
        personas['clinical_anxious'] = Persona(
            name="High-Anxiety Individual",
            category="clinical",
            description="Person with elevated trait anxiety. Hypervigilant to threats, "
                       "avoidant of uncertainty, low confidence in decisions. "
                       "Clark & Watson (1991): Negative affect + physiological hyperarousal.",
            weight=0.07,
            traits={
                'anxiety_level': PersonaTrait('anxiety_level', 0.82, 0.08, 'High trait anxiety'),
                'risk_tolerance': PersonaTrait('risk_tolerance', 0.22, 0.10, 'Very risk-averse'),
                'certainty_seeking': PersonaTrait('certainty_seeking', 0.85, 0.08, 'High need for certainty'),
                'social_desirability': PersonaTrait('social_desirability', 0.60, 0.12, 'Moderate — may underreport'),
                'attention_level': PersonaTrait('attention_level', 0.72, 0.10, 'Hypervigilant'),
                'response_tendency': PersonaTrait('response_tendency', 0.38, 0.12, 'Negative-leaning'),
                'extremity': PersonaTrait('extremity', 0.35, 0.10, 'Moderate — strong on threat items'),
                'engagement': PersonaTrait('engagement', 0.65, 0.10, 'Moderate — anxious engagement'),
            },
            text_style={
                'verbosity': 'moderate',
                'detail_level': 'cautious',
                'coherence': 'moderate',
                'sentiment_alignment': 'negative_leaning'
            },
            applicable_domains=['clinical', 'anxiety', 'health', 'mental_health', 'stress', 'coping']
        )

        # ================================================================
        # LEGAL/FORENSIC PERSONA
        # Scientific basis: Tyler (2006) procedural justice theory,
        # Lind & Tyler (1988) authority legitimacy.
        # ================================================================
        personas['legal_authority_sensitive'] = Persona(
            name="Authority-Sensitive Individual",
            category="legal",
            description="Person highly responsive to authority and procedure. "
                       "Justice-focused, literal interpreter, rule-following. "
                       "Tyler (2006): Procedural fairness drives legitimacy perceptions.",
            weight=0.07,
            traits={
                'authority_sensitivity': PersonaTrait('authority_sensitivity', 0.82, 0.08, 'High deference'),
                'conformity': PersonaTrait('conformity', 0.72, 0.10, 'Rule-following'),
                'need_for_cognition': PersonaTrait('need_for_cognition', 0.55, 0.12, 'Moderate — follows rules'),
                'response_consistency': PersonaTrait('response_consistency', 0.80, 0.08, 'Very consistent'),
                'social_desirability': PersonaTrait('social_desirability', 0.58, 0.10, 'Moderate'),
                'response_tendency': PersonaTrait('response_tendency', 0.52, 0.10, 'Neutral — by the book'),
                'extremity': PersonaTrait('extremity', 0.18, 0.08, 'Low — measured responses'),
                'engagement': PersonaTrait('engagement', 0.70, 0.10, 'Attentive — takes seriously'),
            },
            text_style={
                'verbosity': 'precise',
                'detail_level': 'literal',
                'coherence': 'very_high',
                'sentiment_alignment': 'procedural'
            },
            applicable_domains=['legal', 'forensic', 'justice', 'authority', 'compliance', 'procedural']
        )

        # ================================================================
        # SPORTS/COMPETITION PERSONA
        # Scientific basis: Vealey (1986) sport confidence,
        # Deci & Ryan (2000) intrinsic motivation in competition.
        # ================================================================
        # (competitive_achiever consolidated with earlier definition — v1.0.4.6)

        # ================================================================
        # RELATIONSHIPS/ATTACHMENT PERSONA — Secure Attachment
        # Scientific basis: Brennan et al. (1998) ECR dimensions,
        # Mikulincer & Shaver (2007) attachment in adulthood.
        # ================================================================
        personas['secure_attached'] = Persona(
            name="Securely Attached Individual",
            category="relationships",
            description="Person with secure attachment style. Comfortable with intimacy "
                       "and autonomy, trusting of others, effective emotion regulation. "
                       "Brennan et al. (1998): Low anxiety, low avoidance on ECR.",
            weight=0.08,
            traits={
                'attachment_anxiety': PersonaTrait('attachment_anxiety', 0.22, 0.10, 'Low'),
                'attachment_avoidance': PersonaTrait('attachment_avoidance', 0.20, 0.10, 'Low'),
                'trust_tendency': PersonaTrait('trust_tendency', 0.75, 0.10, 'High — trusting'),
                'cooperation_tendency': PersonaTrait('cooperation_tendency', 0.72, 0.10, 'Cooperative'),
                'empathy': PersonaTrait('empathy', 0.70, 0.10, 'High empathy'),
                'response_tendency': PersonaTrait('response_tendency', 0.62, 0.10, 'Positive-leaning'),
                'extremity': PersonaTrait('extremity', 0.20, 0.08, 'Low — balanced'),
                'social_desirability': PersonaTrait('social_desirability', 0.50, 0.10, 'Moderate — honest'),
            },
            text_style={
                'verbosity': 'moderate',
                'detail_level': 'reflective',
                'coherence': 'high',
                'sentiment_alignment': 'positive_balanced'
            },
            applicable_domains=['relationships', 'attachment', 'intimacy', 'trust', 'social', 'interpersonal']
        )

        # ================================================================
        # FINANCIAL DECISION PERSONA — Loss-Averse Saver
        # Scientific basis: Kahneman & Tversky (1979) Prospect Theory,
        # Thaler (1999) mental accounting.
        # ================================================================
        personas['loss_averse_saver'] = Persona(
            name="Loss-Averse Saver",
            category="financial",
            description="Person with strong loss aversion and status quo bias. "
                       "Prefers safe options, avoids risk, values certainty. "
                       "Kahneman & Tversky (1979): Losses loom ~2.25× larger than gains.",
            weight=0.07,
            traits={
                'risk_tolerance': PersonaTrait('risk_tolerance', 0.18, 0.10, 'Very risk-averse'),
                'loss_aversion': PersonaTrait('loss_aversion', 0.85, 0.08, 'Strong loss aversion'),
                'certainty_seeking': PersonaTrait('certainty_seeking', 0.82, 0.08, 'High'),
                'status_quo_bias': PersonaTrait('status_quo_bias', 0.78, 0.10, 'Strong status quo preference'),
                'information_seeking': PersonaTrait('information_seeking', 0.65, 0.10, 'Moderate — careful research'),
                'response_tendency': PersonaTrait('response_tendency', 0.45, 0.10, 'Conservative-cautious'),
                'extremity': PersonaTrait('extremity', 0.15, 0.08, 'Low — cautious responses'),
                'engagement': PersonaTrait('engagement', 0.70, 0.10, 'Moderate — careful decisions'),
            },
            text_style={
                'verbosity': 'cautious',
                'detail_level': 'risk_aware',
                'coherence': 'high',
                'sentiment_alignment': 'conservative'
            },
            applicable_domains=['financial', 'investment', 'risk', 'behavioral_economics', 'insurance', 'saving']
        )

        # ================================================================
        # COMMUNICATION/MEDIA PERSONA — Persuasion-Resistant
        # Scientific basis: Sagarin et al. (2002) resistance to persuasion,
        # Brehm (1966) psychological reactance theory.
        # ================================================================
        personas['persuasion_resistant'] = Persona(
            name="Persuasion-Resistant Individual",
            category="communication",
            description="Person highly resistant to persuasion attempts. "
                       "Strong reactance to pressure, counterargues automatically, "
                       "maintains initial positions. Brehm (1966): Reactance theory.",
            weight=0.06,
            traits={
                'reactance': PersonaTrait('reactance', 0.82, 0.08, 'High reactance'),
                'persuasion_knowledge': PersonaTrait('persuasion_knowledge', 0.78, 0.10, 'High'),
                'need_for_cognition': PersonaTrait('need_for_cognition', 0.70, 0.10, 'Moderate-high'),
                'conformity': PersonaTrait('conformity', 0.25, 0.10, 'Low — independent'),
                'source_skepticism': PersonaTrait('source_skepticism', 0.80, 0.08, 'Very skeptical'),
                'response_tendency': PersonaTrait('response_tendency', 0.45, 0.10, 'Resistant-negative'),
                'extremity': PersonaTrait('extremity', 0.35, 0.10, 'Moderate — firm positions'),
                'response_consistency': PersonaTrait('response_consistency', 0.82, 0.08, 'Very consistent'),
            },
            text_style={
                'verbosity': 'assertive',
                'detail_level': 'counterarguing',
                'coherence': 'high',
                'sentiment_alignment': 'resistant'
            },
            applicable_domains=['communication', 'persuasion', 'advertising', 'propaganda', 'misinformation', 'media']
        )

        # ================================================================
        # NARRATIVE THINKER (v1.0.4.9)
        # Scientific basis: Green & Brock (2000) Transportation theory,
        # Transportability Scale; Appel & Richter (2010) need for narrative.
        # High narrative engagement predicts persuasion via transported states.
        # ================================================================

        personas['narrative_thinker'] = Persona(
            name="Narrative Thinker",
            category="cognitive",
            description="Person who processes information through narrative frameworks. "
                       "High transportation tendency, vivid imagery, emotional engagement with stories. "
                       "Green & Brock (2000): Transported readers show d=0.5-0.7 attitude shifts.",
            weight=0.08,
            traits={
                # Core 8 traits
                'attention_level': PersonaTrait('attention_level', 0.88, 0.06, 'High — absorbed in content'),
                'response_consistency': PersonaTrait('response_consistency', 0.75, 0.08, 'Consistent — narrative coherence'),
                'scale_use_breadth': PersonaTrait('scale_use_breadth', 0.68, 0.10, 'Moderate-broad — emotional range'),
                'acquiescence': PersonaTrait('acquiescence', 0.52, 0.08, 'Neutral — content-driven agreement'),
                'social_desirability': PersonaTrait('social_desirability', 0.45, 0.10, 'Low-moderate — authentic expression'),
                'reading_speed': PersonaTrait('reading_speed', 0.48, 0.10, 'Slower — immersive processing'),
                'response_tendency': PersonaTrait('response_tendency', 0.62, 0.12, 'Slightly positive — narrative optimism'),
                'extremity': PersonaTrait('extremity', 0.30, 0.10, 'Moderate — nuanced but emotionally moved'),
                # Domain-specific traits
                'engagement': PersonaTrait('engagement', 0.85, 0.07, 'Very high — transported processing'),
                'elaboration': PersonaTrait('elaboration', 0.80, 0.08, 'High — rich narrative detail'),
                'imagery_vividness': PersonaTrait('imagery_vividness', 0.82, 0.08, 'Strong mental imagery'),
                'emotional_engagement': PersonaTrait('emotional_engagement', 0.78, 0.10, 'High empathic response'),
            },
            text_style={
                'verbosity': 'elaborate',
                'detail_level': 'narrative_rich',
                'coherence': 'very_high',
                'sentiment_alignment': 'emotionally_engaged'
            },
            applicable_domains=['narrative', 'communication', 'media', 'persuasion',
                                'advertising', 'fiction', 'storytelling', 'entertainment']
        )

        # ================================================================
        # SOCIAL COMPARER (v1.0.4.9)
        # Scientific basis: Gibbons & Buunk (1999) Iowa-Netherlands
        # Comparison Orientation Measure (INCOM); Festinger (1954) social
        # comparison theory. High comparers show reference-dependent evaluation.
        # ================================================================

        personas['social_comparer'] = Persona(
            name="Social Comparer",
            category="social",
            description="Person who habitually evaluates self and options relative to others. "
                       "Frequent upward/downward comparisons, reference-dependent judgments, "
                       "sensitive to social rank. Gibbons & Buunk (1999): INCOM M=3.8/5 for high comparers.",
            weight=0.10,
            traits={
                # Core 8 traits
                'attention_level': PersonaTrait('attention_level', 0.78, 0.08, 'Good — vigilant to social cues'),
                'response_consistency': PersonaTrait('response_consistency', 0.55, 0.12, 'Lower — comparison-driven variability'),
                'scale_use_breadth': PersonaTrait('scale_use_breadth', 0.65, 0.10, 'Moderate — context-sensitive'),
                'acquiescence': PersonaTrait('acquiescence', 0.55, 0.10, 'Slightly elevated — conformity tendency'),
                'social_desirability': PersonaTrait('social_desirability', 0.72, 0.08, 'High — image-conscious'),
                'reading_speed': PersonaTrait('reading_speed', 0.62, 0.10, 'Moderate — scanning for social info'),
                'response_tendency': PersonaTrait('response_tendency', 0.55, 0.14, 'Variable — depends on comparison direction'),
                'extremity': PersonaTrait('extremity', 0.45, 0.12, 'Moderate-high — polarized by comparisons'),
                # Domain-specific traits
                'comparison_orientation': PersonaTrait('comparison_orientation', 0.82, 0.08, 'Very high — habitual comparer'),
                'reference_dependence': PersonaTrait('reference_dependence', 0.78, 0.10, 'High — anchors to others'),
                'status_sensitivity': PersonaTrait('status_sensitivity', 0.75, 0.10, 'High — rank-aware'),
                'envy_proneness': PersonaTrait('envy_proneness', 0.62, 0.12, 'Moderate-high — upward comparison effect'),
            },
            text_style={
                'verbosity': 'moderate',
                'detail_level': 'comparative',
                'coherence': 'moderate',
                'sentiment_alignment': 'reference_dependent'
            },
            applicable_domains=['social_psychology', 'consumer_behavior', 'social_media',
                                'marketing', 'wellbeing', 'self_esteem', 'body_image']
        )

        # ================================================================
        # GRATEFUL OPTIMIST (v1.0.4.9)
        # Scientific basis: McCullough et al. (2002) GQ-6 gratitude scale,
        # Emmons & McCullough (2003) gratitude and wellbeing interventions,
        # Scheier & Carver (1985) LOT dispositional optimism.
        # ================================================================

        personas['grateful_optimist'] = Persona(
            name="Grateful Optimist",
            category="positive_psychology",
            description="Person with dispositional gratitude and optimistic outlook. "
                       "Positive interpretation bias, high life satisfaction, appreciation-focused. "
                       "Emmons & McCullough (2003): Gratitude journaling d=0.4 on wellbeing.",
            weight=0.08,
            traits={
                # Core 8 traits
                'attention_level': PersonaTrait('attention_level', 0.82, 0.07, 'Good — attentive to positives'),
                'response_consistency': PersonaTrait('response_consistency', 0.72, 0.08, 'Consistent — stable positive bias'),
                'scale_use_breadth': PersonaTrait('scale_use_breadth', 0.58, 0.10, 'Moderate — skewed positive'),
                'acquiescence': PersonaTrait('acquiescence', 0.62, 0.08, 'Elevated — agreeable disposition'),
                'social_desirability': PersonaTrait('social_desirability', 0.68, 0.08, 'Higher — positive self-presentation'),
                'reading_speed': PersonaTrait('reading_speed', 0.58, 0.10, 'Moderate — thorough engagement'),
                'response_tendency': PersonaTrait('response_tendency', 0.72, 0.08, 'Positive — gratitude/optimism bias'),
                'extremity': PersonaTrait('extremity', 0.28, 0.08, 'Moderate — positive but not extreme'),
                # Domain-specific traits
                'gratitude': PersonaTrait('gratitude', 0.85, 0.07, 'Very high — GQ-6 top quartile'),
                'optimism': PersonaTrait('optimism', 0.80, 0.08, 'High — LOT-R top quartile'),
                'life_satisfaction': PersonaTrait('life_satisfaction', 0.78, 0.10, 'High — SWLS above average'),
                'positive_affect': PersonaTrait('positive_affect', 0.75, 0.10, 'High — PANAS positive'),
            },
            text_style={
                'verbosity': 'warm',
                'detail_level': 'appreciative',
                'coherence': 'high',
                'sentiment_alignment': 'positive_genuine'
            },
            applicable_domains=['positive_psychology', 'health_psychology', 'wellbeing',
                                'gratitude', 'life_satisfaction', 'resilience', 'mental_health']
        )

        # ================================================================
        # MORAL ABSOLUTIST (v1.0.4.9)
        # Scientific basis: Tetlock et al. (2000) sacred values and
        # taboo trade-offs; Haidt (2001) moral intuitionism; Graham et al.
        # (2013) Moral Foundations Theory. Strong moral convictions produce
        # high extremity and resistance to trade-offs.
        # ================================================================

        personas['moral_absolutist'] = Persona(
            name="Moral Absolutist",
            category="moral",
            description="Person with strong deontological moral convictions. "
                       "Refuses trade-offs on sacred values, high moral outrage, principled responding. "
                       "Tetlock et al. (2000): Sacred value holders show extreme rejection of taboo trade-offs.",
            weight=0.07,
            traits={
                # Core 8 traits
                'attention_level': PersonaTrait('attention_level', 0.85, 0.06, 'High — morally vigilant'),
                'response_consistency': PersonaTrait('response_consistency', 0.80, 0.07, 'Very consistent — principled'),
                'scale_use_breadth': PersonaTrait('scale_use_breadth', 0.75, 0.10, 'Broad — strong opinions both ways'),
                'acquiescence': PersonaTrait('acquiescence', 0.48, 0.08, 'Moderate — agrees with moral items, rejects immoral'),
                'social_desirability': PersonaTrait('social_desirability', 0.70, 0.08, 'High — moral self-image'),
                'reading_speed': PersonaTrait('reading_speed', 0.52, 0.10, 'Slower — deliberate moral processing'),
                'response_tendency': PersonaTrait('response_tendency', 0.60, 0.15, 'Variable — depends on moral content'),
                'extremity': PersonaTrait('extremity', 0.78, 0.08, 'Very high — moral convictions are strong'),
                # Domain-specific traits
                'engagement': PersonaTrait('engagement', 0.85, 0.07, 'Very high — morally invested'),
                'moral_conviction': PersonaTrait('moral_conviction', 0.88, 0.06, 'Very high — sacred value holder'),
                'moral_outrage': PersonaTrait('moral_outrage', 0.75, 0.10, 'High — strong reactive emotion'),
                'trade_off_resistance': PersonaTrait('trade_off_resistance', 0.85, 0.07, 'Very high — refuses taboo trade-offs'),
            },
            text_style={
                'verbosity': 'passionate',
                'detail_level': 'principled',
                'coherence': 'very_high',
                'sentiment_alignment': 'morally_charged'
            },
            applicable_domains=['moral_psychology', 'ethical_judgment', 'political_psychology',
                                'justice', 'fairness', 'bioethics', 'environmental_ethics']
        )

        # ================================================================
        # DIGITAL NATIVE (v1.0.4.9)
        # Scientific basis: Prensky (2001) digital native concept,
        # Ophir et al. (2009) cognitive control and media multitasking,
        # Uncapher & Wagner (2018) media multitasking and attention.
        # Habitual multitaskers show faster but less consistent processing.
        # ================================================================

        personas['digital_native'] = Persona(
            name="Digital Native",
            category="technology",
            description="Person raised with digital technology, habitual multitasker. "
                       "Fast processing, moderate attention fragmentation, high tech comfort. "
                       "Ophir et al. (2009): Heavy media multitaskers show d=0.4 attention cost.",
            weight=0.10,
            traits={
                # Core 8 traits
                'attention_level': PersonaTrait('attention_level', 0.72, 0.10, 'Moderate — fragmented by multitasking'),
                'response_consistency': PersonaTrait('response_consistency', 0.58, 0.12, 'Lower — context-switching cost'),
                'scale_use_breadth': PersonaTrait('scale_use_breadth', 0.62, 0.10, 'Moderate — quick judgments'),
                'acquiescence': PersonaTrait('acquiescence', 0.50, 0.10, 'Neutral'),
                'social_desirability': PersonaTrait('social_desirability', 0.42, 0.12, 'Low — authentic digital culture'),
                'reading_speed': PersonaTrait('reading_speed', 0.85, 0.07, 'Very fast — skimming habit'),
                'response_tendency': PersonaTrait('response_tendency', 0.58, 0.12, 'Slightly positive — tech-optimistic'),
                'extremity': PersonaTrait('extremity', 0.35, 0.10, 'Moderate — quick but not polarized'),
                # Domain-specific traits
                'engagement': PersonaTrait('engagement', 0.65, 0.12, 'Moderate — brief bursts of focus'),
                'tech_comfort': PersonaTrait('tech_comfort', 0.90, 0.05, 'Very high — native facility'),
                'multitasking_tendency': PersonaTrait('multitasking_tendency', 0.82, 0.08, 'High — habitual switcher'),
                'novelty_seeking': PersonaTrait('novelty_seeking', 0.72, 0.10, 'Higher — drawn to new stimuli'),
            },
            text_style={
                'verbosity': 'concise',
                'detail_level': 'surface',
                'coherence': 'moderate',
                'sentiment_alignment': 'casual'
            },
            applicable_domains=['technology', 'social_media', 'digital_wellbeing', 'ai',
                                'online_behavior', 'gaming', 'communication', 'education']
        )

        # ================================================================
        # FINANCIAL DELIBERATOR (v1.0.4.9)
        # Scientific basis: Kahneman & Tversky (1979) prospect theory,
        # Barber & Odean (2001) individual investor behavior,
        # Frederick (2005) cognitive reflection test.
        # Careful, analytical decision-maker in financial contexts.
        # ================================================================

        personas['financial_deliberator'] = Persona(
            name="Financial Deliberator",
            category="economics",
            description="Careful, analytical person in financial decision contexts. "
                       "High cognitive effort, risk-aware, values information before deciding. "
                       "Frederick (2005): High CRT scorers are more patient and risk-neutral.",
            weight=0.08,
            traits={
                # Core 8 traits
                'attention_level': PersonaTrait('attention_level', 0.88, 0.06, 'High — careful evaluation'),
                'response_consistency': PersonaTrait('response_consistency', 0.78, 0.08, 'High — systematic processing'),
                'scale_use_breadth': PersonaTrait('scale_use_breadth', 0.65, 0.10, 'Moderate — calibrated judgments'),
                'acquiescence': PersonaTrait('acquiescence', 0.42, 0.08, 'Low — skeptical, evaluative'),
                'social_desirability': PersonaTrait('social_desirability', 0.38, 0.10, 'Low — values accuracy over impression'),
                'reading_speed': PersonaTrait('reading_speed', 0.45, 0.10, 'Slow — deliberate, careful reading'),
                'response_tendency': PersonaTrait('response_tendency', 0.52, 0.10, 'Centered — avoids optimism bias'),
                'extremity': PersonaTrait('extremity', 0.25, 0.08, 'Low — calibrated, moderate positions'),
                # Domain-specific traits
                'engagement': PersonaTrait('engagement', 0.82, 0.07, 'High — invested in financial decisions'),
                'need_for_cognition': PersonaTrait('need_for_cognition', 0.80, 0.08, 'High — enjoys complex analysis'),
                'risk_tolerance': PersonaTrait('risk_tolerance', 0.48, 0.12, 'Moderate — risk-aware not risk-averse'),
                'information_seeking': PersonaTrait('information_seeking', 0.82, 0.08, 'High — gathers before deciding'),
            },
            text_style={
                'verbosity': 'analytical',
                'detail_level': 'quantitative',
                'coherence': 'very_high',
                'sentiment_alignment': 'measured'
            },
            applicable_domains=['behavioral_economics', 'financial_psychology', 'consumer_behavior',
                                'investment', 'risk', 'judgment', 'decision_making']
        )

        return personas

    def _build_domain_keywords(self) -> Dict[str, List[str]]:
        """Build keyword mappings for automatic domain detection.

        v1.0.0: Expanded with 30+ domains based on training on 210 QSF files.
        v2.0.0: Restructured with tiered keywords (compound > two-word > specific single).
                 Generic single words removed to prevent false positives.
                 Used by detect_domains() Phase 2 for individual keyword scoring.
        """
        return {
            'consumer_behavior': [
                # Compound phrases (3+ words) - highest specificity
                'willingness to pay', 'purchase intention study', 'consumer decision making',
                'brand loyalty program', 'shopping cart abandonment', 'consumer trust in',
                'product review credibility', 'online shopping behavior',
                # Two-word phrases - high specificity
                'consumer behavior', 'consumer trust', 'consumer preference', 'consumer choice',
                'purchase intention', 'buying behavior', 'brand loyalty', 'brand perception',
                'product review', 'product evaluation', 'product perception', 'product quality',
                'shopping behavior', 'customer satisfaction', 'customer experience',
                'retail experience', 'consumer confidence', 'consumer attitude',
                # Domain-specific single words (unambiguous)
                'consumer', 'purchase', 'buying', 'shopping', 'retail', 'e-commerce',
                'customer', 'consumption', 'wtp', 'checkout', 'sweater', 'loyalty program',
                't-shirt', 'coffee shop'
            ],
            'marketing': [
                # Compound phrases
                'social media marketing', 'advertising effectiveness study',
                'brand positioning strategy', 'influencer marketing effect',
                'celebrity endorsement effectiveness study', 'persuasion knowledge model test',
                'corporate social responsibility marketing',
                # Two-word phrases
                'marketing strategy', 'advertising campaign', 'brand endorsement',
                'marketing message', 'advertising appeal', 'marketing persuasion',
                'influencer endorsement', 'marketing communication', 'woke washing',
                'scientific pitch', 'ad campaign', 'persuasion knowledge',
                'celebrity endorsement', 'brand authenticity',
                # Domain-specific single words
                'marketing', 'advertising', 'advertisement', 'branding', 'endorsement',
                'influencer', 'woke-washed', 'campaign'
            ],
            'ai': [
                # Compound phrases
                'artificial intelligence generated', 'ai generated content',
                'algorithm aversion effect', 'algorithm appreciation effect',
                'human versus ai', 'ai versus human', 'ai recommendation system',
                'ai content detection', 'ai content labels',
                # Two-word phrases
                'artificial intelligence', 'machine learning', 'algorithm aversion',
                'algorithm appreciation', 'ai generated', 'ai recommendation',
                'ai disclosure', 'ai trust', 'ai tutor', 'ai doctor', 'ai manager',
                'ai service', 'ai friend', 'ai told', 'chatbot interaction',
                'recommendation system', 'ai labeling', 'ai detection',
                # Domain-specific single words
                'ai-generated', 'chatbot', 'automation', 'algorithm', 'robot'
            ],
            'technology': [
                # Compound phrases
                'technology adoption model study', 'user experience design evaluation',
                'human computer interaction experiment', 'technology acceptance model',
                'digital platform usage behavior',
                # Two-word phrases
                'user experience', 'digital platform', 'technology adoption',
                'software interface', 'digital technology', 'tech adoption',
                'human computer', 'technology acceptance', 'usability testing',
                # Domain-specific single words
                'technology', 'software', 'platform', 'interface', 'usability'
            ],
            'behavioral_economics': [
                # Compound phrases
                'prospect theory prediction', 'loss aversion effect', 'default option nudge',
                'framing effect experiment', 'anchoring and adjustment', 'sunk cost fallacy',
                'intertemporal choice task', 'nudging through defaults',
                'retirement savings default', 'default enrollment option',
                # Two-word phrases
                'prospect theory', 'loss aversion', 'default nudge', 'framing effect',
                'anchoring effect', 'scarcity priming', 'abundance priming',
                'debt priming', 'powerlessness priming', 'time preference',
                'intertemporal choice', 'sunk cost', 'choice architecture',
                'nudge intervention', 'behavioral nudge', 'default option',
                'retirement savings', 'savings behavior',
                # Domain-specific single words
                'nudge', 'nudging', 'heuristic', 'framing', 'intertemporal',
                'self-control'
            ],
            'organizational_behavior': [
                # Compound phrases
                'workplace gossip behavior', 'employee job satisfaction',
                'organizational commitment study', 'leadership style effect',
                'team performance dynamics', 'human resource management',
                'wage gap in stem', 'gap in stem organizations',
                'gender gap in organizations', 'diversity in organizations',
                # Two-word phrases
                'workplace behavior', 'employee engagement', 'job satisfaction',
                'organizational commitment', 'leadership style', 'team performance',
                'gossip at work', 'workplace dynamics', 'employee turnover',
                'employee motivation', 'strategic silence', 'workplace rudeness',
                'workplace politeness', 'human resources', 'manager subordinate',
                'stem organizations', 'stem workforce', 'stem fields',
                'organizational culture', 'organizational dynamics',
                # Domain-specific single words
                'employee', 'workplace', 'subordinate', 'teammate', 'turnover',
                'hr', 'organization', 'organizations'
            ],
            'social_psychology': [
                # Compound phrases
                'minimal group paradigm experiment', 'social value orientation task',
                'ingroup outgroup dynamics', 'bystander effect study',
                'social identity theory', 'social categorization effect',
                'implicit association test measure', 'stereotype threat experiment',
                'cognitive dissonance reduction task', 'social desirability bias measure',
                'self serving bias in attribution',
                # Two-word phrases
                'social influence', 'social proof', 'conformity pressure',
                'social identity', 'group dynamics', 'social categorization',
                'ingroup outgroup', 'bystander effect', 'social comparison',
                'deviance credit', 'social value', 'social dilemma',
                'prosocial behavior', 'implicit association', 'stereotype threat',
                'cognitive dissonance', 'social desirability', 'attribution bias',
                # Domain-specific single words
                'conformity', 'ingroup', 'outgroup', 'prosocial', 'oneness',
                'stereotype', 'prejudice', 'attribution'
            ],
            'economic_games': [
                # Compound phrases
                'public goods game contribution', 'dictator game allocation',
                'trust game investment', 'ultimatum game offer',
                'prisoner dilemma cooperation',
                # Two-word phrases
                'dictator game', 'trust game', 'public goods', 'ultimatum game',
                'prisoner dilemma', 'player a', 'player b',
                # Domain-specific single words
                'trustor', 'trustee', 'endowment', 'payoff', 'payout', 'pgg', 'dg'
            ],
            'norm_elicitation': [
                # Compound phrases
                'personal normative beliefs study', 'social norm intervention effect',
                'norm elicitation task', 'empirical expectations measurement',
                # Two-word phrases
                'norm elicitation', 'empirical norm', 'normative expectations',
                'injunctive norm', 'descriptive norm', 'norm violation',
                'norm message', 'social norm',
                # Domain-specific single words
                'norms'
            ],
            'political_psychology': [
                # Compound phrases
                'political polarization effect', 'partisan identity threat',
                'liberal conservative divide', 'democrat republican difference',
                'affective polarization study', 'political attitude formation',
                'echo chamber effect', 'political media consumption',
                'social media echo chamber', 'political news consumption',
                # Two-word phrases
                'political polarization', 'political identity', 'partisan divide',
                'political attitudes', 'political beliefs', 'political ideology',
                'affective polarization', 'political trust', 'political participation',
                'political engagement', 'voting behavior', 'partisan bias',
                'political party', 'echo chamber', 'ideological divide',
                'media polarization', 'political communication',
                # Domain-specific single words
                'political', 'partisan', 'polarization', 'democrat', 'republican',
                'liberal', 'conservative', 'ideology', 'ideological', 'trump', 'biden'
            ],
            'dishonesty': [
                # Compound phrases
                'die roll reporting task', 'cheating behavior study',
                'overclaiming test paradigm', 'honest reporting behavior',
                # Two-word phrases
                'die roll', 'cheating behavior', 'overclaiming test',
                'truth telling', 'honest reporting', 'ethical behavior',
                'unethical behavior',
                # Domain-specific single words
                'dishonesty', 'honesty', 'lying', 'liar', 'cheating', 'fraud',
                'deception', 'overclaiming', 'misreporting'
            ],
            'punishment': [
                # Compound phrases
                'third party punishment paradigm', 'costly punishment study',
                'norm enforcement mechanism',
                # Two-word phrases
                'third-party punishment', 'costly punishment', 'norm enforcement',
                'punitive justice', 'criminal punishment',
                # Domain-specific single words
                'punishment', 'punisher', 'sanction', 'deterrence', 'retribution'
            ],
            'gender': [
                # Compound phrases
                'gender wage gap study', 'sex differences in behavior',
                'gender stereotype effect', 'gender bias experiment',
                'gender role expectation', 'gender pay gap analysis',
                # Two-word phrases
                'gender differences', 'sex differences', 'gender stereotypes',
                'gender bias', 'gender roles', 'gender identity', 'gender gap',
                'gender pay', 'wage gap', 'gender discrimination', 'gender equality',
                # Domain-specific single words
                'gender', 'masculinity', 'femininity', 'sexism'
            ],
            'power_status': [
                # Compound phrases
                'high power low power manipulation', 'social dominance orientation scale',
                'power priming manipulation task', 'socioeconomic status effect on',
                'status hierarchy perception study',
                # Two-word phrases
                'high power', 'low power', 'power dynamics', 'status hierarchy',
                'socioeconomic status', 'social hierarchy', 'power imbalance',
                'social dominance', 'power priming', 'status threat',
                # Domain-specific single words
                'hierarchy', 'dominance', 'ses', 'powerlessness'
            ],
            'charitable_giving': [
                # Compound phrases
                'charitable giving behavior', 'effective altruism donation',
                'warm glow effect', 'donation behavior study',
                # Two-word phrases
                'charitable giving', 'donation behavior', 'warm glow',
                'effective altruism', 'charitable donation',
                # Domain-specific single words
                'charity', 'donation', 'charitable', 'philanthropy', 'nonprofit', 'ngo'
            ],
            'covid': [
                # Compound phrases
                'covid-19 vaccine hesitancy', 'pandemic behavior change',
                'social distancing compliance', 'mask wearing behavior',
                # Two-word phrases
                'vaccine hesitancy', 'social distancing', 'mask wearing',
                'pandemic behavior', 'health behavior',
                # Domain-specific single words
                'covid', 'covid-19', 'coronavirus', 'pandemic', 'vaccine',
                'vaccination', 'lockdown'
            ],
            'emotions': [
                # Compound phrases
                'discrete emotions theory', 'emotional regulation strategy',
                'mood induction procedure', 'affect intensity measure',
                # Two-word phrases
                'emotional regulation', 'mood induction', 'affect intensity',
                'emotional contagion', 'discrete emotions', 'emotional response',
                # Domain-specific single words
                'emotion', 'mood', 'sadness', 'happiness', 'anger', 'anxiety',
                'arousal'
            ],
            'deontology_utilitarianism': [
                # Compound phrases
                'trolley problem dilemma', 'moral foundations theory',
                'deontological moral judgment', 'consequentialist reasoning',
                # Two-word phrases
                'moral foundations', 'trolley problem', 'moral judgment',
                'moral reasoning', 'ethical dilemma',
                # Domain-specific single words
                'deontology', 'utilitarianism', 'deontological', 'consequentialist',
                'trolley'
            ],
            'accuracy_misinformation': [
                # Compound phrases
                'fake news detection accuracy', 'truth discernment ability',
                'conspiracy belief formation', 'misinformation susceptibility study',
                # Two-word phrases
                'fake news', 'fact checking', 'truth discernment', 'conspiracy belief',
                'belief accuracy', 'misinformation sharing', 'news accuracy',
                # Domain-specific single words
                'misinformation', 'conspiracy', 'disinformation', 'fake'
            ],
            'psychological_ownership': [
                # Compound phrases
                'psychological ownership of product', 'endowment effect experiment',
                'mere ownership effect study', 'virtual ownership feeling',
                'touch effect on ownership',
                # Two-word phrases
                'psychological ownership', 'endowment effect', 'ownership feelings',
                'self extension', 'mere ownership', 'virtual ownership',
                'perceived ownership',
                # Domain-specific single words
                'possess', 'territory', 'ownership', 'endowment'
            ],
            'health_psychology': [
                # Compound phrases
                'health behavior change intervention', 'patient adherence study',
                'medical decision making', 'health risk perception',
                'vaccine hesitancy among', 'health intervention effectiveness',
                # Two-word phrases
                'health behavior', 'behavior change', 'medical decision',
                'health intervention', 'patient adherence', 'health risk',
                'health communication', 'health literacy', 'health promotion',
                # Domain-specific single words
                'wellness', 'nutrition', 'exercise', 'disease', 'prevention',
                'treatment', 'addiction', 'habit'
            ],
            'environmental': [
                # Compound phrases
                'sustainable behavior intervention', 'green consumption choice',
                'carbon footprint reduction', 'environmental attitude study',
                'climate change belief', 'sustainable default option',
                # Two-word phrases
                'sustainable behavior', 'green consumption', 'carbon footprint',
                'environmental attitude', 'climate change', 'sustainable default',
                'environmental concern', 'renewable energy',
                # Domain-specific single words
                'sustainability', 'eco', 'recycling', 'pollution',
                'conservation', 'renewable', 'unsustainable', 'environmental'
            ],
            'pricing': [
                # Compound phrases
                'willingness to pay measure', 'price sensitivity measurement study',
                'pay what you want pricing', 'reference price effect experiment',
                'price fairness perception study',
                # Two-word phrases
                'willingness to pay', 'price sensitivity', 'price perception',
                'premium pricing', 'dynamic pricing', 'price fairness',
                'reference price', 'price framing',
                # Domain-specific single words
                'pricing', 'auction', 'bid', 'wtp'
            ],
            'hedonic_consumption': [
                # Compound phrases
                'hedonic versus utilitarian product choice', 'hedonic consumption experience study',
                'luxury brand perception experiment', 'pleasure seeking consumption behavior',
                # Two-word phrases
                'hedonic consumption', 'hedonic value', 'hedonic experience',
                'luxury consumption', 'hedonic motivation', 'pleasure seeking',
                # Domain-specific single words
                'hedonic', 'luxury', 'aesthetic', 'entertainment', 'indulgence'
            ],
            'utilitarian_consumption': [
                # Compound phrases
                'utilitarian versus hedonic product choice', 'functional product evaluation study',
                'utilitarian purchase motivation experiment', 'practical value assessment task',
                # Two-word phrases
                'utilitarian consumption', 'utilitarian value', 'functional value',
                'utilitarian product', 'practical value', 'functional benefit',
                # Domain-specific single words
                'utilitarian', 'functional', 'durability', 'necessity', 'practical'
            ],
            'privacy': [
                # Compound phrases
                'personal information disclosure', 'privacy paradox study',
                'data protection concern', 'surveillance capitalism effect',
                # Two-word phrases
                'data privacy', 'personal information', 'privacy concern',
                'privacy paradox', 'information disclosure', 'surveillance concern',
                # Domain-specific single words
                'privacy', 'surveillance', 'gdpr', 'anonymity'
            ],
            'fairness': [
                # Compound phrases
                'distributive justice perception', 'procedural fairness study',
                'inequality aversion experiment',
                # Two-word phrases
                'distributive justice', 'procedural fairness', 'inequality aversion',
                'fairness perception', 'perceived fairness',
                # Domain-specific single words
                'fairness', 'equity', 'inequality', 'distributive'
            ],
            'negotiation': [
                # Compound phrases
                'negotiation strategy effectiveness study', 'first offer anchoring effect',
                'distributive bargaining task experiment', 'integrative negotiation outcome',
                # Two-word phrases
                'bargaining behavior', 'negotiation strategy', 'offer counteroffer',
                'first offer', 'negotiation outcome', 'integrative negotiation',
                'distributive bargaining',
                # Domain-specific single words
                'negotiation', 'bargaining', 'counteroffer', 'concession'
            ],
            'personalization': [
                # Compound phrases
                'personalized recommendation system effect', 'tailored content delivery study',
                'customized user experience experiment', 'personalization privacy tradeoff',
                # Two-word phrases
                'personalized recommendation', 'tailored content',
                'customized experience', 'personalized marketing',
                'recommendation engine', 'content personalization',
                # Domain-specific single words
                'personalized', 'personalization', 'tailored', 'customized'
            ],
            'education': [
                # Compound phrases
                'educational intervention effectiveness study', 'student learning outcome measure',
                'growth mindset intervention experiment', 'classroom behavior management study',
                'academic performance motivation effect',
                # Two-word phrases
                'educational intervention', 'student learning', 'academic performance',
                'learning outcome', 'growth mindset', 'classroom behavior',
                'student motivation', 'academic achievement', 'test anxiety',
                # Domain-specific single words
                'education', 'student', 'teacher', 'tutor', 'academic',
                'school', 'course', 'classroom'
            ],
            'media_communication': [
                # Compound phrases
                'social media echo chamber', 'media framing effect',
                'news consumption pattern', 'media influence on attitudes',
                # Two-word phrases
                'social media', 'media consumption', 'echo chamber',
                'media framing', 'news consumption', 'media influence',
                'media effects', 'media literacy', 'information environment',
                # Domain-specific single words
                'media'
            ],
            'risk_perception': [
                # Compound phrases
                'risk perception and behavior', 'perceived risk assessment',
                'risk communication effect',
                # Two-word phrases
                'risk perception', 'perceived risk', 'risk communication',
                'risk assessment', 'risk aversion', 'risk taking',
                # Domain-specific single words
                'risk-taking', 'hazard'
            ],
            'decision_making': [
                # Compound phrases
                'decision making under uncertainty', 'choice overload effect',
                'decision fatigue study', 'nudging through default options',
                'default option effect on',
                # Two-word phrases
                'decision making', 'choice overload', 'decision fatigue',
                'choice architecture', 'decision quality', 'default option',
                'default options', 'option selection', 'choice behavior',
                # Domain-specific single words
                'decision-making'
            ],
            # ---- New domains added for broader behavioral research coverage ----
            'moral_psychology': [
                # Compound phrases
                'moral foundations theory test', 'moral judgment paradigm',
                'moral identity centrality measure', 'moral disengagement scale',
                'moral licensing effect study', 'moral cleansing behavior',
                # Two-word phrases
                'moral judgment', 'moral identity', 'moral reasoning',
                'moral intuition', 'moral foundations', 'moral disengagement',
                'moral licensing', 'moral cleansing', 'moral conviction',
                'moral outrage', 'ethical judgment', 'moral dilemma',
                'moral emotion', 'moral cognition',
                # Domain-specific single words
                'morality', 'moral', 'ethical', 'virtue', 'wrongness'
            ],
            'prosocial_behavior': [
                # Compound phrases
                'prosocial behavior motivation study', 'altruistic punishment paradigm',
                'bystander intervention effect', 'helping behavior experiment',
                'volunteer motivation study', 'empathy altruism hypothesis',
                # Two-word phrases
                'prosocial behavior', 'altruistic behavior', 'helping behavior',
                'cooperative behavior', 'volunteer behavior', 'empathic concern',
                'altruistic motivation', 'prosocial motivation', 'social cooperation',
                'empathy altruism',
                # Domain-specific single words
                'altruism', 'altruistic', 'prosocial', 'volunteering',
                'helping', 'cooperation', 'empathy'
            ],
            'cultural_psychology': [
                # Compound phrases
                'cross cultural comparison study', 'individualism collectivism dimension',
                'cultural tightness looseness measure', 'hofstede cultural dimension',
                'independent self construal measure', 'cultural value orientation',
                # Two-word phrases
                'cross cultural', 'cultural differences', 'cultural values',
                'individualism collectivism', 'cultural norms', 'cultural tightness',
                'cultural looseness', 'self construal', 'cultural identity',
                'cultural psychology', 'acculturation stress',
                # Domain-specific single words
                'cross-cultural', 'multicultural', 'collectivism', 'individualism',
                'acculturation', 'bicultural', 'ethnocentrism'
            ],
            'computational_social_science': [
                # Compound phrases
                'online behavior tracking study', 'digital trace data analysis',
                'social media behavior analysis', 'computational text analysis',
                'online experiment platform', 'digital footprint analysis',
                # Two-word phrases
                'digital behavior', 'online behavior', 'computational analysis',
                'digital trace', 'social network', 'online experiment',
                'digital footprint', 'web scraping', 'text mining',
                'sentiment analysis', 'network analysis',
                # Domain-specific single words
                'computational', 'digital', 'algorithmic', 'bot', 'tweet'
            ],
            'sports_psychology': [
                # Compound phrases
                'athletic performance under pressure', 'mental toughness training effect',
                'choking under pressure study', 'team cohesion performance link',
                'sport motivation climate study',
                # Two-word phrases
                'sports performance', 'athletic performance', 'mental toughness',
                'choking under', 'team cohesion', 'sport motivation',
                'performance anxiety', 'sport psychology', 'motor performance',
                'competition anxiety', 'exercise motivation',
                # Domain-specific single words
                'athlete', 'athletic', 'coaching', 'sportsman', 'referee'
            ],
            'behavioral_neuroscience': [
                # Compound phrases
                'neural correlates of decision', 'brain imaging study of',
                'cognitive load manipulation task', 'working memory capacity task',
                'dual process theory test', 'executive function assessment',
                # Two-word phrases
                'cognitive load', 'working memory', 'executive function',
                'neural correlates', 'brain imaging', 'cognitive control',
                'dual process', 'implicit cognition', 'cognitive resources',
                'response time', 'reaction time',
                # Domain-specific single words
                'neuroimaging', 'fmri', 'eeg', 'neurocognitive', 'cortisol',
                'psychophysiology', 'biomarker'
            ]
        }

    def detect_domains(self, study_description: str, study_title: str = "") -> List[str]:
        """
        Automatically detect relevant research domains from study description.

        v2.0.0: Major rewrite with 5-phase scoring algorithm:
          Phase 1 - Compound phrase scoring (3+ words = 20pts, 2 words = 10pts)
          Phase 2 - Individual keyword scoring with word-boundary matching
          Phase 3 - Negative keyword penalties (expanded to all domains)
          Phase 4 - Cross-domain suppression (strong signals penalize unrelated domains)
          Phase 5 - Context-aware boosts for experimental manipulation patterns

        Returns 1-3 genuinely relevant domains, never 0.
        """
        import re

        title_text = study_title.lower().strip()
        desc_text = study_description.lower().strip()
        combined_text = f"{title_text} {desc_text}"

        if not combined_text.strip():
            return ['consumer_behavior']

        # ----------------------------------------------------------------
        # Negative keywords: presence of these penalizes a domain heavily.
        # Organized per-domain to reduce false positives across all domains.
        # ----------------------------------------------------------------
        domain_negative_keywords: Dict[str, List[str]] = {
            'ai': ['polarization', 'partisan', 'democrat', 'republican', 'liberal',
                   'conservative', 'ideology', 'voting', 'political party', 'election',
                   'gender wage', 'wage gap', 'stem organizations', 'charity', 'donation',
                   'vaccine hesitancy', 'covid', 'pandemic', 'trolley', 'moral dilemma',
                   'dictator game', 'public goods game'],
            'gender': ['polarization', 'partisan', 'democrat', 'republican', 'liberal',
                       'conservative', 'ideology', 'voting', 'political party', 'election',
                       'ai generated', 'artificial intelligence', 'algorithm', 'chatbot',
                       'vaccine', 'covid', 'pandemic', 'echo chamber', 'fake news',
                       'nudge', 'default option', 'retirement savings'],
            'organizational_behavior': ['polarization', 'partisan', 'democrat', 'republican',
                                         'election', 'ideology', 'liberal', 'conservative',
                                         'ai generated', 'artificial intelligence', 'vaccine',
                                         'covid', 'pandemic', 'echo chamber',
                                         'shopping', 'purchase', 'brand', 'charity', 'donation',
                                         'trolley', 'moral dilemma'],
            'norm_elicitation': ['polarization', 'partisan', 'democrat', 'republican',
                                 'election', 'ideology', 'voting', 'ai generated',
                                 'artificial intelligence', 'consumer', 'purchase', 'brand',
                                 'vaccine', 'covid', 'gender wage', 'wage gap',
                                 'echo chamber', 'social media'],
            'psychological_ownership': ['polarization', 'partisan', 'democrat', 'republican',
                                         'election', 'ideology', 'voting', 'political',
                                         'ai generated', 'artificial intelligence', 'vaccine',
                                         'covid', 'gender', 'wage gap', 'echo chamber'],
            'consumer_behavior': ['polarization', 'partisan', 'democrat', 'republican',
                                   'election', 'ideology', 'voting', 'vaccine', 'covid',
                                   'pandemic', 'dictator game', 'public goods game',
                                   'trolley', 'moral dilemma', 'gender wage', 'wage gap',
                                   'echo chamber'],
            'political_psychology': ['consumer', 'purchase', 'shopping', 'brand', 'product',
                                      'charity', 'donation', 'dictator game', 'public goods',
                                      'trolley', 'moral dilemma', 'gender wage', 'wage gap',
                                      'stem organizations', 'nudge', 'retirement savings',
                                      'default option'],
            'behavioral_economics': ['polarization', 'partisan', 'democrat', 'republican',
                                      'election', 'ideology', 'vaccine', 'covid', 'pandemic',
                                      'ai generated', 'artificial intelligence', 'gender wage',
                                      'wage gap', 'echo chamber', 'social media echo',
                                      'dictator game', 'public goods game'],
            'social_psychology': ['ai generated', 'artificial intelligence', 'algorithm',
                                   'vaccine', 'covid', 'pandemic', 'purchase', 'shopping',
                                   'consumer', 'brand', 'product review', 'retirement savings',
                                   'nudge', 'default option', 'trolley', 'moral dilemma'],
            'health_psychology': ['polarization', 'partisan', 'democrat', 'republican',
                                   'election', 'ideology', 'consumer', 'purchase', 'shopping',
                                   'brand', 'dictator game', 'public goods game', 'trolley',
                                   'moral dilemma', 'echo chamber', 'nudge', 'default option',
                                   'retirement savings', 'ai generated',
                                   # Context: "health product marketing" is marketing, not health psych
                                   'health product marketing', 'health food brand',
                                   'health claim advertising', 'health supplement market'],
            'emotions': ['polarization', 'partisan', 'vaccine', 'covid', 'ai generated',
                         'consumer', 'purchase', 'dictator game', 'trolley', 'nudge',
                         'default option', 'echo chamber', 'gender wage', 'wage gap',
                         # Context: "emotional appeal in advertising" is marketing
                         'emotional appeal advertising', 'emotional branding',
                         'emotional marketing strategy'],
            'economic_games': ['polarization', 'partisan', 'consumer', 'purchase', 'brand',
                                'ai generated', 'artificial intelligence', 'vaccine', 'covid',
                                'echo chamber', 'gender wage', 'wage gap'],
            'environmental': ['polarization', 'partisan', 'democrat', 'republican',
                               'ai generated', 'artificial intelligence', 'consumer trust',
                               'brand loyalty', 'dictator game', 'vaccine', 'covid',
                               'gender wage', 'wage gap', 'echo chamber'],
            'charitable_giving': ['polarization', 'partisan', 'ai generated',
                                    'artificial intelligence', 'consumer', 'purchase',
                                    'vaccine', 'covid', 'echo chamber', 'gender wage',
                                    'nudge', 'default option', 'retirement savings'],
            'dishonesty': ['polarization', 'partisan', 'consumer', 'purchase', 'brand',
                            'ai generated', 'artificial intelligence', 'vaccine', 'covid',
                            'echo chamber', 'gender wage', 'nudge', 'default option'],
            'covid': ['polarization', 'partisan', 'consumer', 'purchase', 'brand',
                       'ai generated', 'artificial intelligence', 'dictator game',
                       'echo chamber', 'gender wage', 'nudge', 'default option',
                       'retirement savings'],
            'accuracy_misinformation': ['consumer', 'purchase', 'brand', 'shopping',
                                         'dictator game', 'vaccine hesitancy', 'gender wage',
                                         'wage gap', 'nudge', 'default option',
                                         'retirement savings', 'ai generated content label'],
            'fairness': ['polarization', 'partisan', 'ai generated', 'artificial intelligence',
                          'consumer', 'purchase', 'vaccine', 'covid', 'echo chamber',
                          'nudge', 'default option'],
            'privacy': ['polarization', 'partisan', 'consumer', 'purchase', 'brand',
                         'vaccine', 'covid', 'dictator game', 'echo chamber',
                         'gender wage', 'wage gap', 'nudge', 'default option'],
            'punishment': ['consumer', 'purchase', 'brand', 'ai generated',
                            'artificial intelligence', 'vaccine', 'covid', 'echo chamber',
                            'gender wage', 'nudge', 'default option'],
            'deontology_utilitarianism': ['consumer', 'purchase', 'brand', 'ai generated',
                                            'vaccine', 'covid', 'echo chamber', 'nudge',
                                            'default option', 'gender wage', 'dictator game'],
            'media_communication': ['consumer', 'purchase', 'brand', 'dictator game',
                                      'vaccine hesitancy', 'gender wage', 'wage gap',
                                      'nudge', 'default option', 'retirement savings',
                                      'trolley', 'moral dilemma'],
            'risk_perception': ['consumer', 'purchase', 'brand', 'ai generated',
                                 'echo chamber', 'gender wage', 'dictator game',
                                 'nudge', 'default option', 'trolley'],
            'decision_making': ['polarization', 'partisan', 'echo chamber', 'gender wage',
                                 'vaccine hesitancy', 'consumer', 'brand', 'dictator game'],
            # --- Negative keywords for previously uncovered domains ---
            'marketing': ['polarization', 'partisan', 'democrat', 'republican',
                          'vaccine', 'covid', 'pandemic', 'dictator game', 'public goods',
                          'trolley', 'moral dilemma', 'gender wage', 'wage gap',
                          'retirement savings', 'echo chamber'],
            'technology': ['polarization', 'partisan', 'vaccine', 'covid', 'charity',
                           'donation', 'dictator game', 'trolley', 'moral dilemma',
                           'gender wage', 'wage gap', 'echo chamber'],
            'pricing': ['polarization', 'partisan', 'vaccine', 'covid', 'echo chamber',
                         'gender wage', 'trolley', 'moral dilemma', 'dictator game'],
            'hedonic_consumption': ['polarization', 'partisan', 'vaccine', 'covid',
                                      'echo chamber', 'gender wage', 'dictator game',
                                      'trolley', 'moral dilemma'],
            'utilitarian_consumption': ['polarization', 'partisan', 'vaccine', 'covid',
                                          'echo chamber', 'gender wage', 'dictator game',
                                          'trolley', 'moral dilemma'],
            'negotiation': ['polarization', 'partisan', 'vaccine', 'covid', 'echo chamber',
                             'consumer', 'brand', 'ai generated', 'gender wage',
                             'trolley', 'moral dilemma'],
            'personalization': ['polarization', 'partisan', 'vaccine', 'covid',
                                 'echo chamber', 'dictator game', 'trolley',
                                 'gender wage', 'wage gap'],
            'education': ['polarization', 'partisan', 'consumer', 'purchase', 'brand',
                           'vaccine hesitancy', 'dictator game', 'echo chamber',
                           'gender wage', 'trolley', 'moral dilemma'],
            'power_status': ['vaccine', 'covid', 'ai generated', 'echo chamber',
                              'consumer', 'purchase', 'brand', 'trolley', 'dictator game'],
            # --- Negative keywords for new domains ---
            'moral_psychology': ['consumer', 'purchase', 'brand', 'ai generated',
                                   'vaccine hesitancy', 'echo chamber', 'gender wage',
                                   'nudge', 'default option', 'retirement savings',
                                   'dictator game', 'public goods game'],
            'prosocial_behavior': ['consumer', 'purchase', 'brand', 'ai generated',
                                     'vaccine', 'covid', 'echo chamber', 'gender wage',
                                     'polarization', 'partisan', 'trolley',
                                     'nudge', 'default option'],
            'cultural_psychology': ['consumer', 'purchase', 'brand', 'ai generated',
                                      'vaccine', 'covid', 'echo chamber', 'dictator game',
                                      'trolley', 'nudge', 'default option',
                                      'retirement savings'],
            'computational_social_science': ['trolley', 'moral dilemma', 'dictator game',
                                                'gender wage', 'retirement savings',
                                                'nudge', 'default option'],
            'sports_psychology': ['consumer', 'purchase', 'brand', 'ai generated',
                                    'vaccine', 'covid', 'echo chamber', 'polarization',
                                    'partisan', 'trolley', 'dictator game', 'gender wage',
                                    'nudge', 'default option', 'retirement savings'],
            'behavioral_neuroscience': ['consumer', 'purchase', 'brand', 'echo chamber',
                                          'polarization', 'partisan', 'dictator game',
                                          'gender wage', 'retirement savings',
                                          'nudge', 'default option'],
            # --- Context-aware negative keywords for cross-domain false positives ---
            # "health" in marketing context should not trigger health_psychology
            # Adding marketing-specific health phrases as negatives for health_psychology
        }

        # ----------------------------------------------------------------
        # Cross-domain suppression groups: when a "signal domain" scores
        # highly, domains NOT in its affinity group get penalized.
        # ----------------------------------------------------------------
        domain_affinity_groups: Dict[str, List[str]] = {
            'political_psychology': ['media_communication', 'accuracy_misinformation',
                                      'social_psychology', 'emotions'],
            'ai': ['technology', 'consumer_behavior', 'privacy', 'decision_making',
                   'computational_social_science'],
            'consumer_behavior': ['marketing', 'pricing', 'hedonic_consumption',
                                   'utilitarian_consumption', 'ai', 'personalization'],
            'health_psychology': ['covid', 'risk_perception', 'emotions',
                                   'decision_making', 'behavioral_neuroscience'],
            'behavioral_economics': ['decision_making', 'pricing', 'consumer_behavior',
                                      'risk_perception', 'economic_games'],
            'gender': ['organizational_behavior', 'fairness', 'social_psychology',
                       'power_status', 'cultural_psychology'],
            'environmental': ['behavioral_economics', 'consumer_behavior',
                               'decision_making', 'prosocial_behavior'],
            'covid': ['health_psychology', 'risk_perception', 'media_communication'],
            'media_communication': ['political_psychology', 'accuracy_misinformation',
                                     'social_psychology', 'technology',
                                     'computational_social_science'],
            'social_psychology': ['moral_psychology', 'prosocial_behavior', 'emotions',
                                   'cultural_psychology', 'power_status', 'norm_elicitation'],
            'moral_psychology': ['deontology_utilitarianism', 'social_psychology',
                                  'dishonesty', 'punishment', 'prosocial_behavior'],
            'prosocial_behavior': ['charitable_giving', 'social_psychology',
                                    'moral_psychology', 'economic_games'],
            'cultural_psychology': ['social_psychology', 'moral_psychology',
                                     'organizational_behavior'],
            'behavioral_neuroscience': ['decision_making', 'emotions',
                                         'health_psychology', 'sports_psychology'],
            'sports_psychology': ['behavioral_neuroscience', 'emotions',
                                   'education'],
            'economic_games': ['behavioral_economics', 'fairness',
                                'prosocial_behavior', 'punishment'],
            'computational_social_science': ['media_communication', 'ai',
                                               'technology', 'political_psychology'],
        }

        domain_scores: Dict[str, float] = {}

        # ================================================================
        # PHASE 1: Score keywords from domain_keywords with tiered weighting
        # Keywords are already organized into tiers within _build_domain_keywords:
        #   - 3+ word phrases get weight 20
        #   - 2-word phrases get weight 10
        #   - Single words get weight 3
        # ================================================================
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                kw = keyword.lower()
                word_count = len(kw.split())

                # Determine weight based on phrase length (tier)
                if word_count >= 3:
                    weight = 20.0
                elif word_count == 2:
                    weight = 10.0
                else:
                    weight = 3.0

                # Always use word-boundary matching for precision
                pattern = r'\b' + re.escape(kw) + r'\b'
                if re.search(pattern, combined_text):
                    domain_scores[domain] = domain_scores.get(domain, 0.0) + weight
                    # Title matches get 50% bonus (title is more indicative)
                    if title_text and re.search(pattern, title_text):
                        domain_scores[domain] += weight * 0.5

        # ================================================================
        # PHASE 2: Apply negative keyword penalties
        # Each negative keyword match subtracts points proportional to how
        # strongly it contradicts the domain.
        # ================================================================
        domains_to_remove = []
        for domain, neg_keywords in domain_negative_keywords.items():
            if domain not in domain_scores:
                continue
            penalty = 0.0
            for neg_kw in neg_keywords:
                neg_pattern = r'\b' + re.escape(neg_kw) + r'\b'
                if re.search(neg_pattern, combined_text):
                    # Multi-word negative keywords are stronger signals
                    neg_word_count = len(neg_kw.split())
                    penalty += 5.0 * neg_word_count
            if penalty > 0:
                domain_scores[domain] = domain_scores[domain] - penalty
                if domain_scores[domain] <= 0:
                    domains_to_remove.append(domain)
        for d in domains_to_remove:
            del domain_scores[d]

        # ================================================================
        # PHASE 3: Cross-domain suppression
        # If a "signal domain" has a very strong score (>= 25), domains
        # outside its affinity group get suppressed by 30%.
        # This prevents unrelated domains from sneaking in due to generic
        # keyword overlap.
        # ================================================================
        if domain_scores:
            top_score = max(domain_scores.values())
            strong_domains = [d for d, s in domain_scores.items() if s >= 25.0]

            for signal_domain in strong_domains:
                if signal_domain in domain_affinity_groups:
                    affinity = set(domain_affinity_groups[signal_domain])
                    affinity.add(signal_domain)  # The signal domain itself is always allowed
                    for other_domain in list(domain_scores.keys()):
                        if other_domain not in affinity:
                            # Suppress by 30% for each strong signal that excludes it
                            domain_scores[other_domain] *= 0.7

        # ================================================================
        # PHASE 3b: Cross-domain synergy boosting
        # When two related domains are both detected, boost them together
        # since they commonly co-occur in real behavioral research.
        # Each pair: (domain_a, domain_b, boost_amount)
        # ================================================================
        synergy_pairs: list = [
            # Economics + Consumer
            ('behavioral_economics', 'consumer_behavior', 8.0),
            ('behavioral_economics', 'decision_making', 8.0),
            ('behavioral_economics', 'pricing', 6.0),
            # Consumer + Marketing
            ('consumer_behavior', 'marketing', 8.0),
            ('consumer_behavior', 'pricing', 6.0),
            ('consumer_behavior', 'personalization', 5.0),
            # Political + Media
            ('political_psychology', 'media_communication', 8.0),
            ('political_psychology', 'accuracy_misinformation', 7.0),
            # Social + Moral/Prosocial
            ('social_psychology', 'moral_psychology', 7.0),
            ('social_psychology', 'prosocial_behavior', 7.0),
            ('social_psychology', 'cultural_psychology', 6.0),
            ('social_psychology', 'emotions', 6.0),
            # Health related
            ('health_psychology', 'covid', 8.0),
            ('health_psychology', 'risk_perception', 6.0),
            # Economics / Games
            ('economic_games', 'behavioral_economics', 6.0),
            ('economic_games', 'fairness', 6.0),
            ('economic_games', 'prosocial_behavior', 5.0),
            # Moral / Ethics
            ('moral_psychology', 'deontology_utilitarianism', 8.0),
            ('moral_psychology', 'dishonesty', 6.0),
            ('moral_psychology', 'punishment', 5.0),
            # Environmental
            ('environmental', 'behavioral_economics', 5.0),
            ('environmental', 'prosocial_behavior', 5.0),
            # Gender + Organization
            ('gender', 'organizational_behavior', 7.0),
            ('gender', 'fairness', 5.0),
            # Tech / AI
            ('ai', 'technology', 7.0),
            ('ai', 'privacy', 5.0),
            # Neuroscience
            ('behavioral_neuroscience', 'decision_making', 6.0),
            ('behavioral_neuroscience', 'emotions', 5.0),
            # Charitable + Prosocial
            ('charitable_giving', 'prosocial_behavior', 7.0),
            # Cultural
            ('cultural_psychology', 'moral_psychology', 5.0),
        ]

        for domain_a, domain_b, boost in synergy_pairs:
            if domain_a in domain_scores and domain_b in domain_scores:
                # Both domains detected -> boost each by the synergy amount
                domain_scores[domain_a] += boost
                domain_scores[domain_b] += boost

        # ================================================================
        # PHASE 4: Context-aware boosts
        # Detect specific experimental patterns in the description and
        # boost the most relevant domain(s).
        # ================================================================
        context_boost_patterns: Dict[str, List[tuple]] = {
            # pattern -> (domain_to_boost, boost_amount)
            'consumer_behavior': [
                (r'product\s+review', 15.0),
                (r'consumer\s+trust\s+in', 15.0),
                (r'brand\s+evaluat', 10.0),
                (r'purchase\s+decision', 15.0),
                (r'willingness\s+to\s+pay', 15.0),
            ],
            'ai': [
                (r'ai[\s-]generated\s+content', 20.0),
                (r'ai[\s-]generated\s+label', 20.0),
                (r'algorithm\s+(aversion|appreciation)', 15.0),
                (r'human\s+vs\.?\s+ai', 15.0),
                (r'ai\s+disclosure', 15.0),
                (r'content\s+label', 10.0),
            ],
            'political_psychology': [
                (r'political\s+polariz', 20.0),
                (r'partisan\s+(divide|bias|identity)', 15.0),
                (r'echo\s+chamber', 15.0),
                (r'(liberal|conservative)\s+(media|news|voter)', 15.0),
                (r'(democrat|republican)\s+(voter|supporter|participant)', 15.0),
            ],
            'behavioral_economics': [
                (r'nudg(e|ing)\s+(retirement|savings|default|option)', 20.0),
                (r'default\s+(option|enrollment|choice)', 15.0),
                (r'framing\s+(effect|manipulation|condition)', 15.0),
                (r'loss\s+aversion', 15.0),
                (r'retirement\s+savings', 15.0),
            ],
            'health_psychology': [
                (r'vaccine\s+hesitanc', 20.0),
                (r'health\s+(intervention|behavior|risk)', 15.0),
                (r'patient\s+(adherence|compliance)', 15.0),
                (r'(medical|health)\s+decision', 15.0),
            ],
            'gender': [
                (r'gender\s+(wage|pay)\s+gap', 20.0),
                (r'wage\s+gap\s+in\s+stem', 20.0),
                (r'sex\s+difference', 15.0),
                (r'gender\s+(bias|stereotype|discrimination)', 15.0),
                (r'stem\s+organizat', 10.0),
            ],
            'media_communication': [
                (r'social\s+media\s+echo', 20.0),
                (r'echo\s+chamber', 15.0),
                (r'media\s+(consumption|influence|framing|effects)', 15.0),
                (r'news\s+consumption', 15.0),
            ],
            'risk_perception': [
                (r'risk\s+perception', 15.0),
                (r'perceived\s+risk', 15.0),
                (r'risk\s+(communicat|assess)', 10.0),
            ],
            'organizational_behavior': [
                (r'stem\s+organizat', 15.0),
                (r'(wage|pay)\s+gap\s+in\s+(stem|organizations)', 20.0),
                (r'workplace\s+(behavior|dynamics|culture)', 15.0),
                (r'employee\s+(engagement|satisfaction|turnover)', 15.0),
                (r'(gender|diversity)\s+in\s+organizations', 15.0),
            ],
            'decision_making': [
                (r'decision\s+making\s+under', 15.0),
                (r'choice\s+(overload|architecture)', 15.0),
                (r'default\s+option', 15.0),
                (r'nudg(e|ing)\s+.{0,20}default', 15.0),
                (r'through\s+default', 10.0),
            ],
            'covid': [
                (r'covid[\s-]?19', 15.0),
                (r'vaccine\s+hesitanc', 15.0),
                (r'pandemic\s+behavior', 15.0),
                (r'mask\s+wearing', 15.0),
            ],
            'environmental': [
                (r'carbon\s+footprint', 15.0),
                (r'climate\s+change', 15.0),
                (r'sustainab(le|ility)\s+(behavior|choice|consumption)', 15.0),
            ],
            'charitable_giving': [
                (r'charitab(le|ly)\s+giving', 15.0),
                (r'donation\s+behavior', 15.0),
                (r'warm\s+glow', 15.0),
            ],
            'accuracy_misinformation': [
                (r'fake\s+news', 15.0),
                (r'misinformation\s+(shar|suscept|spread)', 15.0),
                (r'truth\s+discernment', 15.0),
                (r'conspiracy\s+belief', 15.0),
            ],
            # --- Context boosts for new domains ---
            'moral_psychology': [
                (r'moral\s+(foundation|judgment|reasoning)', 15.0),
                (r'moral\s+(identity|licensing|disengagement)', 15.0),
                (r'ethical\s+(dilemma|judgment)', 10.0),
            ],
            'prosocial_behavior': [
                (r'prosocial\s+behavior', 15.0),
                (r'altruistic\s+(behavior|punishment|motivation)', 15.0),
                (r'helping\s+behavior', 15.0),
                (r'empathy[\s-]altruism', 15.0),
            ],
            'cultural_psychology': [
                (r'cross[\s-]cultural\s+(comparison|difference|study)', 20.0),
                (r'individualism[\s-]collectivism', 15.0),
                (r'cultural\s+(tightness|looseness|values|norms)', 15.0),
                (r'self[\s-]construal', 15.0),
            ],
            'computational_social_science': [
                (r'digital\s+(trace|footprint|behavior)', 15.0),
                (r'online\s+(behavior|experiment)', 15.0),
                (r'(text|sentiment|network)\s+analysis', 10.0),
            ],
            'sports_psychology': [
                (r'(athletic|sport)\s+performance', 15.0),
                (r'mental\s+toughness', 15.0),
                (r'choking\s+under\s+pressure', 20.0),
                (r'team\s+cohesion', 15.0),
            ],
            'behavioral_neuroscience': [
                (r'cognitive\s+load', 15.0),
                (r'working\s+memory', 15.0),
                (r'neural\s+correlat', 15.0),
                (r'dual[\s-]process', 15.0),
                (r'executive\s+function', 15.0),
            ],
        }

        for domain, patterns in context_boost_patterns.items():
            for pattern_str, boost in patterns:
                if re.search(pattern_str, combined_text):
                    domain_scores[domain] = domain_scores.get(domain, 0.0) + boost
                    # Extra boost if found in title
                    if title_text and re.search(pattern_str, title_text):
                        domain_scores[domain] += boost * 0.5

        # ================================================================
        # PHASE 5: Adaptive filter and rank
        # - Minimum absolute score threshold of 8 (avoids weak matches)
        # - Relative threshold adapts based on score distribution:
        #   * If top domain dominates (large gap to #2), lower threshold
        #     to be more inclusive of weaker secondary domains
        #   * If many domains score similarly, raise threshold to be
        #     more selective and avoid noisy results
        # - Cap at 3 domains
        # - Always return at least 1 domain
        # ================================================================
        if not domain_scores:
            return ['consumer_behavior']

        # Sort by score descending
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)

        top_score = sorted_domains[0][1]

        # Adaptive relative threshold based on score concentration
        if len(sorted_domains) >= 2:
            second_score = sorted_domains[1][1]
            # Ratio of second-highest to highest score
            concentration_ratio = second_score / top_score if top_score > 0 else 0.0

            if concentration_ratio < 0.3:
                # One domain dominates strongly - be more inclusive of
                # secondary domains (lower the relative threshold)
                relative_pct = 0.25
            elif concentration_ratio > 0.7:
                # Many domains score similarly - be more selective
                # to avoid returning too many loosely-related domains
                relative_pct = 0.45
            else:
                # Normal distribution - use standard threshold
                relative_pct = 0.35
        else:
            relative_pct = 0.35

        min_threshold = max(8.0, top_score * relative_pct)

        filtered = [d for d, s in sorted_domains if s >= min_threshold]

        # Cap at 3 domains
        result = filtered[:3] if filtered else [sorted_domains[0][0]]

        # Ensure we never return an empty list
        return result if result else ['consumer_behavior']

    def get_personas_for_domains(
        self,
        domains: List[str],
        include_response_styles: bool = True
    ) -> Dict[str, Persona]:
        """
        Get relevant personas for the detected domains.

        Args:
            domains: List of research domain names
            include_response_styles: Whether to include universal response style personas

        Returns:
            Dict of persona_name -> Persona objects
        """
        selected_personas = {}

        # Always include response style personas if requested
        if include_response_styles:
            for name, persona in self.personas.items():
                if persona.category == 'response_style':
                    selected_personas[name] = persona

        # Add domain-specific personas
        for name, persona in self.personas.items():
            if persona.category == 'response_style':
                continue

            # Check if persona applies to any of the detected domains
            for domain in domains:
                if domain in persona.applicable_domains or 'all' in persona.applicable_domains:
                    selected_personas[name] = persona
                    break

        return selected_personas

    def generate_participant_profile(
        self,
        persona: Persona,
        participant_id: int,
        study_seed: int
    ) -> Dict[str, float]:
        """
        Generate unique trait values for a participant based on their persona.

        Uses the persona's base traits but adds individual variation to ensure
        no two participants have identical trait profiles.

        Args:
            persona: The assigned persona
            participant_id: Unique participant identifier
            study_seed: Study-level seed for reproducibility

        Returns:
            Dict of trait_name -> trait_value (0-1 scale)
        """
        # Create unique seed for this participant using stable hash for reproducibility
        unique_seed = _stable_int_hash(f"{study_seed}_{participant_id}_{persona.name}") % (2**31)
        rng = np.random.RandomState(unique_seed)

        traits = {}

        # ================================================================
        # CORRELATED RESPONSE STYLE TRAITS
        # Acquiescence, Extremity, and Social Desirability Covariance
        # ================================================================
        # In real data, these three response biases are NOT independent.
        # They co-vary in systematic ways documented in the literature:
        #
        # Baumgartner & Steenkamp (2001, Marketing Research):
        #   - Acquiescence correlates with Extremity at r ~ 0.15
        #   - Both are "content-free" response tendencies
        #
        # Paulhus (2002, Psychological Assessment):
        #   - Social Desirability correlates with Acquiescence at r ~ 0.25
        #   - Both involve agreeable/positive responding
        #
        # Greenleaf (1992); Baumgartner & Steenkamp (2001):
        #   - Extremity negatively correlates with Social Desirability at r ~ -0.10
        #   - SD respondents moderate their responses to seem "normal"
        #
        # Implementation: Generate z-scores from a multivariate normal with
        # the empirical correlation structure, then transform to each trait's
        # persona-specific mean and SD using the Cholesky decomposition.
        # ================================================================
        _correlated_traits = ['acquiescence', 'extremity', 'social_desirability']
        _corr_means = []
        _corr_sds = []
        _corr_trait_objects = []
        _has_all_corr_traits = True

        for _ct in _correlated_traits:
            if _ct in persona.traits:
                _corr_trait_objects.append(persona.traits[_ct])
                _corr_means.append(persona.traits[_ct].base_mean)
                _corr_sds.append(persona.traits[_ct].base_sd)
            else:
                _has_all_corr_traits = False
                break

        if _has_all_corr_traits and len(_corr_trait_objects) == 3:
            # Empirical correlation matrix from published research:
            # [acquiescence, extremity, social_desirability]
            _R = np.array([
                [1.00, 0.15, 0.25],   # Acquiescence
                [0.15, 1.00, -0.10],   # Extremity
                [0.25, -0.10, 1.00],   # Social Desirability
            ])

            # Cholesky decomposition for correlated sampling
            try:
                _L = np.linalg.cholesky(_R)
                # Generate independent z-scores
                _z = rng.normal(0.0, 1.0, size=3)
                # Transform to correlated z-scores
                _corr_z = _L @ _z

                # Convert correlated z-scores to trait values using each
                # trait's persona-specific mean and SD
                for idx, _ct in enumerate(_correlated_traits):
                    _value = _corr_means[idx] + _corr_z[idx] * _corr_sds[idx]
                    traits[_ct] = float(np.clip(_value, 0.01, 0.99))
            except np.linalg.LinAlgError:
                # Fallback to independent sampling if Cholesky fails
                # (should not happen with valid correlation matrix)
                for _ct in _correlated_traits:
                    _trait = persona.traits[_ct]
                    _value = rng.normal(_trait.base_mean, _trait.base_sd)
                    traits[_ct] = float(np.clip(_value, 0.01, 0.99))
        # else: traits will be generated independently below

        # Generate remaining traits independently
        for trait_name, trait in persona.traits.items():
            if trait_name in traits:
                # Already generated as part of correlated set
                continue
            # Generate value from normal distribution centered on persona mean
            value = rng.normal(trait.base_mean, trait.base_sd)
            # Clip to valid range
            traits[trait_name] = float(np.clip(value, 0.01, 0.99))

        # ================================================================
        # G-FACTOR: General Evaluation Tendency (Podsakoff et al., 2003)
        # ================================================================
        # Each participant has a stable general evaluation tendency (g-factor)
        # that represents their overall positivity/negativity in ratings.
        # Some people rate everything slightly higher, others slightly lower.
        #
        # This is grounded in Common Method Variance (CMV) research:
        #   - Podsakoff et al. (2003): Method factors account for ~25% of
        #     variance in self-report measures on average
        #   - The g-factor loads differentially on different construct types:
        #     * Attitudes/evaluations: loading ~0.25 (high CMV susceptibility)
        #     * Behavioral intentions: loading ~0.15 (moderate CMV)
        #     * Factual/behavioral reports: loading ~0.08 (low CMV)
        #     * Risk/threat perceptions: loading ~0.12 (moderate CMV)
        #
        # The g-factor is drawn from N(0, 1) and stored as a latent score.
        # It produces within-person coherence across scales beyond what
        # the existing latent DV correlation system provides.
        #
        # Reference: Podsakoff, P. M., MacKenzie, S. B., Lee, J.-Y., &
        #   Podsakoff, N. P. (2003). "Common method biases in behavioral
        #   research." Journal of Applied Psychology, 88(5), 879-903.
        # ================================================================
        g_factor_z = float(rng.normal(0.0, 1.0))
        # Engaged/consistent personas have stronger g-factor expression
        # (they respond more coherently), while careless respondents have
        # weaker expression (noise dilutes the latent factor)
        consistency = traits.get('response_consistency', 0.65)
        g_factor_strength = 0.12 + (consistency - 0.5) * 0.16
        g_factor_strength = float(np.clip(g_factor_strength, 0.04, 0.22))
        traits['_g_factor_z'] = g_factor_z
        traits['_g_factor_strength'] = g_factor_strength

        return traits


# ================================================================
# TEXT GENERATION TEMPLATES
# ================================================================

class TextResponseGenerator:
    """
    Generates realistic open-ended text responses based on persona and study context.

    Uses template-based generation with extensive persona-specific variation to create
    plausible qualitative data that resembles real human survey responses. Templates
    are designed to capture natural language patterns including:
    - Varied sentence structures and lengths
    - Appropriate hedging and qualifiers
    - Personal voice and perspective
    - Response style consistent with numeric patterns

    Ensures no duplicate responses within a dataset by tracking used responses.
    """

    def __init__(self):
        self.templates = self._build_templates()
        self.sentence_starters = self._build_sentence_starters()
        self.hedges = self._build_hedges()
        self.intensifiers = self._build_intensifiers()
        self._used_responses: set = set()  # Track used responses for uniqueness
        self._variation_phrases = self._build_variation_phrases()

    def reset_used_responses(self):
        """Reset the set of used responses for a new dataset."""
        self._used_responses = set()

    def _build_variation_phrases(self) -> Dict[str, List[str]]:
        """Build phrases that can be prepended/appended to make responses unique."""
        return {
            'time_phrases': [
                "At first,", "Initially,", "After thinking about it,", "Upon reflection,",
                "Looking at it now,", "Considering everything,", "In retrospect,",
                "After some thought,", "Thinking it over,", "On second thought,",
            ],
            'personal_phrases': [
                "Personally,", "For me,", "In my experience,", "From my perspective,",
                "Speaking for myself,", "In my view,", "To me,", "As I see it,",
                "From where I stand,", "In my opinion,",
            ],
            'certainty_phrases': [
                "I'm fairly sure that", "I believe that", "I think", "I feel like",
                "It seems to me that", "My impression is that", "I'd say that",
                "I'm pretty confident that", "I'm inclined to think", "I would argue that",
            ],
            'ending_phrases': [
                " That's my take on it.", " That's how I see it.", " Those are my thoughts.",
                " That's my perspective.", " That sums up my view.", " That's where I stand.",
                "", "", "",  # Some empty to vary whether ending is added
            ]
        }

    def _build_sentence_starters(self) -> Dict[str, List[str]]:
        """Build varied sentence starters for natural language generation."""
        return {
            'positive': [
                "I really liked", "I was impressed by", "I appreciated",
                "What stood out to me was", "The best part was",
                "I found myself enjoying", "I was pleasantly surprised by",
                "I thought it was great that", "I loved how",
            ],
            'negative': [
                "I wasn't convinced by", "I had concerns about",
                "I didn't really connect with", "I found it disappointing that",
                "I was skeptical of", "I struggled with",
                "I didn't see the appeal of", "I was put off by",
            ],
            'neutral': [
                "I noticed that", "I observed", "From what I saw",
                "Looking at this", "Considering the", "Based on what was shown",
                "It seemed like", "I would say that",
            ],
            'engaged': [
                "After careful consideration,", "Thinking about this carefully,",
                "From my perspective,", "In my honest opinion,",
                "Reflecting on this,", "After reviewing everything,",
            ],
            'brief': [
                "Basically,", "Overall,", "In short,", "Simply put,", "",
            ]
        }

    def _build_hedges(self) -> List[str]:
        """Build hedging phrases for natural uncertainty."""
        return [
            "I think", "I feel like", "In my opinion", "I believe",
            "It seems to me", "From what I can tell", "I'd say",
            "Personally", "To me", "I suppose", "I guess",
        ]

    def _build_intensifiers(self) -> Dict[str, List[str]]:
        """Build intensifiers for sentiment strength."""
        return {
            'strong_positive': ["really", "definitely", "absolutely", "totally", "very"],
            'mild_positive': ["somewhat", "fairly", "pretty", "kind of", "reasonably"],
            'strong_negative': ["really", "definitely", "absolutely", "totally", "quite"],
            'mild_negative': ["somewhat", "a bit", "slightly", "kind of", "a little"],
        }

    def _build_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Build extensive template library for different response types and personas.

        v1.0.7.9: MAJOR OVERHAUL — Added question-type-specific template banks
        (opinion_response, explanation_response, feeling_response, description_response,
        evaluation_response) that produce content-responsive answers instead of
        meta-commentary about the task. The old task_summary templates described the
        PROCESS of answering ("I thought about {topic}") rather than actually
        ANSWERING the question.
        """
        return {
            # v1.0.8.0: Opinion templates — EXPANDED with diverse perspectives
            'opinion_response': {
                'engaged': [
                    "I think {topic} is really important and my views on it are pretty clear. I've thought about this a lot and I feel confident in where I stand.",
                    "When it comes to {topic}, I have strong opinions based on my own experiences. I've seen enough to know how I feel about it.",
                    "{topic} is something I feel I can speak to honestly. My perspective comes from real experience not just reading about it.",
                    "I have genuine views on {topic} that I've developed over time. This isn't something I take lightly.",
                    "My take on {topic} is based on what I've personally observed and experienced. I try to be thoughtful about it.",
                    "{topic} matters to me and I think my position on it is well-reasoned. I've considered different angles.",
                    "I've formed my views on {topic} through a mix of experience and reflection. I feel pretty solid about where I land.",
                    "There's a lot to say about {topic} and I have real opinions on it. I tried to be honest about what I actually think.",
                    "I've given {topic} a lot of thought over the years. My opinions have been tested and refined through real experience.",
                    "{topic} comes up a lot in my life and I've developed a clear perspective. I feel I can back up my views with real examples.",
                    "My position on {topic} isn't just off the top of my head. I've lived this and have strong views because of that.",
                    "I pay attention to {topic} more than most people and that informs where I stand. I feel my view is well-grounded.",
                ],
                'satisficer': [
                    "{topic} is fine I guess.",
                    "I don't think too hard about {topic}.",
                    "{topic}, it's ok.",
                    "Not much to say about {topic}.",
                    "My view on {topic} is pretty simple.",
                    "{topic} is whatever.",
                    "Haven't thought about {topic} that much honestly.",
                    "{topic} is not something I spend a lot of time on.",
                ],
                'extreme': [
                    "I feel VERY strongly about {topic} and I don't think there's room for debate here.",
                    "{topic} is absolutely something I have extreme views on. No middle ground for me.",
                    "I have zero doubt about where I stand on {topic}. My position is crystal clear.",
                    "There's no question in my mind about {topic}. I feel extremely strongly.",
                    "When it comes to {topic}, I'm all in on my position. Anyone who disagrees hasn't thought it through.",
                    "My views on {topic} couldn't be stronger. This is a hill I'll die on.",
                ],
                'careless': [
                    "{topic} idk",
                    "{topic} whatever",
                    "meh {topic}",
                    "{topic} ok",
                    "idk about {topic}",
                    "{topic} i guess",
                    "yeah {topic}",
                    "{topic} sure",
                    "{topic} no opinion",
                    "lol {topic}",
                ],
                'default': [
                    "I have views on {topic} and I tried to express them honestly. It's something I've thought about before.",
                    "My perspective on {topic} is based on my own experiences. I gave my genuine take.",
                    "{topic} — I have real opinions and I shared them as honestly as I could.",
                    "When it comes to {topic}, I know where I stand. I answered based on that.",
                    "I shared my actual views on {topic}. Nothing more nothing less.",
                    "{topic} is something I have thoughts on and I tried to capture them accurately.",
                    "I answered honestly about {topic}. My views come from personal experience.",
                    "My thoughts on {topic} are based on how I actually see things day to day.",
                    "I gave my genuine perspective on {topic}. It reflects how I really feel.",
                    "{topic} is something I can weigh in on from personal experience.",
                ],
            },
            # v1.0.8.0: Explanation templates — EXPANDED
            'explanation_response': {
                'engaged': [
                    "The reason I feel this way about {topic} is that my personal experiences have shaped my view. I've seen enough to know where I stand.",
                    "I made my choice about {topic} because it aligns with my values. I thought carefully about what matters to me.",
                    "My reasoning on {topic} comes from real-life experience. When you've dealt with something directly it changes your perspective.",
                    "I can explain my position on {topic} — it comes down to what I believe is right based on what I've seen and experienced.",
                    "The way I see {topic}, there are clear reasons for my position. My past experience informs my thinking.",
                    "My rationale for how I responded about {topic} is straightforward. I went with what I genuinely believe.",
                    "Several factors influenced my thinking about {topic}. I weighed different considerations and this is where I landed.",
                    "I responded the way I did about {topic} because of specific things I've observed. Not just gut feeling but actual reasons.",
                    "My explanation for where I stand on {topic} is rooted in things I've personally dealt with. Real situations shaped this view.",
                    "There's a clear logic behind my views on {topic}. I can trace my reasoning back to specific experiences.",
                ],
                'satisficer': [
                    "Just how I feel about {topic}.",
                    "No big reason, {topic} is just my view.",
                    "Its just {topic}, went with my gut.",
                    "Thats just how I see {topic}.",
                    "Idk hard to explain, just {topic}.",
                    "No deep reason, thats just my take on {topic}.",
                ],
                'extreme': [
                    "The reason is obvious if you think about {topic} — I can't understand how anyone sees it differently.",
                    "My reasoning on {topic} is rock solid. I know exactly why I feel this way.",
                    "It's clear to me why {topic} matters so much. The evidence is overwhelming.",
                    "My explanation is simple: {topic} is one of those things where the answer is obvious to anyone paying attention.",
                    "I feel this way about {topic} because the facts are undeniable. There's really no other reasonable position.",
                ],
                'careless': [
                    "idk why, just {topic}",
                    "{topic} because",
                    "no reason really, {topic}",
                    "just because {topic}",
                    "cant explain it {topic}",
                    "{topic} dunno",
                ],
                'default': [
                    "My thinking on {topic} is based on my personal experience and values. That's the honest explanation.",
                    "I can explain my response to {topic} — it comes from how I actually see things.",
                    "The reason behind my views on {topic} is pretty straightforward. I went with my genuine perspective.",
                    "When it comes to {topic}, my reasoning is based on what I've actually experienced.",
                    "I responded the way I did because {topic} connects to things I've lived through. That's the real reason.",
                    "My position on {topic} makes sense when you consider my background and experience. That's where it comes from.",
                ],
            },
            # v1.0.8.0: Feeling/emotional reaction templates — EXPANDED
            'feeling_response': {
                'engaged': [
                    "When I think about {topic}, I have a genuine emotional response. It's something that affects me on a personal level.",
                    "{topic} stirs up real feelings for me. I tried to express those honestly because I think it matters.",
                    "My emotional reaction to {topic} is genuine. I feel connected to this issue in a way that's hard to fully articulate.",
                    "I have real feelings about {topic} that go beyond just an opinion. This touches something deeper for me.",
                    "{topic} evokes strong emotions for me because of my personal experiences. I didn't hold back.",
                    "How I feel about {topic} is complicated but genuine. I tried to capture the full range of my emotions.",
                    "{topic} hits home for me in a way that's hard to explain unless you've been through it. The emotions are very real.",
                    "I'm not usually this open about my feelings but {topic} genuinely moves me. I felt compelled to express that.",
                    "There's an emotional weight to {topic} for me. It connects to things in my life that matter deeply.",
                    "My reaction to {topic} was more emotional than I expected. It clearly taps into something meaningful for me.",
                ],
                'satisficer': [
                    "{topic} doesn't make me feel much either way.",
                    "I feel ok about {topic}.",
                    "Not strong feelings about {topic}.",
                    "{topic} — I don't feel too strongly.",
                ],
                'extreme': [
                    "I feel INCREDIBLY strongly about {topic}. My emotions are intense and I make no apologies.",
                    "{topic} triggers an extreme emotional response in me. I can't be neutral about this.",
                    "My feelings about {topic} are powerful and deeply held. This is visceral for me.",
                ],
                'careless': [
                    "{topic} dont care",
                    "whatever about {topic}",
                    "no feelings on {topic}",
                    "{topic} meh",
                ],
                'default': [
                    "My feelings about {topic} are genuine. I tried to express how I actually feel about it.",
                    "When it comes to {topic}, I have real emotions. I shared those as honestly as I could.",
                    "{topic} makes me feel a certain way and I tried to capture that in my response.",
                    "I have genuine emotional reactions to {topic}. My answer reflects those feelings.",
                ],
            },
            # v1.0.8.0: Description templates — EXPANDED
            'description_response': {
                'engaged': [
                    "My experience with {topic} was something I remember clearly. I tried to describe what actually happened and how it went.",
                    "I can describe my interaction with {topic} in detail. There were specific things that stood out to me.",
                    "What happened with {topic} was memorable enough that I can give a real description. I paid attention throughout.",
                    "I observed several things about {topic} that I tried to capture. The details matter to me.",
                    "Let me describe what stood out about {topic}. There were specific moments and details that I noticed and want to share.",
                    "I remember {topic} pretty vividly. The details are fresh in my mind and I tried to be specific about what I observed.",
                    "When it comes to describing {topic}, I can speak from direct experience. I noticed particular things that shaped my impression.",
                    "My encounter with {topic} had several notable aspects. I tried to describe not just what happened but how it felt to be part of it.",
                ],
                'satisficer': [
                    "{topic} was fine. Normal.",
                    "Nothing much to describe about {topic}.",
                    "{topic}, it went as expected.",
                    "Pretty standard experience with {topic}.",
                    "{topic} was uneventful.",
                ],
                'extreme': [
                    "The experience with {topic} was absolutely remarkable — either amazingly good or terrible, hard to forget.",
                    "What happened with {topic} was extreme. There's no way to describe it as average.",
                    "{topic} was one of the most intense experiences I can describe. Nothing about it was moderate.",
                ],
                'careless': [
                    "{topic} was ok",
                    "fine {topic}",
                    "normal {topic}",
                    "{topic} nothing special",
                ],
                'default': [
                    "My experience with {topic} was what it was. I described it as honestly as I could.",
                    "I tried to give an accurate description of {topic} based on what I observed.",
                    "What I noticed about {topic} is what I shared. Tried to be accurate.",
                    "I described {topic} based on what I actually saw and experienced. Nothing exaggerated.",
                    "My description of {topic} reflects what actually happened from my perspective.",
                    "I tried to be straightforward about {topic}. Just shared what I noticed.",
                ],
            },
            # v1.0.8.0: Evaluation/rating explanation templates — EXPANDED
            'evaluation_response': {
                'engaged': [
                    "I evaluated {topic} carefully and my assessment is based on real criteria that matter to me. I tried to be fair and thorough.",
                    "My rating of {topic} reflects what I genuinely think. I considered the strengths and weaknesses before making my judgment.",
                    "I took the evaluation of {topic} seriously. My assessment is based on what I actually observed, not assumptions.",
                    "When assessing {topic}, I tried to be balanced but honest. Some aspects stood out more than others.",
                    "I looked at {topic} from multiple angles before settling on my evaluation. There were strengths and weaknesses I tried to weigh fairly.",
                    "My assessment of {topic} accounts for both the positives and negatives. I didn't just go with my first impression.",
                    "I spent real time thinking about how to evaluate {topic}. My rating considers specific criteria that matter to me.",
                    "Evaluating {topic} required me to think carefully about what standards I was applying. I tried to be consistent and fair.",
                ],
                'satisficer': [
                    "{topic} is average.",
                    "My rating of {topic} is middle of the road.",
                    "{topic} is ok, not great not bad.",
                    "Can't rate {topic} too high or too low. Its in between.",
                    "Average marks for {topic}.",
                ],
                'extreme': [
                    "My assessment of {topic} is extreme and I stand by it completely. The quality was either outstanding or awful.",
                    "I rated {topic} at the extreme end because that's where it deserves to be. No hedging.",
                    "My evaluation of {topic} is at the far end of the scale and I don't think that's unfair at all.",
                ],
                'careless': [
                    "{topic} is fine whatever",
                    "idk {topic} is ok",
                    "{topic} average",
                    "cant really evaluate {topic}",
                ],
                'default': [
                    "I evaluated {topic} based on my honest impressions. My rating reflects what I actually think.",
                    "My assessment of {topic} is genuine. I tried to be fair in my evaluation.",
                    "I rated {topic} based on my real experience with it.",
                    "My evaluation of {topic} is straightforward. I went with my honest impression.",
                    "I assessed {topic} based on what I know and have experienced. It's a genuine rating.",
                    "I tried to be fair when evaluating {topic}. My scores reflect real opinions.",
                ],
            },
            # v1.0.8.3: CREATIVE BELIEF templates — conspiracy theories, paranormal, superstitions
            # These generate ACTUAL CONTENT (specific conspiracy theories / beliefs), not opinions about beliefs.
            # Grounded in: Brotherton et al. (2013) Generic Conspiracist Beliefs Scale;
            # Swami et al. (2011) Conspiracy Mentality; Douglas et al. (2017) psychological underpinnings.
            'creative_belief_response': {
                'engaged': [
                    "Ok so I know this sounds crazy but I genuinely think the government has way more surveillance tech than they let on. Like there are patents for stuff that supposedly doesn't exist. I've gone down that rabbit hole and there's actually decent evidence if you look for it.",
                    "My conspiracy theory is about big pharma. I think they deliberately suppress certain cheap remedies because there's no profit in curing people with generic drugs. A friend of mine worked in the industry and told me some things that really opened my eyes.",
                    "I believe there's a lot more coordination between major media outlets than people realize. Not like a secret society exactly but more like they all get the same talking points and run with them. The stories that get buried are more telling than the ones they push.",
                    "Honestly I think a lot of historical events were way more complex than what we're taught. The official stories are simplified to the point of being misleading. When you actually read declassified documents it paints a completely different picture.",
                    "I'm convinced that social media algorithms are specifically designed to keep people addicted and angry. Not as a side effect but as the actual goal. Internal documents from these companies basically confirm it but nothing changes.",
                    "My craziest one is that I think the food industry knowingly puts addictive compounds in processed food. The amount of sugar in everything is not an accident. They have scientists whose entire job is making food impossible to stop eating.",
                    "I believe that most major political scandals are orchestrated distractions from actual policy changes happening behind the scenes. While everyone argues about the latest controversy, laws get passed that nobody reads or discusses.",
                    "Here's mine: I think the education system is deliberately designed to produce compliant workers rather than critical thinkers. The way schools are structured — bells, rows, standardized tests — it's basically factory training.",
                    "I genuinely believe there are technologies being suppressed because they'd disrupt too many industries. Energy tech specifically. Too many patents have been bought and shelved by major corporations.",
                    "My conspiracy theory is that the housing market is manipulated by a handful of investment firms who deliberately restrict supply to keep prices high. The data actually supports this more than most people realize.",
                ],
                'satisficer': [
                    "I guess the moon landing stuff is kind of interesting. Not sure I fully believe it but some of the photo evidence is weird.",
                    "Idk maybe that the government hides UFO stuff. Seems possible.",
                    "Something about how social media listens to your conversations. Happens too often to be coincidence.",
                    "I think big companies definitely collude on prices. Not really that crazy I guess.",
                    "Maybe that some elections have been tampered with. Wouldn't be surprised.",
                ],
                'extreme': [
                    "I am 100% CERTAIN that the deep state is real and actively controlling politics behind the scenes. Anyone who denies it hasn't been paying attention. The evidence is OVERWHELMING if you actually look.",
                    "Wake up people! The entire financial system is designed to keep ordinary people in debt. Central banks, Wall Street, all of it. This isn't even a conspiracy it's just how it works and they don't even hide it anymore!",
                    "I KNOW for a FACT that major world events are planned. Not coincidences. Everything from economic crashes to political upheavals. There are people pulling strings and they've been doing it for decades.",
                    "The media is COMPLETELY controlled. Every single major outlet. If you still trust mainstream news you are being manipulated and that's just the truth. Independent journalists who get too close to the truth get silenced.",
                ],
                'careless': [
                    "illuminati or whatever",
                    "idk aliens",
                    "government stuff lol",
                    "moon landing maybe fake idk",
                    "the usual ones I guess",
                ],
                'default': [
                    "I think there's probably some truth to the idea that large corporations have more influence over government policy than they should. It's not exactly a conspiracy theory since lobbying is legal, but the extent of it feels wrong.",
                    "I've always thought it was suspicious how certain news stories dominate the cycle while other important stuff gets buried. Whether it's coordinated or just how incentives work, the effect is the same.",
                    "My conspiracy-ish belief is that planned obsolescence is real. Companies definitely make products that break after a certain time so you have to buy new ones. Apple basically admitted to this with the battery thing.",
                    "I think social media companies know exactly how harmful their products are for mental health and don't care because engagement equals profit. Not sure if that counts as conspiracy or just capitalism.",
                    "I believe pharmaceutical companies have way too much influence over what doctors prescribe. The amount of money they spend on marketing to doctors is kind of crazy when you look at the numbers.",
                    "I think most conspiracy theories have a kernel of truth that gets distorted. Like the government probably does keep some things secret, but it's more about incompetence than grand plans.",
                ],
            },
            # v1.0.8.3: PERSONAL DISCLOSURE templates — secrets, private information, confessions
            # Grounded in: Pennebaker (1997) Opening Up; Jourard (1971) Self-Disclosure;
            # Cozby (1973) self-disclosure literature; Dindia & Allen (1992) meta-analysis.
            'personal_disclosure_response': {
                'engaged': [
                    "Something my family knows that most people don't is that I struggled really badly with anxiety in my early twenties. I barely left the house for almost a year. I've gotten a lot better but it shaped who I am.",
                    "My family knows I almost dropped out of college. I was one semester away from quitting because I felt completely lost about what I wanted to do with my life. My mom talked me into finishing and I'm glad she did.",
                    "Nobody outside my family knows that I was adopted. My parents told me when I was twelve and it changed how I understood a lot of things about myself. I've never told friends because I don't want it to define me.",
                    "Here's something personal: I have a learning disability that I've never told anyone at work about. My family helped me develop workarounds when I was young and most people have no idea.",
                    "My family knows I went through a really dark period after a breakup where I basically stopped functioning. I called in sick to work for weeks. They were the ones who got me through it.",
                    "Something only my family would know is that I secretly send money to a relative who's struggling. I don't talk about it because I don't want to embarrass them or make it weird.",
                    "My family knows I considered joining the military instead of going to college. I was actually at the recruiting office. Changed my mind at the last minute and never told anyone else.",
                    "Only my family knows about a medical scare I had a few years ago. I kept it completely private from friends and coworkers because I didn't want the attention or pity.",
                    "Something personal is that my family went through bankruptcy when I was a teenager. We lost our house. It fundamentally changed my relationship with money and I still feel its effects.",
                    "My family knows I was bullied pretty severely in middle school. I've never told my current friends because by the time I met them I'd become a very different person.",
                ],
                'satisficer': [
                    "I guess my family knows I hate my job. I don't tell other people that.",
                    "Something private I guess is that I failed a class in school once. Not a big deal but I never told friends.",
                    "My family knows I don't like some of our relatives. That's about it.",
                    "I have a tattoo in a hidden spot that only family has seen. It's not that exciting.",
                    "My family knows about a dumb thing I did as a teenager. Rather not say what exactly.",
                ],
                'extreme': [
                    "My family knows something about me that would SHOCK everyone I know. I completely reinvented myself when I moved cities. The person I am now is nothing like who I was before and that's very intentional.",
                    "There's a family secret that I will take to my grave. It involves something that happened when I was young that would completely change how people see my family. I've never said a word to anyone.",
                    "My family knows I have a side of me that I never show ANYONE else. It's not bad exactly, but it's so different from my public persona that people literally would not believe it.",
                    "My family knows about a decision I made that I will NEVER regret even though everyone else would judge me for it. It was the hardest choice I've ever made and I'd do it again in a heartbeat.",
                    "Only my family knows the REAL reason I left my last job. What I tell people is completely different from what actually happened and if the truth came out it would change how people see that whole situation.",
                ],
                'careless': [
                    "idk family stuff",
                    "nothing really comes to mind",
                    "rather not say",
                    "pass",
                    "my family knows stuff I guess",
                ],
                'default': [
                    "My family knows that I'm much more sensitive than I let on. At work and with friends I seem pretty easy-going but at home I actually worry about things a lot more than people would guess.",
                    "Something my family knows is that I almost made a completely different career choice. I was accepted to a different program but switched last minute. Sometimes I wonder what would have happened.",
                    "Only my family really knows how much I struggled with a health issue a few years back. I kept it private from everyone else because I didn't want to be treated differently.",
                    "My family knows I'm not as confident as I seem. I put on a good front but inside I second-guess myself more than anyone would expect.",
                    "Something personal that only my family would know is that I have a really emotional attachment to a specific place from my childhood. Going back there hits me harder than I'd ever admit to friends.",
                    "My family knows I went through a period where I seriously questioned some major life decisions. It wasn't a crisis exactly but it was a tough time that I kept very private.",
                ],
            },
            # v1.0.8.3: CREATIVE NARRATIVE templates — general narrative/story generation
            # For "tell us your craziest/wildest/most memorable X" type questions.
            # Grounded in: Green & Brock (2000) narrative transportation; Bruner (1991) narrative psychology.
            'creative_narrative_response': {
                'engaged': [
                    "The craziest thing related to {topic} that I've experienced was completely unexpected. I was in a situation where everything I assumed turned out to be wrong. It really changed how I think about {topic} going forward.",
                    "I have a wild story about {topic}. Without going into too many details, I witnessed something that most people would not believe. It happened a few years ago and I still think about it regularly.",
                    "My most memorable experience with {topic} happened when I was least expecting it. The situation was so bizarre that I called someone immediately afterward just to have a witness to what I'd seen.",
                    "When it comes to {topic}, I once had an experience that completely defied my expectations. I went in thinking one thing and came out with a completely different perspective. It was genuinely eye-opening.",
                    "The wildest thing about {topic} in my life was a situation that escalated beyond anything I could have predicted. Looking back, the chain of events seems almost too perfect to be coincidental.",
                    "I have a {topic} story that I rarely tell because people don't believe me. But it genuinely happened and it's one of those things that makes me question certain assumptions.",
                    "My experience with {topic} took such an unexpected turn that I still bring it up years later. The contrast between what I expected and what actually happened was striking.",
                    "Here's my {topic} story: I found myself in a situation where the normal rules didn't seem to apply. Everyone involved was as confused as I was. It was surreal honestly.",
                ],
                'satisficer': [
                    "I don't have a super crazy {topic} story. Just normal stuff I guess.",
                    "Nothing too wild about {topic} in my life. Pretty standard.",
                    "I've seen some stuff related to {topic} but nothing that stands out.",
                    "My experience with {topic} is pretty boring honestly. Nothing worth writing about.",
                ],
                'extreme': [
                    "My {topic} experience was absolutely INSANE. I'm not exaggerating at all. What happened was so extreme that people literally don't believe me when I tell them.",
                    "The craziest {topic} thing in my life was genuinely unbelievable. It was one of those once-in-a-lifetime situations that changes everything about how you see the world.",
                    "I have a {topic} story that would blow your mind. The intensity of what happened was off the charts and I still feel the effects of it today.",
                ],
                'careless': [
                    "idk nothing comes to mind about {topic}",
                    "{topic} nothing special",
                    "cant think of anything about {topic}",
                    "dont really have a {topic} story",
                ],
                'default': [
                    "I've had some interesting experiences related to {topic} over the years. Nothing too dramatic but a few things that made me think differently about it.",
                    "When it comes to {topic}, I have a few stories that stand out. The most interesting one involved a situation that didn't go the way anyone expected.",
                    "My most notable {topic} experience was when I realized that my assumptions were completely off. The reality was very different from what I'd imagined.",
                    "I've encountered {topic} in ways that surprised me. One experience in particular sticks with me because it was so different from what I'd expected.",
                    "I have a {topic} story from a few years back that I think about sometimes. It wasn't dramatic but it gave me a new perspective.",
                ],
            },
            # v1.0.8.3: PERSONAL STORY templates — "tell us about a time when..."
            'personal_story_response': {
                'engaged': [
                    "There was a time when {topic} came up in my life in a really unexpected way. I was dealing with a situation at work and it forced me to confront how I actually felt versus how I thought I should feel.",
                    "I remember a specific experience with {topic} that stays with me. It happened during a difficult period in my life and the way it played out taught me something important about myself.",
                    "My most significant experience with {topic} was when I had to make a real decision about it, not just think about it abstractly. The pressure of the situation made me see things differently.",
                    "I have a personal story about {topic} from when I was younger. At the time I didn't fully understand what was happening but looking back now it makes a lot more sense.",
                    "My experience with {topic} really came into focus during a conversation with someone close to me. They said something that made me realize I'd been thinking about it all wrong.",
                ],
                'satisficer': [
                    "I had a thing related to {topic} once. It was ok.",
                    "Something about {topic} happened a while back. Don't remember all the details.",
                    "I guess I've dealt with {topic} before. Nothing major.",
                ],
                'careless': [
                    "cant think of a {topic} story",
                    "idk {topic}",
                    "nothing about {topic} comes to mind",
                ],
                'default': [
                    "I've had experiences with {topic} that shaped how I think about it. One situation in particular stands out because it was different from what I expected.",
                    "There was a moment involving {topic} that I still think about. It wasn't dramatic but it changed my perspective.",
                    "I've dealt with {topic} in my own life and it gave me a real understanding of what it means beyond just the abstract idea.",
                    "My personal experience with {topic} taught me that the reality is more nuanced than I originally thought.",
                ],
            },
            # LEGACY: task_summary kept for backward compatibility but improved
            'task_summary': {
                'engaged': [
                    "I thought carefully about {topic} and tried to give honest, thoughtful responses. I reflected on my true opinions and experiences.",
                    "I considered multiple aspects of {topic} before responding. I found myself thinking about how this relates to my own experiences.",
                    "I reflected on {topic} and how it made me feel. I tried to be honest even when I wasn't entirely sure how I felt.",
                    "{topic} is something I've thought about before. I gave thoughtful answers based on my genuine opinions.",
                    "I thought about {topic} carefully and shared my honest impressions. Some aspects resonated with me more than others.",
                    "When it comes to {topic}, I have some experience and tried to respond as accurately as I could.",
                    "I focused on {topic} and gave responses that matched my actual views. I tried not to overthink but also wanted to be accurate.",
                    "I took my time thinking about {topic} and shared my genuine perspective. My past experiences definitely shaped some of my answers.",
                    "{topic} is something I care about. I tried to provide meaningful and honest responses.",
                    "I have real views on {topic} and tried to articulate my genuine reactions.",
                ],
                'satisficer': [
                    "{topic} is fine.",
                    "Shared my views on {topic}.",
                    "{topic}, gave my thoughts.",
                    "Gave my thoughts on {topic}.",
                    "{topic} is what it is.",
                    "Responded about {topic}.",
                    "Shared my opinions on {topic}.",
                    "{topic}. Yeah I have views on that.",
                    "{topic}. Answered honestly.",
                    "My take on {topic}.",
                ],
                'extreme': [
                    "I have very strong feelings about {topic} and I think my answers clearly show that.",
                    "I absolutely had strong opinions about {topic}. I know what I think and I didn't hold back.",
                    "Very clear reactions about {topic}. This definitely struck a chord with me, one way or the other.",
                    "I had extremely strong feelings about {topic} and my answers reflect that. No middle ground for me.",
                    "Strong feelings about {topic}! My responses make my position very clear.",
                    "I felt very strongly about {topic} and didn't hold back in my responses.",
                ],
                'careless': [
                    "{topic} idk",
                    "{topic} whatever",
                    "{topic} is fine",
                    "meh {topic}",
                    "{topic} ok",
                    "sure {topic}",
                    "idk about {topic}",
                    "{topic}",
                    "{topic} i guess",
                    "dont care about {topic}",
                    "yeah {topic}",
                    "{topic} sure",
                ],
                'default': [
                    "I thought about {topic} and tried to give honest responses based on how I actually feel.",
                    "{topic} is something I have opinions on. I shared my genuine reactions.",
                    "I answered about {topic} based on my actual views and experiences.",
                    "I thought about {topic} and gave my honest impressions.",
                    "{topic} — I tried to be accurate about my views.",
                    "I thought about {topic} honestly and gave it reasonable thought.",
                    "{topic} is something I've considered before. I shared my perspective as accurately as I could.",
                    "I reflected on {topic} and shared my genuine thoughts.",
                    "When it comes to {topic}, I shared my honest thoughts.",
                    "I gave my honest views about {topic}.",
                    "{topic} — I gave my real opinions.",
                    "I took {topic} seriously and answered thoughtfully.",
                ]
            },
            'product_evaluation': {
                'positive': [
                    "I feel positively about {product}. The {feature} aspect really stands out to me. Overall a favorable impression.",
                    "I'm impressed with {product}. The {feature} is exactly what I'd hope for. My view is quite positive.",
                    "{product} made a strong positive impression on me. I particularly appreciated the {feature}.",
                    "I had a good impression of {product}. The {feature} stood out and I think it works well.",
                    "{product} feels like a strong option. The {feature} is appealing and it seems well thought out.",
                    "Really positive about {product}. The {feature} is well done and I'd speak favorably about it.",
                    "I'm genuinely interested in {product}. The {feature} addresses something I care about.",
                    "Excellent {product}! The {feature} makes it stand out from similar options I've seen.",
                    "I'm genuinely interested in this {product}. The {feature} addresses something I've been looking for.",
                    "This {product} exceeded my expectations. The {feature} is thoughtfully designed.",
                ],
                'negative': [
                    "Not impressed with {product}. The {feature} doesn't appeal to me at all.",
                    "{product} doesn't meet my expectations. I've seen better alternatives.",
                    "I have issues with {product}. Not convinced about the {feature} or the overall approach.",
                    "Disappointing take on {product}. There are better options available.",
                    "{product} doesn't do much for me. The {feature} isn't convincing.",
                    "I'm not impressed with {product}. The {feature} feels lacking.",
                    "Can't say I'm a fan of {product}. The {feature} doesn't work for me.",
                    "{product} seems underwhelming. The {feature} doesn't stand out.",
                    "I'd pass on {product}. The {feature} isn't enough to change my mind.",
                    "Not for me. {product} and the {feature} don't match what I'm looking for.",
                ],
                'neutral': [
                    "{product} is okay. Nothing special about the {feature} but it's not bad either.",
                    "Average impression of {product}. I could see it going either way.",
                    "{product} is decent. The {feature} is standard.",
                    "It's fine, but the {feature} doesn't make {product} stand out.",
                    "{product} seems acceptable. I don't feel strongly about the {feature} either way.",
                    "Middle of the road on {product}. The {feature} is fine but nothing special.",
                    "Could take it or leave it. {product} and the {feature} are just okay.",
                    "{product} is adequate. The {feature} is unremarkable.",
                    "Neither impressed nor disappointed by {product}. The {feature} is average.",
                    "It's alright. {product} is what I'd expect and the {feature} is fine.",
                ]
            },
            'ai_reaction': {
                'positive': [
                    "I think AI recommendations are helpful. They save time and often find good options that I wouldn't have discovered on my own.",
                    "I trust AI to give relevant suggestions based on my preferences. It's usually pretty accurate.",
                    "AI assistance makes shopping easier and more personalized. I appreciate when technology helps me find what I need.",
                    "AI recommendations feel efficient and generally point me to good choices. I'm comfortable with this kind of assistance.",
                    "I like that AI can analyze patterns to suggest things I might like. It's saved me time many times.",
                    "AI recommendations have introduced me to things I ended up loving. I think the technology is useful.",
                    "I'm generally positive about AI assistance. When it works well, it really enhances the experience.",
                    "AI suggestions often align with my tastes. I think the technology has gotten quite good.",
                ],
                'negative': [
                    "I don't really trust AI recommendations. I prefer to decide on my own without algorithmic influence.",
                    "AI suggestions feel impersonal. I'd rather get advice from real people who understand context.",
                    "I'm concerned about how AI uses my data to make recommendations. Privacy is important to me.",
                    "AI guidance feels hit-or-miss. I've had enough bad recommendations that I don't rely on it.",
                    "I'm skeptical of AI recommendations. They often miss what I'm actually looking for.",
                    "AI assistance feels like it's pushing me toward certain choices. I prefer to explore on my own.",
                    "I don't like the idea of algorithms deciding what I should see or buy. It feels manipulative.",
                    "AI recommendations rarely match my actual preferences. Human judgment is more nuanced.",
                ],
                'neutral': [
                    "AI recommendations can be useful sometimes, but I still like to research on my own.",
                    "I take AI suggestions as one input among many. They're not always right.",
                    "AI helps but isn't always accurate for my specific needs. I use it as a starting point.",
                    "AI is fine as a starting point, but I still double-check on my own before deciding.",
                    "Mixed feelings about AI recommendations. Sometimes helpful, sometimes not relevant.",
                    "I use AI suggestions selectively. They work better for some things than others.",
                    "AI assistance is okay but I wouldn't rely on it completely. Human judgment still matters.",
                    "I'm neutral on AI recommendations. They have their place but aren't a replacement for my own research.",
                ]
            },
            'experience_description': {
                'hedonic': [
                    "It was an enjoyable experience. I felt {emotion} while interacting with the {product}. Overall quite pleasant.",
                    "Using this {product} was fun and pleasurable. I found myself genuinely enjoying the experience.",
                    "I really enjoyed the experience. It was entertaining and satisfying in a way I didn't expect.",
                    "The experience felt enjoyable and engaging. The {product} made the interaction pleasant.",
                    "I had a good time with this {product}. It was more enjoyable than I anticipated.",
                    "The experience was genuinely pleasurable. I felt {emotion} throughout.",
                    "Really enjoyed this. The {product} made it a fun experience.",
                    "The interaction was delightful. I felt positive emotions while using the {product}.",
                ],
                'utilitarian': [
                    "The {product} functioned as expected and helped me accomplish my goal efficiently.",
                    "It was practical and got the job done. That's what matters most to me.",
                    "The {product} served its purpose well. Good functionality and easy to use.",
                    "The {product} was straightforward and effective for what I needed.",
                    "Functional and efficient. The {product} did exactly what it was supposed to do.",
                    "I appreciate that the {product} helped me complete my task without complications.",
                    "The practical aspects of the {product} worked well. It was useful.",
                    "Good functional experience. The {product} met my utilitarian needs.",
                ]
            },
            'general_feedback': {
                'engaged': [
                    "I shared my genuine opinions based on my actual experiences and views.",
                    "I appreciated the chance to share my thoughts. My responses reflect how I really feel.",
                    "I tried to answer thoughtfully and accurately. This is something I care about.",
                    "I engaged seriously and gave honest answers based on my real perspective.",
                    "I provided careful responses based on my actual opinions and experiences.",
                ],
                'brief': [
                    "Shared my thoughts.",
                    "Gave my honest take.",
                    "Done, answered honestly.",
                    "Answered based on my views.",
                    "Finished, shared my opinions.",
                ],
                'critical': [
                    "Some parts were unclear but I answered based on my best interpretation.",
                    "I wasn't sure about everything but gave my honest opinion.",
                    "It was fine overall though some aspects were hard to pin down.",
                    "Had to think hard about some parts but answered honestly.",
                ]
            },
            'reasoning': {
                'engaged': [
                    "My reasoning was based on {topic}. I considered different angles and answered accordingly.",
                    "I thought about my past experiences related to {topic} when answering. Those shaped my responses.",
                    "My answers reflect my values and priorities when it comes to {topic}. I tried to be consistent.",
                    "I based my responses on what I know about {topic} and my personal views.",
                    "I considered multiple factors related to {topic}. My responses reflect that balanced view.",
                ],
                'intuitive': [
                    "I went with my gut feeling on most answers. My initial reactions usually reflect my true opinions.",
                    "Answered based on instinct mostly. Didn't overthink it.",
                    "My responses are based on my immediate reaction to {topic}.",
                    "I trusted my first impressions when answering these questions.",
                ],
            }
        }

    def _add_natural_variation(self, response: str, traits: Dict[str, float], rng: random.Random) -> str:
        """Add natural language variations based on persona traits."""
        attention = traits.get('attention_level', 0.5)

        # Sometimes add a hedge
        if traits.get('response_consistency', 0.5) < 0.6 and rng.random() < 0.25 and len(response) >= 2:
            hedge = rng.choice(self.hedges)
            response = f"{hedge}, {response[0].lower()}{response[1:]}"

        # Typos for low attention
        if attention < 0.5 and rng.random() < 0.2:
            response = response.lower()

        # Self-correction for highly attentive personas (attention > 0.85)
        if attention > 0.85 and rng.random() < 0.15:
            self_corrections = [
                "Well, actually, ",
                "Let me rephrase that — ",
                "Or rather, I should say, ",
                "Actually, on second thought, ",
                "What I really mean is, ",
            ]
            correction = rng.choice(self_corrections)
            sentences = response.split('. ')
            if len(sentences) > 1:
                # Insert self-correction before a random non-first sentence
                insert_pos = rng.randint(1, len(sentences) - 1)
                sent = sentences[insert_pos]
                if len(sent) >= 2:
                    sentences[insert_pos] = correction + sent[0].lower() + sent[1:]
                    response = '. '.join(sentences)
            elif len(response) >= 2:
                response = correction + response[0].lower() + response[1:]

        # Trailing off for satisficers (attention < 0.4)
        if attention < 0.4 and rng.random() < 0.20:
            # Strip existing trailing punctuation before adding ellipsis
            response = response.rstrip('.!?') + "..."

        # Parenthetical asides for engaged personas (attention > 0.7)
        if attention > 0.7 and rng.random() < 0.15:
            parentheticals = [
                "(which I found interesting)",
                "(at least from my experience)",
                "(if that makes sense)",
                "(which surprised me a bit)",
                "(worth noting, I think)",
                "(not that it changes my overall view)",
            ]
            aside = rng.choice(parentheticals)
            sentences = response.split('. ')
            if len(sentences) > 1:
                # Insert the parenthetical after the first sentence
                sentences[0] = sentences[0] + " " + aside
                response = '. '.join(sentences)

        # Add filler phrases for engaged responders
        fillers = [
            " To be honest,", " I have to say,", " Looking back,", " On reflection,",
        ]
        if attention > 0.8 and rng.random() < 0.15:
            filler = rng.choice(fillers)
            sentences = response.split('. ')
            if len(sentences) > 1:
                insert_pos = rng.randint(1, len(sentences) - 1)
                if len(sentences[insert_pos]) >= 2:
                    sentences[insert_pos] = filler + sentences[insert_pos][0].lower() + sentences[insert_pos][1:]
                response = '. '.join(sentences)

        return response

    def _combine_templates(self, templates: List[str], context: Dict[str, str], traits: Dict[str, float], rng: random.Random) -> str:
        """Combine multiple templates for more varied responses."""
        # Select primary template
        primary = rng.choice(templates)

        try:
            response = primary.format(**context)
        except KeyError:
            response = primary
            for key, value in context.items():
                response = response.replace(f"{{{key}}}", str(value))

        # For highly engaged personas, sometimes combine two templates into a
        # multi-sentence response joined by a transition phrase
        if traits.get('attention_level', 0.5) > 0.8 and rng.random() < 0.30 and len(templates) > 1:
            # Pick a second template different from the primary
            remaining = [t for t in templates if t != primary]
            if remaining:
                secondary = rng.choice(remaining)
                try:
                    second_response = secondary.format(**context)
                except KeyError:
                    second_response = secondary
                    for key, value in context.items():
                        second_response = second_response.replace(f"{{{key}}}", str(value))

                transition_phrases = [
                    " Additionally, ",
                    " On top of that, ",
                    " I'd also add that ",
                    " Beyond that, ",
                    " Another thing worth mentioning is that ",
                    " Furthermore, ",
                    " I should also note that ",
                    " Along those lines, ",
                ]
                transition = rng.choice(transition_phrases)
                # Join the two responses: primary + transition + second (lowercased start)
                if len(second_response) >= 2:
                    combined_second = second_response[0].lower() + second_response[1:]
                    # Ensure primary ends with proper punctuation before joining
                    if not response.rstrip().endswith(('.', '!', '?')):
                        response = response.rstrip() + '.'
                    response = response.rstrip() + transition + combined_second

        return response

    def _make_response_unique(self, response: str, rng: random.Random, max_attempts: int = 10, context: Optional[Dict[str, str]] = None) -> str:
        """Ensure response is unique by adding variation if needed.

        Args:
            response: The generated response text.
            rng: Seeded random number generator for reproducibility.
            max_attempts: Maximum number of variation attempts before fallback.
            context: Optional context dict (with keys like 'topic', 'stimulus', etc.)
                     used by topic-relevant detail strategy.
        """
        original_response = response
        attempt = 0

        while response in self._used_responses and attempt < max_attempts:
            attempt += 1
            # Try different variation strategies (6 total, cycling through)
            strategy = attempt % 6

            if strategy == 0:
                # Add a time phrase at the beginning
                time_phrase = rng.choice(self._variation_phrases['time_phrases'])
                if len(original_response) >= 2:
                    response = f"{time_phrase} {original_response[0].lower()}{original_response[1:]}"
                else:
                    response = f"{time_phrase} {original_response}"
            elif strategy == 1:
                # Add a personal phrase at the beginning
                personal_phrase = rng.choice(self._variation_phrases['personal_phrases'])
                if len(original_response) >= 2:
                    response = f"{personal_phrase} {original_response[0].lower()}{original_response[1:]}"
                else:
                    response = f"{personal_phrase} {original_response}"
            elif strategy == 2:
                # Add an ending phrase
                ending = rng.choice([e for e in self._variation_phrases['ending_phrases'] if e])
                response = f"{original_response}{ending}"
            elif strategy == 3:
                # Combine certainty phrase with slight rewording
                certainty = rng.choice(self._variation_phrases['certainty_phrases'])
                # Find a natural break point
                if ". " in original_response:
                    parts = original_response.split(". ", 1)
                    response = f"{certainty} {parts[0].lower()}. {parts[1]}"
                elif len(original_response) >= 2:
                    response = f"{certainty} {original_response[0].lower()}{original_response[1:]}"
                else:
                    response = f"{certainty} {original_response}"
            elif strategy == 4:
                # Paraphrase by restructuring: swap order of first two sentences
                sentences = original_response.split('. ')
                if len(sentences) >= 2:
                    # Swap the first two sentences
                    sentences[0], sentences[1] = sentences[1], sentences[0]
                    response = '. '.join(sentences)
                else:
                    # Single sentence -- fall back to adding a time phrase
                    time_phrase = rng.choice(self._variation_phrases['time_phrases'])
                    if len(original_response) >= 2:
                        response = f"{time_phrase} {original_response[0].lower()}{original_response[1:]}"
                    else:
                        response = f"{time_phrase} {original_response}"
            else:
                # Strategy 5: Add a topic-relevant detail insertion
                topic = (context or {}).get('topic', '')
                if topic:
                    # Extract the first sentence of the original response
                    first_sentence = original_response.split('. ')[0]
                    if not first_sentence.endswith('.'):
                        first_sentence += '.'
                    topic_prefix = f"When it comes to {topic} specifically, "
                    # Build: topic prefix + lowercased first sentence + rest
                    lowered_first = first_sentence[0].lower() + first_sentence[1:] if len(first_sentence) >= 2 else first_sentence
                    rest_parts = original_response.split('. ', 1)
                    if len(rest_parts) > 1:
                        response = f"{topic_prefix}{lowered_first} {rest_parts[1]}"
                    else:
                        response = f"{topic_prefix}{lowered_first}"
                else:
                    # No topic available -- fall back to a personal phrase
                    personal_phrase = rng.choice(self._variation_phrases['personal_phrases'])
                    if len(original_response) >= 2:
                        response = f"{personal_phrase} {original_response[0].lower()}{original_response[1:]}"
                    else:
                        response = f"{personal_phrase} {original_response}"

        # If still not unique after max attempts, add a unique identifier phrase
        if response in self._used_responses:
            unique_modifiers = [
                "To add my perspective,", "In my case,", "What I noticed was that",
                "From what I observed,", "Speaking honestly,", "Being straightforward,",
                "Candidly speaking,", "Truth be told,", "Honestly speaking,", "Frankly,"
            ]
            modifier = rng.choice(unique_modifiers)
            if len(original_response) >= 2:
                response = f"{modifier} {original_response[0].lower()}{original_response[1:]}"
            else:
                response = f"{modifier} {original_response}"

        self._used_responses.add(response)
        return response

    def generate_response(
        self,
        response_type: str,
        persona_style: str,
        context: Dict[str, str],
        traits: Dict[str, float],
        seed: int
    ) -> str:
        """
        Generate a text response based on persona and context.

        Args:
            response_type: Type of response (task_summary, product_evaluation, etc.)
            persona_style: Persona style key (engaged, satisficer, etc.)
            context: Dict with placeholders like {stimulus}, {topic}, {product}
                     v1.0.5.0: Also contains behavioral_summary, response_pattern,
                     intensity, consistency_score, trait_* dimensions, entities,
                     question_intent, condition_framing, etc.
            traits: Participant trait values
            seed: Random seed for reproducibility

        Returns:
            Generated text response with natural variation, guaranteed unique within dataset
        """
        rng = random.Random(seed)

        # v1.0.5.0: Enhanced behavioral-profile-driven style selection
        # Uses full trait vector and behavioral pattern for precise style matching
        _straight_lined = context.get("straight_lined") == "true"
        _beh_pattern = context.get("response_pattern", "")
        try:
            _intensity = float(context.get("intensity", "0.5"))
        except (ValueError, TypeError):
            _intensity = 0.5
        try:
            _consistency = float(context.get("consistency_score", "0.5"))
        except (ValueError, TypeError):
            _consistency = 0.5
        try:
            _sd = float(context.get("trait_social_desirability", "0.3"))
        except (ValueError, TypeError):
            _sd = 0.3
        try:
            _extremity = float(context.get("trait_extremity", "0.4"))
        except (ValueError, TypeError):
            _extremity = 0.4

        if _straight_lined:
            persona_style = "careless"
        elif _beh_pattern in ("strongly_positive", "strongly_negative") and _intensity > 0.6:
            persona_style = "extreme" if _extremity > 0.6 else "engaged"
        elif _beh_pattern in ("strongly_positive", "strongly_negative"):
            if persona_style == "default":
                persona_style = "engaged"
        elif _beh_pattern == "consistently_neutral" and _consistency > 0.8:
            if persona_style not in ("careless", "satisficer"):
                persona_style = "satisficer"  # Consistent neutrality = low differentiation

        # v1.0.7.9: Smart template bank selection based on question intent.
        # Instead of always falling back to task_summary (meta-commentary about the task),
        # use question_intent from context to select content-responsive template banks.
        _question_intent = context.get("question_intent", "")
        _intent_to_bank = {
            "opinion": "opinion_response",
            "explanation": "explanation_response",
            "causal_explanation": "explanation_response",
            "decision_explanation": "explanation_response",
            "emotional_reaction": "feeling_response",
            "description": "description_response",
            "evaluation": "evaluation_response",
            "prediction": "opinion_response",  # predictions use opinion-like templates
            # v1.0.8.3: New content-generative intents
            "creative_belief": "creative_belief_response",
            "personal_disclosure": "personal_disclosure_response",
            "creative_narrative": "creative_narrative_response",
            "personal_story": "personal_story_response",
            "recall": "personal_story_response",  # recall maps to personal story
            # v1.0.8.4: Dedicated routing for recommendation and hypothetical
            "recommendation": "opinion_response",  # recommendations use opinion bank
            "hypothetical": "opinion_response",  # hypotheticals use opinion bank
            # v1.0.8.5: Additional intent routing
            "comparison": "evaluation_response",  # comparisons use evaluation templates
        }
        # Try: explicit response_type → intent-based bank → task_summary fallback
        _effective_type = response_type
        if response_type not in self.templates or response_type == 'task_summary':
            # If we have a question_intent, use the mapped content-responsive bank
            if _question_intent and _question_intent in _intent_to_bank:
                _mapped = _intent_to_bank[_question_intent]
                if _mapped in self.templates:
                    _effective_type = _mapped
            elif not _question_intent:
                # No intent info — try to detect from context/topic
                _topic_lower = context.get("topic", "").lower()
                _stim_lower = context.get("stimulus", "").lower()
                _combined = _topic_lower + " " + _stim_lower
                if any(w in _combined for w in ('why', 'reason', 'explain', 'because')):
                    _effective_type = "explanation_response"
                elif any(w in _combined for w in ('feel', 'emotion', 'feelings')):
                    _effective_type = "feeling_response"
                elif any(w in _combined for w in ('describe', 'experience', 'happened')):
                    _effective_type = "description_response"
                elif any(w in _combined for w in ('rate', 'evaluate', 'assess')):
                    _effective_type = "evaluation_response"
                else:
                    _effective_type = "opinion_response"  # default to opinion, not meta-commentary
        type_templates = self.templates.get(_effective_type, self.templates.get('opinion_response', self.templates['task_summary']))

        # Get style-specific templates or fall back to sentiment or default
        style_templates = type_templates.get(persona_style, [])
        if not style_templates and "sentiment" in context:
            style_templates = type_templates.get(context["sentiment"], [])
        if not style_templates:
            style_templates = type_templates.get('default', [])

        if not style_templates:
            style_templates = ["Response about {topic}."]

        # Shuffle templates to increase variety across participants
        shuffled_templates = style_templates.copy()
        rng.shuffle(shuffled_templates)

        # Generate main response
        response = self._combine_templates(shuffled_templates, context, traits, rng)

        # Add natural variation
        response = self._add_natural_variation(response, traits, rng)

        # v1.0.5.0: Comprehensive behavioral coherence for fallback generator
        # Uses full trait profile + behavioral pattern for rich tone matching
        _topic = context.get("topic", "this")
        if _beh_pattern in ("strongly_positive", "moderately_positive"):
            _resp_lower = response.lower()
            _neg_words = ['bad', 'terrible', 'hate', 'upset', 'awful', 'frustrated',
                          'disappointed', 'angry', 'worst', 'horrible']
            _pos_words = ['good', 'like', 'enjoy', 'positive', 'great', 'appreciate',
                          'happy', 'love', 'glad', 'pleased']
            if any(w in _resp_lower for w in _neg_words):
                if not any(w in _resp_lower for w in _pos_words):
                    response = f"Overall I feel positively about {_topic}. {response}"
        elif _beh_pattern in ("strongly_negative", "moderately_negative"):
            _resp_lower = response.lower()
            _pos_words = ['good', 'great', 'love', 'enjoy', 'wonderful', 'amazing',
                          'happy', 'pleased', 'appreciate']
            _neg_words = ['bad', 'dislike', 'concern', 'negative', 'frustrated',
                          'disappointed', 'upset', 'worried']
            if any(w in _resp_lower for w in _pos_words):
                if not any(w in _resp_lower for w in _neg_words):
                    response = f"I have concerns about {_topic}. {response}"

        # v1.0.5.0: Intensity-driven response modulation
        # Extreme raters (high intensity) get more emphatic language
        if _intensity > 0.7 and _extremity > 0.5:
            if _beh_pattern in ("strongly_positive",) and rng.random() < 0.5:
                _emphatics = [
                    f" I feel really strongly about {_topic}.",
                    f" {_topic} is genuinely important to me.",
                    f" I can't emphasize enough how I feel about {_topic}.",
                ]
                response = response.rstrip()
                if not response.endswith('.'):
                    response += '.'
                response += rng.choice(_emphatics)
            elif _beh_pattern in ("strongly_negative",) and rng.random() < 0.5:
                _emphatics = [
                    f" I feel really strongly about {_topic} and not in a good way.",
                    f" {_topic} seriously concerns me.",
                    f" I can't overstate my frustration with {_topic}.",
                ]
                response = response.rstrip()
                if not response.endswith('.'):
                    response += '.'
                response += rng.choice(_emphatics)

        # v1.0.8.0: Social desirability modulation — EXPANDED hedges
        if _sd > 0.7 and rng.random() < 0.4:
            _sd_hedges = [
                "I tried to be fair in my responses. ",
                "I want to give a balanced perspective. ",
                "I considered different viewpoints. ",
                "I tried to be honest without being harsh. ",
                "I wanted to give a thoughtful response. ",
                "I tried to be respectful in how I answered. ",
            ]
            response = rng.choice(_sd_hedges) + response

        # v1.0.5.0: Straight-liner truncation — behavioral coherence with numeric pattern
        if _straight_lined:
            words = response.split()
            if len(words) > 8:
                response = ' '.join(words[:6])

        # v1.0.5.0: Entity injection — if named entities were detected but aren't
        # mentioned in the response, weave them in naturally
        _entities = context.get("entities")
        if _entities and isinstance(_entities, list) and len(_entities) > 0:
            _resp_lower = response.lower()
            _missing_entities = [e for e in _entities if e.lower() not in _resp_lower]
            if _missing_entities and rng.random() < 0.6:
                _ent = rng.choice(_missing_entities)
                if len(response) > 20:
                    response = response.rstrip('.')
                    response += f", especially regarding {_ent}."

        # v1.0.8.0: Domain-neutral follow-up thoughts — EXPANDED
        followups = {
            'positive': [
                " Overall I came away with a positive impression.",
                " I'd be interested in learning more about this.",
                " This left me feeling good about it.",
                " I think this is heading in the right direction.",
                " I'm genuinely supportive of where this is going.",
                " This reinforced my positive outlook.",
                " I'd be happy to see more of this kind of thing.",
                " My positive feelings about this are pretty strong.",
            ],
            'negative': [
                " Overall I wasn't that impressed.",
                " It didn't really resonate with me.",
                " I was hoping for something better honestly.",
                " I think there are real problems here.",
                " This is concerning to me.",
                " I wish things were handled differently.",
                " This confirmed some of my concerns.",
                " I don't see this improving without real changes.",
            ],
            'neutral': [
                " I don't have particularly strong feelings about it.",
                " Could go either way on this.",
                " My feelings are mixed.",
                " I can see both sides of this.",
                " I'm still forming my opinion.",
                " It's hard to commit to a strong position either way.",
                " I'd need more information to feel strongly.",
                " There are arguments on both sides that make sense to me.",
            ]
        }

        sentiment = context.get("sentiment", "neutral")
        if traits.get('attention_level', 0.5) > 0.75 and rng.random() < 0.35:
            if sentiment in followups:
                response += rng.choice(followups[sentiment])

        # Add reasoning for some engaged respondents
        if persona_style == 'engaged' and rng.random() < 0.25 and 'reasoning' in self.templates:
            reasoning_templates = self.templates['reasoning'].get('engaged', [])
            if reasoning_templates:
                try:
                    reasoning = rng.choice(reasoning_templates).format(**context)
                    response += f" {reasoning}"
                except KeyError:
                    pass

        # Ensure response is unique within this dataset
        response = self._make_response_unique(response.strip(), rng, context=context)

        return response


# ================================================================
# IMAGE/STIMULUS EVALUATION HANDLER
# ================================================================

class StimulusEvaluationHandler:
    """
    Handles generation of realistic responses for image/stimulus evaluations.

    For studies where participants evaluate visual stimuli, products, ads, etc.
    """

    def __init__(self):
        self.evaluation_dimensions = {
            'aesthetic': ['attractive', 'beautiful', 'ugly', 'appealing', 'unappealing'],
            'quality': ['high quality', 'low quality', 'professional', 'amateur'],
            'emotional': ['happy', 'sad', 'excited', 'bored', 'anxious'],
            'trustworthiness': ['trustworthy', 'suspicious', 'credible', 'fake'],
            'relevance': ['relevant', 'irrelevant', 'useful', 'useless']
        }

    def generate_evaluation_response(
        self,
        scale_min: int,
        scale_max: int,
        dimension: str,
        traits: Dict[str, float],
        condition_effect: float,
        seed: int
    ) -> int:
        """
        Generate a numeric evaluation response for a stimulus.

        Args:
            scale_min: Minimum scale value
            scale_max: Maximum scale value
            dimension: Evaluation dimension (aesthetic, quality, etc.)
            traits: Participant trait values
            condition_effect: Effect of experimental condition (-1 to 1)
            seed: Random seed

        Returns:
            Integer scale response
        """
        rng = np.random.RandomState(seed)

        # Base response from traits
        base_tendency = traits.get('response_tendency', 0.5)
        variance = traits.get('variance_tendency', 1.0)

        # Adjust for condition effect
        adjusted_tendency = base_tendency + (condition_effect * 0.15)
        adjusted_tendency = np.clip(adjusted_tendency, 0.1, 0.9)

        # Generate response
        scale_range = scale_max - scale_min
        center = scale_min + (adjusted_tendency * scale_range)
        sd = (scale_range / 4) * variance

        response = rng.normal(center, sd)

        # Extreme responder adjustment
        if traits.get('extreme_tendency', 0) > 0.5 and rng.random() < traits['extreme_tendency']:
            if response > (scale_min + scale_max) / 2:
                response = scale_max - rng.uniform(0, 1)
            else:
                response = scale_min + rng.uniform(0, 1)

        return int(np.clip(round(response), scale_min, scale_max))


# Export main classes
__all__ = [
    'PersonaLibrary',
    'Persona',
    'PersonaTrait',
    'TextResponseGenerator',
    'StimulusEvaluationHandler'
]
