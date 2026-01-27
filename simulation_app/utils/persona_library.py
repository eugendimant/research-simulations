"""
Comprehensive Persona Library for Behavioral Science Simulations
================================================================
A theory-grounded library of behavioral personas for generating realistic
synthetic data across marketing, management, consumer behavior, AI,
behavioral economics, and psychology research domains.

Based on established literature:
- Manning & Horton (2025) - LLM simulation personas
- Krosnick (1991) - Satisficing in survey responses
- Greenleaf (1992) - Extreme response styles
- Paulhus (1991) - Socially desirable responding
- John & Robins (1994) - Individual differences in self-report
"""

import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import hashlib
import re


@dataclass
class PersonaTrait:
    """Represents a single trait dimension with mean and variance."""
    name: str
    base_mean: float  # 0-1 scale
    base_sd: float    # Variance around mean
    description: str


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

    Domains covered:
    - Consumer Behavior & Marketing
    - AI & Technology Attitudes
    - Behavioral Economics & Decision Making
    - Organizational Behavior & Management
    - Social Psychology
    - Health Psychology
    - Environmental Psychology
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

        personas['engaged_responder'] = Persona(
            name="Engaged Responder",
            category="response_style",
            description="Attentive participant who reads carefully and provides thoughtful responses. "
                       "Shows natural variance reflecting genuine opinion differences across items.",
            weight=0.30,
            traits={
                'attention_level': PersonaTrait('attention_level', 0.95, 0.03, 'Very high attention'),
                'response_consistency': PersonaTrait('response_consistency', 0.85, 0.08, 'Consistent but not rigid'),
                'scale_use_breadth': PersonaTrait('scale_use_breadth', 0.70, 0.12, 'Uses full scale range'),
                'acquiescence': PersonaTrait('acquiescence', 0.50, 0.10, 'Balanced agreement tendency'),
                'social_desirability': PersonaTrait('social_desirability', 0.45, 0.12, 'Low SD bias'),
                'reading_speed': PersonaTrait('reading_speed', 0.60, 0.15, 'Moderate pace'),
            },
            text_style={
                'verbosity': 'moderate',
                'detail_level': 'specific',
                'coherence': 'high',
                'sentiment_alignment': 'consistent'
            },
            applicable_domains=['all']
        )

        personas['satisficer'] = Persona(
            name="Satisficer",
            category="response_style",
            description="Participant who puts in minimal cognitive effort. Uses scale midpoints, "
                       "straight-lines on matrices, and provides short text responses.",
            weight=0.20,
            traits={
                'attention_level': PersonaTrait('attention_level', 0.70, 0.10, 'Moderate attention'),
                'response_consistency': PersonaTrait('response_consistency', 0.60, 0.15, 'More random'),
                'scale_use_breadth': PersonaTrait('scale_use_breadth', 0.35, 0.10, 'Prefers midpoints'),
                'acquiescence': PersonaTrait('acquiescence', 0.55, 0.08, 'Slight yes-bias'),
                'social_desirability': PersonaTrait('social_desirability', 0.50, 0.10, 'Neutral'),
                'reading_speed': PersonaTrait('reading_speed', 0.85, 0.10, 'Fast/rushing'),
            },
            text_style={
                'verbosity': 'minimal',
                'detail_level': 'vague',
                'coherence': 'low',
                'sentiment_alignment': 'neutral'
            },
            applicable_domains=['all']
        )

        personas['extreme_responder'] = Persona(
            name="Extreme Responder",
            category="response_style",
            description="Participant who consistently uses scale endpoints (1s and 7s). "
                       "Shows strong opinions but may lack nuance.",
            weight=0.08,
            traits={
                'attention_level': PersonaTrait('attention_level', 0.82, 0.08, 'Good attention'),
                'response_consistency': PersonaTrait('response_consistency', 0.75, 0.10, 'Fairly consistent'),
                'scale_use_breadth': PersonaTrait('scale_use_breadth', 0.90, 0.05, 'Extreme endpoints'),
                'acquiescence': PersonaTrait('acquiescence', 0.50, 0.20, 'Variable'),
                'social_desirability': PersonaTrait('social_desirability', 0.40, 0.15, 'Lower concern'),
                'reading_speed': PersonaTrait('reading_speed', 0.70, 0.12, 'Moderate'),
            },
            text_style={
                'verbosity': 'moderate',
                'detail_level': 'emphatic',
                'coherence': 'moderate',
                'sentiment_alignment': 'strong'
            },
            applicable_domains=['all']
        )

        personas['acquiescent_responder'] = Persona(
            name="Acquiescent Responder",
            category="response_style",
            description="Participant who tends to agree with statements regardless of content. "
                       "Shows yes-saying bias that may inflate positive item scores.",
            weight=0.07,
            traits={
                'attention_level': PersonaTrait('attention_level', 0.75, 0.10, 'Moderate attention'),
                'response_consistency': PersonaTrait('response_consistency', 0.55, 0.12, 'Less consistent'),
                'scale_use_breadth': PersonaTrait('scale_use_breadth', 0.55, 0.10, 'Upper half bias'),
                'acquiescence': PersonaTrait('acquiescence', 0.75, 0.08, 'Strong yes-bias'),
                'social_desirability': PersonaTrait('social_desirability', 0.60, 0.10, 'Some SD bias'),
                'reading_speed': PersonaTrait('reading_speed', 0.75, 0.10, 'Faster'),
            },
            text_style={
                'verbosity': 'moderate',
                'detail_level': 'agreeable',
                'coherence': 'moderate',
                'sentiment_alignment': 'positive'
            },
            applicable_domains=['all']
        )

        personas['careless_responder'] = Persona(
            name="Careless Responder",
            category="response_style",
            description="Participant showing clear inattention - fails attention checks, "
                       "inconsistent reversed items, implausible response patterns.",
            weight=0.05,
            traits={
                'attention_level': PersonaTrait('attention_level', 0.40, 0.15, 'Low attention'),
                'response_consistency': PersonaTrait('response_consistency', 0.30, 0.15, 'Very inconsistent'),
                'scale_use_breadth': PersonaTrait('scale_use_breadth', 0.50, 0.25, 'Random'),
                'acquiescence': PersonaTrait('acquiescence', 0.50, 0.20, 'Random'),
                'social_desirability': PersonaTrait('social_desirability', 0.50, 0.20, 'Unconcerned'),
                'reading_speed': PersonaTrait('reading_speed', 0.95, 0.03, 'Very fast'),
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
        # CONSUMER BEHAVIOR & MARKETING PERSONAS
        # ================================================================

        personas['brand_loyalist'] = Persona(
            name="Brand Loyalist",
            category="consumer",
            description="Consumer with strong brand attachments, high brand trust, "
                       "resistant to switching. Values consistency and familiarity.",
            weight=0.12,
            traits={
                'brand_attachment': PersonaTrait('brand_attachment', 0.82, 0.08, 'High attachment'),
                'price_sensitivity': PersonaTrait('price_sensitivity', 0.35, 0.12, 'Less price sensitive'),
                'novelty_seeking': PersonaTrait('novelty_seeking', 0.30, 0.10, 'Low novelty seeking'),
                'risk_aversion': PersonaTrait('risk_aversion', 0.70, 0.10, 'Risk averse'),
                'wtp_baseline': PersonaTrait('wtp_baseline', 0.65, 0.12, 'Higher WTP'),
                'purchase_involvement': PersonaTrait('purchase_involvement', 0.70, 0.10, 'High involvement'),
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
                       "High price sensitivity, enjoys the thrill of finding bargains.",
            weight=0.15,
            traits={
                'brand_attachment': PersonaTrait('brand_attachment', 0.35, 0.12, 'Low attachment'),
                'price_sensitivity': PersonaTrait('price_sensitivity', 0.85, 0.08, 'Very price sensitive'),
                'novelty_seeking': PersonaTrait('novelty_seeking', 0.55, 0.12, 'Moderate'),
                'risk_aversion': PersonaTrait('risk_aversion', 0.50, 0.12, 'Moderate'),
                'wtp_baseline': PersonaTrait('wtp_baseline', 0.35, 0.10, 'Lower WTP'),
                'purchase_involvement': PersonaTrait('purchase_involvement', 0.75, 0.10, 'High for deals'),
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
                       "immediate gratification. Lower self-control in buying contexts.",
            weight=0.10,
            traits={
                'brand_attachment': PersonaTrait('brand_attachment', 0.45, 0.15, 'Variable'),
                'price_sensitivity': PersonaTrait('price_sensitivity', 0.40, 0.15, 'Lower when excited'),
                'novelty_seeking': PersonaTrait('novelty_seeking', 0.75, 0.10, 'High novelty seeking'),
                'risk_aversion': PersonaTrait('risk_aversion', 0.35, 0.12, 'Lower risk aversion'),
                'wtp_baseline': PersonaTrait('wtp_baseline', 0.60, 0.15, 'Variable, often higher'),
                'purchase_involvement': PersonaTrait('purchase_involvement', 0.45, 0.15, 'Quick decisions'),
                'self_control': PersonaTrait('self_control', 0.35, 0.10, 'Lower self-control'),
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
                       "and corporate responsibility in purchase decisions.",
            weight=0.12,
            traits={
                'ethical_concern': PersonaTrait('ethical_concern', 0.85, 0.08, 'High ethical concern'),
                'price_sensitivity': PersonaTrait('price_sensitivity', 0.45, 0.12, 'Willing to pay premium'),
                'brand_attachment': PersonaTrait('brand_attachment', 0.55, 0.12, 'Values-based loyalty'),
                'information_seeking': PersonaTrait('information_seeking', 0.80, 0.08, 'Researches thoroughly'),
                'wtp_baseline': PersonaTrait('wtp_baseline', 0.65, 0.10, 'Higher for ethical'),
                'purchase_involvement': PersonaTrait('purchase_involvement', 0.80, 0.08, 'Very involved'),
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
                       "experiences from products. Prioritizes hedonic over utilitarian benefits.",
            weight=0.12,
            traits={
                'hedonic_motivation': PersonaTrait('hedonic_motivation', 0.85, 0.08, 'High hedonic drive'),
                'utilitarian_motivation': PersonaTrait('utilitarian_motivation', 0.40, 0.12, 'Lower utilitarian'),
                'emotional_intensity': PersonaTrait('emotional_intensity', 0.75, 0.10, 'Strong emotions'),
                'novelty_seeking': PersonaTrait('novelty_seeking', 0.70, 0.10, 'Seeks new experiences'),
                'wtp_baseline': PersonaTrait('wtp_baseline', 0.60, 0.12, 'Pays for pleasure'),
                'aesthetic_sensitivity': PersonaTrait('aesthetic_sensitivity', 0.75, 0.10, 'Values aesthetics'),
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
                       "and value for money. Focuses on product performance over experience.",
            weight=0.12,
            traits={
                'hedonic_motivation': PersonaTrait('hedonic_motivation', 0.35, 0.10, 'Lower hedonic drive'),
                'utilitarian_motivation': PersonaTrait('utilitarian_motivation', 0.85, 0.08, 'High utilitarian'),
                'emotional_intensity': PersonaTrait('emotional_intensity', 0.40, 0.12, 'More rational'),
                'novelty_seeking': PersonaTrait('novelty_seeking', 0.40, 0.12, 'Prefers proven'),
                'wtp_baseline': PersonaTrait('wtp_baseline', 0.50, 0.10, 'Value-focused'),
                'information_seeking': PersonaTrait('information_seeking', 0.75, 0.10, 'Researches specs'),
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
        # ================================================================

        personas['tech_enthusiast'] = Persona(
            name="Tech Enthusiast",
            category="technology",
            description="Early adopter with positive attitudes toward new technology and AI. "
                       "High tech self-efficacy, sees benefits over risks.",
            weight=0.15,
            traits={
                'tech_affinity': PersonaTrait('tech_affinity', 0.88, 0.07, 'Very high'),
                'ai_attitude': PersonaTrait('ai_attitude', 0.82, 0.08, 'Very positive'),
                'ai_anxiety': PersonaTrait('ai_anxiety', 0.25, 0.10, 'Low anxiety'),
                'perceived_ai_competence': PersonaTrait('perceived_ai_competence', 0.80, 0.08, 'High trust'),
                'human_uniqueness_belief': PersonaTrait('human_uniqueness_belief', 0.40, 0.12, 'Open to AI'),
                'tech_self_efficacy': PersonaTrait('tech_self_efficacy', 0.85, 0.08, 'High confidence'),
                'algorithm_appreciation': PersonaTrait('algorithm_appreciation', 0.80, 0.08, 'Appreciates algorithms'),
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
                       "and loss of human control. Prefers human alternatives.",
            weight=0.15,
            traits={
                'tech_affinity': PersonaTrait('tech_affinity', 0.35, 0.10, 'Low'),
                'ai_attitude': PersonaTrait('ai_attitude', 0.30, 0.10, 'Negative'),
                'ai_anxiety': PersonaTrait('ai_anxiety', 0.75, 0.10, 'High anxiety'),
                'perceived_ai_competence': PersonaTrait('perceived_ai_competence', 0.35, 0.12, 'Low trust'),
                'human_uniqueness_belief': PersonaTrait('human_uniqueness_belief', 0.80, 0.08, 'Strong belief'),
                'tech_self_efficacy': PersonaTrait('tech_self_efficacy', 0.40, 0.12, 'Lower confidence'),
                'algorithm_aversion': PersonaTrait('algorithm_aversion', 0.75, 0.10, 'Prefers humans'),
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
                       "appropriate tasks but values human judgment for important decisions.",
            weight=0.20,
            traits={
                'tech_affinity': PersonaTrait('tech_affinity', 0.60, 0.12, 'Moderate'),
                'ai_attitude': PersonaTrait('ai_attitude', 0.55, 0.12, 'Balanced'),
                'ai_anxiety': PersonaTrait('ai_anxiety', 0.50, 0.12, 'Moderate concern'),
                'perceived_ai_competence': PersonaTrait('perceived_ai_competence', 0.55, 0.12, 'Task-dependent'),
                'human_uniqueness_belief': PersonaTrait('human_uniqueness_belief', 0.60, 0.10, 'Moderate'),
                'tech_self_efficacy': PersonaTrait('tech_self_efficacy', 0.60, 0.12, 'Adequate'),
                'algorithm_appreciation': PersonaTrait('algorithm_appreciation', 0.55, 0.12, 'Context-dependent'),
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
                       "personal information, skeptical of data collection practices.",
            weight=0.12,
            traits={
                'privacy_concern': PersonaTrait('privacy_concern', 0.88, 0.06, 'Very high'),
                'data_sharing_willingness': PersonaTrait('data_sharing_willingness', 0.25, 0.10, 'Very low'),
                'institutional_trust': PersonaTrait('institutional_trust', 0.30, 0.10, 'Low trust'),
                'tech_affinity': PersonaTrait('tech_affinity', 0.50, 0.15, 'Variable'),
                'ai_attitude': PersonaTrait('ai_attitude', 0.40, 0.12, 'Cautious'),
                'personalization_preference': PersonaTrait('personalization_preference', 0.30, 0.12, 'Prefers generic'),
            },
            text_style={
                'verbosity': 'moderate',
                'detail_level': 'privacy_aware',
                'coherence': 'high',
                'sentiment_alignment': 'protective'
            },
            applicable_domains=['technology', 'privacy', 'data', 'personalization', 'surveillance']
        )

        # ================================================================
        # BEHAVIORAL ECONOMICS PERSONAS
        # ================================================================

        personas['loss_averse'] = Persona(
            name="Loss-Averse Decision Maker",
            category="behavioral_economics",
            description="Individual who weighs losses more heavily than equivalent gains. "
                       "Risk-seeking in loss domain, risk-averse in gain domain.",
            weight=0.18,
            traits={
                'loss_aversion': PersonaTrait('loss_aversion', 0.80, 0.10, 'High loss aversion'),
                'risk_tolerance': PersonaTrait('risk_tolerance', 0.35, 0.12, 'Low in gains'),
                'status_quo_bias': PersonaTrait('status_quo_bias', 0.70, 0.10, 'Prefers status quo'),
                'endowment_effect': PersonaTrait('endowment_effect', 0.72, 0.10, 'Strong ownership effect'),
                'regret_anticipation': PersonaTrait('regret_anticipation', 0.75, 0.10, 'High regret concern'),
                'time_preference': PersonaTrait('time_preference', 0.55, 0.12, 'Moderate patience'),
            },
            text_style={
                'verbosity': 'moderate',
                'detail_level': 'risk_focused',
                'coherence': 'high',
                'sentiment_alignment': 'cautious'
            },
            applicable_domains=['behavioral_economics', 'decision_making', 'risk', 'finance']
        )

        personas['present_biased'] = Persona(
            name="Present-Biased Individual",
            category="behavioral_economics",
            description="Person who heavily discounts future outcomes, preferring immediate "
                       "rewards. Struggles with self-control and long-term planning.",
            weight=0.12,
            traits={
                'time_preference': PersonaTrait('time_preference', 0.25, 0.10, 'Very impatient'),
                'self_control': PersonaTrait('self_control', 0.35, 0.12, 'Low self-control'),
                'planning_tendency': PersonaTrait('planning_tendency', 0.35, 0.12, 'Poor planning'),
                'impulsivity': PersonaTrait('impulsivity', 0.75, 0.10, 'High impulsivity'),
                'future_orientation': PersonaTrait('future_orientation', 0.30, 0.12, 'Low'),
                'commitment_device_use': PersonaTrait('commitment_device_use', 0.40, 0.15, 'Variable'),
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
                       "decisions based on expected utility. Less susceptible to biases.",
            weight=0.10,
            traits={
                'need_for_cognition': PersonaTrait('need_for_cognition', 0.85, 0.08, 'High'),
                'information_seeking': PersonaTrait('information_seeking', 0.82, 0.08, 'Thorough'),
                'susceptibility_to_bias': PersonaTrait('susceptibility_to_bias', 0.30, 0.10, 'Lower'),
                'decision_time': PersonaTrait('decision_time', 0.75, 0.10, 'Takes time'),
                'regret_anticipation': PersonaTrait('regret_anticipation', 0.60, 0.12, 'Moderate'),
                'consistency_seeking': PersonaTrait('consistency_seeking', 0.80, 0.08, 'High'),
            },
            text_style={
                'verbosity': 'detailed',
                'detail_level': 'analytical',
                'coherence': 'very_high',
                'sentiment_alignment': 'reasoned'
            },
            applicable_domains=['behavioral_economics', 'decision_making', 'judgment']
        )

        personas['social_comparer'] = Persona(
            name="Social Comparer",
            category="behavioral_economics",
            description="Individual highly influenced by social comparisons and relative standing. "
                       "Strong reactions to inequality and social reference points.",
            weight=0.12,
            traits={
                'social_comparison_orientation': PersonaTrait('social_comparison_orientation', 0.82, 0.08, 'High'),
                'inequality_aversion': PersonaTrait('inequality_aversion', 0.75, 0.10, 'Strong'),
                'status_concern': PersonaTrait('status_concern', 0.78, 0.10, 'High status concern'),
                'envy_proneness': PersonaTrait('envy_proneness', 0.65, 0.12, 'Moderate-high'),
                'fairness_concern': PersonaTrait('fairness_concern', 0.75, 0.10, 'High'),
                'conformity': PersonaTrait('conformity', 0.65, 0.12, 'Moderate-high'),
            },
            text_style={
                'verbosity': 'moderate',
                'detail_level': 'comparison_focused',
                'coherence': 'high',
                'sentiment_alignment': 'relative'
            },
            applicable_domains=['behavioral_economics', 'social_comparison', 'inequality', 'fairness']
        )

        # ================================================================
        # ORGANIZATIONAL BEHAVIOR PERSONAS
        # ================================================================

        personas['high_performer'] = Persona(
            name="High Performer",
            category="organizational",
            description="Highly engaged employee with strong work ethic, high job satisfaction, "
                       "and organizational commitment. Proactive and achievement-oriented.",
            weight=0.15,
            traits={
                'work_engagement': PersonaTrait('work_engagement', 0.85, 0.08, 'Very engaged'),
                'job_satisfaction': PersonaTrait('job_satisfaction', 0.80, 0.10, 'High satisfaction'),
                'organizational_commitment': PersonaTrait('organizational_commitment', 0.78, 0.10, 'Committed'),
                'proactive_personality': PersonaTrait('proactive_personality', 0.82, 0.08, 'Very proactive'),
                'achievement_motivation': PersonaTrait('achievement_motivation', 0.85, 0.08, 'High'),
                'ocb_tendency': PersonaTrait('ocb_tendency', 0.75, 0.10, 'High citizenship'),
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
                       "May be experiencing burnout or job dissatisfaction.",
            weight=0.12,
            traits={
                'work_engagement': PersonaTrait('work_engagement', 0.30, 0.12, 'Low engagement'),
                'job_satisfaction': PersonaTrait('job_satisfaction', 0.35, 0.12, 'Low satisfaction'),
                'organizational_commitment': PersonaTrait('organizational_commitment', 0.30, 0.10, 'Low'),
                'turnover_intention': PersonaTrait('turnover_intention', 0.70, 0.12, 'High'),
                'burnout': PersonaTrait('burnout', 0.70, 0.12, 'High burnout'),
                'ocb_tendency': PersonaTrait('ocb_tendency', 0.30, 0.12, 'Low citizenship'),
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
                       "and individualized consideration. High emotional intelligence.",
            weight=0.08,
            traits={
                'leadership_self_efficacy': PersonaTrait('leadership_self_efficacy', 0.82, 0.08, 'High'),
                'emotional_intelligence': PersonaTrait('emotional_intelligence', 0.80, 0.08, 'High EI'),
                'vision_articulation': PersonaTrait('vision_articulation', 0.78, 0.10, 'Strong vision'),
                'empowerment_orientation': PersonaTrait('empowerment_orientation', 0.80, 0.08, 'Empowering'),
                'ethical_leadership': PersonaTrait('ethical_leadership', 0.78, 0.10, 'High ethics'),
                'change_orientation': PersonaTrait('change_orientation', 0.75, 0.10, 'Change-positive'),
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
        # ================================================================

        personas['prosocial_individual'] = Persona(
            name="Prosocial Individual",
            category="social",
            description="Person with strong prosocial orientation, high empathy, and concern for "
                       "others' welfare. Likely to cooperate and help.",
            weight=0.15,
            traits={
                'empathy': PersonaTrait('empathy', 0.82, 0.08, 'High empathy'),
                'altruism': PersonaTrait('altruism', 0.78, 0.10, 'High altruism'),
                'cooperation_tendency': PersonaTrait('cooperation_tendency', 0.80, 0.08, 'Cooperative'),
                'trust_propensity': PersonaTrait('trust_propensity', 0.70, 0.10, 'Trusting'),
                'social_responsibility': PersonaTrait('social_responsibility', 0.78, 0.10, 'Responsible'),
                'moral_identity': PersonaTrait('moral_identity', 0.75, 0.10, 'Strong moral identity'),
            },
            text_style={
                'verbosity': 'warm',
                'detail_level': 'other_focused',
                'coherence': 'high',
                'sentiment_alignment': 'caring'
            },
            applicable_domains=['social_psychology', 'prosocial_behavior', 'cooperation', 'altruism']
        )

        personas['individualist'] = Persona(
            name="Individualist",
            category="social",
            description="Person focused on personal goals and self-interest. Lower concern for "
                       "collective outcomes, competitive orientation.",
            weight=0.12,
            traits={
                'individualism': PersonaTrait('individualism', 0.82, 0.08, 'High'),
                'competition_orientation': PersonaTrait('competition_orientation', 0.75, 0.10, 'Competitive'),
                'self_interest': PersonaTrait('self_interest', 0.78, 0.10, 'Self-focused'),
                'trust_propensity': PersonaTrait('trust_propensity', 0.45, 0.12, 'Lower trust'),
                'cooperation_tendency': PersonaTrait('cooperation_tendency', 0.35, 0.12, 'Lower'),
                'status_concern': PersonaTrait('status_concern', 0.70, 0.10, 'Status-seeking'),
            },
            text_style={
                'verbosity': 'direct',
                'detail_level': 'self_focused',
                'coherence': 'high',
                'sentiment_alignment': 'achievement_oriented'
            },
            applicable_domains=['social_psychology', 'cooperation', 'competition', 'social_dilemmas']
        )

        personas['conformist'] = Persona(
            name="Conformist",
            category="social",
            description="Person who tends to follow social norms and group opinions. High need for "
                       "belonging, influenced by majority views.",
            weight=0.12,
            traits={
                'conformity': PersonaTrait('conformity', 0.80, 0.08, 'High conformity'),
                'need_for_belonging': PersonaTrait('need_for_belonging', 0.78, 0.10, 'High'),
                'social_influence_susceptibility': PersonaTrait('social_influence_susceptibility', 0.75, 0.10, 'High'),
                'uniqueness_seeking': PersonaTrait('uniqueness_seeking', 0.30, 0.10, 'Low'),
                'opinion_leadership': PersonaTrait('opinion_leadership', 0.35, 0.12, 'Low'),
                'social_anxiety': PersonaTrait('social_anxiety', 0.55, 0.12, 'Moderate'),
            },
            text_style={
                'verbosity': 'moderate',
                'detail_level': 'norm_following',
                'coherence': 'high',
                'sentiment_alignment': 'agreeable'
            },
            applicable_domains=['social_psychology', 'conformity', 'social_influence', 'norms']
        )

        # ================================================================
        # PSYCHOLOGICAL OWNERSHIP PERSONAS (Specific to ownership research)
        # ================================================================

        personas['high_ownership'] = Persona(
            name="High Psychological Ownership",
            category="ownership",
            description="Person who readily develops psychological ownership feelings. Strong "
                       "sense of 'mine', territorial, invests self in objects/ideas.",
            weight=0.15,
            traits={
                'ownership_tendency': PersonaTrait('ownership_tendency', 0.85, 0.08, 'Very high'),
                'territorial_behavior': PersonaTrait('territorial_behavior', 0.78, 0.10, 'High'),
                'self_extension': PersonaTrait('self_extension', 0.80, 0.08, 'Strong'),
                'control_need': PersonaTrait('control_need', 0.75, 0.10, 'High control need'),
                'attachment_style': PersonaTrait('attachment_style', 0.72, 0.10, 'Strong attachment'),
                'endowment_effect': PersonaTrait('endowment_effect', 0.78, 0.10, 'Strong'),
            },
            text_style={
                'verbosity': 'possessive',
                'detail_level': 'ownership_focused',
                'coherence': 'high',
                'sentiment_alignment': 'attached'
            },
            applicable_domains=['psychological_ownership', 'consumer_behavior', 'branding']
        )

        personas['low_ownership'] = Persona(
            name="Low Psychological Ownership",
            category="ownership",
            description="Person who rarely develops ownership feelings. Detached, sharing-oriented, "
                       "less affected by endowment effects.",
            weight=0.10,
            traits={
                'ownership_tendency': PersonaTrait('ownership_tendency', 0.30, 0.10, 'Low'),
                'territorial_behavior': PersonaTrait('territorial_behavior', 0.32, 0.12, 'Low'),
                'self_extension': PersonaTrait('self_extension', 0.35, 0.12, 'Weak'),
                'control_need': PersonaTrait('control_need', 0.40, 0.12, 'Lower control need'),
                'sharing_orientation': PersonaTrait('sharing_orientation', 0.75, 0.10, 'Sharing-oriented'),
                'endowment_effect': PersonaTrait('endowment_effect', 0.35, 0.12, 'Weak'),
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
        # ================================================================

        personas['health_conscious'] = Persona(
            name="Health-Conscious Individual",
            category="health",
            description="Person highly attentive to health behaviors, nutrition, and wellness. "
                       "Proactive about prevention, health-motivated decisions.",
            weight=0.15,
            traits={
                'health_consciousness': PersonaTrait('health_consciousness', 0.85, 0.08, 'Very high'),
                'prevention_focus': PersonaTrait('prevention_focus', 0.80, 0.08, 'Prevention-oriented'),
                'self_efficacy_health': PersonaTrait('self_efficacy_health', 0.78, 0.10, 'High'),
                'health_locus_control': PersonaTrait('health_locus_control', 0.75, 0.10, 'Internal'),
                'risk_perception': PersonaTrait('risk_perception', 0.70, 0.10, 'Moderate-high'),
                'information_seeking_health': PersonaTrait('information_seeking_health', 0.80, 0.08, 'Seeks info'),
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
                       "Lower engagement with health behaviors, external locus of control.",
            weight=0.10,
            traits={
                'health_consciousness': PersonaTrait('health_consciousness', 0.35, 0.12, 'Low'),
                'health_locus_control': PersonaTrait('health_locus_control', 0.30, 0.10, 'External'),
                'self_efficacy_health': PersonaTrait('self_efficacy_health', 0.35, 0.12, 'Low'),
                'prevention_focus': PersonaTrait('prevention_focus', 0.35, 0.12, 'Low'),
                'risk_perception': PersonaTrait('risk_perception', 0.50, 0.15, 'Variable'),
                'optimistic_bias': PersonaTrait('optimistic_bias', 0.65, 0.12, 'Some denial'),
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
        # ================================================================

        personas['eco_warrior'] = Persona(
            name="Environmental Activist",
            category="environmental",
            description="Strongly pro-environmental individual. High environmental concern, "
                       "engages in sustainable behaviors, willing to sacrifice for environment.",
            weight=0.10,
            traits={
                'environmental_concern': PersonaTrait('environmental_concern', 0.90, 0.06, 'Very high'),
                'biospheric_values': PersonaTrait('biospheric_values', 0.88, 0.07, 'Strong'),
                'sustainable_behavior': PersonaTrait('sustainable_behavior', 0.85, 0.08, 'Consistent'),
                'sacrifice_willingness': PersonaTrait('sacrifice_willingness', 0.78, 0.10, 'High'),
                'climate_anxiety': PersonaTrait('climate_anxiety', 0.70, 0.12, 'Moderate-high'),
                'environmental_identity': PersonaTrait('environmental_identity', 0.85, 0.08, 'Core identity'),
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
                       "concern, prioritizes economic considerations.",
            weight=0.08,
            traits={
                'environmental_concern': PersonaTrait('environmental_concern', 0.30, 0.12, 'Low'),
                'climate_skepticism': PersonaTrait('climate_skepticism', 0.75, 0.10, 'Skeptical'),
                'sustainable_behavior': PersonaTrait('sustainable_behavior', 0.35, 0.12, 'Low'),
                'sacrifice_willingness': PersonaTrait('sacrifice_willingness', 0.25, 0.10, 'Low'),
                'economic_priority': PersonaTrait('economic_priority', 0.80, 0.08, 'Economy first'),
                'biospheric_values': PersonaTrait('biospheric_values', 0.30, 0.12, 'Lower'),
            },
            text_style={
                'verbosity': 'direct',
                'detail_level': 'economic_focused',
                'coherence': 'high',
                'sentiment_alignment': 'skeptical'
            },
            applicable_domains=['environmental_psychology', 'sustainability', 'climate', 'skepticism']
        )

        return personas

    def _build_domain_keywords(self) -> Dict[str, List[str]]:
        """Build keyword mappings for automatic domain detection."""
        return {
            'consumer_behavior': [
                'consumer', 'purchase', 'buying', 'shopping', 'product', 'brand',
                'retail', 'e-commerce', 'customer', 'consumption', 'wtp', 'willingness to pay',
                'shopping cart', 'checkout', 'price', 'promotion', 'discount'
            ],
            'marketing': [
                'marketing', 'advertising', 'ad', 'promotion', 'branding', 'brand',
                'campaign', 'persuasion', 'message', 'appeal', 'endorsement',
                'influencer', 'social media marketing'
            ],
            'ai': [
                'ai', 'artificial intelligence', 'algorithm', 'machine learning',
                'automation', 'robot', 'chatbot', 'recommendation system', 'personalization',
                'ai-generated', 'ai recommendation', 'algorithm aversion', 'algorithm appreciation'
            ],
            'technology': [
                'technology', 'tech', 'digital', 'online', 'app', 'software',
                'platform', 'interface', 'ux', 'user experience', 'adoption'
            ],
            'behavioral_economics': [
                'decision', 'choice', 'risk', 'loss', 'gain', 'framing', 'nudge',
                'bias', 'heuristic', 'prospect theory', 'time preference', 'discount',
                'intertemporal', 'self-control', 'commitment', 'default'
            ],
            'organizational_behavior': [
                'employee', 'workplace', 'job', 'organization', 'manager', 'leader',
                'leadership', 'team', 'motivation', 'engagement', 'satisfaction',
                'turnover', 'commitment', 'performance', 'hr', 'human resources'
            ],
            'social_psychology': [
                'social', 'group', 'norm', 'conformity', 'influence', 'persuasion',
                'cooperation', 'competition', 'trust', 'fairness', 'reciprocity',
                'prosocial', 'altruism', 'helping', 'identity'
            ],
            'psychological_ownership': [
                'ownership', 'psychological ownership', 'mine', 'possess', 'territory',
                'endowment', 'attachment', 'self-extension', 'control'
            ],
            'health_psychology': [
                'health', 'wellness', 'nutrition', 'exercise', 'medical', 'disease',
                'prevention', 'treatment', 'behavior change', 'habit', 'addiction'
            ],
            'environmental': [
                'environment', 'sustainability', 'green', 'eco', 'climate', 'carbon',
                'recycling', 'waste', 'pollution', 'conservation', 'renewable'
            ],
            'pricing': [
                'price', 'pricing', 'cost', 'value', 'premium', 'discount',
                'willingness to pay', 'wtp', 'auction', 'bid'
            ],
            'hedonic_consumption': [
                'hedonic', 'pleasure', 'enjoyment', 'fun', 'experience', 'aesthetic',
                'emotional', 'excitement', 'entertainment', 'luxury'
            ],
            'utilitarian_consumption': [
                'utilitarian', 'functional', 'practical', 'efficiency', 'useful',
                'performance', 'quality', 'durability', 'necessity'
            ],
            'privacy': [
                'privacy', 'data', 'personal information', 'surveillance', 'tracking',
                'consent', 'gdpr', 'data protection', 'anonymity'
            ],
            'fairness': [
                'fairness', 'justice', 'equality', 'equity', 'distribution',
                'procedural', 'distributive', 'inequality', 'bias'
            ]
        }

    def detect_domains(self, study_description: str, study_title: str = "") -> List[str]:
        """
        Automatically detect relevant research domains from study description.

        Args:
            study_description: Text describing the research study
            study_title: Optional title of the study

        Returns:
            List of detected domain names, sorted by relevance
        """
        combined_text = f"{study_title} {study_description}".lower()

        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in combined_text:
                    # Weight by keyword length (longer = more specific)
                    score += len(keyword.split())
            if score > 0:
                domain_scores[domain] = score

        # Sort by score and return
        sorted_domains = sorted(domain_scores.keys(), key=lambda x: domain_scores[x], reverse=True)
        return sorted_domains if sorted_domains else ['consumer_behavior']  # Default

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
        # Create unique seed for this participant
        unique_seed = hash(f"{study_seed}_{participant_id}_{persona.name}") % (2**32)
        rng = np.random.RandomState(unique_seed)

        traits = {}
        for trait_name, trait in persona.traits.items():
            # Generate value from normal distribution centered on persona mean
            value = rng.normal(trait.base_mean, trait.base_sd)
            # Clip to valid range
            traits[trait_name] = float(np.clip(value, 0.01, 0.99))

        return traits


# ================================================================
# TEXT GENERATION TEMPLATES
# ================================================================

class TextResponseGenerator:
    """
    Generates realistic open-ended text responses based on persona and study context.

    Uses template-based generation with persona-specific variation to create
    plausible qualitative data without requiring an LLM.
    """

    def __init__(self):
        self.templates = self._build_templates()

    def _build_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Build template library for different response types and personas."""
        return {
            'task_summary': {
                'engaged': [
                    "I carefully reviewed the {stimulus} and thought about my honest reactions to it.",
                    "The study asked me to evaluate {topic} and I considered multiple aspects before responding.",
                    "I looked at the {stimulus} presented and reflected on how it made me feel and what I thought.",
                    "This survey was about {topic}. I tried to give thoughtful answers based on my genuine opinions.",
                    "I examined the {stimulus} closely and answered questions about my impressions and preferences.",
                ],
                'satisficer': [
                    "Looked at {topic} and answered questions.",
                    "Survey about {stimulus}.",
                    "Rated some things about {topic}.",
                    "Answered questions.",
                    "{topic} evaluation.",
                ],
                'extreme': [
                    "This was a really interesting study about {topic}! I had strong feelings about the {stimulus}.",
                    "I absolutely loved/hated the {stimulus}. Very clear opinions on {topic}.",
                    "Strong reactions to {topic}. The {stimulus} definitely affected me.",
                ],
                'careless': [
                    "idk",
                    "stuff",
                    "ok",
                    "survey",
                    "asdfgh",
                    "done",
                ],
                'default': [
                    "I viewed a {stimulus} and answered questions about {topic}.",
                    "The study involved evaluating {topic}.",
                    "I gave my opinions on the {stimulus} shown.",
                ]
            },
            'product_evaluation': {
                'positive': [
                    "This {product} looks really appealing to me. I like the {feature} and would consider buying it.",
                    "Great {product}! The {feature} caught my attention immediately.",
                    "I'm impressed with this {product}. It seems high quality.",
                    "Very nice {product}. I can see myself using this.",
                ],
                'negative': [
                    "Not interested in this {product}. The {feature} doesn't appeal to me.",
                    "This {product} doesn't meet my expectations.",
                    "I wouldn't purchase this {product}. Not convinced about the {feature}.",
                    "Disappointing {product}. There are better options available.",
                ],
                'neutral': [
                    "This {product} is okay. Nothing special about the {feature}.",
                    "Average {product}. Might consider it if the price is right.",
                    "The {product} is decent. {feature} is standard.",
                ]
            },
            'ai_reaction': {
                'positive': [
                    "I think AI recommendations are helpful. They save time and often find good options.",
                    "I trust AI to give relevant suggestions based on my preferences.",
                    "AI assistance makes shopping easier and more personalized.",
                ],
                'negative': [
                    "I don't really trust AI recommendations. I prefer to decide on my own.",
                    "AI suggestions feel impersonal. I'd rather get advice from real people.",
                    "I'm concerned about how AI uses my data to make recommendations.",
                ],
                'neutral': [
                    "AI recommendations can be useful sometimes, but I still like to research on my own.",
                    "I take AI suggestions as one input among many.",
                    "AI helps but isn't always accurate for my specific needs.",
                ]
            },
            'experience_description': {
                'hedonic': [
                    "It was an enjoyable experience. I felt {emotion} while using the {product}.",
                    "Using this {product} was fun and pleasurable.",
                    "I really enjoyed the experience. It was entertaining and satisfying.",
                ],
                'utilitarian': [
                    "The {product} functioned as expected and helped me accomplish my goal efficiently.",
                    "It was practical and got the job done.",
                    "The {product} served its purpose well. Good functionality.",
                ]
            }
        }

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
            traits: Participant trait values
            seed: Random seed for reproducibility

        Returns:
            Generated text response
        """
        rng = random.Random(seed)

        # Get templates for this response type
        type_templates = self.templates.get(response_type, self.templates['task_summary'])

        # Get style-specific templates or default
        style_templates = type_templates.get(persona_style, type_templates.get('default', []))

        if not style_templates:
            style_templates = type_templates.get('default', ["Response about {topic}."])

        # Select template
        template = rng.choice(style_templates)

        # Fill in context
        try:
            response = template.format(**context)
        except KeyError:
            # If context is missing keys, use partial formatting
            response = template
            for key, value in context.items():
                response = response.replace(f"{{{key}}}", value)

        # Add persona-based variation
        if traits.get('attention_level', 0.5) < 0.5:
            # Lower attention = potential typos, shorter responses
            if rng.random() < 0.3:
                response = response.lower()

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
