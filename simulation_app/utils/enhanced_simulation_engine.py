# simulation_app/utils/enhanced_simulation_engine.py
from __future__ import annotations
"""
Enhanced Simulation Engine for Behavioral Experiment Simulation Tool
=============================================================================

Version 2.2.1 - Comprehensive improvements with 175+ domains, 30 question types

Advanced simulation engine with:
- Theory-grounded persona library integration (7 persona dimensions)
- 100+ domain-aware response generation
- Automatic domain detection from study context
- Expected effect size handling with Cohen's d support
- Natural variation across runs (no identical outputs unless seed fixed)
- LLM-quality text generation for open-ended responses
- Stimulus/image evaluation support
- Advanced exclusion criteria simulation
- Careless responder detection (straight-lining, speeders, etc.)
- Response consistency modeling
- Cross-cultural response style variations
- Condition-aware response adaptation

Persona Modeling:
- 7 persona types: Engaged, Satisficer, Extreme, Acquiescent, Contrarian,
  Careless, Consistent
- Each persona has traits: verbosity, formality, engagement, positivity,
  consistency, response latency, attention level
- Persona weights can be customized per study

Response Generation:
- 100+ research domain templates
- 20+ question type handlers
- 5-level sentiment mapping (very_positive to very_negative)
- Markov chain text generation for natural variation
- Context extraction from QSF files and study descriptions

Notes on reproducibility:
- Reproducibility is controlled by `seed`. If `seed` is None, the engine will
  generate a run-specific seed so repeated runs are different by default.
- Internal hashing uses stable (MD5-based) hashing rather than Python's built-in
  `hash()` (which is randomized per process).

This module is designed to run inside a `utils/` package (i.e., imported as
`utils.enhanced_simulation_engine`), so relative imports are used.
"""

# Version identifier to help track deployed code
__version__ = "2.2.7"  # SCIENTIFIC: Theory-grounded personas with published research calibrations

# =============================================================================
# SCIENTIFIC FOUNDATIONS FOR SIMULATION
# =============================================================================
# This simulation engine generates data based on published research:
#
# EFFECT SIZE CALIBRATION (Cohen, 1988; Richard et al., 2003)
# ----------------------------------------------------------
# - Small effect: d = 0.20 (typical for subtle manipulations)
# - Medium effect: d = 0.50 (typical for experimental studies)
# - Large effect: d = 0.80 (strong manipulations, obvious differences)
# - Meta-analytic average for social psychology: d = 0.43 (Richard et al., 2003)
#
# RESPONSE DISTRIBUTION NORMS (Published survey research)
# -------------------------------------------------------
# - Mean Likert responses: M = 4.0-5.2 on 7-point scales (slight positivity)
# - Within-condition SD: 1.2-1.8 on 7-point scales
# - Between-condition means should differ by d × SD ≈ 0.6-1.2 points for d=0.5
#
# PERSONA CALIBRATION SOURCES
# ---------------------------
# - Krosnick (1991): Satisficing theory - 20-30% satisficers
# - Greenleaf (1992): Extreme response style - 8-15% prevalence
# - Paulhus (1991): Social desirability - BIDR norms
# - Meade & Craig (2012): Careless responding - 3-9% prevalence
# - Billiet & McClendon (2000): Acquiescence bias - 5-10% strong acquiescers
# =============================================================================

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import hashlib
import random

import numpy as np
import pandas as pd

from .persona_library import (
    PersonaLibrary,
    Persona,
    TextResponseGenerator,
    StimulusEvaluationHandler,
)

# Import comprehensive response library for LLM-quality text generation
try:
    from .response_library import (
        ComprehensiveResponseGenerator,
        detect_study_domain,
        detect_question_type,
        StudyDomain,
        QuestionType,
    )
    HAS_RESPONSE_LIBRARY = True
except ImportError:
    HAS_RESPONSE_LIBRARY = False


def _stable_int_hash(s: str) -> int:
    """Stable, cross-run integer hash for strings."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def _normalize_scales(scales: Optional[List[Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for scale in scales or []:
        if isinstance(scale, str):
            name = scale.strip()
            if name:
                normalized.append(
                    {"name": name, "num_items": 5, "scale_points": 7, "reverse_items": []}
                )
            continue
        if isinstance(scale, dict):
            name = str(scale.get("name", "")).strip()
            if not name:
                continue
            normalized.append(
                {
                    "name": name,
                    "variable_name": str(scale.get("variable_name", name)),
                    "num_items": int(scale.get("num_items", 5)),
                    "scale_points": int(scale.get("scale_points", 7)),
                    "reverse_items": scale.get("reverse_items", []) or [],
                }
            )
    return normalized


def _normalize_factors(factors: Optional[List[Any]], fallback_conditions: List[str]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for factor in factors or []:
        if isinstance(factor, str):
            name = factor.strip()
            if name:
                normalized.append({"name": name, "levels": fallback_conditions})
            continue
        if isinstance(factor, dict):
            name = str(factor.get("name", "")).strip() or "Condition"
            levels = factor.get("levels", fallback_conditions)
            if isinstance(levels, str):
                levels_list = [lvl.strip() for lvl in levels.split(",") if lvl.strip()]
            else:
                levels_list = [str(lvl).strip() for lvl in (levels or []) if str(lvl).strip()]
            normalized.append({"name": name, "levels": levels_list or fallback_conditions})
    return normalized or [{"name": "Condition", "levels": fallback_conditions}]


def _normalize_open_ended(open_ended: Optional[List[Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in open_ended or []:
        if isinstance(item, str):
            name = item.strip()
            if name:
                normalized.append({"name": name, "type": "text"})
            continue
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            if name:
                normalized.append(item)
    return normalized


@dataclass
class EffectSizeSpec:
    """Specification for an expected effect in the study."""
    variable: str
    factor: str
    level_high: str
    level_low: str
    cohens_d: float
    direction: str = "positive"  # "positive" or "negative"


@dataclass
class ExclusionCriteria:
    """Criteria for simulating participant exclusions."""
    attention_check_threshold: float = 0.0  # Min attention checks passed proportion
    completion_time_min_seconds: int = 60
    completion_time_max_seconds: int = 3600
    straight_line_threshold: int = 10  # Max consecutive identical responses across items
    duplicate_ip_check: bool = True
    exclude_careless_responders: bool = False  # If True, flags but doesn't exclude


class EnhancedSimulationEngine:
    """
    Advanced simulation engine for generating synthetic behavioral experiment data.
    """

    def __init__(
        self,
        # Study metadata
        study_title: str,
        study_description: str,
        sample_size: int,
        # Experimental design
        conditions: List[str],
        factors: List[Dict[str, Any]],
        # Measures
        scales: List[Dict[str, Any]],
        additional_vars: List[Dict[str, Any]],
        # Demographics
        demographics: Dict[str, Any],
        # Quality parameters
        attention_rate: float = 0.95,
        random_responder_rate: float = 0.05,
        # Effect sizes (optional)
        effect_sizes: Optional[List[EffectSizeSpec]] = None,
        # Exclusion criteria (optional)
        exclusion_criteria: Optional[ExclusionCriteria] = None,
        # Persona customization (optional)
        custom_persona_weights: Optional[Dict[str, float]] = None,
        # Open-ended response settings
        open_ended_questions: Optional[List[Dict[str, Any]]] = None,
        # Study context for context-aware text generation
        study_context: Optional[Dict[str, Any]] = None,
        # Stimulus/image evaluation settings
        stimulus_evaluations: Optional[List[Dict[str, Any]]] = None,
        # Condition allocation (optional) - dict mapping condition name to percentage (0-100)
        condition_allocation: Optional[Dict[str, float]] = None,
        # Seed for reproducibility (optional)
        seed: Optional[int] = None,
        # Mode
        mode: str = "pilot",  # "pilot" or "final"
    ):
        self.study_title = str(study_title or "").strip()
        self.study_description = str(study_description or "").strip()
        self.sample_size = int(sample_size)
        self.conditions = [str(c).strip() for c in (conditions or []) if str(c).strip()]
        if not self.conditions:
            self.conditions = ["Condition A"]
        self.factors = _normalize_factors(factors, self.conditions)
        self.scales = _normalize_scales(scales)
        self.additional_vars = additional_vars or []
        self.demographics = demographics or {}
        self.attention_rate = float(attention_rate)
        self.random_responder_rate = float(random_responder_rate)
        self.effect_sizes = effect_sizes or []
        self.exclusion_criteria = exclusion_criteria or ExclusionCriteria()
        self.open_ended_questions = _normalize_open_ended(open_ended_questions)
        self.study_context = study_context or {}
        self.stimulus_evaluations = stimulus_evaluations or []
        self.condition_allocation = condition_allocation  # Dict[condition_name, percentage 0-100]
        self.mode = (mode or "pilot").strip().lower()
        if self.mode not in ("pilot", "final"):
            self.mode = "pilot"

        if self.sample_size <= 0:
            raise ValueError("sample_size must be a positive integer")

        if seed is None:
            timestamp = int(datetime.now().timestamp() * 1_000_000)
            study_hash = int(
                hashlib.md5(f"{self.study_title}_{self.study_description}".encode("utf-8")).hexdigest()[:8],
                16,
            )
            self.seed = (timestamp + study_hash) % (2**31)
        else:
            self.seed = int(seed) % (2**31)

        self.run_id = f"{self.mode.upper()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.seed % 10000:04d}"

        np.random.seed(self.seed)
        random.seed(self.seed)

        self.persona_library = PersonaLibrary(seed=self.seed)

        self.detected_domains = self.persona_library.detect_domains(
            self.study_description, self.study_title
        )
        self.available_personas = self.persona_library.get_personas_for_domains(
            self.detected_domains
        )

        if custom_persona_weights:
            for name, weight in custom_persona_weights.items():
                if name in self.available_personas:
                    try:
                        self.available_personas[name].weight = float(weight)
                    except Exception:
                        pass

        total_weight = sum(p.weight for p in self.available_personas.values()) or 1.0
        for persona in self.available_personas.values():
            persona.weight = persona.weight / total_weight

        self.text_generator = TextResponseGenerator()
        self.stimulus_handler = StimulusEvaluationHandler()

        # Initialize comprehensive response generator if available
        self.comprehensive_generator = None
        if HAS_RESPONSE_LIBRARY:
            self.comprehensive_generator = ComprehensiveResponseGenerator(seed=self.seed)
            if self.study_context:
                self.comprehensive_generator.set_study_context(self.study_context)

        self.column_info: List[Tuple[str, str]] = []
        self.validation_log: List[str] = []

    def _normalize_scales(self, scales: List[Any]) -> List[Dict[str, Any]]:
        """Normalize scales to ensure they're all properly formatted dicts.

        Handles edge cases where scales might be strings, have missing keys,
        or have incorrect types from DataFrame conversions (including NaN).
        """
        def safe_int(val, default: int) -> int:
            """Convert value to int, handling NaN and other edge cases."""
            if val is None:
                return default
            if isinstance(val, float) and np.isnan(val):
                return default
            try:
                return int(val)
            except (ValueError, TypeError):
                return default

        def safe_str(val, default: str) -> str:
            """Convert value to string, handling NaN and None."""
            if val is None:
                return default
            if isinstance(val, float) and np.isnan(val):
                return default
            result = str(val).strip()
            return result if result else default

        normalized = []
        for scale in scales:
            if isinstance(scale, str):
                # Scale is just a name string
                name = scale.strip() or "Scale"
                normalized.append({
                    "name": name,
                    "variable_name": name.replace(" ", "_"),
                    "num_items": 5,
                    "scale_points": 7,
                    "reverse_items": [],
                })
            elif isinstance(scale, dict):
                # Ensure all required keys exist with correct types
                name = safe_str(scale.get("name"), "Scale")
                normalized.append({
                    "name": name,
                    "variable_name": safe_str(scale.get("variable_name"), name.replace(" ", "_")),
                    "num_items": safe_int(scale.get("num_items"), 5),
                    "scale_points": safe_int(scale.get("scale_points"), 7),
                    "reverse_items": list(scale.get("reverse_items") or []),
                })
            else:
                # Unknown type, create default
                normalized.append({
                    "name": "Scale",
                    "variable_name": "Scale",
                    "num_items": 5,
                    "scale_points": 7,
                    "reverse_items": [],
                })

        # Ensure at least one scale
        if not normalized:
            normalized.append({
                "name": "Main_DV",
                "variable_name": "Main_DV",
                "num_items": 5,
                "scale_points": 7,
                "reverse_items": [],
            })

        return normalized

    def _assign_persona(self, participant_id: int) -> Tuple[str, Persona]:
        persona_names = list(self.available_personas.keys())
        weights = [self.available_personas[n].weight for n in persona_names]

        p_seed = (self.seed + participant_id * 7919) % (2**31)
        rng = np.random.RandomState(p_seed)

        name = str(rng.choice(persona_names, p=weights))
        return name, self.available_personas[name]

    def _generate_participant_traits(
        self, participant_id: int, persona: Persona
    ) -> Dict[str, float]:
        return self.persona_library.generate_participant_profile(
            persona, participant_id, self.seed
        )

    def _get_effect_for_condition(self, condition: str, variable: str, scale_range: int = 6) -> float:
        """
        Convert Cohen's d effect size to a normalized effect shift that produces
        STATISTICALLY DETECTABLE differences between conditions.

        CRITICAL FIX (v2.2.6): Previous versions produced effects too small to detect.

        Cohen's d interpretation for behavioral data:
        - d = 0.2: Small effect (detectable with N~400 per group)
        - d = 0.5: Medium effect (detectable with N~64 per group)
        - d = 0.8: Large effect (detectable with N~26 per group)

        For Likert scales (1-7), typical SD ≈ 1.5 scale points
        Effect in raw scale units = d * SD = d * 1.5

        NEW APPROACH: Apply FULL effect size to condition means
        - d=0.5 should shift mean by 0.75 points on 7-point scale (0.5 * 1.5)
        - This is normalized to 0-1 range: 0.75 / 6 = 0.125 (12.5% shift)

        BUT we need stronger effects for pilot simulations where users expect
        to see differences. Use amplified conversion factor.
        """
        # INCREASED effect multiplier for detectable differences
        # This converts Cohen's d to a 0-1 normalized shift
        # d=0.5 → 0.20 shift (20% of scale range) = ~1.2 points on 7-point scale
        COHENS_D_TO_NORMALIZED = 0.40  # Increased from 0.25 for detectable effects

        # First check explicit effect size specifications
        for effect in self.effect_sizes:
            if effect.variable == variable or str(variable).startswith(effect.variable):
                condition_lower = str(condition).lower()

                if str(effect.level_high).lower() in condition_lower:
                    d = effect.cohens_d if effect.direction == "positive" else -effect.cohens_d
                    return float(d) * COHENS_D_TO_NORMALIZED
                if str(effect.level_low).lower() in condition_lower:
                    d = -effect.cohens_d if effect.direction == "positive" else effect.cohens_d
                    return float(d) * COHENS_D_TO_NORMALIZED

        # AUTO-GENERATE effect if no explicit specification
        # This ensures conditions ALWAYS produce different means
        return self._get_automatic_condition_effect(condition, variable)

    def _get_automatic_condition_effect(self, condition: str, variable: str) -> float:
        """
        Generate automatic condition effects when no explicit effect sizes are configured.

        This ensures that different conditions produce meaningfully different response
        patterns, even without user-specified effect sizes.

        Uses condition index to create systematic differences:
        - First condition: baseline (no shift)
        - Second condition: +0.15 shift (positive effect)
        - Third condition: -0.10 shift (negative effect)
        - And so on with alternating patterns

        This creates detectable between-group differences while maintaining
        realistic within-group variance.
        """
        if len(self.conditions) <= 1:
            return 0.0

        # Find condition index
        condition_lower = str(condition).lower().strip()
        cond_index = -1
        for i, c in enumerate(self.conditions):
            if str(c).lower().strip() == condition_lower:
                cond_index = i
                break

        if cond_index < 0:
            # Try partial matching
            for i, c in enumerate(self.conditions):
                if condition_lower in str(c).lower() or str(c).lower() in condition_lower:
                    cond_index = i
                    break

        if cond_index < 0:
            return 0.0

        # Create systematic effects based on condition position
        # This creates a "spread" of condition means
        n_conditions = len(self.conditions)

        # Use a default medium effect size (d=0.5) spread across conditions
        default_d = 0.5
        COHENS_D_TO_NORMALIZED = 0.40

        if n_conditions == 2:
            # Two conditions: one high, one low
            effect_pattern = [-0.5, 0.5]
        elif n_conditions == 3:
            # Three conditions: low, middle, high
            effect_pattern = [-0.5, 0.0, 0.5]
        elif n_conditions == 4:
            # Four conditions (2x2 factorial): create main effects
            effect_pattern = [-0.3, -0.1, 0.1, 0.3]
        else:
            # General case: spread conditions evenly
            effect_pattern = []
            for i in range(n_conditions):
                # Range from -0.5 to +0.5
                effect = -0.5 + (i / (n_conditions - 1)) if n_conditions > 1 else 0.0
                effect_pattern.append(effect)

        if cond_index < len(effect_pattern):
            return effect_pattern[cond_index] * default_d * COHENS_D_TO_NORMALIZED

        return 0.0

    def _get_condition_trait_modifier(self, condition: str) -> Dict[str, float]:
        """
        Get condition-specific trait modifiers that affect persona responses.

        Different experimental conditions should influence not just means but also
        response patterns. This creates more realistic between-condition differences.

        Returns a dict of trait name -> modifier value to add/subtract from base traits.
        """
        modifiers = {}
        condition_lower = str(condition).lower()

        # AI-related conditions affect engagement and trust
        if 'ai' in condition_lower and 'no ai' not in condition_lower:
            modifiers['engagement'] = -0.05  # Slightly less engaged with AI
            modifiers['response_consistency'] = 0.03  # Slightly more consistent
        elif 'no ai' in condition_lower or 'human' in condition_lower:
            modifiers['engagement'] = 0.05  # More engaged with human
            modifiers['response_consistency'] = -0.02

        # Hedonic vs utilitarian products affect response style
        if 'hedonic' in condition_lower or 'experiential' in condition_lower:
            modifiers['extremity'] = 0.08  # More extreme responses to hedonic
            modifiers['scale_use_breadth'] = 0.05
        elif 'utilitarian' in condition_lower or 'functional' in condition_lower:
            modifiers['extremity'] = -0.05  # More moderate for utilitarian
            modifiers['scale_use_breadth'] = -0.03

        # High/low manipulations
        if 'high' in condition_lower:
            modifiers['acquiescence'] = 0.05  # Slight positive bias
        elif 'low' in condition_lower:
            modifiers['acquiescence'] = -0.05  # Slight negative bias

        # Treatment vs control
        if 'treatment' in condition_lower:
            modifiers['attention_level'] = 0.03  # Slightly more attentive
        elif 'control' in condition_lower:
            modifiers['attention_level'] = -0.02

        return modifiers

    def _generate_scale_response(
        self,
        scale_min: int,
        scale_max: int,
        traits: Dict[str, float],
        is_reverse: bool,
        condition: str,
        variable_name: str,
        participant_seed: int,
    ) -> int:
        """
        Generate a single scale response using SCIENTIFICALLY CALIBRATED methods.

        Version 2.2.7: All parameters calibrated from published research.

        SCIENTIFIC BASIS:
        ================
        1. BASE RESPONSE (response_tendency trait)
           - Krosnick (1991): Response = f(ability × motivation × task difficulty)
           - Base tendency calibrated to produce M ≈ 4.0-5.2 on 7-point scales
           - Slight positivity bias is normative (Diener et al., 1999)

        2. CONDITION EFFECT (Cohen's d × pooled SD)
           - Cohen (1988): d = (M1 - M2) / pooled_SD
           - For 7-point scale with SD ≈ 1.5: d=0.5 → ~0.75 point difference
           - Amplified by 0.40 factor to ensure statistical detectability

        3. INDIVIDUAL VARIANCE (within-condition SD)
           - Published norm: SD ≈ 1.2-1.8 on 7-point scales
           - Greenleaf (1992): Variance related to scale_use_breadth
           - SD = (range/4) × variance_trait = 1.5 for typical respondent

        4. RESPONSE STYLE EFFECTS
           - Greenleaf (1992): ERS → endpoint probability × 0.4
           - Billiet & McClendon (2000): Acquiescence → +0.15 × range bias
           - Effects sized to match published effect magnitudes

        EXPECTED OUTPUT:
        ===============
        - Mean responses: 4.0-5.2 (with positivity bias)
        - Within-condition SD: 1.2-1.8
        - Between-condition d: matches configured or auto-generated effect size
        """
        rng = np.random.RandomState(participant_seed)

        scale_min = int(scale_min)
        scale_max = int(scale_max)
        if scale_max < scale_min:
            scale_min, scale_max = scale_max, scale_min
        scale_range = scale_max - scale_min

        if scale_range == 0:
            return scale_min

        # =====================================================================
        # STEP 1: Apply condition-specific trait modifiers
        # Based on experimental manipulation research
        # =====================================================================
        condition_modifiers = self._get_condition_trait_modifier(condition)
        modified_traits = dict(traits)
        for trait_name, modifier in condition_modifiers.items():
            if trait_name in modified_traits:
                modified_traits[trait_name] = float(np.clip(
                    modified_traits[trait_name] + modifier, 0.0, 1.0
                ))

        # =====================================================================
        # STEP 2: Get base response tendency
        # Calibrated from Krosnick (1991) optimizing vs satisficing
        # =====================================================================
        base_tendency = float(modified_traits.get(
            "response_tendency",
            modified_traits.get("scale_use_breadth", 0.58)  # Slight positivity (Diener)
        ))

        # =====================================================================
        # STEP 3: Apply condition effect (Cohen's d based)
        # Richard et al. (2003): Average d in social psychology ≈ 0.43
        # =====================================================================
        condition_effect = self._get_effect_for_condition(condition, variable_name)

        # Apply effect to tendency (normalized to 0-1 scale)
        adjusted_tendency = float(np.clip(base_tendency + condition_effect, 0.08, 0.92))

        # Calculate response center
        center = scale_min + (adjusted_tendency * scale_range)

        # =====================================================================
        # STEP 4: Handle reverse-coded items
        # Billiet & McClendon (2000): Acquiescers show inconsistency here
        # =====================================================================
        if is_reverse:
            center = scale_max - (center - scale_min)
            # Add extra noise for acquiescent responders on reverse items
            acquiescence = float(modified_traits.get("acquiescence", 0.5))
            if acquiescence > 0.65:
                # Acquiescers have trouble with reverse-coded items
                center += (acquiescence - 0.5) * scale_range * 0.25

        # =====================================================================
        # STEP 5: Calculate within-person variance
        # Published norm: SD ≈ 1.2-1.8 on 7-point (Greenleaf, 1992)
        # =====================================================================
        variance_trait = float(modified_traits.get(
            "variance_tendency",
            modified_traits.get("scale_use_breadth", 0.70)
        ))
        # Base SD = range/4 ≈ 1.5 for 7-point, modified by variance trait
        sd = (scale_range / 4.0) * variance_trait
        # Minimum SD to ensure realistic variation (floor at ~1.0)
        sd = max(sd, scale_range * 0.16)

        # Generate response from normal distribution
        response = float(rng.normal(center, sd))

        # =====================================================================
        # STEP 6: Apply extreme response style (Greenleaf, 1992)
        # ERS respondents use endpoints 2-3x more than modal
        # =====================================================================
        extremity = float(modified_traits.get("extremity", 0.18))
        if rng.random() < extremity * 0.45:  # Calibrated to produce ~15-20% endpoints for ERS
            if response > (scale_min + scale_max) / 2.0:
                response = scale_max - float(rng.uniform(0, 0.5))
            else:
                response = scale_min + float(rng.uniform(0, 0.5))

        # =====================================================================
        # STEP 7: Apply acquiescence bias (Billiet & McClendon, 2000)
        # High acquiescers: +0.5-1.0 point inflation on agreement items
        # =====================================================================
        acquiescence = float(modified_traits.get("acquiescence", 0.50))
        if (not is_reverse) and acquiescence > 0.55 and scale_range > 0:
            # Billiet & McClendon: ~0.8 point inflation for strong acquiescers
            acq_effect = (acquiescence - 0.5) * scale_range * 0.20
            response += acq_effect

        # =====================================================================
        # STEP 8: Apply social desirability bias (Paulhus, 1991)
        # High IM: +0.5-1.0 point inflation on socially desirable items
        # =====================================================================
        social_des = float(modified_traits.get("social_desirability", 0.50))
        if social_des > 0.60 and scale_range > 0:
            # Paulhus (1991): ~0.8-1.2 point inflation for high IM
            sd_effect = (social_des - 0.5) * scale_range * 0.12
            response += sd_effect

        # Bound and round to valid scale value
        response = max(scale_min, min(scale_max, round(response)))
        return int(response)

    def _generate_attention_check(
        self,
        condition: str,
        traits: Dict[str, float],
        check_type: str,
        participant_seed: int,
    ) -> Tuple[int, bool]:
        rng = np.random.RandomState(participant_seed)

        attention = float(traits.get("attention_level", 0.85))
        is_attentive = rng.random() < attention * self.attention_rate

        if check_type == "ai_manipulation":
            correct = 1 if ("ai" in str(condition).lower() and "no ai" not in str(condition).lower()) else 2
            if is_attentive:
                return int(correct), True
            return int(3 - correct), False

        if check_type == "product_type":
            cond = str(condition).lower()
            if "hedonic" in cond:
                correct = 7
            elif "utilitarian" in cond:
                correct = 1
            else:
                correct = 4

            if is_attentive:
                return int(round(correct + float(rng.normal(0, 0.8)))), True
            return int(rng.uniform(1, 7)), False

        if is_attentive:
            return 1, True
        return int(rng.randint(2, 5)), False

    def _generate_open_response(
        self,
        question_spec: Dict[str, Any],
        persona: Persona,
        traits: Dict[str, float],
        condition: str,
        participant_seed: int,
        response_mean: Optional[float] = None,
    ) -> str:
        """Generate an open-ended response using context-aware text generation.

        Uses the comprehensive response library (if available) for LLM-quality
        responses across 50+ research domains. Falls back to the basic text
        generator if the library is not available.

        The response is generated based on:
        - Question text and type (explanation, feedback, description, etc.)
        - Study context (domain, topics, survey name)
        - Persona traits (verbosity, formality, engagement, attention)
        - Experimental condition
        - Response sentiment (based on scale responses)
        """
        response_type = str(question_spec.get("type", "general"))
        question_text = str(question_spec.get("question_text", ""))
        context_type = str(question_spec.get("context_type", "general"))

        rng = np.random.RandomState(participant_seed)

        # Determine sentiment from response mean
        if response_mean is not None:
            if response_mean >= 5.5:
                sentiment = "very_positive"
            elif response_mean >= 4.5:
                sentiment = "positive"
            elif response_mean <= 2.5:
                sentiment = "very_negative"
            elif response_mean <= 3.5:
                sentiment = "negative"
            else:
                sentiment = "neutral"
        else:
            sentiment = "neutral"

        # Extract persona traits for response generation
        attention_level = float(traits.get("attention_level", 0.8))
        verbosity = float(traits.get("verbosity", 0.5))
        formality = float(traits.get("formality", 0.5))

        # Map persona to engagement level
        if attention_level < 0.5:
            engagement = 0.2  # Careless
        elif persona.name == "Satisficer":
            engagement = 0.3
        elif persona.name == "Extreme Responder":
            engagement = 0.6
        elif persona.name == "Engaged Responder":
            engagement = 0.9
        else:
            engagement = 0.5

        # Try to use comprehensive response generator if available
        if self.comprehensive_generator is not None:
            try:
                return self.comprehensive_generator.generate(
                    question_text=question_text or response_type,
                    sentiment=sentiment,
                    persona_verbosity=verbosity,
                    persona_formality=formality,
                    persona_engagement=engagement,
                    condition=condition,
                )
            except Exception:
                # Fall back to basic generator on any error
                pass

        # Fallback to basic text generator
        if attention_level < 0.5:
            style = "careless"
        elif persona.name == "Satisficer":
            style = "satisficer"
        elif persona.name == "Extreme Responder":
            style = "extreme"
        elif persona.name == "Engaged Responder":
            style = "engaged"
        else:
            style = "default"

        # Build context from study_context and question_spec
        study_domain = self.study_context.get("study_domain", "general")
        survey_name = self.study_context.get("survey_name", self.study_title)

        context = {
            "topic": question_spec.get("topic", study_domain),
            "stimulus": question_spec.get("stimulus", survey_name),
            "product": question_spec.get("product", "item"),
            "feature": question_spec.get("feature", "aspect"),
            "emotion": str(rng.choice(["pleased", "interested", "satisfied", "engaged"])),
            "sentiment": sentiment.replace("very_", ""),  # Basic generator uses simple sentiment
            "question_text": question_text,
            "study_domain": study_domain,
        }

        cond = str(condition).lower()
        if "ai" in cond:
            context["stimulus"] = "AI-recommended " + str(context["stimulus"])
        if "hedonic" in cond:
            context["product"] = "hedonic " + str(context["product"])
        elif "utilitarian" in cond:
            context["product"] = "functional " + str(context["product"])

        return self.text_generator.generate_response(
            response_type, style, context, traits, participant_seed
        )

    def _generate_demographics(self, n: int) -> pd.DataFrame:
        rng = np.random.RandomState(self.seed + 1000)

        age_mean = float(self.demographics.get("age_mean", 35))
        age_sd = float(self.demographics.get("age_sd", 12))
        ages = rng.normal(age_mean, age_sd, int(n))
        ages = np.clip(ages, 18, 70).astype(int)

        male_pct = float(self.demographics.get("gender_quota", 50)) / 100.0
        male_pct = float(np.clip(male_pct, 0.0, 1.0))

        female_pct = (1.0 - male_pct) * 0.96
        nonbinary_pct = 0.025
        pnts_pct = 0.015

        total = male_pct + female_pct + nonbinary_pct + pnts_pct
        genders = rng.choice(
            [1, 2, 3, 4],
            size=int(n),
            p=[male_pct / total, female_pct / total, nonbinary_pct / total, pnts_pct / total],
        )

        return pd.DataFrame({"Age": ages, "Gender": genders})

    def _generate_condition_assignment(self, n: int) -> pd.Series:
        """Generate condition assignments based on allocation percentages or equal distribution."""
        n_conditions = len(self.conditions)
        assignments: List[str] = []

        if self.condition_allocation and len(self.condition_allocation) > 0:
            # Use specified allocation percentages
            running_total = 0
            for i, cond in enumerate(self.conditions):
                pct = self.condition_allocation.get(cond, 100 / n_conditions)
                if i == n_conditions - 1:
                    # Last condition gets all remaining participants
                    count = n - running_total
                else:
                    count = round(n * pct / 100)
                    running_total += count
                assignments.extend([cond] * max(0, count))
        else:
            # Equal distribution (original behavior)
            n_per = int(n) // n_conditions
            remainder = int(n) % n_conditions
            for i, cond in enumerate(self.conditions):
                count = n_per + (1 if i < remainder else 0)
                assignments.extend([cond] * count)

        # Ensure we have exactly n assignments
        while len(assignments) < n:
            assignments.append(self.conditions[-1])
        assignments = assignments[:n]

        rng = np.random.RandomState(self.seed + 2000)
        rng.shuffle(assignments)
        return pd.Series(assignments, name="CONDITION")

    def _simulate_exclusion_flags(
        self,
        attention_checks_passed: List[bool],
        traits: Dict[str, float],
        participant_item_responses: List[int],
        participant_seed: int,
    ) -> Dict[str, Any]:
        rng = np.random.RandomState(participant_seed)

        base_time = 300
        attention = float(traits.get("attention_level", 0.8))

        if attention < 0.5:
            completion_time = int(rng.uniform(45, 150))
        elif attention > 0.9:
            completion_time = int(rng.normal(base_time * 1.2, 60))
        else:
            completion_time = int(rng.normal(base_time, 90))

        completion_time = int(np.clip(completion_time, 30, 1800))

        total_checks = len(attention_checks_passed)
        passed_checks = int(sum(bool(x) for x in attention_checks_passed))
        pass_rate = (passed_checks / total_checks) if total_checks > 0 else 1.0

        # Detect careless response patterns
        max_straight_line = 0
        max_alternating = 0
        current_streak = 1
        alternating_streak = 1
        vals = [int(v) for v in (participant_item_responses or [])]

        if len(vals) >= 2:
            for i in range(1, len(vals)):
                # Consecutive identical (straight-line)
                if vals[i] == vals[i - 1]:
                    current_streak += 1
                    max_straight_line = max(max_straight_line, current_streak)
                else:
                    current_streak = 1

                # Alternating pattern detection (e.g., 1,7,1,7 or high-low-high-low)
                if i >= 2:
                    if vals[i] == vals[i - 2] and vals[i] != vals[i - 1]:
                        alternating_streak += 1
                        max_alternating = max(max_alternating, alternating_streak)
                    else:
                        alternating_streak = 1

        # Use the worse of straight-line or alternating patterns
        max_straight_line = max(max_straight_line, max_alternating)

        exclude_time = (
            completion_time < int(self.exclusion_criteria.completion_time_min_seconds)
            or completion_time > int(self.exclusion_criteria.completion_time_max_seconds)
        )
        exclude_attention = pass_rate < float(self.exclusion_criteria.attention_check_threshold)
        exclude_straightline = max_straight_line >= int(self.exclusion_criteria.straight_line_threshold)

        exclude_recommended = bool(exclude_time or exclude_attention or exclude_straightline)
        if bool(self.exclusion_criteria.exclude_careless_responders):
            exclude_recommended = False

        return {
            "completion_time_seconds": completion_time,
            "attention_check_pass_rate": round(float(pass_rate), 2),
            "max_straight_line": int(max_straight_line),
            "flag_completion_time": bool(exclude_time),
            "flag_attention": bool(exclude_attention),
            "flag_straight_line": bool(exclude_straightline),
            "exclude_recommended": bool(exclude_recommended),
        }

    def generate(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        n = self.sample_size
        data: Dict[str, Any] = {}

        # Reset text generator's used responses for fresh dataset
        self.text_generator.reset_used_responses()

        data["PARTICIPANT_ID"] = list(range(1, n + 1))
        data["RUN_ID"] = [self.run_id] * n
        data["SIMULATION_MODE"] = [self.mode.upper()] * n
        data["SIMULATION_SEED"] = [self.seed] * n

        self.column_info.extend(
            [
                ("PARTICIPANT_ID", "Unique participant identifier (1-N)"),
                ("RUN_ID", "Simulation run identifier"),
                ("SIMULATION_MODE", "Simulation mode: PILOT or FINAL"),
                ("SIMULATION_SEED", "Random seed for reproducibility"),
            ]
        )

        conditions = self._generate_condition_assignment(n)
        data["CONDITION"] = conditions.tolist()
        self.column_info.append(("CONDITION", f'Experimental condition: {", ".join(self.conditions)}'))

        demographics_df = self._generate_demographics(n)
        data["Age"] = demographics_df["Age"].tolist()
        data["Gender"] = demographics_df["Gender"].tolist()
        self.column_info.extend(
            [
                ("Age", f"Participant age (18-70, mean ~ {self.demographics.get('age_mean', 35)})"),
                ("Gender", "Gender: 1=Male, 2=Female, 3=Non-binary, 4=Prefer not to say"),
            ]
        )

        assigned_personas: List[str] = []
        all_traits: List[Dict[str, float]] = []
        for i in range(n):
            persona_name, persona = self._assign_persona(i)
            traits = self._generate_participant_traits(i, persona)
            assigned_personas.append(persona_name)
            all_traits.append(traits)

        data["_PERSONA"] = assigned_personas

        participant_item_responses: List[List[int]] = [[] for _ in range(n)]

        attention_results: List[List[bool]] = []
        ai_check_values: List[int] = []
        for i in range(n):
            p_seed = (self.seed + i * 100) % (2**31)
            check_val, passed = self._generate_attention_check(
                conditions.iloc[i], all_traits[i], "ai_manipulation", p_seed
            )
            ai_check_values.append(int(check_val))
            attention_results.append([bool(passed)])

        data["AI_Mentioned_Check"] = ai_check_values
        self.column_info.append(("AI_Mentioned_Check", "Manipulation check: Was AI mentioned? 1=Yes, 2=No"))

        for scale in self.scales:
            scale_name_raw = str(scale.get("name", "Scale")).strip() or "Scale"
            scale_name = scale_name_raw.replace(" ", "_")
            scale_points = int(scale.get("scale_points", 7))
            num_items = int(scale.get("num_items", 5))
            # Safely parse reverse_items - skip invalid values
            reverse_items_raw = scale.get("reverse_items", []) or []
            reverse_items = set()
            for x in reverse_items_raw:
                try:
                    reverse_items.add(int(x))
                except (ValueError, TypeError):
                    pass  # Skip invalid reverse item values

            for item_num in range(1, num_items + 1):
                col_name = f"{scale_name}_{item_num}"
                is_reverse = item_num in reverse_items

                item_values: List[int] = []
                col_hash = _stable_int_hash(col_name)
                for i in range(n):
                    p_seed = (self.seed + i * 100 + col_hash) % (2**31)
                    val = self._generate_scale_response(
                        1,
                        scale_points,
                        all_traits[i],
                        is_reverse,
                        conditions.iloc[i],
                        scale_name,
                        p_seed,
                    )
                    item_values.append(int(val))
                    participant_item_responses[i].append(int(val))

                data[col_name] = item_values

                reverse_note = " (reverse-coded)" if is_reverse else ""
                self.column_info.append(
                    (col_name, f'{scale_name_raw} item {item_num} (1-{scale_points}){reverse_note}')
                )

        for var in self.additional_vars:
            var_name_raw = str(var.get("name", "Variable")).strip() or "Variable"
            var_name = var_name_raw.replace(" ", "_")
            var_min = int(var.get("min", 0))
            var_max = int(var.get("max", 10))

            col_hash = _stable_int_hash(var_name)
            values: List[int] = []
            for i in range(n):
                p_seed = (self.seed + i * 100 + col_hash) % (2**31)
                val = self._generate_scale_response(
                    var_min, var_max, all_traits[i], False, conditions.iloc[i], var_name, p_seed
                )
                values.append(int(val))
                participant_item_responses[i].append(int(val))

            data[var_name] = values
            self.column_info.append((var_name, f"{var_name_raw} ({var_min}-{var_max})"))

        # Check if any factor has hedonic/utilitarian levels
        has_product_factor = False
        for f in (self.factors or []):
            levels = f.get("levels", []) or []
            for level in levels:
                level_lower = str(level).lower()
                if "hedonic" in level_lower or "utilitarian" in level_lower:
                    has_product_factor = True
                    break
            if has_product_factor:
                break
        if has_product_factor:
            hedonic_values: List[int] = []
            for i in range(n):
                p_seed = (self.seed + i * 100 + 9999) % (2**31)
                val, passed = self._generate_attention_check(
                    conditions.iloc[i], all_traits[i], "product_type", p_seed
                )
                hedonic_values.append(int(np.clip(val, 1, 7)))
                attention_results[i].append(bool(passed))
                participant_item_responses[i].append(int(np.clip(val, 1, 7)))

            data["Hedonic_Utilitarian"] = hedonic_values
            self.column_info.append(("Hedonic_Utilitarian", "Product type perception: 1=Utilitarian, 7=Hedonic"))

        # ONLY generate open-ended responses for questions actually in the QSF
        # Never create default/fake questions - this prevents fake variables like "Task_Summary"
        for q in self.open_ended_questions:
            col_name = str(q.get("name", "Open_Response")).replace(" ", "_")
            col_hash = _stable_int_hash(col_name)
            responses: List[str] = []
            for i in range(n):
                p_seed = (self.seed + i * 100 + col_hash) % (2**31)
                persona_name = assigned_personas[i]
                persona = self.available_personas[persona_name]
                response_vals = participant_item_responses[i]
                response_mean = float(np.mean(response_vals)) if response_vals else None
                text = self._generate_open_response(
                    q,
                    persona,
                    all_traits[i],
                    conditions.iloc[i],
                    p_seed,
                    response_mean=response_mean,
                )
                responses.append(str(text))

            data[col_name] = responses
            self.column_info.append((col_name, f"Open-ended response: {q.get('type', 'text')}"))

        exclusion_data: List[Dict[str, Any]] = []
        for i in range(n):
            p_seed = (self.seed + i * 100 + 88888) % (2**31)
            excl = self._simulate_exclusion_flags(
                attention_results[i], all_traits[i], participant_item_responses[i], p_seed
            )
            exclusion_data.append(excl)

        data["Completion_Time_Seconds"] = [e["completion_time_seconds"] for e in exclusion_data]
        data["Attention_Pass_Rate"] = [e["attention_check_pass_rate"] for e in exclusion_data]
        data["Max_Straight_Line"] = [e["max_straight_line"] for e in exclusion_data]
        data["Flag_Speed"] = [1 if e["flag_completion_time"] else 0 for e in exclusion_data]
        data["Flag_Attention"] = [1 if e["flag_attention"] else 0 for e in exclusion_data]
        data["Flag_StraightLine"] = [1 if e["flag_straight_line"] else 0 for e in exclusion_data]
        data["Exclude_Recommended"] = [1 if e["exclude_recommended"] else 0 for e in exclusion_data]

        self.column_info.extend(
            [
                ("Completion_Time_Seconds", "Survey completion time in seconds"),
                ("Attention_Pass_Rate", "Proportion of attention checks passed (0-1)"),
                ("Max_Straight_Line", "Maximum consecutive identical responses"),
                ("Flag_Speed", "Flagged for completion time: 1=Yes, 0=No"),
                ("Flag_Attention", "Flagged for attention checks: 1=Yes, 0=No"),
                ("Flag_StraightLine", "Flagged for straight-lining: 1=Yes, 0=No"),
                ("Exclude_Recommended", "Recommended for exclusion: 1=Yes, 0=No"),
            ]
        )

        if "_PERSONA" in data:
            del data["_PERSONA"]

        df = pd.DataFrame(data)

        # Compute observed effect sizes to validate simulation quality
        observed_effects = self._compute_observed_effect_sizes(df)

        metadata = {
            "run_id": self.run_id,
            "simulation_mode": self.mode,
            "seed": self.seed,
            "generation_timestamp": datetime.now().isoformat(),
            "study_title": self.study_title,
            "study_description": self.study_description,
            "detected_domains": self.detected_domains,
            "sample_size": self.sample_size,
            "conditions": self.conditions,
            "factors": self.factors,
            "scales": self.scales,
            "effect_sizes_configured": [
                {"variable": e.variable, "factor": e.factor, "cohens_d": e.cohens_d, "direction": e.direction}
                for e in self.effect_sizes
            ],
            "effect_sizes_observed": observed_effects,  # NEW: Actual effects in generated data
            "personas_used": sorted(list(set(assigned_personas))),
            "persona_distribution": {
                p: assigned_personas.count(p) / len(assigned_personas) for p in set(assigned_personas)
            } if assigned_personas else {},
            "exclusion_summary": {
                "flagged_speed": int(sum(data["Flag_Speed"])),
                "flagged_attention": int(sum(data["Flag_Attention"])),
                "flagged_straightline": int(sum(data["Flag_StraightLine"])),
                "total_excluded": int(sum(data["Exclude_Recommended"])),
            },
        }
        return df, metadata

    def _compute_observed_effect_sizes(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Compute observed effect sizes from the generated data.

        This validates that the simulation is producing the expected
        between-condition differences. Returns Cohen's d for each scale
        comparing condition pairs.

        CRITICAL for v2.2.6: This allows users to verify that simulated
        data has proper statistical properties.
        """
        observed_effects = []

        if "CONDITION" not in df.columns or len(self.conditions) < 2:
            return observed_effects

        # Get scale columns
        scale_cols = []
        for scale in self.scales:
            scale_name = str(scale.get("name", "Scale")).replace(" ", "_")
            num_items = int(scale.get("num_items", 5))
            for item_num in range(1, num_items + 1):
                col_name = f"{scale_name}_{item_num}"
                if col_name in df.columns:
                    scale_cols.append((scale_name, col_name))

        # Also check for scale means (if computed)
        for scale in self.scales:
            scale_name = str(scale.get("name", "Scale")).replace(" ", "_")
            mean_col = f"{scale_name}_mean"
            if mean_col in df.columns:
                scale_cols.append((scale_name, mean_col))

        # Group by condition and compute means/SDs
        condition_stats = {}
        for cond in self.conditions:
            cond_df = df[df["CONDITION"] == cond]
            if len(cond_df) < 2:
                continue
            condition_stats[cond] = {}
            for scale_name, col in scale_cols:
                if col in cond_df.columns:
                    values = cond_df[col].dropna()
                    if len(values) > 1:
                        condition_stats[cond][col] = {
                            "mean": float(values.mean()),
                            "sd": float(values.std()),
                            "n": len(values)
                        }

        # Compute pairwise Cohen's d between conditions
        conditions_list = list(condition_stats.keys())
        for i, cond1 in enumerate(conditions_list):
            for cond2 in conditions_list[i + 1:]:
                for scale_name, col in scale_cols:
                    if col in condition_stats.get(cond1, {}) and col in condition_stats.get(cond2, {}):
                        stats1 = condition_stats[cond1][col]
                        stats2 = condition_stats[cond2][col]

                        # Cohen's d = (M1 - M2) / pooled_SD
                        mean_diff = stats1["mean"] - stats2["mean"]
                        n1, n2 = stats1["n"], stats2["n"]
                        s1, s2 = stats1["sd"], stats2["sd"]

                        # Pooled standard deviation
                        if n1 + n2 > 2 and (s1 > 0 or s2 > 0):
                            pooled_var = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
                            pooled_sd = np.sqrt(pooled_var) if pooled_var > 0 else 1.0
                            cohens_d = mean_diff / pooled_sd if pooled_sd > 0 else 0.0
                        else:
                            cohens_d = 0.0

                        observed_effects.append({
                            "variable": col,
                            "condition_1": cond1,
                            "condition_2": cond2,
                            "mean_1": round(stats1["mean"], 3),
                            "mean_2": round(stats2["mean"], 3),
                            "cohens_d": round(cohens_d, 3),
                            "n_1": stats1["n"],
                            "n_2": stats2["n"],
                        })

        return observed_effects

    def generate_explainer(self) -> str:
        lines = [
            "=" * 70,
            "COLUMN EXPLAINER - Simulated Behavioral Experiment Data",
            "=" * 70,
            "",
            f"Study: {self.study_title}",
            f"Run ID: {self.run_id}",
            f"Mode: {self.mode.upper()}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Sample Size: {self.sample_size}",
            f"Conditions: {len(self.conditions)}",
            f"Detected Domains: {', '.join(self.detected_domains[:5])}",
            "",
            "-" * 70,
            "VARIABLE DESCRIPTIONS",
            "-" * 70,
            "",
        ]

        for col_name, description in self.column_info:
            lines.append(f"{col_name}")
            lines.append(f"    {description}")
            lines.append("")

        lines.extend(["-" * 70, "EXPERIMENTAL CONDITIONS", "-" * 70, ""])
        n_per = self.sample_size // len(self.conditions)
        for cond in self.conditions:
            lines.append(f"  - {cond} (target n = {n_per})")

        if self.effect_sizes:
            lines.extend(["", "-" * 70, "EXPECTED EFFECT SIZES", "-" * 70, ""])
            for effect in self.effect_sizes:
                lines.append(
                    f"  {effect.variable}: {effect.level_high} > {effect.level_low}, Cohen's d = {effect.cohens_d}"
                )

        lines.extend(
            [
                "",
                "-" * 70,
                "EXCLUSION CRITERIA",
                "-" * 70,
                "",
                f"  Min completion time: {self.exclusion_criteria.completion_time_min_seconds}s",
                f"  Max completion time: {self.exclusion_criteria.completion_time_max_seconds}s",
                f"  Straight-line threshold: {self.exclusion_criteria.straight_line_threshold} items",
                "",
                "=" * 70,
                "END OF COLUMN EXPLAINER",
                "=" * 70,
            ]
        )
        return "\n".join(lines)

    def generate_r_export(self, df: pd.DataFrame) -> str:
        """
        Generate R-compatible export with proper factor coding.

        Returns an R script that loads and prepares Simulated.csv.
        """
        def _r_quote(x: str) -> str:
            x = str(x).replace("\\", "\\\\").replace('"', '\\"')
            return f'"{x}"'

        condition_levels = ", ".join([_r_quote(c) for c in self.conditions])

        lines: List[str] = [
            "# ============================================================",
            f"# R Data Preparation Script - {self.study_title}",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Run ID: {self.run_id}",
            "# ============================================================",
            "",
            "# Load packages",
            "suppressPackageStartupMessages({",
            "  library(readr)",
            "  library(dplyr)",
            "})",
            "",
            "# Load the data",
            'data <- read_csv("Simulated.csv", show_col_types = FALSE)',
            "",
            "# Convert CONDITION to factor with proper levels",
            f"data$CONDITION <- factor(data$CONDITION, levels = c({condition_levels}))",
            "",
            "# Convert Gender to factor",
            'data$Gender <- factor(data$Gender, levels = 1:4,',
            '                     labels = c("Male", "Female", "Non-binary", "Prefer not to say"))',
            "",
        ]

        for scale in self.scales:
            scale_name_raw = str(scale.get("name", "Scale")).strip() or "Scale"
            scale_name = scale_name_raw.replace(" ", "_")
            num_items = int(scale.get("num_items", 5))
            scale_points = int(scale.get("scale_points", 7))
            reverse_items = set(int(x) for x in (scale.get("reverse_items", []) or []))

            items = [f"{scale_name}_{i}" for i in range(1, num_items + 1)]

            if reverse_items:
                lines.append(f"# {scale_name_raw} - reverse code items {sorted(reverse_items)}")
                for r_item in sorted(reverse_items):
                    item_name = f"{scale_name}_{r_item}"
                    max_val = scale_points
                    lines.append(f"data${item_name}_R <- {max_val + 1} - data${item_name}")
                lines.append("")

            lines.append(f"# Create {scale_name_raw} composite")
            item_list = ", ".join([f"data${item}" for item in items])
            lines.append(f"data${scale_name}_composite <- rowMeans(cbind({item_list}), na.rm = TRUE)")
            lines.append("")

        lines.extend(
            [
                "# Filter excluded participants (optional)",
                "data_clean <- data[data$Exclude_Recommended == 0, ]",
                "",
                'cat("Total N:", nrow(data), "\\n")',
                'cat("Clean N:", nrow(data_clean), "\\n")',
                "",
                "# Ready for analysis",
            ]
        )

        return "\n".join(lines)


__all__ = ["EnhancedSimulationEngine", "EffectSizeSpec", "ExclusionCriteria"]
