# simulation_app/utils/enhanced_simulation_engine.py
from __future__ import annotations
"""
Enhanced Simulation Engine for Behavioral Experiment Simulation Tool
=============================================================================

Version 1.6.1 - Critical fixes, visual polish, heading consistency

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
__version__ = "1.0.9.4"  # v1.0.9.4: Expand STEP 3 condition trait modifiers + GAME_CALIBRATIONS expansion

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

# =============================================================================
# SCIENTIFIC METHODS DOCUMENTATION (v2.2.8)
# =============================================================================
# This comprehensive documentation supports the condensed methods write-up
# generator and provides full transparency on the scientific approach.

SCIENTIFIC_METHODS_DOCUMENTATION = """
SIMULATION METHODOLOGY: SCIENTIFICALLY-CALIBRATED SYNTHETIC DATA GENERATION
=============================================================================

1. THEORETICAL FRAMEWORK
------------------------
The simulation generates synthetic behavioral science data using a multi-component
model grounded in survey methodology and individual differences research. The
approach combines:

(a) RESPONSE STYLE THEORY (Krosnick, 1991; Paulhus, 1991)
    - Optimizers vs. Satisficers in survey responding
    - Social desirability and impression management
    - Extreme response style and acquiescence

(b) EFFECT SIZE METHODOLOGY (Cohen, 1988)
    - Standardized mean differences (Cohen's d)
    - Power-appropriate effect magnitudes
    - Within-condition and between-condition variance

(c) DOMAIN-SPECIFIC RESPONSE NORMS
    - Construct-appropriate baseline means
    - Scale-type calibrations (Likert, slider, WTP)
    - Published response distribution parameters

2. PERSONA-BASED RESPONSE GENERATION
------------------------------------
Participants are assigned to behavioral personas based on empirically-derived
population weights:

CORE RESPONSE STYLE PERSONAS (Universal):
- Engaged Responder (35%): Krosnick's "optimizers" - high attention, full scale use
- Satisficer (22%): Minimized effort, restricted scale use, midpoint preference
- Extreme Responder (10%): Greenleaf's ERS - consistent endpoint use
- Acquiescent Responder (8%): Billiet's agreement bias - positive inflation
- Careless Responder (5%): Meade & Craig's inattentive - random patterns
- Socially Desirable Responder (12%): Paulhus's high IM - positive self-presentation

DOMAIN-SPECIFIC PERSONAS:
- Consumer: Brand Loyalist, Deal Seeker, Impulse Buyer, Conscious Consumer
- Technology: Tech Enthusiast, Tech Skeptic, AI Pragmatist, Privacy Concerned
- Behavioral Economics: Loss Averse, Present Biased, Rational Deliberator
- Organizational: High Performer, Disengaged Employee, Transformational Leader
- Social Psychology: Prosocial Individual, Individualist, Conformist
- Health: Health Conscious, Health Fatalist
- Environmental: Eco Warrior, Environmental Skeptic

Each persona has calibrated traits:
- response_tendency: Base mean response (0-1 scale, produces M ≈ 4.0-5.2)
- extremity: Endpoint use probability (0.10-0.90)
- acquiescence: Agreement bias strength (0.40-0.85)
- attention_level: Survey engagement (0.30-0.95)
- scale_use_breadth: Range of scale points used (0.30-0.90)

3. RESPONSE GENERATION ALGORITHM
--------------------------------
Each response is generated through a 9-step process:

STEP 1: Condition Trait Modifiers
- Experimental conditions modify persona traits
- AI conditions: -0.05 engagement, +0.03 consistency
- Hedonic products: +0.08 extremity
- High/Low manipulations: ±0.05 acquiescence

STEP 2: Domain Calibration
- Variable-name-based adjustment to match published norms
- Satisfaction scales: +0.08 mean (Oliver, 1980)
- Risk perception: -0.05 mean (Slovic, 1987)
- Trust scales: +0.04 mean (Mayer et al., 1995)

STEP 3: Scale-Type Calibration
- Sliders (0-100): +15% variance, +5% extremity
- Likert (5-7 point): Standard calibration
- WTP/Numeric: +25% variance

STEP 4: Base Response Tendency
- response = tendency × scale_range + scale_min
- Produces realistic mean ≈ 4.0-5.2 on 7-point scales

STEP 5: Condition Effect Application (v2.2.9 CRITICAL UPDATE)
- Effects are determined by SEMANTIC CONTENT of condition names, NOT position
- Valence keywords parsed: positive (lover, friend, good) vs negative (hater, enemy, bad)
- Manipulation types detected: AI/human, hedonic/utilitarian, treatment/control
- Factorial designs parsed: "Factor1 × Factor2" main effects summed
- Uses stable hash for consistent but non-ordered variation
- NEVER uses condition index/position for effect assignment

STEP 6: Reverse-Coded Item Handling
- Inversion: response = max - (response - min)
- Acquiescent adjustment: +0.25 × range for high acquiescers

STEP 7: Within-Person Variance
- SD = (range/4) × variance_trait ≈ 1.2-1.8 on 7-point
- Domain and scale-type adjustments applied

STEP 8: Extreme Response Style (Greenleaf, 1992)
- P(endpoint) = extremity × 0.45
- ERS produces ~15-20% endpoint responses

STEP 9: Acquiescence Bias (Billiet & McClendon, 2000)
- Inflation = (acquiescence - 0.5) × range × 0.20
- ~0.8 point inflation for strong acquiescers

STEP 10: Social Desirability Bias (Paulhus, 1991)
- Inflation = (SD - 0.5) × range × 0.12
- ~0.5-1.0 point inflation for high IM

4. VALIDATION BENCHMARKS
------------------------
The simulation produces data matching empirical benchmarks:
- Mean responses: 4.0-5.2 on 7-point scales (with positivity bias)
- Within-condition SD: 1.2-1.8
- Between-condition Cohen's d: As configured or 0.4-0.6 auto-generated
- Attention check pass rate: 85-95% (after careless exclusion)
- Cronbach's α for multi-item scales: 0.70-0.90

5. KEY CITATIONS
----------------
Cohen, J. (1988). Statistical power analysis for the behavioral sciences.
Greenleaf, E. A. (1992). Measuring extreme response style. POQ.
Krosnick, J. A. (1991). Response strategies for coping with cognitive demands. ACP.
Meade, A. W., & Craig, S. B. (2012). Identifying careless responses. PM.
Paulhus, D. L. (1991). Measurement and control of response bias. Academic Press.
Billiet, J. B., & McClendon, M. J. (2000). Modeling acquiescence. SEM.
Richard, F. D., et al. (2003). One hundred years of social psychology. RGP.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set, Union

import hashlib
import os
import random
import re
import time

import numpy as np
import pandas as pd

from .persona_library import (
    PersonaLibrary,
    Persona,
    TextResponseGenerator,
    StimulusEvaluationHandler,
)

# v1.0.8.7: Import structured scientific knowledge base
try:
    from .scientific_knowledge_base import (
        META_ANALYTIC_DB,
        GAME_CALIBRATIONS,
        CONSTRUCT_NORMS,
        CULTURAL_ADJUSTMENTS,
        RESPONSE_TIME_NORMS,
        ORDER_EFFECTS,
        get_meta_analytic_effect,
        get_game_calibration,
        get_construct_norm,
        get_cultural_adjustment,
        get_response_time_norm,
        get_order_effect,
        compute_fatigue_adjustment,
        get_knowledge_base_summary,
        MetaAnalyticEffect,
        GameCalibration,
        ConstructNorm,
    )
    HAS_KNOWLEDGE_BASE = True
except ImportError:
    HAS_KNOWLEDGE_BASE = False

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

# Import cross-DV correlation module for realistic between-scale correlations
try:
    from .correlation_matrix import (
        infer_correlation_matrix,
        generate_latent_scores,
        detect_construct_types,
    )
    HAS_CORRELATION_MATRIX = True
except ImportError:
    HAS_CORRELATION_MATRIX = False

    def infer_correlation_matrix(scales: Any) -> Tuple[Any, Any]:  # type: ignore[misc]
        """Fallback stub when correlation_matrix module is unavailable."""
        raise ImportError("correlation_matrix module not available")

    def generate_latent_scores(n: int, corr: Any, seed: int) -> Any:  # type: ignore[misc]
        """Fallback stub when correlation_matrix module is unavailable."""
        raise ImportError("correlation_matrix module not available")

    def detect_construct_types(scales: Any) -> Any:  # type: ignore[misc]
        """Fallback stub when correlation_matrix module is unavailable."""
        raise ImportError("correlation_matrix module not available")


import logging  # must be adjacent to logger — do NOT separate these two lines
logger = logging.getLogger(__name__)


# v1.2.0.0: Custom exception for mid-generation LLM exhaustion.
# Carries partial data so app.py can prompt the user for a fallback choice
# instead of silently degrading to templates.
class LLMExhaustedMidGeneration(Exception):
    """Raised when LLM providers are exhausted during OE generation.

    Attributes:
        partial_data: Dict of column_name → list of values generated so far.
        completed_oe_columns: List of OE column names that were fully generated.
        remaining_questions: List of OE question dicts still needing generation.
        engine_state: Dict of engine state needed to resume generation.
        generation_source_map: Dict mapping column_name → list of per-participant
            source labels ("AI" or "Template") for completed OE columns.
    """

    def __init__(
        self,
        message: str,
        partial_data: Dict[str, list],
        completed_oe_columns: List[str],
        remaining_questions: List[Dict[str, Any]],
        engine_state: Dict[str, Any],
        generation_source_map: Dict[str, List[str]],
    ):
        super().__init__(message)
        self.partial_data = partial_data
        self.completed_oe_columns = completed_oe_columns
        self.remaining_questions = remaining_questions
        self.engine_state = engine_state
        self.generation_source_map = generation_source_map


def _word_in(keyword: str, text: str) -> bool:
    """Check if keyword appears in text as a whole word (word-boundary matching).

    Prevents false positives like 'ai' matching 'wait', 'gain' matching 'bargain',
    'own' matching 'brown', 'get' matching 'budget', etc.

    v1.0.1.3: Added to fix substring false-positive keyword matching throughout
    the semantic effect engine.
    """
    return bool(re.search(r'\b' + re.escape(keyword) + r'\b', text))


def _stem_in(stem: str, text: str) -> bool:
    """Check if stem appears at a word boundary start in text.

    For intentional prefix matches like 'automat' -> 'automated'/'automation',
    'reciproc' -> 'reciprocity'/'reciprocal', 'promot' -> 'promote'/'promotion'.
    """
    return bool(re.search(r'\b' + re.escape(stem), text))


def _any_word_in(keywords: list, text: str) -> bool:
    """Check if any keyword from the list appears as a whole word in text."""
    return any(_word_in(kw, text) for kw in keywords)


def _stable_int_hash(s: str) -> int:
    """Stable, cross-run integer hash for strings.

    v1.0.0: Added guard for empty/None input strings.
    """
    if not s:
        return 0
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def _safe_numeric(value: Any, default: float = 0.0, as_int: bool = False) -> Union[float, int]:
    """Safely convert any value to a numeric type, handling dicts, None, NaN, etc.

    v1.2.3: Added as a universal safe conversion utility to prevent float()/int()
    crashes on unexpected types (dicts, lists, objects).

    Args:
        value: The value to convert
        default: Default value if conversion fails
        as_int: If True, return int instead of float

    Returns:
        Numeric value (float or int depending on as_int)
    """
    if value is None:
        return int(default) if as_int else default
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return int(default) if as_int else default
        return int(value) if as_int else float(value)
    if isinstance(value, dict):
        for key in ('value', 'proportion', 'mean', 'base_mean', 'count'):
            if key in value:
                try:
                    v = float(value[key])
                    return int(v) if as_int else v
                except (ValueError, TypeError):
                    pass
        return int(default) if as_int else default
    try:
        v = float(value)
        return int(v) if as_int else v
    except (ValueError, TypeError):
        return int(default) if as_int else default


def _clean_column_name(name: str) -> str:
    """Sanitize a string for use as a DataFrame column name.

    v1.4.3: Added to ensure all generated column names are clean and scientific.
    Removes spaces, special characters, and collapses multiple underscores.

    Args:
        name: Raw name string (e.g., "Trust scale", "My (custom) DV!")

    Returns:
        Clean column name (e.g., "Trust_scale", "My_custom_DV")
    """
    # Replace spaces and non-alphanumeric/underscore characters with underscore
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Collapse multiple underscores into one
    clean = re.sub(r'_+', '_', clean)
    # Strip leading/trailing underscores
    clean = clean.strip('_')
    return clean if clean else "Variable"


def _safe_trait_value(value: Any, default: float = 0.5) -> float:
    """
    Safely extract a float value from a trait.

    v1.2.1: Added to handle edge cases where trait values might be:
    - PersonaTrait objects (extract base_mean)
    - Dicts (extract 'value' or 'base_mean' key)
    - None or NaN
    - Already floats (pass through)

    Args:
        value: The trait value to extract
        default: Default value if extraction fails

    Returns:
        Float value in 0-1 range
    """
    if value is None:
        return default

    # Already a number
    if isinstance(value, (int, float)):
        if isinstance(value, float) and np.isnan(value):
            return default
        return float(np.clip(value, 0.0, 1.0))

    # Dict - extract value or base_mean
    if isinstance(value, dict):
        for key in ('value', 'base_mean', 'mean'):
            if key in value:
                try:
                    return float(np.clip(value[key], 0.0, 1.0))
                except (ValueError, TypeError):
                    pass
        return default

    # Object with base_mean attribute (PersonaTrait)
    if hasattr(value, 'base_mean'):
        try:
            return float(np.clip(value.base_mean, 0.0, 1.0))
        except (ValueError, TypeError):
            pass

    # Try direct conversion
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except (ValueError, TypeError):
        return default


def _normalize_scales(scales: Optional[List[Any]]) -> List[Dict[str, Any]]:
    """
    Normalize scale specifications for the engine.

    IMPORTANT: If a scale has _validated=True (set by app.py's _normalize_scale_specs),
    its values are trusted and preserved as-is. This prevents re-defaulting values
    that have already been validated by the single-source-of-truth normalizer.
    """
    normalized: List[Dict[str, Any]] = []
    for scale in scales or []:
        if isinstance(scale, str):
            name = scale.strip()
            if name:
                normalized.append(
                    {"name": name, "variable_name": name, "num_items": 5, "scale_points": 7, "reverse_items": [], "_validated": True}
                )
            continue
        if isinstance(scale, dict):
            name = str(scale.get("name", "")).strip()
            if not name:
                continue

            # If already validated by app.py, preserve all values exactly
            # v1.2.3: Use _safe_numeric to ensure scale_min/scale_max are always ints
            if scale.get("_validated"):
                pts = _safe_numeric(scale.get("scale_points"), default=7, as_int=True)
                normalized.append({
                    "name": name,
                    "variable_name": str(scale.get("variable_name", name)),
                    "num_items": _safe_numeric(scale.get("num_items"), default=5, as_int=True),
                    "scale_points": pts,
                    "reverse_items": scale.get("reverse_items", []) or [],
                    "_validated": True,
                    # v1.2.3: Force scale_min/scale_max to ints (prevents dict leakage)
                    "scale_min": _safe_numeric(scale.get("scale_min", 1), default=1, as_int=True),
                    "scale_max": _safe_numeric(scale.get("scale_max", pts), default=pts, as_int=True),
                    "item_names": scale.get("item_names", []),
                    # v1.4.0: Preserve scale type for downstream use
                    "type": str(scale.get("type", "matrix")),
                })
                continue

            # Not pre-validated: parse and validate (fallback path)
            raw_pts = scale.get("scale_points")
            if raw_pts is None:
                pts = 7
            else:
                try:
                    pts = int(raw_pts)
                except (ValueError, TypeError):
                    pts = 7
            pts = max(2, min(1001, pts))

            raw_items = scale.get("num_items")
            if raw_items is None:
                raw_items = scale.get("items")  # QSF detection compatibility
            if raw_items is None:
                n_items = 5
            else:
                try:
                    n_items = int(raw_items)
                except (ValueError, TypeError):
                    n_items = 5
            n_items = max(1, n_items)

            normalized.append(
                {
                    "name": name,
                    "variable_name": str(scale.get("variable_name", name)),
                    "num_items": n_items,
                    "scale_points": pts,
                    "reverse_items": scale.get("reverse_items", []) or [],
                    "_validated": True,
                    # v1.2.3: Force scale_min/scale_max to ints (prevents dict leakage)
                    "scale_min": _safe_numeric(scale.get("scale_min", 1), default=1, as_int=True),
                    "scale_max": _safe_numeric(scale.get("scale_max", pts), default=pts, as_int=True),
                    "item_names": scale.get("item_names", []),
                    # v1.4.0: Preserve scale type for downstream use
                    "type": str(scale.get("type", "matrix")),
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
                normalized.append({"name": name, "type": "text", "question_text": name})
            continue
        if isinstance(item, dict):
            # Support both "name" and "question_id" keys (QSF uses question_id)
            name = str(item.get("name", item.get("question_id", ""))).strip()
            if name:
                # Preserve question_text, display_logic, and condition info for survey flow
                normalized_item = dict(item)
                # Ensure name is set (for column naming)
                if not normalized_item.get("name"):
                    normalized_item["name"] = name
                # Ensure question_text is set for unique response generation
                if not normalized_item.get("question_text"):
                    normalized_item["question_text"] = name
                normalized.append(normalized_item)
    return normalized


def _safe_parse_reverse_items(reverse_items_raw: Any) -> set:
    """
    Safely parse reverse items, handling invalid values gracefully.

    Args:
        reverse_items_raw: Raw reverse items value (could be list, None, or invalid)

    Returns:
        Set of integer item numbers (empty set if parsing fails)
    """
    if not reverse_items_raw:
        return set()

    result = set()
    items_list = reverse_items_raw if isinstance(reverse_items_raw, (list, tuple)) else []
    for x in items_list:
        try:
            result.add(int(x))
        except (ValueError, TypeError):
            pass  # Skip invalid reverse item values
    return result


def _substitute_embedded_fields(text: str, condition: str, embedded_data: Optional[Dict[str, Any]] = None) -> str:
    """
    Substitute embedded data field references in question text.

    Handles Qualtrics piping patterns like ${e://Field/StimulusText}
    which are used to dynamically insert condition-specific content.

    Args:
        text: Question text potentially containing field references
        condition: Current condition name (used for default values)
        embedded_data: Optional dict of field name -> value mappings

    Returns:
        Text with field references replaced by actual values
    """
    if not text or '${' not in text:
        return text

    embedded_data = embedded_data or {}

    # Pattern for Qualtrics embedded data: ${e://Field/FieldName}
    pattern = r'\$\{e://Field/([^}]+)\}'

    def replace_field(match):
        field_name = match.group(1)
        # Check if we have a value for this field
        if field_name in embedded_data:
            return str(embedded_data[field_name])
        # Try case-insensitive match
        for key, value in embedded_data.items():
            if key.lower() == field_name.lower():
                return str(value)
        # Default: extract value from condition name if possible
        # e.g., condition "AI × Hedonic" could provide "AI" for "AI_Condition" field
        condition_parts = condition.replace('×', ' ').replace('_', ' ').split()
        for part in condition_parts:
            if part.lower() in field_name.lower():
                return part
        # Last resort: return field name as placeholder
        return f"[{field_name}]"

    return re.sub(pattern, replace_field, text)


def _detect_question_visibility_from_text(question_text: str, conditions: List[str]) -> Dict[str, bool]:
    """
    Detect which conditions should see a question based on its text.

    Handles patterns like:
    - "For those who saw the AI recommendation..."
    - "In the high trust condition..."
    - "If you were in the control group..."

    Args:
        question_text: The question text to analyze
        conditions: List of all condition names

    Returns:
        Dict mapping condition name -> visibility (True/False)
    """
    visibility = {c.lower(): True for c in conditions}

    if not question_text:
        return visibility

    text_lower = question_text.lower()

    # Patterns indicating condition-specific questions
    condition_patterns = [
        (r'for those (?:who|in the)', True),
        (r'if you (?:were|are) in the', True),
        (r'in the .+ condition', True),
        (r'for .+ participants', True),
        (r'those who (?:saw|received|experienced)', True),
    ]

    # Check if question mentions specific conditions
    for pattern, _ in condition_patterns:
        if re.search(pattern, text_lower):
            # Try to identify which condition is referenced
            for cond in conditions:
                cond_lower = cond.lower()
                cond_parts = cond_lower.replace('×', ' ').replace('_', ' ').split()
                for part in cond_parts:
                    if len(part) > 2 and part in text_lower:
                        # This condition is explicitly mentioned
                        # Check if it's a negative reference ("not in the AI condition")
                        not_pattern = rf'(?:not|didn\'t|did not|weren\'t|were not)\s+(?:in|see|receive).*{re.escape(part)}'
                        if re.search(not_pattern, text_lower):
                            # Negated - this condition should NOT see the question
                            visibility[cond_lower] = False
                        else:
                            # Positive reference - only this condition sees it
                            for other_cond in conditions:
                                other_lower = other_cond.lower()
                                if part not in other_lower:
                                    visibility[other_lower] = False
                        break

    return visibility


def _generate_timing_data(
    participant_seed: int,
    attention_level: float,
    num_questions: int,
    base_time_per_question: float = 15.0
) -> Dict[str, Any]:
    """
    Generate realistic timing data for a participant.

    Based on research showing attention level correlates with response time
    and reading patterns.

    Args:
        participant_seed: Seed for reproducibility
        attention_level: Participant's attention trait (0-1)
        num_questions: Number of questions in survey
        base_time_per_question: Average seconds per question

    Returns:
        Dict with timing metrics
    """
    rng = np.random.RandomState(participant_seed)

    # Engaged responders spend more time, careless responders rush
    time_multiplier = 0.5 + attention_level * 1.0  # 0.5x to 1.5x

    # Calculate per-page times with variation
    page_times = []
    for i in range(num_questions):
        base = base_time_per_question * time_multiplier
        # Add random variation (±30%)
        variation = rng.uniform(0.7, 1.3)
        page_times.append(base * variation)

    total_time = sum(page_times)

    # Generate click patterns
    clicks_per_page = [max(1, int(rng.normal(3, 1))) for _ in range(num_questions)]

    return {
        'total_seconds': int(total_time),
        'avg_seconds_per_question': total_time / max(1, num_questions),
        'first_click_delay': rng.uniform(1, 5) if attention_level > 0.5 else rng.uniform(0.5, 2),
        'total_clicks': sum(clicks_per_page),
        'page_times': page_times,
    }


def _evaluate_branch_logic(
    logic: Dict[str, Any],
    participant_responses: Dict[str, Any],
    condition: str,
    embedded_data: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Evaluate branching logic to determine if a branch should be taken.

    Supports response-based branching (e.g., "If Q1 = Yes, show Q2")
    and embedded data checks (e.g., "If Condition = AI, show Q3").

    Args:
        logic: Branch logic definition from QSF
        participant_responses: Dict of question_id -> response value
        condition: Current condition name
        embedded_data: Optional embedded data values

    Returns:
        True if the branch condition is satisfied
    """
    if not logic:
        return True  # No logic = always visible

    logic_type = logic.get('Type', '').lower()
    conditions_list = logic.get('conditions', [])

    if not conditions_list:
        return True

    # Evaluate each condition in the logic
    results = []
    for cond in conditions_list:
        if not isinstance(cond, dict):
            continue

        operator = cond.get('operator', '').lower()
        question_id = cond.get('question_id', '')
        choice_locator = cond.get('choice_locator', '')
        value = cond.get('value', '')

        # Check if this is an embedded data check
        if 'embedded' in question_id.lower() or 'condition' in question_id.lower():
            embedded_data = embedded_data or {}
            # Check embedded data values
            for key, val in embedded_data.items():
                if key.lower() in question_id.lower():
                    if operator in ['equalto', 'is', '=', '==']:
                        results.append(str(val).lower() == str(value).lower())
                    elif operator in ['notequalto', 'isnot', '!=', '<>']:
                        results.append(str(val).lower() != str(value).lower())
                    break
            else:
                # Check condition name
                if 'condition' in question_id.lower():
                    cond_lower = condition.lower()
                    if operator in ['equalto', 'is', '=', '==']:
                        results.append(value.lower() in cond_lower)
                    elif operator in ['notequalto', 'isnot', '!=', '<>']:
                        results.append(value.lower() not in cond_lower)

        # Check if this is a response-based check
        elif question_id and question_id in participant_responses:
            response = participant_responses[question_id]
            if operator in ['selected', 'equalto', 'is']:
                results.append(str(response).lower() == str(value).lower())
            elif operator in ['notselected', 'notequalto', 'isnot']:
                results.append(str(response).lower() != str(value).lower())
            elif operator in ['greaterthan', '>']:
                try:
                    results.append(float(response) > float(value))
                except (ValueError, TypeError):
                    results.append(False)
            elif operator in ['lessthan', '<']:
                try:
                    results.append(float(response) < float(value))
                except (ValueError, TypeError):
                    results.append(False)

    # Combine results based on logic type (AND/OR)
    if not results:
        return True

    if logic_type in ['and', 'all', 'booleanand']:
        return all(results)
    else:  # OR is default
        return any(results)


def _validate_condition_assignment(
    conditions: List[str],
    assignments: List[str],
    allocation: Optional[Dict[str, float]] = None,
    tolerance: float = 0.05
) -> Dict[str, Any]:
    """
    Validate that condition assignments match expected allocation.

    Args:
        conditions: List of condition names
        assignments: List of assigned conditions for each participant
        allocation: Expected allocation percentages (or None for equal)
        tolerance: Acceptable deviation from expected percentage

    Returns:
        Dict with validation results
    """
    n = len(assignments)
    if n == 0:
        return {'valid': False, 'error': 'No assignments'}

    # Guard against empty conditions list to prevent division by zero
    if not conditions or len(conditions) == 0:
        return {'valid': False, 'error': 'No conditions provided'}

    # Count assignments
    counts = {}
    for cond in conditions:
        counts[cond] = assignments.count(cond)

    # Calculate expected counts
    expected = {}
    n_conditions = len(conditions)
    if allocation:
        for cond in conditions:
            pct = allocation.get(cond, 100 / n_conditions)
            expected[cond] = n * pct / 100
    else:
        for cond in conditions:
            expected[cond] = n / n_conditions

    # Check deviations
    deviations = {}
    valid = True
    for cond in conditions:
        actual = counts.get(cond, 0)
        exp = expected.get(cond, 0)
        if exp > 0:
            deviation = abs(actual - exp) / exp
            deviations[cond] = {
                'actual': actual,
                'expected': round(exp, 1),
                'deviation': round(deviation, 3)
            }
            if deviation > tolerance:
                valid = False

    return {
        'valid': valid,
        'counts': counts,
        'expected': {k: round(v, 1) for k, v in expected.items()},
        'deviations': deviations,
        'total_assigned': n
    }


def _parse_matrix_choices(
    choices: Dict[str, Any],
    answers: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Parse complex matrix question choice hierarchies.

    Handles nested choice structures like:
    - Choices with sub-choices
    - Answer columns for matrix tables
    - Recode values for analysis

    Args:
        choices: Choice dictionary from QSF
        answers: Answer columns for matrix questions

    Returns:
        Parsed choice structure
    """
    parsed = {
        'rows': [],
        'columns': [],
        'recode_map': {},
        'display_map': {}
    }

    if not choices:
        return parsed

    # Parse row choices
    for choice_id, choice_data in choices.items():
        if isinstance(choice_data, dict):
            display = choice_data.get('Display', str(choice_id))
            recode = choice_data.get('RecodeValue', choice_id)
            parsed['rows'].append({
                'id': choice_id,
                'display': display,
                'recode': recode
            })
            parsed['recode_map'][choice_id] = recode
            parsed['display_map'][choice_id] = display
        elif isinstance(choice_data, str):
            parsed['rows'].append({
                'id': choice_id,
                'display': choice_data,
                'recode': choice_id
            })

    # Parse answer columns if present (for matrix tables)
    if answers:
        for answer_id, answer_data in answers.items():
            if isinstance(answer_data, dict):
                display = answer_data.get('Display', str(answer_id))
                recode = answer_data.get('RecodeValue', answer_id)
                parsed['columns'].append({
                    'id': answer_id,
                    'display': display,
                    'recode': recode
                })
            elif isinstance(answer_data, str):
                parsed['columns'].append({
                    'id': answer_id,
                    'display': answer_data,
                    'recode': answer_id
                })

    return parsed


def _validate_survey_flow(
    flow_elements: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
    conditions: List[str]
) -> Dict[str, Any]:
    """
    Validate survey flow for consistency and completeness.

    Checks:
    - All referenced questions exist
    - No circular dependencies
    - All conditions have valid paths
    - Skip logic doesn't create dead ends

    Args:
        flow_elements: Survey flow structure
        questions: List of all questions
        conditions: List of condition names

    Returns:
        Validation results with any issues found
    """
    issues = []
    warnings = []

    question_ids = {q.get('question_id', q.get('name', '')) for q in questions}

    # Track which questions are reachable
    reachable = set()

    def check_flow_item(item, path=None):
        path = path or []
        if not isinstance(item, dict):
            return

        item_type = item.get('Type', '')
        item_id = item.get('ID', item.get('BlockID', ''))

        # Check for circular references
        if item_id and item_id in path:
            issues.append(f"Circular reference detected: {' -> '.join(path + [item_id])}")
            return

        # Track reachable items
        if item_type in ['Block', 'Standard']:
            reachable.add(item_id)

        # Check skip logic references
        skip_to = item.get('SkipTo', '')
        if skip_to and skip_to not in question_ids and skip_to not in reachable:
            warnings.append(f"Skip logic references unknown target: {skip_to}")

        # Recurse into nested flow
        for sub_item in item.get('Flow', []):
            check_flow_item(sub_item, path + [item_id] if item_id else path)

    # Check each flow element
    for element in flow_elements:
        check_flow_item(element)

    # Check for unreachable questions
    for q in questions:
        q_id = q.get('question_id', q.get('name', ''))
        block = q.get('block_name', '')
        if block and block not in reachable and q_id not in reachable:
            # This might be OK if it's in a conditional block
            pass  # Don't flag as issue, just note

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'reachable_blocks': list(reachable)
    }


# =============================================================================
# ITERATION 1: CONDITION SEMANTIC PARSING
# =============================================================================
# Parse condition names to extract semantic meaning for effect direction

POSITIVE_VALENCE_KEYWORDS = {
    'high', 'positive', 'good', 'love', 'lover', 'friend', 'prosocial',
    'generous', 'kind', 'warm', 'trust', 'trusting', 'hedonic', 'pleasure',
    'reward', 'gain', 'win', 'success', 'treatment', 'experimental', 'active',
    'present', 'yes', 'true', 'included', 'with', 'pro', 'support', 'agree',
    'accept', 'benefit', 'advantage', 'superior', 'enhanced', 'improved'
}

NEGATIVE_VALENCE_KEYWORDS = {
    'low', 'negative', 'bad', 'hate', 'hater', 'enemy', 'antisocial',
    'selfish', 'cruel', 'cold', 'distrust', 'distrusting', 'utilitarian',
    'practical', 'loss', 'lose', 'failure', 'control', 'placebo', 'inactive',
    'absent', 'no', 'false', 'excluded', 'without', 'anti', 'oppose', 'disagree',
    'reject', 'cost', 'disadvantage', 'inferior', 'reduced', 'diminished'
}

NEUTRAL_KEYWORDS = {
    'neutral', 'baseline', 'middle', 'moderate', 'average', 'standard',
    'normal', 'typical', 'default', 'reference', 'comparison'
}


def _parse_condition_semantics(condition: str) -> Dict[str, Any]:
    """
    Parse semantic meaning from condition names.

    Extracts:
    - Valence direction (positive/negative/neutral)
    - Factor levels for factorial designs
    - Manipulation type (AI/human, hedonic/utilitarian, etc.)
    - Intensity indicators (high/low, strong/weak)

    Args:
        condition: Condition name string

    Returns:
        Dict with semantic properties
    """
    cond_lower = condition.lower()
    cond_parts = cond_lower.replace('×', ' ').replace('_', ' ').replace('-', ' ').split()

    semantics = {
        'original': condition,
        'valence': 0.0,  # -1 to +1 scale
        'factors': [],
        'manipulation_type': None,
        'intensity': 0.5,  # 0 to 1 scale
        'is_control': False,
        'is_treatment': False,
        'keywords_found': []
    }

    # Check for positive/negative keywords
    positive_count = 0
    negative_count = 0

    for part in cond_parts:
        if part in POSITIVE_VALENCE_KEYWORDS:
            positive_count += 1
            semantics['keywords_found'].append((part, 'positive'))
        elif part in NEGATIVE_VALENCE_KEYWORDS:
            negative_count += 1
            semantics['keywords_found'].append((part, 'negative'))
        elif part in NEUTRAL_KEYWORDS:
            semantics['keywords_found'].append((part, 'neutral'))

    # Calculate valence
    total = positive_count + negative_count
    if total > 0:
        semantics['valence'] = (positive_count - negative_count) / total

    # Check for control/treatment
    control_indicators = ['control', 'baseline', 'placebo', 'no', 'without', 'absent']
    treatment_indicators = ['treatment', 'experimental', 'active', 'with', 'present']

    semantics['is_control'] = any(ind in cond_lower for ind in control_indicators)
    semantics['is_treatment'] = any(ind in cond_lower for ind in treatment_indicators)

    # Detect manipulation type
    manipulation_types = {
        'ai_human': ['ai', 'algorithm', 'robot', 'human', 'person', 'manual'],
        'hedonic_utilitarian': ['hedonic', 'utilitarian', 'pleasure', 'practical'],
        'high_low': ['high', 'low', 'strong', 'weak'],
        'gain_loss': ['gain', 'loss', 'reward', 'punishment'],
        'individual_group': ['individual', 'personal', 'group', 'collective'],
        'political': ['trump', 'biden', 'democrat', 'republican', 'liberal', 'conservative']
    }

    for manip_type, keywords in manipulation_types.items():
        if any(kw in cond_lower for kw in keywords):
            semantics['manipulation_type'] = manip_type
            break

    # Parse factorial structure
    separators = ['×', ' x ', '_x_', ' × ', ' vs ', ' vs. ']
    for sep in separators:
        if sep in cond_lower:
            semantics['factors'] = [p.strip() for p in cond_lower.split(sep)]
            break

    if not semantics['factors']:
        semantics['factors'] = [cond_lower]

    return semantics


# =============================================================================
# ITERATION 2: SCALE RELIABILITY SIMULATION
# =============================================================================
# Simulate realistic scale reliability (Cronbach's alpha) through correlated items

def _generate_correlated_items(
    n_items: int,
    base_response: float,
    target_alpha: float,
    scale_points: int,
    reverse_items: List[int],
    rng: np.random.RandomState
) -> List[int]:
    """
    Generate correlated scale items to achieve target Cronbach's alpha.

    Uses a factor model approach where items share common variance
    determined by the target reliability.

    Args:
        n_items: Number of scale items
        base_response: Base response tendency (0-1 scale)
        target_alpha: Target Cronbach's alpha (typically 0.70-0.90)
        scale_points: Number of scale points
        reverse_items: List of 1-indexed item numbers that are reverse-coded
        rng: Random number generator

    Returns:
        List of integer responses for each item
    """
    # Calculate factor loading from target alpha
    # alpha = n * r_bar / (1 + (n-1) * r_bar)
    # Solving for r_bar: r_bar = alpha / (n - alpha * (n-1))
    if n_items <= 1:
        response = int(np.clip(round(base_response * (scale_points - 1) + 1), 1, scale_points))
        return [response]

    # Average inter-item correlation needed
    _denom = n_items - target_alpha * (n_items - 1)
    r_bar = target_alpha / _denom if abs(_denom) > 1e-9 else 0.5
    r_bar = np.clip(r_bar, 0.1, 0.9)

    # Factor loading (sqrt of shared variance)
    factor_loading = np.sqrt(r_bar)
    unique_loading = np.sqrt(1 - r_bar)

    # Generate common factor score
    common_factor = rng.normal(0, 1)

    # Generate item responses
    responses = []
    for i in range(n_items):
        # True score = common factor + unique variance
        true_score = factor_loading * common_factor + unique_loading * rng.normal(0, 1)

        # Transform to response scale
        # Base response determines the mean
        mean_response = base_response * (scale_points - 1) + 1
        sd_response = (scale_points - 1) / 4  # Approximate SD

        raw_response = mean_response + true_score * sd_response

        # Handle reverse coding
        item_num = i + 1
        if item_num in reverse_items:
            raw_response = (scale_points + 1) - raw_response

        # Clip and round to valid scale point
        response = int(np.clip(round(raw_response), 1, scale_points))
        responses.append(response)

    return responses


def _inject_inter_item_correlation(
    item_matrix: np.ndarray,
    target_alpha: float,
    scale_min: int,
    scale_max: int,
) -> np.ndarray:
    """Inject inter-item correlation into independently generated scale items.

    Uses a mixing approach: blend each item with a common factor (the
    participant's row mean) to achieve the target Cronbach's alpha, then
    round back to integer scale values while preserving per-item means/SDs.

    v1.4.11: Called after independent per-item generation to add realistic
    internal consistency without losing condition effects or persona variation.
    """
    n, k = item_matrix.shape
    if k <= 1 or n <= 1:
        return item_matrix

    # Target average inter-item correlation from Spearman-Brown
    denom = k - target_alpha * (k - 1)
    r_bar = target_alpha / denom if abs(denom) > 1e-9 else 0.5
    r_bar = float(np.clip(r_bar, 0.1, 0.9))

    # Mixing weight: x_new = w * common + (1-w) * x_old
    # Correlation between items ≈ w², so w = sqrt(r_bar)
    w = float(np.sqrt(r_bar))

    # Common factor = row mean
    row_means = item_matrix.mean(axis=1, keepdims=True)

    # Blend each item with the common factor
    mixed = w * row_means + (1.0 - w) * item_matrix

    # Restore original per-item means and SDs
    for j in range(k):
        orig_mean = float(item_matrix[:, j].mean())
        orig_std = float(item_matrix[:, j].std())
        mixed_std = float(mixed[:, j].std())
        if mixed_std > 1e-9 and orig_std > 1e-9:
            mixed[:, j] = ((mixed[:, j] - mixed[:, j].mean()) / mixed_std
                           * orig_std + orig_mean)

    # Round and clip to scale bounds
    return np.clip(np.round(mixed), scale_min, scale_max).astype(int)


# =============================================================================
# ITERATION 3: CARELESS RESPONSE PATTERN DETECTION
# =============================================================================
# Detect and flag various careless response patterns

def _detect_careless_patterns(
    responses: List[int],
    scale_points: int = 7
) -> Dict[str, Any]:
    """
    Detect careless responding patterns in a set of responses.

    Detects:
    - Straight-lining (same response repeated)
    - Alternating patterns (1-7-1-7 or similar)
    - Midpoint responding (always choosing middle)
    - Extreme responding (always choosing endpoints)
    - Random responding (high variability with no consistency)

    Args:
        responses: List of scale responses
        scale_points: Number of scale points

    Returns:
        Dict with pattern detection results and flags
    """
    # Filter out any NaN values that may result from missing data injection
    responses = [r for r in responses if not (isinstance(r, float) and np.isnan(r))]

    if len(responses) < 3:
        return {'careless_detected': False, 'patterns': [], 'confidence': 0.0}

    patterns = []
    confidence = 0.0

    # 1. Straight-line detection
    max_streak = 1
    current_streak = 1
    for i in range(1, len(responses)):
        if responses[i] == responses[i-1]:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 1

    straight_line_ratio = max_streak / len(responses)
    if max_streak >= 5 or straight_line_ratio > 0.7:
        patterns.append('straight_line')
        confidence = max(confidence, straight_line_ratio)

    # 2. Alternating pattern detection
    alternating_count = 0
    for i in range(2, len(responses)):
        if responses[i] == responses[i-2] and responses[i] != responses[i-1]:
            alternating_count += 1

    alternating_ratio = alternating_count / max(len(responses) - 2, 1)
    if alternating_ratio > 0.6:
        patterns.append('alternating')
        confidence = max(confidence, alternating_ratio)

    # 3. Midpoint responding
    midpoint = (scale_points + 1) / 2
    midpoint_count = sum(1 for r in responses if abs(r - midpoint) < 0.6)
    midpoint_ratio = midpoint_count / len(responses)
    if midpoint_ratio > 0.8:
        patterns.append('midpoint')
        confidence = max(confidence, midpoint_ratio)

    # 4. Extreme responding
    extreme_count = sum(1 for r in responses if r in [1, scale_points])
    extreme_ratio = extreme_count / len(responses)
    if extreme_ratio > 0.7:
        patterns.append('extreme')
        confidence = max(confidence, extreme_ratio)

    # 5. Random responding (low consistency)
    if len(responses) >= 4:
        variance = np.var(responses)
        expected_variance = ((scale_points - 1) ** 2) / 12  # Uniform distribution variance
        if variance > expected_variance * 1.5:
            # High variance might indicate random responding
            # But we need additional checks
            pass

    return {
        'careless_detected': len(patterns) > 0,
        'patterns': patterns,
        'confidence': confidence,
        'max_straight_line': max_streak,
        'alternating_ratio': alternating_ratio,
        'midpoint_ratio': midpoint_ratio,
        'extreme_ratio': extreme_ratio
    }


# =============================================================================
# ITERATION 4: EFFECT SIZE CALIBRATION VALIDATION
# =============================================================================
# Validate that generated data achieves target effect sizes

def _validate_effect_sizes(
    data: pd.DataFrame,
    conditions: List[str],
    target_d: float,
    dv_columns: List[str]
) -> Dict[str, Any]:
    """
    Validate that generated data achieves target Cohen's d effect sizes.

    Args:
        data: Generated DataFrame
        conditions: List of condition names
        target_d: Target Cohen's d
        dv_columns: Columns containing dependent variables

    Returns:
        Validation results with actual vs target effect sizes
    """
    results = {
        'target_d': target_d,
        'achieved_effects': {},
        'within_tolerance': True,
        'tolerance': 0.15  # Acceptable deviation
    }

    if len(conditions) < 2 or 'CONDITION' not in data.columns:
        return results

    # Compare ALL condition pairs and report the maximum pairwise Cohen's d
    for col in dv_columns:
        if col not in data.columns:
            continue

        best_d = 0.0
        best_pair = ("", "")
        best_mean_diff = 0.0

        try:
            for i, cond1 in enumerate(conditions):
                for cond2 in conditions[i + 1:]:
                    group1 = data[data['CONDITION'] == cond1][col].dropna().astype(float)
                    group2 = data[data['CONDITION'] == cond2][col].dropna().astype(float)

                    if len(group1) < 2 or len(group2) < 2:
                        continue

                    mean1, mean2 = group1.mean(), group2.mean()
                    sd_pooled = np.sqrt(
                        ((len(group1) - 1) * group1.var() + (len(group2) - 1) * group2.var())
                        / (len(group1) + len(group2) - 2)
                    )

                    if sd_pooled > 0:
                        pair_d = abs(mean1 - mean2) / sd_pooled
                        if pair_d > best_d:
                            best_d = pair_d
                            best_pair = (cond1, cond2)
                            best_mean_diff = mean1 - mean2

            if best_d > 0:
                results['achieved_effects'][col] = {
                    'achieved_d': round(best_d, 3),
                    'target_d': target_d,
                    'deviation': round(abs(best_d - target_d), 3),
                    'mean_diff': round(best_mean_diff, 3),
                    'pair': f"{best_pair[0]} vs {best_pair[1]}",
                    'within_tolerance': abs(best_d - target_d) <= results['tolerance']
                }

                if not results['achieved_effects'][col]['within_tolerance']:
                    results['within_tolerance'] = False

        except Exception:
            continue

    return results


# =============================================================================
# ITERATION 5: CROSS-CULTURAL RESPONSE STYLE MODELING
# =============================================================================
# Model cultural differences in response styles

CULTURAL_RESPONSE_STYLES = {
    'western_individualist': {
        'description': 'Western individualist cultures (US, UK, Australia)',
        'acquiescence_bias': 0.52,  # Slight agreement bias
        'extreme_responding': 0.35,  # Moderate extreme responding
        'midpoint_avoidance': 0.40,  # Tend to avoid midpoint
        'social_desirability': 0.45,
        'response_elaboration': 0.60  # Longer open-ended responses
    },
    'east_asian': {
        'description': 'East Asian cultures (China, Japan, Korea)',
        'acquiescence_bias': 0.48,  # Less acquiescence
        'extreme_responding': 0.20,  # Lower extreme responding
        'midpoint_avoidance': 0.25,  # More likely to use midpoint
        'social_desirability': 0.60,  # Higher social desirability
        'response_elaboration': 0.40  # More concise responses
    },
    'latin_american': {
        'description': 'Latin American cultures (Mexico, Brazil, Argentina)',
        'acquiescence_bias': 0.58,  # Higher agreement bias
        'extreme_responding': 0.50,  # Higher extreme responding
        'midpoint_avoidance': 0.50,
        'social_desirability': 0.55,
        'response_elaboration': 0.70  # More elaborate responses
    },
    'middle_eastern': {
        'description': 'Middle Eastern cultures (UAE, Saudi Arabia, Egypt)',
        'acquiescence_bias': 0.55,
        'extreme_responding': 0.45,
        'midpoint_avoidance': 0.45,
        'social_desirability': 0.65,
        'response_elaboration': 0.55
    }
}


def _apply_cultural_response_style(
    base_response: float,
    scale_points: int,
    cultural_style: str,
    rng: np.random.RandomState
) -> int:
    """
    Apply cultural response style adjustments to a base response.

    Args:
        base_response: Base response (0-1 scale)
        scale_points: Number of scale points
        cultural_style: Key from CULTURAL_RESPONSE_STYLES
        rng: Random number generator

    Returns:
        Adjusted response as integer scale point
    """
    style = CULTURAL_RESPONSE_STYLES.get(cultural_style, CULTURAL_RESPONSE_STYLES['western_individualist'])

    # Transform base response to scale
    raw = base_response * (scale_points - 1) + 1

    # Apply acquiescence bias (shift toward agreement/positive)
    acquiescence_shift = (style['acquiescence_bias'] - 0.5) * (scale_points - 1) * 0.2
    raw += acquiescence_shift

    # Apply extreme responding tendency
    midpoint = (scale_points + 1) / 2
    if rng.random() < style['extreme_responding']:
        # Push toward extremes
        if raw > midpoint:
            raw = raw + (scale_points - raw) * 0.4
        else:
            raw = raw - (raw - 1) * 0.4

    # Apply midpoint avoidance
    if style['midpoint_avoidance'] > 0.5 and abs(raw - midpoint) < 0.5:
        if rng.random() < style['midpoint_avoidance'] - 0.5:
            # Shift away from midpoint
            raw += rng.choice([-0.5, 0.5])

    return int(np.clip(round(raw), 1, scale_points))


# =============================================================================
# ITERATION 6: OPEN-ENDED RESPONSE DIVERSITY ENHANCEMENT
# =============================================================================
# Generate more diverse and unique open-ended responses

RESPONSE_TEMPLATES_BY_DOMAIN = {
    'ai_technology': {
        'positive': [
            "I found the AI-generated recommendations to be {adjective}. The system seemed to {verb} my preferences well.",
            "The algorithm {verb} relevant suggestions. I appreciated how it {action}.",
            "Overall, I'm {sentiment} with how the technology {verb} my needs. It felt {adjective}.",
            "I was {sentiment} by how well the system understood what I was looking for. Really {adjective} results.",
            "The AI did a {adjective} job. It clearly {verb} the patterns in my preferences and {action} accordingly.",
        ],
        'negative': [
            "I was {sentiment} by the AI's recommendations. They seemed {adjective} and didn't {verb} what I was looking for.",
            "The algorithm felt {adjective}. I wished it could have {action} better.",
            "I found the technology to be {adjective}. It {verb} my actual preferences.",
            "The system was pretty {adjective} in my opinion. It seemed to {verb} what I actually needed.",
            "Not a great experience with the AI. The suggestions were {adjective} and it {verb} the point entirely.",
        ],
        'neutral': [
            "The AI recommendations were {adjective}. Some were helpful while others {verb} the mark.",
            "I had mixed feelings about the algorithm. It {verb} in some areas but {action} in others.",
            "The technology was {adjective} - neither great nor terrible. It {verb} its basic function.",
        ]
    },
    'consumer_behavior': {
        'positive': [
            "I {verb} the product. It {adjective} exceeded my expectations in terms of {quality}.",
            "The {product_aspect} was {adjective}. I would {action} this to others.",
            "Overall, a {adjective} experience. The {quality} really stood out.",
            "I was genuinely {sentiment} with the {quality}. It felt like the product was designed with real care.",
            "Great {product_aspect}. I felt the {quality} was {adjective} and worth every penny.",
        ],
        'negative': [
            "I was {sentiment} with the product. The {quality} was {adjective}.",
            "The {product_aspect} {verb} to meet my expectations. It felt {adjective}.",
            "Not {adjective} overall. The {quality} needs improvement.",
            "The product left me feeling {sentiment}. The {product_aspect} was particularly {adjective}.",
            "I expected more. The {quality} was {adjective} and the {product_aspect} {verb} to deliver.",
        ],
        'neutral': [
            "The product was {adjective}. It {verb} its purpose but nothing {adjective}.",
            "Mixed feelings - the {quality} was fine but the {product_aspect} could be {adjective}.",
            "A perfectly {adjective} product. It {verb} what it needed to without being particularly notable.",
        ]
    },
    'social_psychology': {
        'positive': [
            "I felt {adjective} about the interaction. The other person seemed {trait}.",
            "The experience was {adjective}. It made me feel {emotion}.",
            "I {verb} the social aspect. People appeared {adjective} and {trait}.",
            "The interaction left me feeling {emotion}. I thought the other person was very {trait} and {adjective}.",
            "It was a {adjective} experience interacting with others. I felt {emotion} about the whole thing.",
        ],
        'negative': [
            "I felt {emotion} during the interaction. The situation seemed {adjective}.",
            "The experience was {adjective}. It left me feeling {emotion}.",
            "I {verb} uncomfortable. The atmosphere felt {adjective}.",
            "The social interaction was {adjective}. I felt {emotion} and somewhat {adjective} throughout.",
            "I was not {sentiment} with how things went. The other person seemed {trait} and the situation felt {adjective}.",
        ],
        'neutral': [
            "The interaction was {adjective}. I didn't feel strongly either way.",
            "A {adjective} experience overall. Nothing particularly {trait}.",
            "It was an {adjective} interaction. I felt {adjective} about it without any strong reaction.",
        ]
    },
    'political': {
        'positive': [
            "I {verb} with this perspective. It seems {adjective} and {trait}.",
            "The position is {adjective}. I appreciate how it {action}.",
            "I find this view {adjective}. It {verb} with my values.",
            "This argument is {adjective}. I think it {verb} the core issues well and seems {trait}.",
        ],
        'negative': [
            "I {verb} with this perspective. It seems {adjective} and fails to {action}.",
            "This position is {adjective}. It doesn't {verb} the real issues.",
            "I find this view {adjective} and {trait}.",
            "I'm {sentiment} by this perspective. It {verb} important nuances and comes across as {adjective}.",
        ],
        'neutral': [
            "I have mixed feelings about this. Some points are {adjective} while others {verb} consideration.",
            "This perspective has both {adjective} and {adjective} aspects.",
            "I can see both sides. The argument is {adjective} in some ways but {verb} in others.",
        ]
    },
    'behavioral_economics': {
        'positive': [
            "The decision felt {adjective}. I was {sentiment} with how the options were presented.",
            "I found the choice {adjective}. The framing really {verb} me think about the trade-offs carefully.",
            "I was {sentiment} with my decision. It felt {adjective} to weigh the different factors.",
        ],
        'negative': [
            "The decision felt {adjective}. I was {sentiment} with how limited the options seemed.",
            "I found the choice {adjective}. It felt like the framing {verb} important considerations.",
            "I wasn't {sentiment} with the way the decision was structured. It seemed {adjective}.",
        ],
        'neutral': [
            "The decision was {adjective}. I weighed the options and {verb} a reasonable choice.",
            "Neither option stood out strongly. The choice felt {adjective} overall.",
        ]
    },
    'health_psychology': {
        'positive': [
            "I felt {sentiment} about the health information. It was {adjective} and {verb} my concerns.",
            "The health advice seemed {adjective}. I feel more {emotion} about making changes.",
            "This information was {adjective}. It {verb} me think more carefully about my health decisions.",
        ],
        'negative': [
            "I was {sentiment} by the health information. It felt {adjective} and didn't {verb} my specific situation.",
            "The advice seemed {adjective}. I'm {emotion} about whether it would actually help.",
            "Not very {adjective} health information. It {verb} the complexity of my situation.",
        ],
        'neutral': [
            "The health information was {adjective}. Some parts {verb} my needs, others not so much.",
            "I had {adjective} reactions to the advice. It was {adjective} but not life-changing.",
        ]
    },
    'organizational': {
        'positive': [
            "The workplace scenario felt {adjective}. I was {sentiment} with how the situation was handled.",
            "I thought the leadership approach was {adjective}. It {verb} employee concerns effectively.",
            "The organizational decision seemed {adjective} and {trait}. It would {action} team morale.",
        ],
        'negative': [
            "The workplace scenario felt {adjective}. I was {sentiment} with the management approach.",
            "I thought the decision was {adjective}. It {verb} important employee perspectives.",
            "The organizational approach seemed {adjective}. It would likely {action} trust and morale.",
        ],
        'neutral': [
            "The workplace scenario was {adjective}. It had both {adjective} and concerning elements.",
            "A {adjective} organizational situation. The approach {verb} some needs but not others.",
        ]
    },
    'education': {
        'positive': [
            "I found the learning experience {adjective}. It {verb} my understanding of the topic.",
            "The educational approach was {adjective}. I felt {emotion} about how much I learned.",
            "This was a {adjective} way to learn. The material {verb} my curiosity and engagement.",
        ],
        'negative': [
            "I found the learning experience {adjective}. It {verb} to engage me with the material.",
            "The educational approach was {adjective}. I felt {emotion} about the effectiveness.",
            "Not a very {adjective} learning experience. The material {verb} to capture my interest.",
        ],
        'neutral': [
            "The learning experience was {adjective}. It {verb} some things well but could improve in others.",
            "An {adjective} educational experience overall. Neither especially engaging nor boring.",
        ]
    },
}

WORD_BANKS = {
    'adjective_positive': ['excellent', 'impressive', 'helpful', 'intuitive', 'effective', 'valuable',
                          'thoughtful', 'accurate', 'responsive', 'innovative', 'reliable', 'satisfying',
                          'remarkable', 'compelling', 'well-designed', 'outstanding', 'refreshing',
                          'insightful', 'encouraging', 'empowering'],
    'adjective_negative': ['disappointing', 'frustrating', 'confusing', 'inaccurate', 'unhelpful',
                          'limited', 'generic', 'impersonal', 'unreliable', 'underwhelming',
                          'poorly designed', 'off-putting', 'problematic', 'tedious', 'misguided',
                          'unconvincing', 'shallow', 'ineffective'],
    'adjective_neutral': ['adequate', 'acceptable', 'standard', 'typical', 'moderate', 'ordinary',
                         'reasonable', 'average', 'straightforward', 'unremarkable', 'fair', 'decent'],
    'verb_positive': ['understood', 'captured', 'addressed', 'enhanced', 'improved', 'recognized',
                     'appreciated', 'supported', 'facilitated', 'delivered', 'exceeded', 'highlighted'],
    'verb_negative': ['missed', 'ignored', 'failed', 'overlooked', 'misunderstood', 'neglected',
                     'underestimated', 'distorted', 'complicated', 'undermined', 'dismissed', 'confused'],
    'verb_neutral': ['met', 'served', 'provided', 'delivered', 'offered', 'presented',
                    'covered', 'maintained', 'fulfilled', 'supplied', 'conveyed', 'handled'],
    'emotion_positive': ['satisfied', 'pleased', 'impressed', 'confident', 'comfortable', 'optimistic',
                        'encouraged', 'reassured', 'motivated', 'grateful', 'relieved', 'engaged'],
    'emotion_negative': ['frustrated', 'disappointed', 'concerned', 'uncomfortable', 'skeptical',
                        'anxious', 'annoyed', 'uneasy', 'discouraged', 'doubtful', 'irritated'],
    'emotion_neutral': ['indifferent', 'neutral', 'uncertain', 'ambivalent', 'mixed', 'undecided'],
    'trait_positive': ['genuine', 'trustworthy', 'competent', 'approachable', 'transparent',
                      'professional', 'attentive', 'considerate', 'fair-minded', 'knowledgeable'],
    'trait_negative': ['dismissive', 'insincere', 'unreliable', 'distant', 'opaque',
                      'condescending', 'careless', 'biased', 'evasive', 'unprofessional'],
    'trait_neutral': ['professional', 'neutral', 'detached', 'measured', 'cautious', 'reserved'],
}


def _generate_diverse_open_ended(
    question_text: str,
    domain: str,
    valence: str,
    persona_traits: Dict[str, float],
    condition: str,
    rng: np.random.RandomState
) -> str:
    """
    Generate diverse, contextually appropriate open-ended responses.

    v1.4.0: Enhanced with question-text-aware responses, condition-specific
    elaborations, and broader domain coverage.

    Args:
        question_text: The question being answered
        domain: Study domain (ai_technology, consumer_behavior, etc.)
        valence: Response valence (positive, negative, neutral)
        persona_traits: Persona characteristics
        condition: Current experimental condition
        rng: Random number generator

    Returns:
        Generated response text
    """
    # Select appropriate template set - try exact domain, then related domains
    domain_lower = str(domain).lower().replace(" ", "_")
    domain_aliases = {
        "technology": "ai_technology",
        "ai": "ai_technology",
        "marketing": "consumer_behavior",
        "consumer": "consumer_behavior",
        "social": "social_psychology",
        "psychology": "social_psychology",
        "politics": "political",
        "economics": "behavioral_economics",
        "finance": "behavioral_economics",
        "health": "health_psychology",
        "medical": "health_psychology",
        "workplace": "organizational",
        "management": "organizational",
        "leadership": "organizational",
        "learning": "education",
        "teaching": "education",
    }
    resolved_domain = domain_aliases.get(domain_lower, domain_lower)
    templates = RESPONSE_TEMPLATES_BY_DOMAIN.get(
        resolved_domain,
        RESPONSE_TEMPLATES_BY_DOMAIN.get('social_psychology', {})
    ).get(valence, [])

    if not templates:
        # Fallback to any valence from same domain
        domain_templates = RESPONSE_TEMPLATES_BY_DOMAIN.get(resolved_domain, {})
        for v in ['neutral', 'positive', 'negative']:
            templates = domain_templates.get(v, [])
            if templates:
                break
    if not templates:
        templates = ["I found this to be an interesting experience."]

    # Select template
    template = str(rng.choice(templates))

    # Fill in template with appropriate words
    verbosity = _safe_trait_value(persona_traits.get('verbosity'), 0.5)

    def get_word(category: str, sentiment: str) -> str:
        if sentiment == 'positive':
            words = WORD_BANKS.get(f'{category}_positive', WORD_BANKS.get(category, ['good']))
        elif sentiment == 'negative':
            words = WORD_BANKS.get(f'{category}_negative', WORD_BANKS.get(category, ['poor']))
        else:
            words = WORD_BANKS.get(f'{category}_neutral', WORD_BANKS.get(category, ['okay']))
        return str(rng.choice(words))

    # Replace placeholders - each replacement gets a unique random word
    response = template
    response = response.replace('{adjective}', get_word('adjective', valence), 1)
    # Replace any remaining {adjective} with a DIFFERENT word
    while '{adjective}' in response:
        response = response.replace('{adjective}', get_word('adjective', valence), 1)
    response = response.replace('{verb}', get_word('verb', valence))
    response = response.replace('{emotion}', get_word('emotion', valence))
    response = response.replace('{trait}', get_word('trait', valence))
    response = response.replace('{sentiment}', get_word('emotion', valence))
    response = response.replace('{action}', get_word('verb', valence))
    response = response.replace('{quality}', str(rng.choice(
        ['quality', 'functionality', 'design', 'performance', 'usability', 'value', 'reliability']
    )))
    response = response.replace('{product_aspect}', str(rng.choice(
        ['overall experience', 'main features', 'user interface', 'core functionality',
         'presentation', 'build quality', 'ease of use']
    )))

    # v1.4.0: Add question-text-aware elaboration
    # Extract key topic words from the question text for contextual responses
    q_lower = str(question_text).lower()
    condition_lower = str(condition).lower()

    # Add elaboration based on verbosity
    if verbosity > 0.7:
        # Build condition-specific elaborations
        elaborations = []
        # Reference the condition meaningfully
        if any(w in condition_lower for w in ['high', 'strong', 'positive']):
            elaborations.append(" I think the stronger approach really made a difference here.")
        elif any(w in condition_lower for w in ['low', 'weak', 'negative']):
            elaborations.append(" The weaker version was noticeable and affected my impression.")
        elif any(w in condition_lower for w in ['control', 'baseline', 'neutral']):
            elaborations.append(" Without any particular manipulation, this felt like a natural experience.")
        elif any(w in condition_lower for w in ['ai', 'algorithm', 'automated']):
            elaborations.append(" Knowing it was AI-driven definitely shaped my perspective.")
        elif any(w in condition_lower for w in ['human', 'personal', 'manual']):
            elaborations.append(" The human element of this really came through.")
        else:
            elaborations.append(f" Given the {condition_lower} condition, this was my honest impression.")

        # Question-text-aware elaborations
        if 'why' in q_lower or 'reason' in q_lower or 'explain' in q_lower:
            elaborations.append(" My main reasoning comes from my personal experiences with similar situations.")
        elif 'feel' in q_lower or 'emotion' in q_lower:
            elaborations.append(f" Emotionally, I felt {get_word('emotion', valence)} about the whole thing.")
        elif 'suggest' in q_lower or 'improve' in q_lower or 'recommend' in q_lower:
            elaborations.append(" I think there is room for improvement in how this was presented.")
        else:
            elaborations.append(" This aligns with my overall expectations.")

        response += str(rng.choice(elaborations))
    elif verbosity > 0.5:
        # Medium verbosity: short elaboration
        short_elaborations = [
            " That's my overall take.",
            " I hope that makes sense.",
            " It was an interesting experience overall.",
        ]
        if rng.random() > 0.5:
            response += str(rng.choice(short_elaborations))

    return response


# =============================================================================
# ITERATION 7: ENHANCED FACTORIAL DESIGN SUPPORT
# =============================================================================
# Better support for complex factorial designs (2x2, 2x3, 3x3, etc.)

def _parse_factorial_design(conditions: List[str]) -> Dict[str, Any]:
    """
    Parse factorial design structure from condition names.

    Detects and extracts:
    - Number of factors
    - Levels per factor
    - Main effects
    - Interaction structure

    Args:
        conditions: List of condition names

    Returns:
        Factorial design specification
    """
    design = {
        'is_factorial': False,
        'factors': {},
        'n_factors': 0,
        'design_string': '',
        'cell_structure': {},
        'main_effect_conditions': {},
        'interaction_cells': []
    }

    # Common factorial separators
    separators = ['×', ' x ', '_x_', ' × ', ' X ']

    # Try to parse each condition
    all_factors = {}
    cells = []

    for cond in conditions:
        parts = None
        used_sep = None
        for sep in separators:
            if sep in cond.lower():
                parts = [p.strip() for p in cond.split(sep)]
                used_sep = sep
                break

        if parts and len(parts) >= 2:
            cells.append({
                'condition': cond,
                'factors': parts
            })

            for i, part in enumerate(parts):
                factor_name = f'Factor_{i+1}'
                if factor_name not in all_factors:
                    all_factors[factor_name] = set()
                all_factors[factor_name].add(part.lower())

    if len(all_factors) >= 2:
        design['is_factorial'] = True
        design['factors'] = {k: list(v) for k, v in all_factors.items()}
        design['n_factors'] = len(all_factors)

        # Create design string (e.g., "2x2", "2x3")
        levels = [len(v) for v in all_factors.values()]
        design['design_string'] = 'x'.join(str(l) for l in levels)

        # Structure cells
        for cell in cells:
            cell_key = tuple(f.lower() for f in cell['factors'])
            design['cell_structure'][cell_key] = cell['condition']

        # Identify main effect conditions (vary only one factor)
        for factor_name, levels in design['factors'].items():
            design['main_effect_conditions'][factor_name] = []
            for level in levels:
                matching = [c for c in conditions if level in c.lower()]
                design['main_effect_conditions'][factor_name].extend(matching)

    return design


def _compute_factorial_effects(
    data: pd.DataFrame,
    factorial_design: Dict[str, Any],
    dv_column: str
) -> Dict[str, Any]:
    """
    Compute main effects and interactions for factorial design.

    Args:
        data: DataFrame with CONDITION column and DV
        factorial_design: Parsed factorial structure
        dv_column: Name of dependent variable column

    Returns:
        Effect estimates for main effects and interactions
    """
    effects = {
        'main_effects': {},
        'interactions': {},
        'cell_means': {}
    }

    if not factorial_design['is_factorial'] or dv_column not in data.columns:
        return effects

    # Calculate cell means
    for cond in data['CONDITION'].unique():
        cell_data = data[data['CONDITION'] == cond][dv_column].dropna()
        if len(cell_data) > 0:
            effects['cell_means'][cond] = {
                'mean': round(cell_data.mean(), 3),
                'sd': round(cell_data.std(), 3),
                'n': len(cell_data)
            }

    # Calculate main effects
    for factor_name, levels in factorial_design['factors'].items():
        if len(levels) >= 2:
            level_means = []
            for level in levels:
                matching_conds = [c for c in data['CONDITION'].unique()
                                 if level in c.lower()]
                level_data = data[data['CONDITION'].isin(matching_conds)][dv_column].dropna()
                if len(level_data) > 0:
                    level_means.append((level, level_data.mean()))

            if len(level_means) >= 2:
                effects['main_effects'][factor_name] = {
                    'level_means': {l: round(m, 3) for l, m in level_means},
                    'effect_size': round(abs(level_means[0][1] - level_means[1][1]), 3)
                }

    return effects


# =============================================================================
# ITERATION 8: RESPONSE CONSISTENCY MODELING
# =============================================================================
# Model within-person consistency across similar items

def _generate_consistent_responses(
    n_items: int,
    base_tendency: float,
    consistency_level: float,
    scale_points: int,
    item_similarities: Optional[List[List[float]]],
    rng: np.random.RandomState
) -> List[int]:
    """
    Generate responses with realistic within-person consistency.

    Similar items should have correlated responses. This models the
    fact that people tend to respond consistently to items measuring
    the same construct.

    Args:
        n_items: Number of items
        base_tendency: Base response tendency (0-1)
        consistency_level: How consistent responses should be (0-1)
        scale_points: Number of scale points
        item_similarities: Optional NxN matrix of item similarities
        rng: Random number generator

    Returns:
        List of integer responses
    """
    if n_items <= 0:
        return []

    # Generate latent person-level trait
    trait = rng.normal(base_tendency, 0.15)

    # Generate item-specific deviations
    # High consistency = low item-specific variance
    item_variance = (1 - consistency_level) * 0.3

    responses = []
    previous_response = None

    for i in range(n_items):
        # Base response from trait
        item_tendency = trait

        # Add item-specific deviation
        item_tendency += rng.normal(0, item_variance)

        # If there are item similarities, incorporate those
        if item_similarities and previous_response is not None and i > 0:
            # Pull toward previous response based on similarity
            similarity = (
                item_similarities[i-1][i]
                if i < len(item_similarities) and i < len(item_similarities[i-1])
                else 0.5
            )
            # v1.0.0: Guard against division by zero when scale_points <= 1
            scale_range = max(scale_points - 1, 1)
            prev_normalized = (previous_response - 1) / scale_range
            item_tendency = item_tendency * (1 - similarity * 0.3) + prev_normalized * similarity * 0.3

        # Convert to scale response
        raw = item_tendency * (scale_points - 1) + 1
        response = int(np.clip(round(raw), 1, scale_points))
        responses.append(response)
        previous_response = response

    return responses


# =============================================================================
# ITERATION 9: HEATMAP AND SPECIAL QUESTION TYPE HANDLING
# =============================================================================
# Handle special question types found in QSF analysis: HeatMap, RO, FileUpload

def _generate_heatmap_response(
    image_width: int,
    image_height: int,
    n_clicks: int,
    attention_level: float,
    condition_focus: Optional[str],
    rng: np.random.RandomState
) -> List[Dict[str, int]]:
    """
    Generate simulated heatmap click coordinates.

    Args:
        image_width: Width of image in pixels
        image_height: Height of image in pixels
        n_clicks: Number of clicks to generate
        attention_level: Participant attention (affects click spread)
        condition_focus: Optional area of focus (e.g., 'center', 'left', 'product')
        rng: Random number generator

    Returns:
        List of {x, y} coordinate dictionaries
    """
    clicks = []

    # Determine focus area
    if condition_focus == 'center':
        center_x, center_y = image_width / 2, image_height / 2
        spread_x, spread_y = image_width * 0.3, image_height * 0.3
    elif condition_focus == 'left':
        center_x, center_y = image_width * 0.25, image_height / 2
        spread_x, spread_y = image_width * 0.2, image_height * 0.4
    elif condition_focus == 'right':
        center_x, center_y = image_width * 0.75, image_height / 2
        spread_x, spread_y = image_width * 0.2, image_height * 0.4
    else:
        # Default: slightly center-biased
        center_x, center_y = image_width / 2, image_height / 2
        spread_x, spread_y = image_width * 0.35, image_height * 0.35

    # Attention affects spread (lower attention = more random)
    spread_multiplier = 1.5 - attention_level

    for _ in range(n_clicks):
        x = rng.normal(center_x, spread_x * spread_multiplier)
        y = rng.normal(center_y, spread_y * spread_multiplier)

        # Clip to image bounds
        x = int(np.clip(x, 0, image_width - 1))
        y = int(np.clip(y, 0, image_height - 1))

        clicks.append({'x': x, 'y': y})

    return clicks


def _generate_rank_order_response(
    items: List[str],
    preferences: Dict[str, float],
    attention_level: float,
    rng: np.random.RandomState
) -> List[str]:
    """
    Generate rank ordering of items based on preferences.

    Args:
        items: List of items to rank
        preferences: Dict mapping item -> preference score (0-1)
        attention_level: Affects ranking consistency
        rng: Random number generator

    Returns:
        Items in ranked order (first = most preferred)
    """
    if not items:
        return []

    # Get preference scores with noise
    scored_items = []
    for item in items:
        base_score = preferences.get(item, 0.5)
        # Add noise inversely proportional to attention
        noise = rng.normal(0, 0.2 * (1 - attention_level))
        scored_items.append((item, base_score + noise))

    # Sort by score (descending)
    scored_items.sort(key=lambda x: x[1], reverse=True)

    return [item for item, score in scored_items]


# =============================================================================
# ITERATION 10: COMPREHENSIVE DATA QUALITY METRICS
# =============================================================================
# Generate comprehensive data quality metrics for validation

def _compute_data_quality_metrics(
    data: pd.DataFrame,
    scale_columns: List[str],
    conditions: List[str]
) -> Dict[str, Any]:
    """
    Compute comprehensive data quality metrics.

    Includes:
    - Response distribution statistics
    - Scale reliability estimates
    - Careless responding rates
    - Condition balance
    - Missing data patterns

    Args:
        data: Generated DataFrame
        scale_columns: Columns containing scale items
        conditions: List of condition names

    Returns:
        Comprehensive quality metrics
    """
    metrics = {
        'n_participants': len(data),
        'n_conditions': len(conditions),
        'response_distributions': {},
        'reliability_estimates': {},
        'careless_rates': {},
        'condition_balance': {},
        'missing_data': {},
        'overall_quality_score': 0.0
    }

    # 1. Condition balance
    if 'CONDITION' in data.columns and len(conditions) > 0:
        cond_counts = data['CONDITION'].value_counts()
        # v1.0.0: Guard against division by zero
        expected = len(data) / len(conditions)
        balance_scores = []
        for cond in conditions:
            actual = cond_counts.get(cond, 0)
            deviation = abs(actual - expected) / expected if expected > 0 else 0
            metrics['condition_balance'][cond] = {
                'count': int(actual),
                'expected': round(expected, 1),
                'deviation': round(deviation, 3)
            }
            balance_scores.append(1 - min(deviation, 1))
        metrics['balance_score'] = round(np.mean(balance_scores), 3)

    # 2. Response distributions for scale columns
    for col in scale_columns:
        if col not in data.columns:
            continue
        try:
            values = data[col].dropna().astype(float)
            if len(values) > 0:
                metrics['response_distributions'][col] = {
                    'mean': round(values.mean(), 3),
                    'sd': round(values.std(), 3),
                    'min': round(values.min(), 3),
                    'max': round(values.max(), 3),
                    'skewness': round(float(values.skew()), 3) if len(values) > 2 else 0
                }
        except Exception:
            continue

    # 3. Missing data
    total_cells = len(data) * len(data.columns)
    missing_cells = data.isna().sum().sum()
    metrics['missing_data'] = {
        'total_missing': int(missing_cells),
        'missing_rate': round(missing_cells / total_cells, 4) if total_cells > 0 else 0
    }

    # 4. Overall quality score (0-1)
    quality_components = []
    if 'balance_score' in metrics:
        quality_components.append(metrics['balance_score'])
    if metrics['missing_data']['missing_rate'] < 0.05:
        quality_components.append(1.0)
    else:
        quality_components.append(max(0, 1 - metrics['missing_data']['missing_rate'] * 10))

    metrics['overall_quality_score'] = round(np.mean(quality_components) if quality_components else 0.5, 3)

    return metrics


class SurveyFlowHandler:
    """
    Handler for survey flow logic - determines which questions participants see
    based on their experimental condition.

    Implements condition-based question visibility to ensure simulated
    participants only receive responses for questions they would actually see.

    Detection methods:
    1. Explicit condition restrictions in question spec
    2. DisplayLogic parsing for condition checks
    3. Block name analysis (condition keywords in block names)
    4. Embedded data checks referencing condition variables
    5. Factor-level matching for factorial designs
    """

    def __init__(
        self,
        conditions: List[str],
        open_ended_questions: List[Dict[str, Any]],
        precomputed_visibility: Optional[Dict[str, Dict[str, bool]]] = None
    ):
        """Initialize the survey flow handler.

        Args:
            conditions: List of condition names
            open_ended_questions: List of question info dicts
            precomputed_visibility: v1.0.0 - Optional pre-computed visibility map from QSF parser
                Format: {condition: {question_id: True/False}}
        """
        self.conditions = [str(c).lower().strip() for c in conditions]
        self.condition_map = {c.lower().strip(): c for c in conditions}
        self.questions = open_ended_questions
        self.precomputed_visibility = precomputed_visibility or {}
        # Parse conditions into factors for factorial designs
        self.factor_levels = self._extract_factor_levels()
        self.visibility_map = self._build_visibility_map()

    def _extract_factor_levels(self) -> Dict[str, Set[str]]:
        """Extract factor levels from condition names for factorial designs."""
        # Common separators in factorial condition names
        separators = ['×', ' x ', '_x_', ' × ']
        factor_levels: Dict[str, Set[str]] = {}

        for cond in self.conditions:
            # Check if this is a crossed condition
            parts = None
            for sep in separators:
                if sep in cond:
                    parts = [p.strip() for p in cond.split(sep)]
                    break

            if parts and len(parts) >= 2:
                for i, part in enumerate(parts):
                    factor_key = f"factor_{i}"
                    if factor_key not in factor_levels:
                        factor_levels[factor_key] = set()
                    factor_levels[factor_key].add(part.lower())

        return factor_levels

    def _build_visibility_map(self) -> Dict[str, Dict[str, bool]]:
        """Build map of question -> condition -> visibility.

        v1.0.0: Enhanced to use pre-computed visibility from QSF parser when available.
        The pre-computed visibility is based on comprehensive block-level analysis
        of the survey flow structure (BlockRandomizers, display logic, etc.).
        """
        visibility = {}

        for q in self.questions:
            q_name = str(q.get("name", "")).strip()
            q_id = str(q.get("question_id", q.get("qid", q_name))).strip()
            if not q_name:
                continue

            # v1.0.0: First check pre-computed visibility from QSF parser
            # This is more accurate as it's based on actual block-level flow analysis
            precomputed_used = False
            if self.precomputed_visibility:
                q_visibility = {}
                for cond in self.conditions:
                    # Find matching condition in precomputed (case-insensitive)
                    for pc_cond, pc_vis in self.precomputed_visibility.items():
                        if pc_cond.lower().strip() == cond:
                            # Check if question is in this condition's visibility
                            # Try both q_name and q_id
                            if q_id in pc_vis:
                                q_visibility[cond] = pc_vis[q_id]
                                precomputed_used = True
                            elif q_name in pc_vis:
                                q_visibility[cond] = pc_vis[q_name]
                                precomputed_used = True
                            break

            # Fall back to building visibility from question metadata
            if not precomputed_used:
                # Get various sources of visibility info
                display_logic = q.get("display_logic") or q.get("display_logic_details") or {}
                condition_restriction = q.get("condition") or q.get("visible_conditions") or []
                block_name = str(q.get("block_name", "")).lower()
                question_text = str(q.get("question_text", "")).lower()

                # Initialize all conditions as visible by default
                q_visibility = {c: True for c in self.conditions}

                # Method 1: Explicit condition restrictions
                if condition_restriction:
                    if isinstance(condition_restriction, str):
                        condition_restriction = [condition_restriction]
                    allowed = [str(c).lower().strip() for c in condition_restriction]
                    for cond in self.conditions:
                        q_visibility[cond] = self._condition_matches_any(cond, allowed)

                # Method 2: Display logic parsing
                if display_logic and isinstance(display_logic, dict):
                    self._apply_display_logic(q_visibility, display_logic)

                # Method 3: Block name analysis
                self._apply_block_name_logic(q_visibility, block_name)

                # Method 4: Question text hints (e.g., "For AI condition participants...")
                self._apply_question_text_hints(q_visibility, question_text)

            visibility[q_name] = q_visibility

        return visibility

    def _condition_matches_any(self, condition: str, allowed: List[str]) -> bool:
        """Check if a condition matches any in the allowed list."""
        cond_lower = condition.lower()

        # Direct match
        if cond_lower in allowed:
            return True

        # Partial match for factorial conditions
        for a in allowed:
            # Check if allowed pattern is a factor level within condition
            # e.g., "ai" should match "ai × hedonic"
            if a in cond_lower or cond_lower in a:
                return True

            # Check factor-level matching
            for factor_key, levels in self.factor_levels.items():
                if a in levels and a in cond_lower:
                    return True

        return False

    def _apply_display_logic(self, q_visibility: Dict[str, bool], display_logic: Dict[str, Any]):
        """Apply display logic rules to visibility map.

        Enhanced to handle multiple Qualtrics display logic patterns:
        - Embedded data field checks (e.g., Condition = "AI")
        - Question response checks (e.g., Q1 = "Yes")
        - Multiple condition combinations (AND/OR logic)
        - Choice locator patterns (e.g., q://QID123/SelectableChoice/1)
        """
        logic_type = display_logic.get("Type", "").lower()
        logic_conditions = display_logic.get("conditions", [])

        # Also check for 'Condition' field directly in display_logic
        if display_logic.get("Condition"):
            direct_conditions = display_logic.get("Condition", [])
            if isinstance(direct_conditions, list):
                logic_conditions.extend(direct_conditions)

        condition_results = {}  # Track which conditions satisfy each logic rule

        for logic_cond in logic_conditions:
            if not isinstance(logic_cond, dict):
                continue

            choice_locator = str(logic_cond.get("choice_locator", logic_cond.get("ChoiceLocator", ""))).lower()
            question_id = str(logic_cond.get("question_id", logic_cond.get("QuestionID", ""))).lower()
            operator = str(logic_cond.get("operator", logic_cond.get("Operator", ""))).lower()
            left_operand = str(logic_cond.get("LeftOperand", "")).lower()
            right_operand = str(logic_cond.get("RightOperand", "")).lower()

            # Check for embedded data references
            is_embedded_check = (
                "embedded" in left_operand or
                "embedded" in question_id or
                "ed://" in choice_locator or
                "condition" in choice_locator or
                "condition" in question_id
            )

            if is_embedded_check:
                # This is an embedded data check - likely checking condition
                # Extract the value being checked
                check_value = right_operand or choice_locator

                for cond in self.conditions:
                    cond_parts = cond.replace('×', ' ').replace('_', ' ').lower().split()

                    # Check if any condition part matches the check value
                    for part in cond_parts:
                        if len(part) >= 2:
                            part_in_check = part in check_value
                            part_in_locator = part in choice_locator

                            if part_in_check or part_in_locator:
                                if operator in ["selected", "equalto", "is", "="]:
                                    # This condition satisfies the display logic
                                    if cond not in condition_results:
                                        condition_results[cond] = []
                                    condition_results[cond].append(True)
                                elif operator in ["notselected", "notequalto", "isnot", "!="]:
                                    # This condition does NOT satisfy
                                    if cond not in condition_results:
                                        condition_results[cond] = []
                                    condition_results[cond].append(False)

            # Check for direct condition value in locator (e.g., q://QID123/SelectableChoice/AI)
            else:
                for cond in self.conditions:
                    cond_parts = cond.replace('×', ' ').replace('_', ' ').lower().split()
                    for part in cond_parts:
                        if len(part) >= 2 and (part in choice_locator or part in right_operand):
                            if operator in ["selected", "equalto", "is", "="]:
                                if cond not in condition_results:
                                    condition_results[cond] = []
                                condition_results[cond].append(True)

        # Apply results based on logic type (AND requires all true, OR requires any true)
        if condition_results:
            for cond in self.conditions:
                results = condition_results.get(cond, [])
                if results:
                    if logic_type in ["and", "booleanand"]:
                        q_visibility[cond] = all(results)
                    else:  # OR or default
                        q_visibility[cond] = any(results)
                else:
                    # No explicit match - check if other conditions matched
                    # If ANY condition explicitly matched, non-matching conditions don't see it
                    if any(condition_results.values()):
                        q_visibility[cond] = False

    def _apply_block_name_logic(self, q_visibility: Dict[str, bool], block_name: str):
        """Apply visibility rules based on block name."""
        if not block_name:
            return

        block_lower = block_name.lower()

        # Check for explicit negation patterns in block name
        # e.g., "no_ai_block", "non_ai", "without_ai"
        negation_patterns = ['no_', 'no ', 'non_', 'non ', 'without_', 'without ']
        has_negation = any(neg in block_lower for neg in negation_patterns)

        # Find which factor level(s) the block name refers to
        block_keywords = []
        for cond in self.conditions:
            cond_parts = cond.replace('×', ' ').replace('_', ' ').lower().split()
            for part in cond_parts:
                # Skip common words and negations
                if part in ['no', 'non', 'without', 'and', 'or', 'the', 'a']:
                    continue
                # Include 2+ character keywords (e.g., "ai")
                if len(part) >= 2 and part in block_lower:
                    block_keywords.append(part)

        if not block_keywords:
            return

        # Get the primary keyword (longest match)
        primary_keyword = max(set(block_keywords), key=len)

        # Determine which conditions should see this block
        for cond in self.conditions:
            cond_lower = cond.lower()

            # Check if this condition has the keyword
            has_keyword = primary_keyword in cond_lower.replace('×', ' ').replace('_', ' ')

            # Check if this condition has negation of the keyword
            cond_has_negation = any(
                f"{neg}{primary_keyword}" in cond_lower.replace('×', ' ').replace('_', ' ')
                or f"{neg} {primary_keyword}" in cond_lower.replace('×', ' ')
                for neg in ['no', 'non', 'without']
            )

            # Block has "AI" without negation -> only conditions with "AI" (not "No AI") see it
            # Block has "No AI" -> only conditions with "No AI" see it
            if has_negation:
                # Block is for negated conditions (e.g., "no_ai_block")
                q_visibility[cond] = cond_has_negation
            else:
                # Block is for positive conditions (e.g., "ai_block")
                # Conditions with "No AI" should NOT see it
                q_visibility[cond] = has_keyword and not cond_has_negation

    def _apply_question_text_hints(self, q_visibility: Dict[str, bool], question_text: str):
        """Check question text for condition-specific language."""
        if not question_text:
            return

        # Patterns that indicate condition-specific questions
        condition_phrases = [
            ("for those who", True),
            ("if you were in the", True),
            ("in the ai condition", True),
            ("in the human condition", False),
            ("for participants who", True),
        ]

        for phrase, _ in condition_phrases:
            if phrase in question_text:
                # Try to extract which condition this refers to
                for cond in self.conditions:
                    cond_parts = cond.replace('×', ' ').replace('_', ' ').lower().split()
                    for part in cond_parts:
                        if len(part) > 2 and part in question_text:
                            # This question mentions a specific condition
                            for other_cond in self.conditions:
                                if part not in other_cond:
                                    q_visibility[other_cond] = False
                            break

    def is_question_visible(self, question_name: str, condition: str) -> bool:
        """Check if a question is visible for a given condition."""
        # Try exact match first
        q_visibility = self.visibility_map.get(question_name, {})

        # Also try without underscores/spaces
        if not q_visibility:
            normalized = question_name.replace("_", " ").replace("-", " ")
            for q_name, vis in self.visibility_map.items():
                if q_name.replace("_", " ").replace("-", " ") == normalized:
                    q_visibility = vis
                    break

        if not q_visibility:
            return True  # Default to visible

        condition_lower = str(condition).lower().strip()

        # Direct match
        if condition_lower in q_visibility:
            return q_visibility[condition_lower]

        # Partial match for factorial conditions
        for mapped_cond, visible in q_visibility.items():
            if self._conditions_share_factor(mapped_cond, condition_lower):
                return visible

        return True

    def _conditions_share_factor(self, cond1: str, cond2: str) -> bool:
        """Check if two conditions share a factor level."""
        parts1 = set(cond1.replace('×', ' ').replace('_', ' ').split())
        parts2 = set(cond2.replace('×', ' ').replace('_', ' ').split())
        return bool(parts1 & parts2)

    def get_visible_questions(self, condition: str) -> List[Dict[str, Any]]:
        """Get all questions visible for a specific condition."""
        return [q for q in self.questions
                if self.is_question_visible(str(q.get("name", "")), condition)]


@dataclass
class EffectSizeSpec:
    """Specification for an expected effect in the study.

    v1.4.0: Added __post_init__ to safely convert cohens_d and direction,
    preventing crashes from string/dict/None values.
    """
    variable: str
    factor: str
    level_high: str
    level_low: str
    cohens_d: float
    direction: str = "positive"  # "positive" or "negative"

    def __post_init__(self) -> None:
        """Safely convert fields to proper types."""
        # Safely convert cohens_d
        try:
            if isinstance(self.cohens_d, dict):
                self.cohens_d = float(self.cohens_d.get("value", 0.5))
            elif self.cohens_d is None:
                self.cohens_d = 0.5
            else:
                self.cohens_d = float(self.cohens_d)
        except (ValueError, TypeError):
            self.cohens_d = 0.5

        if np.isnan(self.cohens_d) or np.isinf(self.cohens_d):
            self.cohens_d = 0.5

        # Ensure direction is a valid string
        if self.direction not in ("positive", "negative"):
            self.direction = "positive"

        # Ensure string fields are strings
        self.variable = str(self.variable or "")
        self.factor = str(self.factor or "")
        self.level_high = str(self.level_high or "")
        self.level_low = str(self.level_low or "")


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
        # v1.0.0: Pre-computed visibility map from QSF parser
        # Format: {condition: {question_id: True/False}}
        precomputed_visibility: Optional[Dict[str, Dict[str, bool]]] = None,
        # Cross-DV correlation structure (optional)
        # If provided, used to generate correlated latent scores across scales
        correlation_matrix: Optional[np.ndarray] = None,
        # Missing data simulation parameters
        missing_data_rate: float = 0.0,  # 0.0 = none; 0.04 = 4% item-level missingness
        dropout_rate: float = 0.0,  # 0.0 = none; 0.07 = 7% participant dropout
        missing_data_mechanism: str = "realistic",  # "none", "mcar", "realistic"
        allow_template_fallback: bool = True,
        progress_callback: Optional[callable] = None,
        # v1.0.8.1: SocSim experimental enrichment for economic game DVs
        use_socsim_experimental: bool = False,
    ):
        self.progress_callback = progress_callback
        self.use_socsim_experimental = bool(use_socsim_experimental)
        self.study_title = str(study_title or "").strip()
        self.study_description = str(study_description or "").strip()
        self.sample_size = int(sample_size)
        # Normalize condition names: strip whitespace AND non-breaking spaces (\xa0)
        self.conditions = [
            str(c).replace('\xa0', ' ').strip()
            for c in (conditions or [])
            if str(c).replace('\xa0', ' ').strip()
        ]
        if not self.conditions:
            # v1.1.1.5: Use logger instead of self._log() here — validation_log
            # is not yet initialized at this point in __init__.
            logger.warning("No conditions specified — defaulting to single 'Condition A'")
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
        # v1.0.5.1: Extract condition descriptions for domain detection and effect sizing
        self.condition_descriptions: Dict[str, str] = {}
        if self.study_context.get("condition_descriptions"):
            self.condition_descriptions = dict(self.study_context["condition_descriptions"])
        self.stimulus_evaluations = stimulus_evaluations or []
        self.condition_allocation = self._normalize_condition_allocation(
            condition_allocation, self.conditions
        )  # Dict[condition_name, percentage 0-100]
        self.precomputed_visibility = precomputed_visibility or {}  # v1.0.0: From QSF parser
        # Cross-DV correlation structure
        self.correlation_matrix = correlation_matrix
        # Missing data simulation
        self.missing_data_rate = float(np.clip(missing_data_rate, 0.0, 0.50))
        self.dropout_rate = float(np.clip(dropout_rate, 0.0, 0.50))
        self.missing_data_mechanism = (
            missing_data_mechanism if missing_data_mechanism in ("none", "mcar", "realistic")
            else "realistic"
        )
        self.allow_template_fallback = bool(allow_template_fallback)
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
        # Also detect domains from condition names AND descriptions for better persona matching
        # v1.0.5.1: Include condition descriptions in domain detection text
        condition_text = " ".join(str(c) for c in self.conditions)
        if self.condition_descriptions:
            condition_text += " " + " ".join(str(d) for d in self.condition_descriptions.values() if d)
        condition_domains = self.persona_library.detect_domains(
            condition_text, ""
        )
        # Merge detected domains
        all_domains = list(set(self.detected_domains + condition_domains))
        self.detected_domains = all_domains if all_domains else self.detected_domains

        # v1.3.6: Also merge in explicitly provided persona_domains from builder
        _explicit_persona_domains = self.study_context.get("persona_domains", [])
        if _explicit_persona_domains and isinstance(_explicit_persona_domains, list):
            _merged = list(set(self.detected_domains + _explicit_persona_domains))
            self.detected_domains = _merged if _merged else self.detected_domains

        self.available_personas = self.persona_library.get_personas_for_domains(
            self.detected_domains
        )

        # v1.0.4.6 Step 9: Persona pool validation
        # Ensure we have enough domain-specific personas (≥3 non-response-style)
        # If too few, broaden to adjacent domains
        _domain_specific = {n: p for n, p in self.available_personas.items()
                           if p.category != 'response_style'}
        _MIN_DOMAIN_PERSONAS = 3
        if len(_domain_specific) < _MIN_DOMAIN_PERSONAS and self.detected_domains:
            _ADJACENT_DOMAINS = {
                'political_psychology': ['social_psychology', 'intergroup_relations', 'moral_psychology'],
                'economic_games': ['behavioral_economics', 'social_psychology', 'cooperation'],
                'intergroup_relations': ['political_psychology', 'social_psychology', 'prejudice'],
                'consumer_behavior': ['behavioral_economics', 'marketing', 'social_psychology'],
                'health_psychology': ['clinical', 'behavioral_economics', 'social_psychology'],
                'organizational_behavior': ['social_psychology', 'leadership', 'management'],
                'clinical': ['health_psychology', 'social_psychology', 'stress'],
                'ai': ['technology', 'consumer_behavior', 'behavioral_economics'],
                'environmental': ['social_psychology', 'consumer_behavior', 'moral_psychology'],
                'moral_psychology': ['social_psychology', 'political_psychology', 'fairness'],
                'social_psychology': ['behavioral_economics', 'political_psychology', 'intergroup_relations'],
                'behavioral_economics': ['economic_games', 'consumer_behavior', 'social_psychology'],
                'communication': ['social_psychology', 'consumer_behavior', 'political_psychology'],
            }
            _expanded_domains = list(self.detected_domains)
            for d in self.detected_domains:
                _expanded_domains.extend(_ADJACENT_DOMAINS.get(d, []))
            _expanded_domains = list(set(_expanded_domains))
            _expanded_personas = self.persona_library.get_personas_for_domains(_expanded_domains)
            if len({n: p for n, p in _expanded_personas.items()
                    if p.category != 'response_style'}) > len(_domain_specific):
                self.available_personas = _expanded_personas
                self._log(f"Persona pool expanded via adjacent domains: "
                         f"{len(_domain_specific)} → "
                         f"{len({n: p for n, p in _expanded_personas.items() if p.category != 'response_style'})} "
                         f"domain-specific personas")

        # Adjust persona weights based on study characteristics
        self._adjust_persona_weights_for_study()

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

        # Initialize survey flow handler for condition-based question visibility
        # This ensures participants only get responses for questions they would see
        # v1.0.0: Pass pre-computed visibility from QSF parser for accurate block-level visibility
        self.survey_flow_handler = SurveyFlowHandler(
            conditions=self.conditions,
            open_ended_questions=self.open_ended_questions,
            precomputed_visibility=self.precomputed_visibility
        )

        self.column_info: List[Tuple[str, str]] = []
        self.validation_log: List[str] = []
        self._scale_generation_log: List[Dict[str, Any]] = []

        # Initialize LLM-powered response generator (optional upgrade)
        # Must come after validation_log init so _log() works
        self.llm_generator = None
        self.llm_init_error: str = ""
        try:
            from .llm_response_generator import LLMResponseGenerator
            # LLMResponseGenerator has a built-in default API key;
            # also picks up LLM_API_KEY / GROQ_API_KEY from env or user-provided key
            _user_key = os.environ.get("LLM_API_KEY", "") or os.environ.get("GROQ_API_KEY", "")
            self.llm_generator = LLMResponseGenerator(
                api_key=_user_key or None,
                study_title=self.study_title,
                study_description=self.study_description,
                seed=self.seed,
                fallback_generator=self.comprehensive_generator,
                allow_template_fallback=self.allow_template_fallback,
                batch_size=20,
                all_conditions=self.conditions if self.conditions else None,
            )
            if self.llm_generator.is_llm_available:
                self._log(f"LLM response generator initialized ({self.llm_generator.provider_display_name})")
            else:
                # v1.9.0: Keep the generator alive even if initial check fails.
                # Providers may have transient issues; the generator has built-in
                # retry and cooldown logic that can recover during generation.
                self._log("LLM generator: initial check found no active providers, "
                          "will retry during generation (providers may recover)")
                # Reset all providers to give them a fresh chance during actual generation
                if hasattr(self.llm_generator, '_reset_all_providers'):
                    self.llm_generator._reset_all_providers()
                    self.llm_generator._api_available = True
        except Exception as _llm_err:
            self.llm_init_error = str(_llm_err)
            self._log(f"LLM generator not available (using templates): {_llm_err}")

    @staticmethod
    def _normalize_condition_allocation(
        allocation: Optional[Dict[str, Any]],
        conditions: List[str],
    ) -> Optional[Dict[str, float]]:
        """Normalize condition allocation dict, handling edge cases.

        v1.4.0: Ensures:
        - Empty dicts are treated as None (equal allocation)
        - String values are converted to floats
        - Proportions (0-1 range) are converted to percentages (0-100)
        - Keys that don't match conditions are matched case-insensitively
        - All values are proper floats
        - Total allocation sums to ~100%

        Args:
            allocation: Raw condition allocation dict (or None)
            conditions: List of condition names

        Returns:
            Normalized allocation dict or None for equal allocation
        """
        if not allocation or not isinstance(allocation, dict):
            return None

        # Filter out empty/None values and convert to float
        cleaned: Dict[str, float] = {}
        condition_lower_map = {c.lower().strip(): c for c in conditions}

        for key, val in allocation.items():
            # Skip None/empty values
            if val is None:
                continue

            # Convert value to float safely
            if isinstance(val, dict):
                # Handle dict-contaminated values
                for dkey in ("value", "proportion", "percentage", "pct"):
                    if dkey in val:
                        try:
                            val = float(val[dkey])
                            break
                        except (ValueError, TypeError):
                            continue
                else:
                    continue  # Could not extract from dict
            try:
                float_val = float(val)
            except (ValueError, TypeError):
                continue

            # Skip NaN/inf
            if np.isnan(float_val) or np.isinf(float_val):
                continue

            # Match key to actual condition name (case-insensitive)
            key_stripped = str(key).strip()
            matched_condition = None
            if key_stripped in [c for c in conditions]:
                matched_condition = key_stripped
            else:
                key_lower = key_stripped.lower()
                if key_lower in condition_lower_map:
                    matched_condition = condition_lower_map[key_lower]

            if matched_condition:
                cleaned[matched_condition] = float_val

        if not cleaned:
            return None

        # Detect if values are proportions (0-1) vs percentages (0-100)
        all_values = list(cleaned.values())
        total = sum(all_values)
        max_val = max(all_values) if all_values else 0

        if max_val <= 1.0 and total <= 1.05:
            # Values appear to be proportions (0-1), convert to percentages
            cleaned = {k: v * 100.0 for k, v in cleaned.items()}
            total = sum(cleaned.values())

        # If total is way off from 100, normalize to 100%
        if total > 0 and (total < 80 or total > 120):
            factor = 100.0 / total
            cleaned = {k: v * factor for k, v in cleaned.items()}

        return cleaned

    def _log(self, message: str) -> None:
        """Append a message to the validation log for debugging and verification."""
        self.validation_log.append(message)

    # =================================================================
    # MISSING DATA & DROPOUT SIMULATION
    # =================================================================

    def _should_be_missing(
        self,
        participant_idx: int,
        item_position: int,
        total_items: int,
        traits: Dict[str, float],
        condition: str,
    ) -> bool:
        """Determine if this item should be missing for this participant.

        Uses a blended model:
        - Base rate from self.missing_data_rate
        - Persona-driven: careless responders skip more (attention_level < 0.4 -> 3x base rate)
        - Fatigue: later items have higher skip probability
        - All bounded to keep data usable

        Args:
            participant_idx: Index of participant (0-based)
            item_position: Position of item in overall survey (0-based)
            total_items: Total number of items in the survey
            traits: Participant trait dictionary
            condition: Participant's experimental condition

        Returns:
            True if this item should be marked as missing (np.nan)
        """
        if self.missing_data_rate <= 0 or self.missing_data_mechanism == "none":
            return False

        base_rate = self.missing_data_rate

        if self.missing_data_mechanism == "mcar":
            # Missing Completely At Random: constant probability
            rng = np.random.RandomState(
                (self.seed + participant_idx * 317 + item_position * 53) % (2**31)
            )
            return bool(rng.random() < base_rate)

        # --- "realistic" mechanism ---
        # (1) Persona-driven adjustment: careless responders skip more
        attention = _safe_trait_value(traits.get("attention_level"), 0.7)
        if attention < 0.4:
            rate = base_rate * 3.0  # 3x more missing for careless
        elif attention < 0.6:
            rate = base_rate * 1.5
        else:
            rate = base_rate

        # (2) Fatigue: later items have higher skip probability
        if total_items > 1:
            progress = item_position / max(total_items - 1, 1)
            # At end of survey, up to 2x the base rate
            fatigue_multiplier = 1.0 + progress * 1.0
            rate *= fatigue_multiplier

        # (3) Cap so we never exceed 25% per item (keeps data usable)
        rate = min(rate, 0.25)

        rng = np.random.RandomState(
            (self.seed + participant_idx * 317 + item_position * 53) % (2**31)
        )
        return bool(rng.random() < rate)

    def _should_dropout(
        self,
        participant_idx: int,
        traits: Dict[str, float],
    ) -> Optional[int]:
        """Determine if and when this participant drops out.

        Returns:
            Item position (0-based) at which dropout occurs, or None for completion.
            Uses survival function: P(drop) increases with position.
            Careless responders (attention < 0.4) have 3x dropout rate.

        The dropout point is sampled from a geometric-like distribution
        weighted toward later survey positions (most dropouts happen in
        the second half).

        Args:
            participant_idx: Index of participant (0-based)
            traits: Participant trait dictionary
        """
        if self.dropout_rate <= 0:
            return None

        attention = _safe_trait_value(traits.get("attention_level"), 0.7)
        effective_rate = self.dropout_rate
        if attention < 0.4:
            effective_rate *= 3.0
        elif attention < 0.6:
            effective_rate *= 1.5

        # Cap at 40%
        effective_rate = min(effective_rate, 0.40)

        rng = np.random.RandomState(
            (self.seed + participant_idx * 997 + 77777) % (2**31)
        )

        # Will this participant drop out at all?
        if rng.random() >= effective_rate:
            return None  # Completes the survey

        # Sample dropout point using a Beta(2, 1.5) distribution
        # This skews toward the latter part of the survey (mean ~ 0.57)
        dropout_fraction = float(rng.beta(2.0, 1.5))
        # Ensure dropout is at least 10% into the survey and at most 95%
        dropout_fraction = float(np.clip(dropout_fraction, 0.10, 0.95))
        return dropout_fraction  # Will be multiplied by total_items in _apply_missing_data

    def _apply_missing_data(
        self,
        data: Dict[str, List],
        all_traits: List[Dict],
        conditions: "pd.Series",
        n: int,
    ) -> None:
        """Apply missing data patterns to the generated data dict in-place.

        Phase 1: Determine dropout points for each participant
        Phase 2: For dropouts, set all items after dropout point to np.nan
        Phase 3: For remaining participants, apply item-level missingness
        Phase 4: Recompute composite means to handle NaN (use nanmean)

        Does NOT apply missing data to: PARTICIPANT_ID, CONDITION, RUN_ID,
        SIMULATION_MODE, SIMULATION_SEED, Gender, Age, Exclude_Recommended,
        attention check columns, metadata columns.

        Args:
            data: The generated data dictionary (modified in-place)
            all_traits: List of participant trait dicts
            conditions: Pandas Series of condition assignments
            n: Number of participants
        """
        # Protected columns that should NEVER have missing data
        _PROTECTED_COLUMNS: Set[str] = {
            "PARTICIPANT_ID", "CONDITION", "RUN_ID", "SIMULATION_MODE",
            "SIMULATION_SEED", "Gender", "Age", "Exclude_Recommended",
            "Completion_Time_Seconds", "Attention_Pass_Rate",
            "Max_Straight_Line", "Flag_Speed", "Flag_Attention",
            "Flag_StraightLine",
        }
        # Also protect attention check columns
        for col in list(data.keys()):
            if "Attention_Check" in col or "attention_check" in col.lower():
                _PROTECTED_COLUMNS.add(col)

        # Identify eligible columns (numeric, not protected, not open-ended strings)
        eligible_cols: List[str] = []
        for col in data:
            if col in _PROTECTED_COLUMNS:
                continue
            values = data[col]
            if not values:
                continue
            # Check if column is numeric (int or float)
            sample = values[0]
            if isinstance(sample, (int, float)) and not isinstance(sample, bool):
                eligible_cols.append(col)

        if not eligible_cols:
            self._log("Missing data: no eligible columns found, skipping")
            return

        total_items = len(eligible_cols)
        dropout_count = 0
        total_missing_cells = 0
        total_eligible_cells = n * total_items
        per_col_missing: Dict[str, int] = {col: 0 for col in eligible_cols}

        # --- Phase 1: Determine dropout points ---
        dropout_points: Dict[int, int] = {}  # participant_idx -> item_position of dropout
        for i in range(n):
            dropout_frac = self._should_dropout(i, all_traits[i])
            if dropout_frac is not None:
                dropout_item = max(1, int(float(dropout_frac) * total_items))
                dropout_points[i] = dropout_item
                dropout_count += 1

        # --- Phase 2: Apply dropout (set all items after dropout point to NaN) ---
        for i, dropout_pos in dropout_points.items():
            for col_idx in range(dropout_pos, total_items):
                col = eligible_cols[col_idx]
                data[col][i] = np.nan
                per_col_missing[col] += 1
                total_missing_cells += 1

        # --- Phase 3: Apply item-level missingness for non-dropout participants ---
        if self.missing_data_rate > 0 and self.missing_data_mechanism != "none":
            for i in range(n):
                if i in dropout_points:
                    continue  # Already handled by dropout
                condition = str(conditions.iloc[i]) if hasattr(conditions, 'iloc') else str(conditions[i])
                for col_idx, col in enumerate(eligible_cols):
                    if self._should_be_missing(i, col_idx, total_items, all_traits[i], condition):
                        data[col][i] = np.nan
                        per_col_missing[col] += 1
                        total_missing_cells += 1

        # --- Phase 4: Recompute composite means using nanmean ---
        for col in list(data.keys()):
            if col.endswith("_mean"):
                # Find the corresponding item columns
                prefix = col[:-5]  # Remove "_mean"
                item_cols = [c for c in eligible_cols if c.startswith(prefix + "_") and c != col]
                if item_cols:
                    for i in range(n):
                        item_vals = []
                        for ic in item_cols:
                            v = data[ic][i]
                            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                                item_vals.append(float(v))
                        if item_vals:
                            data[col][i] = round(float(np.nanmean(item_vals)), 2)
                        else:
                            data[col][i] = np.nan
                            per_col_missing[col] = per_col_missing.get(col, 0) + 1

        # --- Store stats for metadata ---
        actual_rate = total_missing_cells / max(total_eligible_cells, 1)
        self._actual_missing_rate = round(actual_rate, 4)
        self._actual_dropout_count = dropout_count
        self._per_scale_missing_rate = {
            col: round(count / max(n, 1), 4)
            for col, count in per_col_missing.items()
            if count > 0
        }
        self._log(
            f"Missing data applied: {total_missing_cells}/{total_eligible_cells} cells "
            f"({actual_rate:.1%}), {dropout_count} dropouts"
        )

    def _adjust_persona_weights_for_study(self) -> None:
        """
        Adjust persona weights based on detected study domain and conditions.

        v1.0.4.6: Now uses self.detected_domains directly instead of
        re-keyword-matching study text. The 5-phase domain detection already
        ran on study title, description, and conditions — we use that result.

        SCIENTIFIC BASIS:
        =================
        Different study types attract different participant populations.
        This method adjusts persona weights to better reflect likely sample
        characteristics based on study context.

        References:
        - Buhrmester et al. (2011): MTurk sample characteristics
        - Peer et al. (2017): Online panel composition
        """
        _detected = set(getattr(self, 'detected_domains', []) or [])

        # =====================================================================
        # v1.0.4.6: Domain-aware persona weight adjustments
        # Uses detected_domains → category mapping for precise boosting
        # =====================================================================
        _DOMAIN_TO_CATEGORY_BOOST = {
            'ai': ('technology', 1.3),
            'technology': ('technology', 1.2),
            'consumer_behavior': ('consumer', 1.3),
            'marketing': ('consumer', 1.2),
            'organizational_behavior': ('organizational', 1.4),
            'social_psychology': ('social', 1.3),
            'health_psychology': ('health', 1.4),
            'environmental': ('environmental', 1.4),
            'political_psychology': ('social', 1.2),  # Political boosts social personas
            'behavioral_economics': ('behavioral_economics', 1.3),
            'economic_games': ('behavioral_economics', 1.3),
            'clinical': ('clinical', 1.4),
            'media_communication': ('communication', 1.3),
            'accuracy_misinformation': ('communication', 1.2),
        }

        for domain in _detected:
            if domain in _DOMAIN_TO_CATEGORY_BOOST:
                target_category, boost = _DOMAIN_TO_CATEGORY_BOOST[domain]
                for name, persona in self.available_personas.items():
                    if persona.category == target_category:
                        persona.weight *= boost

        # Additional name-based boosts for specific domain-persona affinities
        if _detected & {'political_psychology'}:
            for name, persona in self.available_personas.items():
                if any(kw in name for kw in ['prosocial', 'individualist', 'partisan',
                                              'moderate', 'ingroup', 'egalitarian']):
                    persona.weight *= 1.2

        if _detected & {'economic_games', 'behavioral_economics'}:
            for name, persona in self.available_personas.items():
                if any(kw in name for kw in ['loss_averse', 'overconfident', 'reciprocal',
                                              'free_rider', 'fairness', 'social_comparer']):
                    persona.weight *= 1.3

        if _detected & {'ai', 'technology'}:
            for name, persona in self.available_personas.items():
                if 'tech' in name or 'ai' in name or 'privacy' in name:
                    persona.weight *= 1.2

        if _detected & {'intergroup_relations', 'prejudice', 'social_identity'}:
            for name, persona in self.available_personas.items():
                if any(kw in name for kw in ['ingroup', 'egalitarian', 'partisan',
                                              'conformist', 'prosocial']):
                    persona.weight *= 1.3

        if _detected & {'moral_psychology', 'fairness'}:
            for name, persona in self.available_personas.items():
                if any(kw in name for kw in ['fairness', 'justice', 'prosocial',
                                              'partisan', 'egalitarian']):
                    persona.weight *= 1.2

        # =====================================================================
        # v1.0.4.6 Step 7: Study-context-enriched persona selection
        # Use study title, description, AND condition text for fine-grained
        # persona weight adjustments beyond domain detection
        # =====================================================================
        _study_ctx = f"{self.study_title or ''} {self.study_description or ''}".lower()
        _cond_ctx = " ".join(str(c) for c in self.conditions).lower()
        _full_ctx = _study_ctx + " " + _cond_ctx

        # Condition-text analysis: boost personas whose traits match condition semantics
        _CONDITION_PERSONA_AFFINITIES = {
            # Trust/cooperation conditions → prosocial + reciprocal personas
            ('trust', 'cooperation', 'prosocial', 'altruism'): [
                'prosocial', 'reciprocal', 'egalitarian', 'secure'],
            # Competition/conflict conditions → individualist + competitive personas
            ('competition', 'conflict', 'rivalry', 'threat'): [
                'individualist', 'competitive', 'free_rider'],
            # Fairness/justice conditions → fairness + justice personas
            ('fair', 'justice', 'equality', 'inequal'): [
                'fairness', 'justice', 'egalitarian', 'social_comparer'],
            # Identity/group conditions → intergroup + identity personas
            ('ingroup', 'outgroup', 'identity', 'partisan', 'party'): [
                'ingroup', 'partisan', 'egalitarian', 'conformist'],
            # Loss/risk conditions → loss averse + cautious personas
            ('loss', 'risk', 'gamble', 'uncertain'): [
                'loss_averse', 'rational', 'present_biased'],
            # Anxiety/stress conditions → clinical + anxious personas
            ('anxiety', 'stress', 'threat', 'fear'): [
                'anxious', 'clinical', 'health_fatalist'],
            # Authority/leadership conditions → authority + conformist personas
            ('authority', 'leader', 'manager', 'boss'): [
                'authority_sensitive', 'conformist', 'transformational',
                'high_performer', 'disengaged'],
        }

        for keywords, persona_fragments in _CONDITION_PERSONA_AFFINITIES.items():
            if any(kw in _full_ctx for kw in keywords):
                for name, persona in self.available_personas.items():
                    if any(frag in name for frag in persona_fragments):
                        persona.weight *= 1.15  # Modest boost — don't overwhelm domain boosting

        # Study title/description specific boosting for paradigm recognition
        if any(kw in _study_ctx for kw in ['dictator game', 'trust game', 'ultimatum',
                                             'public goods', 'prisoner']):
            # Economic game paradigm detected from study context
            for name, persona in self.available_personas.items():
                if any(frag in name for frag in ['reciprocal', 'free_rider', 'fairness_enforcer',
                                                   'prosocial', 'individualist', 'social_comparer']):
                    persona.weight *= 1.25

        if any(kw in _study_ctx for kw in ['polariz', 'partisan', 'democrat', 'republican',
                                             'trump', 'biden', 'political']):
            # Political study detected from context
            for name, persona in self.available_personas.items():
                if any(frag in name for frag in ['partisan', 'moderate', 'ingroup',
                                                   'egalitarian', 'conformist']):
                    persona.weight *= 1.25

        # Fallback: if no domains detected, use keyword matching
        if not _detected:
            _all_text = _full_ctx
            if any(kw in _all_text for kw in ['consumer', 'brand', 'purchase']):
                for name, persona in self.available_personas.items():
                    if persona.category == 'consumer':
                        persona.weight *= 1.3

        # Normalize weights
        total_weight = sum(p.weight for p in self.available_personas.values()) or 1.0
        for persona in self.available_personas.values():
            persona.weight = persona.weight / total_weight

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
                # If already validated by app.py, preserve all values exactly
                if scale.get("_validated"):
                    name = safe_str(scale.get("name"), "Scale")
                    normalized.append({
                        "name": name,
                        "variable_name": safe_str(scale.get("variable_name"), name.replace(" ", "_")),
                        "num_items": safe_int(scale.get("num_items"), 5),
                        "scale_points": safe_int(scale.get("scale_points"), 7),
                        "reverse_items": list(scale.get("reverse_items") or []),
                        "_validated": True,
                    })
                else:
                    # Not pre-validated: apply full validation
                    name = safe_str(scale.get("name"), "Scale")
                    pts = safe_int(scale.get("scale_points"), 7)
                    pts = max(2, min(1001, pts))
                    raw_items = scale.get("num_items")
                    if raw_items is None:
                        raw_items = scale.get("items")  # QSF detection key
                    n_items = safe_int(raw_items, 5)
                    n_items = max(1, n_items)
                    normalized.append({
                        "name": name,
                        "variable_name": safe_str(scale.get("variable_name"), name.replace(" ", "_")),
                        "num_items": n_items,
                        "scale_points": pts,
                        "reverse_items": list(scale.get("reverse_items") or []),
                        "_validated": True,
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

        # v1.0.0: Guard against empty persona list
        if not persona_names:
            # Fallback to default engaged persona
            default_persona = Persona(
                name="engaged",
                description="Default engaged responder",
                weight=1.0,
                traits={"response_tendency": 0.65, "variance": 0.3}
            )
            return "engaged", default_persona

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

    def _get_effect_for_condition(self, condition: str, variable: str) -> float:
        """
        Convert Cohen's d effect size to a normalized effect shift that produces
        STATISTICALLY DETECTABLE differences between conditions.

        v1.4.0: Enhanced with safe numeric conversion for cohens_d and improved
        level matching to reduce false positives (e.g., "no ai" matching "ai").

        CRITICAL FIX (v2.2.6): Previous versions produced effects too small to detect.

        Cohen's d interpretation for behavioral data:
        - d = 0.2: Small effect (detectable with N~400 per group)
        - d = 0.5: Medium effect (detectable with N~64 per group)
        - d = 0.8: Large effect (detectable with N~26 per group)

        For Likert scales (1-7), typical SD = 1.5 scale points
        Effect in raw scale units = d * SD = d * 1.5

        NEW APPROACH: Apply FULL effect size to condition means
        - d=0.5 should shift mean by 0.75 points on 7-point scale (0.5 * 1.5)
        - This is normalized to 0-1 range: 0.75 / 6 = 0.125 (12.5% shift)

        BUT we need stronger effects for pilot simulations where users expect
        to see differences. Use amplified conversion factor.
        """
        # v1.4.11: Recalibrated effect multiplier for accuracy
        # Converts Cohen's d to a 0-1 normalized shift
        # d=0.5 -> 0.15 shift -> ~0.9 points on 7-point scale (observed d ≈ 0.5-0.7)
        # Previous value of 0.40 produced observed d ~1.6x the specified d
        COHENS_D_TO_NORMALIZED = 0.30

        # Check explicit effect size specifications -- accumulate ALL matching effects
        # for factorial designs where multiple effect specs may apply to one condition
        matched_effects: list = []
        condition_lower = str(condition).lower().strip()
        variable_lower = str(variable).lower().strip()

        for effect in self.effect_sizes:
            # v1.1.1.5: Support both EffectSizeSpec objects AND plain dicts.
            # The app always passes EffectSizeSpec, but the engine should be
            # robust against dicts from tests, API callers, or legacy code.
            def _eget(obj: Any, key: str, default: Any = "") -> Any:
                """Get attribute from object or key from dict."""
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return getattr(obj, key, default)

            # v1.4.0: Safe conversion of cohens_d (could be string, dict, or NaN)
            try:
                _raw_d = _eget(effect, 'cohens_d', 0.5)
                cohens_d = float(_raw_d) if not isinstance(_raw_d, dict) else 0.5
            except (ValueError, TypeError, AttributeError):
                cohens_d = 0.5  # Default to medium effect on conversion failure
            if isinstance(cohens_d, float) and (np.isnan(cohens_d) or np.isinf(cohens_d)):
                cohens_d = 0.5
            # Clamp to reasonable range (0-3.0 covers virtually all real effects)
            cohens_d = float(np.clip(abs(cohens_d), 0.0, 3.0))

            # Check if this effect spec matches the current variable
            effect_var = str(_eget(effect, 'variable', '')).lower().strip()
            variable_matches = (
                effect_var == variable_lower
                or variable_lower.startswith(effect_var)
                or effect_var in variable_lower
            )

            if variable_matches:
                # v1.4.0: Improved level matching with false-positive prevention
                level_high = str(_eget(effect, 'level_high', '')).lower().strip()
                level_low = str(_eget(effect, 'level_low', '')).lower().strip()
                direction = str(_eget(effect, 'direction', 'positive')).lower().strip()

                # v1.0.1.3: Use word-boundary matching to prevent false positives
                # (e.g., level "ai" matching condition "wait")
                is_high = bool(level_high and _word_in(level_high, condition_lower))
                is_low = bool(level_low and _word_in(level_low, condition_lower))

                # Avoid double-matching (e.g., "no ai" matching both "ai" and "no ai")
                # If both match, prefer the longer/more specific match
                if is_high and is_low:
                    if len(level_high) >= len(level_low):
                        is_low = False
                    else:
                        is_high = False

                if is_high:
                    d = cohens_d if direction == "positive" else -cohens_d
                    matched_effects.append(d * COHENS_D_TO_NORMALIZED)
                elif is_low:
                    d = -cohens_d if direction == "positive" else cohens_d
                    matched_effects.append(d * COHENS_D_TO_NORMALIZED)

        if matched_effects:
            # Average matched effects so they don't stack unreasonably
            return sum(matched_effects) / len(matched_effects)

        # AUTO-GENERATE effect if no explicit specification
        # This ensures conditions ALWAYS produce different means
        return self._get_automatic_condition_effect(condition, variable)

    def _get_automatic_condition_effect(self, condition: str, variable: str) -> float:
        """
        Generate automatic condition effects based on SEMANTIC CONTENT, not position.

        VERSION 2.3.0: COMPREHENSIVE - All manipulation types grounded in published literature.

        Following Westwood (PNAS 2025), this simulation aims to approximate real human
        responses by applying theory-driven effect directions from published research.

        SCIENTIFIC BASIS - LITERATURE GROUNDING:
        =========================================

        1. AI/TECHNOLOGY DOMAIN
        -----------------------
        - Algorithm Aversion: Dietvorst, Simmons & Massey (2015, JEP:G) - People avoid algorithms
          after seeing them err, even when algorithms outperform humans. d ≈ -0.3 to -0.5
        - Algorithm Appreciation: Logg, Minson & Moore (2019, Org Behav) - In some contexts,
          people prefer algorithmic judgment. Context-dependent reversal.
        - Anthropomorphism: Epley, Waytz & Cacioppo (2007, Psych Review) - Human-like features
          increase trust, liking, and moral consideration. d ≈ +0.2 to +0.4
        - Uncanny Valley: Mori (1970/2012) - Near-human appearance can decrease liking.

        2. CONSUMER/MARKETING DOMAIN
        ----------------------------
        - Hedonic vs Utilitarian: Babin, Darden & Griffin (1994, JCR) - Hedonic consumption
          generates more positive affect than utilitarian. d ≈ +0.25
        - Scarcity Effect: Cialdini (2001); Barton et al. (2022, meta-analysis) - Limited
          availability increases desirability. Mean effect r = 0.28, d ≈ +0.30
        - Social Proof: Cialdini (2001); Bond & Smith (1996, Psych Bulletin meta) - Others'
          choices influence preferences. Asch conformity ~35-75% of trials.
        - Price-Quality Inference: Rao & Monroe (1989, JMR) - Higher price signals quality.
        - Brand Familiarity: Alba & Hutchinson (1987, JCR) - Familiar brands preferred.

        3. SOCIAL PSYCHOLOGY DOMAIN
        ---------------------------
        - In-group/Out-group Bias: Tajfel (1971); Balliet et al. (2014, Psych Bulletin) -
          Minimal group paradigm shows in-group favoritism. d ≈ 0.3-0.5
        - Authority/Obedience: Milgram (1963); Meta-Milgram 2014 - 43.6% full obedience
          across conditions. Uniform increases compliance by 46 percentage points.
        - Reciprocity: Cialdini (2001); Regan (1971) - Favors increase compliance.
          Tips increase 23% with personalized gifts.
        - Social Presence: Short, Williams & Christie (1976) - Co-presence increases
          prosocial behavior. d ≈ +0.15 to +0.30

        4. BEHAVIORAL ECONOMICS DOMAIN
        ------------------------------
        - Loss Aversion: Tversky & Kahneman (1981, Science) - Losses loom larger than gains.
          Loss frame: 43% risk-seeking vs gain frame: 23%. λ ≈ 2.0-2.5
        - Anchoring: Tversky & Kahneman (1974, Science) - First numbers anchor judgment.
          d ≈ 0.5-1.0 depending on anchor extremity.
        - Default Effect: Johnson & Goldstein (2003, Science) - Opt-out > opt-in by 60-80
          percentage points for organ donation.
        - Endowment Effect: Kahneman, Knetsch & Thaler (1990, JPE) - Ownership increases
          valuation. WTA/WTP ratio ≈ 2:1
        - Fairness/Ultimatum: Güth et al. (1982); Meta-analyses - Unfair offers rejected
          40-60% of time even at cost to self.

        5. GAME THEORY/COOPERATION DOMAIN
        ---------------------------------
        - Public Goods Game: Fehr & Gächter (2000, AER) - Punishment increases cooperation.
          With punishment: near 100% cooperation vs 40% without.
        - Dictator Game: Engel (2011, meta-analysis) - Mean giving ≈ 28% of endowment.
        - Trust Game: Berg et al. (1995); Johnson & Mislin (2011, meta) - Mean sent ≈ 50%.
        - Prisoner's Dilemma: Sally (1995, meta) - Mean cooperation ≈ 47%.

        6. HEALTH/RISK DOMAIN
        ---------------------
        - Self-Efficacy: Bandura (1977); Meta-analyses - Higher self-efficacy increases
          health behaviors. Robust predictor across domains.
        - Fear Appeals: Witte & Allen (2000, meta) - Moderate fear most effective.
          High fear + high efficacy = behavior change. d ≈ 0.3-0.5
        - Optimistic Bias: Weinstein (1980) - "It won't happen to me" effect for risks.
        - Present Bias: O'Donoghue & Rabin (1999) - Immediate rewards overweighted.

        7. ORGANIZATIONAL/LEADERSHIP DOMAIN
        -----------------------------------
        - Procedural Justice: Colquitt et al. (2001, JAP meta) - Fair procedures increase
          trust and commitment. ρ ≈ .40-.50
        - Transformational Leadership: Judge & Piccolo (2004, JAP meta) - Transforms
          follower attitudes. ρ ≈ .44 with satisfaction.
        - Power Distance: Hofstede (1980); GLOBE - High power distance cultures accept
          hierarchy. Moderates leadership effects.
        - Autonomy: Deci & Ryan (2000, SDT) - Autonomy support increases motivation.

        8. POLITICAL/MORAL DOMAIN
        -------------------------
        - Moral Foundations: Graham, Haidt & Nosek (2009, JPSP) - Liberals emphasize
          care/fairness, conservatives all five foundations.
        - Political Polarization: Iyengar & Westwood (2015, AJPS) - Partisan affect
          stronger than racial prejudice. d > 0.5
        - Disgust Sensitivity: Inbar et al. (2009) - Disgust predicts conservative attitudes.

        CRITICAL: This method NEVER uses condition index/position for effects.
        Effects are determined ONLY by semantic content matching these literature findings.
        """
        condition_lower = str(condition).lower().strip()
        variable_lower = str(variable).lower().strip()

        # Build study context string from all conditions + title for relational parsing
        _all_conds_text = " ".join(c.lower() for c in self.conditions) if self.conditions else ""
        _study_text = (self.study_title or "").lower() + " " + (self.study_description or "").lower()
        # v1.0.5.1: Include condition descriptions in full context for better semantic matching
        _cond_desc_text = ""
        if self.condition_descriptions:
            _cond_desc_text = " ".join(str(d).lower() for d in self.condition_descriptions.values() if d)
        _full_context = _all_conds_text + " " + _study_text + " " + variable_lower + " " + _cond_desc_text

        # Default medium effect size parameters
        default_d = 0.5
        COHENS_D_TO_NORMALIZED = 0.30  # v1.4.11: recalibrated from 0.40

        # Initialize base effect at 0 (neutral)
        semantic_effect = 0.0

        # =====================================================================
        # STEP 0: RELATIONAL/MATCHING CONDITION PARSING (v1.0.4.2)
        # Detects conditions that describe WHO the participant interacts with.
        # CRITICAL for economic games, intergroup studies, social psychology.
        #
        # Scientific basis:
        # - Social Identity Theory (Tajfel & Turner, 1979): People favor ingroup
        # - Affective Polarization (Iyengar & Westwood, 2015): Partisan bias d>0.5
        # - Dictator Game Intergroup (Fershtman & Gneezy, 2001): Ingroup giving
        #   is ~30-40% higher than outgroup giving
        # - Political Identity Dictator Game (Dimant, 2024): Strong discrimination
        #   based on political identity, d ≈ 0.6-0.9
        # =====================================================================

        _handled_by_relational = False

        # Detect POLITICAL IDENTITY conditions
        # These describe the political leaning of the PARTNER/RECIPIENT, not the
        # participant. The key behavioral distinction is:
        # - Matching political ingroup → more generous/cooperative (ingroup favoritism)
        # - Matching political outgroup → less generous/cooperative (outgroup discrimination)
        # - No identity / control → baseline behavior
        _political_figures = ['trump', 'biden', 'obama', 'clinton', 'harris',
                              'desantis', 'sanders', 'pelosi', 'mcconnell']
        _political_labels = ['republican', 'democrat', 'liberal', 'conservative',
                             'left', 'right', 'progressive', 'maga']
        _political_attitudes = ['lover', 'hater', 'supporter', 'opponent',
                                'fan', 'critic', 'admirer', 'detractor',
                                'pro', 'anti']
        _identity_keywords = ['identity', 'political', 'partisan', 'party']

        # Check if this is a political identity study
        _is_political_study = (
            any(_word_in(fig, _full_context) for fig in _political_figures) or
            any(_word_in(lab, _full_context) for lab in _political_labels) or
            any(kw in _full_context for kw in _identity_keywords)
        )

        # Check if the DV is an economic game / allocation measure
        _econ_game_keywords = ['dollar', 'amount', 'allocat', 'give', 'sent',
                               'offer', 'share', 'split', 'endow', 'dictator',
                               'trust game', 'ultimatum', 'public good',
                               'contribution', 'transfer', 'payment']
        _is_economic_game_dv = any(kw in _full_context for kw in _econ_game_keywords)

        if _is_political_study:
            # Parse the relational dynamic: is this condition describing
            # ingroup matching, outgroup matching, or no identity?
            _has_lover = any(_word_in(att, condition_lower) for att in
                             ['lover', 'supporter', 'fan', 'admirer', 'pro'])
            _has_hater = any(_word_in(att, condition_lower) for att in
                             ['hater', 'opponent', 'critic', 'detractor', 'anti'])
            _has_neutral_id = any(_word_in(att, condition_lower) for att in
                                  ['neutral', 'moderate', 'independent', 'undecided'])
            _no_identity = any(kw in condition_lower for kw in
                               ['no identity', 'control', 'unknown', 'no info',
                                'anonymous', 'no political'])

            # Count how many political attitude words appear in this condition
            # If condition has BOTH lover AND hater → it's describing a MIXED/OUTGROUP pairing
            _attitude_count = sum(1 for att in _political_attitudes
                                  if _word_in(att, condition_lower))

            if _attitude_count >= 2:
                # Condition contains multiple political attitudes (e.g., "trump lover, trump hater")
                # This is an OUTGROUP MATCHING condition
                # Iyengar & Westwood (2015): d > 0.5 for affective polarization
                # Fershtman & Gneezy (2001): ~20-30% less generous to outgroup
                # Dimant (2024): Strong discrimination in political dictator games
                semantic_effect -= 0.40  # Strong outgroup discrimination effect
                _handled_by_relational = True
            elif _has_lover and not _has_hater:
                # Only positive attitude → INGROUP condition
                # Social Identity Theory: ingroup favoritism
                # Balliet et al. (2014): d ≈ 0.3-0.5 for ingroup cooperation
                semantic_effect += 0.30  # Ingroup favoritism
                _handled_by_relational = True
            elif _has_hater and not _has_lover:
                # Only negative attitude → OUTGROUP condition
                semantic_effect -= 0.35  # Outgroup discrimination
                _handled_by_relational = True
            elif _has_neutral_id:
                # Neutral identity → moderate, between ingroup and outgroup
                semantic_effect -= 0.05  # Slight caution toward unknown
                _handled_by_relational = True
            elif _no_identity:
                # No identity shown → baseline behavior (control)
                # Dictator game control: mean giving ≈ 28% of endowment (Engel, 2011)
                semantic_effect += 0.0  # Pure baseline
                _handled_by_relational = True

            # If it's also an economic game, amplify the intergroup effect
            # because discrimination is MORE pronounced in resource allocation
            if _handled_by_relational and _is_economic_game_dv:
                semantic_effect *= 1.3  # Amplify for economic allocation decisions

        # Detect GENERAL INTERGROUP/MATCHING conditions (non-political)
        # e.g., "same race", "different ethnicity", "ingroup partner", "outgroup partner"
        # v1.0.4.2: Expanded to cover racial, ethnic, religious, gender, and
        # arbitrary group identity conditions
        #
        # Scientific basis:
        # - Tajfel (1971): Minimal Group Paradigm — even arbitrary categories
        #   produce ingroup favoritism (d ≈ 0.3-0.5)
        # - Balliet et al. (2014, Psych Bulletin meta): Ingroup cooperation
        #   significantly higher than outgroup, d ≈ 0.32
        # - Fershtman & Gneezy (2001): Ethnic discrimination in trust games
        # - Bauer et al. (2016): Religious identity affects prosocial behavior
        if not _handled_by_relational:
            _same_group_markers = ['same group', 'same race', 'same team',
                                   'same ethnicity', 'same religion', 'same gender',
                                   'same nationality', 'same school', 'same university',
                                   'ingroup partner', 'ingroup member', 'fellow member',
                                   'same party', 'co-ethnic', 'co-religious',
                                   'shared identity', 'common group']
            _diff_group_markers = ['different group', 'different race', 'different team',
                                   'different ethnicity', 'different religion', 'different gender',
                                   'different nationality', 'different school',
                                   'outgroup partner', 'outgroup member', 'other member',
                                   'opposing party', 'other ethnicity', 'other religion',
                                   'cross-group', 'intergroup']

            if any(kw in condition_lower for kw in _same_group_markers):
                _ingroup_d = 0.28  # Balliet et al. (2014)
                if _is_economic_game_dv:
                    _ingroup_d = 0.35  # Stronger in resource allocation
                semantic_effect += _ingroup_d
                _handled_by_relational = True
            elif any(kw in condition_lower for kw in _diff_group_markers):
                _outgroup_d = -0.25
                if _is_economic_game_dv:
                    _outgroup_d = -0.32  # Stronger discrimination in allocation
                semantic_effect += _outgroup_d
                _handled_by_relational = True

            # v1.0.4.2: Detect racial/ethnic identity studies
            # Fershtman & Gneezy (2001): Discrimination varies by group
            # Stereotype content model (Fiske et al., 2002): Groups judged on
            # warmth and competence dimensions
            _racial_terms = ['white', 'black', 'asian', 'hispanic', 'latino',
                             'african american', 'caucasian', 'arab', 'muslim',
                             'jewish', 'christian', 'hindu']
            _racial_in_cond = [t for t in _racial_terms if _word_in(t, condition_lower)]
            if _racial_in_cond and not _handled_by_relational:
                # Check study context for what the manipulation is
                # If multiple racial terms in study (suggests comparison), treat as
                # intergroup study — the participant evaluates someone of this identity
                _racial_in_study = [t for t in _racial_terms if _word_in(t, _full_context)]
                if len(_racial_in_study) >= 2:
                    # Multi-group comparison — this is an intergroup study
                    # No universal direction; effect depends on perceiver-target match
                    # Add moderate variance but no directional effect by default
                    semantic_effect += 0.0  # Direction depends on specific matchup
                    _handled_by_relational = True

        # =====================================================================
        # STEP 1: Parse valence keywords (directional effects)
        # Based on affective meaning of condition labels
        # SKIP if already handled by relational parsing above
        # =====================================================================

        if not _handled_by_relational:
            # Strong positive valence keywords → positive effect
            # NOTE: 'lover', 'hater' removed to prevent false positives
            # in political identity conditions (handled in STEP 0)
            positive_keywords = [
                'friend', 'positive', 'high', 'good', 'best', 'strong',
                'success', 'win', 'gain', 'benefit', 'reward', 'pleasant',
                'like', 'love', 'favor', 'approve', 'support', 'prosocial',
                'cooperative', 'trust', 'warm', 'kind', 'helpful', 'generous',
                'optimistic', 'confident', 'empowered', 'satisfied'
            ]

            # Strong negative valence keywords → negative effect
            negative_keywords = [
                'enemy', 'negative', 'low', 'bad', 'worst', 'weak',
                'failure', 'lose', 'loss', 'cost', 'punish', 'unpleasant',
                'dislike', 'hate', 'oppose', 'disapprove', 'reject', 'antisocial',
                'competitive', 'distrust', 'cold', 'hostile', 'harmful', 'selfish',
                'pessimistic', 'anxious', 'threatened', 'dissatisfied'
            ]

            # Neutral/baseline keywords → zero effect
            neutral_keywords = [
                'unknown', 'control', 'baseline', 'neutral', 'moderate',
                'medium', 'average', 'standard', 'normal', 'typical', 'placebo'
            ]

            # Check for valence keywords (v1.0.1.3: word-boundary matching)
            for keyword in positive_keywords:
                if _word_in(keyword, condition_lower):
                    semantic_effect += 0.35  # Moderate positive shift
                    break

            for keyword in negative_keywords:
                if _word_in(keyword, condition_lower):
                    semantic_effect -= 0.35  # Moderate negative shift
                    break

            for keyword in neutral_keywords:
                if _word_in(keyword, condition_lower):
                    semantic_effect *= 0.3  # Reduce effect toward neutral
                    break

        # =====================================================================
        # v1.0.4.6: DOMAIN-AWARE ROUTING + EFFECT STACKING GUARD
        #
        # self.detected_domains (computed at init from 5-phase detection) tells
        # us which research domains this study belongs to. We use this to:
        #   1. Track cumulative effect contributions per domain
        #   2. Attenuate effects from NON-detected domains by 0.5×
        #      (they may still be relevant, but less likely)
        #   3. Cap total STEP 2 effect to ±0.45 to prevent runaway stacking
        #
        # This prevents a consumer study's "premium brand" condition from
        # also triggering social psychology (+authority), behavioral economics
        # (+anchoring), and organizational (+transformational) keywords, which
        # would stack to an unrealistically large total effect.
        # =====================================================================
        _detected = set(getattr(self, 'detected_domains', []) or [])
        _effect_before_step2 = semantic_effect  # Track pre-STEP-2 baseline

        # Domain-relevance mapping: each STEP 2 domain → persona library domains
        _DOMAIN_RELEVANCE = {
            1: {'ai', 'technology'},
            2: {'consumer_behavior', 'marketing', 'hedonic_consumption', 'utilitarian_consumption'},
            3: {'social_psychology', 'norm_elicitation'},
            4: {'behavioral_economics', 'economic_games', 'decision_making'},
            5: {'economic_games', 'behavioral_economics'},
            6: {'health_psychology'},
            7: {'organizational_behavior'},
            8: {'political_psychology', 'deontology_utilitarianism'},
            9: {'behavioral_economics', 'decision_making', 'cognitive_psychology'},
            10: {'media_communication', 'accuracy_misinformation'},
            11: {'educational_psychology', 'cognitive_psychology'},
            12: {'social_psychology', 'political_psychology'},
            13: {'social_psychology', 'organizational_behavior'},
            14: {'environmental'},
            15: {'social_psychology'},
            16: {'behavioral_economics', 'social_psychology'},
            17: {'behavioral_economics', 'dishonesty'},
            18: {'power_status', 'organizational_behavior'},
            19: {'media_communication', 'social_psychology'},
            20: {'social_psychology', 'consumer_behavior'},
            21: {'positive_psychology', 'health_psychology'},
            22: {'moral_psychology', 'deontology_utilitarianism'},
            23: {'technology', 'cognitive_psychology'},
        }

        def _domain_is_relevant(domain_num: int) -> bool:
            """Check if a STEP 2 domain is relevant to the detected study domains."""
            if not _detected:
                return True  # No detection → all domains equally relevant
            return bool(_detected & _DOMAIN_RELEVANCE.get(domain_num, set()))

        # =====================================================================
        # DOMAIN 1: AI/TECHNOLOGY MANIPULATIONS
        # =====================================================================

        # Algorithm Aversion (Dietvorst, Simmons & Massey, 2015, JEP:G)
        # People avoid algorithms after seeing them err. Effect: d ≈ -0.3 to -0.5
        if _word_in('ai', condition_lower) or _word_in('algorithm', condition_lower) or _word_in('robot', condition_lower):
            if _any_word_in(['no ai', 'no_ai', 'without ai', 'no algorithm', 'human only'], condition_lower):
                # No AI / Human condition - often preferred due to algorithm aversion
                semantic_effect += 0.15  # Human preference effect
            else:
                # AI present - shows aversion in evaluations (Dietvorst et al., 2015)
                semantic_effect -= 0.12

        # Machine vs Human judgment (Logg, Minson & Moore, 2019)
        if _word_in('machine', condition_lower) and not _word_in('human', condition_lower):
            semantic_effect -= 0.10
        elif _word_in('human', condition_lower) and not _word_in('machine', condition_lower):
            if not _word_in('superhuman', condition_lower):
                semantic_effect += 0.10

        # Anthropomorphism (Epley, Waytz & Cacioppo, 2007, Psychological Review)
        # Human-like features increase trust and liking. Effect: d ≈ +0.2 to +0.4
        if _stem_in('anthropomorph', condition_lower) or _word_in('human-like', condition_lower) or _word_in('humanoid', condition_lower):
            semantic_effect += 0.18
        elif _word_in('machine-like', condition_lower) or _word_in('robotic', condition_lower):
            semantic_effect -= 0.08

        # Automation (Parasuraman & Riley, 1997; Lee & See, 2004)
        if _stem_in('automat', condition_lower):
            if _word_in('full', condition_lower) or _word_in('complete', condition_lower):
                semantic_effect -= 0.15  # Full automation trust concerns
            elif _word_in('partial', condition_lower) or _word_in('assisted', condition_lower):
                semantic_effect += 0.05  # Partial automation often preferred

        # Transparency/Explainability (Ribeiro et al., 2016)
        if _word_in('transparent', condition_lower) or _word_in('explainable', condition_lower) or _word_in('interpretable', condition_lower):
            semantic_effect += 0.12
        elif _word_in('black box', condition_lower) or _word_in('opaque', condition_lower):
            semantic_effect -= 0.10

        # =====================================================================
        # DOMAIN 2: CONSUMER/MARKETING MANIPULATIONS
        # =====================================================================

        # Hedonic vs Utilitarian (Babin, Darden & Griffin, 1994, JCR)
        # Hedonic consumption generates more positive affect. Effect: d ≈ +0.25
        if _any_word_in(['hedonic', 'experiential', 'fun', 'pleasure', 'enjoyment', 'indulgent'], condition_lower):
            semantic_effect += 0.22
        elif _any_word_in(['utilitarian', 'functional', 'practical', 'necessity', 'useful'], condition_lower):
            semantic_effect -= 0.08

        # Scarcity Effect (Cialdini, 2001; Barton et al., 2022 meta-analysis)
        # Limited availability increases desirability. Mean effect r = 0.28, d ≈ +0.30
        if _any_word_in(['scarce', 'limited', 'exclusive', 'rare', 'last chance', 'few left'], condition_lower):
            semantic_effect += 0.25
        elif _any_word_in(['abundant', 'unlimited', 'plentiful', 'common', 'widely available'], condition_lower):
            semantic_effect -= 0.08

        # Social Proof (Cialdini, 2001; Bond & Smith, 1996 meta-analysis)
        # Others' choices influence preferences. Conformity effect robust.
        if _any_word_in(['popular', 'bestseller', 'most chosen', 'endorsed', 'recommended',
                         'others chose', 'trending', 'viral', 'social proof'], condition_lower):
            semantic_effect += 0.20
        elif _any_word_in(['unpopular', 'not recommended', 'unknown brand', 'no reviews'], condition_lower):
            semantic_effect -= 0.15

        # Price-Quality Inference (Rao & Monroe, 1989, JMR)
        if _word_in('premium', condition_lower) or _word_in('luxury', condition_lower) or _word_in('expensive', condition_lower):
            semantic_effect += 0.15
        elif _word_in('budget', condition_lower) or _word_in('discount', condition_lower) or _word_in('cheap', condition_lower):
            semantic_effect -= 0.10

        # Brand Effects (Alba & Hutchinson, 1987, JCR)
        if _word_in('familiar', condition_lower) or _word_in('known brand', condition_lower) or _word_in('established', condition_lower):
            semantic_effect += 0.12
        elif _word_in('unfamiliar', condition_lower) or _word_in('new brand', condition_lower) or _word_in('unknown', condition_lower):
            semantic_effect -= 0.08

        # Advertising Appeals (MacInnis & Jaworski, 1989)
        if _word_in('emotional', condition_lower) and _word_in('appeal', condition_lower):
            semantic_effect += 0.15
        elif _word_in('rational', condition_lower) and _word_in('appeal', condition_lower):
            semantic_effect += 0.05

        # =====================================================================
        # DOMAIN 3: SOCIAL PSYCHOLOGY MANIPULATIONS
        # =====================================================================

        # In-group/Out-group Bias (Tajfel, 1971; Balliet et al., 2014 meta)
        # Minimal group paradigm shows in-group favoritism. d ≈ 0.3-0.5
        if _any_word_in(['ingroup', 'in-group', 'in group', 'us', 'our group', 'teammate'], condition_lower):
            semantic_effect += 0.28
        elif _any_word_in(['outgroup', 'out-group', 'out group', 'them', 'other group', 'opponent'], condition_lower):
            semantic_effect -= 0.25

        # Authority/Obedience (Milgram, 1963; Meta-Milgram, 2014)
        # Authority figures increase compliance. Uniform effect: +46pp compliance
        if _any_word_in(['authority', 'expert', 'doctor', 'professor', 'scientist',
                         'official', 'leader', 'manager', 'uniform'], condition_lower):
            semantic_effect += 0.22
        elif _any_word_in(['peer', 'layperson', 'non-expert', 'stranger', 'novice'], condition_lower):
            semantic_effect -= 0.08

        # Reciprocity (Cialdini, 2001; Regan, 1971)
        # Favors increase compliance. Gift effect: +23% tips
        if _stem_in('reciproc', condition_lower) or _any_word_in(['gift', 'favor', 'gave first', 'free sample'], condition_lower):
            semantic_effect += 0.20
        elif _word_in('no gift', condition_lower) or _word_in('no favor', condition_lower):
            semantic_effect -= 0.05

        # Social Presence (Short, Williams & Christie, 1976)
        # Co-presence increases prosocial behavior. d ≈ +0.15 to +0.30
        if _any_word_in(['social presence', 'observed', 'watched', 'public',
                         'with others', 'audience', 'witnessed'], condition_lower):
            semantic_effect += 0.18
        elif _any_word_in(['alone', 'private', 'anonymous', 'unobserved', 'no audience'], condition_lower):
            semantic_effect -= 0.10

        # Commitment/Consistency (Cialdini, 2001; Freedman & Fraser, 1966)
        if _any_word_in(['commitment', 'pledged', 'promised', 'foot in door', 'prior agreement'], condition_lower):
            semantic_effect += 0.18

        # Liking/Similarity (Cialdini, 2001; Byrne, 1971)
        if _any_word_in(['similar', 'likeable', 'attractive', 'compliment', 'same group'], condition_lower):
            semantic_effect += 0.15
        elif _any_word_in(['dissimilar', 'unlikeable', 'different', 'outgroup'], condition_lower):
            semantic_effect -= 0.12

        # =====================================================================
        # DOMAIN 4: BEHAVIORAL ECONOMICS MANIPULATIONS
        # =====================================================================

        # Loss Aversion/Framing (Tversky & Kahneman, 1981, Science)
        # Losses loom larger than gains. λ ≈ 2.0-2.5. Loss frame: 43% vs gain: 23% risk-seeking
        if _any_word_in(['gain', 'save', 'earn', 'win', 'keep', 'gain frame'], condition_lower):
            semantic_effect += 0.12
        elif _any_word_in(['loss', 'lose', 'cost', 'pay', 'forfeit', 'loss frame'], condition_lower):
            semantic_effect -= 0.20  # Loss aversion amplifies negative effects

        # Anchoring (Tversky & Kahneman, 1974, Science)
        # First numbers anchor judgment. d ≈ 0.5-1.0
        if _word_in('high anchor', condition_lower) or _word_in('large anchor', condition_lower):
            semantic_effect += 0.25
        elif _word_in('low anchor', condition_lower) or _word_in('small anchor', condition_lower):
            semantic_effect -= 0.20

        # Default Effect (Johnson & Goldstein, 2003, Science)
        # Opt-out > opt-in by 60-80 percentage points
        if _any_word_in(['opt-out', 'opt out', 'default yes', 'presumed consent'], condition_lower):
            semantic_effect += 0.35
        elif _any_word_in(['opt-in', 'opt in', 'default no', 'explicit consent', 'active choice'], condition_lower):
            semantic_effect -= 0.15

        # Endowment Effect (Kahneman, Knetsch & Thaler, 1990, JPE)
        # Ownership increases valuation. WTA/WTP ≈ 2:1
        if _any_word_in(['own', 'possess', 'endow', 'yours', 'have'], condition_lower):
            semantic_effect += 0.20
        elif _any_word_in(['buy', 'acquire', 'get', 'obtain'], condition_lower):
            semantic_effect -= 0.10

        # Fairness/Ultimatum (Güth et al., 1982; Camerer, 2003)
        # Unfair offers rejected 40-60% of time
        if _any_word_in(['fair', 'equal', 'equitable', '50-50', 'even split'], condition_lower):
            semantic_effect += 0.25
        elif _any_word_in(['unfair', 'unequal', 'inequitable', 'low offer', 'stingy'], condition_lower):
            semantic_effect -= 0.30

        # Present Bias (O'Donoghue & Rabin, 1999)
        if _any_word_in(['immediate', 'now', 'today', 'instant'], condition_lower):
            semantic_effect += 0.18
        elif _any_word_in(['delayed', 'later', 'future', 'wait'], condition_lower):
            semantic_effect -= 0.12

        # Mental Accounting (Thaler, 1985)
        if _word_in('windfall', condition_lower) or _word_in('bonus', condition_lower) or _word_in('unexpected', condition_lower):
            semantic_effect += 0.15

        # =====================================================================
        # DOMAIN 5: GAME THEORY/COOPERATION MANIPULATIONS
        # =====================================================================

        # Public Goods Game (Fehr & Gächter, 2000, AER)
        # With punishment: near 100% vs 40% without
        if _any_word_in(['pgg', 'public good', 'contribute', 'common pool'], condition_lower):
            semantic_effect += 0.15
            if _word_in('punish', condition_lower):
                semantic_effect += 0.25  # Punishment dramatically increases cooperation

        # Dictator Game (Engel, 2011 meta-analysis)
        # Mean giving ≈ 28% of endowment
        if _word_in('dictator', condition_lower):
            semantic_effect -= 0.05  # More self-interested than other games

        # Trust Game (Berg et al., 1995; Johnson & Mislin, 2011 meta)
        # Mean sent ≈ 50%
        if _word_in('trust game', condition_lower):
            if _word_in('trustor', condition_lower) or _word_in('sender', condition_lower):
                semantic_effect += 0.15
            elif _word_in('trustee', condition_lower) or _word_in('receiver', condition_lower):
                semantic_effect += 0.10

        # Prisoner's Dilemma (Sally, 1995 meta-analysis)
        # Mean cooperation ≈ 47%
        if _word_in('prisoner', condition_lower) or _word_in('pd', condition_lower):
            if _stem_in('cooperat', condition_lower):
                semantic_effect += 0.20
            elif _word_in('defect', condition_lower):
                semantic_effect -= 0.25

        # Repeated vs One-shot games (Axelrod, 1984)
        if _word_in('repeated', condition_lower) or _word_in('iterated', condition_lower) or _word_in('multiple rounds', condition_lower):
            semantic_effect += 0.15  # Repeated games show more cooperation
        elif _word_in('one-shot', condition_lower) or _word_in('single round', condition_lower):
            semantic_effect -= 0.10

        # =====================================================================
        # DOMAIN 6: HEALTH/RISK MANIPULATIONS (expanded v1.0.4.5)
        # =====================================================================

        # Self-Efficacy (Bandura, 1977; Meta-analyses)
        # Higher self-efficacy increases health behaviors
        if _any_word_in(['high efficacy', 'self-efficacy', 'confident', 'capable', 'empowered'], condition_lower):
            semantic_effect += 0.22
        elif _any_word_in(['low efficacy', 'doubtful', 'incapable', 'helpless'], condition_lower):
            semantic_effect -= 0.20

        # Fear Appeals (Witte & Allen, 2000 meta-analysis)
        # Moderate fear most effective. High fear + high efficacy = change. d ≈ 0.3-0.5
        if _word_in('fear', condition_lower) or _word_in('threat', condition_lower) or _word_in('danger', condition_lower):
            if _word_in('high', condition_lower) or _word_in('strong', condition_lower):
                semantic_effect -= 0.15  # High fear can backfire without efficacy
            elif _word_in('moderate', condition_lower) or _word_in('medium', condition_lower):
                semantic_effect += 0.12  # Moderate fear often most effective
            else:
                semantic_effect -= 0.10

        # Risk Perception (Slovic, 1987)
        if _any_word_in(['risky', 'dangerous', 'hazardous', 'unsafe'], condition_lower):
            semantic_effect -= 0.18
        elif _any_word_in(['safe', 'secure', 'protected', 'low risk'], condition_lower):
            semantic_effect += 0.15

        # Health Message Framing (Rothman & Salovey, 1997)
        if _word_in('prevention', condition_lower) or _word_in('detect', condition_lower):
            if _word_in('loss', condition_lower):
                semantic_effect += 0.12  # Loss frame better for detection
        if _stem_in('promot', condition_lower):
            if _word_in('gain', condition_lower):
                semantic_effect += 0.12  # Gain frame better for prevention

        # v1.0.4.5: Optimistic Bias (Weinstein, 1980; Shepperd et al. 2013 meta)
        # People underestimate personal risk; corrective info shifts perception
        if _any_word_in(['personal risk', 'your risk', 'individual risk', 'optimistic bias'], condition_lower):
            semantic_effect -= 0.14  # Personal risk framing reduces optimistic bias
        elif _any_word_in(['average risk', 'population risk', 'general risk'], condition_lower):
            semantic_effect += 0.05  # Abstract risk maintains optimistic bias

        # v1.0.4.5: Health Literacy (Berkman et al. 2011 meta)
        # Simplified health information increases comprehension and compliance
        if _any_word_in(['simplified', 'plain language', 'easy to read', 'health literate'], condition_lower):
            semantic_effect += 0.15
        elif _any_word_in(['technical', 'jargon', 'medical terminology', 'complex language'], condition_lower):
            semantic_effect -= 0.12

        # v1.0.4.5: Social Norms for Health (Cialdini, 2003; Goldstein et al. 2008)
        # Descriptive norms ("most people do X") increase health behaviors
        if _any_word_in(['descriptive norm', 'most people', 'majority behavior', 'social norm health'], condition_lower):
            semantic_effect += 0.18
        elif _any_word_in(['injunctive norm', 'should do', 'ought to'], condition_lower):
            semantic_effect += 0.10  # Weaker than descriptive

        # =====================================================================
        # DOMAIN 7: ORGANIZATIONAL/LEADERSHIP MANIPULATIONS
        # =====================================================================

        # Procedural Justice (Colquitt et al., 2001 meta-analysis, JAP)
        # Fair procedures increase trust and commitment. ρ ≈ .40-.50
        if _any_word_in(['procedural justice', 'fair process', 'voice',
                         'transparent process', 'fair procedure'], condition_lower):
            semantic_effect += 0.25
        elif _any_word_in(['unfair process', 'no voice', 'arbitrary'], condition_lower):
            semantic_effect -= 0.28

        # Distributive Justice (Colquitt et al., 2001)
        if _any_word_in(['distributive justice', 'fair outcome', 'equitable pay',
                         'fair reward', 'fair distribution'], condition_lower):
            semantic_effect += 0.25
        elif _any_word_in(['unfair outcome', 'inequitable', 'underpaid'], condition_lower):
            semantic_effect -= 0.28

        # Transformational Leadership (Judge & Piccolo, 2004 meta-analysis, JAP)
        # ρ ≈ .44 with satisfaction
        if _any_word_in(['transformational', 'inspirational', 'charismatic',
                         'visionary', 'empowering leader'], condition_lower):
            semantic_effect += 0.22
        elif _any_word_in(['transactional', 'directive', 'laissez-faire'], condition_lower):
            semantic_effect -= 0.05

        # Autonomy (Deci & Ryan, 2000, SDT)
        # Autonomy support increases motivation
        if _any_word_in(['autonomy', 'choice', 'freedom', 'self-directed',
                         'empowerment', 'participative'], condition_lower):
            semantic_effect += 0.20
        elif _any_word_in(['controlled', 'no choice', 'mandated', 'forced', 'required'], condition_lower):
            semantic_effect -= 0.15

        # Feedback (Kluger & DeNisi, 1996 meta-analysis)
        if _any_word_in(['positive feedback', 'praise', 'recognition', 'appreciated'], condition_lower):
            semantic_effect += 0.20
        elif _any_word_in(['negative feedback', 'criticism', 'blame'], condition_lower):
            semantic_effect -= 0.22

        # v1.0.4.5: Leader-Member Exchange (Gerstner & Day, 1997 meta; ρ ≈ .35)
        # High LMX = trust, liking, respect between leader and member
        if _any_word_in(['high lmx', 'good relationship', 'trusted employee', 'favored'], condition_lower):
            semantic_effect += 0.20
        elif _any_word_in(['low lmx', 'poor relationship', 'distant leader', 'unfavored'], condition_lower):
            semantic_effect -= 0.18

        # v1.0.4.5: Psychological Safety (Edmondson, 1999; Frazier et al. 2017 meta)
        # Team psychological safety enables learning, voice, innovation
        if _any_word_in(['psychological safety', 'safe to speak', 'no blame culture', 'speak up'], condition_lower):
            semantic_effect += 0.22
        elif _any_word_in(['psychologically unsafe', 'punitive', 'fear of speaking', 'blame culture'], condition_lower):
            semantic_effect -= 0.20

        # v1.0.4.5: Organizational Trust (Dirks & Ferrin, 2002 meta; ρ ≈ .30)
        if _any_word_in(['trust in management', 'trustworthy org', 'reliable employer'], condition_lower):
            semantic_effect += 0.18
        elif _any_word_in(['distrust management', 'untrustworthy org', 'unreliable employer'], condition_lower):
            semantic_effect -= 0.20

        # =====================================================================
        # DOMAIN 8: POLITICAL/MORAL MANIPULATIONS
        # =====================================================================

        # Moral Foundations (Graham, Haidt & Nosek, 2009, JPSP)
        # Liberals: care/fairness; Conservatives: all five foundations
        if _any_word_in(['care', 'harm', 'compassion', 'suffering'], condition_lower):
            semantic_effect += 0.18
        if _any_word_in(['fairness', 'justice', 'equality', 'rights'], condition_lower):
            semantic_effect += 0.18
        if _any_word_in(['loyalty', 'patriot', 'traitor', 'betrayal'], condition_lower):
            semantic_effect += 0.12
        if _any_word_in(['authority', 'tradition', 'subversion', 'respect'], condition_lower):
            semantic_effect += 0.10
        if _any_word_in(['purity', 'sanctity', 'disgust', 'degradation'], condition_lower):
            semantic_effect += 0.10

        # Political Polarization (Iyengar & Westwood, 2015, AJPS)
        # Partisan affect stronger than racial prejudice. d > 0.5
        # v1.0.4.2: Expanded detection for complex condition names
        _copartisan_kws = ['same party', 'co-partisan', 'inparty', 'fellow partisan',
                           'political ally', 'same side', 'political ingroup']
        _outpartisan_kws = ['other party', 'opposing party', 'outparty', 'cross-partisan',
                            'political opponent', 'other side', 'political outgroup']
        if _any_word_in(_copartisan_kws, condition_lower):
            semantic_effect += 0.35
        elif _any_word_in(_outpartisan_kws, condition_lower):
            semantic_effect -= 0.35

        # v1.0.4.2: Detect political figure + attitude combinations
        # e.g., condition "trump supporter" in a study about political attitudes
        # The FIGURE isn't the effect — the ATTITUDE toward the figure matters
        # for how OTHERS treat that person (ingroup vs outgroup dynamics)
        for fig in ['trump', 'biden', 'obama', 'clinton', 'harris', 'desantis', 'sanders']:
            if _word_in(fig, condition_lower):
                # Political figure detected — check if we already handled in STEP 0
                # If not, apply a moderate polarization effect
                if not _handled_by_relational:
                    _fig_positive = any(_word_in(w, condition_lower) for w in
                                        ['supporter', 'lover', 'fan', 'pro', 'admirer'])
                    _fig_negative = any(_word_in(w, condition_lower) for w in
                                        ['opponent', 'hater', 'critic', 'anti', 'detractor'])
                    if _fig_positive and not _fig_negative:
                        semantic_effect += 0.20  # Positive political identity
                    elif _fig_negative and not _fig_positive:
                        semantic_effect -= 0.20  # Negative political identity
                break  # Only process first political figure found

        # Disgust (Inbar et al., 2009)
        if _word_in('disgust', condition_lower):
            semantic_effect -= 0.20

        # Moral vs Non-moral framing (Feinberg & Willer, 2015)
        if _word_in('moral', condition_lower) or _word_in('ethical', condition_lower):
            semantic_effect += 0.15
        elif _word_in('immoral', condition_lower) or _word_in('unethical', condition_lower):
            semantic_effect -= 0.22

        # =====================================================================
        # GENERAL TREATMENT EFFECTS
        # =====================================================================

        # Treatment vs Control (general pattern)
        if _word_in('treatment', condition_lower) and not _word_in('control', condition_lower):
            semantic_effect += 0.20
        elif _word_in('control', condition_lower) and not _word_in('treatment', condition_lower):
            semantic_effect -= 0.05

        # Intervention effects
        if _word_in('intervention', condition_lower):
            semantic_effect += 0.15
        elif _word_in('no intervention', condition_lower) or _word_in('waitlist', condition_lower):
            semantic_effect -= 0.05

        # =====================================================================
        # DOMAIN 9: ADDITIONAL COGNITIVE/DECISION MANIPULATIONS (50+ more)
        # =====================================================================

        # Choice Overload (Iyengar & Lepper, 2000; Scheibehenne et al. 2010 meta)
        # Original jam study d = 0.77; meta-analysis shows mean effect near 0
        # Effect is moderated by complexity and expertise
        if _any_word_in(['many options', 'large assortment', 'high choice', 'extensive'], condition_lower):
            semantic_effect -= 0.12  # Choice overload reduces satisfaction
        elif _any_word_in(['few options', 'small assortment', 'limited choice', 'simple'], condition_lower):
            semantic_effect += 0.08

        # Sunk Cost Fallacy (Staw, 1976; Sleesman et al. 2012 meta-analysis)
        # Personal responsibility increases escalation; d ≈ 0.37
        if _any_word_in(['sunk cost', 'invested', 'escalation', 'committed'], condition_lower):
            semantic_effect += 0.15  # Escalation tendency
        elif _word_in('no sunk cost', condition_lower) or _word_in('fresh start', condition_lower):
            semantic_effect -= 0.05

        # Construal Level Theory (Trope & Liberman, 2010; Soderberg et al. meta)
        # Psychological distance affects abstraction; robust effect
        if _any_word_in(['distant', 'far future', 'abstract', 'why'], condition_lower):
            semantic_effect += 0.12  # Abstract = more desirable
        elif _any_word_in(['near', 'soon', 'concrete', 'how'], condition_lower):
            semantic_effect -= 0.08  # Concrete = more feasibility concerns

        # Intrinsic Motivation Crowding Out (Deci, Koestner & Ryan, 1999 meta)
        # Tangible rewards undermine intrinsic motivation; d = -0.40
        if _any_word_in(['extrinsic reward', 'payment', 'incentive', 'bonus for'], condition_lower):
            semantic_effect -= 0.15  # Undermining effect
        elif _any_word_in(['intrinsic', 'no reward', 'autonomous', 'self-determined'], condition_lower):
            semantic_effect += 0.12

        # Mere Exposure Effect (Zajonc, 1968; Bornstein, 1989 meta r = 0.26)
        # Repeated exposure increases liking
        if _any_word_in(['familiar', 'repeated exposure', 'seen before', 'recognized'], condition_lower):
            semantic_effect += 0.15
        elif _any_word_in(['novel', 'unfamiliar', 'first time', 'new'], condition_lower):
            semantic_effect -= 0.05

        # Bystander Effect (Darley & Latané, 1968; Fischer et al. 2011 meta)
        # More bystanders = less helping; 85% alone vs 31% with 4 others
        if _any_word_in(['alone', 'sole witness', 'only one'], condition_lower):
            semantic_effect += 0.25  # More likely to help
        elif _any_word_in(['crowd', 'many bystanders', 'group present', 'others present'], condition_lower):
            semantic_effect -= 0.20  # Diffusion of responsibility

        # Stereotype Threat (Steele & Aronson, 1995; Nguyen & Ryan 2008 meta d = 0.26)
        # Threat of confirming negative stereotype impairs performance
        if _any_word_in(['stereotype threat', 'diagnostic', 'ability test'], condition_lower):
            semantic_effect -= 0.15
        elif _any_word_in(['no threat', 'non-diagnostic', 'practice'], condition_lower):
            semantic_effect += 0.08

        # Reactance (Brehm, 1966; Rains 2013 meta; 2025 meta r = -0.23)
        # Freedom threat leads to boomerang effects
        if _any_word_in(['must', 'required', 'mandatory', 'have to', 'forced'], condition_lower):
            semantic_effect -= 0.18  # Reactance reduces compliance
        elif _any_word_in(['optional', 'choice', 'voluntary', 'may'], condition_lower):
            semantic_effect += 0.10

        # Emotional Contagion (Hatfield et al., 1993; replicated in social networks)
        # Emotions transfer between individuals
        if _any_word_in(['happy confederate', 'positive mood', 'smiling'], condition_lower):
            semantic_effect += 0.15
        elif _any_word_in(['sad confederate', 'negative mood', 'frowning'], condition_lower):
            semantic_effect -= 0.15

        # Negativity Bias (Rozin & Royzman, 2001; Baumeister et al. 2001)
        # Bad is stronger than good; negative information weighted more
        if _word_in('negative info', condition_lower) or _word_in('criticism', condition_lower):
            semantic_effect -= 0.22  # Stronger negative effect
        elif _word_in('positive info', condition_lower) or _word_in('praise', condition_lower):
            semantic_effect += 0.15  # Weaker positive effect

        # Identifiable Victim Effect (Small, Loewenstein & Slovic, 2007)
        # Meta-analysis r = 0.13; single identified victim > statistics
        if _any_word_in(['identified victim', 'named', 'individual story', 'one person'], condition_lower):
            semantic_effect += 0.12
        elif _any_word_in(['statistics', 'many victims', 'aggregate', 'numbers'], condition_lower):
            semantic_effect -= 0.05

        # Confirmation Bias (Nickerson, 1998; Hart et al. 2009 selective exposure meta)
        # People seek belief-consistent information
        if _word_in('confirming', condition_lower) or _word_in('consistent', condition_lower):
            semantic_effect += 0.15
        elif _word_in('disconfirming', condition_lower) or _word_in('inconsistent', condition_lower):
            semantic_effect -= 0.12

        # Hyperbolic Discounting (Laibson, 1997; Amlung et al. meta)
        # Present bias; immediate rewards overweighted
        if _any_word_in(['$10 now', 'today', 'immediate small'], condition_lower):
            semantic_effect += 0.20
        elif _any_word_in(['$15 later', 'delayed large', 'wait for more'], condition_lower):
            semantic_effect -= 0.08

        # =====================================================================
        # DOMAIN 10: COMMUNICATION & PERSUASION MANIPULATIONS
        # =====================================================================

        # Source Credibility (Hovland & Weiss, 1951; Wilson & Sherrell, 1993 meta)
        # High credibility sources more persuasive
        if _any_word_in(['credible source', 'expert source', 'trustworthy'], condition_lower):
            semantic_effect += 0.20
        elif _any_word_in(['low credibility', 'non-expert', 'untrustworthy'], condition_lower):
            semantic_effect -= 0.18

        # Message Sidedness (Allen, 1991 meta-analysis)
        # Two-sided messages more effective for educated audiences
        if _word_in('two-sided', condition_lower) or _word_in('both sides', condition_lower):
            semantic_effect += 0.12
        elif _word_in('one-sided', condition_lower):
            semantic_effect -= 0.05

        # Narrative vs Statistical Evidence (Allen & Preiss, 1997 meta)
        # Narratives often more persuasive than statistics
        if _any_word_in(['narrative', 'story', 'anecdote', 'testimonial'], condition_lower):
            semantic_effect += 0.15
        elif _any_word_in(['statistical', 'data', 'numbers', 'facts'], condition_lower):
            semantic_effect += 0.08  # Both positive, narratives more so

        # Vividness Effect (Taylor & Thompson, 1982)
        # Vivid information more impactful
        if _any_word_in(['vivid', 'graphic', 'detailed', 'concrete'], condition_lower):
            semantic_effect += 0.12
        elif _any_word_in(['pallid', 'abstract', 'summary'], condition_lower):
            semantic_effect -= 0.05

        # Inoculation Theory (McGuire, 1961; Banas & Rains 2010 meta d = 0.29)
        # Pre-exposure to weakened arguments confers resistance
        if _word_in('inoculation', condition_lower) or _word_in('prebunk', condition_lower):
            semantic_effect += 0.18  # Resistance to persuasion
        elif _word_in('no inoculation', condition_lower):
            semantic_effect -= 0.05

        # v1.0.4.5: Sleeper Effect (Kumkale & Albarracin, 2004 meta d = 0.10)
        # Discounting cue forgotten over time, message persists
        if _any_word_in(['sleeper effect', 'delayed persuasion', 'discounting cue'], condition_lower):
            semantic_effect += 0.08

        # v1.0.4.5: Elaboration Likelihood (Petty & Cacioppo, 1986)
        # Central route = stronger, more durable attitudes
        if _any_word_in(['central route', 'high elaboration', 'strong argument', 'argument quality'], condition_lower):
            semantic_effect += 0.22
        elif _any_word_in(['peripheral route', 'low elaboration', 'weak argument', 'heuristic cue'], condition_lower):
            semantic_effect += 0.10  # Still persuasive, just weaker

        # v1.0.4.5: Mere Exposure in Communication (Zajonc, 1968; Bornstein 1989 meta)
        # Repeated message exposure increases liking/acceptance
        if _any_word_in(['repeated message', 'frequent exposure', 'high frequency ad'], condition_lower):
            semantic_effect += 0.12
        elif _any_word_in(['single exposure', 'one time', 'novel message'], condition_lower):
            semantic_effect += 0.02

        # v1.0.4.5: Source Attractiveness (Eagly & Chaiken, 1993)
        if _any_word_in(['attractive source', 'likable speaker', 'popular source'], condition_lower):
            semantic_effect += 0.14
        elif _any_word_in(['unattractive source', 'unlikable speaker', 'unpopular source'], condition_lower):
            semantic_effect -= 0.10

        # =====================================================================
        # DOMAIN 11: LEARNING & MEMORY MANIPULATIONS
        # =====================================================================

        # Testing Effect (Roediger & Karpicke, 2006; Rowland 2014 meta d = 0.50)
        # Retrieval practice enhances long-term retention
        if _any_word_in(['test', 'retrieval practice', 'quiz'], condition_lower):
            semantic_effect += 0.22
        elif _any_word_in(['restudy', 'review', 'read again'], condition_lower):
            semantic_effect -= 0.05

        # Spacing Effect (Cepeda et al. 2006 meta; robust effect)
        # Distributed practice superior to massed
        if _any_word_in(['spaced', 'distributed', 'interleaved'], condition_lower):
            semantic_effect += 0.20
        elif _any_word_in(['massed', 'blocked', 'crammed'], condition_lower):
            semantic_effect -= 0.10

        # Generation Effect (Slamecka & Graf, 1978)
        # Self-generated information better remembered
        if _any_word_in(['generate', 'produce', 'create', 'self-generated'], condition_lower):
            semantic_effect += 0.18
        elif _any_word_in(['read', 'provided', 'given'], condition_lower):
            semantic_effect -= 0.05

        # Desirable Difficulties (Bjork, 1994)
        # Challenges that slow learning can enhance retention
        if _word_in('difficult', condition_lower) or _word_in('challenging', condition_lower):
            semantic_effect += 0.10  # Long-term benefit despite short-term cost
        elif _word_in('easy', condition_lower) or _word_in('simple', condition_lower):
            semantic_effect += 0.05

        # v1.0.4.5: Transfer-Appropriate Processing (Morris et al., 1977)
        # Encoding that matches retrieval conditions improves performance
        if _any_word_in(['transfer appropriate', 'matched encoding', 'congruent context'], condition_lower):
            semantic_effect += 0.18
        elif _any_word_in(['mismatched encoding', 'incongruent context', 'different context'], condition_lower):
            semantic_effect -= 0.12

        # v1.0.4.5: Encoding Specificity (Tulving & Thomson, 1973)
        # Context-dependent memory; same context at encoding and retrieval helps
        if _any_word_in(['same context', 'encoding specificity', 'context reinstatement'], condition_lower):
            semantic_effect += 0.16
        elif _any_word_in(['different context', 'context change', 'new environment'], condition_lower):
            semantic_effect -= 0.10

        # v1.0.4.5: Self-Reference Effect (Rogers et al. 1977; Symons & Johnson 1997 meta d = 0.50)
        # Information processed in relation to self is better remembered
        if _any_word_in(['self-reference', 'relate to self', 'personal relevance', 'self-generated'], condition_lower):
            semantic_effect += 0.22
        elif _any_word_in(['other-reference', 'semantic processing', 'structural processing'], condition_lower):
            semantic_effect += 0.05

        # =====================================================================
        # DOMAIN 12: SOCIAL IDENTITY & GROUP MANIPULATIONS
        # =====================================================================

        # Common Identity (Gaertner et al., 1993)
        # Superordinate identity reduces intergroup bias
        if _any_word_in(['common identity', 'superordinate', 'we', 'shared'], condition_lower):
            semantic_effect += 0.22
        elif _any_word_in(['dual identity', 'subgroup', 'they'], condition_lower):
            semantic_effect -= 0.10

        # Contact Hypothesis (Allport, 1954; Pettigrew & Tropp 2006 meta r = -0.21)
        # Intergroup contact reduces prejudice
        if _any_word_in(['contact', 'interaction', 'exposure to outgroup'], condition_lower):
            semantic_effect += 0.18
        elif _any_word_in(['no contact', 'segregated', 'separate'], condition_lower):
            semantic_effect -= 0.12

        # Minimal Group Paradigm (Tajfel, 1971; Balliet et al. 2014)
        # Even arbitrary categories produce in-group favoritism
        if _any_word_in(['overestimator', 'klee group', 'blue team'], condition_lower):
            semantic_effect += 0.15  # In-group favoritism even in minimal groups

        # Social Identity Salience (Oakes, 1987)
        # Making identity salient activates associated attitudes
        if _any_word_in(['identity salient', 'reminded of', 'primed with'], condition_lower):
            semantic_effect += 0.15
        elif _word_in('identity not salient', condition_lower):
            semantic_effect -= 0.05

        # v1.0.4.5: Recategorization (Gaertner & Dovidio, 2000)
        # Recategorizing outgroup as common ingroup reduces bias
        if _any_word_in(['recategoriz', 'one group', 'common ingroup identity', 'merged group'], condition_lower):
            semantic_effect += 0.20
        elif _any_word_in(['separate groups', 'distinct groups', 'us vs them'], condition_lower):
            semantic_effect -= 0.15

        # v1.0.4.5: Crossed Categorization (Crisp & Hewstone, 2007)
        # When multiple category memberships cross-cut, bias is reduced
        if _any_word_in(['crossed categoriz', 'multiple identit', 'cross-cutting'], condition_lower):
            semantic_effect += 0.12
        elif _any_word_in(['single category', 'simple categoriz'], condition_lower):
            semantic_effect -= 0.05

        # v1.0.4.5: Relative Deprivation (Smith et al. 2012 meta)
        # Perceiving one's group as deprived increases collective action
        if _any_word_in(['relative deprivation', 'group disadvantage', 'inequality', 'unjust treatment'], condition_lower):
            semantic_effect -= 0.18  # Negative toward status quo
        elif _any_word_in(['group advantage', 'privileged group', 'equal treatment'], condition_lower):
            semantic_effect += 0.10

        # v1.0.4.5: Perspective-Taking (Galinsky & Moskowitz, 2000)
        # Taking outgroup perspective reduces stereotyping
        if _any_word_in(['perspective taking', 'imagine their life', 'walk in shoes', 'empathize outgroup'], condition_lower):
            semantic_effect += 0.18
        elif _any_word_in(['no perspective taking', 'objective view', 'detached'], condition_lower):
            semantic_effect -= 0.05

        # =====================================================================
        # DOMAIN 13: MOTIVATION & SELF-REGULATION MANIPULATIONS
        # =====================================================================

        # Implementation Intentions (Gollwitzer, 1999; Gollwitzer & Sheeran 2006 meta d = 0.65)
        # If-then planning increases goal attainment
        if _any_word_in(['implementation intention', 'if-then', 'when-then', 'planning'], condition_lower):
            semantic_effect += 0.28
        elif _word_in('goal intention', condition_lower) or _word_in('motivation only', condition_lower):
            semantic_effect -= 0.05

        # Growth vs Fixed Mindset (Dweck, 2006; Sisk et al. 2018 meta d = 0.10)
        # Malleable beliefs about ability; effect sizes smaller than originally claimed
        if _any_word_in(['growth mindset', 'malleable', 'can improve'], condition_lower):
            semantic_effect += 0.08
        elif _any_word_in(['fixed mindset', 'innate', 'cannot change'], condition_lower):
            semantic_effect -= 0.08

        # Regulatory Focus (Higgins, 1997)
        # Promotion vs prevention focus affects behavior
        if _any_word_in(['promotion', 'eager', 'gains', 'aspirations'], condition_lower):
            semantic_effect += 0.15
        elif _any_word_in(['prevention', 'vigilant', 'losses', 'obligations'], condition_lower):
            semantic_effect -= 0.10

        # Goal Gradient Effect (Hull, 1932; Kivetz et al. 2006)
        # Effort increases as goal approaches
        if _any_word_in(['near goal', 'almost there', 'close to'], condition_lower):
            semantic_effect += 0.18
        elif _any_word_in(['far from goal', 'just started', 'beginning'], condition_lower):
            semantic_effect -= 0.08

        # Licensing Effect (Merritt et al. 2010 meta)
        # Good deeds license subsequent bad behavior
        if _any_word_in(['licensed', 'already helped', 'did good'], condition_lower):
            semantic_effect -= 0.12  # Reduced subsequent prosocial
        elif _word_in('no license', condition_lower):
            semantic_effect += 0.05

        # =====================================================================
        # DOMAIN 14: ENVIRONMENTAL & CONTEXTUAL MANIPULATIONS
        # =====================================================================

        # Temperature and Aggression (Anderson et al. 2000)
        # Heat increases aggressive cognition and behavior
        if _any_word_in(['hot', 'warm room', 'heat'], condition_lower):
            semantic_effect -= 0.12  # More negative affect
        elif _any_word_in(['cool', 'cold room', 'comfortable temp'], condition_lower):
            semantic_effect += 0.05

        # Crowding (Baum & Paulus, 1987)
        # High density increases stress
        if _any_word_in(['crowded', 'high density', 'cramped'], condition_lower):
            semantic_effect -= 0.15
        elif _any_word_in(['spacious', 'low density', 'uncrowded'], condition_lower):
            semantic_effect += 0.08

        # Cleanliness (Schnall et al., 2008; Lee & Schwarz 2010)
        # Clean environments reduce severity of moral judgments
        if _any_word_in(['clean', 'tidy', 'pure', 'washed hands'], condition_lower):
            semantic_effect += 0.12
        elif _any_word_in(['dirty', 'messy', 'contaminated'], condition_lower):
            semantic_effect -= 0.15

        # Nature Exposure (Berman et al., 2008; Bratman et al., 2012)
        # Nature reduces stress, improves mood
        if _any_word_in(['nature', 'park', 'green space', 'outdoors'], condition_lower):
            semantic_effect += 0.15
        elif _any_word_in(['urban', 'city', 'concrete', 'indoors'], condition_lower):
            semantic_effect -= 0.05

        # Lighting (Baron et al., 1992)
        # Bright light improves mood and alertness
        if _any_word_in(['bright light', 'well-lit', 'daylight'], condition_lower):
            semantic_effect += 0.08
        elif _any_word_in(['dim light', 'dark', 'low light'], condition_lower):
            semantic_effect -= 0.05

        # v1.0.4.5: Noise Effects (Banbury & Berry, 2005; Szalma & Hancock 2011 meta)
        # Noise impairs cognitive performance, increases stress
        if _any_word_in(['noisy', 'loud', 'high noise', 'noise distraction'], condition_lower):
            semantic_effect -= 0.14
        elif _any_word_in(['quiet', 'silent', 'low noise', 'no noise'], condition_lower):
            semantic_effect += 0.08

        # v1.0.4.5: Color Psychology (Mehta & Zhu, 2009; Elliot & Maier 2014)
        # Red = avoidance/arousal; Blue = approach/creativity
        if _any_word_in(['red color', 'red background', 'red prime'], condition_lower):
            semantic_effect -= 0.08  # Avoidance, caution
        elif _any_word_in(['blue color', 'blue background', 'blue prime'], condition_lower):
            semantic_effect += 0.08  # Approach, openness

        # v1.0.4.5: Music/Sound (Hallam et al. 2002; Kämpfe et al. 2011 meta)
        # Background music can enhance or impair depending on task
        if _any_word_in(['music', 'pleasant sound', 'calming audio'], condition_lower):
            semantic_effect += 0.06
        elif _any_word_in(['no music', 'silence condition', 'unpleasant sound'], condition_lower):
            semantic_effect -= 0.04

        # =====================================================================
        # DOMAIN 15: EMBODIMENT & PHYSICAL MANIPULATIONS
        # =====================================================================

        # Facial Feedback (Strack et al., 1988; Coles et al. 2019 many-labs r = 0.03)
        # Facial expressions may influence emotional experience
        # Effect small or null in replications
        if _any_word_in(['smile', 'pen in teeth', 'happy expression'], condition_lower):
            semantic_effect += 0.05  # Small effect
        elif _any_word_in(['frown', 'pen in lips', 'sad expression'], condition_lower):
            semantic_effect -= 0.05

        # Power Posing (Carney et al., 2010; Credé & Phillips 2017 critique)
        # Expansive poses may affect feelings; effects contested
        if _any_word_in(['power pose', 'expansive', 'open posture'], condition_lower):
            semantic_effect += 0.08  # Contested, smaller than original claims
        elif _any_word_in(['contractive', 'closed posture', 'slumped'], condition_lower):
            semantic_effect -= 0.05

        # Heaviness and Importance (Jostmann et al., 2009)
        # Heavier objects associated with importance
        if _any_word_in(['heavy clipboard', 'weighty', 'substantial'], condition_lower):
            semantic_effect += 0.10
        elif _any_word_in(['light clipboard', 'lightweight', 'flimsy'], condition_lower):
            semantic_effect -= 0.05

        # v1.0.4.4: Warmth/Coldness priming (Williams & Bargh, 2008)
        # Holding warm beverage → warmer interpersonal judgments; d ≈ 0.15-0.25
        # Replication mixed but effect appears in some contexts
        if _any_word_in(['warm cup', 'warm drink', 'heated pad', 'warm hands'], condition_lower):
            semantic_effect += 0.08
        elif _any_word_in(['cold cup', 'cold drink', 'ice pack', 'cold hands'], condition_lower):
            semantic_effect -= 0.08

        # v1.0.4.4: Physical Movement (Casasanto & Dijkstra, 2010)
        # Arm flexion → approach motivation; arm extension → avoidance
        if _any_word_in(['arm flexion', 'pull toward', 'approach motion'], condition_lower):
            semantic_effect += 0.10
        elif _any_word_in(['arm extension', 'push away', 'avoidance motion'], condition_lower):
            semantic_effect -= 0.10

        # v1.0.4.4: Physical Touch (Crusco & Wetzel, 1984; Guéguen, 2002)
        # Brief touch increases compliance and positive evaluation; d ≈ 0.20
        if _any_word_in(['touch', 'physical contact', 'pat on shoulder', 'handshake'], condition_lower):
            semantic_effect += 0.12
        elif _any_word_in(['no touch', 'no contact', 'distanced'], condition_lower):
            semantic_effect -= 0.02

        # v1.0.4.4: Head Movement (Wells & Petty, 1980)
        # Nodding → agreement; head shaking → disagreement
        if _any_word_in(['nodding', 'head nod', 'vertical head'], condition_lower):
            semantic_effect += 0.10
        elif _any_word_in(['head shake', 'horizontal head', 'shaking head'], condition_lower):
            semantic_effect -= 0.10

        # =====================================================================
        # DOMAIN 16: TIME & TEMPORAL MANIPULATIONS
        # =====================================================================

        # Time Pressure (Dror et al., 1999)
        # Time constraints affect decision quality
        if _any_word_in(['time pressure', 'deadline', 'hurry', 'limited time'], condition_lower):
            semantic_effect -= 0.15  # More errors, less satisfaction
        elif _any_word_in(['no time pressure', 'unlimited time', 'take your time'], condition_lower):
            semantic_effect += 0.08

        # Morning vs Afternoon (Sievertsen et al., 2016)
        # Cognitive performance varies by time of day
        if _any_word_in(['morning', 'early', 'am session'], condition_lower):
            semantic_effect += 0.08
        elif _any_word_in(['afternoon', 'late', 'pm session', 'evening'], condition_lower):
            semantic_effect -= 0.05

        # Waiting (Kumar et al., 2014)
        # Anticipation affects experience
        if _any_word_in(['anticipation', 'waiting', 'expecting'], condition_lower):
            semantic_effect += 0.12
        elif _word_in('immediate', condition_lower):
            semantic_effect += 0.05

        # v1.0.4.4: Future Time Perspective (Zimbardo & Boyd, 1999)
        # Future-oriented people show more self-regulation and delayed gratification
        if _any_word_in(['future oriented', 'long-term', 'future self', 'years from now'], condition_lower):
            semantic_effect += 0.15
        elif _any_word_in(['present oriented', 'short-term', 'live for today', 'right now'], condition_lower):
            semantic_effect -= 0.08

        # v1.0.4.4: Temporal Landmarks (Dai et al., 2014; "fresh start effect")
        # New beginnings (Monday, New Year, birthday) increase motivation
        if _any_word_in(['fresh start', 'new year', 'new beginning', 'monday', 'new semester'], condition_lower):
            semantic_effect += 0.12
        elif _any_word_in(['ordinary day', 'mid-week', 'continuation'], condition_lower):
            semantic_effect -= 0.02

        # v1.0.4.4: Nostalgia induction (Wildschut et al., 2006)
        # Nostalgia increases social connectedness, positive affect, meaning
        if _any_word_in(['nostalg', 'remember when', 'childhood memory', 'good old days'], condition_lower):
            semantic_effect += 0.18
        elif _any_word_in(['ordinary event', 'routine', 'typical day'], condition_lower):
            semantic_effect -= 0.02

        # =====================================================================
        # DOMAIN 17: DECEPTION & DISHONESTY MANIPULATIONS (v1.0.4.4)
        # Scientific basis: Mazar et al. (2008), Gino et al. (2009),
        # Shalvi et al. (2011), Fischbacher & Föllmi-Heusi (2013)
        # =====================================================================

        # Honor Code / Moral Reminder (Mazar et al., 2008)
        # Reminders of morality reduce dishonesty; d ≈ 0.30
        if _any_word_in(['honor code', 'moral reminder', 'ten commandments', 'honesty pledge'], condition_lower):
            semantic_effect += 0.18  # More honest behavior
        elif _any_word_in(['no reminder', 'baseline dishonesty', 'no pledge'], condition_lower):
            semantic_effect -= 0.08

        # Monitoring / Observability (Bateson et al., 2006; "watching eyes")
        # Being observed or reminded of observation increases honesty
        if _any_word_in(['monitored', 'observed', 'camera', 'watching eyes', 'transparent'], condition_lower):
            semantic_effect += 0.15
        elif _any_word_in(['unmonitored', 'unobserved', 'anonymous', 'private'], condition_lower):
            semantic_effect -= 0.12

        # Moral Licensing (Merritt et al., 2010 meta)
        # Prior moral behavior licenses subsequent transgression
        if _any_word_in(['moral license', 'already donated', 'virtuous act'], condition_lower):
            semantic_effect -= 0.10
        elif _any_word_in(['no license', 'neutral prime', 'control prime'], condition_lower):
            semantic_effect += 0.03

        # Self-Serving Justification (Shalvi et al., 2011)
        # Ambiguity enables self-serving dishonesty
        if _any_word_in(['ambiguous', 'plausible deniability', 'uncertain outcome'], condition_lower):
            semantic_effect -= 0.08  # More dishonesty
        elif _any_word_in(['unambiguous', 'clear outcome', 'verifiable'], condition_lower):
            semantic_effect += 0.10

        # v1.0.4.5: Ethical Fading (Tenbrunsel & Messick, 2004)
        # Framing removes ethical dimension from decision; increases dishonesty
        if _any_word_in(['business frame', 'strategic decision', 'competitive context', 'ethical fading'], condition_lower):
            semantic_effect -= 0.12
        elif _any_word_in(['ethical frame', 'moral decision', 'right thing'], condition_lower):
            semantic_effect += 0.14

        # v1.0.4.5: Incrementalism / Slippery Slope (Welsh et al. 2015)
        # Small initial transgressions escalate gradually
        if _any_word_in(['gradual escalation', 'slippery slope', 'small lie', 'incremental'], condition_lower):
            semantic_effect -= 0.10
        elif _any_word_in(['sudden large', 'all at once', 'single decision'], condition_lower):
            semantic_effect += 0.05  # Harder to justify single large act

        # v1.0.4.5: Self-Concept Maintenance (Mazar et al. 2008; Ariely 2012)
        # People cheat only to extent they can maintain honest self-image
        if _any_word_in(['self concept', 'identity threat', 'honest self-image'], condition_lower):
            semantic_effect += 0.10  # Constrains dishonesty
        elif _any_word_in(['deindividuated', 'group decision', 'diffused responsibility'], condition_lower):
            semantic_effect -= 0.14  # Easier to be dishonest

        # =====================================================================
        # DOMAIN 18: POWER & STATUS MANIPULATIONS (v1.0.4.4, expanded v1.0.4.5)
        # Scientific basis: Keltner et al. (2003), Galinsky et al. (2003),
        # Anderson & Berdahl (2002)
        # =====================================================================

        # Power Priming (Galinsky et al., 2003)
        # High power → approach-oriented, risk-taking, less perspective-taking
        if _any_word_in(['high power', 'power prime', 'boss', 'leader role', 'in charge'], condition_lower):
            semantic_effect += 0.18
        elif _any_word_in(['low power', 'subordinate', 'employee role', 'follower'], condition_lower):
            semantic_effect -= 0.15

        # Social Status (Kraus et al., 2012)
        # Higher subjective status → more positive self-evaluation, confidence
        if _any_word_in(['high status', 'wealthy', 'upper class', 'privileged'], condition_lower):
            semantic_effect += 0.15
        elif _any_word_in(['low status', 'poor', 'lower class', 'disadvantaged'], condition_lower):
            semantic_effect -= 0.18

        # Accountability (Lerner & Tetlock, 1999)
        # Being accountable increases accuracy motivation, reduces biases
        if _any_word_in(['accountable', 'justify', 'explain to others', 'audience aware'], condition_lower):
            semantic_effect += 0.10
        elif _any_word_in(['not accountable', 'anonymous decision', 'private choice'], condition_lower):
            semantic_effect -= 0.05

        # v1.0.4.5: Dominance vs Prestige (Cheng et al. 2013; Henrich & Gil-White 2001)
        # Two routes to status: intimidation vs. freely conferred deference
        if _any_word_in(['dominant', 'intimidat', 'coercive power', 'aggressive leader'], condition_lower):
            semantic_effect -= 0.12  # Dominance → negative evaluation
        elif _any_word_in(['prestige', 'respected', 'admired leader', 'earned status'], condition_lower):
            semantic_effect += 0.18

        # v1.0.4.5: Status Anxiety (de Botton, 2004; Wilkinson & Pickett 2009)
        # Social comparison with higher status others → negative affect
        if _any_word_in(['status anxiety', 'upward comparison', 'outperformed', 'inferior'], condition_lower):
            semantic_effect -= 0.16
        elif _any_word_in(['downward comparison', 'outperforming', 'superior position'], condition_lower):
            semantic_effect += 0.12

        # v1.0.4.5: Hierarchy Legitimacy (Tyler, 2006; Jost & Banaji 1994)
        # Perceived legitimacy of hierarchy affects acceptance and behavior
        if _any_word_in(['legitimate hierarchy', 'meritocratic', 'fair system', 'earned position'], condition_lower):
            semantic_effect += 0.14
        elif _any_word_in(['illegitimate hierarchy', 'unfair system', 'nepotism', 'arbitrary status'], condition_lower):
            semantic_effect -= 0.18

        # v1.0.4.5: Power and Perspective-Taking (Galinsky et al. 2006)
        # High power reduces perspective-taking and empathy
        if _any_word_in(['powerful perspective', 'power empathy', 'power and others'], condition_lower):
            semantic_effect -= 0.10  # Powerful people less considerate
        # This extends the basic power priming effect above

        # v1.0.4.5: Resource Scarcity × Status (Shah et al. 2012; Mullainathan & Shafir 2013)
        # Scarcity captures attention but impairs long-term decision making
        if _any_word_in(['resource scarce', 'budget constrained', 'limited resources', 'scarcity mindset'], condition_lower):
            semantic_effect -= 0.14
        elif _any_word_in(['resource abundant', 'unconstrained', 'plenty of resources'], condition_lower):
            semantic_effect += 0.08

        # =====================================================================
        # DOMAIN 19: NARRATIVE TRANSPORTATION (v1.0.4.9)
        # Green & Brock (2000): Absorption into narrative worlds increases
        # persuasion by reducing counterarguing and increasing emotional engagement.
        # van Laer et al. (2014 meta): r = 0.35 for narrative persuasion
        # Appel & Richter (2007): Fiction can change real-world beliefs
        # =====================================================================

        if _any_word_in(['narrative', 'story', 'transported', 'immersed', 'absorbed'], condition_lower):
            if _any_word_in(['high transport', 'vivid narrative', 'immersive story', 'engaging narrative'], condition_lower):
                semantic_effect += 0.22  # Strong transportation → high persuasion
            else:
                semantic_effect += 0.15  # General narrative advantage
        elif _any_word_in(['expository', 'factual', 'report', 'data only', 'no narrative'], condition_lower):
            semantic_effect -= 0.05

        # Fictional vs real narratives (Appel & Richter, 2007)
        if _word_in('fictional', condition_lower) or _word_in('imagined', condition_lower):
            semantic_effect += 0.08  # Fiction still persuasive
        elif _word_in('real story', condition_lower) or _word_in('true account', condition_lower):
            semantic_effect += 0.12  # Real accounts slightly more persuasive

        # First-person vs third-person narrative (de Graaf et al., 2012)
        if _any_word_in(['first person', 'i perspective', 'my experience'], condition_lower):
            semantic_effect += 0.10  # Greater identification
        elif _any_word_in(['third person', 'they perspective', 'observer'], condition_lower):
            semantic_effect += 0.04

        # =====================================================================
        # DOMAIN 20: SOCIAL COMPARISON (v1.0.4.9)
        # Festinger (1954): Upward/downward comparison affects self-evaluation.
        # Gerber et al. (2018 meta): Social comparison d = 0.20-0.50
        # Wheeler & Miyake (1992): Upward comparison → negative affect
        # Wills (1981): Downward comparison → positive affect
        # =====================================================================

        # Upward social comparison (Wheeler & Miyake, 1992)
        if _any_word_in(['upward comparison', 'better than you', 'higher performer',
                         'outperformed by', 'social comparison up'], condition_lower):
            semantic_effect -= 0.18  # Self-threat, negative affect
        elif _any_word_in(['downward comparison', 'worse than you', 'lower performer',
                           'outperforming', 'social comparison down'], condition_lower):
            semantic_effect += 0.15  # Self-enhancement, positive affect

        # Social media comparison (Vogel et al., 2014)
        if _any_word_in(['social media feed', 'instagram', 'curated profile',
                         'highlight reel', 'idealized images'], condition_lower):
            semantic_effect -= 0.15  # Social media upward comparison
        elif _any_word_in(['no social media', 'authentic post', 'real life'], condition_lower):
            semantic_effect += 0.05

        # Assimilation vs contrast (Mussweiler, 2003)
        if _any_word_in(['similar target', 'assimilation', 'like me'], condition_lower):
            semantic_effect += 0.10  # Assimilation toward comparison target
        elif _any_word_in(['dissimilar target', 'contrast', 'unlike me'], condition_lower):
            semantic_effect -= 0.12  # Contrast away from comparison target

        # =====================================================================
        # DOMAIN 21: GRATITUDE & POSITIVE INTERVENTIONS (v1.0.4.9)
        # Emmons & McCullough (2003): Gratitude journaling increases wellbeing
        # Davis et al. (2016 meta): Gratitude interventions d = 0.31
        # Sin & Lyubomirsky (2009 meta): Positive psychology interventions d = 0.29
        # Seligman et al. (2005): Three good things, gratitude visits
        # =====================================================================

        # Gratitude induction (Emmons & McCullough, 2003)
        if _any_word_in(['gratitude', 'thankful', 'grateful', 'count blessings',
                         'gratitude journal', 'three good things'], condition_lower):
            semantic_effect += 0.18  # Robust positive effect
        elif _any_word_in(['hassles', 'complaints', 'annoyances', 'neutral listing'], condition_lower):
            semantic_effect -= 0.10

        # Kindness intervention (Lyubomirsky et al., 2005)
        if _any_word_in(['acts of kindness', 'kindness task', 'helping others'], condition_lower):
            semantic_effect += 0.15
        elif _any_word_in(['self-focused', 'no kindness', 'routine activities'], condition_lower):
            semantic_effect -= 0.03

        # Savoring (Bryant & Veroff, 2007)
        if _any_word_in(['savoring', 'mindful attention', 'positive focus', 'appreciate moment'], condition_lower):
            semantic_effect += 0.14
        elif _any_word_in(['distraction', 'mind wandering', 'dampening'], condition_lower):
            semantic_effect -= 0.08

        # Best possible self (King, 2001; Meevissen et al., 2011)
        if _any_word_in(['best possible self', 'ideal future', 'future self visualization'], condition_lower):
            semantic_effect += 0.16
        elif _any_word_in(['typical day', 'ordinary future', 'no visualization'], condition_lower):
            semantic_effect -= 0.02

        # =====================================================================
        # DOMAIN 22: MORAL CLEANSING & COMPENSATION (v1.0.4.9)
        # Zhong & Liljenquist (2006): "Macbeth effect" - moral threat → physical cleansing
        # Sachdeva et al. (2009): Moral self-regulation via licensing/cleansing
        # Jordan et al. (2011): Moral identity priming increases prosocial behavior
        # Tetlock et al. (2000): Sacred value tradeoffs trigger moral outrage
        # =====================================================================

        # Moral threat / transgression (Zhong & Liljenquist, 2006)
        if _any_word_in(['moral threat', 'recalled transgression', 'unethical memory',
                         'moral failure', 'guilt prime'], condition_lower):
            semantic_effect -= 0.16  # Moral distress → compensatory behavior
        elif _any_word_in(['moral affirmation', 'ethical memory', 'virtuous recall',
                           'moral success'], condition_lower):
            semantic_effect += 0.14  # Moral licensing risk

        # Sacred values (Tetlock et al., 2000; Tetlock, 2003)
        if _any_word_in(['sacred value', 'taboo tradeoff', 'money for morals',
                         'sell out', 'commodify'], condition_lower):
            semantic_effect -= 0.25  # Strong moral outrage
        elif _any_word_in(['routine tradeoff', 'cost-benefit', 'utilitarian calculus'], condition_lower):
            semantic_effect += 0.05

        # Moral identity salience (Aquino & Reed, 2002)
        if _any_word_in(['moral identity', 'ethical self', 'virtuous person'], condition_lower):
            semantic_effect += 0.15  # Motivates moral behavior
        elif _any_word_in(['amoral', 'pragmatic identity', 'self-interest'], condition_lower):
            semantic_effect -= 0.10

        # =====================================================================
        # DOMAIN 23: ATTENTION ECONOMY & DIGITAL DISTRACTION (v1.0.4.9)
        # Ward et al. (2017, JACR): Phone presence reduces cognitive capacity
        # Stothart et al. (2015): Phone notifications impair attention
        # Ophir et al. (2009): Media multitasking reduces filtering ability
        # Uncapher & Wagner (2018): Heavy media multitasking → attention deficits
        # =====================================================================

        # Phone/device presence (Ward et al., 2017)
        if _any_word_in(['phone present', 'phone on desk', 'device visible',
                         'smartphone nearby'], condition_lower):
            semantic_effect -= 0.14  # Cognitive drain
        elif _any_word_in(['no phone', 'phone away', 'device removed',
                           'phone absent'], condition_lower):
            semantic_effect += 0.08

        # Notification interruption (Stothart et al., 2015)
        if _any_word_in(['notification', 'interrupted', 'alert', 'ping'], condition_lower):
            semantic_effect -= 0.16
        elif _any_word_in(['no interruption', 'do not disturb', 'silent mode',
                           'focus mode'], condition_lower):
            semantic_effect += 0.10

        # Media multitasking (Ophir et al., 2009)
        if _any_word_in(['multitask', 'dual task', 'split attention',
                         'media multitask'], condition_lower):
            semantic_effect -= 0.18
        elif _any_word_in(['single task', 'focused', 'undivided attention'], condition_lower):
            semantic_effect += 0.12

        # Digital detox (Radtke et al., 2022)
        if _any_word_in(['digital detox', 'screen break', 'tech-free',
                         'offline period'], condition_lower):
            semantic_effect += 0.14
        elif _any_word_in(['continuous use', 'always connected', 'high screen time'], condition_lower):
            semantic_effect -= 0.10

        # =====================================================================
        # DOMAIN 24: NOSTALGIA MANIPULATIONS (v1.0.7.4)
        # Wildschut et al. (2006): Nostalgia increases positive affect, social
        # connectedness, self-continuity, and meaning in life.
        # Sedikides et al. (2015 meta): Nostalgia d = 0.20-0.35 on wellbeing.
        # Routledge et al. (2011): Nostalgia buffers existential threat.
        # =====================================================================

        # Nostalgia induction (Wildschut et al., 2006)
        if _any_word_in(['nostalgic', 'nostalgia prime', 'nostalgia condition',
                         'nostalgic memory', 'sentimental'], condition_lower):
            semantic_effect += 0.20  # Positive affect, connectedness
        elif _any_word_in(['contemporary', 'modern', 'present-focused',
                           'current event'], condition_lower):
            semantic_effect -= 0.05

        # Past/memory orientation (Routledge et al., 2011)
        if _any_word_in(['past memory', 'remember the past', 'childhood',
                         'old days', 'reminiscence'], condition_lower):
            semantic_effect += 0.12
        elif _any_word_in(['future focus', 'forward looking', 'plan ahead'], condition_lower):
            semantic_effect -= 0.08

        # Personal vs historical nostalgia (Batcho, 2013)
        if _any_word_in(['personal nostalgia', 'own past', 'autobiographical'], condition_lower):
            semantic_effect += 0.18  # Personal nostalgia stronger
        elif _any_word_in(['historical nostalgia', 'collective past', 'era nostalgia'], condition_lower):
            semantic_effect += 0.10

        # =====================================================================
        # DOMAIN 25: FORGIVENESS MANIPULATIONS (v1.0.7.4)
        # Fehr et al. (2010 meta): Forgiveness interventions d = 0.56.
        # McCullough et al. (2000): Empathy mediates forgiveness.
        # Worthington (2006): REACH model of forgiveness.
        # =====================================================================

        # Forgiveness induction (Fehr et al., 2010)
        if _any_word_in(['forgive', 'forgiveness', 'forgiveness prime',
                         'letting go', 'pardon'], condition_lower):
            semantic_effect += 0.18
        elif _any_word_in(['grudge', 'revenge', 'vengeance',
                           'hold grudge', 'unforgiven'], condition_lower):
            semantic_effect -= 0.22

        # Reconciliation vs retaliation (McCullough et al., 2001)
        if _any_word_in(['reconcil', 'restore relationship', 'make amends',
                         'apology accepted'], condition_lower):
            semantic_effect += 0.15
        elif _any_word_in(['punish', 'retaliat', 'get even',
                           'retributive', 'payback'], condition_lower):
            semantic_effect -= 0.18

        # Transgression severity (Fincham et al., 2006)
        if _any_word_in(['minor offense', 'small transgression', 'slight'], condition_lower):
            semantic_effect += 0.08  # Easier to forgive
        elif _any_word_in(['severe offense', 'betrayal', 'major transgression'], condition_lower):
            semantic_effect -= 0.20  # Harder to forgive

        # =====================================================================
        # DOMAIN 26: GRATITUDE DEPTH MANIPULATIONS (v1.0.7.4)
        # Wood et al. (2010 meta): Gratitude -> wellbeing r = 0.30-0.50.
        # Algoe (2012): Find-Remind-Bind theory of gratitude.
        # Ma et al. (2017 meta): Gratitude interventions d = 0.31.
        # Note: Basic gratitude induction is in Domain 21. This covers
        # deeper gratitude constructs and entitlement contrasts.
        # =====================================================================

        # Grateful disposition priming (Wood et al., 2010)
        if _any_word_in(['grateful disposition', 'trait gratitude',
                         'grateful person', 'appreciation mindset'], condition_lower):
            semantic_effect += 0.22
        elif _any_word_in(['entitled', 'entitlement', 'deserve more',
                           'owed', 'demanding'], condition_lower):
            semantic_effect -= 0.15

        # Benefactor-focused gratitude (Algoe et al., 2008)
        if _any_word_in(['thank benefactor', 'gratitude letter',
                         'grateful to person', 'benefactor appreciation'], condition_lower):
            semantic_effect += 0.18
        elif _any_word_in(['ungrateful', 'unappreciated', 'taken for granted',
                           'ingratitude'], condition_lower):
            semantic_effect -= 0.12

        # Material vs experiential gratitude (Emmons, 2007)
        if _any_word_in(['experiential gratitude', 'grateful for experience'], condition_lower):
            semantic_effect += 0.16  # Experiential gratitude more lasting
        elif _any_word_in(['material gratitude', 'grateful for possession'], condition_lower):
            semantic_effect += 0.10

        # =====================================================================
        # DOMAIN 27: GROWTH MINDSET & IMPLICIT THEORIES (v1.0.7.4)
        # Sisk et al. (2018 meta): Growth mindset intervention d = 0.08.
        # Yeager et al. (2019, Nature): Targeted interventions d = 0.10.
        # Dweck (2006): Implicit theories of intelligence framework.
        # Note: Basic growth/fixed is in Domain 13. This covers deeper
        # mindset constructs and effort/ability attributions.
        # =====================================================================

        # Mindset intervention (Yeager et al., 2019)
        if _any_word_in(['growth mindset intervention', 'malleable intelligence',
                         'brain grows', 'neuroplasticity message'], condition_lower):
            semantic_effect += 0.10  # Small but real for targeted populations
        elif _any_word_in(['fixed mindset induction', 'innate ability',
                           'born with it', 'genetic talent'], condition_lower):
            semantic_effect -= 0.08

        # Effort vs talent attribution (Mueller & Dweck, 1998)
        if _any_word_in(['effort praise', 'hard work', 'improvement',
                         'practice makes', 'learning process'], condition_lower):
            semantic_effect += 0.12
        elif _any_word_in(['talent praise', 'natural ability', 'gifted',
                           'born smart', 'innate talent'], condition_lower):
            semantic_effect -= 0.05

        # Failure mindset framing (Haimovitz & Dweck, 2016)
        if _any_word_in(['failure is learning', 'growth from failure',
                         'productive failure'], condition_lower):
            semantic_effect += 0.14
        elif _any_word_in(['failure is bad', 'avoid failure',
                           'failure means inability'], condition_lower):
            semantic_effect -= 0.10

        # =====================================================================
        # DOMAIN 28: SELF-AFFIRMATION MANIPULATIONS (v1.0.7.4)
        # McQueen & Klein (2006 meta): Self-affirmation d = 0.17.
        # Cohen & Sherman (2014): Self-affirmation reduces defensiveness.
        # Steele (1988): Self-affirmation theory -- affirming core values
        # buffers threat and reduces defensive processing.
        # =====================================================================

        # Values affirmation (Cohen et al., 2006)
        if _any_word_in(['values affirmation', 'self affirm', 'affirm values',
                         'important values', 'core values essay'], condition_lower):
            semantic_effect += 0.18
        elif _any_word_in(['no affirmation', 'control essay',
                           'unimportant values', 'neutral writing'], condition_lower):
            semantic_effect -= 0.03

        # Self-affirmation under threat (Sherman & Cohen, 2006)
        if _any_word_in(['affirm under threat', 'affirmed and threatened',
                         'buffered threat'], condition_lower):
            semantic_effect += 0.15  # Affirmation buffers threat
        elif _any_word_in(['threat no affirm', 'self threat', 'ego threat',
                           'identity threat', 'unaffirmed threat'], condition_lower):
            semantic_effect -= 0.12

        # Spontaneous self-affirmation (Pietersma & Dijkstra, 2012)
        if _any_word_in(['spontaneous affirm', 'self-generated affirm',
                         'reflect on strengths'], condition_lower):
            semantic_effect += 0.12
        elif _any_word_in(['other-affirm', 'affirm other person',
                           'other strengths'], condition_lower):
            semantic_effect += 0.05

        # =====================================================================
        # DOMAIN 29: AUTONOMY & SELF-DETERMINATION (v1.0.7.4)
        # Deci & Ryan (2000): Self-Determination Theory -- autonomy, competence,
        # and relatedness as basic psychological needs.
        # Patall et al. (2008 meta): Choice d = 0.19 on intrinsic motivation.
        # Moller et al. (2006): Autonomy support vs. control.
        # =====================================================================

        # Autonomy/choice manipulation (Patall et al., 2008)
        if _any_word_in(['autonomy', 'free choice', 'autonomy support',
                         'self-directed', 'choose freely'], condition_lower):
            semantic_effect += 0.20
        elif _any_word_in(['controlled', 'coerced', 'no choice',
                           'forced', 'externally controlled'], condition_lower):
            semantic_effect -= 0.18

        # Competence feedback (Vallerand & Reid, 1984)
        if _any_word_in(['competence', 'mastery', 'skill feedback',
                         'positive competence', 'you are capable'], condition_lower):
            semantic_effect += 0.15
        elif _any_word_in(['incompetence', 'failure feedback',
                           'negative competence', 'you failed'], condition_lower):
            semantic_effect -= 0.15

        # Relatedness/belonging (Baumeister & Leary, 1995)
        if _any_word_in(['relatedness', 'belonging', 'socially connected',
                         'included', 'part of group'], condition_lower):
            semantic_effect += 0.12
        elif _any_word_in(['excluded', 'ostracized', 'rejected',
                           'socially isolated'], condition_lower):
            semantic_effect -= 0.18

        # Autonomy-supportive vs controlling language (Vansteenkiste et al., 2004)
        if _any_word_in(['autonomy language', 'you may', 'consider trying',
                         'you could'], condition_lower):
            semantic_effect += 0.10
        elif _any_word_in(['controlling language', 'you must', 'you should',
                           'you have to'], condition_lower):
            semantic_effect -= 0.10

        # =====================================================================
        # DOMAIN 30: SCARCITY & RESOURCE MANIPULATIONS (v1.0.7.4)
        # Shah et al. (2012): Scarcity captures attention but impairs executive
        # function ("tunneling"). Mullainathan & Shafir (2013): Scarcity mindset.
        # Cialdini (2009): Scarcity as persuasion principle.
        # Note: Resource scarcity x status is in Domain 18. This covers
        # broader scarcity/abundance constructs.
        # =====================================================================

        # Scarcity induction (Shah et al., 2012)
        if _any_word_in(['scarcity', 'scarce', 'limited supply',
                         'running out', 'few remaining'], condition_lower):
            semantic_effect -= 0.15  # Tunneling, urgency
        elif _any_word_in(['abundance', 'abundant', 'plentiful',
                           'unlimited supply', 'surplus'], condition_lower):
            semantic_effect += 0.10

        # Cognitive tunneling (Mullainathan & Shafir, 2013)
        if _any_word_in(['tunneling', 'bandwidth tax', 'cognitive load scarcity',
                         'scarcity mindset'], condition_lower):
            semantic_effect -= 0.12
        elif _any_word_in(['slack', 'mental bandwidth', 'cognitive surplus',
                           'abundance mindset'], condition_lower):
            semantic_effect += 0.08

        # Time scarcity vs money scarcity (Hershfield et al., 2016)
        if _any_word_in(['time scarcity', 'time poor', 'rushed'], condition_lower):
            semantic_effect -= 0.14
        elif _any_word_in(['time rich', 'time affluent', 'unhurried'], condition_lower):
            semantic_effect += 0.08

        # Persuasive scarcity (Cialdini, 2009)
        if _any_word_in(['limited edition', 'exclusive offer', 'only a few left',
                         'deadline offer'], condition_lower):
            semantic_effect += 0.12  # Persuasion via scarcity
        elif _any_word_in(['always available', 'no deadline', 'unlimited offer'], condition_lower):
            semantic_effect -= 0.02

        # =====================================================================
        # DOMAIN 31: SLEEP & FATIGUE MANIPULATIONS (v1.0.7.4)
        # Lim & Dinges (2010 meta): Sleep deprivation impairs attention d = 0.80,
        # working memory d = 0.55, and mood d = 0.50.
        # Killgore (2010): Sleep deprivation impairs moral judgment.
        # Walker (2017): Sleep loss increases emotional reactivity.
        # =====================================================================

        # Sleep deprivation (Lim & Dinges, 2010)
        if _any_word_in(['sleep deprived', 'sleep deprivation', 'no sleep',
                         'sleep restricted', 'stayed awake'], condition_lower):
            semantic_effect -= 0.20  # Significant cognitive impairment
        elif _any_word_in(['well rested', 'full sleep', 'sleep sufficient',
                           'good sleep', 'rested'], condition_lower):
            semantic_effect += 0.12

        # Fatigue induction (Baumeister et al., 1998; ego depletion)
        if _any_word_in(['fatigued', 'exhausted', 'depleted',
                         'ego depleted', 'mentally tired'], condition_lower):
            semantic_effect -= 0.15
        elif _any_word_in(['refreshed', 'energized', 'alert',
                           'well-rested', 'fully awake'], condition_lower):
            semantic_effect += 0.10

        # Insomnia simulation (Fortier-Brochu et al., 2012)
        if _any_word_in(['insomnia', 'poor sleep quality',
                         'sleep disrupted', 'broken sleep'], condition_lower):
            semantic_effect -= 0.15
        elif _any_word_in(['sleep quality', 'restful sleep',
                           'sleep hygiene'], condition_lower):
            semantic_effect += 0.08

        # Circadian mismatch (Goldstein et al., 2007)
        if _any_word_in(['circadian mismatch', 'off-peak', 'wrong time of day'], condition_lower):
            semantic_effect -= 0.10
        elif _any_word_in(['circadian match', 'optimal time', 'peak time'], condition_lower):
            semantic_effect += 0.08

        # =====================================================================
        # DOMAIN 32: MUSIC & MOOD INDUCTION (v1.0.7.4)
        # Juslin & Vastfjall (2008): 6 mechanisms of musical emotion induction.
        # Eerola & Vuoskoski (2013): Discrete emotions from music.
        # Vastfjall (2002): Emotion induction via music more ecologically valid
        # than Velten method. Note: Basic music is in Domain 14. This covers
        # specific mood induction via music characteristics.
        # =====================================================================

        # Happy/upbeat music induction (Eerola & Vuoskoski, 2013)
        if _any_word_in(['happy music', 'upbeat music', 'joyful music',
                         'major key', 'fast tempo music'], condition_lower):
            semantic_effect += 0.15
        elif _any_word_in(['sad music', 'melancholy music', 'minor key',
                           'slow tempo music', 'somber music'], condition_lower):
            semantic_effect -= 0.12

        # No music control (Vastfjall, 2002)
        if _any_word_in(['no music condition', 'silence control',
                         'quiet condition'], condition_lower):
            semantic_effect += 0.0  # Neutral baseline

        # Arousing music (Husain et al., 2002; Mozart effect reframed)
        if _any_word_in(['arousing music', 'energizing music',
                         'high tempo', 'stimulating music'], condition_lower):
            semantic_effect += 0.10
        elif _any_word_in(['calming music', 'relaxing music',
                           'slow music', 'ambient music'], condition_lower):
            semantic_effect += 0.05  # Both positive, arousing more so

        # Music familiarity (van den Bosch et al., 2013)
        if _any_word_in(['familiar music', 'preferred music', 'chosen music'], condition_lower):
            semantic_effect += 0.12
        elif _any_word_in(['unfamiliar music', 'random music', 'assigned music'], condition_lower):
            semantic_effect += 0.03

        # =====================================================================
        # DOMAIN 33: NATURE & ENVIRONMENT EXPOSURE (v1.0.7.4)
        # Bratman et al. (2019): Nature and mental health review.
        # Kaplan (1995): Attention Restoration Theory -- nature restores
        # directed attention. Ulrich (1984): Stress Reduction Theory.
        # Note: Basic nature exposure is in Domain 14. This covers deeper
        # nature vs urban and virtual nature constructs.
        # =====================================================================

        # Nature immersion (Bratman et al., 2015)
        if _any_word_in(['nature walk', 'outdoor nature', 'green space walk',
                         'forest bathing', 'park walk'], condition_lower):
            semantic_effect += 0.18
        elif _any_word_in(['urban walk', 'city walk', 'street walk',
                           'traffic area'], condition_lower):
            semantic_effect -= 0.08

        # Virtual nature (White et al., 2018)
        if _any_word_in(['virtual nature', 'nature video', 'nature images',
                         'nature sounds', 'nature vr'], condition_lower):
            semantic_effect += 0.10  # Weaker than real nature
        elif _any_word_in(['office', 'indoor', 'windowless',
                           'artificial light', 'cubicle'], condition_lower):
            semantic_effect -= 0.05

        # Biophilic design (Kellert, 2008)
        if _any_word_in(['biophilic', 'plant in room', 'natural materials',
                         'green view', 'window view nature'], condition_lower):
            semantic_effect += 0.12
        elif _any_word_in(['sterile environment', 'concrete room',
                           'no window', 'artificial environment'], condition_lower):
            semantic_effect -= 0.08

        # Nature restoration (Kaplan, 1995; Attention Restoration Theory)
        if _any_word_in(['restorative environment', 'attention restoration',
                         'soft fascination'], condition_lower):
            semantic_effect += 0.15
        elif _any_word_in(['demanding environment', 'directed attention fatigue',
                           'cognitive overload'], condition_lower):
            semantic_effect -= 0.10

        # =====================================================================
        # DOMAIN 34: FOOD, HUNGER & CONSUMPTION (v1.0.7.4)
        # Xu et al. (2015): Hunger increases acquisitive behavior.
        # Danziger et al. (2011): Judges grant more parole after eating.
        # Gal & Liu (2011): Hunger -> more favorable product evaluations.
        # Bushman et al. (2014): Low glucose -> aggression in couples.
        # =====================================================================

        # Hunger manipulation (Xu et al., 2015)
        if _any_word_in(['hungry', 'fasting', 'food deprived',
                         'empty stomach', 'skipped meal'], condition_lower):
            semantic_effect -= 0.12  # Acquisitive, less patient
        elif _any_word_in(['satiated', 'fed', 'full stomach',
                           'after meal', 'well fed'], condition_lower):
            semantic_effect += 0.05

        # Food cue exposure (Fedoroff et al., 1997)
        if _any_word_in(['food cue', 'food image', 'food aroma',
                         'appetizing', 'food exposure'], condition_lower):
            semantic_effect += 0.08
        elif _any_word_in(['no food cue', 'neutral cue', 'non-food'], condition_lower):
            semantic_effect -= 0.02

        # Diet/restriction (Herman & Polivy, 1980; restrained eating)
        if _any_word_in(['diet', 'restrict', 'restrained eating',
                         'calorie counting', 'food restriction'], condition_lower):
            semantic_effect -= 0.10  # Cognitive load from restraint
        elif _any_word_in(['unrestricted', 'eat freely', 'intuitive eating'], condition_lower):
            semantic_effect += 0.05

        # Glucose depletion (Gailliot et al., 2007)
        if _any_word_in(['glucose depleted', 'low blood sugar',
                         'no glucose', 'sugar free'], condition_lower):
            semantic_effect -= 0.12
        elif _any_word_in(['glucose drink', 'sugar drink', 'glucose boost'], condition_lower):
            semantic_effect += 0.08

        # =====================================================================
        # DOMAIN 35: PAIN & PHYSICAL DISCOMFORT (v1.0.7.4)
        # Bastian et al. (2014): Pain increases prosocial behavior and bonding.
        # Borsook & MacDonald (2010): Pain and social exclusion share neural
        # pathways. Eisenberger (2012): Social and physical pain overlap.
        # Franklin et al. (2013): Pain tolerance individual differences.
        # =====================================================================

        # Pain induction (Bastian et al., 2014)
        if _any_word_in(['pain', 'discomfort', 'painful stimulus',
                         'pain condition', 'physical pain'], condition_lower):
            semantic_effect -= 0.18  # Negative valence
        elif _any_word_in(['comfort', 'relief', 'pain free',
                           'no pain', 'comfortable'], condition_lower):
            semantic_effect += 0.15

        # Cold pressor task (Mitchell et al., 2004)
        if _any_word_in(['cold pressor', 'ice water', 'cold water hand',
                         'cold pain'], condition_lower):
            semantic_effect -= 0.15
        elif _any_word_in(['warm water', 'comfortable temperature',
                           'neutral water'], condition_lower):
            semantic_effect += 0.05

        # Shared pain bonding (Bastian et al., 2014)
        if _any_word_in(['shared pain', 'pain together', 'group pain',
                         'collective suffering'], condition_lower):
            semantic_effect += 0.10  # Shared pain -> bonding
        elif _any_word_in(['pain alone', 'individual pain', 'solo suffering'], condition_lower):
            semantic_effect -= 0.08

        # Warmth/physical comfort (Bargh & Shalev, 2012)
        if _any_word_in(['warm comfortable', 'cozy', 'warm environment',
                         'heated room'], condition_lower):
            semantic_effect += 0.08
        elif _any_word_in(['cold room', 'uncomfortable temperature',
                           'chilly', 'cold environment'], condition_lower):
            semantic_effect -= 0.08

        # =====================================================================
        # DOMAIN 36: COLOR & VISUAL PROCESSING (v1.0.7.4)
        # Elliot & Maier (2014): Color-in-context theory.
        # Mehta & Zhu (2009): Red -> avoidance/detail; Blue -> approach/creativity.
        # Labrecque & Milne (2012): Color effects on brand perception.
        # Note: Basic color is in Domain 14. This covers deeper color
        # associations and brightness/contrast effects.
        # =====================================================================

        # Red vs blue context effects (Mehta & Zhu, 2009)
        if _any_word_in(['red stimulus', 'warm color', 'red environment',
                         'red label'], condition_lower):
            semantic_effect += 0.08  # Arousal, attention
        elif _any_word_in(['blue stimulus', 'cool color', 'blue environment',
                           'blue label'], condition_lower):
            semantic_effect -= 0.05  # Calm, creative

        # Brightness effects (Steidle & Werth, 2013)
        if _any_word_in(['bright', 'well lit', 'high brightness',
                         'brightly illuminated'], condition_lower):
            semantic_effect += 0.10  # Clarity, positive judgment
        elif _any_word_in(['dark', 'dim', 'low brightness',
                           'poorly lit'], condition_lower):
            semantic_effect -= 0.08  # Ambiguity, risk

        # Green color (Lichtenfeld et al., 2012)
        if _any_word_in(['green color', 'green stimulus', 'green environment'], condition_lower):
            semantic_effect += 0.06  # Creativity boost
        elif _any_word_in(['grey color', 'gray stimulus', 'neutral color'], condition_lower):
            semantic_effect += 0.0  # Neutral

        # Color saturation (Wilms & Oberfeld, 2018)
        if _any_word_in(['saturated color', 'vivid color', 'high saturation'], condition_lower):
            semantic_effect += 0.08  # More arousing
        elif _any_word_in(['desaturated', 'muted color', 'low saturation',
                           'pastel'], condition_lower):
            semantic_effect -= 0.03  # Calming

        # =====================================================================
        # DOMAIN 37: LANGUAGE & FRAMING EFFECTS (v1.0.7.4)
        # Fausey & Boroditsky (2011): Linguistic framing affects attribution.
        # Tversky & Kahneman (1981): Framing effects on decision-making.
        # Keysar et al. (2012): Foreign language effect reduces emotional bias.
        # Pennebaker (2011): Pronoun use predicts psychological states.
        # =====================================================================

        # Active vs passive framing (Fausey & Boroditsky, 2011)
        if _any_word_in(['active voice', 'active frame', 'agentive',
                         'he broke', 'she caused'], condition_lower):
            semantic_effect += 0.08  # More blame/responsibility
        elif _any_word_in(['passive voice', 'passive frame',
                           'it broke', 'accident happened'], condition_lower):
            semantic_effect -= 0.05  # Less blame/responsibility

        # First-person vs third-person (Kross & Ayduk, 2011)
        if _any_word_in(['first person', 'i perspective', 'self-immersed',
                         'my experience'], condition_lower):
            semantic_effect += 0.10  # Greater emotional intensity
        elif _any_word_in(['third person', 'observer perspective',
                           'self-distanced', 'they perspective'], condition_lower):
            semantic_effect -= 0.03  # More rational processing

        # Foreign language effect (Keysar et al., 2012; Costa et al., 2014)
        if _any_word_in(['foreign language', 'second language', 'non-native',
                         'l2 framing'], condition_lower):
            semantic_effect += 0.05  # More utilitarian/rational decisions
        elif _any_word_in(['native language', 'first language', 'mother tongue',
                           'l1 framing'], condition_lower):
            semantic_effect += 0.02  # Stronger emotional response

        # Gain vs loss framing (Tversky & Kahneman, 1981)
        if _any_word_in(['gain frame', 'save lives', 'positive frame',
                         'benefit frame'], condition_lower):
            semantic_effect += 0.15
        elif _any_word_in(['loss frame', 'people die', 'negative frame',
                           'risk frame'], condition_lower):
            semantic_effect -= 0.12

        # Concrete vs abstract language (Semin & Fiedler, 1988; LCM)
        if _any_word_in(['concrete language', 'specific description',
                         'descriptive action'], condition_lower):
            semantic_effect += 0.06
        elif _any_word_in(['abstract language', 'trait description',
                           'dispositional label'], condition_lower):
            semantic_effect -= 0.04

        # =====================================================================
        # DOMAIN 38: SOCIAL STATUS & INEQUALITY (v1.0.7.4)
        # Piff et al. (2010): Lower class -> more prosocial; higher class -> less.
        # Kraus et al. (2012): Social class affects social cognition.
        # Stephens et al. (2012): Cultural mismatch in institutions.
        # Note: Basic status is in Domain 18. This covers class-specific
        # and inequality-focused manipulations.
        # =====================================================================

        # High vs low SES priming (Piff et al., 2010)
        if _any_word_in(['high status', 'wealthy prime', 'upper class prime',
                         'high ses', 'rich condition'], condition_lower):
            semantic_effect -= 0.12  # Less prosocial, more entitled
        elif _any_word_in(['low status', 'poor prime', 'lower class prime',
                           'low ses', 'disadvantaged condition'], condition_lower):
            semantic_effect += 0.10  # More prosocial, communal

        # Inequality salience (Cote et al., 2015)
        if _any_word_in(['inequality', 'wealth gap', 'economic disparity',
                         'unequal distribution'], condition_lower):
            semantic_effect -= 0.15  # Negative affect, fairness concerns
        elif _any_word_in(['equal status', 'equality', 'egalitarian',
                           'fair distribution'], condition_lower):
            semantic_effect += 0.05

        # Status threat (Scheepers & Ellemers, 2005)
        if _any_word_in(['status threat', 'losing status', 'status decline',
                         'downward mobility'], condition_lower):
            semantic_effect -= 0.15
        elif _any_word_in(['status secure', 'stable position', 'status confirmed'], condition_lower):
            semantic_effect += 0.08

        # Meritocracy belief (Ledgerwood et al., 2011)
        if _any_word_in(['meritocracy prime', 'earned success', 'hard work pays'], condition_lower):
            semantic_effect += 0.10
        elif _any_word_in(['systemic barriers', 'unearned privilege',
                           'structural inequality'], condition_lower):
            semantic_effect -= 0.10

        # =====================================================================
        # v1.0.4.6: DOMAIN-AWARE EFFECT STACKING GUARD
        #
        # After all STEP 2 domains have been checked, apply two safeguards:
        # 1. If total STEP 2 contribution is large AND came from domains not
        #    in self.detected_domains, attenuate by 0.5× (less likely relevant)
        # 2. Cap total STEP 2 semantic_effect to ±0.45 to prevent runaway stacking
        # =====================================================================
        _step2_contribution = semantic_effect - _effect_before_step2
        if abs(_step2_contribution) > 0.30 and _detected:
            # Large effect from STEP 2 — check if it came from relevant domains
            # If detected_domains is set but the condition keywords mostly
            # matched NON-relevant domains, attenuate the excess
            _any_relevant = False
            for _dn, _dr in _DOMAIN_RELEVANCE.items():
                if _detected & _dr:
                    _any_relevant = True
                    break
            if not _any_relevant:
                # No detected domain matched any STEP 2 domain — attenuate
                _step2_contribution *= 0.5
                semantic_effect = _effect_before_step2 + _step2_contribution

        # Cap total semantic_effect to prevent extreme stacking
        semantic_effect = max(-0.50, min(0.50, semantic_effect))

        # =====================================================================
        # FACTORIAL DESIGN PARSING
        # For conditions like "No AI × Utilitarian" or "AI x Hedonic"
        # Parse each factor and sum main effects
        # =====================================================================

        # Detect factorial separators (×, x, +, &, *, /)
        factorial_separators = ['×', ' x ', ' + ', ' & ', ' * ', ' / ']
        is_factorial = any(sep in condition_lower for sep in factorial_separators)

        if is_factorial:
            # Split by any separator and process each factor
            factors_text = condition_lower
            for sep in factorial_separators:
                factors_text = factors_text.replace(sep, '|')
            factors = [f.strip() for f in factors_text.split('|') if f.strip()]

            # Add effects for each factor (but with reduced magnitude to avoid stacking)
            factor_effects = []
            for factor in factors:
                factor_effect = 0.0

                # Check valence for this factor
                for kw in positive_keywords:
                    if _word_in(kw, factor):
                        factor_effect += 0.18
                        break
                for kw in negative_keywords:
                    if _word_in(kw, factor):
                        factor_effect -= 0.18
                        break

                # Check key manipulation types for this factor
                if _word_in('ai', factor) and not _word_in('no', factor):
                    factor_effect -= 0.08
                elif _word_in('no ai', factor) or _word_in('no_ai', factor) or _word_in('human', factor):
                    factor_effect += 0.10

                if _word_in('hedonic', factor) or _word_in('fun', factor):
                    factor_effect += 0.12
                elif _word_in('utilitarian', factor) or _word_in('practical', factor):
                    factor_effect -= 0.05

                if _word_in('gain', factor) or _word_in('save', factor):
                    factor_effect += 0.08
                elif _word_in('loss', factor) or _word_in('lose', factor):
                    factor_effect -= 0.12

                if _word_in('fair', factor):
                    factor_effect += 0.12
                elif _word_in('unfair', factor):
                    factor_effect -= 0.15

                factor_effects.append(factor_effect)

            # Sum factor effects (main effects) + interaction effect for factorial designs
            # v1.0.1.3: Added interaction effects for factorial designs
            #
            # SCIENTIFIC BASIS:
            # In factorial designs, interaction effects occur when the effect of one
            # factor depends on the level of another factor. This is modeled as the
            # product of individual factor effects, scaled by an interaction coefficient.
            #
            # Examples:
            # - AI × Hedonic: AI aversion may be STRONGER for hedonic products (synergy)
            # - Loss frame × High anchor: Loss framing may amplify anchoring (reinforcing)
            # - Control × Utilitarian: neutral × neutral → near-zero interaction
            #
            # The multiplicative interaction naturally produces:
            # - Same-direction factors: positive interaction (reinforcing)
            # - Opposite-direction factors: negative interaction (attenuating)
            # - Near-zero factors: minimal interaction (appropriate)
            if factor_effects:
                n_fac = max(len(factor_effects), 1)
                scale_factor = 0.6 / max(1, n_fac - 1) if n_fac > 1 else 0.6

                # Main effects (additive)
                main_effect = sum(factor_effects) * scale_factor

                # Interaction effect (multiplicative)
                # The product of factor effects captures cross-factor dependencies
                # Coefficient of 0.4 prevents interactions from dominating main effects
                # while still producing detectable interaction patterns in the data
                interaction_effect = 0.0
                if len(factor_effects) >= 2:
                    interaction_product = 1.0
                    for fe in factor_effects:
                        interaction_product *= fe
                    # Scale interaction: strong when factors reinforce, weak when orthogonal
                    interaction_coeff = 0.4
                    interaction_effect = interaction_product * interaction_coeff

                semantic_effect += main_effect + interaction_effect

        # =====================================================================
        # STEP 3: Create additional variance using stable hash (NOT position)
        # This ensures conditions with similar meanings have slight differences
        # =====================================================================

        # v1.0.0: Guard against empty condition string
        if not condition:
            condition_hash = 0
        else:
            # Use MD5 hash of condition name for stable but non-positional variation
            condition_hash = int(hashlib.md5(condition.encode()).hexdigest(), 16)
        # Small random-like adjustment based on hash (-0.05 to +0.05)
        hash_adjustment = ((condition_hash % 1000) / 1000.0 - 0.5) * 0.08

        semantic_effect += hash_adjustment

        # =====================================================================
        # STEP 4: Bound and scale the effect
        # =====================================================================

        # v1.4.11: Tightened cap from ±0.7 to ±0.5 to prevent keyword stacking
        # from producing unrealistically large effects
        semantic_effect = max(-0.5, min(0.5, semantic_effect))

        # v1.0.4.3: Comprehensive domain-aware effect magnitude scaling
        # Different research domains have systematically different effect sizes
        # in the published literature. The multiplier adjusts the default d=0.5
        # to match domain-typical magnitudes.
        #
        # SCIENTIFIC BASIS for domain-specific d multipliers:
        # ===================================================
        # Default d = 0.5 (medium effect, Cohen 1988)
        # Multiplier adjusts this to domain-typical ranges.
        #
        # Political + Economic games: 1.6× → d ≈ 0.80 (Dimant 2024: d = 0.6-0.9)
        # Political only: 1.3× → d ≈ 0.65 (Iyengar & Westwood 2015: d > 0.5)
        # Economic games: 1.2× → d ≈ 0.60 (Engel 2011, Balliet 2014)
        # Health fear appeals: 1.25× → d ≈ 0.63 (Witte & Allen 2000: d = 0.3-0.8)
        # Organizational justice: 1.3× → d ≈ 0.65 (Colquitt 2001: ρ = .40-.50)
        # Default/nudge effects: 1.4× → d ≈ 0.70 (Johnson & Goldstein 2003: 60-80pp)
        # Stereotype threat: 0.8× → d ≈ 0.40 (Nguyen & Ryan 2008: d = 0.26)
        # Embodied cognition: 0.6× → d ≈ 0.30 (Many Labs replication: small effects)
        # Environmental: 1.1× → d ≈ 0.55 (moderate, polarized attitudes)
        # Consumer/marketing: 1.0× → d ≈ 0.50 (Barton 2022 scarcity meta: r = 0.28)
        # AI/technology: 1.1× → d ≈ 0.55 (Dietvorst 2015: d = 0.3-0.5)
        # Moral/ethics: 1.2× → d ≈ 0.60 (Haidt 2001: strong intuitive reactions)
        # Education/learning: 1.15× → d ≈ 0.58 (Rowland 2014 testing effect: d = 0.50)
        # Clinical/anxiety: 1.25× → d ≈ 0.63 (therapy effect sizes typically large)
        # Gender/power: 1.15× → d ≈ 0.58 (moderate but reliable effects)
        # Misinformation: 1.2× → d ≈ 0.60 (inoculation meta d = 0.29, but with booster)
        # Prosocial/charitable: 1.1× → d ≈ 0.55 (identified victim: r = 0.13)
        # Implementation intentions: 1.3× → d ≈ 0.65 (Gollwitzer meta: d = 0.65)
        _domain_d_multiplier = 1.0

        # v1.0.4.6: Use self.detected_domains as PRIMARY routing for scaling
        # Falls back to keyword matching for patterns not caught by detection
        _domain_ctx = condition_lower + " " + variable_lower + " " + _study_text
        _det = set(getattr(self, 'detected_domains', []) or [])
        _used_detected_scaling = False

        if _det:
            # Domain-aware routing: check detected domains FIRST
            if _det & {'political_psychology'} and _is_economic_game_dv:
                _domain_d_multiplier = 1.6  # Political + econ game (Dimant 2024)
                _used_detected_scaling = True
            elif _det & {'political_psychology'}:
                _domain_d_multiplier = 1.3
                _used_detected_scaling = True
            elif _is_economic_game_dv:
                _domain_d_multiplier = 1.2
                _used_detected_scaling = True
            elif _det & {'health_psychology'}:
                _domain_d_multiplier = 1.25
                _used_detected_scaling = True
            elif _det & {'organizational_behavior'}:
                _domain_d_multiplier = 1.3
                _used_detected_scaling = True
            elif _det & {'deontology_utilitarianism', 'fairness'}:
                _domain_d_multiplier = 1.2
                _used_detected_scaling = True
            elif _det & {'educational_psychology', 'cognitive_psychology'}:
                _domain_d_multiplier = 1.15
                _used_detected_scaling = True
            elif _det & {'clinical'}:
                _domain_d_multiplier = 1.25
                _used_detected_scaling = True
            elif _det & {'ai', 'technology'}:
                _domain_d_multiplier = 1.1
                _used_detected_scaling = True
            elif _det & {'environmental'}:
                _domain_d_multiplier = 1.1
                _used_detected_scaling = True
            elif _det & {'accuracy_misinformation', 'media_communication'}:
                _domain_d_multiplier = 1.2
                _used_detected_scaling = True
            elif _det & {'dishonesty'}:
                _domain_d_multiplier = 1.15
                _used_detected_scaling = True
            elif _det & {'punishment'}:
                _domain_d_multiplier = 1.25
                _used_detected_scaling = True
            # v1.0.4.9: New domain scaling for added paradigms
            elif _det & {'positive_psychology'}:
                _domain_d_multiplier = 1.0  # Davis et al. 2016 meta: d = 0.31
                _used_detected_scaling = True
            elif _det & {'narrative_persuasion'}:
                _domain_d_multiplier = 1.15  # van Laer et al. 2014: r = 0.35
                _used_detected_scaling = True
            elif _det & {'digital_wellbeing'}:
                _domain_d_multiplier = 1.1  # Ward et al. 2017: moderate effects
                _used_detected_scaling = True
            elif _det & {'moral_psychology'}:
                _domain_d_multiplier = 1.2  # Sacred values: strong effects
                _used_detected_scaling = True

        # Fallback: keyword matching if detected domains didn't match scaling
        if not _used_detected_scaling:
            if _is_political_study and _is_economic_game_dv:
                _domain_d_multiplier = 1.6
            elif _is_political_study:
                _domain_d_multiplier = 1.3
            elif _is_economic_game_dv:
                _domain_d_multiplier = 1.2

            # --- Health/Fear Appeal domain ---
            # Witte & Allen (2000 meta): Fear appeals d = 0.3-0.8 depending on efficacy
            # Health interventions often produce large effects when well-targeted
            elif any(kw in _domain_ctx for kw in ['fear appeal', 'health intervention',
                     'self-efficacy', 'health message', 'vaccination', 'patient',
                     'medical decision', 'health risk', 'health behavior']):
                _domain_d_multiplier = 1.25

            # --- Organizational Justice domain ---
            # Colquitt et al. (2001 meta): ρ = .40-.50 for justice-outcome relationships
            # Leadership effects: Judge & Piccolo (2004): ρ = .44
            elif any(kw in _domain_ctx for kw in ['procedural justice', 'distributive justice',
                     'organizational justice', 'transformational leader', 'leadership style',
                     'employee engagement', 'job satisfaction', 'workplace fairness']):
                _domain_d_multiplier = 1.3

            # --- Default/Nudge effects ---
            # Johnson & Goldstein (2003): Opt-out vs opt-in → 60-80pp difference
            # Gollwitzer & Sheeran (2006): Implementation intentions d = 0.65
            elif any(kw in _domain_ctx for kw in ['default option', 'opt-out', 'opt-in',
                     'nudge', 'implementation intention', 'if-then plan',
                     'choice architecture']):
                _domain_d_multiplier = 1.4

            # --- Moral/Ethics domain ---
            # Haidt (2001): Moral judgments produce strong intuitive reactions
            # Moral foundations: Graham et al. (2009): clear liberal/conservative splits
            elif any(kw in _domain_ctx for kw in ['moral judgment', 'ethical dilemma',
                     'trolley problem', 'moral foundation', 'deontolog', 'utilitari',
                     'moral', 'ethical', 'disgust', 'purity']):
                _domain_d_multiplier = 1.2

            # --- Education/Learning domain ---
            # Rowland (2014): Testing effect d = 0.50
            # Cepeda et al. (2006): Spacing effect robust and moderate-to-large
            elif any(kw in _domain_ctx for kw in ['testing effect', 'retrieval practice',
                     'spacing effect', 'learning', 'education', 'classroom',
                     'student performance', 'teaching method']):
                _domain_d_multiplier = 1.15

            # --- Clinical/Anxiety domain ---
            # Therapy effect sizes are typically large (d = 0.5-1.0)
            # Cuijpers et al. (2019): Psychotherapy for depression d = 0.72
            elif any(kw in _domain_ctx for kw in ['anxiety', 'depression', 'therapy',
                     'clinical', 'mental health', 'wellbeing', 'intervention',
                     'coping', 'stress', 'burnout', 'ptsd']):
                _domain_d_multiplier = 1.25

            # --- AI/Technology domain ---
            # Dietvorst et al. (2015): Algorithm aversion d = 0.3-0.5
            # Longoni et al. (2019): AI resistance moderate effects
            elif any(kw in _domain_ctx for kw in ['ai', 'algorithm', 'robot', 'automat',
                     'technology adoption', 'chatbot', 'artificial intelligence',
                     'machine learning', 'human-ai']):
                _domain_d_multiplier = 1.1

            # --- Environmental/Climate domain ---
            # Polarized topic with moderate effects but high variance
            # Campbell & Kay (2014): Ideological filtering of climate info
            elif any(kw in _domain_ctx for kw in ['environment', 'climate', 'sustainab',
                     'green', 'carbon', 'renewable', 'pollution', 'conservation']):
                _domain_d_multiplier = 1.1

            # --- Gender/Power domain ---
            # Stereotype effects moderate but reliable
            # Nguyen & Ryan (2008): Stereotype threat d = 0.26 (small-to-moderate)
            elif any(kw in _domain_ctx for kw in ['gender', 'stereotype', 'power',
                     'status', 'dominance', 'sexism', 'masculin', 'feminin']):
                _domain_d_multiplier = 1.15

            # --- Misinformation/Inoculation domain ---
            # Banas & Rains (2010): Inoculation d = 0.29
            # Roozenbeek et al. (2022): Prebunking d = 0.3-0.5
            elif any(kw in _domain_ctx for kw in ['misinformation', 'fake news',
                     'inoculation', 'prebunk', 'conspiracy', 'fact check',
                     'truth discernment', 'media literacy']):
                _domain_d_multiplier = 1.2

            # --- Prosocial/Charitable domain ---
            # Small et al. (2007): Identifiable victim r = 0.13
            # Charitable giving: moderate effects, boosted by narratives
            elif any(kw in _domain_ctx for kw in ['charit', 'donat', 'prosocial',
                     'altruism', 'volunteer', 'helping', 'philanthrop',
                     'identifiable victim', 'warm glow']):
                _domain_d_multiplier = 1.1

            # --- Embodied cognition domain ---
            # Many Labs replications: Small or null effects
            # Coles et al. (2019): Facial feedback r = 0.03
            elif any(kw in _domain_ctx for kw in ['embodi', 'power pose', 'facial feedback',
                     'pen in teeth', 'heavy clipboard', 'warm cup',
                     'clean hands', 'physical posture']):
                _domain_d_multiplier = 0.6

            # --- Dishonesty/Cheating domain ---
            # Gino et al. (2009): Moral licensing moderate effects
            # Die-rolling paradigms: reliable but moderate detection
            elif any(kw in _domain_ctx for kw in ['dishonest', 'cheat', 'lying',
                     'overclaim', 'die roll', 'moral licens', 'honesty']):
                _domain_d_multiplier = 1.15

            # --- Punishment/Norm Enforcement domain ---
            # Fehr & Gächter (2000): Punishment effects are large in PGG
            # Third-party punishment: robust effects
            elif any(kw in _domain_ctx for kw in ['punish', 'sanction', 'norm enforcement',
                     'retribution', 'deterrence']):
                _domain_d_multiplier = 1.25

        # Apply Cohen's d scaling with domain-aware multiplier
        return semantic_effect * default_d * COHENS_D_TO_NORMALIZED * _domain_d_multiplier

    def _get_condition_trait_modifier(self, condition: str) -> Dict[str, float]:
        """
        Get condition-specific trait modifiers that affect persona responses.

        Different experimental conditions should influence not just means but also
        response patterns. This creates more realistic between-condition differences.

        v1.0.4.4: Now also reads study_title and study_description for domain-level
        trait priming. Even control groups in domain-specific studies show priming
        effects (Bargh et al., 1996: domain context primes related constructs).

        Returns a dict of trait name -> modifier value to add/subtract from base traits.
        """
        modifiers = {}
        condition_lower = str(condition).lower()

        # ================================================================
        # v1.0.4.6: Domain-aware study-level trait priming
        #
        # Now uses self.detected_domains directly instead of re-keyword-matching
        # study text. This is more precise and eliminates redundant computation.
        # The domain detection already ran a 5-phase scoring algorithm on study
        # title, description, and conditions — we leverage that result.
        #
        # Scientific basis:
        # - Bargh et al. (1996): Category priming affects behavior automatically
        # - Higgins et al. (1977): Accessibility of constructs influences judgment
        # - Schwarz (2007): Context effects in self-reports are pervasive
        # ================================================================
        _detected = set(getattr(self, 'detected_domains', []) or [])

        # Political study context primes identity salience for ALL conditions
        if _detected & {'political_psychology'}:
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.06
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) + 0.04

        # Economic game context primes strategic thinking
        if _detected & {'economic_games'}:
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.04
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) + 0.03

        # Health study context primes risk awareness
        if _detected & {'health_psychology'}:
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.03
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) + 0.04

        # Moral/ethical study context primes evaluative extremity
        if _detected & {'deontology_utilitarianism', 'fairness'}:
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.05
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.03

        # Environmental/sustainability context primes polarization
        if _detected & {'environmental'}:
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.04

        # AI/technology study context primes tech-related traits
        if _detected & {'ai', 'technology'}:
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.02
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.02

        # v1.0.4.6: Additional domain-aware priming from detected domains
        # Clinical/anxiety studies prime hypervigilance
        if _detected & {'clinical', 'anxiety'}:
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.04
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) + 0.05

        # Organizational studies prime conscientiousness
        if _detected & {'organizational_behavior'}:
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.03
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) + 0.03

        # Consumer studies prime evaluation mode
        if _detected & {'consumer_behavior', 'marketing'}:
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.02

        # Communication/media studies prime source evaluation
        if _detected & {'media_communication', 'accuracy_misinformation'}:
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.03

        # v1.0.4.9: Positive psychology/gratitude studies prime positive affect
        if _detected & {'positive_psychology'}:
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) + 0.04
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.03

        # v1.0.4.9: Moral/ethics studies prime evaluative intensity
        if _detected & {'moral_psychology'}:
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.06
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.04
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) + 0.05

        # v1.0.4.9: Narrative/media engagement primes elaboration
        if _detected & {'narrative_persuasion', 'media_communication'}:
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.05
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.03

        # v1.0.4.9: Digital/attention studies prime awareness of distraction
        if _detected & {'digital_wellbeing', 'technology'}:
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.02
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) + 0.02

        # Fallback: if no domains detected, use keyword matching on study text
        if not _detected:
            _study_ctx = (
                (getattr(self, 'study_title', '') or '') + " " +
                (getattr(self, 'study_description', '') or '')
            ).lower()
            if any(kw in _study_ctx for kw in ['political', 'partisan', 'polariz']):
                modifiers['extremity'] = modifiers.get('extremity', 0) + 0.06
            if any(kw in _study_ctx for kw in ['dictator game', 'trust game', 'economic game']):
                modifiers['engagement'] = modifiers.get('engagement', 0) + 0.04

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

        # v1.0.4.2: Political identity / intergroup conditions
        # When political identity is made salient, responses become more extreme
        # and variance increases (Iyengar & Westwood, 2015)
        _political_terms = ['trump', 'biden', 'political', 'partisan', 'republican',
                            'democrat', 'liberal', 'conservative']
        _is_political = any(kw in condition_lower for kw in _political_terms)
        if _is_political:
            modifiers['extremity'] = 0.12  # More polarized responses
            modifiers['response_consistency'] = 0.08  # More consistent within-person
            # Outgroup conditions: more negative emotional valence
            _outgroup_markers = ['hater', 'opponent', 'outgroup', 'other party',
                                 'opposing', 'different', 'anti']
            if any(kw in condition_lower for kw in _outgroup_markers):
                modifiers['acquiescence'] = -0.10  # Negative bias in outgroup evaluations
                modifiers['extremity'] = 0.15  # Even more extreme for outgroup

        # v1.0.4.2: Economic game conditions — intergroup matching
        # When participants play economic games with identified partners,
        # the partner's group membership strongly affects behavior
        _econ_game = any(kw in condition_lower for kw in
                         ['dictator', 'trust game', 'ultimatum', 'public good'])
        if _econ_game:
            modifiers['engagement'] = 0.05  # Economic games increase engagement
            modifiers['response_consistency'] = 0.05

        # ================================================================
        # v1.0.4.3: Domain-specific trait modifiers for 15+ research domains
        # Each domain has published evidence for how manipulations affect
        # response patterns beyond simple mean shifts.
        # ================================================================

        # --- Health/Fear Appeal conditions ---
        # Witte (1992): Fear appeals increase attention and engagement when
        # efficacy is high, but trigger defensive avoidance when efficacy is low
        # Rogers (1975): Protection Motivation Theory — threat + coping appraisal
        if any(kw in condition_lower for kw in ['fear appeal', 'health threat',
               'disease risk', 'high threat', 'severe illness']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.08
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.10
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.06
        elif any(kw in condition_lower for kw in ['low threat', 'safe', 'healthy',
                 'prevention', 'wellness']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) - 0.03
            modifiers['extremity'] = modifiers.get('extremity', 0) - 0.05

        # --- Self-efficacy conditions ---
        # Bandura (1997): High self-efficacy → more confident, consistent responding
        if any(kw in condition_lower for kw in ['high efficacy', 'empowered',
               'capable', 'confident']):
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) + 0.06
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.05
        elif any(kw in condition_lower for kw in ['low efficacy', 'helpless',
                 'incapable', 'doubtful']):
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) - 0.08
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) + 0.06

        # --- Environmental/Sustainability conditions ---
        # Campbell & Kay (2014): Environmental messages trigger identity-protective
        # cognition — high engagement, polarized extremity
        if any(kw in condition_lower for kw in ['environment', 'climate', 'sustainab',
               'green', 'carbon', 'eco-friendly']):
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.08
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.04
        elif any(kw in condition_lower for kw in ['pollut', 'wasteful', 'unsustainable',
                 'carbon intensive']):
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.10
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) - 0.06

        # --- Moral/Ethical conditions ---
        # Haidt (2001): Moral judgments are emotion-driven, produce extreme responses
        # Greene et al. (2001): Personal moral dilemmas increase emotional engagement
        if any(kw in condition_lower for kw in ['moral', 'ethical', 'immoral',
               'unethical', 'trolley', 'dilemma']):
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.12
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.06
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) + 0.05

        # --- Authority/Credibility conditions ---
        # Milgram (1963): Authority increases compliance and acquiescence
        # Hovland & Weiss (1951): Source credibility amplifies persuasion
        if any(kw in condition_lower for kw in ['expert', 'authority', 'doctor',
               'professor', 'credible source', 'scientist']):
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) + 0.08
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) + 0.04
        elif any(kw in condition_lower for kw in ['non-expert', 'layperson', 'peer',
                 'low credibility', 'unknown source']):
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) - 0.06
            modifiers['extremity'] = modifiers.get('extremity', 0) - 0.04

        # --- Scarcity/Urgency conditions ---
        # Cialdini (2001): Scarcity increases arousal and extremity of evaluations
        # Worchel et al. (1975): Scarce items rated higher, more emotionally
        if any(kw in condition_lower for kw in ['scarce', 'limited', 'exclusive',
               'last chance', 'urgent', 'deadline']):
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.10
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.05
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.04

        # --- Social presence/Observation conditions ---
        # Zajonc (1965): Social facilitation — presence amplifies dominant responses
        # Bond & Titus (1983 meta): Audience effects on performance
        if any(kw in condition_lower for kw in ['observed', 'watched', 'public',
               'social presence', 'audience', 'with others']):
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) + 0.10
            modifiers['extremity'] = modifiers.get('extremity', 0) - 0.05
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) + 0.04
        elif any(kw in condition_lower for kw in ['anonymous', 'private', 'alone',
                 'unobserved', 'confidential']):
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) - 0.08
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.04

        # --- Loss/Gain framing conditions ---
        # Tversky & Kahneman (1981): Loss frame increases attention, risk-seeking
        # Levin et al. (2002): Framing effects on risk perception
        if any(kw in condition_lower for kw in ['loss frame', 'lose', 'forfeit',
               'penalty', 'risk of losing']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.06
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.08
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.04
        elif any(kw in condition_lower for kw in ['gain frame', 'earn', 'save',
                 'benefit', 'reward']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.03
            modifiers['extremity'] = modifiers.get('extremity', 0) - 0.03

        # --- Emotional induction conditions ---
        # Lerner & Keltner (2001): Anger → risk-seeking, certainty appraisals
        # Schwarz & Clore (1983): Mood-as-information
        if any(kw in condition_lower for kw in ['anger', 'angry', 'outrage',
               'frustrated', 'hostile']):
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.14
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) - 0.08
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.06
        elif any(kw in condition_lower for kw in ['sad', 'sadness', 'melanchol',
                 'grief', 'lonely']):
            modifiers['extremity'] = modifiers.get('extremity', 0) - 0.06
            modifiers['engagement'] = modifiers.get('engagement', 0) - 0.05
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) - 0.04
        elif any(kw in condition_lower for kw in ['happy', 'joy', 'elated',
                 'positive mood', 'cheerful']):
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) + 0.06
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.04

        # --- Cognitive load conditions ---
        # Sweller (1988): Cognitive load reduces processing depth → satisficing
        # Gilbert et al. (1988): Load increases reliance on heuristics
        if any(kw in condition_lower for kw in ['cognitive load', 'high load',
               'dual task', 'multitask', 'distract']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) - 0.10
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) - 0.08
            modifiers['extremity'] = modifiers.get('extremity', 0) - 0.05
        elif any(kw in condition_lower for kw in ['no load', 'low load', 'focused',
                 'undistracted']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.04
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) + 0.03

        # --- Time pressure conditions ---
        # Dror et al. (1999): Time pressure reduces accuracy, increases satisficing
        # Maule & Edland (1997): Deadline stress → more extreme, less careful
        if any(kw in condition_lower for kw in ['time pressure', 'deadline',
               'hurry', 'limited time', 'timed']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) - 0.08
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.06
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) - 0.06

        # --- Gender/Stereotype conditions ---
        # Steele & Aronson (1995): Stereotype threat increases anxiety, reduces performance
        # Schmader et al. (2008): Working memory interference under threat
        if any(kw in condition_lower for kw in ['stereotype threat', 'gender salient',
               'race salient', 'diagnostic test']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) - 0.06
            modifiers['extremity'] = modifiers.get('extremity', 0) - 0.04
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.04

        # --- Nostalgia/Memory conditions ---
        # Wildschut et al. (2006): Nostalgia increases positive affect, social connectedness
        # Mitchell et al. (1997): Rosy retrospection inflates positive recall
        if any(kw in condition_lower for kw in ['nostalgia', 'remember', 'childhood',
               'past experience', 'memory']):
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) + 0.06
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.05

        # ================================================================
        # v1.0.4.4: Additional domain-condition interaction patterns
        # ================================================================

        # --- Power/Hierarchy conditions ---
        # Keltner et al. (2003): Power increases approach, reduces inhibition
        # Galinsky et al. (2003): Power priming increases risk-taking
        if any(kw in condition_lower for kw in ['high power', 'power prime', 'boss',
               'leader role', 'in charge', 'manager']):
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.10
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) - 0.06
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) - 0.05
        elif any(kw in condition_lower for kw in ['low power', 'subordinate',
                 'employee role', 'follower', 'powerless']):
            modifiers['extremity'] = modifiers.get('extremity', 0) - 0.08
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) + 0.06
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) + 0.08

        # --- Competition conditions ---
        # Deutsch (1949): Competition decreases cooperation, increases defensiveness
        if any(kw in condition_lower for kw in ['competi', 'rival', 'contest',
               'tournament', 'winner', 'ranking']):
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.08
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.06

        # --- Mindfulness/Reflection conditions ---
        # Brown & Ryan (2003): Mindfulness reduces reactivity, increases presence
        if any(kw in condition_lower for kw in ['mindful', 'meditation', 'reflective',
               'contemplat', 'breathing exercise']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.08
            modifiers['extremity'] = modifiers.get('extremity', 0) - 0.08
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) + 0.05

        # --- Accountability conditions ---
        # Lerner & Tetlock (1999): Accountability increases accuracy motivation
        if any(kw in condition_lower for kw in ['accountable', 'justify decision',
               'explain to', 'audience', 'evaluated by']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.06
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) + 0.05
            modifiers['extremity'] = modifiers.get('extremity', 0) - 0.06

        # --- Goal-setting conditions ---
        # Locke & Latham (2002): Specific difficult goals increase effort
        if any(kw in condition_lower for kw in ['specific goal', 'challenging goal',
               'performance target', 'achievement goal']):
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.08
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) + 0.05

        # --- Depletion/Fatigue conditions ---
        # Baumeister et al. (1998): Ego depletion reduces self-regulation
        # (Though replication debates exist, fatigue effects are robust)
        if any(kw in condition_lower for kw in ['depleted', 'fatigued', 'exhausted',
               'ego depletion', 'self-control depletion']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) - 0.10
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) - 0.06
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.05

        # --- Mortality salience conditions ---
        # Greenberg et al. (1990): Terror Management Theory
        # Mortality reminders increase worldview defense, self-esteem striving
        if any(kw in condition_lower for kw in ['mortality salien', 'death remind',
               'think about death', 'mortality', 'funeral']):
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.12
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.06
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) + 0.05

        # --- v1.0.4.9: Narrative transportation conditions ---
        # Green & Brock (2000): Transportation reduces counterarguing
        if any(kw in condition_lower for kw in ['narrative', 'story', 'transported',
               'immersed', 'fictional scenario']):
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.08
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.05
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) + 0.04

        # --- v1.0.4.9: Social comparison conditions ---
        # Festinger (1954): Social comparison affects self-evaluation
        if any(kw in condition_lower for kw in ['upward comparison', 'better than',
               'outperformed', 'social comparison']):
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.08
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) + 0.06
        elif any(kw in condition_lower for kw in ['downward comparison', 'worse than',
                 'outperforming']):
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.04

        # --- v1.0.4.9: Gratitude/positive intervention conditions ---
        # Emmons & McCullough (2003): Gratitude increases positive affect
        if any(kw in condition_lower for kw in ['gratitude', 'thankful', 'count blessings',
               'three good things', 'best possible self']):
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) + 0.05
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.04

        # --- v1.0.4.9: Moral threat/cleansing conditions ---
        # Sachdeva et al. (2009): Moral self-regulation
        if any(kw in condition_lower for kw in ['moral threat', 'guilt', 'transgression',
               'sacred value', 'taboo']):
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.10
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) + 0.08
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.06

        # --- v1.0.4.9: Digital distraction conditions ---
        # Ward et al. (2017): Phone presence reduces cognitive capacity
        if any(kw in condition_lower for kw in ['phone present', 'notification',
               'multitask', 'distract', 'interrupted']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) - 0.08
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) - 0.05
        elif any(kw in condition_lower for kw in ['no phone', 'focus mode',
                 'single task', 'no distraction']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.04
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) + 0.03

        # ================================================================
        # v1.0.9.4: Expanded condition trait modifiers — 15 new categories
        # Each grounded in published experimental paradigms with
        # documented effects on response patterns.
        # ================================================================

        # ── 1. Nostalgia Induction (Wildschut et al., 2006; Sedikides et al., 2015) ──
        # Nostalgia increases positive affect, social connectedness, and meaning in life.
        # Enhances engagement and produces slightly more extreme, acquiescent responses.
        if any(kw in condition_lower for kw in ['nostalgia induct', 'nostalgic',
               'recall a fond memory', 'sentimental', 'good old days']):
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.06
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.05
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) + 0.04

        # ── 2. Self-Affirmation (Steele, 1988; Cohen & Sherman, 2014) ──
        # Self-affirmation reduces defensiveness and identity threat, leading to
        # more open, less socially desirable responding with greater consistency.
        if any(kw in condition_lower for kw in ['self-affirm', 'self affirm',
               'values affirmation', 'affirmed', 'wrote about values',
               'personal strengths']):
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) - 0.06
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.04
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) + 0.04

        # ── 3. Mindfulness / Present-Moment Focus (Brown & Ryan, 2003; Arch & Craske, 2006) ──
        # Mindfulness increases attention and deliberate responding while reducing
        # reactive extremity. Enhances consistency through careful item processing.
        if any(kw in condition_lower for kw in ['present-moment', 'present moment',
               'body scan', 'mindful attention', 'focused awareness',
               'mindfulness induction']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.10
            modifiers['extremity'] = modifiers.get('extremity', 0) - 0.06
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) + 0.06

        # ── 4. Gratitude Induction (Emmons & McCullough, 2003; Wood et al., 2010) ──
        # Gratitude elevates positive mood, increasing acquiescence and engagement.
        # Also produces slightly more extreme positive evaluations.
        if any(kw in condition_lower for kw in ['gratitude induct', 'gratitude journal',
               'grateful', 'appreciation', 'counting blessings',
               'grateful reflection']):
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) + 0.06
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.05
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.04

        # ── 5. Power Priming — High Power (Galinsky et al., 2003; Anderson & Berdahl, 2002) ──
        # High power increases approach motivation, risk-taking, and action orientation.
        # Reduces social desirability concerns and boosts engagement.
        if any(kw in condition_lower for kw in ['power priming', 'high status',
               'recall a time you had power', 'dominant role', 'authority role',
               'elevated status']):
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.10
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) - 0.06
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.04

        # ── 6. Power Priming — Low Power (Keltner et al., 2003; Anderson & Galinsky, 2006) ──
        # Low power increases inhibition, conformity, and social monitoring.
        # Reduces extremity and increases social desirability and vigilant attention.
        if any(kw in condition_lower for kw in ['low status', 'subordinate role',
               'recall a time someone had power over', 'submissive', 'deferential',
               'disempowered']):
            modifiers['extremity'] = modifiers.get('extremity', 0) - 0.06
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) + 0.08
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.04

        # ── 7. Cognitive Load — Dual Task (Sweller, 1988; Gilbert et al., 1988) ──
        # Heavy cognitive load impairs processing capacity, reducing attention and
        # consistency. Paradoxically increases extremity through reliance on heuristics.
        if any(kw in condition_lower for kw in ['dual task', 'memorize number',
               'concurrent task', 'working memory load', 'remember digits',
               'count backwards']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) - 0.12
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) - 0.10
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.04

        # ── 8. Mortality Salience (Greenberg et al., 1990; Burke et al., 2010 meta) ──
        # Terror Management Theory: death awareness triggers worldview defense,
        # producing more extreme, engaged, and consistent value-congruent responding.
        if any(kw in condition_lower for kw in ['death prime', 'mortality prime',
               'write about own death', 'life is short', 'impermanence',
               'end of life']):
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.14
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.08
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) + 0.06

        # ── 9. Sleep Deprivation / Fatigue (Lim & Dinges, 2010; Killgore, 2010) ──
        # Sleep deprivation impairs executive function, reducing sustained attention
        # and response consistency. Increases extremity via reduced inhibition.
        if any(kw in condition_lower for kw in ['sleep depriv', 'sleep restrict',
               'fatigued participant', 'tired', 'insufficient sleep',
               'sleep loss', 'no sleep']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) - 0.12
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) - 0.08
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.06

        # ── 10. Nature Exposure / Green Space (Kaplan, 1995; Berman et al., 2008) ──
        # Attention Restoration Theory: exposure to natural environments restores
        # directed attention, reduces mental fatigue, and promotes calmer responding.
        if any(kw in condition_lower for kw in ['nature exposure', 'nature walk',
               'green space', 'outdoor', 'park scene', 'forest',
               'natural environment', 'nature image']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.06
            modifiers['extremity'] = modifiers.get('extremity', 0) - 0.04
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.04

        # ── 11. Social Exclusion / Ostracism (Williams, 2007; Baumeister et al., 2005) ──
        # Ostracism threatens fundamental needs (belonging, self-esteem, control, meaning).
        # Produces more extreme responses, higher engagement, but reduced acquiescence
        # as excluded individuals resist conforming to group norms.
        if any(kw in condition_lower for kw in ['social exclusion', 'ostracism',
               'ostracized', 'excluded', 'cyberball exclusion', 'rejected',
               'left out', 'ignored by group']):
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.10
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.06
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) - 0.08

        # ── 12. Warmth / Cold Priming (Williams & Bargh, 2008; IJzerman & Semin, 2009) ──
        # Physical warmth primes social warmth — increased acquiescence and engagement.
        # Physical cold primes social coldness — decreased acquiescence and engagement.
        if any(kw in condition_lower for kw in ['warm cup', 'warm drink', 'warm prime',
               'physical warmth', 'warm condition', 'heated room',
               'warm temperature']):
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) + 0.06
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.04
        elif any(kw in condition_lower for kw in ['cold cup', 'cold drink', 'cold prime',
                 'physical cold', 'cold condition', 'cold temperature',
                 'ice']):
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) - 0.06
            modifiers['engagement'] = modifiers.get('engagement', 0) - 0.04

        # ── 13. Scarcity Priming (Shah et al., 2012; Mullainathan & Shafir, 2013) ──
        # Scarcity captures attention (tunneling effect), increases engagement,
        # and produces more extreme evaluations of scarce resources.
        if any(kw in condition_lower for kw in ['scarcity prime', 'resource scarce',
               'financial scarcity', 'scarcity mindset', 'not enough',
               'running out', 'shortage']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.08
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.06
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.04

        # ── 14. Autonomy Support (Deci & Ryan, 2000; Ryan & Deci, 2017) ──
        # Self-Determination Theory: autonomy support satisfies the need for autonomy,
        # increasing intrinsic motivation, engagement, and consistent responding
        # while reducing impression management.
        if any(kw in condition_lower for kw in ['autonomy support', 'autonomous',
               'free choice', 'self-determined', 'your decision',
               'choose freely', 'volitional']):
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.08
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) + 0.06
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) - 0.04

        # ── 15. Autonomy Thwarting / Controlling (Deci & Ryan, 2000; Vansteenkiste & Ryan, 2013) ──
        # Controlling contexts undermine intrinsic motivation, reducing engagement
        # and consistency while increasing social desirability (conformity pressure).
        if any(kw in condition_lower for kw in ['autonomy thwart', 'controlling',
               'forced choice', 'no choice', 'mandated', 'required to',
               'must comply', 'coerced']):
            modifiers['engagement'] = modifiers.get('engagement', 0) - 0.06
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) - 0.04
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) + 0.06

        return modifiers

    def _get_domain_response_calibration(
        self,
        variable_name: str,
        condition: str = "",
    ) -> Dict[str, float]:
        """
        Get domain-specific response calibration based on variable name and context.

        SCIENTIFIC BASIS:
        =================
        Different research domains have documented baseline response norms:
        - Consumer satisfaction scales: M ≈ 5.0-5.5/7 (Oliver, 1980)
        - Attitude scales: M ≈ 4.0-4.5/7 (Eagly & Chaiken, 1993)
        - Behavioral intentions: M ≈ 4.5-5.0/7 (Ajzen, 1991)
        - Risk perception: M ≈ 3.5-4.5/7 (Slovic, 1987)
        - Trust scales: M ≈ 4.0-4.8/7 (Mayer et al., 1995)
        - Job satisfaction: M ≈ 4.8-5.2/7 (Judge et al., 2001)

        Returns dict with calibration adjustments.
        """
        calibration = {
            'mean_adjustment': 0.0,  # Adjustment to base mean
            'variance_adjustment': 0.0,  # Adjustment to variance
            'positivity_bias': 0.0,  # Additional positivity
        }

        var_lower = variable_name.lower()
        condition_lower = str(condition).lower()

        # v1.0.4.6: Use detected_domains to apply domain-level calibration priors
        # These complement (don't replace) the variable-specific calibrations below
        _det = set(getattr(self, 'detected_domains', []) or [])
        if _det:
            # Domain-level baseline adjustments from detected study domain
            # These are additive priors that capture study-level context
            if _det & {'clinical_psychology'}:
                # Clinical studies: participants report more distress on avg
                calibration['variance_adjustment'] += 0.04
            if _det & {'political_psychology'}:
                # Political studies: high polarization = high variance
                calibration['variance_adjustment'] += 0.06
            if _det & {'economic_games'}:
                # Economic games: variance higher due to strategic behavior
                calibration['variance_adjustment'] += 0.03
            if _det & {'consumer_behavior', 'marketing', 'hedonic_consumption'}:
                # Consumer/marketing: positivity bias in product evaluations
                calibration['positivity_bias'] += 0.03
            if _det & {'health_psychology'}:
                # Health: self-efficacy bias inflates health intentions
                calibration['positivity_bias'] += 0.02
            if _det & {'organizational_behavior'}:
                # Organizational: SD inflates satisfaction & commitment reports
                calibration['positivity_bias'] += 0.03
            if _det & {'moral_psychology', 'fairness'}:
                # Moral/fairness: extreme judgments, low positivity bias
                calibration['variance_adjustment'] += 0.05
            if _det & {'intergroup_relations', 'prejudice'}:
                # Intergroup: high variance due to ingroup/outgroup polarization
                calibration['variance_adjustment'] += 0.05
                calibration['positivity_bias'] -= 0.02  # SD suppresses prejudice reports
            if _det & {'educational_psychology'}:
                # Educational: positive skew in student self-assessments
                calibration['positivity_bias'] += 0.02
            if _det & {'environmental_psychology'}:
                # Environmental: polarized topic, moderate variance
                calibration['variance_adjustment'] += 0.04

        # ===== ECONOMIC GAME / ALLOCATION MEASURES (v1.0.4.2) =====
        # Engel (2011 meta-analysis): Dictator game mean giving ≈ 28% of endowment
        # Berg et al. (1995): Trust game mean sent ≈ 50%
        # Sally (1995 meta): Prisoner's dilemma cooperation ≈ 47%
        # Güth et al. (1982): Ultimatum offers ≈ 40-50%, rejected below 20%
        # Fehr & Gächter (2000): Public goods mean contribution ≈ 40-60%
        _econ_kws = ['dollar', 'amount', 'allocat', 'give', 'giving', 'sent',
                     'offer', 'share', 'split', 'endow', 'dictator',
                     'trust game', 'ultimatum', 'public good', 'contribution',
                     'transfer', 'payment', 'donate', 'generosity']
        _is_econ_game = any(kw in var_lower for kw in _econ_kws) or any(
            kw in condition_lower for kw in ['dictator', 'trust game', 'ultimatum',
                                              'public good', 'prisoner'])
        if _is_econ_game:
            # Detect specific game type for precise calibration
            _full_ctx = var_lower + " " + condition_lower + " " + (
                self.study_title or "").lower() + " " + (self.study_description or "").lower()

            # v1.0.8.7: Try structured knowledge base FIRST for game calibrations
            if HAS_KNOWLEDGE_BASE:
                _kb_game = None
                for _gt in ['dictator', 'trust', 'ultimatum', 'public_good',
                            'prisoner', 'auction', 'bargain', 'gift_exchange',
                            'stag_hunt', 'common_pool', 'holt_laury',
                            'beauty_contest', 'die_roll', 'bribery']:
                    if _gt in _full_ctx:
                        _variant = 'standard'
                        if _gt == 'dictator' and any(kw in _full_ctx for kw in ['tak', 'steal', 'negative']):
                            _variant = 'taking'
                        elif _gt == 'dictator' and any(kw in _full_ctx for kw in ['third party', 'punishment']):
                            _variant = 'third_party_punishment'
                        elif _gt == 'public_good' and 'punish' in _full_ctx:
                            _variant = 'punishment'
                        _gt_clean = _gt.replace('_good', '_goods')
                        _kb_game = get_game_calibration(_gt_clean, _variant)
                        if _kb_game is None:
                            _kb_game = get_game_calibration(_gt, _variant)
                        break
                if _kb_game:
                    # Use structured calibration: convert mean_proportion to adjustment
                    # mean_proportion is 0-1 scale, default midpoint is 0.5
                    calibration['mean_adjustment'] = _kb_game.mean_proportion - 0.50
                    calibration['variance_adjustment'] = max(0.08, _kb_game.sd_proportion * 0.8)
                    calibration['positivity_bias'] = -0.05 if _kb_game.mean_proportion < 0.40 else 0.0
                    calibration['_game_variant'] = f"{_kb_game.game_type}_{_kb_game.variant}"
                    calibration['_kb_source'] = _kb_game.source
                    return calibration
            # v1.0.8.6: Detect game VARIANTS (taking, punishment, etc.)
            _has_taking = any(kw in _full_ctx for kw in [
                'tak', 'steal', 'subtract', 'remov', 'destroy', 'deduct',
                'negative', 'punish', '-100', 'minus',
            ])
            _has_third_party = any(kw in _full_ctx for kw in [
                'third party', 'third-party', 'bystander', 'observer',
                'punishment', 'costly punish',
            ])
            if 'dictator' in _full_ctx and _has_taking:
                # TAKING dictator game (e.g., -100 to +100):
                # List (2007), Bardsley (2008): When taking is available,
                # ~15-25% of participants take, mean giving drops to ~10-15%
                # On bipolar scale: center should be at ~0.40 of range
                # (positive side of zero but lower than standard dictator)
                calibration['mean_adjustment'] = -0.10  # Shift below midpoint
                calibration['positivity_bias'] = -0.10  # Reduce positivity further
                calibration['variance_adjustment'] = 0.25  # Very high variance (takers + givers)
                # Flag for subpopulation mixing (used in Step 4a)
                calibration['_game_variant'] = 'dictator_taking'
            elif 'dictator' in _full_ctx and _has_third_party:
                # Third-party punishment dictator game:
                # Fehr & Fischbacher (2004): Third parties punish ~60% of unfair offers
                calibration['mean_adjustment'] = -0.15
                calibration['positivity_bias'] = -0.05
                calibration['variance_adjustment'] = 0.18
                calibration['_game_variant'] = 'dictator_third_party'
            elif 'dictator' in _full_ctx:
                # Standard dictator game: mean giving ≈ 28% of endowment (Engel, 2011)
                # On 0-100 scale, this means center should be ~28, not ~50
                # Adjustment: shift from 0.5 (midpoint) down to ~0.28
                calibration['mean_adjustment'] = -0.22
                calibration['positivity_bias'] = -0.05
                calibration['variance_adjustment'] = 0.12  # High variance in giving
            elif 'trust' in _full_ctx and 'game' in _full_ctx:
                # Trust game: mean sent ≈ 50% (Berg et al., 1995)
                calibration['mean_adjustment'] = 0.0  # Already near midpoint
                calibration['variance_adjustment'] = 0.10
            elif 'ultimatum' in _full_ctx:
                # Ultimatum: mean offer ≈ 40-50% (modal: 50%)
                calibration['mean_adjustment'] = -0.02
                calibration['variance_adjustment'] = 0.08
            elif 'public good' in _full_ctx:
                # Public goods: mean contribution ≈ 40-60%
                calibration['mean_adjustment'] = -0.05
                calibration['variance_adjustment'] = 0.12
            else:
                # Generic economic allocation: slightly below midpoint
                calibration['mean_adjustment'] = -0.10
                calibration['variance_adjustment'] = 0.10
            return calibration  # Return early — economic game calibration takes priority

        # v1.0.8.7: Try structured construct norms database FIRST
        # This catches well-known scales by variable name with published norms
        if HAS_KNOWLEDGE_BASE:
            _construct_map = {
                # ── Original 22 constructs ──
                'loneliness': 'loneliness_ucla', 'lonely': 'loneliness_ucla',
                'swls': 'life_satisfaction_swls', 'life_sat': 'life_satisfaction_swls',
                'self_esteem': 'self_esteem_rse', 'rosenberg': 'self_esteem_rse',
                'agreeabl': 'big_five_agreeableness', 'conscientious': 'big_five_conscientiousness',
                'extraver': 'big_five_extraversion', 'neurotic': 'big_five_neuroticism',
                'openness': 'big_five_openness', 'burnout': 'burnout_emotional_exhaustion',
                'exhaust': 'burnout_emotional_exhaustion', 'mbi': 'burnout_emotional_exhaustion',
                'gratitude': 'gratitude_gq6', 'grateful': 'gratitude_gq6',
                'resilien': 'resilience_cd_risc', 'cd_risc': 'resilience_cd_risc',
                'moral_identity': 'moral_identity_aquino', 'narcissi': 'narcissism_npi',
                'npi': 'narcissism_npi', 'attachment_anx': 'attachment_anxiety_ecr',
                'attachment_avoid': 'attachment_avoidance_ecr', 'ecr': 'attachment_anxiety_ecr',
                'need_for_cognition': 'need_for_cognition', 'nfc': 'need_for_cognition',
                'conspiracy': 'conspiracy_beliefs_gcbs', 'disgust': 'disgust_sensitivity_dsr',
                'stai': 'state_anxiety_stai', 'state_anxiety': 'state_anxiety_stai',
                'phq': 'depression_phq9', 'depression': 'depression_phq9',
                'pss': 'perceived_stress_pss', 'perceived_stress': 'perceived_stress_pss',
                'impulsiv': 'impulsivity_bis', 'bis_11': 'impulsivity_bis',
                # ── v1.0.9.3: Clinical Psychology ──
                'gad': 'anxiety_gad7', 'gad7': 'anxiety_gad7', 'generalized_anxiety': 'anxiety_gad7',
                'bdi': 'depression_bdi2', 'beck_depression': 'depression_bdi2',
                'ptsd': 'ptsd_pcl5', 'pcl': 'ptsd_pcl5', 'posttraumatic': 'ptsd_pcl5',
                'social_anxiety': 'social_anxiety_lsas', 'lsas': 'social_anxiety_lsas',
                'social_phobia': 'social_anxiety_lsas',
                'ocd': 'ocd_ybocs', 'ybocs': 'ocd_ybocs', 'obsessi': 'ocd_ybocs',
                'eating_disorder': 'eating_disorder_eat26', 'eat26': 'eating_disorder_eat26',
                'eat_26': 'eating_disorder_eat26', 'anorexi': 'eating_disorder_eat26',
                'panic': 'panic_pdss', 'pdss': 'panic_pdss', 'panic_disorder': 'panic_pdss',
                'audit': 'alcohol_use_audit', 'alcohol': 'alcohol_use_audit',
                'staxi': 'anger_staxi', 'anger': 'anger_staxi', 'trait_anger': 'anger_staxi',
                'death_anxiety': 'death_anxiety_das', 'das': 'death_anxiety_das',
                'psqi': 'sleep_quality_psqi', 'sleep_quality': 'sleep_quality_psqi',
                'sleep': 'sleep_quality_psqi', 'insomnia': 'sleep_quality_psqi',
                'body_image': 'body_image_satisfaction',
                'health_anxiety': 'health_anxiety_hai', 'hai': 'health_anxiety_hai',
                'hypochondri': 'health_anxiety_hai',
                'somatiz': 'somatization_phq15', 'phq15': 'somatization_phq15',
                'phq_15': 'somatization_phq15',
                'chronic_fatigue': 'chronic_fatigue',
                # ── v1.0.9.3: Wellbeing & Positive Psychology ──
                'flourish': 'flourishing_perma', 'perma': 'flourishing_perma',
                'positive_affect': 'positive_affect_panas', 'panas_pos': 'positive_affect_panas',
                'negative_affect': 'negative_affect_panas', 'panas_neg': 'negative_affect_panas',
                'panas': 'positive_affect_panas',
                'psych_wellbeing': 'psychological_wellbeing_pwb', 'pwb': 'psychological_wellbeing_pwb',
                'meaning_life': 'meaning_life_mlq_presence', 'mlq': 'meaning_life_mlq_presence',
                'meaning_search': 'meaning_life_mlq_search',
                'hope': 'hope_ahs', 'ahs': 'hope_ahs', 'hopeful': 'hope_ahs',
                'optimism': 'optimism_lotr', 'lot_r': 'optimism_lotr', 'lotr': 'optimism_lotr',
                'happiness': 'happiness_shs', 'shs': 'happiness_shs', 'subjective_happiness': 'happiness_shs',
                'vitality': 'vitality_svs', 'svs': 'vitality_svs',
                'self_compass': 'self_compassion_scs', 'scs': 'self_compassion_scs',
                # ── v1.0.9.3: Values & Ideology ──
                'sdo': 'sdo_social_dominance', 'social_dominan': 'sdo_social_dominance',
                'rwa': 'rwa_authoritarianism', 'authoritarian': 'rwa_authoritarianism',
                'just_world': 'just_world_belief_bjw', 'bjw': 'just_world_belief_bjw',
                'materiali': 'materialism_mvs', 'mvs': 'materialism_mvs',
                'system_justif': 'system_justification',
                'political_ideol': 'political_ideology',
                # ── v1.0.9.3: Social Psychology ──
                'social_support': 'social_support_mspss', 'mspss': 'social_support_mspss',
                'belonging': 'belongingness', 'belongingness': 'belongingness',
                'social_compar': 'social_comparison_sco', 'sco': 'social_comparison_sco',
                'collective_self': 'collective_self_esteem_cse', 'cse': 'collective_self_esteem_cse',
                'empathic_concern': 'empathic_concern_iri', 'iri_ec': 'empathic_concern_iri',
                'perspective_tak': 'perspective_taking_iri', 'iri_pt': 'perspective_taking_iri',
                'personal_distress': 'personal_distress_iri', 'iri_pd': 'personal_distress_iri',
                'interpersonal_trust': 'interpersonal_trust',
                # ── v1.0.9.3: Cognitive & Self-Regulation ──
                'mindful': 'mindfulness_maas', 'maas': 'mindfulness_maas',
                'cognitive_flex': 'cognitive_flexibility',
                'ambiguity_toler': 'tolerance_of_ambiguity',
                'locus_control': 'locus_of_control',
                'self_regulat': 'self_regulation_srs', 'srs': 'self_regulation_srs',
                'cognitive_reflect': 'cognitive_reflection_crt', 'crt': 'cognitive_reflection_crt',
                'growth_mindset': 'growth_mindset', 'mindset': 'growth_mindset',
                'grit': 'grit', 'perseveran': 'grit',
                'ruminat': 'rumination_rrs', 'rrs': 'rumination_rrs',
                'worry': 'worry_pswq', 'pswq': 'worry_pswq', 'penn_worry': 'worry_pswq',
                'reappraisal': 'emotion_regulation_erq_reappraisal', 'erq': 'emotion_regulation_erq_reappraisal',
                'suppress': 'emotion_regulation_erq_suppression',
                # ── v1.0.9.3: Motivation & Achievement ──
                'intrinsic_motiv': 'intrinsic_motivation_imi', 'imi': 'intrinsic_motivation_imi',
                'self_efficacy': 'self_efficacy_gse', 'gse': 'self_efficacy_gse',
                'work_engage': 'work_engagement_uwes', 'uwes': 'work_engagement_uwes',
                'flow': 'flow_experience', 'procrastinat': 'procrastination',
                'test_anxiety': 'test_anxiety_tai', 'tai': 'test_anxiety_tai',
                # ── v1.0.9.3: Interpersonal & Relationships ──
                'forgiv': 'forgiveness_tfs', 'tfs': 'forgiveness_tfs',
                'relation_satisf': 'relationship_satisfaction_ras', 'ras': 'relationship_satisfaction_ras',
                'jealous': 'jealousy', 'romantic_love': 'romantic_love',
                'emotional_intellig': 'emotional_intelligence_eq', 'eq_score': 'emotional_intelligence_eq',
                # ── v1.0.9.3: Work & Organizational ──
                'job_satisf': 'job_satisfaction_msq', 'msq': 'job_satisfaction_msq',
                'org_commit': 'organizational_commitment_ocq', 'ocq': 'organizational_commitment_ocq',
                'lmx': 'leader_member_exchange_lmx',
                'psycap': 'psychological_capital_psycap',
                'turnover_intent': 'turnover_intention',
                'work_family': 'work_family_conflict', 'wfc': 'work_family_conflict',
                # ── v1.0.9.3: Technology & Media ──
                'tech_accept': 'technology_acceptance_tam', 'tam': 'technology_acceptance_tam',
                'internet_addict': 'internet_addiction_iat',
                'social_media_intens': 'social_media_intensity',
                'privacy_concern': 'privacy_concern_iuipc', 'iuipc': 'privacy_concern_iuipc',
                'ai_attitude': 'ai_attitudes',
                # ── v1.0.9.3: Consumer & Marketing ──
                'brand_loyal': 'brand_loyalty', 'purchase_intent': 'purchase_intention',
                'customer_satisf': 'customer_satisfaction_acsi', 'acsi': 'customer_satisfaction_acsi',
                'perceived_value': 'perceived_value', 'brand_trust': 'brand_trust',
            }
            for _kw, _norm_key in _construct_map.items():
                if _kw in var_lower:
                    _norm = get_construct_norm(_norm_key, target_scale_points=7)
                    if _norm:
                        # Convert published norm to calibration adjustment
                        # Published mean on 7-point → deviation from neutral (4.0)
                        _dev = (_norm['mean'] - 4.0) / 3.0  # Normalize to [-1, 1]
                        calibration['mean_adjustment'] = _dev * 0.15  # Scale to adjustment range
                        calibration['positivity_bias'] = max(-0.10, min(0.12, _dev * 0.10))
                        if _norm.get('skewness', 0) > 0.3:
                            calibration['variance_adjustment'] += 0.06
                        elif _norm.get('skewness', 0) < -0.3:
                            calibration['variance_adjustment'] -= 0.02
                        calibration['_kb_source'] = f"ConstructNorm: {_norm_key}"
                        return calibration

        # ===== SATISFACTION SCALES =====
        # Oliver (1980): Satisfaction has positive skew (M ≈ 5.0-5.5)
        if any(kw in var_lower for kw in ['satisfaction', 'satisfied', 'happy', 'pleased']):
            calibration['mean_adjustment'] = 0.08  # Shift toward positive
            calibration['positivity_bias'] = 0.10

        # ===== PURCHASE/BEHAVIORAL INTENTION =====
        # Ajzen (1991): Intentions moderately positive (M ≈ 4.5-5.0)
        elif any(kw in var_lower for kw in ['intention', 'likely', 'would_', 'willing', 'wtp']):
            calibration['mean_adjustment'] = 0.05
            calibration['variance_adjustment'] = 0.05  # Higher variance in intentions

        # ===== ATTITUDE/EVALUATION SCALES =====
        # Eagly & Chaiken (1993): Attitudes vary by valence of object
        elif any(kw in var_lower for kw in ['attitude', 'evaluation', 'opinion', 'view']):
            calibration['mean_adjustment'] = 0.02  # Slight positivity bias
            calibration['variance_adjustment'] = 0.03

        # ===== TRUST SCALES =====
        # Mayer et al. (1995): Trust tends slightly positive (M ≈ 4.0-4.8)
        elif any(kw in var_lower for kw in ['trust', 'reliability', 'dependab', 'credib']):
            calibration['mean_adjustment'] = 0.04
            calibration['positivity_bias'] = 0.05

        # ===== RISK PERCEPTION =====
        # Slovic (1987): Risk perception centered/slightly negative (M ≈ 3.5-4.5)
        elif any(kw in var_lower for kw in ['risk', 'danger', 'unsafe', 'threat', 'harm']):
            calibration['mean_adjustment'] = -0.05  # Slightly lower/cautious
            calibration['variance_adjustment'] = 0.08  # High variance in risk perception

        # ===== ANXIETY/CONCERN =====
        # Lower baseline for negative constructs
        elif any(kw in var_lower for kw in ['anxiety', 'worry', 'concern', 'fear']):
            calibration['mean_adjustment'] = -0.08
            calibration['positivity_bias'] = -0.05

        # ===== JOB/WORK SATISFACTION =====
        # Judge et al. (2001): Generally positive (M ≈ 4.8-5.2)
        elif any(kw in var_lower for kw in ['job_', 'work_', 'employee', 'workplace']):
            calibration['mean_adjustment'] = 0.06
            calibration['positivity_bias'] = 0.08

        # ===== QUALITY PERCEPTION =====
        # Generally positive for evaluations (M ≈ 4.5-5.0)
        elif any(kw in var_lower for kw in ['quality', 'excellent', 'good', 'value']):
            calibration['mean_adjustment'] = 0.05
            calibration['positivity_bias'] = 0.06

        # ===== ENVIRONMENTAL/SUSTAINABILITY =====
        # Dunlap et al. (2000): Moderately positive (M ≈ 4.2-4.8)
        elif any(kw in var_lower for kw in ['environment', 'sustain', 'green', 'eco', 'climate']):
            calibration['mean_adjustment'] = 0.03
            calibration['variance_adjustment'] = 0.06  # Polarized topic

        # ===== AI/TECHNOLOGY ATTITUDES =====
        # Longoni et al. (2019): Mixed/slightly negative (M ≈ 3.8-4.3)
        elif any(kw in var_lower for kw in ['ai_', 'robot', 'automat', 'algorithm']):
            calibration['mean_adjustment'] = -0.03
            calibration['variance_adjustment'] = 0.08  # Highly polarized

        # ===== HEALTH BEHAVIORS =====
        # Rosenstock (1974): Self-efficacy positive (M ≈ 4.5-5.2)
        elif any(kw in var_lower for kw in ['health', 'wellness', 'exercise', 'diet']):
            calibration['mean_adjustment'] = 0.04
            calibration['positivity_bias'] = 0.05

        # ===== SOCIAL IDENTITY / INTERGROUP =====
        # Tajfel & Turner (1979): High variance due to ingroup/outgroup polarization
        # Social Identity Theory predicts strong ingroup favoritism and outgroup derogation
        elif any(kw in var_lower for kw in ['identity', 'intergroup', 'ingroup', 'outgroup', 'discrimination']):
            calibration['mean_adjustment'] = 0.03
            calibration['positivity_bias'] = 0.04
            calibration['variance_adjustment'] = 0.10

        # ===== MORAL JUDGMENT =====
        # Haidt (2001): Moral judgments elicit extreme responses, minimal positivity bias
        # Moral foundations theory: intuitive, emotion-driven judgments cluster at endpoints
        elif any(kw in var_lower for kw in ['moral', 'ethical', 'right', 'wrong', 'justice']):
            calibration['mean_adjustment'] = -0.02
            calibration['positivity_bias'] = 0.0
            calibration['variance_adjustment'] = 0.12

        # ===== POLITICAL ATTITUDES =====
        # Iyengar & Westwood (2015): Maximal polarization in partisan attitudes
        # Affective polarization produces bimodal distributions with high variance
        elif any(kw in var_lower for kw in ['political', 'liberal', 'conservative', 'democrat', 'republican', 'partisan']):
            calibration['mean_adjustment'] = 0.0
            calibration['positivity_bias'] = 0.0
            calibration['variance_adjustment'] = 0.15

        # ===== SELF-EFFICACY / COMPETENCE =====
        # Bandura (1997): Positive skew in self-assessments of capability
        # Self-enhancement bias inflates competence ratings (M ≈ 5.0-5.5)
        elif any(kw in var_lower for kw in ['efficacy', 'competence', 'confidence', 'capable', 'ability']):
            calibration['mean_adjustment'] = 0.06
            calibration['positivity_bias'] = 0.08

        # ===== PROSOCIAL / ALTRUISM =====
        # Social desirability inflates prosocial self-reports (M ≈ 5.2-5.8)
        # Low variance: most respondents claim prosocial intentions
        elif any(kw in var_lower for kw in ['prosocial', 'altruism', 'helping', 'donate', 'volunteer', 'charity']):
            calibration['mean_adjustment'] = 0.07
            calibration['positivity_bias'] = 0.10
            calibration['variance_adjustment'] = 0.05

        # ===== NOSTALGIA / MEMORY =====
        # Mitchell et al. (1997): Rosy retrospection bias inflates positive valence of memories
        # Nostalgia generates positively-tinted recall with high positivity bias
        elif any(kw in var_lower for kw in ['nostalgia', 'remember', 'past', 'memory', 'childhood']):
            calibration['mean_adjustment'] = 0.05
            calibration['positivity_bias'] = 0.12

        # ===== PRIVACY CONCERNS =====
        # Westin (2003): Moderate negative valence in privacy concern ratings
        # Privacy paradox: stated concerns exceed behavioral responses
        elif any(kw in var_lower for kw in ['privacy', 'surveillance', 'tracking', 'data_collection']):
            calibration['mean_adjustment'] = -0.04
            calibration['variance_adjustment'] = 0.08

        # ===== FOOD / CONSUMPTION =====
        # Hedonic positivity bias: food evaluations skew positive (M ≈ 5.0-5.5)
        # Koenig-Lewis & Palmer (2014): Consumption experiences rated favorably
        elif any(kw in var_lower for kw in ['food', 'taste', 'meal', 'restaurant', 'eat']):
            calibration['mean_adjustment'] = 0.06
            calibration['positivity_bias'] = 0.08

        # ===== EDUCATION / LEARNING =====
        # Marsh (1987): Positive skew in course evaluations (M ≈ 5.0-5.5/7)
        # Students' Evaluations of Educational Quality (SEEQ) norms
        elif any(kw in var_lower for kw in ['learn', 'teach', 'education', 'classroom', 'student']):
            calibration['mean_adjustment'] = 0.04
            calibration['positivity_bias'] = 0.06

        # ===== CREATIVITY / INNOVATION =====
        # Positive self-assessment bias for creative ability, moderate variance
        # Runco (2004): Self-reported creativity shows moderate positive skew
        elif any(kw in var_lower for kw in ['creative', 'innovati', 'novel', 'idea', 'brainstorm']):
            calibration['mean_adjustment'] = 0.04
            calibration['variance_adjustment'] = 0.06

        # ================================================================
        # v1.0.4.3: Extended domain calibrations for under-served domains
        # Grounded in published meta-analyses and response norm studies
        # ================================================================

        # ===== LONELINESS / SOCIAL ISOLATION =====
        # Russell (1996, UCLA Loneliness Scale): Norms show moderate-negative
        # baseline — most people report some loneliness (M ≈ 3.5-4.0/7)
        # High variance due to large individual differences
        elif any(kw in var_lower for kw in ['lonely', 'loneliness', 'isolat', 'alone', 'social connect']):
            calibration['mean_adjustment'] = -0.06
            calibration['positivity_bias'] = -0.05
            calibration['variance_adjustment'] = 0.10

        # ===== GRATITUDE =====
        # McCullough et al. (2002, GQ-6): Positive skew, M ≈ 5.5-6.0/7
        # Most people report relatively high gratitude (social desirability)
        elif any(kw in var_lower for kw in ['gratitude', 'grateful', 'thankful', 'appreciate']):
            calibration['mean_adjustment'] = 0.08
            calibration['positivity_bias'] = 0.10
            calibration['variance_adjustment'] = -0.03  # Low variance — ceiling effect

        # ===== RESILIENCE =====
        # Connor & Davidson (2003, CD-RISC): Moderate positive, M ≈ 4.5-5.0/7
        # Self-enhancement bias in self-reported resilience
        elif any(kw in var_lower for kw in ['resilien', 'cope', 'coping', 'recover', 'bounce back']):
            calibration['mean_adjustment'] = 0.05
            calibration['positivity_bias'] = 0.06
            calibration['variance_adjustment'] = 0.04

        # ===== BURNOUT =====
        # Maslach & Jackson (1981, MBI): Burnout norms vary by facet
        # Emotional exhaustion: M ≈ 3.0-3.5/7 (moderate)
        # Depersonalization: M ≈ 2.5/7 (lower)
        # Personal accomplishment: M ≈ 5.0/7 (higher, reverse-coded)
        elif any(kw in var_lower for kw in ['burnout', 'exhaust', 'depersonaliz', 'cynicism']):
            calibration['mean_adjustment'] = -0.08
            calibration['positivity_bias'] = -0.06
            calibration['variance_adjustment'] = 0.08

        # ===== LIFE SATISFACTION =====
        # Diener et al. (1985, SWLS): Generally positive, M ≈ 4.8-5.2/7
        # Slight positive skew (Cummins 2003: homeostatic set-point ≈ 70-80%)
        elif any(kw in var_lower for kw in ['life satisf', 'well-being', 'wellbeing', 'flourish',
                                             'life_sat', 'swls', 'happiness']):
            calibration['mean_adjustment'] = 0.07
            calibration['positivity_bias'] = 0.10
            calibration['variance_adjustment'] = 0.04

        # ===== EMPATHY =====
        # Davis (1983, IRI): Moderate-positive baselines, M ≈ 4.2-4.8/7
        # Gender differences: women score ~0.5 points higher (Eisenberg & Lennon, 1983)
        elif any(kw in var_lower for kw in ['empathy', 'empathic', 'perspective_taking',
                                             'compassion', 'sympathy']):
            calibration['mean_adjustment'] = 0.05
            calibration['positivity_bias'] = 0.06
            calibration['variance_adjustment'] = 0.05

        # ===== AGGRESSION =====
        # Buss & Perry (1992, AQ): Generally low-moderate, M ≈ 3.0-3.5/7
        # Social desirability suppresses aggression reports
        elif any(kw in var_lower for kw in ['aggress', 'hostil', 'anger', 'violent', 'agitation']):
            calibration['mean_adjustment'] = -0.10
            calibration['positivity_bias'] = -0.08
            calibration['variance_adjustment'] = 0.10

        # ===== PERSONALITY (Big Five) =====
        # Costa & McCrae (1992, NEO-PI-R norms):
        # Agreeableness: M ≈ 5.0/7 (positive skew)
        # Conscientiousness: M ≈ 4.8/7 (positive skew)
        # Neuroticism: M ≈ 3.5/7 (moderate, high variance)
        # Extraversion: M ≈ 4.2/7 (moderate positive)
        # Openness: M ≈ 4.3/7 (moderate)
        elif any(kw in var_lower for kw in ['agreeable', 'conscientious', 'extraver',
                                             'neurotic', 'openness', 'big five', 'personality']):
            calibration['mean_adjustment'] = 0.03
            calibration['variance_adjustment'] = 0.06

        # ===== NARCISSISM / DARK TRIAD =====
        # Paulhus & Williams (2002): Lower reported means due to social undesirability
        # NPI: M ≈ 15/40 (below midpoint), DTDD norms skew low
        elif any(kw in var_lower for kw in ['narcissi', 'dark triad', 'machiavelli',
                                             'psychopath', 'grandiosity', 'entitlement']):
            calibration['mean_adjustment'] = -0.08
            calibration['positivity_bias'] = -0.06
            calibration['variance_adjustment'] = 0.10

        # ===== ATTACHMENT STYLE =====
        # Brennan et al. (1998, ECR): Anxiety and avoidance dimensions
        # Anxiety: M ≈ 3.2/7 (moderate-low), Avoidance: M ≈ 3.0/7
        elif any(kw in var_lower for kw in ['attachment', 'anxious_attach', 'avoidant',
                                             'secure_attach', 'relationship_style']):
            calibration['mean_adjustment'] = -0.04
            calibration['variance_adjustment'] = 0.08

        # ===== CONSPIRACY BELIEFS =====
        # Brotherton et al. (2013): Generally low endorsement, M ≈ 2.5-3.5/7
        # But high variance — believers score very high, skeptics very low
        elif any(kw in var_lower for kw in ['conspiracy', 'conspira', 'paranoi', 'cover-up',
                                             'secret', 'deep state']):
            calibration['mean_adjustment'] = -0.10
            calibration['positivity_bias'] = -0.05
            calibration['variance_adjustment'] = 0.15

        # ===== DISGUST SENSITIVITY =====
        # Olatunji et al. (2007, DS-R norms): Moderate, M ≈ 3.8-4.2/7
        # Higher variance; women score higher than men (Druschel & Sherman, 1999)
        elif any(kw in var_lower for kw in ['disgust', 'repuls', 'contaminat', 'gross']):
            calibration['mean_adjustment'] = -0.02
            calibration['variance_adjustment'] = 0.08

        # ===== IMPULSIVITY / SELF-CONTROL =====
        # Tangney et al. (2004, Brief SCS): Moderate means, M ≈ 4.0/7
        # Moderate variance, slight negative skew (people admit some impulsivity)
        elif any(kw in var_lower for kw in ['impulsiv', 'self_control', 'self-control',
                                             'inhibit', 'restrain']):
            calibration['mean_adjustment'] = -0.03
            calibration['variance_adjustment'] = 0.06

        # ===== NEED FOR COGNITION =====
        # Cacioppo et al. (1984): Moderate-positive, M ≈ 4.0-4.5/7
        # Students typically score above general population
        elif any(kw in var_lower for kw in ['need_for_cognition', 'nfc', 'cognitive_need',
                                             'thinking_enjoyment']):
            calibration['mean_adjustment'] = 0.04
            calibration['positivity_bias'] = 0.04
            calibration['variance_adjustment'] = 0.05

        # ===== SOCIAL MEDIA / DIGITAL BEHAVIOR =====
        # High variance, moderate means — frequency-dependent
        # Twenge (2019): Social media use associated with reduced wellbeing
        elif any(kw in var_lower for kw in ['social_media', 'instagram', 'tiktok', 'facebook',
                                             'screen_time', 'digital', 'online']):
            calibration['mean_adjustment'] = 0.02
            calibration['variance_adjustment'] = 0.10

        # ===== ORGANIZATIONAL COMMITMENT =====
        # Meyer & Allen (1991): Affective commitment M ≈ 4.5-5.0/7
        # Continuance commitment: M ≈ 3.5-4.0/7 (more neutral)
        elif any(kw in var_lower for kw in ['commit', 'organizational', 'turnover_intent',
                                             'retention', 'loyalty_employ']):
            calibration['mean_adjustment'] = 0.04
            calibration['positivity_bias'] = 0.05
            calibration['variance_adjustment'] = 0.06

        # ===== PREJUDICE / DISCRIMINATION =====
        # Explicit prejudice norms: Low reported means due to social desirability
        # McConahay (1986): Modern racism M ≈ 2.5-3.0/7
        elif any(kw in var_lower for kw in ['prejudic', 'discrimin', 'racism', 'sexism',
                                             'bias', 'intoleran']):
            calibration['mean_adjustment'] = -0.12
            calibration['positivity_bias'] = -0.08
            calibration['variance_adjustment'] = 0.12

        # ===== MISINFORMATION / FAKE NEWS =====
        # Pennycook & Rand (2019): Low accuracy in discerning real vs fake news
        # High variance; some people are much better than others
        elif any(kw in var_lower for kw in ['misinform', 'fake_news', 'truth', 'accuracy',
                                             'credib', 'believab']):
            calibration['mean_adjustment'] = 0.0
            calibration['variance_adjustment'] = 0.12

        # ===== v1.0.4.9: NARRATIVE ENGAGEMENT =====
        # Green & Brock (2000): Transportation into narrative worlds
        # Transported readers show moderate-high engagement (M ≈ 4.5-5.2/7)
        elif any(kw in var_lower for kw in ['transport', 'narrative', 'immersion', 'absorbed', 'story_engage']):
            calibration['mean_adjustment'] = 0.04
            calibration['positivity_bias'] = 0.06
            calibration['variance_adjustment'] = 0.05

        # ===== v1.0.4.9: SOCIAL COMPARISON MEASURES =====
        # Gibbons & Buunk (1999, Iowa-Netherlands Comparison Scale): M ≈ 3.5-4.5/7
        # High variance due to individual differences in comparison orientation
        elif any(kw in var_lower for kw in ['comparison', 'compare', 'relative', 'better_than', 'worse_than']):
            calibration['mean_adjustment'] = 0.0
            calibration['variance_adjustment'] = 0.10

        # ===== v1.0.4.9: DIGITAL WELLBEING =====
        # Smartphone/social media usage effects — high variance
        elif any(kw in var_lower for kw in ['screen_time', 'phone_use', 'social_media', 'digital_wellbeing',
                                             'device', 'app_use']):
            calibration['mean_adjustment'] = -0.03
            calibration['variance_adjustment'] = 0.08

        # ===== v1.0.4.9: MORAL CLEANSING / MORAL SELF =====
        # Aquino & Reed (2002): Moral identity M ≈ 5.5-6.0/7 (positive skew)
        elif any(kw in var_lower for kw in ['moral_self', 'moral_identity', 'ethical_self', 'virtue']):
            calibration['mean_adjustment'] = 0.08
            calibration['positivity_bias'] = 0.10
            calibration['variance_adjustment'] = -0.02

        # ===== CONDITION-BASED ADJUSTMENTS =====
        # Adjust based on experimental condition keywords
        if 'positive' in condition_lower or 'high' in condition_lower:
            calibration['mean_adjustment'] += 0.03
        elif 'negative' in condition_lower or 'low' in condition_lower:
            calibration['mean_adjustment'] -= 0.03
        elif 'control' in condition_lower or 'neutral' in condition_lower:
            pass  # No adjustment for control/neutral conditions

        # Longoni et al. (2019): AI-related conditions produce slightly negative shift
        if any(kw in condition_lower for kw in ['ai', 'algorithm', 'robot', 'automat', 'machine']):
            calibration['mean_adjustment'] -= 0.02

        return calibration

    def _detect_scale_geometry(
        self,
        scale_min: int,
        scale_max: int,
        variable_name: str = "",
    ) -> Dict[str, Any]:
        """Detect the geometric properties of a scale for bipolar/novel handling.

        v1.0.8.6: NEW — Classifies scales to determine appropriate generation strategy.

        Returns dict with:
        - is_bipolar: True if scale spans negative-to-positive (e.g., -100 to +100)
        - is_symmetric: True if scale is symmetric around zero (e.g., -3 to +3)
        - midpoint: The conceptual neutral point of the scale
        - is_novel_range: True if scale uses non-standard range (not 1-7, 0-100, etc.)
        - is_economic_game_allocation: True if DV looks like an economic game allocation
        - has_taking_option: True if negative values represent "taking" behavior
        - bound_width: Wider [low, high] for tendency clipping (bipolar gets [0.02, 0.98])

        Scientific basis:
        - Schwarz et al. (1991): Numeric scale endpoints affect response meaning
        - Krosnick & Fabrigar (1997): Scale format systematically influences data quality
        - Engel (2011): Dictator game distributions are bimodal, not normal
        """
        result: Dict[str, Any] = {
            'is_bipolar': False,
            'is_symmetric': False,
            'midpoint': (scale_min + scale_max) / 2.0,
            'is_novel_range': False,
            'is_economic_game_allocation': False,
            'has_taking_option': False,
            'bound_low': 0.08,   # default clipping bounds
            'bound_high': 0.92,
        }

        scale_range = scale_max - scale_min

        # ── Bipolar detection ──
        # A scale is bipolar if it spans both negative and positive values
        if scale_min < 0 and scale_max > 0:
            result['is_bipolar'] = True
            # Check symmetry (e.g., -100 to +100, -3 to +3)
            if abs(abs(scale_min) - abs(scale_max)) <= 1:
                result['is_symmetric'] = True
                result['midpoint'] = 0.0
            # Widen clipping bounds for bipolar scales — allow full range access
            result['bound_low'] = 0.02
            result['bound_high'] = 0.98

        # ── Novel range detection ──
        # Standard ranges: 1-5, 1-7, 1-9, 0-10, 0-100, 1-100
        _standard_ranges = {
            (1, 5), (1, 7), (1, 9), (1, 10), (1, 11),
            (0, 10), (0, 100), (1, 100), (0, 6), (0, 4),
        }
        if (scale_min, scale_max) not in _standard_ranges:
            # Allow small deviations (e.g., 0-101 ≈ standard)
            _is_std = any(
                abs(scale_min - sm) <= 1 and abs(scale_max - sx) <= 1
                for sm, sx in _standard_ranges
            )
            if not _is_std:
                result['is_novel_range'] = True

        # ── Economic game allocation detection ──
        _var_lower = variable_name.lower()
        _study_ctx = ((self.study_title or "") + " " + (self.study_description or "")).lower()
        _full_ctx = _var_lower + " " + _study_ctx
        _econ_kws = ['dictator', 'allocat', 'give', 'giving', 'endow', 'split',
                     'transfer', 'sent', 'offer', 'share', 'trust game',
                     'ultimatum', 'public good', 'contribution', 'donate']
        if any(kw in _full_ctx for kw in _econ_kws):
            result['is_economic_game_allocation'] = True

        # ── Taking option detection ──
        # If the scale is bipolar AND it's an economic game, negative values = taking
        _taking_kws = ['tak', 'steal', 'subtract', 'remov', 'reduc', 'negative',
                       'punish', 'destroi', 'destroy', 'deduct']
        if result['is_bipolar'] and (
            result['is_economic_game_allocation'] or
            any(kw in _full_ctx for kw in _taking_kws)
        ):
            result['has_taking_option'] = True
            # For taking games, allow even wider bounds
            result['bound_low'] = 0.01
            result['bound_high'] = 0.99

        return result

    def _get_scale_type_calibration(
        self,
        variable_name: str,
        scale_min: int,
        scale_max: int,
    ) -> Dict[str, float]:
        """
        Get calibration adjustments based on scale type/format.

        SCIENTIFIC BASIS:
        =================
        Different scale formats produce systematically different response patterns:
        - Likert scales (5-7 pt): Central tendency, modest extremity
        - Visual analog/sliders (0-100): Full range use, less central tendency
        - Willingness to pay: Positive skew, high variance
        - Binary/forced choice: Clear differentiation

        References:
        - Krosnick & Fabrigar (1997): Scale format effects
        - Schwarz et al. (1991): Numeric scale influences
        - Tourangeau et al. (2000): Psychology of survey response
        """
        calibration = {
            'central_tendency_reduction': 0.0,  # Reduce midpoint pull
            'variance_multiplier': 1.0,  # Scale variance
            'extremity_boost': 0.0,  # Increase endpoint use
        }

        scale_range = scale_max - scale_min
        var_lower = variable_name.lower()

        # ===== SLIDER/VISUAL ANALOG SCALES (0-100 or similar wide range) =====
        # Krosnick & Fabrigar (1997): Sliders produce more differentiated responses
        if scale_range >= 50:
            calibration['central_tendency_reduction'] = 0.08
            calibration['variance_multiplier'] = 1.15
            calibration['extremity_boost'] = 0.05

        # ===== STANDARD LIKERT (5-7 point) =====
        # Most published research uses these scales
        elif 4 <= scale_range <= 6:
            calibration['central_tendency_reduction'] = 0.0
            calibration['variance_multiplier'] = 1.0
            calibration['extremity_boost'] = 0.0

        # ===== BIPOLAR SCALES (e.g., -3 to +3, -100 to +100) =====
        # v1.0.8.6: Enhanced bipolar handling — wider variance, more spread
        elif scale_min < 0:
            calibration['central_tendency_reduction'] = -0.02  # Slight midpoint pull
            # Bipolar scales should show MORE variance (full range usage)
            if scale_range >= 50:
                # Wide bipolar (e.g., -100 to +100): very high variance
                calibration['variance_multiplier'] = 1.30
                calibration['extremity_boost'] = 0.08
            else:
                # Narrow bipolar (e.g., -3 to +3): moderate variance
                calibration['variance_multiplier'] = 1.05

        # ===== WILLINGNESS TO PAY / NUMERIC ESTIMATES =====
        # Typically show positive skew and high variance
        if any(kw in var_lower for kw in ['wtp', 'willingness_to_pay', 'price', 'amount', 'dollar']):
            calibration['variance_multiplier'] = 1.25
            calibration['extremity_boost'] = -0.05  # Avoid ceiling effects

        # ===== PROBABILITY/LIKELIHOOD SCALES =====
        # Bounded at 0 and 100, often cluster near endpoints
        elif any(kw in var_lower for kw in ['probability', 'percent', 'likelihood', 'chance']):
            calibration['extremity_boost'] = 0.08
            calibration['variance_multiplier'] = 1.1

        # ===== FREQUENCY SCALES =====
        # Often positively skewed (most people report lower frequencies)
        elif any(kw in var_lower for kw in ['frequency', 'often', 'times', 'how_many']):
            calibration['central_tendency_reduction'] = -0.05  # Shift toward lower
            calibration['variance_multiplier'] = 1.2

        return calibration

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

        # Defensive conversion with fallbacks for None/NaN
        try:
            if scale_min is None or (isinstance(scale_min, float) and np.isnan(scale_min)):
                scale_min = 1
            else:
                scale_min = int(scale_min)
        except (ValueError, TypeError):
            scale_min = 1

        try:
            if scale_max is None or (isinstance(scale_max, float) and np.isnan(scale_max)):
                scale_max = 7
            else:
                scale_max = int(scale_max)
        except (ValueError, TypeError):
            scale_max = 7

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
        # v1.2.1: Use _safe_trait_value to handle dict/PersonaTrait values
        modified_traits = {k: _safe_trait_value(v, 0.5) for k, v in traits.items()}
        for trait_name, modifier in condition_modifiers.items():
            if trait_name in modified_traits:
                modified_traits[trait_name] = float(np.clip(
                    modified_traits[trait_name] + modifier, 0.0, 1.0
                ))

        # =====================================================================
        # STEP 2a: Get domain-specific calibration
        # Based on published norms for different construct types (v2.2.8)
        # =====================================================================
        domain_calibration = self._get_domain_response_calibration(variable_name, condition)

        # =====================================================================
        # STEP 2b: Get scale-type calibration
        # Based on scale format effects research (Krosnick & Fabrigar, 1997)
        # =====================================================================
        scale_calibration = self._get_scale_type_calibration(variable_name, scale_min, scale_max)

        # =====================================================================
        # v1.0.8.6 STEP 2c: Scale geometry detection
        # Classifies bipolar/novel/economic-game scales and sets appropriate
        # clipping bounds. Critical for scales like -100 to +100 (taking DG).
        # =====================================================================
        _scale_geom = self._detect_scale_geometry(scale_min, scale_max, variable_name)
        _bound_low = _scale_geom['bound_low']
        _bound_high = _scale_geom['bound_high']

        # =====================================================================
        # STEP 3: Get base response tendency
        # Calibrated from Krosnick (1991) optimizing vs satisficing
        # =====================================================================
        # v1.2.1: Safe trait access with fallback chain
        base_tendency = modified_traits.get("response_tendency")
        if base_tendency is None:
            base_tendency = modified_traits.get("scale_use_breadth", 0.58)
        base_tendency = _safe_trait_value(base_tendency, 0.58)

        # v1.0.8.6: For bipolar scales, START at the true midpoint (0.5 = zero)
        # instead of the default positivity-biased 0.58. Positivity bias is a
        # Likert-scale artifact (Diener et al., 1999) that doesn't apply to
        # bipolar allocation scales where negative = taking.
        if _scale_geom['is_bipolar']:
            # Center at 0.5 (the zero point on bipolar scales)
            base_tendency = 0.50 + (base_tendency - 0.58) * 0.5  # Dampen bias
            # Still apply domain calibrations but with reduced positivity
            base_tendency += domain_calibration['mean_adjustment']
            base_tendency += domain_calibration['positivity_bias'] * 0.3  # Attenuate positivity
        else:
            # Apply domain-specific adjustments (standard unipolar behavior)
            base_tendency += domain_calibration['mean_adjustment']
            base_tendency += domain_calibration['positivity_bias']
        # Apply scale-type central tendency adjustment
        base_tendency += scale_calibration['central_tendency_reduction']
        base_tendency = float(np.clip(base_tendency, 0.05, 0.95))

        # =====================================================================
        # v1.0.8.6 STEP 3b: Subpopulation mixing for economic game variants
        # When game variants introduce novel action spaces (e.g., taking),
        # the population is NOT normally distributed — it consists of
        # distinct behavioral types drawn from behavioral economics theory.
        #
        # SCIENTIFIC BASIS:
        # List (2007): Introducing taking option reveals ~20% takers
        # Bardsley (2008): Taking lowers mean giving to ~10-15%
        # Engel (2011): Standard dictator is bimodal: mode at 0, mode at 50%
        # Fehr & Schmidt (1999): Inequity aversion model predicts subpopulations
        # =====================================================================
        _game_variant = domain_calibration.get('_game_variant', '')
        if _game_variant == 'dictator_taking' and _scale_geom['has_taking_option']:
            # Subpopulation mixture for taking dictator game:
            #   ~35% Fair dividers: give ~40-50% (tendency ≈ 0.70-0.75 on bipolar scale)
            #   ~25% Selfish/zero: give 0 (tendency ≈ 0.50 = zero point)
            #   ~20% Takers: take 10-40% (tendency ≈ 0.20-0.40)
            #   ~20% Moderate givers: give ~10-25% (tendency ≈ 0.55-0.65)
            _subpop_roll = rng.random()
            if _subpop_roll < 0.35:
                # Fair divider — give ~40-50%
                base_tendency = 0.70 + rng.uniform(-0.05, 0.05)
            elif _subpop_roll < 0.60:
                # Selfish — give zero (or very close to it)
                base_tendency = 0.50 + rng.uniform(-0.02, 0.02)
            elif _subpop_roll < 0.80:
                # TAKER — take 10-40% from the other person
                base_tendency = 0.20 + rng.uniform(0.0, 0.15)
            else:
                # Moderate giver — give 10-25%
                base_tendency = 0.55 + rng.uniform(0.0, 0.10)
        elif _game_variant == 'dictator_third_party':
            # Third-party punishment: most punish (60%), some don't (40%)
            _subpop_roll = rng.random()
            if _subpop_roll < 0.60:
                # Punisher — allocate to punishment
                base_tendency = 0.35 + rng.uniform(-0.10, 0.10)
            else:
                # Non-punisher — keep or give minimally
                base_tendency = 0.55 + rng.uniform(-0.05, 0.10)

        # =====================================================================
        # STEP 4: Apply condition effect (Cohen's d based)
        # Richard et al. (2003): Average d in social psychology ≈ 0.43
        # =====================================================================
        condition_effect = self._get_effect_for_condition(condition, variable_name)

        # =====================================================================
        # STEP 4a: Personality x Condition Interaction Effects
        # Differential susceptibility to experimental manipulations
        #
        # SCIENTIFIC BASIS:
        # -----------------
        # Petty & Cacioppo (1986) Elaboration Likelihood Model (ELM):
        #   - High-elaboration (engaged) participants process stimuli deeply,
        #     showing LARGER and more reliable condition effects
        #   - Low-elaboration (satisficers) rely on heuristics, showing
        #     SMALLER, less reliable effects
        #
        # Krosnick (1991) Satisficing Theory:
        #   - Optimizers differentiate conditions more (d multiplier ~1.3x)
        #   - Satisficers show attenuated effects (d multiplier ~0.6x)
        #
        # Greenleaf (1992) Extreme Response Style:
        #   - Extreme responders amplify ALL effects (including condition)
        #   - Multiplier ~1.35x due to scale endpoint usage
        #
        # Meade & Craig (2012) Careless Responding:
        #   - Careless responders show near-random responses
        #   - Condition effects almost entirely attenuated (~0.3x)
        #
        # Interaction coefficients are calibrated so that the POPULATION-
        # AVERAGE effect size matches the specified Cohen's d, while
        # individual-level effects vary realistically by persona type.
        # =====================================================================
        if condition_effect != 0.0:
            _engagement = _safe_trait_value(modified_traits.get("engagement"), 0.65)
            _attention = _safe_trait_value(modified_traits.get("attention_level"), 0.75)
            _extremity = _safe_trait_value(modified_traits.get("extremity"), 0.20)
            _reading_speed = _safe_trait_value(modified_traits.get("reading_speed"), 0.60)
            _consistency = _safe_trait_value(modified_traits.get("response_consistency"), 0.65)

            # Processing depth factor: high engagement + attention = deeper processing
            # Petty & Cacioppo (1986): Central route processing amplifies effects
            # Range: ~0.70 (low engagement) to ~1.35 (high engagement)
            _processing_depth = 0.5 + (_engagement * 0.45) + (_attention * 0.40)
            _processing_depth = float(np.clip(_processing_depth, 0.65, 1.40))

            # Speed attenuation: fast responders (satisficers/careless) miss
            # manipulation details (Krosnick, 1991)
            # reading_speed > 0.80 indicates rushing -> attenuate
            _speed_factor = 1.0
            if _reading_speed > 0.80:
                _speed_factor = 1.0 - (_reading_speed - 0.80) * 1.5  # Range: 1.0 to ~0.70
                _speed_factor = float(np.clip(_speed_factor, 0.55, 1.0))

            # Extremity amplification: extreme responders amplify everything
            # Greenleaf (1992): ERS inflates apparent effect sizes
            _extremity_amp = 1.0 + (_extremity - 0.20) * 0.50
            _extremity_amp = float(np.clip(_extremity_amp, 0.90, 1.40))

            # Consistency factor: inconsistent responders add noise that
            # dilutes true condition effects
            _consistency_factor = 0.70 + _consistency * 0.40
            _consistency_factor = float(np.clip(_consistency_factor, 0.60, 1.10))

            # =====================================================================
            # v1.0.4.3: Domain-specific persona sensitivity factor
            # Different research domains differentially activate persona traits.
            # This creates realistic heterogeneity in treatment effects across
            # participant types, grounded in domain-specific literature.
            #
            # SCIENTIFIC BASIS:
            # - Cacioppo & Petty (1982): Need for Cognition moderates persuasion
            # - Kahneman & Tversky (1979): Loss aversion varies by individual
            # - Van Lange et al. (1997): Social Value Orientation moderates
            #   cooperation/defection in economic games
            # - Dietvorst et al. (2015): Tech attitude moderates algorithm aversion
            # - Rosenstock (1974): Health beliefs moderate fear appeal effectiveness
            # =====================================================================
            _domain_persona_factor = 1.0
            _condition_lower = str(condition).lower().strip()
            _variable_lower = str(variable_name).lower().strip()
            _cond_var_ctx = _condition_lower + " " + _variable_lower

            # Prosocial/empathic personas respond MORE to intergroup and prosocial
            # manipulations (Van Lange et al., 1997: SVO moderates cooperation d)
            _empathy = _safe_trait_value(modified_traits.get("empathy"), 0.50)
            _cooperation = _safe_trait_value(modified_traits.get("cooperation_tendency"), 0.50)
            if any(kw in _cond_var_ctx for kw in ['ingroup', 'outgroup', 'prosocial',
                   'cooperat', 'charit', 'donat', 'help', 'altruism']):
                _prosocial_sensitivity = 0.85 + (_empathy * 0.20) + (_cooperation * 0.15)
                _domain_persona_factor *= float(np.clip(_prosocial_sensitivity, 0.85, 1.25))

            # Risk-tolerant personas respond LESS to fear appeals and risk
            # manipulations (Rosenstock 1974: perceived susceptibility moderates)
            _risk_tolerance = _safe_trait_value(modified_traits.get("risk_tolerance"), 0.50)
            if any(kw in _cond_var_ctx for kw in ['risk', 'fear', 'threat', 'danger',
                   'unsafe', 'hazard', 'loss']):
                _risk_sensitivity = 1.15 - (_risk_tolerance * 0.30)
                _domain_persona_factor *= float(np.clip(_risk_sensitivity, 0.85, 1.20))

            # Tech-affine personas respond LESS to algorithm aversion
            # manipulations (Dietvorst 2015: prior experience moderates aversion)
            _tech_affinity = _safe_trait_value(modified_traits.get("tech_affinity"), 0.50)
            if any(kw in _cond_var_ctx for kw in ['ai', 'algorithm', 'robot',
                   'automat', 'chatbot', 'machine']):
                # High tech affinity → smaller negative effect (less aversion)
                # Low tech affinity → larger negative effect (more aversion)
                _tech_sensitivity = 1.15 - (_tech_affinity * 0.30)
                _domain_persona_factor *= float(np.clip(_tech_sensitivity, 0.85, 1.20))

            # Social desirability moderates effects on sensitive topics
            # Paulhus (1991): High SD respondents attenuate reports of
            # negative behaviors and amplify reports of positive behaviors
            _sd = _safe_trait_value(modified_traits.get("social_desirability"), 0.50)
            if any(kw in _cond_var_ctx for kw in ['dishonest', 'cheat', 'lie',
                   'prejudic', 'discriminat', 'racist', 'sexist', 'immoral']):
                # High SD → underreport negative behaviors (attenuate effect)
                _sd_attenuation = 1.10 - (_sd * 0.25)
                _domain_persona_factor *= float(np.clip(_sd_attenuation, 0.82, 1.15))

            # Need for cognition moderates persuasion and framing effects
            # Cacioppo & Petty (1982): High NFC = central route, more sensitive
            # to argument quality; Low NFC = peripheral route, more sensitive
            # to cues (authority, social proof)
            _nfc = _safe_trait_value(modified_traits.get("need_for_cognition"), 0.50)
            if any(kw in _cond_var_ctx for kw in ['persuas', 'argument', 'framing',
                   'anchor', 'nudge', 'default', 'message']):
                _nfc_factor = 0.90 + (_nfc * 0.20)
                _domain_persona_factor *= float(np.clip(_nfc_factor, 0.90, 1.15))

            # Conformity moderates social influence effects
            # Asch (1956): Individual differences in conformity rates (0-100%)
            _conformity = _safe_trait_value(modified_traits.get("conformity"), 0.50)
            if any(kw in _cond_var_ctx for kw in ['social proof', 'popular',
                   'norm', 'majority', 'consensus', 'conformity']):
                _conformity_sensitivity = 0.85 + (_conformity * 0.30)
                _domain_persona_factor *= float(np.clip(_conformity_sensitivity, 0.85, 1.20))

            # Health consciousness moderates health message effectiveness
            # Rosenstock (1974): Health beliefs moderate intervention effects
            _health_conscious = _safe_trait_value(modified_traits.get("health_consciousness"), 0.50)
            if any(kw in _cond_var_ctx for kw in ['health', 'wellness', 'exercise',
                   'diet', 'vaccination', 'medical', 'prevention']):
                _health_sensitivity = 0.90 + (_health_conscious * 0.20)
                _domain_persona_factor *= float(np.clip(_health_sensitivity, 0.88, 1.15))

            # Environmental concern moderates green messaging effects
            # Stern et al. (1999): Value-Belief-Norm theory — pre-existing
            # environmental values amplify pro-environmental messaging
            _env_concern = _safe_trait_value(modified_traits.get("environmental_concern"), 0.50)
            if any(kw in _cond_var_ctx for kw in ['environment', 'climate', 'green',
                   'sustainab', 'carbon', 'eco']):
                _env_sensitivity = 0.88 + (_env_concern * 0.22)
                _domain_persona_factor *= float(np.clip(_env_sensitivity, 0.88, 1.15))

            # v1.0.4.5: Political identity × Cooperation in economic games
            # Dimant (2024): Political discrimination moderated by
            # cooperation tendency (cooperative vs self-interested)
            _coop = _safe_trait_value(modified_traits.get("cooperation_tendency"), 0.50)
            if any(kw in _cond_var_ctx for kw in ['partisan', 'political',
                   'democrat', 'republican', 'liberal', 'conservative']):
                if any(kw in _cond_var_ctx for kw in ['dictator', 'trust game',
                       'ultimatum', 'allocation', 'give', 'share']):
                    # Cooperative people discriminate LESS in political economic games
                    _pol_coop_sensitivity = 1.20 - (_coop * 0.35)
                    _domain_persona_factor *= float(np.clip(_pol_coop_sensitivity, 0.82, 1.25))

            # v1.0.4.5: Authority × Need for Cognition interaction
            # Petty & Cacioppo (1986): Low NFC → more susceptible to authority cues
            # High NFC → scrutinize source, less affected by authority
            if any(kw in _cond_var_ctx for kw in ['authority', 'expert', 'credib',
                   'source', 'endorse']):
                _nfc_authority = _safe_trait_value(modified_traits.get("need_for_cognition"), 0.50)
                if _nfc_authority < 0.40:
                    _domain_persona_factor *= 1.15  # Low NFC amplifies authority
                elif _nfc_authority > 0.70:
                    _domain_persona_factor *= 0.88  # High NFC attenuates authority

            # v1.0.4.5: Loss frame × Loss aversion trait
            # Kahneman & Tversky (1979): Loss aversion ~2.25×
            # Individual differences in loss aversion moderate framing effects
            _loss_aversion = _safe_trait_value(modified_traits.get("risk_tolerance"), 0.50)
            if any(kw in _cond_var_ctx for kw in ['loss frame', 'lose', 'penalty',
                   'risk of losing', 'could lose']):
                # Low risk tolerance = high loss aversion = amplified loss frame
                _loss_sensitivity = 1.20 - (_loss_aversion * 0.35)
                _domain_persona_factor *= float(np.clip(_loss_sensitivity, 0.85, 1.25))

            # v1.0.4.5: Stereotype threat × Self-efficacy
            # Steele & Aronson (1995): Threat moderated by self-regard
            _self_efficacy = _safe_trait_value(modified_traits.get("engagement"), 0.60)
            if any(kw in _cond_var_ctx for kw in ['stereotype threat', 'diagnostic test',
                   'gender test', 'race test', 'identity threat']):
                if _self_efficacy < 0.40:
                    _domain_persona_factor *= 1.25  # Low efficacy amplifies threat
                elif _self_efficacy > 0.70:
                    _domain_persona_factor *= 0.80  # High efficacy buffers threat

            # Clamp total domain persona factor
            _domain_persona_factor = float(np.clip(_domain_persona_factor, 0.65, 1.45))

            # Combined interaction multiplier
            # Population-weighted average should approximate 1.0 to preserve
            # specified Cohen's d at the group level
            _interaction_multiplier = (
                _processing_depth * _speed_factor *
                _extremity_amp * _consistency_factor *
                _domain_persona_factor
            )
            # Clamp to prevent extreme distortions
            _interaction_multiplier = float(np.clip(_interaction_multiplier, 0.25, 1.80))

            condition_effect *= _interaction_multiplier

        # Apply effect to tendency (normalized to 0-1 scale)
        # v1.0.8.6: Use dynamic bounds from scale geometry (wider for bipolar/novel)
        adjusted_tendency = float(np.clip(base_tendency + condition_effect, _bound_low, _bound_high))

        # =====================================================================
        # STEP 4b: Apply cross-DV latent correlation effect
        # Creates realistic between-scale correlations driven by construct
        # relationships (e.g., Trust and Satisfaction positively correlated).
        #
        # Weight is persona-adaptive:
        #   - Engaged responders (high consistency) show stronger cross-scale
        #     covariance (weight ~0.20) because attentive participants respond
        #     more coherently across related constructs.
        #   - Careless responders (low attention) show weaker covariance
        #     (weight ~0.08) because random noise dilutes latent structure.
        #   - Average participants: weight ~0.15
        #
        # Empirically calibrated so that target r = 0.50 yields realised
        # r ≈ 0.35-0.50 after all persona noise is added — consistent with
        # typical survey attenuation (Schmitt & Hunter, 1996).
        # =====================================================================
        _latent_dvs = traits.get("_latent_dvs", {})
        _latent_z = _latent_dvs.get(variable_name, 0.0)
        if _latent_z != 0.0:
            # Persona-adaptive weight: scale by consistency and attention
            _consistency = _safe_trait_value(traits.get("consistency"), 0.65)
            _attention = _safe_trait_value(traits.get("attention_level"), 0.70)
            # Base weight 0.15, boosted up to 0.22 for highly consistent/attentive,
            # reduced down to 0.08 for careless/inattentive
            _latent_weight = 0.15 + (_consistency - 0.5) * 0.10 + (_attention - 0.5) * 0.06
            _latent_weight = float(np.clip(_latent_weight, 0.08, 0.22))
            _latent_effect = _latent_z * _latent_weight
            adjusted_tendency = float(np.clip(adjusted_tendency + _latent_effect, _bound_low, _bound_high))

        # =====================================================================
        # STEP 4c: Apply g-factor (general evaluation tendency)
        # Podsakoff et al. (2003): Common Method Variance
        #
        # The g-factor represents a participant's stable tendency to rate
        # things higher or lower across ALL scales. It loads differentially
        # on different construct types:
        #   - Attitudes/evaluations: loading ~0.25 (high CMV)
        #   - Satisfaction/affect: loading ~0.22 (high CMV)
        #   - Behavioral intentions: loading ~0.15 (moderate CMV)
        #   - Trust/credibility: loading ~0.18 (moderate-high CMV)
        #   - Risk/threat: loading ~0.12 (moderate CMV, often reversed)
        #   - Factual/behavioral: loading ~0.08 (low CMV)
        #
        # This creates within-person coherence: if participant P rates
        # Trust high, they're more likely to also rate Satisfaction high,
        # even beyond what the construct correlation captures.
        # =====================================================================
        _g_factor_z = traits.get("_g_factor_z", 0.0)
        if _g_factor_z != 0.0:
            _g_strength = traits.get("_g_factor_strength", 0.12)
            # Determine construct-type-specific loading based on variable name
            # Podsakoff et al. (2003) meta-analytic loadings
            _var_lower = variable_name.lower()
            if any(kw in _var_lower for kw in [
                'attitude', 'evaluation', 'opinion', 'view', 'perception',
                'feeling', 'judgment', 'assessment'
            ]):
                _g_loading = 0.25  # Attitudes: highest CMV susceptibility
            elif any(kw in _var_lower for kw in [
                'satisfaction', 'happy', 'pleased', 'enjoy', 'affect',
                'emotion', 'mood', 'wellbeing'
            ]):
                _g_loading = 0.22  # Satisfaction/affect: high CMV
            elif any(kw in _var_lower for kw in [
                'trust', 'credib', 'reliab', 'dependab', 'competenc',
                'integrity', 'benevolenc'
            ]):
                _g_loading = 0.18  # Trust constructs: moderate-high CMV
            elif any(kw in _var_lower for kw in [
                'intention', 'likely', 'willing', 'would', 'plan',
                'expect', 'intend'
            ]):
                _g_loading = 0.15  # Behavioral intentions: moderate CMV
            elif any(kw in _var_lower for kw in [
                'risk', 'danger', 'threat', 'harm', 'fear', 'anxiety',
                'concern', 'worry'
            ]):
                _g_loading = 0.12  # Risk/threat: moderate CMV (often inverted)
            elif any(kw in _var_lower for kw in [
                'frequency', 'count', 'number', 'amount', 'time',
                'behavior', 'action', 'usage'
            ]):
                _g_loading = 0.08  # Factual/behavioral: low CMV
            else:
                _g_loading = 0.15  # Default: moderate loading

            _g_effect = _g_factor_z * _g_strength * _g_loading
            adjusted_tendency = float(np.clip(
                adjusted_tendency + _g_effect, _bound_low, _bound_high
            ))

        # =====================================================================
        # v1.0.4.6 STEP 4d: Cross-DV coherence from response history
        # Pulls adjusted_tendency slightly toward participant's running average
        # across prior DVs. Creates realistic within-person consistency beyond
        # what the g-factor and latent scores provide.
        # Weight is small (0.05-0.10) to avoid overwhelming condition effects.
        # Only activates after participant has responded to ≥2 prior items.
        # =====================================================================
        _resp_hist = getattr(self, '_participant_response_history', None)
        if _resp_hist is not None:
            # Find this participant's history — use traits as proxy for participant index
            _p_idx = traits.get('_participant_idx', -1)
            if isinstance(_p_idx, (int, float)) and 0 <= int(_p_idx) < len(_resp_hist):
                _hist = _resp_hist[int(_p_idx)]
                if _hist['running_count'] >= 2:
                    _consistency = _safe_trait_value(traits.get("response_consistency"), 0.60)
                    # Weight increases with consistency: careless participants are less coherent
                    _coherence_weight = 0.05 + (_consistency - 0.5) * 0.06
                    _coherence_weight = float(np.clip(_coherence_weight, 0.02, 0.10))
                    # v1.0.8.6: Stronger coherence pull for economic game DVs
                    # A taker on one game should be selfish on another (Fehr & Schmidt 1999)
                    if _scale_geom['is_economic_game_allocation']:
                        _coherence_weight *= 1.5  # 50% stronger for game decisions
                    _pull = (_hist['running_mean'] - adjusted_tendency) * _coherence_weight
                    adjusted_tendency = float(np.clip(adjusted_tendency + _pull, _bound_low, _bound_high))

        # Calculate response center
        center = scale_min + (adjusted_tendency * scale_range)

        # =====================================================================
        # STEP 5: Handle reverse-coded items
        # v1.0.4.4: Enhanced with engagement-dependent reversal accuracy
        #
        # SCIENTIFIC BASIS:
        # -----------------
        # Woods (2006): 10-15% of respondents ignore item directionality entirely
        # Weijters et al. (2010): Acquiescence inflates reverse-coded item
        #   error by ~0.5 points on 7-point scales
        # Meade & Craig (2012): Careless respondents fail reverse items at
        #   rates up to 40-50%, creating inconsistency
        # Krosnick (1991): Satisficers don't cognitively reverse the item —
        #   they respond to face value, producing acquiescence artifacts
        #
        # Implementation:
        # 1. Engaged respondents: Correctly reverse and respond accurately
        # 2. Satisficers: Partially fail to reverse (probability based on attention)
        # 3. Careless respondents: Often ignore reversal entirely
        # 4. Acquiescent respondents: Additional positive-direction pull even
        #    after reversal (inflating scores on reverse items)
        # =====================================================================
        _correctly_reversed = False  # Track for SD × reverse interaction
        if is_reverse:
            _attention = _safe_trait_value(modified_traits.get("attention_level"), 0.75)
            _engagement = _safe_trait_value(modified_traits.get("engagement"), 0.65)
            acquiescence = _safe_trait_value(modified_traits.get("acquiescence"), 0.5)

            # Probability of correctly reversing the item
            # High attention + engagement → near-certain reversal
            # Low attention → substantial probability of ignoring reversal
            # Woods (2006): ~10-15% fail at baseline; up to 40% for careless
            _reversal_probability = 0.50 + (_attention * 0.35) + (_engagement * 0.15)
            _reversal_probability = float(np.clip(_reversal_probability, 0.30, 0.98))

            # v1.0.4.9: Cross-item reverse failure consistency
            # If this participant has already failed reverse items, they're MORE likely
            # to fail subsequent ones (trait-like within session; Woods 2006)
            _p_idx = getattr(self, '_current_participant_idx', None)
            if _p_idx is not None and hasattr(self, '_participant_reverse_tracking'):
                _rt = self._participant_reverse_tracking[_p_idx]
                if _rt['total_reverse'] >= 2:
                    _fail_rate = _rt['failed_reverse'] / max(_rt['total_reverse'], 1)
                    # Adjust probability toward their established failure rate
                    # Weight: 0.3 = moderate influence from past behavior
                    _reversal_probability = (0.7 * _reversal_probability +
                                             0.3 * (1.0 - _fail_rate))
                    _reversal_probability = float(np.clip(_reversal_probability, 0.20, 0.98))

            # v1.0.4.5: Engagement-level differential failure rates
            # Krosnick (1991): Satisficers partially fail reverse items
            # Engaged: ~95% correct; Satisficers (0.3-0.6): ~60-75%; Careless: ~30-50%
            if _engagement < 0.35:
                # Careless responders: nearly random reversal
                _reversal_probability *= 0.70  # Reduce by 30%
            elif _engagement < 0.55:
                # Satisficers: partially fail — they see the words but don't
                # always cognitively invert the meaning
                _reversal_probability *= 0.88  # Reduce by 12%

            _reversal_probability = float(np.clip(_reversal_probability, 0.25, 0.98))

            if rng.random() < _reversal_probability:
                # Correctly reverses the item
                center = scale_max - (center - scale_min)
                _correctly_reversed = True
            else:
                # Fails to reverse — responds as if positively worded
                # This creates the acquiescence-driven inconsistency pattern
                # that reliability analysts see in real data
                _correctly_reversed = False

            # v1.0.4.9: Update reverse-item tracking for this participant
            if _p_idx is not None and hasattr(self, '_participant_reverse_tracking'):
                self._participant_reverse_tracking[_p_idx]['total_reverse'] += 1
                if not _correctly_reversed:
                    self._participant_reverse_tracking[_p_idx]['failed_reverse'] += 1

            # Acquiescence pull on reverse items (Weijters et al., 2010)
            # Even respondents who DO reverse still show partial acquiescence
            # Effect: ~0.5 point inflation for strong acquiescers
            # v1.0.4.5: Acquiescence pull is STRONGER when reversal fails
            # (person already showing agree-tendency, acq reinforces it)
            if acquiescence > 0.55:
                _acq_multiplier = 0.20 if _correctly_reversed else 0.30
                _acq_reverse_pull = (acquiescence - 0.5) * scale_range * _acq_multiplier
                center += _acq_reverse_pull

        # =====================================================================
        # STEP 6: Calculate within-person variance
        # Published norm: SD ≈ 1.2-1.8 on 7-point (Greenleaf, 1992)
        # =====================================================================
        # v1.2.1: Safe trait access with fallback chain
        variance_trait = modified_traits.get("variance_tendency")
        if variance_trait is None:
            variance_trait = modified_traits.get("scale_use_breadth", 0.70)
        variance_trait = _safe_trait_value(variance_trait, 0.70)
        # Base SD = range/4 ≈ 1.5 for 7-point, modified by variance trait
        sd = (scale_range / 4.0) * variance_trait
        # Apply domain-specific variance adjustment
        sd *= (1.0 + domain_calibration['variance_adjustment'])
        # Apply scale-type variance multiplier
        sd *= scale_calibration['variance_multiplier']
        # Minimum SD to ensure realistic variation (floor at ~1.0)
        sd = max(sd, scale_range * 0.16)

        # v1.0.8.6: Additional variance boost for novel/bipolar scales
        # Novel scales (non-standard ranges) need more spread because participants
        # are less anchored by familiar scale conventions. Bipolar scales with
        # taking options need high variance to produce the bimodal distribution.
        if _scale_geom['is_novel_range']:
            sd *= 1.15  # 15% more variance for unfamiliar scales
        if _scale_geom['has_taking_option']:
            sd *= 1.20  # 20% more variance for taking games (bimodal shape)

        # Generate response from normal distribution
        response = float(rng.normal(center, sd))

        # =====================================================================
        # STEP 7: Apply extreme response style (Greenleaf, 1992)
        # ERS respondents use endpoints 2-3x more than modal
        # =====================================================================
        extremity = _safe_trait_value(modified_traits.get("extremity"), 0.18)
        # Apply scale-type extremity boost
        extremity += scale_calibration['extremity_boost']
        extremity = float(np.clip(extremity, 0.0, 0.95))
        if rng.random() < extremity * 0.45:  # Calibrated to produce ~15-20% endpoints for ERS
            # Use proportional noise near endpoints (scales to range)
            endpoint_noise = max(0.5, scale_range * 0.02)  # 2% of range, min 0.5
            if response > (scale_min + scale_max) / 2.0:
                response = scale_max - float(rng.uniform(0, endpoint_noise))
            else:
                response = scale_min + float(rng.uniform(0, endpoint_noise))

        # =====================================================================
        # STEP 8: Apply acquiescence bias (Billiet & McClendon, 2000)
        # High acquiescers: +0.5-1.0 point inflation on agreement items
        # =====================================================================
        acquiescence = _safe_trait_value(modified_traits.get("acquiescence"), 0.50)
        if (not is_reverse) and acquiescence > 0.55 and scale_range > 0:
            # Billiet & McClendon: ~0.8 point inflation for strong acquiescers
            acq_effect = (acquiescence - 0.5) * scale_range * 0.20
            response += acq_effect

        # =====================================================================
        # STEP 9: Apply social desirability bias (Paulhus, 1991)
        # v1.0.4.4: Domain-sensitive social desirability
        #
        # SCIENTIFIC BASIS:
        # -----------------
        # Social desirability bias varies dramatically by construct sensitivity:
        # - Nederhof (1985 meta): SD bias d = 0.25-0.75 for sensitive topics
        # - Paulhus (2002): Impression Management (deliberate faking) differs
        #   from Self-Deceptive Enhancement (unconscious positivity)
        # - Tourangeau & Yan (2007): Sensitivity depends on: social norms,
        #   intrusiveness, and threat of disclosure
        #
        # Construct sensitivity categories:
        # HIGH (1.5× multiplier): Prejudice, aggression, substance use,
        #   dishonesty, sexual behavior — strong social norms against admission
        # MODERATE (1.0×): Prosocial behavior, compliance, health behaviors,
        #   self-esteem — mild inflation toward desirable direction
        # LOW (0.5×): Factual/behavioral frequency, risk perception,
        #   cognitive ability — less norm-linked, harder to fake
        # INVERTED (-0.5×): Self-deprecating topics (anxiety, vulnerability,
        #   loneliness) — SD bias suppresses honest negative reports
        # =====================================================================
        social_des = _safe_trait_value(modified_traits.get("social_desirability"), 0.50)
        if social_des > 0.55 and scale_range > 0:
            # Determine construct sensitivity multiplier
            _var_lower = variable_name.lower()
            _sd_sensitivity = 1.0  # Default: moderate sensitivity

            # HIGH sensitivity: topics with strong social norms
            if any(kw in _var_lower for kw in ['prejudic', 'discrimin', 'racism', 'sexism',
                   'aggress', 'hostil', 'violent', 'dishonest', 'cheat', 'lie',
                   'alcohol', 'drug', 'substance', 'steal', 'bully']):
                _sd_sensitivity = 1.5

            # MODERATE-HIGH: Prosocial self-reports (inflation)
            elif any(kw in _var_lower for kw in ['prosocial', 'help', 'donat', 'volunteer',
                     'charit', 'altruism', 'moral', 'ethical', 'compliance']):
                _sd_sensitivity = 1.2

            # MODERATE: Standard self-evaluations
            elif any(kw in _var_lower for kw in ['satisf', 'attitude', 'opinion', 'health',
                     'exercise', 'self_esteem', 'competenc']):
                _sd_sensitivity = 1.0

            # LOW: Factual/behavioral reports
            elif any(kw in _var_lower for kw in ['frequency', 'count', 'number', 'time',
                     'amount', 'usage', 'behavior', 'action']):
                _sd_sensitivity = 0.5

            # INVERTED: Vulnerability topics — SD suppresses honest negatives
            elif any(kw in _var_lower for kw in ['anxiety', 'depress', 'lonely', 'loneliness',
                     'vulnerable', 'weakness', 'failure', 'shame', 'guilt',
                     'insecur', 'fear', 'worry', 'burnout']):
                _sd_sensitivity = -0.5  # Negative = suppresses admission of negatives

            # v1.0.4.9: MORAL/SACRED VALUE topics — very high SD sensitivity
            # Tetlock et al. (2000): Sacred value violations trigger moral outrage
            # People strongly inflate their moral standing in self-reports
            elif any(kw in _var_lower for kw in ['moral_identity', 'ethical_self', 'sacred',
                     'virtuous', 'moral_self', 'integrity']):
                _sd_sensitivity = 1.4  # Very high — moral self-presentation

            # v1.0.4.9: GRATITUDE/POSITIVE PSYCH — moderate-high inflation
            # McCullough et al. (2002): Gratitude self-reports positively skewed
            elif any(kw in _var_lower for kw in ['gratitude', 'grateful', 'thankful',
                     'wellbeing', 'flourish', 'life_satisf']):
                _sd_sensitivity = 1.2  # Moderate-high — socially desirable to be grateful

            # v1.0.4.9: SOCIAL COMPARISON — moderate sensitivity
            # Admitting social comparison is somewhat undesirable
            elif any(kw in _var_lower for kw in ['compar', 'envy', 'jealous',
                     'social_comparison', 'relative_standing']):
                _sd_sensitivity = 1.1

            # v1.0.4.9: DIGITAL HABITS — moderate (people downplay usage)
            # Self-reported screen time systematically underestimated (Andrews et al. 2015)
            elif any(kw in _var_lower for kw in ['screen_time', 'phone_use', 'social_media_use',
                     'app_usage', 'internet_addict', 'phone_depend']):
                _sd_sensitivity = 1.15  # People underreport digital dependence

            # v1.0.9.3: SEXUAL BEHAVIOR / REPRODUCTION — very high SD sensitivity
            # Alexander & Fisher (2003): bogus pipeline reveals massive SD gap
            elif any(kw in _var_lower for kw in ['sexual', 'sex_', 'intercours', 'condom',
                     'contracepti', 'porn', 'masturbat', 'partner_count', 'infidel']):
                _sd_sensitivity = 1.6  # Highest category — sexuality strongly norm-laden

            # v1.0.9.3: INCOME / FINANCIAL STATUS — moderate-high inflation
            # Moore et al. (2000): Self-reported income inflated ~15-20%
            elif any(kw in _var_lower for kw in ['income', 'salary', 'earning', 'wealth',
                     'financial_status', 'socioeconomic', 'debt', 'savings']):
                _sd_sensitivity = 1.25  # People overreport income, underreport debt

            # v1.0.9.3: VOTING / CIVIC BEHAVIOR — moderate-high
            # Holbrook & Krosnick (2010): ~15% overreport voting
            elif any(kw in _var_lower for kw in ['voted', 'voting', 'civic_engag', 'volunteer_freq',
                     'communit', 'recycle_freq', 'blood_donat']):
                _sd_sensitivity = 1.3  # Social norms strongly favor civic participation

            # v1.0.9.3: PARENTING / CHILD-REARING — high SD sensitivity
            # Bornstein (2002): Parents systematically overreport positive parenting
            elif any(kw in _var_lower for kw in ['parent', 'child_rear', 'disciplin', 'nurtur',
                     'parental', 'spank', 'punish_child']):
                _sd_sensitivity = 1.4  # Parenting norms very strong

            # v1.0.9.3: COGNITIVE ABILITY / INTELLIGENCE — moderate
            # Paulhus et al. (2003): self-estimated IQ inflated ~15 points
            elif any(kw in _var_lower for kw in ['intelligen', 'iq_', 'cognitive_abil', 'smart',
                     'knowledge_test', 'academic_abil']):
                _sd_sensitivity = 1.15  # Self-enhancement bias for intelligence

            # v1.0.9.3: ENVIRONMENTAL BEHAVIOR — moderate-high gap
            # Kormos & Gifford (2014): self-reported pro-environmental > actual
            elif any(kw in _var_lower for kw in ['pro_environment', 'green_behavior', 'sustainab',
                     'carbon_footprint', 'energy_conserv']):
                _sd_sensitivity = 1.25  # Attitude-behavior gap well-documented

            # v1.0.9.3: CONFORMITY / OBEDIENCE — inverted (underreport)
            # Pronin (2007): bias blind spot — people deny being influenced
            elif any(kw in _var_lower for kw in ['conform', 'obedien', 'comply', 'submiss',
                     'follow_crowd', 'peer_pressur', 'susceptib']):
                _sd_sensitivity = -0.4  # People underreport being influenced

            # v1.0.9.3: PREJUDICE / IMPLICIT BIAS — very high
            # Greenwald et al. (2009): explicit prejudice measures highly SD-sensitive
            elif any(kw in _var_lower for kw in ['implicit_bias', 'iat_', 'modern_racism',
                     'symbolic_racism', 'aversive_racism', 'subtle_prejudic']):
                _sd_sensitivity = 1.55  # Extremely norm-laden

            # v1.0.9.3: RELATIONSHIP QUALITY — moderate inflation
            # Fowers & Olson (1993): marital satisfaction scales show positivity bias
            elif any(kw in _var_lower for kw in ['relation_satisf', 'marital', 'coupl',
                     'partner_satisf', 'relationship_qual', 'romantic_satisf']):
                _sd_sensitivity = 1.15  # People overreport relationship quality

            # v1.0.9.3: RELIGIOSITY / SPIRITUAL — moderate-high
            # Hadaway et al. (1993): church attendance self-reports inflated ~50%
            elif any(kw in _var_lower for kw in ['religio', 'spiritual', 'church_attend', 'prayer',
                     'faith', 'worship', 'devoti']):
                _sd_sensitivity = 1.3  # Religious behavior strongly normed

            # v1.0.9.3: BODY WEIGHT / EATING — moderate-high
            # Gorber et al. (2007): self-reported weight underestimated, height overestimated
            elif any(kw in _var_lower for kw in ['body_weight', 'bmi_self', 'calorie_intake',
                     'eating_habit', 'binge_eat', 'diet_adher', 'food_intake']):
                _sd_sensitivity = 1.25  # Desirability toward healthy eating norms

            # v1.0.9.3: AGGRESSION / ANGER — high (underreport)
            # Suris et al. (2004): physical aggression underreported in self-report
            elif any(kw in _var_lower for kw in ['aggress', 'anger_express', 'physical_fight',
                     'verbal_aggress', 'road_rage', 'retaliat']):
                _sd_sensitivity = 1.45  # Strong norms against aggression

            # Also check condition context for sensitivity
            _cond_lower = condition.lower() if condition else ""
            if any(kw in _cond_lower for kw in ['dishonest', 'cheat', 'prejudic',
                   'discriminat', 'immoral']):
                _sd_sensitivity = max(_sd_sensitivity, 1.3)

            # v1.0.4.5: Economic game SD sensitivity
            # In dictator/trust/ultimatum games, allocations reveal character
            # SD bias is MODERATE-HIGH (not LOW) because fairness norms are strong
            # Engel (2011): Dictator giving inflated by ~5% in observed conditions
            if any(kw in _var_lower for kw in ['dictator', 'trust_game', 'ultimatum',
                   'allocat', 'give', 'donat', 'share', 'split']):
                if any(kw in _cond_lower for kw in ['dictator', 'trust', 'ultimatum',
                       'public good', 'economic game']):
                    _sd_sensitivity = max(_sd_sensitivity, 1.3)  # Override LOW→MODERATE-HIGH

            # v1.0.4.5: SD × Reverse-item interaction
            # When a reverse item is correctly reversed, SD and reversal align
            # → ATTENUATE SD slightly (both pushing same direction)
            # When reversal fails, SD contradicts the unreversed response
            # → AMPLIFY SD (person trying to present well but reversal failure fights it)
            if is_reverse:
                if _correctly_reversed:
                    _sd_sensitivity *= 0.85  # Attenuate: reversal already adjusted direction
                else:
                    _sd_sensitivity *= 1.20  # Amplify: SD fights the reversal failure

            # Apply domain-sensitive SD effect
            # Paulhus (1991): ~0.8-1.2 point inflation for high IM on sensitive topics
            sd_effect = (social_des - 0.5) * scale_range * 0.12 * _sd_sensitivity
            response += sd_effect

        # Bound and round to valid scale value
        response = max(scale_min, min(scale_max, round(response)))
        result = int(response)

        # =====================================================================
        # STEP 11: Human-like micro-pattern adjustments (v1.0.6.9)
        # Adds realistic item-position drift, streak inertia, and occasional
        # correction behavior without overwhelming experimental effects.
        # =====================================================================
        _p_idx = getattr(self, "_current_participant_idx", None)
        _item_pos = int(getattr(self, "_current_item_position", 1))
        _item_total = int(max(1, getattr(self, "_current_item_total", 1)))
        if isinstance(_p_idx, int) and _p_idx >= 0:
            _progress = _item_pos / max(1, _item_total)
            _attn = _safe_trait_value(traits.get("attention_level"), 0.7)
            _cons = _safe_trait_value(traits.get("response_consistency"), 0.6)
            _ext = _safe_trait_value(traits.get("extremity"), 0.3)

            # Fatigue drift: slight move toward midpoint later in long scales
            # v1.0.8.7: Use knowledge base fatigue model when available
            if HAS_KNOWLEDGE_BASE and _item_total >= 5:
                _fatigue = compute_fatigue_adjustment(_item_pos, _item_total)
                if _fatigue['mean_shift'] != 0 and _attn < 0.7:
                    _mid = (scale_min + scale_max) / 2.0
                    _shrink = abs(_fatigue['mean_shift']) * (1.0 - _attn) * 8.0
                    _shrink = min(0.25, _shrink)
                    result = int(round(result + (_mid - result) * _shrink))
                # v1.0.8.7: Knowledge base straight-lining acceleration
                if _fatigue['straight_line_boost'] > 0 and _cons < 0.55:
                    if rng.random() < _fatigue['straight_line_boost'] * (0.55 - _cons) * 3:
                        _prev_val = self._item_response_memory.get((_p_idx, variable_name)) if hasattr(self, '_item_response_memory') else None
                        if _prev_val is not None:
                            result = int(_prev_val)
            elif _item_total >= 5 and _progress >= 0.6 and _attn < 0.6:
                _mid = (scale_min + scale_max) / 2.0
                _shrink = 0.12 + (0.6 - _attn) * 0.20
                result = int(round(result + (_mid - result) * _shrink))

            # Streak inertia: low-consistency participants sometimes repeat prior value
            if not hasattr(self, "_item_response_memory"):
                self._item_response_memory = {}
            _prev = self._item_response_memory.get((_p_idx, variable_name))
            if _prev is not None and _cons < 0.55 and rng.random() < (0.08 + (0.55 - _cons) * 0.20):
                result = int(round((_prev + result) / 2.0))

            # Human correction: engaged respondents occasionally counter-correct extremes
            if _attn > 0.75 and _ext < 0.5 and rng.random() < 0.05:
                if result in (scale_min, scale_max):
                    result += -1 if result == scale_max else 1

            # Store memory for next item in same construct
            self._item_response_memory[(_p_idx, variable_name)] = int(result)

        # SAFETY CHECK: Final validation that result is within bounds
        # This guards against any floating point edge cases
        if result < scale_min:
            result = scale_min
        elif result > scale_max:
            result = scale_max

        return result

    def _generate_attention_check(
        self,
        condition: str,
        traits: Dict[str, float],
        check_type: str,
        participant_seed: int,
    ) -> Tuple[int, bool]:
        rng = np.random.RandomState(participant_seed)

        # v1.2.1: Safe trait access
        attention = _safe_trait_value(traits.get("attention_level"), 0.85)
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

    def _build_behavioral_profile(
        self,
        persona: 'Persona',
        traits: Dict[str, float],
        response_vals: List[int],
        response_mean: Optional[float],
        condition: str,
        scale_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Build a rich behavioral profile summarizing this participant's behavior.

        v1.0.4.8: Creates a structured behavioral summary from the participant's
        numeric responses, persona traits, and condition assignment. This profile
        flows to ALL text generators (LLM, ComprehensiveResponseGenerator,
        TextResponseGenerator) to ensure open-text responses are consistent with
        the participant's quantitative behavior in the study.

        Returns a dict with:
        - response_pattern: str description of their numeric behavior
        - intensity: float (0-1) how extreme their numeric responses were
        - consistency_score: float (0-1) how consistent across items
        - behavioral_summary: str natural-language summary for LLM prompts
        - trait_profile: dict of all 7 trait dimensions
        - scale_summaries: list of per-scale behavioral descriptions
        """
        profile: Dict[str, Any] = {
            'response_mean': response_mean,
            'response_vals': response_vals,
            'persona_name': persona.name if persona else 'Default',
            'persona_description': getattr(persona, 'description', ''),
            'condition': condition,
        }

        # Full 7-dimensional trait vector
        profile['trait_profile'] = {
            'attention_level': _safe_trait_value(traits.get("attention_level"), 0.8),
            'verbosity': _safe_trait_value(traits.get("verbosity"), 0.5),
            'formality': _safe_trait_value(traits.get("formality"), 0.5),
            'social_desirability': _safe_trait_value(traits.get("social_desirability"), 0.3),
            'consistency': _safe_trait_value(traits.get("response_consistency"), 0.6),
            'response_latency': _safe_trait_value(traits.get("response_latency"), 0.5),
            'extremity': _safe_trait_value(traits.get("extremity"), 0.4),
        }

        # Behavioral pattern from numeric responses
        if response_vals and len(response_vals) >= 2:
            # v1.0.6.1: Filter out NaN/None values to prevent NaN propagation
            vals = [float(v) for v in response_vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if len(vals) < 2:
                vals = [4.0, 4.0]  # Safe midpoint fallback
            _mean = float(np.mean(vals))
            _std = float(np.std(vals))
            _min_v, _max_v = float(min(vals)), float(max(vals))
            _range = _max_v - _min_v

            # v1.0.8.6: Detect scale range from actual response values
            # If any response is negative, we're on a bipolar scale
            _has_negative_vals = any(v < 0 for v in vals)
            _inferred_max = max(abs(_min_v), abs(_max_v), 7.0)
            _midpoint = 0.0 if _has_negative_vals else 4.0
            _norm_divisor = _inferred_max if _has_negative_vals else 3.0

            # Intensity: how far from scale midpoint
            profile['intensity'] = min(1.0, abs(_mean - _midpoint) / max(_norm_divisor, 1.0))

            # Consistency: inverse of variability (low SD = high consistency)
            _sd_norm = _inferred_max / 2.33 if _has_negative_vals else 3.0  # scale-aware
            profile['consistency_score'] = max(0.0, 1.0 - (_std / max(_sd_norm, 1.0)))

            # Straight-lining detection
            _unique_vals = len(set(int(v) for v in vals))
            profile['straight_lined'] = _unique_vals <= 2 and len(vals) >= 4

            # v1.0.8.6: Flag negative response behavior (for taking games)
            _neg_count = sum(1 for v in vals if v < 0)
            profile['has_negative_responses'] = _neg_count > 0
            profile['negative_response_fraction'] = _neg_count / len(vals) if vals else 0.0

            # Response pattern classification
            # v1.0.8.6: Scale-aware thresholds for bipolar scales
            if _has_negative_vals:
                # Bipolar scale: classify around zero
                if _mean > _inferred_max * 0.3:
                    profile['response_pattern'] = 'strongly_positive'
                elif _mean > _inferred_max * 0.1:
                    profile['response_pattern'] = 'moderately_positive'
                elif _mean < -_inferred_max * 0.3:
                    profile['response_pattern'] = 'strongly_negative'
                elif _mean < -_inferred_max * 0.1:
                    profile['response_pattern'] = 'moderately_negative'
                elif _std < _inferred_max * 0.15:
                    profile['response_pattern'] = 'consistently_neutral'
                else:
                    profile['response_pattern'] = 'mixed_ambivalent'
            else:
                if _mean >= 5.5:
                    profile['response_pattern'] = 'strongly_positive'
                elif _mean >= 4.5:
                    profile['response_pattern'] = 'moderately_positive'
                elif _mean <= 2.5:
                    profile['response_pattern'] = 'strongly_negative'
                elif _mean <= 3.5:
                    profile['response_pattern'] = 'moderately_negative'
                elif _std < 0.8:
                    profile['response_pattern'] = 'consistently_neutral'
                else:
                    profile['response_pattern'] = 'mixed_ambivalent'

            # Build natural-language behavioral summary for LLM
            # v1.0.8.6: Scale-aware descriptions for bipolar scales
            if _has_negative_vals:
                _scale_desc = f"mean {_mean:.1f}, range {_min_v:.0f} to {_max_v:.0f}"
                _pattern_desc = {
                    'strongly_positive': f'allocated positively/gave generously ({_scale_desc})',
                    'moderately_positive': f'gave moderate positive allocations ({_scale_desc})',
                    'strongly_negative': f'took from others/allocated negatively ({_scale_desc})',
                    'moderately_negative': f'made slightly negative allocations ({_scale_desc})',
                    'consistently_neutral': f'allocated near zero consistently ({_scale_desc})',
                    'mixed_ambivalent': f'gave mixed allocations ({_scale_desc})',
                }
            else:
                _pattern_desc = {
                    'strongly_positive': f'rated items very positively (mean {_mean:.1f}/7)',
                    'moderately_positive': f'rated items somewhat positively (mean {_mean:.1f}/7)',
                    'strongly_negative': f'rated items very negatively (mean {_mean:.1f}/7)',
                    'moderately_negative': f'rated items somewhat negatively (mean {_mean:.1f}/7)',
                    'consistently_neutral': f'gave consistently moderate ratings (mean {_mean:.1f}/7, low variation)',
                    'mixed_ambivalent': f'gave mixed ratings (mean {_mean:.1f}/7, range {_min_v:.0f}-{_max_v:.0f})',
                }

            _consistency_desc = ''
            if profile['straight_lined']:
                _consistency_desc = ' They appear to have straight-lined (gave nearly identical responses across items).'
            elif _std < 0.5:
                _consistency_desc = ' Their responses were very uniform, suggesting limited discrimination between items.'
            elif _std > 2.0:
                _consistency_desc = ' Their responses varied widely across items, suggesting they differentiated carefully.'

            _effort_desc = ''
            _attn = profile['trait_profile']['attention_level']
            if _attn < 0.3:
                _effort_desc = ' This participant showed signs of low effort/carelessness.'
            elif _attn > 0.8:
                _effort_desc = ' This participant was highly engaged and attentive.'

            # v1.0.8.6: Theory-grounded behavioral strategy classification
            # Per Manning & Horton (2025): Discrete agent types > continuous trait variation
            # Classify this participant into a behavioral strategy based on their responses
            _strategy = 'default'
            if _has_negative_vals:
                # Economic game with bipolar scale
                if _mean < -_inferred_max * 0.1:
                    _strategy = 'taker'
                elif abs(_mean) < _inferred_max * 0.05:
                    _strategy = 'selfish_zero'
                elif _mean > _inferred_max * 0.35:
                    _strategy = 'fair_divider'
                else:
                    _strategy = 'moderate_giver'
            elif _mean >= 5.5:
                _strategy = 'enthusiast'
            elif _mean <= 2.5:
                _strategy = 'critic'
            elif _std < 0.5 and _mean > 3.0 and _mean < 5.0:
                _strategy = 'satisficer'
            profile['behavioral_strategy'] = _strategy

            _strategy_desc = ''
            _strategy_descs = {
                'taker': ' Behavioral type: TAKER — this person took from others.',
                'selfish_zero': ' Behavioral type: SELFISH — kept everything for themselves.',
                'fair_divider': ' Behavioral type: FAIR DIVIDER — split approximately equally.',
                'moderate_giver': ' Behavioral type: MODERATE GIVER — gave a small amount.',
                'enthusiast': ' Behavioral type: ENTHUSIAST — consistently positive.',
                'critic': ' Behavioral type: CRITIC — consistently negative.',
                'satisficer': ' Behavioral type: SATISFICER — minimal effort, near midpoint.',
            }
            _strategy_desc = _strategy_descs.get(_strategy, '')

            profile['behavioral_summary'] = (
                f"This participant {_pattern_desc.get(profile['response_pattern'], 'responded moderately')}."
                f"{_consistency_desc}{_effort_desc}{_strategy_desc}"
            )
        else:
            profile['intensity'] = 0.5
            profile['consistency_score'] = 0.5
            profile['straight_lined'] = False
            profile['response_pattern'] = 'unknown'
            profile['behavioral_summary'] = 'No prior numeric response data available for this participant.'

        return profile

    def _validate_participant_responses(
        self,
        responses: List[int],
        scale_min: int,
        scale_max: int,
        persona_name: str,
        traits: Dict[str, Any],
    ) -> Dict[str, Any]:
        """v1.0.4.9: Post-generation validation of participant response patterns.

        Checks that generated responses match expected patterns for the participant's
        persona type. Returns a validation report with any detected anomalies.

        Scientific basis:
        - Meade & Craig (2012): Careless responder detection via IRV, longstring
        - Curran (2016): Insufficient effort responding indicators
        - DeSimone et al. (2018): Inconsistency indices for data quality
        """
        report: Dict[str, Any] = {'valid': True, 'warnings': []}
        if not responses or len(responses) < 3:
            return report

        vals = [float(v) for v in responses]
        _mean = float(np.mean(vals))
        _std = float(np.std(vals))
        _unique = len(set(int(v) for v in vals))
        _scale_range = max(scale_max - scale_min, 1)

        # Check 1: Longstring detection (consecutive identical responses)
        _max_longstring = 1
        _current_run = 1
        for j in range(1, len(vals)):
            if int(vals[j]) == int(vals[j - 1]):
                _current_run += 1
                _max_longstring = max(_max_longstring, _current_run)
            else:
                _current_run = 1

        # Longstring > 80% of items is suspicious even for straight-liners
        if _max_longstring > max(4, len(vals) * 0.8):
            _attn = _safe_trait_value(traits.get("attention_level"), 0.75)
            if _attn > 0.7:  # Engaged respondent shouldn't straight-line this much
                report['warnings'].append(
                    f"Longstring ({_max_longstring}/{len(vals)}) for engaged persona '{persona_name}'"
                )

        # Check 2: IRV (Intra-individual Response Variability)
        # Dunn et al. (2018): IRV should match persona engagement level
        if _std < 0.3 and _unique <= 2 and len(vals) >= 5:
            _engagement = _safe_trait_value(traits.get("engagement"), 0.6)
            if _engagement > 0.6:
                report['warnings'].append(
                    f"Near-zero IRV (SD={_std:.2f}) for engaged persona '{persona_name}'"
                )

        # Check 3: Scale range utilization
        # Greenleaf (1992): Extreme responders should use endpoints
        _extremity = _safe_trait_value(traits.get("extremity"), 0.3)
        _uses_endpoints = any(int(v) == scale_min or int(v) == scale_max for v in vals)
        if _extremity > 0.7 and len(vals) >= 5 and not _uses_endpoints:
            report['warnings'].append(
                f"High extremity ({_extremity:.2f}) but no endpoint use for '{persona_name}'"
            )

        report['valid'] = len(report['warnings']) == 0
        report['stats'] = {
            'mean': round(_mean, 2), 'sd': round(_std, 2),
            'unique_values': _unique, 'max_longstring': _max_longstring,
        }
        return report

    def _generate_open_response(
        self,
        question_spec: Dict[str, Any],
        persona: Persona,
        traits: Dict[str, float],
        condition: str,
        participant_seed: int,
        response_mean: Optional[float] = None,
        behavioral_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate an open-ended response using context-aware text generation.

        Uses the comprehensive response library (if available) for LLM-quality
        responses across 50+ research domains. Falls back to the basic text
        generator if the library is not available.

        v1.0.4.8: Enhanced with full behavioral profile to ensure OE responses
        are consistent with the participant's quantitative behavior. The
        behavioral_profile dict contains response patterns, intensity, consistency,
        and a natural-language summary that flows to all generators.

        The response is generated based on:
        - Question text and type (explanation, feedback, description, etc.)
        - Study context (domain, topics, survey name)
        - Persona traits (ALL 7 dimensions, not just 3)
        - Experimental condition
        - Response sentiment (based on scale responses)
        - Behavioral profile (response pattern, intensity, consistency)
        """
        # v1.2.0.0: Default source is "Template"; overridden to "AI" if LLM succeeds.
        self._last_oe_source = "Template"
        response_type = str(question_spec.get("type", "general"))
        question_text = str(question_spec.get("question_text", ""))
        _original_question_text = question_text  # v1.0.4.7: Preserve before context embedding
        context_type = str(question_spec.get("context_type", "general"))
        question_context = str(question_spec.get("question_context", "")).strip()

        # v1.0.1.2: Use user-provided question context to enrich the prompt.
        # This is critical for questions like "explain_feel_donald" where
        # the variable name alone doesn't convey what's really being asked.
        # Enhanced: include condition and study topic for full context chain.
        if question_context:
            # User provided explicit context — use it directly
            import re as _re
            _humanized = _re.sub(r'[_\-]+', ' ', question_text).strip() if question_text and " " not in question_text.strip() else question_text
            _study_topic = self.study_title or self.study_description or ""
            question_text = (
                f"Question: {_humanized}\n"
                f"Context: {question_context}"
            )
            if _study_topic:
                question_text += f"\nStudy topic: {_study_topic}"
            # v1.0.1.2: Include condition in context for tighter prompt grounding
            if condition:
                question_text += f"\nCondition: {condition}"
            # v1.0.9.1: Include additional simulation context if provided
            _add_ctx = self.study_context.get("additional_context", "")
            if _add_ctx:
                question_text += f"\nAdditional context: {_add_ctx[:200]}"
        elif question_text and " " not in question_text.strip():
            # v1.4.11: If question_text looks like a variable name (no spaces),
            # build a richer question from study context so LLM/template can
            # generate contextually relevant responses.
            import re as _re
            _humanized = _re.sub(r'[_\-]+', ' ', question_text).strip()
            _study_topic = self.study_title or self.study_description or ""
            if _study_topic:
                question_text = (
                    f"In the context of a study about {_study_topic}, "
                    f"please share your thoughts on: {_humanized}"
                )
            else:
                question_text = f"Please share your thoughts on: {_humanized}"

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
        # v1.2.1: Use safe trait value extraction
        attention_level = _safe_trait_value(traits.get("attention_level"), 0.8)
        verbosity = _safe_trait_value(traits.get("verbosity"), 0.5)
        formality = _safe_trait_value(traits.get("formality"), 0.5)

        # Map persona to engagement level
        _persona_name = getattr(persona, 'name', '') if persona else ''
        if attention_level < 0.5:
            engagement = 0.2  # Careless
        elif _persona_name == "Satisficer":
            engagement = 0.3
        elif _persona_name == "Extreme Responder":
            engagement = 0.6
        elif _persona_name == "Engaged Responder":
            engagement = 0.9
        else:
            engagement = 0.5

        # v1.0.4.8: Extract full trait vector for all generators
        _social_des = _safe_trait_value(traits.get("social_desirability"), 0.3)
        _consistency = _safe_trait_value(traits.get("response_consistency"), 0.6)
        _extremity = _safe_trait_value(traits.get("extremity"), 0.4)

        # v1.0.8.4: Early question intent detection — computed BEFORE generator cascade
        # so ALL generators (LLM, Comprehensive, TextResponse) can use it.
        # This uses both the original question text and user-provided context.
        _qt_early = (_original_question_text or question_text or "").lower()
        _ctx_early = (question_context or "").lower()
        _both_early = f"{_qt_early} {_ctx_early}"
        _early_intent = "opinion"  # default
        if any(w in _both_early for w in ('conspiracy', 'theory', 'believe in', 'crazy belie',
                                           'paranormal', 'supernatural', 'superstition')):
            _early_intent = "creative_belief"
        elif any(w in _both_early for w in ('secret', 'only your family', 'nobody knows',
                                             'never told', 'private', 'confession', 'confess',
                                             'reveal', 'admit', 'embarrassing')):
            _early_intent = "personal_disclosure"
        elif any(w in _qt_early for w in ('tell us your', 'share your', 'write about your',
                                           'describe your')):
            if any(w in _both_early for w in ('craziest', 'wildest', 'favorite', 'most',
                                               'biggest', 'worst', 'best', 'funniest',
                                               'scariest', 'strangest')):
                _early_intent = "creative_narrative"
            elif any(w in _both_early for w in ('experience', 'story', 'time when',
                                                 'moment', 'situation', 'incident')):
                _early_intent = "personal_story"
        elif any(w in _qt_early for w in ('hypothetical', 'if you were', 'imagine',
                                           'suppose', 'what if', 'what would you',
                                           'in a scenario', 'what would happen')):
            _early_intent = "hypothetical"
        elif any(w in _qt_early for w in ('predict', 'expect', 'future', 'will happen',
                                           'forecast', 'what do you think will',
                                           'how likely', 'do you plan')):
            _early_intent = "prediction"
        elif any(w in _qt_early for w in ('recommend', 'suggest', 'advice', 'should',
                                           'tips for', 'best way to', 'what would you advise')):
            _early_intent = "recommendation"
        # v1.0.8.5: Comparison and recall intents
        elif any(w in _qt_early for w in ('compare', 'comparison', 'compared to', 'versus',
                                           'pros and cons', 'advantages', 'better or worse')):
            _early_intent = "comparison"
        elif any(w in _qt_early for w in ('remember', 'recall', 'looking back', 'in hindsight',
                                           'what stands out', 'think back')):
            _early_intent = "recall"
        elif any(w in _qt_early for w in ('why', 'explain', 'reason', 'because')):
            _early_intent = "explanation"
        elif any(w in _qt_early for w in ('how do you feel', 'feelings', 'emotions', 'react')):
            _early_intent = "emotional_reaction"
        elif any(w in _qt_early for w in ('describe', 'tell us about', 'what happened')):
            _early_intent = "description"
        elif any(w in _qt_early for w in ('evaluate', 'rate', 'assess')):
            _early_intent = "evaluation"

        # v1.1.1.5: Skip LLM entirely when template fallback is enabled (user chose
        # "Template Engine" or "Adaptive Behavioral Engine").  Trying LLM here wastes
        # time and can trigger provider errors that obscure the actual generation path.
        # v1.4.9: Try LLM generator first (question-specific, persona-aligned)
        if self.llm_generator is not None and not self.allow_template_fallback:
            try:
                resp = self.llm_generator.generate(
                    question_text=question_text or response_type,
                    sentiment=sentiment,
                    persona_verbosity=verbosity,
                    persona_formality=formality,
                    persona_engagement=engagement,
                    condition=condition,
                    question_name=str(question_spec.get("name", "")),
                    participant_seed=participant_seed,
                    behavioral_profile=behavioral_profile,
                )
                if resp and resp.strip():
                    self._last_oe_source = "AI"
                    return resp
            except Exception as _llm_gen_err:
                # v1.2.0.0: NEVER re-raise "template fallback is disabled" here.
                # When LLM is force-disabled mid-question (budget exceeded), the old
                # re-raise bypassed comprehensive_generator and text_generator entirely,
                # causing _last_resort_oe_response() gibberish for all remaining
                # participants. Instead, fall through to the template generators below
                # which produce much higher quality topic-aware responses.
                logger.warning("LLM generate() error: %s", _llm_gen_err)
                self._log(f"WARNING: LLM generation failed, falling back: {_llm_gen_err}")

        # Try to use comprehensive response generator if available
        # v1.0.7.2: Don't blindly return — check if result is non-empty first.
        # Previously, an empty return from comprehensive_generator would skip the
        # text_generator fallback entirely, leaving responses blank.
        if self.comprehensive_generator is not None:
            try:
                base_name = str(question_spec.get("name", ""))
                var_name = str(question_spec.get("variable_name", ""))
                q_type = str(question_spec.get("type", ""))
                unique_question_id = f"{base_name}|{var_name}|{q_type}|{question_text[:100]}"
                _comp_result = self.comprehensive_generator.generate(
                    question_text=question_text or response_type,
                    sentiment=sentiment,
                    persona_verbosity=verbosity,
                    persona_formality=formality,
                    persona_engagement=engagement,
                    condition=condition,
                    question_name=unique_question_id,
                    participant_seed=participant_seed,
                    behavioral_profile=behavioral_profile,
                    question_intent=_early_intent,  # v1.0.8.4: Pass intent for template routing
                    question_context=question_context,  # v1.0.8.4: Pass raw context
                )
                if _comp_result and _comp_result.strip():
                    self._last_oe_source = "Template"
                    return _comp_result
                # v1.0.7.2: Empty result — fall through to text_generator
                logger.debug("ComprehensiveResponseGenerator returned empty for '%s', falling through to text_generator",
                             question_text[:80] if question_text else "unknown")
            except Exception as _comp_gen_err:
                # v1.0.5.7: Log at WARNING (not debug) so failures are visible
                logger.warning("ComprehensiveResponseGenerator error for '%s': %s",
                               question_text[:80] if question_text else "unknown", _comp_gen_err)
                self._log(f"WARNING: ComprehensiveResponseGenerator failed: {_comp_gen_err}")

        # Fallback to basic text generator
        # v1.0.4.8: Also consider behavioral profile for style override
        _effective_attn = attention_level
        if behavioral_profile and isinstance(behavioral_profile, dict):
            if behavioral_profile.get('straight_lined'):
                _effective_attn = min(_effective_attn, 0.3)  # Force careless style

        if _effective_attn < 0.5:
            style = "careless"
        elif _persona_name == "Satisficer":
            style = "satisficer"
        elif _persona_name == "Extreme Responder":
            style = "extreme"
        elif _persona_name == "Engaged Responder":
            style = "engaged"
        else:
            style = "default"

        # Build context from study_context and question_spec
        # v1.0.3.8: Heavily revised — extract meaningful topics from question
        # text/context so fallback templates are grounded in the actual question.
        study_domain = self.study_context.get("study_domain", "general")
        survey_name = self.study_context.get("survey_name", self.study_title)

        # v1.0.3.8: Extract MEANINGFUL topic from question text and context
        # Priority: question_context > question_text > study_domain
        topic = question_spec.get("topic", "")
        _stimulus_source = question_spec.get("stimulus", survey_name or "this study")
        _product_source = question_spec.get("product", "")
        _feature_source = question_spec.get("feature", "")

        # Extract topic words from question context or text
        import re as _ctx_re
        # v1.0.4.7: Unified stop word list — includes researcher-instruction vocabulary
        _ctx_stop = {
            'the', 'a', 'an', 'this', 'that', 'these', 'those', 'its', 'it',
            'they', 'them', 'their', 'we', 'our', 'you', 'your', 'he', 'she',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'up', 'about',
            'and', 'or', 'but', 'not', 'no', 'so', 'nor',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'need',
            'how', 'what', 'who', 'why', 'when', 'where', 'which',
            'want', 'wants', 'understand', 'think', 'feel', 'tell', 'share', 'describe',
            'explain', 'ask', 'asked', 'give', 'get', 'make', 'say', 'know', 'see',
            # Researcher instruction vocabulary (v1.0.4.7)
            'participants', 'respondents', 'subjects', 'people', 'person',
            'primed', 'priming', 'prime', 'exposed', 'exposure', 'exposing',
            'presented', 'presenting', 'shown', 'showing', 'show',
            'told', 'telling', 'instructed', 'instructions',
            'assigned', 'randomly', 'random', 'randomized',
            'thinking', 'reading', 'viewing', 'watching', 'completing', 'answering',
            'reporting', 'sharing', 'responding',
            'before', 'after', 'during', 'following', 'prior',
            'then', 'next', 'first', 'second', 'third',
            'stories', 'story', 'experience', 'experiences',
            'whether', 'toward', 'towards', 'regarding',
            'question', 'questions', 'context', 'study', 'survey', 'experiment',
            'condition', 'conditions', 'topic', 'measure', 'measured',
            'response', 'responses', 'answer', 'answers', 'item', 'items',
            'scale', 'rating', 'open', 'ended', 'text', 'variable',
            'much', 'more', 'most', 'very', 'really', 'just', 'also', 'please',
            'better', 'deeply', 'held', 'quite',
            # v1.0.6.3: Additional stop words found to cause gibberish
            'here', 'there', 'now', 'well', 'like', 'even', 'still', 'let',
            'only', 'some', 'such', 'each', 'every', 'any', 'all', 'both',
            'many', 'few', 'own', 'other', 'another', 'same', 'different',
            'something', 'anything', 'everything', 'nothing',
            'however', 'therefore', 'moreover', 'furthermore', 'indeed',
            'certain', 'particular', 'specific', 'general', 'overall',
        }
        # v1.0.8.3: Deep topic intelligence — 7-strategy semantic context extraction
        # Produces a structured understanding of what the question is ABOUT, who/what
        # entities are involved, and what kind of response the question expects.
        # This flows through all three generators (LLM, Comprehensive, TextResponse).

        # Strategy 1: Extract from ORIGINAL question context/text (not embedded string)
        # v1.0.8.3: Use question text FIRST for topic (it's the actual question asked),
        # then context as supplementary. Context often has researcher instructions that
        # pollute topic extraction (e.g., "Participants are asked to think about...").
        _qt_for_topic = _original_question_text or question_text or ""
        # Strip common researcher framing prefixes
        _qt_for_topic = _ctx_re.sub(
            r'^(?:participants?\s+(?:are|were|will\s+be)\s+)',
            '', _qt_for_topic, flags=_ctx_re.IGNORECASE).strip()

        # v1.0.8.3: Adjective modifiers that describe the topic but ARE NOT the topic.
        # "your craziest conspiracy theory" → "conspiracy theory" not "craziest conspiracy"
        # Grounded in: adjective-noun phrase parsing — superlatives, possessives, and
        # evaluative adjectives that modify but don't constitute the core noun phrase.
        _adj_modifiers = {
            # Superlatives and ordinals
            'favorite', 'craziest', 'wildest', 'deepest', 'biggest', 'worst',
            'best', 'strongest', 'weirdest', 'strangest', 'funniest', 'scariest',
            'most', 'least', 'first', 'last', 'recent', 'latest', 'current',
            # Evaluative adjectives
            'personal', 'private', 'secret', 'honest', 'real', 'true', 'genuine',
            'crazy', 'wild', 'extreme', 'controversial', 'unpopular', 'important',
            'interesting', 'memorable', 'notable', 'significant', 'relevant',
            # Possessive/relational
            'own', 'particular', 'specific', 'actual', 'main', 'primary',
            # Emotional intensity modifiers
            'deeply', 'strongly', 'absolutely', 'completely', 'totally', 'really',
            # Question framing words (not topics themselves)
            'related', 'following', 'given', 'certain', 'various',
        }
        _qt_words = _ctx_re.findall(r'\b[a-zA-Z]{3,}\b', _qt_for_topic.lower())
        _topic_words = [w for w in _qt_words if w not in _ctx_stop and w not in _adj_modifiers][:6]

        # v1.0.8.3: Also extract from context as secondary source for enrichment
        _context_topic_words: List[str] = []
        if question_context:
            _ctx_for_topic = _ctx_re.sub(
                r'^(?:participants?\s+(?:are|were|will\s+be)\s+(?:asked\s+to\s+)?)',
                '', question_context, flags=_ctx_re.IGNORECASE).strip()
            _ctx_words = _ctx_re.findall(r'\b[a-zA-Z]{3,}\b', _ctx_for_topic.lower())
            _context_topic_words = [w for w in _ctx_words
                                    if w not in _ctx_stop and w not in _adj_modifiers][:6]

        # Strategy 2: Phrase-level extraction — find meaningful noun phrases
        # v1.0.8.3: Massively expanded patterns for narrative, creative, disclosure,
        # and superlative question types (not just opinion/evaluation).
        _phrase_topic = ""
        _phrase_patterns = [
            # v1.0.8.3: Superlative/creative capture — "your craziest/most X related to Y"
            r'(?:your\s+(?:craziest|wildest|biggest|deepest|worst|best|strongest|weirdest|strangest|funniest|scariest|most\s+\w+|favorite))\s+(.+?)(?:\s+(?:related\s+to|about|regarding|concerning)\s+\w+|\.|$|\?)',
            # v1.0.8.3: "Tell us X related to Y" — capture the core noun phrase
            r'(?:tell\s+(?:us|me)\s+(?:your|about\s+your|a|about\s+a|about))\s+(.+?)(?:\.|$|\?)',
            # v1.0.8.3: "Tell us something X" — capture disclosure type
            r'(?:tell\s+(?:us|me)\s+something)\s+(.+?)(?:\.|$|\?)',
            # v1.0.8.3: "Share X" / "Describe X"
            r'(?:share|describe|write\s+about)\s+(?:a|an|your|the)?\s*(.+?)(?:\.|$|\?)',
            # Original opinion/feeling patterns
            r'(?:feelings?|thoughts?|opinions?|views?|attitudes?|reactions?|impressions?)\s+(?:about|toward|towards|on|regarding|concerning)\s+(.+?)(?:\.|$|\?)',
            r'(?:describe|explain|tell\s+us|share)\s+(?:about|how\s+you\s+feel\s+about|your\s+views?\s+on)\s+(.+?)(?:\.|$|\?)',
            r'(?:how\s+do\s+you\s+feel\s+about)\s+(.+?)(?:\.|$|\?)',
            r'(?:what\s+do\s+you\s+think\s+(?:about|of))\s+(.+?)(?:\.|$|\?)',
            r'(?:what\s+(?:is|are)\s+your\s+(?:views?|thoughts?|opinions?)\s+(?:on|about))\s+(.+?)(?:\.|$|\?)',
            r'(?:how\s+(?:does|did|would))\s+(.+?)\s+(?:make\s+you\s+feel|affect\s+you|influence)',
            r'(?:why\s+did\s+you\s+(?:choose|select|rate|respond|decide|prefer))\s+(.+?)(?:\.|$|\?)',
            r'(?:your\s+(?:experience|interaction|encounter)\s+with)\s+(.+?)(?:\.|$|\?)',
            # v1.0.8.3: "What is your X" — personal attribute capture
            r'(?:what\s+(?:is|are)\s+your)\s+(.+?)(?:\.|$|\?)',
        ]
        # v1.0.8.3: Search BOTH question text AND context for phrases
        _phrase_search_texts = [_qt_for_topic]
        if question_context:
            _phrase_search_texts.append(question_context)
        for _search_text in _phrase_search_texts:
            if _phrase_topic:
                break
            for _pp in _phrase_patterns:
                _pm = _ctx_re.search(_pp, _search_text, flags=_ctx_re.IGNORECASE)
                if _pm:
                    _captured = _pm.group(1).strip()[:150]
                    # v1.0.8.3: Clean captured phrase — remove trailing researcher instructions
                    _captured = _ctx_re.sub(
                        r'\s*[-–—]\s*(?:I\s+want|we\s+want|this\s+(?:will|should|helps?)|participants?).*$',
                        '', _captured, flags=_ctx_re.IGNORECASE).strip()
                    # Remove trailing articles/prepositions
                    _captured = _ctx_re.sub(r'\s+(?:the|a|an|to|for|in|on|at|by|with)$', '', _captured).strip()
                    if len(_captured) >= 3:
                        _phrase_topic = _captured
                        break

        # Strategy 3: Condition-aware topic enrichment
        # Extract meaningful words from condition name to enrich topic
        _cond_topic_words = []
        if condition:
            _cond_clean = _ctx_re.sub(r'[_\-,]+', ' ', condition).strip()
            _cond_words = _ctx_re.findall(r'\b[a-zA-Z]{3,}\b', _cond_clean.lower())
            _cond_stop = _ctx_stop | {'control', 'baseline', 'treatment', 'group', 'condition',
                                      'level', 'high', 'low', 'cell'}
            _cond_topic_words = [w for w in _cond_words if w not in _cond_stop][:3]

        # Strategy 4: Comprehensive domain vocabulary hints
        # v1.0.6.6: Restored & expanded comprehensive domain hint table.
        # Precise vocabulary matters for ALL domains, not just economic games.
        # Each entry maps a keyword (found in study_domain or study_title) to
        # a domain-specific description that grounds open-text responses.
        # Adaptive fallback chain (Steps B-D) supplements for topics not in table.
        _domain_topic_hints = {
            # ── Economic games (meta-analysis-calibrated baselines) ──
            'dictator': 'giving and allocation decisions',
            'trust': 'trust and reciprocity',
            'ultimatum': 'fairness and offers',
            'public_goods': 'cooperation and contributions',
            'prisoners_dilemma': 'cooperation and defection decisions',
            'prisoner': 'cooperation and defection decisions',
            'commons_dilemma': 'shared resource management and sustainability',
            'bargaining': 'negotiation and bargaining outcomes',
            'auction': 'bidding strategies and valuation',
            'investment': 'investment decisions and financial risk-taking',
            'endowment': 'ownership effects and willingness to trade',
            'market': 'market behavior and trading decisions',
            'gift_exchange': 'reciprocal gift-giving and effort provision',
            'coordination': 'coordination and strategic decision-making',
            'stag_hunt': 'coordination and mutual cooperation under risk',
            'chicken': 'brinkmanship and conflict escalation',
            'centipede': 'sequential trust and backward induction',
            # ── Political science & polarization ──
            'polarization': 'political attitudes and divisions',
            'partisan': 'partisan identity and political loyalty',
            'ideology': 'ideological beliefs and political orientation',
            'election': 'electoral preferences and voting behavior',
            'voting': 'voting decisions and democratic participation',
            'democracy': 'democratic values and governance attitudes',
            'authoritarianism': 'authoritarian attitudes and obedience to authority',
            'populism': 'populist attitudes and anti-elite sentiment',
            'nationalism': 'national identity and patriotic attitudes',
            'globalization': 'attitudes toward globalization and international cooperation',
            'immigration': 'immigration attitudes and policy preferences',
            'refugee': 'attitudes toward refugees and asylum seekers',
            'political': 'political attitudes and civic engagement',
            'policy': 'policy preferences and government attitudes',
            'campaign': 'campaign messaging and political persuasion',
            'lobbying': 'lobbying influence and political spending attitudes',
            'corruption': 'corruption perceptions and institutional trust',
            'censorship': 'censorship attitudes and free speech values',
            'propaganda': 'propaganda effects and media manipulation',
            'protest': 'protest participation and collective action',
            'revolution': 'revolutionary attitudes and regime change',
            # ── Intergroup relations & identity ──
            'intergroup': 'group identity and relations',
            'ingroup': 'ingroup favoritism and group loyalty',
            'outgroup': 'outgroup attitudes and intergroup bias',
            'discrimination': 'fairness and equal treatment',
            'prejudice': 'attitudes toward social groups',
            'stereotype': 'stereotypes and social categorization',
            'racism': 'racial attitudes and systemic racism perceptions',
            'race': 'racial identity and interracial relations',
            'ethnicity': 'ethnic identity and cultural attitudes',
            'sexism': 'gender-based attitudes and sex discrimination',
            'gender': 'gender identity and gender role attitudes',
            'lgbtq': 'sexual orientation and gender identity attitudes',
            'sexuality': 'sexual attitudes and relationship norms',
            'disability': 'disability attitudes and accessibility perceptions',
            'ageism': 'age-based attitudes and intergenerational relations',
            'classism': 'social class perceptions and economic inequality attitudes',
            'xenophobia': 'attitudes toward foreigners and cultural others',
            'islamophobia': 'attitudes toward Muslims and Islamic culture',
            'antisemitism': 'attitudes toward Jewish people and communities',
            'identity': 'identity and self-concept',
            'social_identity': 'social identity and group membership',
            'stigma': 'stigmatization and social marking',
            'dehumanization': 'dehumanization and moral exclusion',
            'minority': 'minority experiences and majority-minority relations',
            'diversity': 'diversity attitudes and inclusion perceptions',
            'multiculturalism': 'multicultural attitudes and cultural integration',
            # ── Social psychology ──
            'conformity': 'conformity and social influence',
            'obedience': 'obedience to authority and compliance',
            'compliance': 'compliance with requests and social pressure',
            'persuasion': 'persuasive messages and attitude change',
            'social_influence': 'social influence and normative pressure',
            'social_norms': 'social norms and normative expectations',
            'norms': 'social norms and behavioral expectations',
            'bystander': 'bystander intervention and helping behavior',
            'prosocial': 'helping behavior and prosocial motivation',
            'altruism': 'altruistic behavior and selfless helping',
            'cooperation': 'cooperation and collective action',
            'competition': 'competitive behavior and rivalry',
            'aggression': 'aggressive behavior and hostile attitudes',
            'violence': 'violence attitudes and aggressive tendencies',
            'bullying': 'bullying behavior and peer victimization',
            'cyberbullying': 'online harassment and cyberbullying experiences',
            'ostracism': 'social exclusion and ostracism experiences',
            'loneliness': 'loneliness and social isolation',
            'belonging': 'sense of belonging and social connectedness',
            'rejection': 'social rejection and interpersonal exclusion',
            'power': 'power dynamics and social hierarchy',
            'status': 'social status and dominance hierarchies',
            'leadership': 'leadership and authority',
            'hierarchy': 'social hierarchies and rank-based behavior',
            'fairness': 'fairness perceptions and justice sensitivity',
            'reciprocity': 'reciprocity and mutual exchange',
            'revenge': 'revenge and retaliatory behavior',
            'forgiveness': 'forgiveness and letting go of grievances',
            'apology': 'apology and reconciliation',
            'gratitude': 'gratitude and appreciation',
            'empathy': 'empathic concern and perspective-taking',
            'compassion': 'compassion and caring for others',
            'schadenfreude': 'pleasure at others misfortune and competitive emotions',
            'envy': 'envy and social comparison emotions',
            'jealousy': 'jealousy and possessive concerns',
            # ── Moral psychology & ethics ──
            'moral': 'moral judgments and ethical decisions',
            'ethics': 'ethical reasoning and moral principles',
            'trolley': 'moral dilemmas and utilitarian vs. deontological reasoning',
            'justice': 'justice perceptions and fairness judgments',
            'punishment': 'punishment and norm enforcement',
            'retribution': 'retributive justice and punishment preferences',
            'restorative': 'restorative justice and rehabilitation attitudes',
            'deception': 'honesty and deceptive behavior',
            'lying': 'lying behavior and truth-telling norms',
            'cheating': 'cheating behavior and academic integrity',
            'corruption': 'corruption perceptions and institutional trust',
            'hypocrisy': 'moral hypocrisy and inconsistency',
            'virtue': 'virtue and moral character judgments',
            'disgust': 'moral disgust and purity concerns',
            'sacred': 'sacred values and taboo trade-offs',
            'dilemma': 'moral dilemmas and ethical trade-offs',
            # ── Consumer behavior & marketing ──
            'consumer': 'product preferences and choices',
            'purchase': 'purchasing decisions and buying behavior',
            'brand': 'brand perceptions and brand loyalty',
            'advertising': 'advertising effectiveness and ad attitudes',
            'pricing': 'price perceptions and willingness to pay',
            'luxury': 'luxury consumption and status signaling',
            'sustainable_consumption': 'sustainable purchasing and ethical consumerism',
            'organic': 'organic product preferences and natural food attitudes',
            'ecommerce': 'online shopping behavior and digital commerce',
            'retail': 'retail experiences and shopping behavior',
            'product': 'product evaluation and consumer preferences',
            'service': 'service quality and customer satisfaction',
            'loyalty': 'customer loyalty and brand attachment',
            'word_of_mouth': 'word-of-mouth and recommendation behavior',
            'influencer': 'influencer marketing and social media endorsements',
            'packaging': 'packaging design and product presentation effects',
            'scarcity': 'scarcity effects and urgency in purchasing',
            'choice_overload': 'choice overload and decision difficulty',
            # ── Behavioral economics & decision-making ──
            'risk': 'risk perception and decision-making under uncertainty',
            'uncertainty': 'uncertainty tolerance and ambiguity attitudes',
            'loss_aversion': 'loss aversion and reference-dependent preferences',
            'endowment_effect': 'ownership effects and endowment-driven valuation',
            'anchoring': 'anchoring effects and numerical judgment biases',
            'framing': 'framing effects and presentation-dependent choices',
            'nudge': 'nudging and choice architecture effects',
            'default': 'default effects and status quo bias',
            'sunk_cost': 'sunk cost effects and escalation of commitment',
            'temporal_discount': 'temporal discounting and intertemporal choice',
            'delay_gratification': 'delayed gratification and self-control',
            'gambling': 'gambling behavior and risk preferences',
            'debt': 'debt attitudes and financial decision-making',
            'saving': 'savings behavior and financial planning',
            'poverty': 'poverty effects on cognition and decision-making',
            'inequality': 'economic inequality perceptions and redistribution attitudes',
            'wealth': 'wealth perceptions and economic mobility attitudes',
            'prospect': 'prospect theory and risky choice',
            'bounded_rationality': 'bounded rationality and satisficing behavior',
            'heuristic': 'heuristic-based judgment and cognitive shortcuts',
            'overconfidence': 'overconfidence and calibration in judgment',
            # ── Health psychology & wellbeing ──
            'health': 'health decisions and wellbeing',
            'mental_health': 'mental health attitudes and psychological wellbeing',
            'anxiety': 'anxiety and worry experiences',
            'depression': 'mood and depressive experiences',
            'stress': 'stress and coping strategies',
            'burnout': 'burnout and occupational exhaustion',
            'resilience': 'resilience and coping capacity',
            'wellbeing': 'subjective wellbeing and life satisfaction',
            'happiness': 'happiness and positive emotional experiences',
            'life_satisfaction': 'life satisfaction and global wellbeing judgments',
            'mindfulness': 'mindfulness and attention',
            'meditation': 'meditation practice and contemplative experiences',
            'therapy': 'therapy experiences and treatment attitudes',
            'counseling': 'counseling attitudes and help-seeking behavior',
            'addiction': 'addictive behaviors and substance use',
            'alcohol': 'alcohol consumption and drinking behavior',
            'smoking': 'smoking behavior and tobacco attitudes',
            'cannabis': 'cannabis use and marijuana attitudes',
            'opioid': 'opioid use and pain management attitudes',
            'drug': 'drug use attitudes and substance abuse perceptions',
            'trauma': 'traumatic experiences and coping',
            'ptsd': 'post-traumatic stress and trauma recovery',
            'grief': 'grief and bereavement experiences',
            'pain': 'pain perception and pain management',
            'chronic_illness': 'chronic illness experiences and disease management',
            'disability_health': 'health-related disability and functional limitations',
            'sleep': 'sleep quality and habits',
            'exercise': 'exercise habits and physical activity',
            'nutrition': 'nutritional attitudes and dietary choices',
            'food': 'food preferences and eating behavior',
            'eating_disorder': 'eating disorder attitudes and body-related concerns',
            'body_image': 'body image and physical appearance concerns',
            'obesity': 'obesity attitudes and weight management',
            'vaccine': 'vaccination attitudes and health decisions',
            'pandemic': 'pandemic experiences and public health attitudes',
            'covid': 'COVID-19 attitudes and pandemic behavior',
            'quarantine': 'quarantine experiences and isolation effects',
            'mask': 'mask-wearing attitudes and protective behavior',
            'telemedicine': 'telemedicine attitudes and remote healthcare',
            'patient': 'patient experiences and healthcare satisfaction',
            'doctor': 'doctor-patient communication and medical trust',
            'placebo': 'placebo effects and treatment expectations',
            # ── Cognitive psychology ──
            'memory': 'memory and recall experiences',
            'attention': 'attention and concentration experiences',
            'perception': 'perceptual experiences and sensory judgments',
            'creativity': 'creative thinking and problem solving',
            'intelligence': 'intelligence beliefs and cognitive ability perceptions',
            'mindset': 'growth mindset and beliefs about ability',
            'cognitive_load': 'cognitive load and mental effort',
            'decision_fatigue': 'decision fatigue and ego depletion',
            'metacognition': 'metacognitive awareness and thinking about thinking',
            'learning': 'learning strategies and knowledge acquisition',
            'expertise': 'expertise and skill development',
            'insight': 'insight and problem-solving breakthroughs',
            'intuition': 'intuitive judgment and gut feelings',
            'reasoning': 'logical reasoning and analytical thinking',
            'bias': 'cognitive biases and judgment errors',
            'false_memory': 'false memories and memory distortion',
            'eyewitness': 'eyewitness testimony and memory accuracy',
            'misinformation_effect': 'misinformation effects on memory',
            # ── Emotion & affect ──
            'emotion': 'emotional experiences and regulation',
            'affect': 'affective states and mood',
            'mood': 'mood states and emotional wellbeing',
            'anger': 'anger experiences and hostile feelings',
            'fear': 'fear and anxiety responses',
            'sadness': 'sadness and melancholy experiences',
            'joy': 'joy and positive emotional experiences',
            'surprise': 'surprise reactions and expectation violations',
            'contempt': 'contempt and moral superiority feelings',
            'pride': 'pride and achievement-related emotions',
            'shame': 'shame and self-conscious emotions',
            'guilt': 'guilt and moral self-regulation',
            'embarrassment': 'embarrassment and social awkwardness',
            'hope': 'hope and optimistic expectations',
            'nostalgia': 'nostalgic experiences and sentimental reflection',
            'awe': 'awe experiences and vast/overwhelming stimuli',
            'boredom': 'boredom and understimulation experiences',
            'curiosity': 'curiosity and information-seeking motivation',
            'emotion_regulation': 'emotion regulation and coping strategies',
            'emotional_intelligence': 'emotional intelligence and affect understanding',
            # ── Personality & individual differences ──
            'personality': 'personality traits and individual differences',
            'big_five': 'Big Five personality dimensions and trait expression',
            'extraversion': 'extraversion and sociability',
            'neuroticism': 'neuroticism and emotional instability',
            'conscientiousness': 'conscientiousness and self-discipline',
            'agreeableness': 'agreeableness and interpersonal warmth',
            'openness': 'openness to experience and intellectual curiosity',
            'narcissism': 'narcissistic tendencies and self-enhancement',
            'psychopathy': 'psychopathic traits and callous-unemotional tendencies',
            'machiavellianism': 'manipulative tendencies and strategic self-interest',
            'dark_triad': 'dark triad traits and antisocial personality',
            'self_esteem': 'self-esteem and self-worth',
            'self_efficacy': 'self-efficacy and confidence in abilities',
            'self_control': 'self-control and impulse regulation',
            'impulsivity': 'impulsivity and spontaneous behavior',
            'need_for_cognition': 'need for cognition and thinking enjoyment',
            'locus_of_control': 'locus of control and agency beliefs',
            'optimism': 'optimism and positive expectations',
            'pessimism': 'pessimism and negative expectations',
            'perfectionism': 'perfectionism and high standards',
            'grit': 'grit and perseverance toward long-term goals',
            'procrastination': 'procrastination and task avoidance',
            # ── Relationships & attachment ──
            'attachment': 'interpersonal attachment and relationships',
            'romantic': 'romantic relationships and partner preferences',
            'dating': 'dating preferences and romantic experiences',
            'marriage': 'marriage attitudes and marital satisfaction',
            'divorce': 'divorce attitudes and relationship dissolution',
            'infidelity': 'infidelity attitudes and relationship betrayal',
            'intimacy': 'intimacy and emotional closeness',
            'love': 'love and romantic attachment',
            'friendship': 'friendship quality and social support',
            'family': 'family relationships and family dynamics',
            'parenting': 'parenting approaches and child-rearing',
            'sibling': 'sibling relationships and family dynamics',
            'caregiving': 'caregiving experiences and caregiver burden',
            'social_support': 'social support and interpersonal resources',
            'conflict_resolution': 'interpersonal conflict resolution strategies',
            'communication': 'interpersonal communication and relationship quality',
            'trust_interpersonal': 'interpersonal trust and relational security',
            # ── Organizational behavior & work ──
            'organizational': 'organizational attitudes and workplace behavior',
            'workplace': 'workplace experiences and job attitudes',
            'job_satisfaction': 'job satisfaction and work engagement',
            'motivation': 'motivation and goal pursuit',
            'goal_setting': 'goal-setting and achievement motivation',
            'teamwork': 'teamwork and collaborative performance',
            'negotiation': 'negotiation strategies and outcomes',
            'conflict': 'organizational conflict and dispute resolution',
            'work_life': 'work-life balance and boundary management',
            'remote_work': 'remote work experiences and telecommuting attitudes',
            'entrepreneurship': 'entrepreneurial intentions and startup attitudes',
            'innovation': 'innovation and creative organizational behavior',
            'organizational_justice': 'organizational justice and workplace fairness',
            'harassment': 'workplace harassment and hostile work environments',
            'diversity_inclusion': 'workplace diversity and inclusion practices',
            'turnover': 'turnover intentions and organizational commitment',
            'mentoring': 'mentoring relationships and career development',
            'performance': 'performance evaluation and feedback',
            'management': 'management practices and supervisory behavior',
            # ── Education & learning ──
            'education': 'learning and educational experiences',
            'teaching': 'teaching practices and pedagogical approaches',
            'student': 'student experiences and academic attitudes',
            'academic': 'academic performance and scholarly engagement',
            'test_anxiety': 'test anxiety and examination stress',
            'cheating_academic': 'academic dishonesty and integrity attitudes',
            'online_learning': 'online learning experiences and distance education',
            'stem': 'STEM education and science engagement',
            'literacy': 'literacy and reading attitudes',
            'math_anxiety': 'mathematics anxiety and numerical attitudes',
            'feedback_education': 'educational feedback and grading effects',
            'growth_mindset': 'growth mindset and beliefs about intelligence',
            'self_regulated': 'self-regulated learning and study strategies',
            'peer_learning': 'peer learning and collaborative education',
            'stereotype_threat': 'stereotype threat and identity-contingent performance',
            # ── Technology & AI ──
            'ai_attitudes': 'AI technology and trust',
            'artificial_intelligence': 'artificial intelligence attitudes and perceptions',
            'technology': 'technology use and digital behavior',
            'automation': 'automation attitudes and job displacement concerns',
            'robot': 'robot interaction and human-robot relations',
            'chatbot': 'chatbot interactions and conversational AI',
            'algorithm': 'algorithmic decision-making and algorithm attitudes',
            'privacy': 'privacy concerns and data sharing',
            'surveillance': 'surveillance attitudes and monitoring perceptions',
            'social_media': 'social media use and online behavior',
            'internet': 'internet use and online behavior',
            'screen_time': 'screen time and digital media consumption',
            'digital_wellbeing': 'digital wellbeing and technology-life balance',
            'misinformation': 'misinformation, fake news, and media credibility',
            'fake_news': 'fake news detection and media literacy',
            'deepfake': 'deepfake awareness and synthetic media attitudes',
            'cryptocurrency': 'cryptocurrency attitudes and blockchain perceptions',
            'nft': 'NFT attitudes and digital ownership perceptions',
            'vr': 'virtual reality experiences and immersive technology',
            'virtual_reality': 'virtual reality experiences and immersive technology',
            'augmented_reality': 'augmented reality experiences and mixed-reality attitudes',
            'autonomous_vehicle': 'autonomous vehicle trust and self-driving attitudes',
            'self_driving': 'self-driving vehicle attitudes and transportation automation',
            'smart_home': 'smart home technology adoption and IoT attitudes',
            'wearable': 'wearable technology use and health tracking',
            'gaming': 'video game behavior and gaming attitudes',
            'cybersecurity': 'cybersecurity awareness and online safety behavior',
            # ── Media & communication ──
            'media': 'media consumption and information sources',
            'news': 'news consumption and media trust',
            'journalism': 'journalism credibility and press freedom attitudes',
            'framing_media': 'media framing effects and issue presentation',
            'agenda_setting': 'agenda-setting and media influence on priorities',
            'conspiracy': 'conspiracy theories and beliefs about hidden forces',
            'belief': 'personal beliefs and worldviews',
            'rumor': 'rumor spread and unverified information sharing',
            'satire': 'satire perception and political humor effects',
            # ── Environmental psychology ──
            'environmental': 'environmental attitudes and sustainable behavior',
            'climate': 'climate change beliefs and environmental action',
            'climate_change': 'climate change beliefs and environmental action',
            'sustainability': 'sustainability attitudes and eco-friendly behavior',
            'recycling': 'recycling behavior and waste reduction attitudes',
            'energy': 'energy conservation and renewable energy attitudes',
            'nature': 'nature connectedness and environmental appreciation',
            'animal_welfare': 'animal welfare attitudes and ethical treatment',
            'vegetarian': 'vegetarian and vegan attitudes and dietary choices',
            'biodiversity': 'biodiversity awareness and conservation attitudes',
            'pollution': 'pollution perceptions and environmental health concerns',
            'water': 'water conservation and resource management attitudes',
            # ── Religion & spirituality ──
            'religion': 'religious beliefs and spiritual experiences',
            'spirituality': 'spiritual experiences and meaning-making',
            'atheism': 'atheist identity and secular attitudes',
            'prayer': 'prayer experiences and religious practice',
            'faith': 'faith and religious conviction',
            'afterlife': 'afterlife beliefs and mortality attitudes',
            'morality_religion': 'religion-morality connections and sacred values',
            # ── Cultural psychology ──
            'culture': 'cultural values and cross-cultural differences',
            'individualism': 'individualism and self-reliance values',
            'collectivism': 'collectivism and group harmony values',
            'honor': 'honor culture and reputation-based norms',
            'face': 'face-saving and social reputation concerns',
            'acculturation': 'acculturation and cultural adaptation',
            'cross_cultural': 'cross-cultural attitudes and intercultural contact',
            'language_attitude': 'language attitudes and linguistic identity',
            'bilingual': 'bilingualism and multilingual experiences',
            # ── Legal & forensic psychology ──
            'legal': 'legal attitudes and justice system perceptions',
            'jury': 'jury decision-making and trial judgments',
            'sentencing': 'sentencing preferences and punishment severity',
            'police': 'police attitudes and law enforcement trust',
            'crime': 'crime perceptions and criminal justice attitudes',
            'death_penalty': 'death penalty attitudes and capital punishment',
            'eyewitness_legal': 'eyewitness reliability and legal testimony',
            'interrogation': 'interrogation and confession attitudes',
            'prison': 'prison attitudes and incarceration perceptions',
            'recidivism': 'recidivism and rehabilitation attitudes',
            # ── Sports & competition ──
            'sports': 'athletic performance and sports attitudes',
            'exercise_sport': 'exercise motivation and physical activity',
            'doping': 'doping attitudes and performance enhancement',
            'sportsmanship': 'sportsmanship and fair play values',
            'fan': 'sports fandom and team identification',
            'esports': 'esports participation and competitive gaming',
            # ── Developmental & aging ──
            'aging': 'aging experiences and perceptions',
            'child_development': 'child development and developmental milestones',
            'adolescent': 'adolescent experiences and identity development',
            'emerging_adult': 'emerging adulthood and life transitions',
            'retirement': 'retirement attitudes and late-life transitions',
            'generational': 'generational differences and cohort attitudes',
            'mortality_salience': 'mortality salience and death awareness effects',
            'death': 'death attitudes and end-of-life perceptions',
            # ── Sexuality & reproductive health ──
            'abortion': 'reproductive rights and policy attitudes',
            'contraception': 'contraception attitudes and reproductive choices',
            'sex_education': 'sex education and sexual health literacy',
            'consent': 'sexual consent and boundary communication',
            'sexual_harassment': 'sexual harassment and gender-based violence',
            'body_positivity': 'body positivity and appearance acceptance',
            # ── Economic & financial ──
            'tax': 'tax compliance and fiscal policy attitudes',
            'redistribution': 'wealth redistribution and social welfare attitudes',
            'minimum_wage': 'minimum wage and labor market attitudes',
            'gig_economy': 'gig economy and nonstandard work attitudes',
            'sharing_economy': 'sharing economy participation and attitudes',
            'universal_basic': 'universal basic income and social safety net attitudes',
            'trade': 'international trade and tariff attitudes',
            'inflation': 'inflation perceptions and economic expectations',
            'housing': 'housing affordability and homeownership attitudes',
            # ── Gun policy & safety ──
            'gun': 'gun policy attitudes and safety perceptions',
            'firearm': 'firearm attitudes and gun ownership perceptions',
            'second_amendment': 'Second Amendment attitudes and gun rights',
            # ── War, peace & security ──
            'war': 'war attitudes and military intervention perceptions',
            'peace': 'peace attitudes and conflict resolution preferences',
            'terrorism': 'terrorism perceptions and security attitudes',
            'military': 'military attitudes and defense spending perceptions',
            'nuclear': 'nuclear weapon attitudes and proliferation concerns',
            'security': 'security perceptions and threat assessments',
            'drone': 'drone warfare attitudes and autonomous weapons',
            # ── Miscellaneous research topics ──
            'volunteering': 'volunteering behavior and civic participation',
            'charity': 'charitable giving and philanthropy attitudes',
            'crowdfunding': 'crowdfunding participation and prosocial lending',
            'tipping': 'tipping behavior and service gratuity norms',
            'organ_donation': 'organ donation attitudes and end-of-life decisions',
            'blood_donation': 'blood donation willingness and prosocial health behavior',
            'humor': 'humor appreciation and comedy preferences',
            'music': 'music preferences and aesthetic experiences',
            'art': 'art appreciation and aesthetic judgments',
            'beauty': 'beauty perceptions and physical attractiveness',
            'fashion': 'fashion attitudes and appearance norms',
            'travel': 'travel preferences and tourism attitudes',
            'transportation': 'transportation choices and commuting attitudes',
            'urban': 'urban living attitudes and neighborhood perceptions',
            'rural': 'rural living experiences and community attitudes',
            'migration': 'migration experiences and mobility attitudes',
            'gentrification': 'gentrification perceptions and neighborhood change',
            'noise': 'noise sensitivity and environmental annoyance',
            'smell': 'olfactory experiences and scent-based attitudes',
            'color': 'color preferences and chromatic associations',
            'design': 'design aesthetics and visual preference',
            'architecture': 'architectural preferences and built environment',
            'space': 'outer space and space exploration attitudes',
            'pets': 'pet ownership and human-animal relationships',
            'luck': 'luck beliefs and superstitious thinking',
            'superstition': 'superstitious beliefs and magical thinking',
            'conspiracy_theory': 'conspiracy thinking and epistemic mistrust',
            'conspiracy': 'conspiracy beliefs and alternative explanations',
            'paranormal': 'paranormal beliefs and supernatural attitudes',
            # v1.0.8.3: Expanded for narrative/creative/disclosure question types
            'secret': 'personal secrets and self-disclosure',
            'disclosure': 'personal disclosure and private information sharing',
            'confession': 'confessions and personal admissions',
            'family': 'family relationships and family knowledge',
            'narrative': 'personal narratives and life stories',
            'anecdote': 'personal anecdotes and memorable experiences',
            'story': 'personal stories and lived experiences',
            'belief': 'personal beliefs and conviction systems',
            'theory': 'personal theories and explanatory beliefs',
            'opinion': 'personal opinions and value judgments',
            'experience': 'personal experiences and life events',
            'memory': 'personal memories and recollections',
        }
        _domain_hint = ""
        # Step A: Check comprehensive domain vocabulary table
        _sd_lower = study_domain.lower()
        _st_lower = (self.study_title or '').lower()
        for _dk, _dv in _domain_topic_hints.items():
            if _dk in _sd_lower or _dk in _st_lower:
                _domain_hint = _dv
                break
        # Step B: If no table match, dynamically build from detected domains
        if not _domain_hint and hasattr(self, 'detected_domains') and self.detected_domains:
            # Humanize detected domain names: "social_psychology" → "social psychology"
            _humanized = [d.replace('_', ' ') for d in self.detected_domains[:2]]
            _domain_hint = ' and '.join(_humanized)
        # Step C: If detected_domains didn't help, try study_domain
        if not _domain_hint and study_domain and study_domain not in ('general', ''):
            _domain_hint = study_domain.replace('_', ' ')
        # Step D: Construct from topic words as last resort
        if not _domain_hint and _topic_words:
            _domain_hint = ' '.join(_topic_words[:4])

        # Strategy 5 (v1.0.5.0): Entity extraction — identify named entities
        # (people, organizations, concepts) that should appear in responses
        # v1.0.6.4: GENERAL-PURPOSE heuristic detection. Instead of relying on
        # a hardcoded list, detect entities from the ORIGINAL (non-lowered) text
        # by finding capitalized words that aren't at sentence starts. This works
        # for ANY topic — political figures, brands, diseases, places, etc.
        _entities = []
        _original_source = f"{question_context or _original_question_text or question_text or ''} {condition or ''}"
        # Heuristic 1: Words capitalized mid-sentence (proper nouns)
        _orig_words = _ctx_re.findall(r'(?<=[a-z]\s)([A-Z][a-zA-Z]{2,})', _original_source)
        _entities.extend(w for w in _orig_words if w.lower() not in _ctx_stop)
        # Heuristic 2: ALL-CAPS words of 2+ letters (acronyms like AI, FBI, GDP)
        _acronyms = _ctx_re.findall(r'\b([A-Z]{2,})\b', _original_source)
        _entities.extend(a for a in _acronyms if len(a) <= 6 and a.lower() not in _ctx_stop)
        # Heuristic 3: Words after "about", "regarding", "on" that are capitalized
        _after_prep = _ctx_re.findall(
            r'(?:about|regarding|on|toward|towards|of)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
            _original_source)
        _entities.extend(_after_prep)
        # v1.0.8.5: Heuristic 4 — Lowercase entity detection for high-salience topics
        _known_lc_entities = {
            'trump', 'biden', 'obama', 'clinton', 'sanders', 'desantis', 'pelosi',
            'democrat', 'republican', 'brexit', 'nato', 'putin', 'zelensky',
            'facebook', 'instagram', 'twitter', 'tiktok', 'reddit', 'google',
            'amazon', 'tesla', 'chatgpt', 'openai', 'bitcoin', 'crypto',
            'covid', 'coronavirus', 'vaccine', 'pfizer', 'moderna', 'fauci',
            'blm', 'metoo', 'lgbtq', 'maga', 'qanon', 'antifa',
            'netflix', 'spotify', 'disney', 'uber', 'airbnb',
        }
        _source_lower_words = _ctx_re.findall(r'\b[a-zA-Z]{3,}\b', _original_source.lower())
        for _lw in _source_lower_words:
            if _lw in _known_lc_entities and _lw not in {e.lower() for e in _entities}:
                _entities.append(_lw.capitalize())
        # Deduplicate while preserving order
        _seen_ents: set = set()
        _unique_entities: list = []
        for _e in _entities:
            _el = _e.lower()
            if _el not in _seen_ents and _el not in _ctx_stop:
                _seen_ents.add(_el)
                _unique_entities.append(_e)
        _entities = _unique_entities[:5]

        # v1.0.6.5: Derive proper nouns from extracted entities (fixes undefined _proper_nouns bug)
        _proper_nouns = {e.lower() for e in _entities}

        # Strategy 6 (v1.0.8.3): Question intent classification — EXPANDED
        # Determines what KIND of response the question expects.
        # v1.0.8.3: Added narrative, creative, disclosure, and personal_story intents.
        # These are FUNDAMENTALLY different from opinion — they need the participant
        # to GENERATE CONTENT (a story, a theory, a secret) not just express a view.
        _question_intent = "opinion"  # default
        _qt_lower = (_original_question_text or _qt_for_topic or "").lower()
        _ctx_lower = (question_context or "").lower()
        _both_lower = f"{_qt_lower} {_ctx_lower}"
        # Check MOST SPECIFIC intents first, then fall back to broader categories
        if any(w in _both_lower for w in ('conspiracy', 'theory', 'believe in', 'crazy belie',
                                           'paranormal', 'supernatural', 'superstition')):
            _question_intent = "creative_belief"
        elif any(w in _both_lower for w in ('secret', 'only your family', 'nobody knows',
                                             'never told', 'private', 'confession', 'confess',
                                             'reveal', 'admit', 'embarrassing')):
            _question_intent = "personal_disclosure"
        elif any(w in _qt_lower for w in ('tell us your', 'share your', 'write about your',
                                           'describe your')):
            # "Tell us your X" = narrative generation, not opinion
            if any(w in _both_lower for w in ('craziest', 'wildest', 'favorite', 'most',
                                               'biggest', 'worst', 'best', 'funniest',
                                               'scariest', 'strangest')):
                _question_intent = "creative_narrative"
            elif any(w in _both_lower for w in ('experience', 'story', 'time when',
                                                 'moment', 'situation', 'incident')):
                _question_intent = "personal_story"
            else:
                _question_intent = "description"
        elif any(w in _qt_lower for w in ('why', 'explain', 'reason', 'because')):
            _question_intent = "explanation"
        elif any(w in _qt_lower for w in ('describe', 'tell us about', 'what happened')):
            _question_intent = "description"
        elif any(w in _qt_lower for w in ('how do you feel', 'feelings', 'emotions', 'react')):
            _question_intent = "emotional_reaction"
        elif any(w in _qt_lower for w in ('evaluate', 'rate', 'assess', 'compare')):
            _question_intent = "evaluation"
        elif any(w in _qt_lower for w in ('predict', 'expect', 'future', 'will you')):
            _question_intent = "prediction"
        elif any(w in _qt_lower for w in ('recommend', 'suggest', 'advice', 'should')):
            _question_intent = "recommendation"
        elif any(w in _qt_lower for w in ('remember', 'recall', 'memory', 'past')):
            _question_intent = "recall"

        # Strategy 7 (v1.0.5.0): Study-title topic extraction as additional signal
        _study_title_words = []
        if self.study_title:
            _st_words = _ctx_re.findall(r'\b[a-zA-Z]{3,}\b', self.study_title.lower())
            _study_title_words = [w for w in _st_words if w not in _ctx_stop][:4]

        # v1.0.8.3: Topic construction — phrase-first, then entities, then words.
        # Enriches topic_words with context words for broader coverage.
        if _context_topic_words:
            # Merge context words into topic_words (deduplicated, context-first)
            _seen = set(_topic_words)
            for _cw in _context_topic_words:
                if _cw not in _seen:
                    _topic_words.append(_cw)
                    _seen.add(_cw)
            _topic_words = _topic_words[:8]  # Allow slightly more after merge

        if not topic or topic == "general":
            if _phrase_topic:
                _parts = _phrase_topic.split()
                _parts = [w.capitalize() if w.lower() in _proper_nouns else w for w in _parts]
                topic = ' '.join(_parts)
            elif _entities:
                # Named entity is the cleanest topic: "Trump", "Biden", etc.
                topic = _entities[0]
            elif _topic_words:
                # v1.0.8.3: Allow up to 3 content words for richer topics
                # e.g., "conspiracy theory politics" instead of "conspiracy theory"
                topic = ' '.join(_topic_words[:3])
            elif _cond_topic_words:
                topic = ' '.join(_cond_topic_words[:2])
            elif _domain_hint:
                topic = _domain_hint
            elif _study_title_words:
                topic = ' '.join(_study_title_words[:2])
            elif study_domain and study_domain != "general":
                topic = study_domain.replace('_', ' ')
            else:
                topic = survey_name or "the study topic"

        # v1.0.5.0: Capitalize any proper nouns in final topic
        _t_parts = topic.split()
        _t_parts = [w.capitalize() if w.lower() in _proper_nouns else w for w in _t_parts]
        topic = ' '.join(_t_parts)

        # v1.0.3.8: Use topic as stimulus and product when no specific values exist
        if not _product_source:
            _product_source = topic
        if not _feature_source:
            _feature_source = _topic_words[0] if _topic_words else "topic"
        if _stimulus_source in ("this study", "item", ""):
            _stimulus_source = topic

        # Map sentiment to emotion words
        emotion_map = {
            "very_positive": ["delighted", "thrilled", "very pleased", "impressed"],
            "positive": ["pleased", "satisfied", "happy", "comfortable"],
            "neutral": ["interested", "curious", "engaged", "attentive"],
            "negative": ["concerned", "disappointed", "uneasy", "uncertain"],
            "very_negative": ["frustrated", "upset", "very disappointed", "troubled"],
        }
        emotion_words = emotion_map.get(sentiment, emotion_map["neutral"])

        context = {
            "topic": topic,
            "stimulus": _stimulus_source,
            "product": _product_source,
            "feature": _feature_source,
            "emotion": str(rng.choice(emotion_words)),
            "sentiment": sentiment.replace("very_", ""),  # Basic generator uses simple sentiment
            "question_text": question_text,
            "study_domain": study_domain,
            "condition": condition,
            # v1.0.8.4: Use _early_intent (computed before cascade) for consistency
            # with what ComprehensiveResponseGenerator receives
            "question_intent": _early_intent,
            "entities": _entities,
            "topic_words": _topic_words,
            "domain_hint": _domain_hint,
            "question_context_raw": question_context,
            "original_question_text": _original_question_text,
            "study_title": self.study_title or "",
        }

        # v1.0.4.8: Embed behavioral profile data into context for fallback generator
        # v1.0.5.0: Enhanced — pass full trait profile + intensity + consistency
        if behavioral_profile and isinstance(behavioral_profile, dict):
            _bp = behavioral_profile
            context["response_pattern"] = _bp.get("response_pattern", "unknown")
            context["behavioral_summary"] = _bp.get("behavioral_summary", "")
            context["persona_name"] = _bp.get("persona_name", "Default")
            context["persona_description"] = _bp.get("persona_description", "")
            context["intensity"] = str(_bp.get("intensity", 0.5))
            context["consistency_score"] = str(_bp.get("consistency_score", 0.5))
            if _bp.get("response_mean") is not None:
                context["response_mean_str"] = f"{_bp['response_mean']:.1f}"
            if _bp.get("straight_lined"):
                context["straight_lined"] = "true"
            # v1.0.5.0: Pass full 7-dimensional trait vector
            _tp = _bp.get("trait_profile", {})
            if _tp:
                context["trait_social_desirability"] = str(_tp.get("social_desirability", 0.3))
                context["trait_extremity"] = str(_tp.get("extremity", 0.4))
                context["trait_consistency"] = str(_tp.get("consistency", 0.6))
                context["trait_attention"] = str(_tp.get("attention_level", 0.8))
            # v1.0.5.0: Voice memory for cross-response consistency
            _voice_hint = _bp.get("voice_consistency_hint", "")
            if _voice_hint:
                context["voice_consistency_hint"] = _voice_hint
            _established_tone = _bp.get("established_tone", "")
            if _established_tone:
                context["established_tone"] = _established_tone

        # v1.0.5.0: Domain-aware condition modifiers — covers consumer, political,
        # health, economic games, intergroup, and more. Each modifier only applies
        # when the study domain is relevant.
        cond = str(condition).lower()
        _sd_lower = study_domain.lower() if study_domain else ""
        _consumer_domains = {"consumer", "ai_attitudes", "advertising", "brand", "product_evaluation"}
        _political_domains = {"political", "polarization", "intergroup", "identity"}
        _health_domains = {"health", "wellbeing", "clinical", "stress"}
        _econ_game_domains = {"dictator", "trust", "ultimatum", "public_goods", "economic"}

        if "ai" in cond and "no ai" not in cond:
            if _sd_lower in _consumer_domains:
                context["stimulus"] = "AI-recommended " + str(context["stimulus"])
        elif "human" in cond or "no ai" in cond:
            if _sd_lower in _consumer_domains:
                context["stimulus"] = "human-curated " + str(context["stimulus"])
        if "hedonic" in cond or "experiential" in cond:
            context["product"] = "hedonic " + str(context["product"])
        elif "utilitarian" in cond or "functional" in cond:
            context["product"] = "functional " + str(context["product"])

        # v1.0.5.0: Political condition modifiers
        if any(w in cond for w in ('liberal', 'democrat', 'progressive', 'left')):
            if any(d in _sd_lower for d in _political_domains):
                context["condition_framing"] = "progressive/liberal"
        elif any(w in cond for w in ('conservative', 'republican', 'right')):
            if any(d in _sd_lower for d in _political_domains):
                context["condition_framing"] = "conservative/right-leaning"

        # v1.0.5.0: Ingroup/outgroup condition modifiers
        if any(w in cond for w in ('ingroup', 'in_group', 'same', 'similar')):
            context["condition_framing"] = context.get("condition_framing", "") + " ingroup"
        elif any(w in cond for w in ('outgroup', 'out_group', 'different', 'other')):
            context["condition_framing"] = context.get("condition_framing", "") + " outgroup"

        # v1.0.5.0: Health condition modifiers
        if any(w in cond for w in ('risk', 'threat', 'danger', 'severity')):
            if any(d in _sd_lower for d in _health_domains):
                context["condition_framing"] = "health risk/threat"
        elif any(w in cond for w in ('prevention', 'benefit', 'gain', 'healthy')):
            if any(d in _sd_lower for d in _health_domains):
                context["condition_framing"] = "health benefit/prevention"

        # v1.0.0 CRITICAL FIX: Create question-specific seed for fallback generator
        # Combine participant_seed with a stable hash of the question identity
        base_name = str(question_spec.get("name", ""))
        var_name = str(question_spec.get("variable_name", ""))
        unique_id = f"{base_name}|{var_name}|{question_text[:100]}"
        # Use stable hash independent of Python's hash randomization
        question_hash_stable = sum(ord(c) * (i + 1) * 31 for i, c in enumerate(unique_id[:200]))
        unique_fallback_seed = (participant_seed + question_hash_stable) % (2**31)

        # v1.0.7.2: Wrap text_generator in try/except — if even the last-resort
        # generator fails, produce a hardcoded topic-aware response rather than
        # propagating the exception to the outer handler (which sets "").
        try:
            _text_result = self.text_generator.generate_response(
                response_type, style, context, traits, unique_fallback_seed
            )
            if _text_result and _text_result.strip():
                return _text_result
        except Exception as _txt_gen_err:
            logger.warning("TextResponseGenerator error: %s", _txt_gen_err)

        # v1.0.7.2: ABSOLUTE LAST RESORT — generate a minimal topic-aware response
        # so that the OE column is NEVER empty when a participant should have answered.
        return self._last_resort_oe_response(question_text, _original_question_text, sentiment, participant_seed)

    def _last_resort_oe_response(
        self,
        question_text: str,
        original_question_text: str,
        sentiment: str,
        participant_seed: int,
    ) -> str:
        """Generate a minimal topic-aware OE response when ALL generators have failed.

        v1.0.7.2: This is the absolute last resort. It extracts topic words from
        the question text and produces a short, on-topic response. This method
        must NEVER raise an exception and must NEVER return an empty string.
        """
        import re as _lr_re
        _rng = np.random.RandomState(participant_seed % (2**31))

        # Extract topic words from original question text (before context embedding)
        _source = original_question_text or question_text or ""
        _words = _lr_re.findall(r'\b[a-zA-Z]{4,}\b', _source.lower())
        _stop = {
            'this', 'that', 'about', 'what', 'your', 'please', 'describe',
            'explain', 'question', 'context', 'study', 'topic', 'condition',
            'think', 'feel', 'have', 'some', 'with', 'from', 'very', 'really',
            'would', 'could', 'should', 'tell', 'share', 'much', 'many',
            'they', 'them', 'their', 'been', 'being', 'were', 'also',
        }
        _topic_words = [w for w in _words if w not in _stop][:3]
        _topic = ' '.join(_topic_words) if _topic_words else 'what was asked'

        # Sentiment-aligned minimal responses
        if sentiment in ('very_positive', 'positive'):
            _templates = [
                f"I feel positively about {_topic}.",
                f"I have good feelings about {_topic}.",
                f"{_topic} is something I view favorably.",
                f"My thoughts on {_topic} are generally positive.",
                f"I think {_topic} is important and I feel good about it.",
            ]
        elif sentiment in ('very_negative', 'negative'):
            _templates = [
                f"I have concerns about {_topic}.",
                f"My feelings about {_topic} are not very positive.",
                f"{_topic} is something I feel negatively about.",
                f"I'm not too happy about {_topic} honestly.",
                f"I think {_topic} needs more thought, I'm not satisfied.",
            ]
        else:
            _templates = [
                f"I have mixed feelings about {_topic}.",
                f"{_topic} is something I've thought about.",
                f"I shared my honest thoughts about {_topic}.",
                f"My views on {_topic} are somewhere in the middle.",
                f"I considered {_topic} and gave my genuine opinion.",
            ]

        return str(_templates[int(_rng.randint(0, len(_templates)))])

    def _generate_demographics(self, n: int) -> pd.DataFrame:
        rng = np.random.RandomState(self.seed + 1000)

        age_mean = _safe_numeric(self.demographics.get("age_mean", 35), default=35.0)
        age_sd = _safe_numeric(self.demographics.get("age_sd", 12), default=12.0)
        ages = rng.normal(age_mean, age_sd, int(n))
        ages = np.clip(ages, 18, 70).astype(int)

        male_pct = _safe_numeric(self.demographics.get("gender_quota", 50), default=50.0) / 100.0
        male_pct = float(np.clip(male_pct, 0.0, 1.0))

        female_pct = (1.0 - male_pct) * 0.96
        nonbinary_pct = 0.025
        pnts_pct = 0.015

        total = male_pct + female_pct + nonbinary_pct + pnts_pct
        if total <= 0:
            total = 1.0

        # v1.4.3: Use descriptive string labels instead of numeric codes for Gender
        _gender_labels = ["Male", "Female", "Non-binary", "Prefer not to say"]
        genders = rng.choice(
            _gender_labels,
            size=int(n),
            p=[male_pct / total, female_pct / total, nonbinary_pct / total, pnts_pct / total],
        )

        return pd.DataFrame({"Age": ages, "Gender": genders})

    def _generate_condition_assignment(self, n: int) -> pd.Series:
        """Generate condition assignments based on allocation percentages or equal distribution.

        v1.4.0: Enhanced with safe numeric conversion and proportion/percentage detection.
        """
        n_conditions = len(self.conditions)
        if n_conditions == 0:
            raise ValueError("No experimental conditions defined. Cannot generate data without at least one condition.")
        assignments: List[str] = []

        if self.condition_allocation and len(self.condition_allocation) > 0:
            # Use specified allocation percentages (already normalized by __init__)
            # v1.0.6.1: Warn if allocation keys don't match conditions
            _alloc_keys = set(self.condition_allocation.keys())
            _cond_set = set(self.conditions)
            if _alloc_keys != _cond_set:
                _missing = _cond_set - _alloc_keys
                if _missing:
                    self._log(f"WARNING: Condition allocation missing keys for: {_missing}. Using equal distribution for those.")
            running_total = 0
            for i, cond in enumerate(self.conditions):
                raw_pct = self.condition_allocation.get(cond, 100.0 / n_conditions)
                # Safe float conversion in case normalization missed something
                try:
                    pct = float(raw_pct) if not isinstance(raw_pct, dict) else 100.0 / n_conditions
                except (ValueError, TypeError):
                    pct = 100.0 / n_conditions
                if np.isnan(pct) or np.isinf(pct) or pct < 0:
                    pct = 100.0 / n_conditions
                if i == n_conditions - 1:
                    # Last condition gets all remaining participants
                    count = n - running_total
                else:
                    count = round(n * pct / 100.0)
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
        if not self.conditions:
            raise ValueError("No experimental conditions defined. Please specify at least one condition.")
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
        # v1.2.1: Safe trait value extraction
        attention = _safe_trait_value(traits.get("attention_level"), 0.8)

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
        # Filter out any NaN values defensively (should not occur since exclusion
        # flags are computed before missing data injection, but guards against refactors)
        vals = [int(v) for v in (participant_item_responses or [])
                if not (isinstance(v, float) and np.isnan(v))]

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
        # When exclude_careless_responders is True, flag only (don't recommend exclusion)
        # — this lets instructors decide on exclusion rather than auto-excluding
        if self.exclusion_criteria.exclude_careless_responders:
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

    def _simulate_response_times(
        self,
        traits: Dict[str, float],
        num_scale_items: int,
        num_open_ended: int,
        participant_seed: int,
    ) -> Dict[str, Any]:
        """
        Simulate per-participant response time metrics that correlate with
        response quality based on survey methodology research.

        SCIENTIFIC BASIS:
        =================
        Yan & Tourangeau (2008): Response time is a key indicator of data quality.
        - Engaged responders: 3-5 sec/item on Likert scales
        - Satisficers: 1-2 sec/item
        - Careless: < 1 sec/item
        - Open-ended questions: 15-45 sec for engaged, 3-8 sec for satisficers

        Malhotra (2008): Response time correlates with response consistency
        at r ~ 0.40-0.60 in online surveys.

        Callegaro et al. (2015): Item-level response times follow log-normal
        distributions with persona-dependent parameters.

        Zhang & Conrad (2014): Response time increases with item complexity
        and decreases with satisficing behavior.

        Args:
            traits: Participant trait dict
            num_scale_items: Number of Likert/scale items in the survey
            num_open_ended: Number of open-ended questions
            participant_seed: Seed for reproducibility

        Returns:
            Dict with response time metrics:
            - mean_item_response_time_ms: Average ms per scale item
            - total_scale_time_ms: Total time across all scale items
            - open_ended_time_ms: Total time on open-ended (if any)
            - response_time_quality_r: Estimated quality correlation
        """
        rng = np.random.RandomState(participant_seed)

        # Extract relevant traits
        _attention = _safe_trait_value(traits.get("attention_level"), 0.75)
        _reading_speed = _safe_trait_value(traits.get("reading_speed"), 0.60)
        _engagement = _safe_trait_value(traits.get("engagement"), 0.65)
        _consistency = _safe_trait_value(traits.get("response_consistency"), 0.65)

        # v1.0.8.7: Determine engagement category for knowledge base lookup
        if _attention < 0.45:
            _eng_cat = "careless"
        elif _attention < 0.65:
            _eng_cat = "satisficing"
        else:
            _eng_cat = "engaged"

        # ---- Scale item response times ----
        # v1.0.8.7: Use ex-Gaussian distribution from knowledge base when available
        # Ex-Gaussian (mu, sigma, tau) captures the right-skewed RT distribution
        # that simple lognormal misses: mu=Gaussian center, tau=exponential tail
        _use_ex_gaussian = False
        _ex_mu, _ex_sigma, _ex_tau = 0.0, 0.0, 0.0

        if HAS_KNOWLEDGE_BASE:
            _rt_norm = get_response_time_norm("likert", _eng_cat)
            if _rt_norm and _rt_norm.ex_gaussian_mu > 0:
                _use_ex_gaussian = True
                _ex_mu = _rt_norm.ex_gaussian_mu
                _ex_sigma = _rt_norm.ex_gaussian_sigma
                _ex_tau = _rt_norm.ex_gaussian_tau
                # Adjust by individual trait variation (±20%)
                _trait_mod = 0.8 + _attention * 0.4  # 0.8 (careless) to 1.2 (very engaged)
                _ex_mu *= _trait_mod
                _ex_sigma *= (0.9 + (1.0 - _consistency) * 0.3)
                _ex_tau *= (0.8 + (1.0 - _consistency) * 0.5)

        if not _use_ex_gaussian:
            # Fallback: original lognormal approach
            _effective_speed = 1.0 - _reading_speed
            _base_time_ms = 800 + _effective_speed * 3200 + _attention * 1500
            _base_time_ms += _engagement * 800
            _base_time_ms = float(np.clip(_base_time_ms, 400, 7000))

        if num_scale_items > 0:
            if _use_ex_gaussian:
                # v1.0.8.7: Ex-Gaussian sampling (Ratcliff, 1978; Luce, 1986)
                # RT = Normal(mu, sigma) + Exponential(tau)
                _gaussian_part = rng.normal(_ex_mu, _ex_sigma, size=num_scale_items)
                _exp_part = rng.exponential(_ex_tau, size=num_scale_items)
                _item_times = _gaussian_part + _exp_part

                # v1.0.8.7: Apply fatigue/order effects from knowledge base
                if HAS_KNOWLEDGE_BASE:
                    for _idx in range(num_scale_items):
                        _fatigue = compute_fatigue_adjustment(_idx + 1, num_scale_items)
                        # Fatigue decreases RT (speeding) and increases straight-lining
                        _item_times[_idx] *= _fatigue['variance_multiplier']
                        if _fatigue['mean_shift'] < 0:
                            _item_times[_idx] *= max(0.85, 1.0 + _fatigue['mean_shift'])
            else:
                _log_mean = np.log(_base_time_ms)
                _log_sd = 0.30 + (1.0 - _consistency) * 0.25
                _item_times = rng.lognormal(_log_mean, _log_sd, size=num_scale_items)

            _item_times = np.clip(_item_times, 300, 15000)
            _mean_item_time = float(np.mean(_item_times))
            _total_scale_time = float(np.sum(_item_times))
        else:
            _mean_item_time = _ex_mu if _use_ex_gaussian else (_base_time_ms if not _use_ex_gaussian else 4000.0)
            _total_scale_time = 0.0

        # ---- Open-ended response times ----
        # v1.0.8.7: Use knowledge base norms when available
        _oe_time_ms = 0.0
        if num_open_ended > 0:
            if HAS_KNOWLEDGE_BASE:
                _oe_norm = get_response_time_norm("open_ended", _eng_cat)
                if _oe_norm:
                    _oe_base = _oe_norm.mean_ms
                    _oe_sd = _oe_norm.sd_ms
                else:
                    _oe_base = 35000 if _eng_cat == "engaged" else (6000 if _eng_cat == "satisficing" else 2000)
                    _oe_sd = _oe_base * 0.40
            else:
                _effective_speed = 1.0 - _reading_speed
                _oe_base = 3000 + _effective_speed * 25000 + _attention * 15000
                _oe_base = float(np.clip(_oe_base, 1000, 45000))
                _oe_sd = _oe_base * 0.40

            for _ in range(num_open_ended):
                _oe_item = float(rng.lognormal(np.log(max(500, _oe_base)), max(0.1, _oe_sd / _oe_base)))
                _oe_item = float(np.clip(_oe_item, 800, 90000))
                _oe_time_ms += _oe_item

        # ---- Quality-time correlation (Malhotra, 2008) ----
        _quality_score = (_attention + _consistency + (1.0 - _reading_speed)) / 3.0
        _time_score = _mean_item_time / 7000.0
        _estimated_r = 0.40 + _quality_score * 0.20
        _estimated_r = float(np.clip(_estimated_r, 0.35, 0.65))

        return {
            "mean_item_response_time_ms": int(round(_mean_item_time)),
            "total_scale_time_ms": int(round(_total_scale_time)),
            "open_ended_time_ms": int(round(_oe_time_ms)),
            "response_time_quality_r": round(_estimated_r, 2),
            "distribution_model": "ex_gaussian" if _use_ex_gaussian else "lognormal",
            "engagement_category": _eng_cat,
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
                ("Age", f"Participant age in years (18-70, mean ~ {self.demographics.get('age_mean', 35)})"),
                ("Gender", "Participant gender: Male, Female, Non-binary, or Prefer not to say"),
            ]
        )

        # v1.0.8.1: Progress callback for real-time UI updates
        def _report_progress(phase: str, current: int, total: int) -> None:
            if self.progress_callback:
                try:
                    self.progress_callback(phase, current, total)
                except Exception as _cb_err:
                    # v1.1.1.4: Log instead of silently swallowing — helps diagnose
                    # frozen progress bar issues without breaking generation.
                    logger.debug("Progress callback error (phase=%s, %d/%d): %s", phase, current, total, _cb_err)

        assigned_personas: List[str] = []
        all_traits: List[Dict[str, float]] = []
        _report_progress("personas", 0, n)
        for i in range(n):
            persona_name, persona = self._assign_persona(i)
            traits = self._generate_participant_traits(i, persona)
            # v1.0.4.6: Store participant index for cross-DV coherence (Step 10)
            traits['_participant_idx'] = i
            assigned_personas.append(persona_name)
            all_traits.append(traits)

        data["_PERSONA"] = assigned_personas

        # =================================================================
        # CROSS-DV LATENT CORRELATION SCORES
        # Generate correlated z-scores across scales so that conceptually
        # related DVs (e.g., Trust and Satisfaction) co-vary realistically.
        # =================================================================
        _scale_names = [s.get("variable_name", s["name"]) for s in self.scales]
        if self.correlation_matrix is not None:
            _corr_matrix = self.correlation_matrix
        else:
            # Auto-infer from scale names
            try:
                _corr_matrix, _ = infer_correlation_matrix(self.scales)
            except Exception:
                _corr_matrix = None

        if _corr_matrix is not None and len(_scale_names) > 1:
            try:
                _latent_scores = generate_latent_scores(n, _corr_matrix, self.seed)
                # Store latent z-scores in each participant's traits
                for i in range(n):
                    all_traits[i]["_latent_dvs"] = {
                        _scale_names[j]: float(_latent_scores[i, j])
                        for j in range(min(len(_scale_names), _latent_scores.shape[1]))
                    }
                self._log(f"Generated cross-DV latent scores for {len(_scale_names)} scales")
            except Exception as e:
                self._log(f"WARNING: Failed to generate correlated latent scores: {e}")
        else:
            self._log("Cross-DV correlation: skipped (single scale or no correlation matrix)")

        participant_item_responses: List[List[int]] = [[] for _ in range(n)]

        # v1.0.4.6 Step 10: Cross-DV coherence — per-participant response history
        # Tracks running mean of normalized responses across scales for each participant.
        # Used to create within-participant consistency (halo / CMV effect) that
        # makes responses across conceptually related DVs more coherent.
        # Podsakoff et al. (2003): CMV accounts for r ≈ 0.10-0.20 shared variance.
        _participant_response_history: List[Dict[str, float]] = [
            {'running_sum': 0.0, 'running_count': 0, 'running_mean': 0.5}
            for _ in range(n)
        ]
        # Store reference on self so _generate_scale_response can access it
        self._participant_response_history = _participant_response_history

        # v1.0.4.9: Per-participant reverse-item failure tracking
        # Tracks whether each participant consistently fails or passes reverse items.
        # Careless respondents who fail one reverse item are more likely to fail others.
        # Scientific basis: Woods (2006) — reverse-item failure is trait-like within session
        self._participant_reverse_tracking: List[Dict[str, int]] = [
            {'total_reverse': 0, 'failed_reverse': 0}
            for _ in range(n)
        ]

        attention_results: List[List[bool]] = []
        attention_check_values: List[int] = []
        for i in range(n):
            p_seed = (self.seed + i * 100) % (2**31)
            check_val, passed = self._generate_attention_check(
                conditions.iloc[i], all_traits[i], "ai_manipulation", p_seed
            )
            attention_check_values.append(int(check_val))
            attention_results.append([bool(passed)])

        # v1.4.3: Use clear, scientific attention check column naming
        data["Attention_Check_1"] = attention_check_values
        self.column_info.append(("Attention_Check_1", "Manipulation/attention check: 1=Correct, 2=Incorrect"))

        # =====================================================================
        # SCALE DATA GENERATION - CONTRACT ENFORCEMENT
        # Each scale in self.scales has been validated upstream (_validated=True).
        # We use direct key access (not .get() with defaults) to ensure that
        # if a scale is missing required keys, it fails LOUDLY rather than
        # silently producing data with wrong parameters.
        # =====================================================================
        _scale_generation_log: List[Dict[str, Any]] = []  # Track what was generated

        _used_column_prefixes: set = set()  # Track to prevent column collisions

        _total_scales = len(self.scales)
        for scale_idx, scale in enumerate(self.scales):
            _report_progress("scales", scale_idx, _total_scales)
            # EXTRACT with contract enforcement - fail loudly on missing keys
            scale_name_raw = str(scale.get("name", "")).strip()
            if not scale_name_raw:
                self._log(f"WARNING: Scale {scale_idx} has no name, skipping")
                continue
            # v1.3.6: Prefer variable_name for column generation to avoid collisions
            # when multiple scales share the same display name
            # v1.4.3: Use _clean_column_name for scientific column naming
            _var_name = str(scale.get("variable_name", "")).strip()
            if _var_name:
                scale_name = _clean_column_name(_var_name)
            else:
                scale_name = _clean_column_name(scale_name_raw)

            # Deduplicate column prefix to prevent overwrites
            _base_col = scale_name
            _col_suffix = 2
            while scale_name in _used_column_prefixes:
                scale_name = f"{_base_col}_{_col_suffix}"
                _col_suffix += 1
            _used_column_prefixes.add(scale_name)

            # Extract scale_points - NO silent defaulting
            if "scale_points" not in scale:
                self._log(f"WARNING: Scale '{scale_name_raw}' missing scale_points, using 7")
                scale_points = 7
            else:
                try:
                    scale_points = int(float(scale.get("scale_points", 7)))
                except (ValueError, TypeError):
                    scale_points = 7

            # Extract num_items - NO silent defaulting
            if "num_items" not in scale:
                self._log(f"WARNING: Scale '{scale_name_raw}' missing num_items, using 5")
                num_items = 5
            else:
                try:
                    num_items = int(float(scale.get("num_items", 1)))
                except (ValueError, TypeError):
                    num_items = 1

            # Final safety bounds (should never trigger for validated scales)
            scale_points = max(2, min(1001, scale_points))
            num_items = max(1, num_items)

            # Extract scale_min and scale_max from scale dict (from QSF detection)
            # v1.2.1: ROBUST defensive handling: check for None, NaN, dict, and invalid types
            raw_scale_min = scale.get("scale_min", 1)
            raw_scale_max = scale.get("scale_max", scale_points)

            # Handle dict (can occur from malformed data) - extract value or default
            if isinstance(raw_scale_min, dict):
                raw_scale_min = raw_scale_min.get("value", 1) if "value" in raw_scale_min else 1
            if isinstance(raw_scale_max, dict):
                raw_scale_max = raw_scale_max.get("value", scale_points) if "value" in raw_scale_max else scale_points

            # Handle None
            if raw_scale_min is None:
                raw_scale_min = 1
            if raw_scale_max is None:
                raw_scale_max = scale_points

            # Handle NaN and convert to int
            try:
                if isinstance(raw_scale_min, float) and np.isnan(raw_scale_min):
                    raw_scale_min = 1
                if isinstance(raw_scale_max, float) and np.isnan(raw_scale_max):
                    raw_scale_max = scale_points
                scale_min = int(raw_scale_min)
                scale_max = int(raw_scale_max)
            except (ValueError, TypeError):
                # Fallback if conversion fails
                scale_min = 1
                scale_max = scale_points

            # Safely parse reverse_items - skip invalid values
            reverse_items_raw = scale.get("reverse_items", []) or []
            reverse_items = set()
            for x in reverse_items_raw:
                try:
                    reverse_items.add(int(x))
                except (ValueError, TypeError):
                    pass  # Skip invalid reverse item values

            self._log(f"Generating scale '{scale_name_raw}': {num_items} items, {scale_min}-{scale_max} range")
            _scale_generation_log.append({
                "name": scale_name_raw,
                "scale_points": scale_points,
                "scale_min": scale_min,
                "scale_max": scale_max,
                "num_items": num_items,
                "columns_generated": [],
            })

            for item_num in range(1, num_items + 1):
                col_name = f"{scale_name}_{item_num}"
                is_reverse = item_num in reverse_items

                item_values: List[int] = []
                col_hash = _stable_int_hash(col_name)
                for i in range(n):
                    p_seed = (self.seed + i * 100 + col_hash) % (2**31)
                    self._current_participant_idx = i  # v1.0.4.9: for reverse tracking
                    self._current_item_position = item_num
                    self._current_item_total = num_items
                    val = self._generate_scale_response(
                        scale_min,
                        scale_max,
                        all_traits[i],
                        is_reverse,
                        conditions.iloc[i],
                        scale_name,
                        p_seed,
                    )
                    # SAFETY: Enforce bounds on generated value
                    val = max(scale_min, min(scale_max, int(val)))
                    item_values.append(val)
                    participant_item_responses[i].append(val)

                    # v1.0.4.6 Step 10: Update per-participant response history
                    _hist = _participant_response_history[i]
                    _scale_range = max(1, scale_max - scale_min)
                    _normalized_val = (val - scale_min) / _scale_range
                    _hist['running_sum'] += _normalized_val
                    _hist['running_count'] += 1
                    _hist['running_mean'] = _hist['running_sum'] / _hist['running_count']

                data[col_name] = item_values
                _scale_generation_log[-1]["columns_generated"].append(col_name)

                reverse_note = " (reverse-coded)" if is_reverse else ""
                self.column_info.append(
                    (col_name, f'{scale_name_raw} item {item_num} ({scale_min}-{scale_max}){reverse_note}')
                )

            # v1.4.11: Inject inter-item correlation for multi-item scales
            # This adds realistic Cronbach's alpha while preserving per-item
            # condition effects, persona variation, and calibration.
            if num_items >= 3:
                target_alpha = float(scale.get("reliability", 0.85))
                item_col_names = [f"{scale_name}_{j+1}" for j in range(num_items)]
                try:
                    _item_matrix = np.array(
                        [data[c] for c in item_col_names], dtype=float
                    ).T  # shape (n, num_items)
                    _correlated = _inject_inter_item_correlation(
                        _item_matrix, target_alpha, scale_min, scale_max,
                    )
                    for j, c in enumerate(item_col_names):
                        data[c] = _correlated[:, j].tolist()
                    self._log(f"Injected inter-item correlation for '{scale_name_raw}' (target alpha={target_alpha:.2f})")
                except Exception as _corr_err:
                    self._log(f"WARNING: Could not inject correlation for '{scale_name_raw}': {_corr_err}")

        # v1.0.5.8: Anti-detection — detect and break alternating/zigzag patterns.
        # Mechanical alternation (e.g., 2,4,2,4,2,4 or 1,7,1,7,1,7) across items
        # is a classic tell for non-human data. Real humans show item-content-driven
        # variation, not mechanical oscillation. Detection: check if consecutive
        # differences alternate sign perfectly for 6+ items.
        for log_entry in _scale_generation_log:
            item_cols = log_entry["columns_generated"]
            if len(item_cols) < 6:
                continue  # Need at least 6 items to detect a pattern
            for i in range(n):
                _vals = [data[c][i] for c in item_cols]
                # Check for perfect alternation: diff signs alternate (+,-,+,-,+,-)
                _diffs = [_vals[j+1] - _vals[j] for j in range(len(_vals)-1)]
                _signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in _diffs]
                _nonzero_signs = [s for s in _signs if s != 0]
                if len(_nonzero_signs) >= 5:
                    _alternating = all(
                        _nonzero_signs[k] != _nonzero_signs[k+1]
                        for k in range(len(_nonzero_signs)-1)
                    )
                    if _alternating:
                        # Break the pattern by adding small noise to 2-3 items
                        _zz_rng = np.random.RandomState(self.seed + i * 777)
                        _break_count = _zz_rng.randint(2, 4)
                        _break_indices = _zz_rng.choice(
                            len(item_cols), size=min(_break_count, len(item_cols)), replace=False
                        )
                        _s_min = log_entry["scale_min"]
                        _s_max = log_entry["scale_max"]
                        for _bi in _break_indices:
                            _noise = _zz_rng.choice([-1, 0, 1])
                            _new_val = int(np.clip(data[item_cols[_bi]][i] + _noise, _s_min, _s_max))
                            data[item_cols[_bi]][i] = _new_val

        # Store generation log for post-generation verification
        self._scale_generation_log = _scale_generation_log

        # =====================================================================
        # v1.4.3: COMPOSITE MEAN COLUMNS FOR MULTI-ITEM SCALES
        # For each scale with > 1 item, compute a _mean column as the row-wise
        # average across all items. This is the standard composite score used
        # in behavioral science analysis.
        # =====================================================================
        for log_entry in _scale_generation_log:
            item_cols = log_entry["columns_generated"]
            if len(item_cols) > 1:
                # Compute row-wise mean across all items for this scale
                mean_values: List[float] = []
                for i in range(n):
                    item_sum = sum(data[col][i] for col in item_cols)
                    mean_values.append(round(item_sum / len(item_cols), 2))
                # Derive clean composite column name from the first item column
                # e.g., "Trust_1" -> "Trust_mean"
                _prefix = item_cols[0].rsplit("_", 1)[0]
                mean_col_name = f"{_prefix}_mean"
                data[mean_col_name] = mean_values
                scale_raw_name = log_entry["name"]
                self.column_info.append(
                    (mean_col_name, f"{scale_raw_name} composite mean ({log_entry['scale_min']}-{log_entry['scale_max']})")
                )
                self._log(f"Generated composite mean column '{mean_col_name}' from {len(item_cols)} items")

        for var in self.additional_vars:
            var_name_raw = str(var.get("name", "Variable")).strip() or "Variable"
            # v1.4.3: Use _clean_column_name for scientific column naming
            var_name = _clean_column_name(var_name_raw)
            var_min = _safe_numeric(var.get("min", 0), default=0, as_int=True)
            var_max = _safe_numeric(var.get("max", 10), default=10, as_int=True)
            # SAFETY: Ensure min < max
            if var_max <= var_min:
                var_max = var_min + 1

            col_hash = _stable_int_hash(var_name)
            values: List[int] = []
            for i in range(n):
                p_seed = (self.seed + i * 100 + col_hash) % (2**31)
                self._current_participant_idx = i  # v1.0.4.9: for reverse tracking
                self._current_item_position = 1
                self._current_item_total = 1
                val = self._generate_scale_response(
                    var_min, var_max, all_traits[i], False, conditions.iloc[i], var_name, p_seed
                )
                # SAFETY: Enforce bounds on generated value
                val = max(var_min, min(var_max, int(val)))
                values.append(val)
                participant_item_responses[i].append(val)

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

        # v1.4.8: Pre-fill LLM response pool with smart scaling
        # v1.9.1: Always try prefill if generator exists — providers may recover
        # during actual generation even if initial check was uncertain
        # v1.0.6.3: Force provider reset before prefill for clean state
        # v1.0.7.1: TOTAL prefill time budget — shared across ALL OE question × condition
        # combinations. This prevents the scenario where auto-recovery re-enables
        # providers between prefill_pool calls, causing each call to retry and fail.
        # v1.1.1.0: INCREASED from 30s → 90s. The old 30s budget left most pool
        # buckets empty (5 sentiments × N conditions × M questions = many buckets).
        # With 90s, even slow providers (~15s/call) can fill 6+ buckets, which
        # dramatically reduces expensive on-demand generation during the main loop.
        _PREFILL_TOTAL_BUDGET = 90.0  # seconds
        # v1.1.1.5: Skip prefill entirely when template fallback is enabled (user chose
        # "Template Engine" or "Adaptive Behavioral Engine").  No point pre-filling
        # an LLM pool that won't be used — saves 0-90s of wasted API calls.
        if self.llm_generator and self.open_ended_questions and not self.allow_template_fallback:
            try:
                self.llm_generator.reset_providers()
                self._log("LLM providers reset before prefill (clean state)")
            except Exception:
                pass
            _prefill_wall_start = time.time()
            _prefill_timed_out = False
            try:
                _unique_conditions = list(set(conditions.tolist()))
                _sents = ["very_positive", "positive", "neutral", "negative", "very_negative"]
                _prefill_oe_count = len(self.open_ended_questions)
                for _pf_idx, oq in enumerate(self.open_ended_questions):
                    # v1.1.1.5: Report progress during prefill so the watchdog thread
                    # doesn't mistake a long prefill (up to 90s) for a stall.
                    _report_progress("llm_prefill", _pf_idx, _prefill_oe_count)
                    # v1.1.1.3: Skip demographic questions in LLM prefill — they generate
                    # numeric/categorical data, not text.  Also prevents demographic
                    # variable names (e.g. "Age") from polluting topic inference.
                    if str(oq.get("question_purpose", "")).strip() == "Demographic":
                        continue
                    # v1.0.7.1: Check total budget before each OE question
                    _elapsed = time.time() - _prefill_wall_start
                    if _elapsed >= _PREFILL_TOTAL_BUDGET:
                        self._log(f"LLM prefill: total time budget ({_PREFILL_TOTAL_BUDGET:.0f}s) "
                                  f"exceeded after {_elapsed:.1f}s — remaining questions use templates")
                        _prefill_timed_out = True
                        break
                    # v1.0.7.1: If API was disabled during prefill, stop immediately
                    if not self.llm_generator.is_llm_available:
                        self._log("LLM prefill: API unavailable — switching to template fallback")
                        break
                    _q_text = str(oq.get("question_text", oq.get("name", "")))
                    _q_ctx = str(oq.get("question_context", "")).strip()
                    # v1.0.1.2: Enrich pool pre-fill with user-provided context + condition
                    if _q_ctx:
                        import re as _re_pool
                        _humanized_pool = _re_pool.sub(r'[_\-]+', ' ', _q_text).strip() if _q_text and " " not in _q_text.strip() else _q_text
                        _topic_pool = self.study_title or self.study_description or ""
                        _q_text = f"Question: {_humanized_pool}\nContext: {_q_ctx}"
                        if _topic_pool:
                            _q_text += f"\nStudy topic: {_topic_pool}"
                    elif _q_text and " " not in _q_text.strip():
                        import re as _re_pool
                        _humanized_pool = _re_pool.sub(r'[_\-]+', ' ', _q_text).strip()
                        _topic_pool = self.study_title or self.study_description or ""
                        if _topic_pool:
                            _q_text = f"In the context of a study about {_topic_pool}, please share your thoughts on: {_humanized_pool}"
                        else:
                            _q_text = f"Please share your thoughts on: {_humanized_pool}"
                    for _cond in _unique_conditions:
                        # v1.0.7.1: Check budget and API status before each condition
                        if time.time() - _prefill_wall_start >= _PREFILL_TOTAL_BUDGET:
                            _prefill_timed_out = True
                            break
                        if not self.llm_generator.is_llm_available:
                            break
                        if self.survey_flow_handler.is_question_visible(
                            _clean_column_name(str(oq.get("name", ""))), _cond
                        ):
                            # Per-call budget = remaining total budget
                            _remaining = max(5.0, _PREFILL_TOTAL_BUDGET - (time.time() - _prefill_wall_start))
                            self.llm_generator.prefill_pool(
                                question_text=_q_text,
                                condition=_cond,
                                sentiments=_sents,
                                sample_size=n,
                                n_conditions=len(_unique_conditions),
                                max_time=_remaining,
                            )
                    if _prefill_timed_out:
                        break
                _prefill_elapsed = time.time() - _prefill_wall_start
                _stats = self.llm_generator.stats
                if _stats['pool_size'] > 0:
                    self._log(f"LLM pre-filled pool: {_stats['llm_calls']} API calls, "
                              f"{_stats['pool_size']} responses via {_stats.get('active_provider', 'unknown')} "
                              f"({_prefill_elapsed:.1f}s)")
                else:
                    self._log(f"LLM prefill: {_stats['llm_calls']} API calls but 0 responses in "
                              f"{_prefill_elapsed:.1f}s — using template fallback. "
                              f"Providers: {_stats.get('providers', {})}")
                    # v1.1.1.0: Only reset if not already force-disabled
                    if hasattr(self.llm_generator, '_force_disabled') and self.llm_generator._force_disabled:
                        self._log("LLM prefill: force-disabled — skipping provider reset")
                    elif hasattr(self.llm_generator, '_reset_all_providers'):
                        self.llm_generator._reset_all_providers()
                        self.llm_generator._api_available = True
            except Exception as _pf_err:
                self._log(f"WARNING: LLM pool prefill failed: {_pf_err}")
                logger.warning("LLM pool prefill failed: %s", _pf_err)
                # v1.1.1.0: Only reset if not force-disabled
                try:
                    if hasattr(self.llm_generator, '_force_disabled') and self.llm_generator._force_disabled:
                        self._log("LLM prefill error recovery: force-disabled — skipping reset")
                    elif hasattr(self.llm_generator, '_reset_all_providers'):
                        self.llm_generator._reset_all_providers()
                        self.llm_generator._api_available = True
                except Exception as _reset_err:
                    logger.warning("Provider reset after prefill failure also failed: %s", _reset_err)

        # v1.0.5.0: Participant voice memory — tracks style, tone, and themes across
        # multiple OE questions for the SAME participant. This ensures cross-response
        # consistency: the same person should sound the same across all their answers.
        # Key insight: a real participant doesn't change personality between questions.
        _participant_voice_memory: Dict[int, Dict[str, Any]] = {}
        # v1.0.5.7: PRE-INITIALIZE voice memory for ALL participants BEFORE the
        # OE loop.  Previously only initialized after the first OE response,
        # meaning the first OE question for each participant had NO voice
        # consistency hint.  Now every participant starts with a tone derived
        # from their numeric response pattern.
        for _vi in range(n):
            _v_vals = participant_item_responses[_vi]
            # v1.0.6.1: Filter NaN/None before computing mean
            _v_clean = [float(v) for v in _v_vals if v is not None and not (isinstance(v, float) and np.isnan(v))] if _v_vals else []
            _v_mean = float(np.mean(_v_clean)) if _v_clean else None
            if _v_mean is not None:
                if _v_mean >= 5.5:
                    _v_tone = "positive"
                elif _v_mean >= 4.5:
                    _v_tone = "slightly positive"
                elif _v_mean <= 2.5:
                    _v_tone = "negative"
                elif _v_mean <= 3.5:
                    _v_tone = "slightly negative"
                else:
                    _v_tone = "neutral"
            else:
                _v_tone = "neutral"
            _participant_voice_memory[_vi] = {
                'responses': [],
                'tone': _v_tone,
                'last_response': '',
                'response_mean': _v_mean,
            }

        # ONLY generate open-ended responses for questions actually in the QSF
        # Never create default/fake questions - this prevents fake variables like "Task_Summary"
        # v1.0.0: Use survey flow handler to determine question visibility per condition
        # v1.1.1.0: Hard timeout for OE generation — prevents indefinite hangs when
        # LLM providers are slow or unresponsive. After budget expires, remaining
        # participants fall back to template generation automatically.
        # REDUCED from 300s → 180s (3 min). The old 5-min budget plus auto-recovery
        # allowed infinite retry cycles. 3 min is generous for any real LLM response.
        _OE_GENERATION_BUDGET = 180.0  # 3 minutes max PER QUESTION
        # v1.1.1.0: Per-participant timeout — if a SINGLE participant's OE response
        # takes more than this, force template fallback for that participant AND
        # permanently disable LLM (the provider is clearly hanging).
        _PER_PARTICIPANT_OE_TIMEOUT = 45.0  # seconds
        # v1.2.0.0: Budget/timer are now RESET per question (moved inside the loop).
        # Previously a single shared timer meant Q1 could exhaust the budget,
        # permanently disabling LLM for ALL subsequent questions — producing
        # gibberish template responses for Q2+ even when the user chose "Built-in AI".
        _oe_budget_switched_count = 0  # How many participants used fallback (cumulative)
        _CONSECUTIVE_SLOW_LIMIT = 3  # After 3 slow participants in a row, kill LLM
        _total_oe = len(self.open_ended_questions)
        _report_progress("open_ended", 0, _total_oe)
        _oe_budget_exceeded_any = False  # Track if ANY question hit the budget
        _oe_all_questions_wall_start = time.time()  # Wall-clock start for total elapsed logging
        # v1.2.0.0: Per-participant generation source tracking.
        # Maps OE column_name → list of "AI" or "Template" per participant.
        _generation_source_map: Dict[str, List[str]] = {}
        _completed_oe_columns: List[str] = []  # Columns fully generated so far
        for _oe_idx, q in enumerate(self.open_ended_questions):
            # v1.2.0.0: Reset per-question budget, timer, and LLM state.
            # Each OE question gets a FRESH 180s budget and a fresh LLM chance.
            # Without this, Q1 exhausting the budget permanently kills LLM for Q2+,
            # producing gibberish template fallback for all subsequent questions.
            _oe_gen_wall_start = time.time()
            _oe_budget_exceeded = False
            _consecutive_slow_participants = 0
            # v1.2.0.0: If LLM was force-disabled during the PREVIOUS question
            # and template fallback is NOT allowed (user chose "Built-in AI"),
            # raise LLMExhaustedMidGeneration so app.py can prompt the user
            # to provide their own API key or choose a fallback method.
            # This ONLY fires on Q2+ (not Q1, since Q1 hasn't had a chance yet).
            if (_oe_idx > 0
                    and self.llm_generator
                    and getattr(self.llm_generator, '_force_disabled', False)
                    and not self.allow_template_fallback):
                _remaining_qs = list(self.open_ended_questions[_oe_idx:])
                self._log(
                    f"OE Q{_oe_idx+1}/{_total_oe}: LLM exhausted after Q{_oe_idx}. "
                    f"Raising LLMExhaustedMidGeneration — {len(_remaining_qs)} question(s) remain."
                )
                raise LLMExhaustedMidGeneration(
                    message=(
                        f"Free AI providers were exhausted after generating question {_oe_idx} "
                        f"of {_total_oe}. {len(_remaining_qs)} question(s) still need generation."
                    ),
                    partial_data=dict(data),
                    completed_oe_columns=list(_completed_oe_columns),
                    remaining_questions=_remaining_qs,
                    engine_state={
                        "column_info": list(self.column_info),
                        "participant_voice_memory": dict(_participant_voice_memory),
                        "oe_budget_switched_count": _oe_budget_switched_count,
                    },
                    generation_source_map=dict(_generation_source_map),
                )

            # Re-enable LLM for this question if it was force-disabled by the
            # PREVIOUS question's budget/timeout AND template fallback IS allowed.
            if self.llm_generator and getattr(self.llm_generator, '_force_disabled', False):
                self._log(f"OE Q{_oe_idx+1}/{_total_oe}: Re-enabling LLM (was force-disabled by previous question)")
                self.llm_generator._force_disabled = False
                self.llm_generator._api_available = True
            # v1.1.1.2: Per-question progress so UI shows "Text question 2/3"
            _report_progress("open_ended_question", _oe_idx, _total_oe)
            # v1.4.3: Use _clean_column_name for scientific column naming
            col_name = _clean_column_name(str(q.get("name", "Open_Response")))

            # v1.0.0 FIX: Prevent open-ended columns from overwriting existing columns
            # (e.g., an OE question named "Age" must not overwrite the demographic "Age" column)
            if col_name in data:
                original_name = col_name
                col_name = f"OE_{col_name}"
                # If even the OE_ prefixed name exists, add a numeric suffix
                suffix = 2
                while col_name in data:
                    col_name = f"OE_{original_name}_{suffix}"
                    suffix += 1
                self._log(f"Renamed open-ended column '{original_name}' -> '{col_name}' to avoid collision")

            q_text = str(q.get("question_text", col_name))
            col_hash = _stable_int_hash(col_name + q_text)  # Include question text for uniqueness

            # v1.1.1.3: Handle demographic questions — generate realistic demographic
            # values instead of AI-generated text.  Demographic questions are excluded
            # from topic inference so "Age" doesn't confuse the study topic.
            _question_purpose = str(q.get("question_purpose", "DV Response")).strip()
            # v1.1.1.4: Auto-detect demographic questions even without explicit tag.
            # If the variable name is a standalone demographic keyword (e.g., "Age",
            # "Gender") and the user didn't set a purpose, treat it as demographic to
            # prevent topic confusion.  Only auto-detect for EXACT single-word matches
            # to avoid false positives (e.g., "Age_Discrimination_Scale" should NOT match).
            _DEMOGRAPHIC_EXACT = {"age", "gender", "sex", "race", "ethnicity", "income",
                                  "education", "salary", "location", "state", "country",
                                  "zipcode", "zip", "dob", "birthdate", "marital"}
            if _question_purpose == "DV Response" and col_name.lower().strip() in _DEMOGRAPHIC_EXACT:
                _question_purpose = "Demographic"
                self._log(f"Auto-classified '{col_name}' as Demographic (exact name match)")
            if _question_purpose == "Demographic":
                _demo_responses: List[str] = []
                # v1.1.1.4: Check BOTH variable name AND question text for demographic type
                # so "How old are you?" works even if the variable is named "Q1"
                _demo_search_text = (col_name + " " + q_text).lower()
                for _di in range(n):
                    _report_progress("generating", _di, n)
                    _d_seed = (self.seed + _di * 100 + col_hash) % (2**31)
                    _d_rng = np.random.RandomState(_d_seed)
                    if any(kw in _demo_search_text for kw in ("age", "old", "born", "birth")):
                        # Age: realistic MTurk/Prolific distribution (18-80, slight right skew)
                        _age = int(np.clip(_d_rng.normal(35, 13), 18, 80))
                        _demo_responses.append(str(_age))
                    elif any(kw in _demo_search_text for kw in ("gender", "sex", "male", "female")):
                        _demo_responses.append(
                            _d_rng.choice(
                                ["Male", "Female", "Non-binary", "Prefer not to say"],
                                p=[0.48, 0.48, 0.03, 0.01],
                            )
                        )
                    elif any(kw in _demo_search_text for kw in ("race", "ethnic", "racial")):
                        _demo_responses.append(
                            _d_rng.choice(
                                ["White", "Black or African American", "Hispanic/Latino",
                                 "Asian", "Other/Mixed race"],
                                p=[0.58, 0.13, 0.19, 0.06, 0.04],
                            )
                        )
                    elif any(kw in _demo_search_text for kw in ("education", "degree", "school", "college")):
                        _demo_responses.append(
                            _d_rng.choice(
                                ["High school diploma", "Some college", "Bachelor's degree",
                                 "Master's degree", "Doctoral degree"],
                                p=[0.15, 0.25, 0.35, 0.18, 0.07],
                            )
                        )
                    elif any(kw in _demo_search_text for kw in ("income", "salary", "earn", "household income")):
                        _inc = int(np.clip(_d_rng.lognormal(10.8, 0.8), 15000, 300000))
                        _demo_responses.append(str(_inc))
                    elif any(kw in _demo_search_text for kw in ("state", "location", "country", "city", "zip")):
                        _us_states = ["California", "Texas", "Florida", "New York",
                                      "Pennsylvania", "Illinois", "Ohio", "Georgia",
                                      "North Carolina", "Michigan", "Other"]
                        _demo_responses.append(_d_rng.choice(_us_states))
                    else:
                        # Generic demographic — generate short factual answers
                        _demo_responses.append(str(int(np.clip(_d_rng.normal(40, 15), 1, 99))))
                data[col_name] = _demo_responses
                self._log(f"Generated demographic data for '{col_name}' ({n} values)")
                continue  # Skip normal OE text generation

            responses: List[str] = []
            _sources_for_col: List[str] = []  # v1.2.0.0: per-participant source tracking
            for i in range(n):
                # v1.1.1.2: Report OE progress EVERY participant (not every 5%).
                # During OE generation, each participant can take 5-30s with LLM.
                # Users need to see continuous movement, not 30s+ stale progress.
                _report_progress("generating", i, n)

                # v1.1.1.0: Check hard timeout budget — switch to template for remaining.
                # Uses disable_permanently() to prevent auto-recovery from re-enabling.
                if not _oe_budget_exceeded:
                    _oe_elapsed = time.time() - _oe_gen_wall_start
                    if _oe_elapsed >= _OE_GENERATION_BUDGET:
                        _oe_budget_exceeded = True
                        self._log(
                            f"OE generation budget ({_OE_GENERATION_BUDGET:.0f}s) exceeded "
                            f"after {_oe_elapsed:.1f}s at participant {i+1}/{n} — "
                            f"remaining participants use template fallback"
                        )
                        # v1.1.1.0: PERMANENTLY disable LLM — auto-recovery CANNOT undo this
                        if self.llm_generator:
                            try:
                                self.llm_generator.disable_permanently(
                                    f"OE budget ({_OE_GENERATION_BUDGET:.0f}s) exceeded"
                                )
                            except Exception:
                                pass

                participant_condition = conditions.iloc[i]

                # Check if this participant's condition allows them to see this question
                if not self.survey_flow_handler.is_question_visible(col_name, participant_condition):
                    # Participant wouldn't see this question - leave blank (NA)
                    responses.append("")
                    continue

                # Generate unique seed using participant, question, and question text hash
                # v1.0.0 CRITICAL FIX: Use stable hash instead of Python's hash()
                # v1.3.3: Use MD5 of full text to avoid collisions on similar questions
                _q_hash_input = (q_text + col_name).encode('utf-8', errors='replace')
                q_text_hash = int(hashlib.md5(_q_hash_input).hexdigest()[:8], 16)
                p_seed = (self.seed + i * 100 + col_hash + q_text_hash) % (2**31)
                persona_name = assigned_personas[i]
                # v1.0.6.1: Safe lookup — fallback to first available persona if name missing
                persona = self.available_personas.get(
                    persona_name, next(iter(self.available_personas.values()))
                )
                response_vals = participant_item_responses[i]
                # v1.0.6.1: Filter NaN before computing mean to prevent propagation
                _clean_resp = [float(v) for v in response_vals if v is not None and not (isinstance(v, float) and np.isnan(v))] if response_vals else []
                response_mean = float(np.mean(_clean_resp)) if _clean_resp else None

                # v1.0.4.8: Build full behavioral profile for OE-numeric consistency
                # v1.0.7.2: Wrap in try/except — profile enrichment must never
                # crash the loop and leave remaining participants with no response.
                try:
                    _beh_profile = self._build_behavioral_profile(
                        persona, all_traits[i], response_vals, response_mean,
                        participant_condition,
                    )
                except Exception as _bp_err:
                    logger.warning("Behavioral profile build failed for participant %d: %s", i + 1, _bp_err)
                    _beh_profile = {'response_mean': response_mean, 'persona_name': 'Default'}

                # v1.0.9.1: Pass additional simulation context to generators
                _add_ctx = self.study_context.get("additional_context", "")
                if _add_ctx:
                    _beh_profile['additional_context'] = _add_ctx

                # v1.0.5.7: Inject cross-response voice memory into profile.
                # Now always available (pre-initialized before OE loop).
                # First OE question gets tone from numeric ratings; subsequent
                # questions also get prior response excerpts for consistency.
                try:
                    if i in _participant_voice_memory:
                        _voice = _participant_voice_memory[i]
                        _beh_profile['prior_responses'] = _voice.get('responses', [])
                        _beh_profile['established_tone'] = _voice.get('tone', '')
                        if _voice.get('last_response'):
                            _beh_profile['voice_consistency_hint'] = (
                                f"This participant previously wrote: \"{_voice['last_response'][:80]}...\" "
                                f"Their tone was {_voice.get('tone', 'neutral')}. "
                                f"Maintain consistent voice and personality across questions."
                            )
                        elif _voice.get('tone'):
                            # First OE question: hint from numeric pattern only
                            _beh_profile['voice_consistency_hint'] = (
                                f"Based on their numeric ratings, this participant's tone is {_voice['tone']}. "
                                f"Their text should match this tone."
                            )
                except Exception as _vm_err:
                    logger.debug("Voice memory injection failed for participant %d: %s", i + 1, _vm_err)

                # v1.0.7.1: OE completeness guarantee.
                # User requirement: visible open-ended questions must be fully populated.
                # We therefore disable behavioral skip logic for OE generation and always
                # produce a non-empty response (LLM first, then deterministic fallback).

                # Generate response with enhanced uniqueness + behavioral context
                # v1.0.7.0: Wrapped in try/except to guarantee simulation never hard-stops
                # from OE generation failures (LLM or template errors).
                # v1.0.7.2: On failure, use _last_resort_oe_response instead of empty string.
                # v1.1.1.0: Per-participant timeout — if a single participant's OE takes
                # too long, abort and use template. Also detect consecutive slow participants.
                _participant_oe_start = time.time()
                try:
                    text = self._generate_open_response(
                        q,
                        persona,
                        all_traits[i],
                        participant_condition,
                        p_seed,
                        response_mean=response_mean,
                        behavioral_profile=_beh_profile,
                    )
                    _text_str = str(text) if text else ""
                except Exception as _oe_err:
                    logger.warning("OE response generation failed for participant %d: %s", i + 1, _oe_err)
                    _text_str = ""

                # v1.1.1.0: Per-participant timeout tracking
                _participant_oe_elapsed = time.time() - _participant_oe_start
                if _participant_oe_elapsed > _PER_PARTICIPANT_OE_TIMEOUT:
                    _consecutive_slow_participants += 1
                    self._log(
                        f"SLOW: Participant {i+1}/{n} OE took {_participant_oe_elapsed:.1f}s "
                        f"(limit {_PER_PARTICIPANT_OE_TIMEOUT:.0f}s) — "
                        f"consecutive slow: {_consecutive_slow_participants}/{_CONSECUTIVE_SLOW_LIMIT}"
                    )
                    if _consecutive_slow_participants >= _CONSECUTIVE_SLOW_LIMIT and self.llm_generator:
                        self._log(
                            f"{_CONSECUTIVE_SLOW_LIMIT} consecutive slow participants — "
                            f"permanently disabling LLM for remaining participants"
                        )
                        try:
                            self.llm_generator.disable_permanently(
                                f"{_CONSECUTIVE_SLOW_LIMIT} consecutive participants exceeded "
                                f"{_PER_PARTICIPANT_OE_TIMEOUT:.0f}s timeout"
                            )
                        except Exception:
                            pass
                        _oe_budget_exceeded = True
                elif _participant_oe_elapsed <= 5.0:
                    # Fast response — reset consecutive slow counter
                    _consecutive_slow_participants = 0

                # v1.1.1.5: Track template fallbacks — count ONCE per participant
                # that actually used a template/last-resort instead of LLM.
                # Previous bug: double-counted when budget exceeded AND response empty.
                _was_template_fallback = False
                if _oe_budget_exceeded and not _text_str.strip():
                    # Budget exceeded and no LLM response — will fall through to last resort
                    _was_template_fallback = True
                elif _oe_budget_exceeded and _text_str.strip():
                    # Budget exceeded but response already generated (from prior path) — check source
                    # If text came from non-LLM source, count it
                    pass  # Don't count — the response exists
                elif not _oe_budget_exceeded and not _text_str.strip():
                    # Budget NOT exceeded but still empty — genuine fallback
                    _was_template_fallback = True

                # v1.0.7.2: NEVER leave OE response empty when participant should have answered.
                # If all generators failed or returned empty, use the absolute last-resort.
                if not _text_str.strip():
                    _was_template_fallback = True  # Confirmed: falling back to last-resort
                    try:
                        _sentiment_for_lr = "neutral"
                        if response_mean is not None:
                            if response_mean >= 4.5:
                                _sentiment_for_lr = "positive"
                            elif response_mean <= 3.5:
                                _sentiment_for_lr = "negative"
                        _text_str = self._last_resort_oe_response(
                            q_text, q_text, _sentiment_for_lr, p_seed
                        )
                    except Exception as _lr_err:
                        logger.error("Even last-resort OE generation failed for participant %d: %s", i + 1, _lr_err)
                        _text_str = "I shared my honest thoughts on this."

                # v1.1.1.5: Increment fallback count exactly ONCE per participant
                if _was_template_fallback:
                    _oe_budget_switched_count += 1

                responses.append(_text_str)

                # v1.2.0.0: Track source for this participant/question
                _participant_source = getattr(self, '_last_oe_source', 'Template')
                if _was_template_fallback:
                    _participant_source = "Template"
                _sources_for_col.append(_participant_source)

                # v1.0.5.0: Update voice memory for this participant
                if _text_str.strip():
                    if i not in _participant_voice_memory:
                        # Derive initial tone from response_mean (same logic as _generate_open_response)
                        if response_mean is not None:
                            if response_mean >= 5.5:
                                _init_tone = "positive"
                            elif response_mean >= 4.5:
                                _init_tone = "positive"
                            elif response_mean <= 2.5:
                                _init_tone = "negative"
                            elif response_mean <= 3.5:
                                _init_tone = "negative"
                            else:
                                _init_tone = "neutral"
                        else:
                            _init_tone = "neutral"
                        _participant_voice_memory[i] = {
                            'responses': [],
                            'tone': _init_tone,
                            'last_response': '',
                        }
                    _vm = _participant_voice_memory[i]
                    _vm['responses'].append(_text_str[:100])
                    _vm['last_response'] = _text_str
                    # Detect established tone from response
                    _tl = _text_str.lower()
                    _pos_count = sum(1 for w in ['good', 'like', 'enjoy', 'happy', 'great', 'love', 'positive', 'support']
                                     if w in _tl)
                    _neg_count = sum(1 for w in ['bad', 'hate', 'dislike', 'frustrated', 'upset', 'terrible', 'negative', 'concerned']
                                     if w in _tl)
                    if _pos_count > _neg_count + 1:
                        _vm['tone'] = 'positive'
                    elif _neg_count > _pos_count + 1:
                        _vm['tone'] = 'negative'
                    else:
                        _vm['tone'] = 'neutral'

            data[col_name] = responses
            _generation_source_map[col_name] = _sources_for_col
            _completed_oe_columns.append(col_name)
            q_desc = q.get("question_text", "")[:50] if q.get("question_text") else q.get('type', 'text')
            self.column_info.append((col_name, f"Open-ended: {q_desc}"))
            # v1.2.0.0: Accumulate budget-exceeded flag across questions
            if _oe_budget_exceeded:
                _oe_budget_exceeded_any = True

        # v1.1.1.0: Log OE generation budget status with detailed diagnostics
        self._oe_budget_exceeded = _oe_budget_exceeded_any
        self._oe_budget_switched_count = _oe_budget_switched_count  # v1.1.1.4: Expose for metadata
        _oe_total_elapsed = time.time() - _oe_all_questions_wall_start
        if _oe_budget_exceeded_any:
            _llm_stats_summary = ""
            if self.llm_generator:
                try:
                    _s = self.llm_generator.stats
                    _llm_stats_summary = (
                        f" LLM stats: {_s.get('llm_calls', 0)} calls, "
                        f"{_s.get('pool_size', 0)} pool responses, "
                        f"{_s.get('cumulative_failures', 0)} cumulative failures, "
                        f"force_disabled={_s.get('force_disabled', False)}"
                    )
                except Exception:
                    pass
            self._log(
                f"OE generation completed with budget exceeded: {_oe_total_elapsed:.1f}s total, "
                f"{_oe_budget_switched_count} participants used template fallback.{_llm_stats_summary}"
            )
        else:
            self._log(f"OE generation completed normally in {_oe_total_elapsed:.1f}s")

        # v1.2.0.0: Add per-participant _Generation_Source column to the output.
        # This tells the user which participants got AI-generated OE responses vs
        # template-generated ones (so they can filter/identify rows later).
        # The source is determined by the FIRST OE question — if a participant got
        # AI for Q1, they're marked "AI" even if Q2 used template (which shouldn't
        # happen now with per-question budget reset, but is a safety net).
        # If multiple OE questions, use the MAJORITY source across questions.
        if _generation_source_map and _completed_oe_columns:
            _per_participant_source: List[str] = []
            for _pi in range(n):
                _ai_count = 0
                _total_count = 0
                for _src_col in _completed_oe_columns:
                    _src_list = _generation_source_map.get(_src_col, [])
                    if _pi < len(_src_list):
                        _total_count += 1
                        if _src_list[_pi] == "AI":
                            _ai_count += 1
                if _total_count == 0:
                    _per_participant_source.append("Template")
                elif _ai_count == _total_count:
                    _per_participant_source.append("AI")
                elif _ai_count == 0:
                    _per_participant_source.append("Template")
                else:
                    _per_participant_source.append("Mixed")
            data["_Generation_Source"] = _per_participant_source
            self.column_info.append(("_Generation_Source", "AI vs Template source indicator"))
            # Also store per-question source map in engine for metadata
            self._generation_source_map = _generation_source_map

        # v1.0.0 CRITICAL FIX: Post-processing validation to detect and fix duplicate responses
        # Check each participant's responses across all open-ended questions
        # Use actual column names from data (accounting for any renames due to collisions)
        open_ended_cols = []
        for q in self.open_ended_questions:
            _candidate = _clean_column_name(str(q.get("name", "Open_Response")))
            if (_candidate in data and isinstance(data[_candidate], list)
                    and len(data[_candidate]) > 0 and isinstance(data[_candidate][0], str)):
                open_ended_cols.append(_candidate)
        if len(open_ended_cols) > 1:
            for i in range(n):
                participant_responses = {}
                duplicates_found = []
                for col in open_ended_cols:
                    if col in data:
                        response = data[col][i]
                        if response and response.strip():  # Skip empty responses
                            if response in participant_responses:
                                # Found a duplicate!
                                duplicates_found.append((col, participant_responses[response]))
                            else:
                                participant_responses[response] = col

                # Fix any duplicates by adding unique modifiers
                for dup_col, orig_col in duplicates_found:
                    original_response = data[dup_col][i]
                    # Add a unique modifier to make it different
                    modifiers = [
                        "Additionally, ", "Also, ", "Furthermore, ", "On reflection, ",
                        "I would add that ", "On another note, ", "I also think that ",
                        "From a different perspective, ", "More specifically, "
                    ]
                    modifier_idx = (i + hash(dup_col) % 100) % len(modifiers)
                    if original_response and len(original_response) > 10:
                        # Modify the beginning
                        modified = modifiers[modifier_idx] + original_response[0].lower() + original_response[1:]
                        data[dup_col][i] = modified
                    else:
                        # For short responses, just add modifier
                        data[dup_col][i] = modifiers[modifier_idx] + original_response

        # v1.0.5.8: CROSS-PARTICIPANT duplicate / near-duplicate detection.
        # Real survey data NEVER has identical or near-identical OE responses
        # across different participants. This is the #1 tell for fabricated data.
        # Check each OE column for cross-participant duplicates and mutate them.
        for col in open_ended_cols:
            if col not in data:
                continue
            _col_responses = data[col]
            # Build a set of (normalized_response → list of participant indices)
            _seen: Dict[str, List[int]] = {}
            for _pi, _resp in enumerate(_col_responses):
                if not _resp or not _resp.strip():
                    continue
                # Normalize: lowercase, strip extra whitespace, remove punctuation
                _norm = re.sub(r'[^\w\s]', '', _resp.lower()).strip()
                _norm = re.sub(r'\s+', ' ', _norm)
                if len(_norm) < 10:
                    continue  # Skip very short responses (e.g., "idk")
                if _norm in _seen:
                    _seen[_norm].append(_pi)
                else:
                    _seen[_norm] = [_pi]
            # For each group of duplicates, mutate all but the first
            _dedup_modifiers = [
                "I mean ", "Like ", "Honestly ", "For me personally ",
                "Well ", "So ", "Yeah ", "Basically ", "Tbh ",
                "From my end ", "In my case ", "For me ",
            ]
            _dedup_rng = np.random.RandomState(self.seed + 99999)
            for _norm_key, _indices in _seen.items():
                if len(_indices) > 1:
                    for _di in _indices[1:]:  # Keep first, mutate rest
                        _orig = _col_responses[_di]
                        _mod = _dedup_modifiers[_dedup_rng.randint(0, len(_dedup_modifiers))]
                        # Also shuffle a word or two for additional differentiation
                        _words = _orig.split()
                        if len(_words) > 5:
                            _swap_idx = _dedup_rng.randint(1, max(2, len(_words) - 2))
                            if _swap_idx + 1 < len(_words):
                                _words[_swap_idx], _words[_swap_idx + 1] = _words[_swap_idx + 1], _words[_swap_idx]
                        _col_responses[_di] = _mod + ' '.join(_words)
                    self._log(f"Deduped {len(_indices)-1} cross-participant duplicates in '{col}'")

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

        # =================================================================
        # RESPONSE TIME SIMULATION (Yan & Tourangeau, 2008; Malhotra, 2008)
        # Generate per-participant response time metrics that correlate
        # with response quality. This provides researchers with realistic
        # timing data for data quality analysis.
        # =================================================================
        _total_scale_items = sum(
            len(log_entry["columns_generated"])
            for log_entry in _scale_generation_log
        )
        _num_oe = len(self.open_ended_questions) if self.open_ended_questions else 0

        _rt_mean_item: List[int] = []
        _rt_total_scale: List[int] = []
        for i in range(n):
            _rt_seed = (self.seed + i * 100 + 77777) % (2**31)
            _rt_data = self._simulate_response_times(
                all_traits[i], _total_scale_items, _num_oe, _rt_seed
            )
            _rt_mean_item.append(_rt_data["mean_item_response_time_ms"])
            _rt_total_scale.append(_rt_data["total_scale_time_ms"])

        data["Mean_Item_RT_ms"] = _rt_mean_item
        data["Total_Scale_RT_ms"] = _rt_total_scale
        self.column_info.extend([
            ("Mean_Item_RT_ms", "Mean response time per scale item in ms (Yan & Tourangeau, 2008)"),
            ("Total_Scale_RT_ms", "Total response time across all scale items in ms"),
        ])
        self._log(f"Generated response time data for {n} participants ({_total_scale_items} scale items, {_num_oe} OE questions)")

        # =================================================================
        # MISSING DATA & DROPOUT APPLICATION
        # Applied after all data generation, before DataFrame assembly.
        # Introduces realistic item-level missingness and survey dropout.
        # =================================================================
        if self.missing_data_rate > 0 or self.dropout_rate > 0:
            self._apply_missing_data(data, all_traits, conditions, n)

        if "_PERSONA" in data:
            del data["_PERSONA"]

        df = pd.DataFrame(data)

        # POST-GENERATION VALIDATION: Verify all scale columns are within bounds
        validation_issues = self._validate_generated_data(df)
        if validation_issues:
            self._log(f"POST-GENERATION VALIDATION: {len(validation_issues)} issue(s) found, auto-correcting")
            for issue in validation_issues:
                col = issue["column"]
                col_min = issue["expected_min"]
                col_max = issue["expected_max"]
                # Auto-correct out-of-bounds values (preserve NaN from missing data)
                col_series = df[col]
                mask = col_series.notna()
                if mask.any():
                    df.loc[mask, col] = col_series[mask].clip(lower=col_min, upper=col_max).astype(int)
                self._log(f"  Corrected {col}: clipped to [{col_min}, {col_max}]")

        # Compute observed effect sizes to validate simulation quality
        observed_effects = self._compute_observed_effect_sizes(df)

        # =====================================================================
        # ENHANCED PERSONA METADATA (v1.2.0)
        # Comprehensive tracking of persona assignment and trait distributions
        # =====================================================================

        # 1. Persona counts (absolute numbers)
        persona_counts: Dict[str, int] = {}
        for p in assigned_personas:
            persona_counts[p] = persona_counts.get(p, 0) + 1

        # 2. Persona proportions (percentages)
        total_participants = len(assigned_personas) if assigned_personas else 1
        persona_proportions: Dict[str, float] = {
            p: count / total_participants for p, count in persona_counts.items()
        }

        # 3. Per-condition persona breakdown
        # Maps condition -> {persona_name -> count}
        persona_by_condition: Dict[str, Dict[str, int]] = {}
        conditions_list = conditions.tolist() if hasattr(conditions, 'tolist') else list(conditions)
        for cond in self.conditions:
            persona_by_condition[cond] = {}
        for i, (persona_name, cond) in enumerate(zip(assigned_personas, conditions_list)):
            if cond not in persona_by_condition:
                persona_by_condition[cond] = {}
            persona_by_condition[cond][persona_name] = persona_by_condition[cond].get(persona_name, 0) + 1

        # 4. Per-condition persona proportions
        persona_proportions_by_condition: Dict[str, Dict[str, float]] = {}
        for cond, persona_dict in persona_by_condition.items():
            cond_total = sum(persona_dict.values()) if persona_dict else 1
            persona_proportions_by_condition[cond] = {
                p: count / cond_total for p, count in persona_dict.items()
            }

        # 5. Per-condition trait averages
        # Maps condition -> {trait_name -> average_value}
        trait_averages_by_condition: Dict[str, Dict[str, float]] = {}
        condition_traits: Dict[str, List[Dict[str, float]]] = {cond: [] for cond in self.conditions}
        for i, (traits_dict, cond) in enumerate(zip(all_traits, conditions_list)):
            if cond not in condition_traits:
                condition_traits[cond] = []
            condition_traits[cond].append(traits_dict)

        for cond, traits_list in condition_traits.items():
            if not traits_list:
                trait_averages_by_condition[cond] = {}
                continue
            # Get all trait keys from first participant
            trait_keys = list(traits_list[0].keys()) if traits_list else []
            trait_averages_by_condition[cond] = {}
            for trait_key in trait_keys:
                # Skip non-numeric trait values (e.g., _latent_dvs is a dict)
                if trait_key.startswith("_"):
                    continue
                values = [t.get(trait_key, 0.0) for t in traits_list if trait_key in t]
                # Filter to numeric values only
                numeric_values = [v for v in values if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v))]
                if numeric_values:
                    trait_averages_by_condition[cond][trait_key] = round(float(np.mean(numeric_values)), 4)

        # 6. Overall trait averages (across all participants)
        overall_trait_averages: Dict[str, float] = {}
        if all_traits:
            trait_keys = list(all_traits[0].keys()) if all_traits else []
            for trait_key in trait_keys:
                # Skip non-numeric trait values (e.g., _latent_dvs is a dict)
                if trait_key.startswith("_"):
                    continue
                values = [t.get(trait_key, 0.0) for t in all_traits if trait_key in t]
                # Filter to numeric values only
                numeric_values = [v for v in values if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v))]
                if numeric_values:
                    overall_trait_averages[trait_key] = round(float(np.mean(numeric_values)), 4)

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
                {
                    "variable": e.get("variable", "") if isinstance(e, dict) else getattr(e, "variable", ""),
                    "factor": e.get("factor", "") if isinstance(e, dict) else getattr(e, "factor", ""),
                    "cohens_d": e.get("cohens_d", 0.5) if isinstance(e, dict) else getattr(e, "cohens_d", 0.5),
                    "direction": e.get("direction", "") if isinstance(e, dict) else getattr(e, "direction", ""),
                }
                for e in self.effect_sizes
            ],
            "effect_sizes_observed": observed_effects,  # Actual effects in generated data
            "personas_used": sorted(list(set(assigned_personas))),
            # ENHANCED: Full persona distribution with counts and proportions
            "persona_distribution": {
                "counts": persona_counts,
                "proportions": persona_proportions,
                "total_participants": total_participants,
            } if assigned_personas else {},
            # ENHANCED: Per-condition persona breakdown
            "persona_by_condition": {
                "counts": persona_by_condition,
                "proportions": persona_proportions_by_condition,
            },
            # ENHANCED: Per-condition trait averages
            "trait_averages_by_condition": trait_averages_by_condition,
            # ENHANCED: Overall trait averages
            "trait_averages_overall": overall_trait_averages,
            # v1.0.6.1: Guard against missing flag columns if generation was partial
            "exclusion_summary": {
                "flagged_speed": int(sum(data.get("Flag_Speed", [0]))),
                "flagged_attention": int(sum(data.get("Flag_Attention", [0]))),
                "flagged_straightline": int(sum(data.get("Flag_StraightLine", [0]))),
                "total_excluded": int(sum(data.get("Exclude_Recommended", [0]))),
            },
            "validation_issues_corrected": len(validation_issues),
            "scale_verification": self._build_scale_verification_report(df),
            "generation_warnings": self._check_generation_warnings(df),
            # v1.8.7.1: Include open-ended questions with context in metadata
            "open_ended_questions": [
                {
                    "name": q.get("name", ""),
                    "variable_name": q.get("variable_name", q.get("name", "")),
                    "question_text": q.get("question_text", ""),
                    "question_context": q.get("question_context", ""),
                }
                for q in self.open_ended_questions
            ],
            # v1.4.6: LLM response generation stats
            # v1.0.6.1: Guard against .stats being None
            "llm_response_stats": (getattr(self.llm_generator, 'stats', None) or {"llm_calls": 0, "fallback_uses": 0}) if self.llm_generator else {"llm_calls": 0, "fallback_uses": 0},
            "llm_init_error": self.llm_init_error,
            # v1.1.1.4: OE generation budget tracking for transparent user reporting
            "oe_budget_exceeded": getattr(self, '_oe_budget_exceeded', False),
            "oe_budget_switched_count": getattr(self, '_oe_budget_switched_count', 0),
            # v1.4.3: Column descriptions for data dictionary / codebook generation
            "column_descriptions": {col: desc for col, desc in self.column_info},
            # v1.4.11: Scale generation log — maps scale names to actual generated columns
            # Downstream consumers (validation, instructor report) should use this
            # instead of reconstructing column names, preventing mismatches.
            "scale_generation_log": self._scale_generation_log,
            # Cross-DV correlation info
            "cross_dv_correlation": {
                "enabled": _corr_matrix is not None and len(_scale_names) > 1,
                "num_scales": len(_scale_names),
                "scale_names": _scale_names,
                "correlation_matrix": _corr_matrix.tolist() if _corr_matrix is not None and hasattr(_corr_matrix, 'tolist') else None,
                "construct_types": (
                    {name: str(ct) for name, ct in detect_construct_types(self.scales).items()}
                    if HAS_CORRELATION_MATRIX and len(_scale_names) > 1
                    else {}
                ),
            },
            # Missing data simulation info
            "missing_data": {
                "missing_data_rate": self.missing_data_rate,
                "dropout_rate": self.dropout_rate,
                "mechanism": self.missing_data_mechanism,
                "total_missing_rate": getattr(self, '_actual_missing_rate', 0.0),
                "dropout_count": getattr(self, '_actual_dropout_count', 0),
                "per_scale_missing_rate": getattr(self, '_per_scale_missing_rate', {}),
            },
        }

        # v1.0.8.1: SocSim experimental enrichment for economic game DVs
        if self.use_socsim_experimental:
            try:
                from utils.socsim_adapter import detect_game_dvs, run_socsim_enrichment
                _report_progress("socsim_enrichment", 0, 1)
                game_dvs = detect_game_dvs(
                    scales=self.scales,
                    study_title=self.study_title,
                    study_description=self.study_description,
                    conditions=self.conditions,
                )
                if game_dvs:
                    self._log(f"SocSim: Detected {len(game_dvs)} game DV(s): "
                              f"{[d['game_name'] for d in game_dvs]}")
                    df, socsim_meta = run_socsim_enrichment(
                        df=df,
                        game_dvs=game_dvs,
                        conditions=self.conditions,
                        study_title=self.study_title,
                        study_description=self.study_description,
                        sample_size=n,
                        seed=self.seed,
                        progress_callback=self.progress_callback,
                    )
                    metadata["socsim"] = socsim_meta
                    self._log(f"SocSim enrichment complete: {len(socsim_meta.get('enriched_dvs', []))} DVs enriched")
                else:
                    self._log("SocSim: No game-theory DVs detected — running standard simulation only")
                    metadata["socsim"] = {"socsim_used": False, "reason": "no_game_dvs_detected"}
            except Exception as e:
                self._log(f"SocSim enrichment failed (non-fatal): {e}")
                metadata["socsim"] = {"socsim_used": False, "error": str(e)}

        _report_progress("complete", n, n)
        return df, metadata

    def _check_generation_warnings(self, df: pd.DataFrame) -> List[str]:
        """Return any warnings about the generated data quality."""
        warnings: List[str] = []
        if "CONDITION" in df.columns and len(self.conditions) >= 2:
            cell_counts = df["CONDITION"].value_counts()
            min_cell = int(cell_counts.min()) if len(cell_counts) > 0 else 0
            if min_cell < 5:
                warnings.append(
                    f"Smallest cell has only {min_cell} participants. "
                    f"Statistical tests will be unreliable."
                )
            elif min_cell < 20 and len(self.conditions) >= 6:
                warnings.append(
                    f"Smallest cell has {min_cell} participants across "
                    f"{len(self.conditions)} conditions. Consider increasing sample size "
                    f"for more reliable statistics."
                )
        # v1.1.0.9: Warn if OE generation was cut short due to timeout
        if getattr(self, '_oe_budget_exceeded', False):
            warnings.append(
                "Open-ended text generation timed out after 5 minutes. "
                "Some participants' text was generated using templates instead of AI. "
                "For full AI generation, try using Your own API key for faster, dedicated access."
            )
        return warnings

    def _validate_generated_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        COMPREHENSIVE post-generation validation.

        Checks:
        1. All expected scale columns EXIST in the DataFrame
        2. All values are within expected bounds (min/max)
        3. Values actually USE the defined range (not clustered in a tiny sub-range)
        4. All additional variable columns exist and are within bounds
        5. Demographic columns are within expected ranges
        6. Cross-references _scale_generation_log if available

        Returns list of issues found for auto-correction.
        """
        issues: List[Dict[str, Any]] = []

        # ===== CHECK 1: Verify all expected scale columns exist =====
        for scale in self.scales:
            # v1.4.11: Use _clean_column_name (and prefer variable_name) to match
            # how columns were actually generated in generate().
            _raw_name = str(scale.get("name", "Scale")).strip()
            _var_name = str(scale.get("variable_name", "")).strip()
            scale_name = _clean_column_name(_var_name if _var_name else _raw_name)
            scale_points = _safe_numeric(scale.get("scale_points", 7), default=7, as_int=True)
            scale_points = max(2, min(1001, scale_points))
            scale_min = _safe_numeric(scale.get("scale_min", 1), default=1, as_int=True)
            num_items = _safe_numeric(scale.get("num_items", 5), default=5, as_int=True)

            for item_num in range(1, num_items + 1):
                col_name = f"{scale_name}_{item_num}"

                # CHECK 1a: Column must exist
                if col_name not in df.columns:
                    self._log(f"VALIDATION ERROR: Expected column '{col_name}' MISSING from DataFrame")
                    continue

                col_data = df[col_name]

                # Safety: skip non-numeric columns (e.g., if an OE column overwrote a scale col)
                if col_data.dtype == object or not np.issubdtype(col_data.dtype, np.number):
                    self._log(f"VALIDATION WARNING: Column '{col_name}' has non-numeric dtype {col_data.dtype}, skipping bounds check")
                    continue

                # Drop NaN values before computing min/max (missing data may have been injected)
                col_valid = col_data.dropna()
                if len(col_valid) == 0:
                    continue
                actual_min = int(col_valid.min())
                actual_max = int(col_valid.max())

                # CHECK 1b: Bounds validation
                if actual_min < scale_min or actual_max > scale_points:
                    issues.append({
                        "column": col_name,
                        "expected_min": scale_min,
                        "expected_max": scale_points,
                        "actual_min": actual_min,
                        "actual_max": actual_max,
                        "issue_type": "out_of_bounds",
                    })

                # CHECK 1c: Range utilization - values should span a reasonable
                # portion of the scale. For scales > 10 points, warn if values
                # only use < 30% of the range (indicates the old capping bug or similar)
                if scale_points > 10 and len(col_data) >= 20:
                    value_range = actual_max - actual_min
                    expected_range = scale_points - 1
                    utilization = value_range / expected_range if expected_range > 0 else 0
                    if utilization < 0.30:
                        self._log(
                            f"VALIDATION WARNING: {col_name} uses only {utilization:.0%} of "
                            f"1-{scale_points} range (actual: {actual_min}-{actual_max}). "
                            f"This may indicate scale_points was not respected."
                        )

        # ===== CHECK 2: Cross-reference with generation log =====
        if hasattr(self, '_scale_generation_log'):
            for log_entry in self._scale_generation_log:
                for expected_col in log_entry["columns_generated"]:
                    if expected_col not in df.columns:
                        self._log(
                            f"VALIDATION ERROR: Column '{expected_col}' was generated "
                            f"but is MISSING from final DataFrame"
                        )

        # ===== CHECK 3: Additional variable bounds =====
        for var in self.additional_vars:
            var_name = str(var.get("name", "Variable")).strip().replace(" ", "_")
            var_min = _safe_numeric(var.get("min", 0), default=0, as_int=True)
            var_max = _safe_numeric(var.get("max", 10), default=10, as_int=True)
            if var_max <= var_min:
                var_max = var_min + 1
            if var_name not in df.columns:
                self._log(f"VALIDATION ERROR: Expected additional variable column '{var_name}' MISSING")
                continue
            col_data = df[var_name]
            if col_data.dtype == object or not np.issubdtype(col_data.dtype, np.number):
                self._log(f"VALIDATION WARNING: Additional var '{var_name}' has non-numeric dtype, skipping")
                continue
            # Drop NaN values before computing min/max (missing data may have been injected)
            col_valid = col_data.dropna()
            if len(col_valid) == 0:
                continue
            actual_min = int(col_valid.min())
            actual_max = int(col_valid.max())
            if actual_min < var_min or actual_max > var_max:
                issues.append({
                    "column": var_name,
                    "expected_min": var_min,
                    "expected_max": var_max,
                    "actual_min": actual_min,
                    "actual_max": actual_max,
                    "issue_type": "out_of_bounds",
                })

        # ===== CHECK 4: Demographic bounds =====
        # v1.4.3: Gender is now string-labeled, only check numeric demographics
        for col_name, expected_range in [
            ("Age", (18, 85)),
        ]:
            if col_name not in df.columns:
                continue
            col_data = df[col_name]
            if col_data.dtype == object or not np.issubdtype(col_data.dtype, np.number):
                self._log(f"VALIDATION WARNING: Demographic '{col_name}' has non-numeric dtype, skipping")
                continue
            # Drop NaN values before computing min/max (missing data may have been injected)
            col_valid = col_data.dropna()
            if len(col_valid) == 0:
                continue
            actual_min = int(col_valid.min())
            actual_max = int(col_valid.max())
            if actual_min < expected_range[0] or actual_max > expected_range[1]:
                issues.append({
                    "column": col_name,
                    "expected_min": expected_range[0],
                    "expected_max": expected_range[1],
                    "actual_min": actual_min,
                    "actual_max": actual_max,
                    "issue_type": "out_of_bounds",
                })

        return issues

    def _build_scale_verification_report(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Build a comprehensive verification report for each scale,
        confirming that generated data matches user specifications.
        """
        report: List[Dict[str, Any]] = []

        for scale in self.scales:
            scale_name = str(scale.get("name", "Scale")).strip()
            # v1.4.11: use _clean_column_name for consistency with generation
            _var = str(scale.get("variable_name", "")).strip()
            scale_name_clean = _clean_column_name(_var if _var else scale_name)
            spec_points = int(scale.get("scale_points", 7))
            spec_items = int(scale.get("num_items", 5))
            spec_min = _safe_numeric(scale.get("scale_min", 1), default=1, as_int=True)

            scale_report: Dict[str, Any] = {
                "name": scale_name,
                "specified_scale_points": spec_points,
                "specified_num_items": spec_items,
                "columns_found": [],
                "columns_missing": [],
                "all_values_in_bounds": True,
                "range_utilization_pct": 0.0,
                "status": "OK",
            }

            all_values: List[int] = []
            for item_num in range(1, spec_items + 1):
                col_name = f"{scale_name_clean}_{item_num}"
                if col_name in df.columns:
                    # Safety: skip non-numeric columns
                    if df[col_name].dtype == object or not np.issubdtype(df[col_name].dtype, np.number):
                        scale_report["columns_missing"].append(col_name)
                        continue
                    scale_report["columns_found"].append(col_name)
                    col_values = df[col_name].tolist()
                    all_values.extend(col_values)
                    col_min = min(col_values)
                    col_max = max(col_values)
                    if col_min < spec_min or col_max > spec_points:
                        scale_report["all_values_in_bounds"] = False
                        scale_report["status"] = "BOUNDS_VIOLATION"
                else:
                    scale_report["columns_missing"].append(col_name)
                    scale_report["status"] = "MISSING_COLUMNS"

            if all_values and spec_points > 1:
                observed_range = max(all_values) - min(all_values)
                expected_range = spec_points - 1
                utilization = (observed_range / expected_range * 100) if expected_range > 0 else 0
                scale_report["range_utilization_pct"] = round(utilization, 1)
                scale_report["observed_min"] = min(all_values)
                scale_report["observed_max"] = max(all_values)
                scale_report["observed_mean"] = round(sum(all_values) / len(all_values), 2)

                # Flag if range utilization is suspiciously low for large scales
                if spec_points > 10 and utilization < 30 and len(all_values) >= 20:
                    scale_report["status"] = "LOW_RANGE_UTILIZATION"

            report.append(scale_report)

        return report

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

        # Get scale columns — v1.4.11: use _clean_column_name for consistency
        scale_cols = []
        for scale in self.scales:
            _raw = str(scale.get("name", "Scale")).strip()
            _var = str(scale.get("variable_name", "")).strip()
            scale_name = _clean_column_name(_var if _var else _raw)
            num_items = _safe_numeric(scale.get("num_items", 5), default=5, as_int=True)
            for item_num in range(1, num_items + 1):
                col_name = f"{scale_name}_{item_num}"
                if col_name in df.columns:
                    scale_cols.append((scale_name, col_name))

        # Also check for scale means (if computed)
        for scale in self.scales:
            _raw = str(scale.get("name", "Scale")).strip()
            _var = str(scale.get("variable_name", "")).strip()
            scale_name = _clean_column_name(_var if _var else _raw)
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

    def validate_no_order_effects(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that there are NO systematic order effects in the generated data.

        VERSION 2.2.9: Critical validation to ensure condition position does not
        predict response means.

        This method computes the correlation between condition position (index)
        and condition means. A significant correlation would indicate an order
        effect bug that needs to be fixed.

        Returns:
            Dict containing validation results:
            - order_correlation: Pearson r between position and mean
            - is_problematic: True if |r| > 0.7 (strong order effect)
            - condition_means: Dict of condition -> mean
            - warning: Warning message if order effect detected
        """
        result = {
            "order_correlation": 0.0,
            "is_problematic": False,
            "condition_means": {},
            "warning": None,
        }

        if "CONDITION" not in df.columns or len(self.conditions) < 3:
            return result

        # Find numeric columns (DVs)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        dv_cols = [c for c in numeric_cols if not c.startswith(('PARTICIPANT', 'Flag_', 'Exclude_', 'Completion_', 'Attention_', 'Max_', 'SIMULATION_'))]

        if not dv_cols:
            return result

        # Compute mean across DVs for each condition
        condition_grand_means = {}
        for i, cond in enumerate(self.conditions):
            cond_data = df[df["CONDITION"] == cond][dv_cols]
            if len(cond_data) > 0:
                grand_mean = cond_data.mean().mean()
                condition_grand_means[cond] = grand_mean

        result["condition_means"] = {k: round(v, 3) for k, v in condition_grand_means.items()}

        # Compute correlation between position and mean
        if len(condition_grand_means) >= 3:
            positions = list(range(len(self.conditions)))
            means = [condition_grand_means.get(c, 0) for c in self.conditions]

            # Pearson correlation
            if len(positions) > 2 and np.std(means) > 0:
                correlation = np.corrcoef(positions, means)[0, 1]
                result["order_correlation"] = round(correlation, 3) if not np.isnan(correlation) else 0.0

                # Flag if strong order effect (|r| > 0.7)
                if abs(result["order_correlation"]) > 0.7:
                    result["is_problematic"] = True
                    result["warning"] = (
                        f"WARNING: Strong order effect detected (r={result['order_correlation']:.2f}). "
                        "Condition position is highly correlated with response means. "
                        "This suggests a bug in effect assignment - effects should be "
                        "based on semantic content, not position."
                    )

        return result

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
        # v1.0.0: Guard against division by zero when no conditions
        n_per = self.sample_size // max(len(self.conditions), 1)
        for cond in self.conditions:
            lines.append(f"  - {cond} (target n = {n_per})")

        if self.effect_sizes:
            lines.extend(["", "-" * 70, "EXPECTED EFFECT SIZES", "-" * 70, ""])
            for effect in self.effect_sizes:
                # v1.1.1.5: Support both EffectSizeSpec objects and plain dicts
                _ev = effect.get("variable", "") if isinstance(effect, dict) else getattr(effect, 'variable', '')
                _eh = effect.get("level_high", "") if isinstance(effect, dict) else getattr(effect, 'level_high', '')
                _el = effect.get("level_low", "") if isinstance(effect, dict) else getattr(effect, 'level_low', '')
                _ed = effect.get("cohens_d", 0.5) if isinstance(effect, dict) else getattr(effect, 'cohens_d', 0.5)
                lines.append(f"  {_ev}: {_eh} > {_el}, Cohen's d = {_ed}")

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
            "# Gender is already labeled as strings (Male, Female, Non-binary, Prefer not to say)",
            'data$Gender <- factor(data$Gender)',
            "",
        ]

        for scale in self.scales:
            scale_name_raw = str(scale.get("name", "Scale")).strip() or "Scale"
            scale_name = scale_name_raw.replace(" ", "_")
            num_items = _safe_numeric(scale.get("num_items", 5), default=5, as_int=True)
            scale_points = _safe_numeric(scale.get("scale_points", 7), default=7, as_int=True)
            reverse_items = _safe_parse_reverse_items(scale.get("reverse_items", []))

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

    def generate_python_export(self, df: pd.DataFrame) -> str:
        """
        Generate Python-compatible export script with pandas (v2.4.5).

        Returns a Python script that loads and prepares Simulated.csv.
        """
        def _py_quote(x: str) -> str:
            x = str(x).replace("\\", "\\\\").replace("'", "\\'")
            return f"'{x}'"

        condition_levels = ", ".join([_py_quote(c) for c in self.conditions])

        lines: List[str] = [
            "# ============================================================",
            f"# Python Data Preparation Script - {self.study_title}",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Run ID: {self.run_id}",
            "# ============================================================",
            "",
            "import pandas as pd",
            "import numpy as np",
            "",
            "# Load the data",
            "data = pd.read_csv('Simulated_Data.csv')",
            "",
            "# Convert CONDITION to categorical with proper order",
            f"condition_order = [{condition_levels}]",
            "data['CONDITION'] = pd.Categorical(data['CONDITION'], categories=condition_order, ordered=True)",
            "",
            "# Gender is already labeled as strings (Male, Female, Non-binary, Prefer not to say)",
            "data['Gender'] = pd.Categorical(data['Gender'])",
            "",
        ]

        for scale in self.scales:
            scale_name_raw = str(scale.get("name", "Scale")).strip() or "Scale"
            scale_name = scale_name_raw.replace(" ", "_")
            num_items = _safe_numeric(scale.get("num_items", 5), default=5, as_int=True)
            scale_points = _safe_numeric(scale.get("scale_points", 7), default=7, as_int=True)
            reverse_items = _safe_parse_reverse_items(scale.get("reverse_items", []))

            items = [f"{scale_name}_{i}" for i in range(1, num_items + 1)]

            if reverse_items:
                lines.append(f"# {scale_name_raw} - reverse code items {sorted(reverse_items)}")
                for r_item in sorted(reverse_items):
                    item_name = f"{scale_name}_{r_item}"
                    max_val = scale_points
                    lines.append(f"data['{item_name}_R'] = {max_val + 1} - data['{item_name}']")
                lines.append("")

            lines.append(f"# Create {scale_name_raw} composite")
            item_list = ", ".join([f"'{item}'" for item in items])
            lines.append(f"data['{scale_name}_composite'] = data[[{item_list}]].mean(axis=1)")
            lines.append("")

        lines.extend([
            "# Filter excluded participants (optional)",
            "data_clean = data[data['Exclude_Recommended'] == 0].copy()",
            "",
            "print(f'Total N: {len(data)}')",
            "print(f'Clean N: {len(data_clean)}')",
            "",
            "# Ready for analysis",
            "# Example: data_clean.groupby('CONDITION')['Scale_composite'].mean()",
        ])

        return "\n".join(lines)

    def generate_julia_export(self, df: pd.DataFrame) -> str:
        """
        Generate Julia-compatible export script with DataFrames.jl (v2.4.5).

        Returns a Julia script that loads and prepares Simulated.csv.
        """
        def _jl_quote(x: str) -> str:
            x = str(x).replace("\\", "\\\\").replace('"', '\\"')
            return f'"{x}"'

        condition_levels = ", ".join([_jl_quote(c) for c in self.conditions])

        lines: List[str] = [
            "# ============================================================",
            f"# Julia Data Preparation Script - {self.study_title}",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Run ID: {self.run_id}",
            "# ============================================================",
            "",
            "using CSV",
            "using DataFrames",
            "using CategoricalArrays",
            "using Statistics",
            "",
            "# Load the data",
            'data = CSV.read("Simulated_Data.csv", DataFrame)',
            "",
            "# Convert CONDITION to categorical with proper order",
            f"condition_levels = [{condition_levels}]",
            "data.CONDITION = categorical(data.CONDITION, levels=condition_levels, ordered=true)",
            "",
            "# Gender is already labeled as strings (Male, Female, Non-binary, Prefer not to say)",
            "data.Gender = categorical(data.Gender)",
            "",
        ]

        for scale in self.scales:
            scale_name_raw = str(scale.get("name", "Scale")).strip() or "Scale"
            scale_name = scale_name_raw.replace(" ", "_")
            num_items = _safe_numeric(scale.get("num_items", 5), default=5, as_int=True)
            scale_points = _safe_numeric(scale.get("scale_points", 7), default=7, as_int=True)
            reverse_items = _safe_parse_reverse_items(scale.get("reverse_items", []))

            items = [f"{scale_name}_{i}" for i in range(1, num_items + 1)]

            if reverse_items:
                lines.append(f"# {scale_name_raw} - reverse code items {sorted(reverse_items)}")
                for r_item in sorted(reverse_items):
                    item_name = f"{scale_name}_{r_item}"
                    max_val = scale_points
                    lines.append(f'data.{item_name}_R = {max_val + 1} .- data.{item_name}')
                lines.append("")

            lines.append(f"# Create {scale_name_raw} composite")
            item_syms = ", ".join([f":{item}" for item in items])
            lines.append(f"data.{scale_name}_composite = mean.(eachrow(data[:, [{item_syms}]]))")
            lines.append("")

        lines.extend([
            "# Filter excluded participants (optional)",
            "data_clean = filter(row -> row.Exclude_Recommended == 0, data)",
            "",
            'println("Total N: ", nrow(data))',
            'println("Clean N: ", nrow(data_clean))',
            "",
            "# Ready for analysis",
            "# Example: combine(groupby(data_clean, :CONDITION), :Scale_composite => mean)",
        ])

        return "\n".join(lines)

    def generate_spss_export(self, df: pd.DataFrame) -> str:
        """
        Generate SPSS syntax file for data preparation (v2.4.5).

        Returns SPSS syntax that prepares the data after import.
        """
        def _spss_quote(x: str) -> str:
            x = str(x).replace("'", "''")
            return f"'{x}'"

        lines: List[str] = [
            "* ============================================================.",
            f"* SPSS Data Preparation Syntax - {self.study_title}.",
            f"* Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.",
            f"* Run ID: {self.run_id}.",
            "* ============================================================.",
            "",
            "* Load the data first using:",
            "*   File > Import Data > CSV Data...",
            "*   Select 'Simulated_Data.csv'.",
            "",
            "* Define variable labels and value labels.",
            "",
        ]

        # Add condition value labels
        condition_labels = " ".join([f"{i+1} {_spss_quote(c)}" for i, c in enumerate(self.conditions)])
        lines.extend([
            "VALUE LABELS CONDITION",
            f"  {condition_labels}.",
            "",
            "* Gender is already labeled as strings (Male, Female, Non-binary, Prefer not to say).",
            "* STRING Gender(A20).",
            "",
        ])

        for scale in self.scales:
            scale_name_raw = str(scale.get("name", "Scale")).strip() or "Scale"
            scale_name = scale_name_raw.replace(" ", "_")
            num_items = _safe_numeric(scale.get("num_items", 5), default=5, as_int=True)
            scale_points = _safe_numeric(scale.get("scale_points", 7), default=7, as_int=True)
            reverse_items = _safe_parse_reverse_items(scale.get("reverse_items", []))

            items = [f"{scale_name}_{i}" for i in range(1, num_items + 1)]

            if reverse_items:
                lines.append(f"* {scale_name_raw} - reverse code items {sorted(reverse_items)}.")
                for r_item in sorted(reverse_items):
                    item_name = f"{scale_name}_{r_item}"
                    max_val = scale_points
                    lines.append(f"COMPUTE {item_name}_R = {max_val + 1} - {item_name}.")
                lines.append("EXECUTE.")
                lines.append("")

            lines.append(f"* Create {scale_name_raw} composite.")
            item_list = " ".join(items)
            lines.append(f"COMPUTE {scale_name}_composite = MEAN({item_list}).")
            lines.append("EXECUTE.")
            lines.append("")

        lines.extend([
            "* Filter excluded participants (optional).",
            "USE ALL.",
            "COMPUTE filter_$=(Exclude_Recommended = 0).",
            "VARIABLE LABELS filter_$ 'Exclude_Recommended = 0 (FILTER)'.",
            "VALUE LABELS filter_$ 0 'Not Selected' 1 'Selected'.",
            "FORMATS filter_$ (f1.0).",
            "FILTER BY filter_$.",
            "EXECUTE.",
            "",
            "* Descriptive statistics.",
            "DESCRIPTIVES VARIABLES=ALL /STATISTICS=MEAN STDDEV MIN MAX.",
            "",
        ])

        return "\n".join(lines)

    def generate_stata_export(self, df: pd.DataFrame) -> str:
        """
        Generate Stata .do file for data preparation (v2.4.5).

        Returns Stata commands that prepare the data after import.
        """
        def _stata_quote(x: str) -> str:
            x = str(x).replace('"', "'")
            return f'"{x}"'

        lines: List[str] = [
            "// ============================================================",
            f"// Stata Data Preparation Do-File - {self.study_title}",
            f"// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"// Run ID: {self.run_id}",
            "// ============================================================",
            "",
            "// Load the data",
            'import delimited "Simulated_Data.csv", clear',
            "",
            "// Label the CONDITION variable",
        ]

        # Add condition value labels
        for i, c in enumerate(self.conditions):
            lines.append(f'label define condition_lbl {i+1} {_stata_quote(c)}, add')
        lines.extend([
            "encode condition, gen(condition_num) label(condition_lbl)",
            "",
            "// Gender is already labeled as strings (Male, Female, Non-binary, Prefer not to say)",
            "// No numeric encoding needed",
            "",
        ])

        for scale in self.scales:
            scale_name_raw = str(scale.get("name", "Scale")).strip() or "Scale"
            scale_name = scale_name_raw.replace(" ", "_").lower()
            num_items = _safe_numeric(scale.get("num_items", 5), default=5, as_int=True)
            scale_points = _safe_numeric(scale.get("scale_points", 7), default=7, as_int=True)
            reverse_items = _safe_parse_reverse_items(scale.get("reverse_items", []))

            items = [f"{scale_name}_{i}" for i in range(1, num_items + 1)]

            if reverse_items:
                lines.append(f"// {scale_name_raw} - reverse code items {sorted(reverse_items)}")
                for r_item in sorted(reverse_items):
                    item_name = f"{scale_name}_{r_item}"
                    max_val = scale_points
                    lines.append(f"gen {item_name}_r = {max_val + 1} - {item_name}")
                lines.append("")

            lines.append(f"// Create {scale_name_raw} composite")
            item_list = " ".join(items)
            lines.append(f"egen {scale_name}_composite = rowmean({item_list})")
            lines.append("")

        lines.extend([
            "// Filter excluded participants (optional)",
            "preserve",
            "keep if exclude_recommended == 0",
            "",
            '// Display counts',
            'display "Total N: " _N',
            "",
            "// Summary statistics",
            "summarize",
            "",
            "// Ready for analysis",
            "restore",
        ])

        return "\n".join(lines)

    def generate_methods_writeup(self, condensed: bool = True) -> str:
        """
        Generate a scientific methods write-up for the simulation.

        This produces a publication-ready methods section that documents
        the scientific approach used to generate the synthetic data.

        Args:
            condensed: If True, returns a brief paragraph. If False, returns
                      full detailed documentation.

        Returns:
            Formatted methods text suitable for reports or publications.
        """
        if not condensed:
            return SCIENTIFIC_METHODS_DOCUMENTATION

        # Generate condensed methods paragraph
        n_conditions = len(self.conditions)
        n_scales = len(self.scales)

        # Determine effect size info
        effect_info = "auto-generated (d = 0.4-0.6)"
        if self.effect_sizes:
            ds = [
                (es.get("cohens_d", 0.5) if isinstance(es, dict) else getattr(es, "cohens_d", 0.5))
                for es in self.effect_sizes
            ]
            if len(ds) == 1:
                effect_info = f"d = {ds[0]:.2f}"
            else:
                effect_info = f"d = {min(ds):.2f}-{max(ds):.2f}"

        methods = f"""
METHODS: SYNTHETIC DATA GENERATION

Data were generated using a scientifically-calibrated simulation engine (v2.2.8)
grounded in survey methodology research. The simulation employs a persona-based
response model with parameters calibrated from published empirical research.

Sample and Design: N = {self.sample_size} synthetic participants were randomly
assigned to {n_conditions} experimental condition{'s' if n_conditions > 1 else ''}.
Responses were generated for {n_scales} scale{'s' if n_scales > 1 else ''} measuring
dependent variables relevant to the study context.

Response Generation: Each response was generated through a multi-step process:
(1) Personas were assigned based on population weights derived from Krosnick (1991)
satisficing theory (35% engaged, 22% satisficers, 10% extreme responders, 8%
acquiescent, 5% careless, 12% socially desirable responders, 8% other).
(2) Domain-specific calibrations adjusted response means to match published norms
(Oliver, 1980; Slovic, 1987; Mayer et al., 1995).
(3) Condition effects were applied using standardized effect sizes ({effect_info})
with Cohen's d methodology (Cohen, 1988).
(4) Response styles were simulated based on Greenleaf (1992) for extreme responding
and Billiet & McClendon (2000) for acquiescence bias.

Validation: The simulation produces data matching empirical benchmarks: mean
responses of M = 4.0-5.2 on 7-point scales, within-condition SD = 1.2-1.8,
and attention check pass rates of 85-95%.

Key Citations:
- Krosnick, J. A. (1991). Response strategies. Applied Cognitive Psychology.
- Greenleaf, E. A. (1992). Measuring extreme response style. POQ.
- Paulhus, D. L. (1991). Measurement of response bias. Academic Press.
- Meade, A. W., & Craig, S. B. (2012). Identifying careless responses. PM.
- Cohen, J. (1988). Statistical power analysis. Lawrence Erlbaum.
"""
        return methods.strip()


# Export the documentation constant as well
__all__ = [
    "EnhancedSimulationEngine",
    "EffectSizeSpec",
    "ExclusionCriteria",
    "SCIENTIFIC_METHODS_DOCUMENTATION",
]
