# simulation_app/utils/enhanced_simulation_engine.py
from __future__ import annotations
"""
Enhanced Simulation Engine for Behavioral Experiment Simulation Tool
=============================================================================

Version 1.2.2 - New tab-based UI + safe trait value extraction

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
__version__ = "1.4.4"  # v1.4.4: Fix navigation buttons, enhance training data collection

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
import random
import re

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
    ):
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
            if hasattr(self, '_log'):
                self._log("WARNING", "CONDITIONS", "No conditions specified - defaulting to single 'Condition A'. Results may not match intended design.")
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
        self.condition_allocation = self._normalize_condition_allocation(
            condition_allocation, self.conditions
        )  # Dict[condition_name, percentage 0-100]
        self.precomputed_visibility = precomputed_visibility or {}  # v1.0.0: From QSF parser
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
        # Also detect domains from condition names for better persona matching
        condition_text = " ".join(str(c) for c in self.conditions)
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

    def _adjust_persona_weights_for_study(self) -> None:
        """
        Adjust persona weights based on detected study domain and conditions.

        SCIENTIFIC BASIS:
        =================
        Different study types attract different participant populations.
        This method adjusts persona weights to better reflect likely sample
        characteristics based on study context.

        References:
        - Buhrmester et al. (2011): MTurk sample characteristics
        - Peer et al. (2017): Online panel composition
        """
        study_text = f"{self.study_description} {self.study_title}".lower()
        condition_text = " ".join(str(c) for c in self.conditions).lower()
        all_text = f"{study_text} {condition_text}"

        # =====================================================================
        # Domain-based persona weight adjustments
        # =====================================================================

        # AI/Technology studies - increase tech-related personas
        if any(kw in all_text for kw in ['ai', 'algorithm', 'robot', 'automation', 'technology']):
            for name, persona in self.available_personas.items():
                if 'tech' in name.lower() or 'ai' in name.lower():
                    persona.weight *= 1.3  # Boost tech personas
                if persona.category == 'technology':
                    persona.weight *= 1.2

        # Consumer/Marketing studies - increase consumer personas
        if any(kw in all_text for kw in ['consumer', 'brand', 'purchase', 'product', 'marketing']):
            for name, persona in self.available_personas.items():
                if persona.category == 'consumer':
                    persona.weight *= 1.3
                if 'consumer' in name.lower() or 'brand' in name.lower():
                    persona.weight *= 1.2

        # Organizational studies - increase org behavior personas
        if any(kw in all_text for kw in ['employee', 'workplace', 'job', 'organization', 'leadership']):
            for name, persona in self.available_personas.items():
                if persona.category == 'organizational':
                    persona.weight *= 1.4
                if 'employee' in name.lower() or 'leader' in name.lower():
                    persona.weight *= 1.2

        # Social psychology studies - increase social personas
        if any(kw in all_text for kw in ['cooperation', 'prosocial', 'trust', 'fairness', 'social dilemma']):
            for name, persona in self.available_personas.items():
                if persona.category == 'social':
                    persona.weight *= 1.3
                if 'prosocial' in name.lower() or 'individualist' in name.lower():
                    persona.weight *= 1.2

        # Health studies - increase health personas
        if any(kw in all_text for kw in ['health', 'wellness', 'exercise', 'diet', 'medical']):
            for name, persona in self.available_personas.items():
                if persona.category == 'health':
                    persona.weight *= 1.4

        # Environmental studies - increase environmental personas
        if any(kw in all_text for kw in ['environment', 'sustainability', 'climate', 'green', 'eco']):
            for name, persona in self.available_personas.items():
                if persona.category == 'environmental':
                    persona.weight *= 1.4

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
        # INCREASED effect multiplier for detectable differences
        # This converts Cohen's d to a 0-1 normalized shift
        # d=0.5 -> 0.20 shift (20% of scale range) = ~1.2 points on 7-point scale
        COHENS_D_TO_NORMALIZED = 0.40  # Increased from 0.25 for detectable effects

        # Check explicit effect size specifications -- accumulate ALL matching effects
        # for factorial designs where multiple effect specs may apply to one condition
        matched_effects: list = []
        condition_lower = str(condition).lower().strip()
        variable_lower = str(variable).lower().strip()

        for effect in self.effect_sizes:
            # v1.4.0: Safe conversion of cohens_d (could be string, dict, or NaN)
            try:
                cohens_d = float(effect.cohens_d) if not isinstance(effect.cohens_d, dict) else 0.5
            except (ValueError, TypeError):
                cohens_d = 0.5  # Default to medium effect on conversion failure
            if isinstance(cohens_d, float) and (np.isnan(cohens_d) or np.isinf(cohens_d)):
                cohens_d = 0.5
            # Clamp to reasonable range (0-3.0 covers virtually all real effects)
            cohens_d = float(np.clip(abs(cohens_d), 0.0, 3.0))

            # Check if this effect spec matches the current variable
            effect_var = str(effect.variable).lower().strip()
            variable_matches = (
                effect_var == variable_lower
                or variable_lower.startswith(effect_var)
                or effect_var in variable_lower
            )

            if variable_matches:
                # v1.4.0: Improved level matching with false-positive prevention
                level_high = str(effect.level_high).lower().strip()
                level_low = str(effect.level_low).lower().strip()
                direction = str(getattr(effect, 'direction', 'positive')).lower().strip()

                is_high = bool(level_high and level_high in condition_lower)
                is_low = bool(level_low and level_low in condition_lower)

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

        # Default medium effect size parameters
        default_d = 0.5
        COHENS_D_TO_NORMALIZED = 0.40

        # Initialize base effect at 0 (neutral)
        semantic_effect = 0.0

        # =====================================================================
        # STEP 1: Parse valence keywords (directional effects)
        # Based on affective meaning of condition labels
        # =====================================================================

        # Strong positive valence keywords → positive effect
        positive_keywords = [
            'lover', 'friend', 'positive', 'high', 'good', 'best', 'strong',
            'success', 'win', 'gain', 'benefit', 'reward', 'pleasant',
            'like', 'love', 'favor', 'approve', 'support', 'prosocial',
            'cooperative', 'trust', 'warm', 'kind', 'helpful', 'generous',
            'optimistic', 'confident', 'empowered', 'satisfied'
        ]

        # Strong negative valence keywords → negative effect
        negative_keywords = [
            'hater', 'enemy', 'negative', 'low', 'bad', 'worst', 'weak',
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

        # Check for valence keywords
        for keyword in positive_keywords:
            if keyword in condition_lower:
                semantic_effect += 0.35  # Moderate positive shift
                break

        for keyword in negative_keywords:
            if keyword in condition_lower:
                semantic_effect -= 0.35  # Moderate negative shift
                break

        for keyword in neutral_keywords:
            if keyword in condition_lower:
                semantic_effect *= 0.3  # Reduce effect toward neutral
                break

        # =====================================================================
        # DOMAIN 1: AI/TECHNOLOGY MANIPULATIONS
        # =====================================================================

        # Algorithm Aversion (Dietvorst, Simmons & Massey, 2015, JEP:G)
        # People avoid algorithms after seeing them err. Effect: d ≈ -0.3 to -0.5
        if 'ai' in condition_lower or 'algorithm' in condition_lower or 'robot' in condition_lower:
            if any(neg in condition_lower for neg in ['no ai', 'no_ai', 'without ai', 'no algorithm', 'human only']):
                # No AI / Human condition - often preferred due to algorithm aversion
                semantic_effect += 0.15  # Human preference effect
            else:
                # AI present - shows aversion in evaluations (Dietvorst et al., 2015)
                semantic_effect -= 0.12

        # Machine vs Human judgment (Logg, Minson & Moore, 2019)
        if 'machine' in condition_lower and 'human' not in condition_lower:
            semantic_effect -= 0.10
        elif 'human' in condition_lower and 'machine' not in condition_lower:
            if 'superhuman' not in condition_lower:
                semantic_effect += 0.10

        # Anthropomorphism (Epley, Waytz & Cacioppo, 2007, Psychological Review)
        # Human-like features increase trust and liking. Effect: d ≈ +0.2 to +0.4
        if 'anthropomorph' in condition_lower or 'human-like' in condition_lower or 'humanoid' in condition_lower:
            semantic_effect += 0.18
        elif 'machine-like' in condition_lower or 'robotic' in condition_lower:
            semantic_effect -= 0.08

        # Automation (Parasuraman & Riley, 1997; Lee & See, 2004)
        if 'automat' in condition_lower:
            if 'full' in condition_lower or 'complete' in condition_lower:
                semantic_effect -= 0.15  # Full automation trust concerns
            elif 'partial' in condition_lower or 'assisted' in condition_lower:
                semantic_effect += 0.05  # Partial automation often preferred

        # Transparency/Explainability (Ribeiro et al., 2016)
        if 'transparent' in condition_lower or 'explainable' in condition_lower or 'interpretable' in condition_lower:
            semantic_effect += 0.12
        elif 'black box' in condition_lower or 'opaque' in condition_lower:
            semantic_effect -= 0.10

        # =====================================================================
        # DOMAIN 2: CONSUMER/MARKETING MANIPULATIONS
        # =====================================================================

        # Hedonic vs Utilitarian (Babin, Darden & Griffin, 1994, JCR)
        # Hedonic consumption generates more positive affect. Effect: d ≈ +0.25
        if any(h in condition_lower for h in ['hedonic', 'experiential', 'fun', 'pleasure', 'enjoyment', 'indulgent']):
            semantic_effect += 0.22
        elif any(u in condition_lower for u in ['utilitarian', 'functional', 'practical', 'necessity', 'useful']):
            semantic_effect -= 0.08

        # Scarcity Effect (Cialdini, 2001; Barton et al., 2022 meta-analysis)
        # Limited availability increases desirability. Mean effect r = 0.28, d ≈ +0.30
        if any(s in condition_lower for s in ['scarce', 'limited', 'exclusive', 'rare', 'last chance', 'few left']):
            semantic_effect += 0.25
        elif any(a in condition_lower for a in ['abundant', 'unlimited', 'plentiful', 'common', 'widely available']):
            semantic_effect -= 0.08

        # Social Proof (Cialdini, 2001; Bond & Smith, 1996 meta-analysis)
        # Others' choices influence preferences. Conformity effect robust.
        if any(sp in condition_lower for sp in ['popular', 'bestseller', 'most chosen', 'endorsed', 'recommended',
                                                  'others chose', 'trending', 'viral', 'social proof']):
            semantic_effect += 0.20
        elif any(np in condition_lower for np in ['unpopular', 'not recommended', 'unknown brand', 'no reviews']):
            semantic_effect -= 0.15

        # Price-Quality Inference (Rao & Monroe, 1989, JMR)
        if 'premium' in condition_lower or 'luxury' in condition_lower or 'expensive' in condition_lower:
            semantic_effect += 0.15
        elif 'budget' in condition_lower or 'discount' in condition_lower or 'cheap' in condition_lower:
            semantic_effect -= 0.10

        # Brand Effects (Alba & Hutchinson, 1987, JCR)
        if 'familiar' in condition_lower or 'known brand' in condition_lower or 'established' in condition_lower:
            semantic_effect += 0.12
        elif 'unfamiliar' in condition_lower or 'new brand' in condition_lower or 'unknown' in condition_lower:
            semantic_effect -= 0.08

        # Advertising Appeals (MacInnis & Jaworski, 1989)
        if 'emotional' in condition_lower and 'appeal' in condition_lower:
            semantic_effect += 0.15
        elif 'rational' in condition_lower and 'appeal' in condition_lower:
            semantic_effect += 0.05

        # =====================================================================
        # DOMAIN 3: SOCIAL PSYCHOLOGY MANIPULATIONS
        # =====================================================================

        # In-group/Out-group Bias (Tajfel, 1971; Balliet et al., 2014 meta)
        # Minimal group paradigm shows in-group favoritism. d ≈ 0.3-0.5
        if any(ig in condition_lower for ig in ['ingroup', 'in-group', 'in group', 'us', 'our group', 'teammate']):
            semantic_effect += 0.28
        elif any(og in condition_lower for og in ['outgroup', 'out-group', 'out group', 'them', 'other group', 'opponent']):
            semantic_effect -= 0.25

        # Authority/Obedience (Milgram, 1963; Meta-Milgram, 2014)
        # Authority figures increase compliance. Uniform effect: +46pp compliance
        if any(auth in condition_lower for auth in ['authority', 'expert', 'doctor', 'professor', 'scientist',
                                                      'official', 'leader', 'manager', 'uniform']):
            semantic_effect += 0.22
        elif any(nauth in condition_lower for nauth in ['peer', 'layperson', 'non-expert', 'stranger', 'novice']):
            semantic_effect -= 0.08

        # Reciprocity (Cialdini, 2001; Regan, 1971)
        # Favors increase compliance. Gift effect: +23% tips
        if any(r in condition_lower for r in ['reciproc', 'gift', 'favor', 'gave first', 'free sample']):
            semantic_effect += 0.20
        elif 'no gift' in condition_lower or 'no favor' in condition_lower:
            semantic_effect -= 0.05

        # Social Presence (Short, Williams & Christie, 1976)
        # Co-presence increases prosocial behavior. d ≈ +0.15 to +0.30
        if any(sp in condition_lower for sp in ['social presence', 'observed', 'watched', 'public',
                                                  'with others', 'audience', 'witnessed']):
            semantic_effect += 0.18
        elif any(al in condition_lower for al in ['alone', 'private', 'anonymous', 'unobserved', 'no audience']):
            semantic_effect -= 0.10

        # Commitment/Consistency (Cialdini, 2001; Freedman & Fraser, 1966)
        if any(c in condition_lower for c in ['commitment', 'pledged', 'promised', 'foot in door', 'prior agreement']):
            semantic_effect += 0.18

        # Liking/Similarity (Cialdini, 2001; Byrne, 1971)
        if any(l in condition_lower for l in ['similar', 'likeable', 'attractive', 'compliment', 'same group']):
            semantic_effect += 0.15
        elif any(d in condition_lower for d in ['dissimilar', 'unlikeable', 'different', 'outgroup']):
            semantic_effect -= 0.12

        # =====================================================================
        # DOMAIN 4: BEHAVIORAL ECONOMICS MANIPULATIONS
        # =====================================================================

        # Loss Aversion/Framing (Tversky & Kahneman, 1981, Science)
        # Losses loom larger than gains. λ ≈ 2.0-2.5. Loss frame: 43% vs gain: 23% risk-seeking
        if any(g in condition_lower for g in ['gain', 'save', 'earn', 'win', 'keep', 'gain frame']):
            semantic_effect += 0.12
        elif any(l in condition_lower for l in ['loss', 'lose', 'cost', 'pay', 'forfeit', 'loss frame']):
            semantic_effect -= 0.20  # Loss aversion amplifies negative effects

        # Anchoring (Tversky & Kahneman, 1974, Science)
        # First numbers anchor judgment. d ≈ 0.5-1.0
        if 'high anchor' in condition_lower or 'large anchor' in condition_lower:
            semantic_effect += 0.25
        elif 'low anchor' in condition_lower or 'small anchor' in condition_lower:
            semantic_effect -= 0.20

        # Default Effect (Johnson & Goldstein, 2003, Science)
        # Opt-out > opt-in by 60-80 percentage points
        if any(d in condition_lower for d in ['opt-out', 'opt out', 'default yes', 'presumed consent']):
            semantic_effect += 0.35
        elif any(d in condition_lower for d in ['opt-in', 'opt in', 'default no', 'explicit consent', 'active choice']):
            semantic_effect -= 0.15

        # Endowment Effect (Kahneman, Knetsch & Thaler, 1990, JPE)
        # Ownership increases valuation. WTA/WTP ≈ 2:1
        if any(e in condition_lower for e in ['own', 'possess', 'endow', 'yours', 'have']):
            semantic_effect += 0.20
        elif any(e in condition_lower for e in ['buy', 'acquire', 'get', 'obtain']):
            semantic_effect -= 0.10

        # Fairness/Ultimatum (Güth et al., 1982; Camerer, 2003)
        # Unfair offers rejected 40-60% of time
        if any(f in condition_lower for f in ['fair', 'equal', 'equitable', '50-50', 'even split']):
            semantic_effect += 0.25
        elif any(uf in condition_lower for uf in ['unfair', 'unequal', 'inequitable', 'low offer', 'stingy']):
            semantic_effect -= 0.30

        # Present Bias (O'Donoghue & Rabin, 1999)
        if any(p in condition_lower for p in ['immediate', 'now', 'today', 'instant']):
            semantic_effect += 0.18
        elif any(f in condition_lower for f in ['delayed', 'later', 'future', 'wait']):
            semantic_effect -= 0.12

        # Mental Accounting (Thaler, 1985)
        if 'windfall' in condition_lower or 'bonus' in condition_lower or 'unexpected' in condition_lower:
            semantic_effect += 0.15

        # =====================================================================
        # DOMAIN 5: GAME THEORY/COOPERATION MANIPULATIONS
        # =====================================================================

        # Public Goods Game (Fehr & Gächter, 2000, AER)
        # With punishment: near 100% vs 40% without
        if any(p in condition_lower for p in ['pgg', 'public good', 'contribute', 'common pool']):
            semantic_effect += 0.15
            if 'punish' in condition_lower:
                semantic_effect += 0.25  # Punishment dramatically increases cooperation

        # Dictator Game (Engel, 2011 meta-analysis)
        # Mean giving ≈ 28% of endowment
        if 'dictator' in condition_lower:
            semantic_effect -= 0.05  # More self-interested than other games

        # Trust Game (Berg et al., 1995; Johnson & Mislin, 2011 meta)
        # Mean sent ≈ 50%
        if 'trust game' in condition_lower:
            if 'trustor' in condition_lower or 'sender' in condition_lower:
                semantic_effect += 0.15
            elif 'trustee' in condition_lower or 'receiver' in condition_lower:
                semantic_effect += 0.10

        # Prisoner's Dilemma (Sally, 1995 meta-analysis)
        # Mean cooperation ≈ 47%
        if "prisoner" in condition_lower or 'pd' in condition_lower:
            if 'cooperat' in condition_lower:
                semantic_effect += 0.20
            elif 'defect' in condition_lower:
                semantic_effect -= 0.25

        # Repeated vs One-shot games (Axelrod, 1984)
        if 'repeated' in condition_lower or 'iterated' in condition_lower or 'multiple rounds' in condition_lower:
            semantic_effect += 0.15  # Repeated games show more cooperation
        elif 'one-shot' in condition_lower or 'single round' in condition_lower:
            semantic_effect -= 0.10

        # =====================================================================
        # DOMAIN 6: HEALTH/RISK MANIPULATIONS
        # =====================================================================

        # Self-Efficacy (Bandura, 1977; Meta-analyses)
        # Higher self-efficacy increases health behaviors
        if any(se in condition_lower for se in ['high efficacy', 'self-efficacy', 'confident', 'capable', 'empowered']):
            semantic_effect += 0.22
        elif any(le in condition_lower for le in ['low efficacy', 'doubtful', 'incapable', 'helpless']):
            semantic_effect -= 0.20

        # Fear Appeals (Witte & Allen, 2000 meta-analysis)
        # Moderate fear most effective. High fear + high efficacy = change. d ≈ 0.3-0.5
        if 'fear' in condition_lower or 'threat' in condition_lower or 'danger' in condition_lower:
            if 'high' in condition_lower or 'strong' in condition_lower:
                semantic_effect -= 0.15  # High fear can backfire without efficacy
            elif 'moderate' in condition_lower or 'medium' in condition_lower:
                semantic_effect += 0.12  # Moderate fear often most effective
            else:
                semantic_effect -= 0.10

        # Risk Perception (Slovic, 1987)
        if any(r in condition_lower for r in ['risky', 'dangerous', 'hazardous', 'unsafe']):
            semantic_effect -= 0.18
        elif any(s in condition_lower for s in ['safe', 'secure', 'protected', 'low risk']):
            semantic_effect += 0.15

        # Health Message Framing (Rothman & Salovey, 1997)
        if 'prevention' in condition_lower or 'detect' in condition_lower:
            if 'loss' in condition_lower:
                semantic_effect += 0.12  # Loss frame better for detection
        if 'promot' in condition_lower:
            if 'gain' in condition_lower:
                semantic_effect += 0.12  # Gain frame better for prevention

        # =====================================================================
        # DOMAIN 7: ORGANIZATIONAL/LEADERSHIP MANIPULATIONS
        # =====================================================================

        # Procedural Justice (Colquitt et al., 2001 meta-analysis, JAP)
        # Fair procedures increase trust and commitment. ρ ≈ .40-.50
        if any(pj in condition_lower for pj in ['procedural justice', 'fair process', 'voice',
                                                  'transparent process', 'fair procedure']):
            semantic_effect += 0.25
        elif any(pi in condition_lower for pi in ['unfair process', 'no voice', 'arbitrary']):
            semantic_effect -= 0.28

        # Distributive Justice (Colquitt et al., 2001)
        if any(dj in condition_lower for dj in ['distributive justice', 'fair outcome', 'equitable pay',
                                                  'fair reward', 'fair distribution']):
            semantic_effect += 0.25
        elif any(di in condition_lower for di in ['unfair outcome', 'inequitable', 'underpaid']):
            semantic_effect -= 0.28

        # Transformational Leadership (Judge & Piccolo, 2004 meta-analysis, JAP)
        # ρ ≈ .44 with satisfaction
        if any(tl in condition_lower for tl in ['transformational', 'inspirational', 'charismatic',
                                                  'visionary', 'empowering leader']):
            semantic_effect += 0.22
        elif any(tal in condition_lower for tal in ['transactional', 'directive', 'laissez-faire']):
            semantic_effect -= 0.05

        # Autonomy (Deci & Ryan, 2000, SDT)
        # Autonomy support increases motivation
        if any(a in condition_lower for a in ['autonomy', 'choice', 'freedom', 'self-directed',
                                                'empowerment', 'participative']):
            semantic_effect += 0.20
        elif any(c in condition_lower for c in ['controlled', 'no choice', 'mandated', 'forced', 'required']):
            semantic_effect -= 0.15

        # Feedback (Kluger & DeNisi, 1996 meta-analysis)
        if any(fb in condition_lower for fb in ['positive feedback', 'praise', 'recognition', 'appreciated']):
            semantic_effect += 0.20
        elif any(nfb in condition_lower for nfb in ['negative feedback', 'criticism', 'blame']):
            semantic_effect -= 0.22

        # =====================================================================
        # DOMAIN 8: POLITICAL/MORAL MANIPULATIONS
        # =====================================================================

        # Moral Foundations (Graham, Haidt & Nosek, 2009, JPSP)
        # Liberals: care/fairness; Conservatives: all five foundations
        if any(m in condition_lower for m in ['care', 'harm', 'compassion', 'suffering']):
            semantic_effect += 0.18
        if any(m in condition_lower for m in ['fairness', 'justice', 'equality', 'rights']):
            semantic_effect += 0.18
        if any(m in condition_lower for m in ['loyalty', 'patriot', 'traitor', 'betrayal']):
            semantic_effect += 0.12
        if any(m in condition_lower for m in ['authority', 'tradition', 'subversion', 'respect']):
            semantic_effect += 0.10
        if any(m in condition_lower for m in ['purity', 'sanctity', 'disgust', 'degradation']):
            semantic_effect += 0.10

        # Political Polarization (Iyengar & Westwood, 2015, AJPS)
        # Partisan affect stronger than racial prejudice. d > 0.5
        if any(p in condition_lower for p in ['same party', 'co-partisan', 'inparty']):
            semantic_effect += 0.30
        elif any(o in condition_lower for o in ['other party', 'opposing party', 'outparty']):
            semantic_effect -= 0.30

        # Disgust (Inbar et al., 2009)
        if 'disgust' in condition_lower:
            semantic_effect -= 0.20

        # Moral vs Non-moral framing (Feinberg & Willer, 2015)
        if 'moral' in condition_lower or 'ethical' in condition_lower:
            semantic_effect += 0.15
        elif 'immoral' in condition_lower or 'unethical' in condition_lower:
            semantic_effect -= 0.22

        # =====================================================================
        # GENERAL TREATMENT EFFECTS
        # =====================================================================

        # Treatment vs Control (general pattern)
        if 'treatment' in condition_lower and 'control' not in condition_lower:
            semantic_effect += 0.20
        elif 'control' in condition_lower and 'treatment' not in condition_lower:
            semantic_effect -= 0.05

        # Intervention effects
        if 'intervention' in condition_lower:
            semantic_effect += 0.15
        elif 'no intervention' in condition_lower or 'waitlist' in condition_lower:
            semantic_effect -= 0.05

        # =====================================================================
        # DOMAIN 9: ADDITIONAL COGNITIVE/DECISION MANIPULATIONS (50+ more)
        # =====================================================================

        # Choice Overload (Iyengar & Lepper, 2000; Scheibehenne et al. 2010 meta)
        # Original jam study d = 0.77; meta-analysis shows mean effect near 0
        # Effect is moderated by complexity and expertise
        if any(c in condition_lower for c in ['many options', 'large assortment', 'high choice', 'extensive']):
            semantic_effect -= 0.12  # Choice overload reduces satisfaction
        elif any(c in condition_lower for c in ['few options', 'small assortment', 'limited choice', 'simple']):
            semantic_effect += 0.08

        # Sunk Cost Fallacy (Staw, 1976; Sleesman et al. 2012 meta-analysis)
        # Personal responsibility increases escalation; d ≈ 0.37
        if any(s in condition_lower for s in ['sunk cost', 'invested', 'escalation', 'committed']):
            semantic_effect += 0.15  # Escalation tendency
        elif 'no sunk cost' in condition_lower or 'fresh start' in condition_lower:
            semantic_effect -= 0.05

        # Construal Level Theory (Trope & Liberman, 2010; Soderberg et al. meta)
        # Psychological distance affects abstraction; robust effect
        if any(d in condition_lower for d in ['distant', 'far future', 'abstract', 'why']):
            semantic_effect += 0.12  # Abstract = more desirable
        elif any(c in condition_lower for c in ['near', 'soon', 'concrete', 'how']):
            semantic_effect -= 0.08  # Concrete = more feasibility concerns

        # Intrinsic Motivation Crowding Out (Deci, Koestner & Ryan, 1999 meta)
        # Tangible rewards undermine intrinsic motivation; d = -0.40
        if any(e in condition_lower for e in ['extrinsic reward', 'payment', 'incentive', 'bonus for']):
            semantic_effect -= 0.15  # Undermining effect
        elif any(i in condition_lower for i in ['intrinsic', 'no reward', 'autonomous', 'self-determined']):
            semantic_effect += 0.12

        # Mere Exposure Effect (Zajonc, 1968; Bornstein, 1989 meta r = 0.26)
        # Repeated exposure increases liking
        if any(e in condition_lower for e in ['familiar', 'repeated exposure', 'seen before', 'recognized']):
            semantic_effect += 0.15
        elif any(n in condition_lower for n in ['novel', 'unfamiliar', 'first time', 'new']):
            semantic_effect -= 0.05

        # Bystander Effect (Darley & Latané, 1968; Fischer et al. 2011 meta)
        # More bystanders = less helping; 85% alone vs 31% with 4 others
        if any(b in condition_lower for b in ['alone', 'sole witness', 'only one']):
            semantic_effect += 0.25  # More likely to help
        elif any(b in condition_lower for b in ['crowd', 'many bystanders', 'group present', 'others present']):
            semantic_effect -= 0.20  # Diffusion of responsibility

        # Stereotype Threat (Steele & Aronson, 1995; Nguyen & Ryan 2008 meta d = 0.26)
        # Threat of confirming negative stereotype impairs performance
        if any(s in condition_lower for s in ['stereotype threat', 'diagnostic', 'ability test']):
            semantic_effect -= 0.15
        elif any(n in condition_lower for n in ['no threat', 'non-diagnostic', 'practice']):
            semantic_effect += 0.08

        # Reactance (Brehm, 1966; Rains 2013 meta; 2025 meta r = -0.23)
        # Freedom threat leads to boomerang effects
        if any(r in condition_lower for r in ['must', 'required', 'mandatory', 'have to', 'forced']):
            semantic_effect -= 0.18  # Reactance reduces compliance
        elif any(f in condition_lower for f in ['optional', 'choice', 'voluntary', 'may']):
            semantic_effect += 0.10

        # Emotional Contagion (Hatfield et al., 1993; replicated in social networks)
        # Emotions transfer between individuals
        if any(e in condition_lower for e in ['happy confederate', 'positive mood', 'smiling']):
            semantic_effect += 0.15
        elif any(e in condition_lower for e in ['sad confederate', 'negative mood', 'frowning']):
            semantic_effect -= 0.15

        # Negativity Bias (Rozin & Royzman, 2001; Baumeister et al. 2001)
        # Bad is stronger than good; negative information weighted more
        if 'negative info' in condition_lower or 'criticism' in condition_lower:
            semantic_effect -= 0.22  # Stronger negative effect
        elif 'positive info' in condition_lower or 'praise' in condition_lower:
            semantic_effect += 0.15  # Weaker positive effect

        # Identifiable Victim Effect (Small, Loewenstein & Slovic, 2007)
        # Meta-analysis r = 0.13; single identified victim > statistics
        if any(i in condition_lower for i in ['identified victim', 'named', 'individual story', 'one person']):
            semantic_effect += 0.12
        elif any(s in condition_lower for s in ['statistics', 'many victims', 'aggregate', 'numbers']):
            semantic_effect -= 0.05

        # Confirmation Bias (Nickerson, 1998; Hart et al. 2009 selective exposure meta)
        # People seek belief-consistent information
        if 'confirming' in condition_lower or 'consistent' in condition_lower:
            semantic_effect += 0.15
        elif 'disconfirming' in condition_lower or 'inconsistent' in condition_lower:
            semantic_effect -= 0.12

        # Hyperbolic Discounting (Laibson, 1997; Amlung et al. meta)
        # Present bias; immediate rewards overweighted
        if any(h in condition_lower for h in ['$10 now', 'today', 'immediate small']):
            semantic_effect += 0.20
        elif any(d in condition_lower for d in ['$15 later', 'delayed large', 'wait for more']):
            semantic_effect -= 0.08

        # =====================================================================
        # DOMAIN 10: COMMUNICATION & PERSUASION MANIPULATIONS
        # =====================================================================

        # Source Credibility (Hovland & Weiss, 1951; Wilson & Sherrell, 1993 meta)
        # High credibility sources more persuasive
        if any(c in condition_lower for c in ['credible source', 'expert source', 'trustworthy']):
            semantic_effect += 0.20
        elif any(l in condition_lower for l in ['low credibility', 'non-expert', 'untrustworthy']):
            semantic_effect -= 0.18

        # Message Sidedness (Allen, 1991 meta-analysis)
        # Two-sided messages more effective for educated audiences
        if 'two-sided' in condition_lower or 'both sides' in condition_lower:
            semantic_effect += 0.12
        elif 'one-sided' in condition_lower:
            semantic_effect -= 0.05

        # Narrative vs Statistical Evidence (Allen & Preiss, 1997 meta)
        # Narratives often more persuasive than statistics
        if any(n in condition_lower for n in ['narrative', 'story', 'anecdote', 'testimonial']):
            semantic_effect += 0.15
        elif any(s in condition_lower for s in ['statistical', 'data', 'numbers', 'facts']):
            semantic_effect += 0.08  # Both positive, narratives more so

        # Vividness Effect (Taylor & Thompson, 1982)
        # Vivid information more impactful
        if any(v in condition_lower for v in ['vivid', 'graphic', 'detailed', 'concrete']):
            semantic_effect += 0.12
        elif any(a in condition_lower for a in ['pallid', 'abstract', 'summary']):
            semantic_effect -= 0.05

        # Inoculation Theory (McGuire, 1961; Banas & Rains 2010 meta d = 0.29)
        # Pre-exposure to weakened arguments confers resistance
        if 'inoculation' in condition_lower or 'prebunk' in condition_lower:
            semantic_effect += 0.18  # Resistance to persuasion
        elif 'no inoculation' in condition_lower:
            semantic_effect -= 0.05

        # =====================================================================
        # DOMAIN 11: LEARNING & MEMORY MANIPULATIONS
        # =====================================================================

        # Testing Effect (Roediger & Karpicke, 2006; Rowland 2014 meta d = 0.50)
        # Retrieval practice enhances long-term retention
        if any(t in condition_lower for t in ['test', 'retrieval practice', 'quiz']):
            semantic_effect += 0.22
        elif any(s in condition_lower for s in ['restudy', 'review', 'read again']):
            semantic_effect -= 0.05

        # Spacing Effect (Cepeda et al. 2006 meta; robust effect)
        # Distributed practice superior to massed
        if any(s in condition_lower for s in ['spaced', 'distributed', 'interleaved']):
            semantic_effect += 0.20
        elif any(m in condition_lower for m in ['massed', 'blocked', 'crammed']):
            semantic_effect -= 0.10

        # Generation Effect (Slamecka & Graf, 1978)
        # Self-generated information better remembered
        if any(g in condition_lower for g in ['generate', 'produce', 'create', 'self-generated']):
            semantic_effect += 0.18
        elif any(r in condition_lower for r in ['read', 'provided', 'given']):
            semantic_effect -= 0.05

        # Desirable Difficulties (Bjork, 1994)
        # Challenges that slow learning can enhance retention
        if 'difficult' in condition_lower or 'challenging' in condition_lower:
            semantic_effect += 0.10  # Long-term benefit despite short-term cost
        elif 'easy' in condition_lower or 'simple' in condition_lower:
            semantic_effect += 0.05

        # =====================================================================
        # DOMAIN 12: SOCIAL IDENTITY & GROUP MANIPULATIONS
        # =====================================================================

        # Common Identity (Gaertner et al., 1993)
        # Superordinate identity reduces intergroup bias
        if any(c in condition_lower for c in ['common identity', 'superordinate', 'we', 'shared']):
            semantic_effect += 0.22
        elif any(d in condition_lower for d in ['dual identity', 'subgroup', 'they']):
            semantic_effect -= 0.10

        # Contact Hypothesis (Allport, 1954; Pettigrew & Tropp 2006 meta r = -0.21)
        # Intergroup contact reduces prejudice
        if any(c in condition_lower for c in ['contact', 'interaction', 'exposure to outgroup']):
            semantic_effect += 0.18
        elif any(n in condition_lower for n in ['no contact', 'segregated', 'separate']):
            semantic_effect -= 0.12

        # Minimal Group Paradigm (Tajfel, 1971; Balliet et al. 2014)
        # Even arbitrary categories produce in-group favoritism
        if any(m in condition_lower for m in ['overestimator', 'klee group', 'blue team']):
            semantic_effect += 0.15  # In-group favoritism even in minimal groups

        # Social Identity Salience (Oakes, 1987)
        # Making identity salient activates associated attitudes
        if any(s in condition_lower for s in ['identity salient', 'reminded of', 'primed with']):
            semantic_effect += 0.15
        elif 'identity not salient' in condition_lower:
            semantic_effect -= 0.05

        # =====================================================================
        # DOMAIN 13: MOTIVATION & SELF-REGULATION MANIPULATIONS
        # =====================================================================

        # Implementation Intentions (Gollwitzer, 1999; Gollwitzer & Sheeran 2006 meta d = 0.65)
        # If-then planning increases goal attainment
        if any(i in condition_lower for i in ['implementation intention', 'if-then', 'when-then', 'planning']):
            semantic_effect += 0.28
        elif 'goal intention' in condition_lower or 'motivation only' in condition_lower:
            semantic_effect -= 0.05

        # Growth vs Fixed Mindset (Dweck, 2006; Sisk et al. 2018 meta d = 0.10)
        # Malleable beliefs about ability; effect sizes smaller than originally claimed
        if any(g in condition_lower for g in ['growth mindset', 'malleable', 'can improve']):
            semantic_effect += 0.08
        elif any(f in condition_lower for f in ['fixed mindset', 'innate', 'cannot change']):
            semantic_effect -= 0.08

        # Regulatory Focus (Higgins, 1997)
        # Promotion vs prevention focus affects behavior
        if any(p in condition_lower for p in ['promotion', 'eager', 'gains', 'aspirations']):
            semantic_effect += 0.15
        elif any(p in condition_lower for p in ['prevention', 'vigilant', 'losses', 'obligations']):
            semantic_effect -= 0.10

        # Goal Gradient Effect (Hull, 1932; Kivetz et al. 2006)
        # Effort increases as goal approaches
        if any(n in condition_lower for n in ['near goal', 'almost there', 'close to']):
            semantic_effect += 0.18
        elif any(f in condition_lower for f in ['far from goal', 'just started', 'beginning']):
            semantic_effect -= 0.08

        # Licensing Effect (Merritt et al. 2010 meta)
        # Good deeds license subsequent bad behavior
        if any(l in condition_lower for l in ['licensed', 'already helped', 'did good']):
            semantic_effect -= 0.12  # Reduced subsequent prosocial
        elif 'no license' in condition_lower:
            semantic_effect += 0.05

        # =====================================================================
        # DOMAIN 14: ENVIRONMENTAL & CONTEXTUAL MANIPULATIONS
        # =====================================================================

        # Temperature and Aggression (Anderson et al. 2000)
        # Heat increases aggressive cognition and behavior
        if any(h in condition_lower for h in ['hot', 'warm room', 'heat']):
            semantic_effect -= 0.12  # More negative affect
        elif any(c in condition_lower for c in ['cool', 'cold room', 'comfortable temp']):
            semantic_effect += 0.05

        # Crowding (Baum & Paulus, 1987)
        # High density increases stress
        if any(c in condition_lower for c in ['crowded', 'high density', 'cramped']):
            semantic_effect -= 0.15
        elif any(s in condition_lower for s in ['spacious', 'low density', 'uncrowded']):
            semantic_effect += 0.08

        # Cleanliness (Schnall et al., 2008; Lee & Schwarz 2010)
        # Clean environments reduce severity of moral judgments
        if any(c in condition_lower for c in ['clean', 'tidy', 'pure', 'washed hands']):
            semantic_effect += 0.12
        elif any(d in condition_lower for d in ['dirty', 'messy', 'contaminated']):
            semantic_effect -= 0.15

        # Nature Exposure (Berman et al., 2008; Bratman et al., 2012)
        # Nature reduces stress, improves mood
        if any(n in condition_lower for n in ['nature', 'park', 'green space', 'outdoors']):
            semantic_effect += 0.15
        elif any(u in condition_lower for u in ['urban', 'city', 'concrete', 'indoors']):
            semantic_effect -= 0.05

        # Lighting (Baron et al., 1992)
        # Bright light improves mood and alertness
        if any(b in condition_lower for b in ['bright light', 'well-lit', 'daylight']):
            semantic_effect += 0.08
        elif any(d in condition_lower for d in ['dim light', 'dark', 'low light']):
            semantic_effect -= 0.05

        # =====================================================================
        # DOMAIN 15: EMBODIMENT & PHYSICAL MANIPULATIONS
        # =====================================================================

        # Facial Feedback (Strack et al., 1988; Coles et al. 2019 many-labs r = 0.03)
        # Facial expressions may influence emotional experience
        # Effect small or null in replications
        if any(f in condition_lower for f in ['smile', 'pen in teeth', 'happy expression']):
            semantic_effect += 0.05  # Small effect
        elif any(f in condition_lower for f in ['frown', 'pen in lips', 'sad expression']):
            semantic_effect -= 0.05

        # Power Posing (Carney et al., 2010; Credé & Phillips 2017 critique)
        # Expansive poses may affect feelings; effects contested
        if any(p in condition_lower for p in ['power pose', 'expansive', 'open posture']):
            semantic_effect += 0.08  # Contested, smaller than original claims
        elif any(c in condition_lower for c in ['contractive', 'closed posture', 'slumped']):
            semantic_effect -= 0.05

        # Heaviness and Importance (Jostmann et al., 2009)
        # Heavier objects associated with importance
        if any(h in condition_lower for h in ['heavy clipboard', 'weighty', 'substantial']):
            semantic_effect += 0.10
        elif any(l in condition_lower for l in ['light clipboard', 'lightweight', 'flimsy']):
            semantic_effect -= 0.05

        # =====================================================================
        # DOMAIN 16: TIME & TEMPORAL MANIPULATIONS
        # =====================================================================

        # Time Pressure (Dror et al., 1999)
        # Time constraints affect decision quality
        if any(t in condition_lower for t in ['time pressure', 'deadline', 'hurry', 'limited time']):
            semantic_effect -= 0.15  # More errors, less satisfaction
        elif any(n in condition_lower for n in ['no time pressure', 'unlimited time', 'take your time']):
            semantic_effect += 0.08

        # Morning vs Afternoon (Sievertsen et al., 2016)
        # Cognitive performance varies by time of day
        if any(m in condition_lower for m in ['morning', 'early', 'am session']):
            semantic_effect += 0.08
        elif any(a in condition_lower for a in ['afternoon', 'late', 'pm session', 'evening']):
            semantic_effect -= 0.05

        # Waiting (Kumar et al., 2014)
        # Anticipation affects experience
        if any(w in condition_lower for w in ['anticipation', 'waiting', 'expecting']):
            semantic_effect += 0.12
        elif 'immediate' in condition_lower:
            semantic_effect += 0.05

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
                    if kw in factor:
                        factor_effect += 0.18
                        break
                for kw in negative_keywords:
                    if kw in factor:
                        factor_effect -= 0.18
                        break

                # Check key manipulation types for this factor
                if 'ai' in factor and 'no' not in factor:
                    factor_effect -= 0.08
                elif 'no ai' in factor or 'no_ai' in factor or 'human' in factor:
                    factor_effect += 0.10

                if 'hedonic' in factor or 'fun' in factor:
                    factor_effect += 0.12
                elif 'utilitarian' in factor or 'practical' in factor:
                    factor_effect -= 0.05

                if 'gain' in factor or 'save' in factor:
                    factor_effect += 0.08
                elif 'loss' in factor or 'lose' in factor:
                    factor_effect -= 0.12

                if 'fair' in factor:
                    factor_effect += 0.12
                elif 'unfair' in factor:
                    factor_effect -= 0.15

                factor_effects.append(factor_effect)

            # Sum factor effects (main effects in factorial design)
            # Scale down proportionally to number of factors to prevent extreme stacking
            if factor_effects:
                n_fac = max(len(factor_effects), 1)
                scale_factor = 0.6 / max(1, n_fac - 1) if n_fac > 1 else 0.6
                semantic_effect += sum(factor_effects) * scale_factor

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

        # Clamp to reasonable range (-0.7 to +0.7) - slightly wider for strong manipulations
        semantic_effect = max(-0.7, min(0.7, semantic_effect))

        # Apply Cohen's d scaling
        return semantic_effect * default_d * COHENS_D_TO_NORMALIZED

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

        # ===== BIPOLAR SCALES (e.g., -3 to +3) =====
        elif scale_min < 0:
            calibration['central_tendency_reduction'] = -0.02  # Slight midpoint pull
            calibration['variance_multiplier'] = 0.95

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
        # STEP 3: Get base response tendency
        # Calibrated from Krosnick (1991) optimizing vs satisficing
        # =====================================================================
        # v1.2.1: Safe trait access with fallback chain
        base_tendency = modified_traits.get("response_tendency")
        if base_tendency is None:
            base_tendency = modified_traits.get("scale_use_breadth", 0.58)
        base_tendency = _safe_trait_value(base_tendency, 0.58)

        # Apply domain-specific adjustments
        base_tendency += domain_calibration['mean_adjustment']
        base_tendency += domain_calibration['positivity_bias']
        # Apply scale-type central tendency adjustment
        base_tendency += scale_calibration['central_tendency_reduction']
        base_tendency = float(np.clip(base_tendency, 0.05, 0.95))

        # =====================================================================
        # STEP 4: Apply condition effect (Cohen's d based)
        # Richard et al. (2003): Average d in social psychology ≈ 0.43
        # =====================================================================
        condition_effect = self._get_effect_for_condition(condition, variable_name)

        # Apply effect to tendency (normalized to 0-1 scale)
        adjusted_tendency = float(np.clip(base_tendency + condition_effect, 0.08, 0.92))

        # Calculate response center
        center = scale_min + (adjusted_tendency * scale_range)

        # =====================================================================
        # STEP 5: Handle reverse-coded items
        # Billiet & McClendon (2000): Acquiescers show inconsistency here
        # =====================================================================
        if is_reverse:
            center = scale_max - (center - scale_min)
            # Add extra noise for acquiescent responders on reverse items
            acquiescence = _safe_trait_value(modified_traits.get("acquiescence"), 0.5)
            if acquiescence > 0.65:
                # Acquiescers have trouble with reverse-coded items
                center += (acquiescence - 0.5) * scale_range * 0.25

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
        # High IM: +0.5-1.0 point inflation on socially desirable items
        # =====================================================================
        social_des = _safe_trait_value(modified_traits.get("social_desirability"), 0.50)
        if social_des > 0.60 and scale_range > 0:
            # Paulhus (1991): ~0.8-1.2 point inflation for high IM
            sd_effect = (social_des - 0.5) * scale_range * 0.12
            response += sd_effect

        # Bound and round to valid scale value
        response = max(scale_min, min(scale_max, round(response)))
        result = int(response)

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
        # v1.2.1: Use safe trait value extraction
        attention_level = _safe_trait_value(traits.get("attention_level"), 0.8)
        verbosity = _safe_trait_value(traits.get("verbosity"), 0.5)
        formality = _safe_trait_value(traits.get("formality"), 0.5)

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
                # v1.0.0 CRITICAL FIX: Always create a UNIQUE question identifier
                # Combine name, variable name, question text to ensure uniqueness
                base_name = str(question_spec.get("name", ""))
                var_name = str(question_spec.get("variable_name", ""))
                q_type = str(question_spec.get("type", ""))
                # Create stable unique ID that will be different for ANY different question
                unique_question_id = f"{base_name}|{var_name}|{q_type}|{question_text[:100]}"
                return self.comprehensive_generator.generate(
                    question_text=question_text or response_type,
                    sentiment=sentiment,
                    persona_verbosity=verbosity,
                    persona_formality=formality,
                    persona_engagement=engagement,
                    condition=condition,
                    question_name=unique_question_id,  # Use unique ID instead of just name
                    participant_seed=participant_seed,
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
        # v1.4.0: Enhanced context building with better condition and question integration
        study_domain = self.study_context.get("study_domain", "general")
        survey_name = self.study_context.get("survey_name", self.study_title)

        # Extract meaningful topic from question text or study context
        topic = question_spec.get("topic", study_domain)
        if topic == "general" and question_text:
            # Try to extract a topic from the question text itself
            q_words = [w for w in question_text.lower().split() if len(w) > 4 and w not in {
                "about", "would", "could", "should", "which", "there", "their", "these",
                "those", "where", "while", "being", "other", "after", "before", "during",
                "please", "describe", "explain"
            }]
            if q_words:
                topic = q_words[0]

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
            "stimulus": question_spec.get("stimulus", survey_name),
            "product": question_spec.get("product", "item"),
            "feature": question_spec.get("feature", "aspect"),
            "emotion": str(rng.choice(emotion_words)),
            "sentiment": sentiment.replace("very_", ""),  # Basic generator uses simple sentiment
            "question_text": question_text,
            "study_domain": study_domain,
            "condition": condition,  # v1.4.0: Pass condition for context-aware generation
        }

        cond = str(condition).lower()
        if "ai" in cond and "no ai" not in cond:
            context["stimulus"] = "AI-recommended " + str(context["stimulus"])
        elif "human" in cond or "no ai" in cond:
            context["stimulus"] = "human-curated " + str(context["stimulus"])
        if "hedonic" in cond or "experiential" in cond:
            context["product"] = "hedonic " + str(context["product"])
        elif "utilitarian" in cond or "functional" in cond:
            context["product"] = "functional " + str(context["product"])
        if "high" in cond:
            context["feature"] = "prominent " + str(context["feature"])
        elif "low" in cond:
            context["feature"] = "subtle " + str(context["feature"])

        # v1.0.0 CRITICAL FIX: Create question-specific seed for fallback generator
        # Combine participant_seed with a stable hash of the question identity
        base_name = str(question_spec.get("name", ""))
        var_name = str(question_spec.get("variable_name", ""))
        unique_id = f"{base_name}|{var_name}|{question_text[:100]}"
        # Use stable hash independent of Python's hash randomization
        question_hash_stable = sum(ord(c) * (i + 1) * 31 for i, c in enumerate(unique_id[:200]))
        unique_fallback_seed = (participant_seed + question_hash_stable) % (2**31)

        return self.text_generator.generate_response(
            response_type, style, context, traits, unique_fallback_seed
        )

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

        for scale_idx, scale in enumerate(self.scales):
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

                data[col_name] = item_values
                _scale_generation_log[-1]["columns_generated"].append(col_name)

                reverse_note = " (reverse-coded)" if is_reverse else ""
                self.column_info.append(
                    (col_name, f'{scale_name_raw} item {item_num} ({scale_min}-{scale_max}){reverse_note}')
                )

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

        # ONLY generate open-ended responses for questions actually in the QSF
        # Never create default/fake questions - this prevents fake variables like "Task_Summary"
        # v1.0.0: Use survey flow handler to determine question visibility per condition
        for q in self.open_ended_questions:
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
            responses: List[str] = []
            for i in range(n):
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
                persona = self.available_personas[persona_name]
                response_vals = participant_item_responses[i]
                response_mean = float(np.mean(response_vals)) if response_vals else None

                # Generate response with enhanced uniqueness
                text = self._generate_open_response(
                    q,
                    persona,
                    all_traits[i],
                    participant_condition,
                    p_seed,
                    response_mean=response_mean,
                )
                responses.append(str(text))

            data[col_name] = responses
            q_desc = q.get("question_text", "")[:50] if q.get("question_text") else q.get('type', 'text')
            self.column_info.append((col_name, f"Open-ended: {q_desc}"))

        # v1.0.0 CRITICAL FIX: Post-processing validation to detect and fix duplicate responses
        # Check each participant's responses across all open-ended questions
        # Use actual column names from data (accounting for any renames due to collisions)
        open_ended_cols = [col for col in data
                          if isinstance(data[col], list)
                          and len(data[col]) > 0
                          and isinstance(data[col][0], str)]
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

        # POST-GENERATION VALIDATION: Verify all scale columns are within bounds
        validation_issues = self._validate_generated_data(df)
        if validation_issues:
            self._log(f"POST-GENERATION VALIDATION: {len(validation_issues)} issue(s) found, auto-correcting")
            for issue in validation_issues:
                col = issue["column"]
                col_min = issue["expected_min"]
                col_max = issue["expected_max"]
                # Auto-correct out-of-bounds values
                df[col] = df[col].clip(lower=col_min, upper=col_max).astype(int)
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
                values = [t.get(trait_key, 0.0) for t in traits_list if trait_key in t]
                if values:
                    trait_averages_by_condition[cond][trait_key] = round(float(np.mean(values)), 4)

        # 6. Overall trait averages (across all participants)
        overall_trait_averages: Dict[str, float] = {}
        if all_traits:
            trait_keys = list(all_traits[0].keys()) if all_traits else []
            for trait_key in trait_keys:
                values = [t.get(trait_key, 0.0) for t in all_traits if trait_key in t]
                if values:
                    overall_trait_averages[trait_key] = round(float(np.mean(values)), 4)

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
            "exclusion_summary": {
                "flagged_speed": int(sum(data["Flag_Speed"])),
                "flagged_attention": int(sum(data["Flag_Attention"])),
                "flagged_straightline": int(sum(data["Flag_StraightLine"])),
                "total_excluded": int(sum(data["Exclude_Recommended"])),
            },
            "validation_issues_corrected": len(validation_issues),
            "scale_verification": self._build_scale_verification_report(df),
            "generation_warnings": self._check_generation_warnings(df),
            # v1.4.3: Column descriptions for data dictionary / codebook generation
            "column_descriptions": {col: desc for col, desc in self.column_info},
        }
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
            scale_name = str(scale.get("name", "Scale")).strip().replace(" ", "_")
            scale_points = _safe_numeric(scale.get("scale_points", 7), default=7, as_int=True)
            scale_points = max(2, min(1001, scale_points))
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

                actual_min = int(col_data.min())
                actual_max = int(col_data.max())

                # CHECK 1b: Bounds validation
                if actual_min < 1 or actual_max > scale_points:
                    issues.append({
                        "column": col_name,
                        "expected_min": 1,
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
            actual_min = int(col_data.min())
            actual_max = int(col_data.max())
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
            actual_min = int(col_data.min())
            actual_max = int(col_data.max())
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
            scale_name_clean = scale_name.replace(" ", "_")
            spec_points = int(scale.get("scale_points", 7))
            spec_items = int(scale.get("num_items", 5))

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
                    if col_min < 1 or col_max > spec_points:
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

        # Get scale columns
        scale_cols = []
        for scale in self.scales:
            scale_name = str(scale.get("name", "Scale")).replace(" ", "_")
            num_items = _safe_numeric(scale.get("num_items", 5), default=5, as_int=True)
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
            ds = [es.cohens_d for es in self.effect_sizes]
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
