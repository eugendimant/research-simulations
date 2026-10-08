# simulation_app/utils/enhanced_simulation_engine.py
from __future__ import annotations
"""
Enhanced Simulation Engine for Behavioral Experiment Simulation Tool
=============================================================================

Module version: see __version__ below (the app-wide version is tracked in
utils/__init__.py and app.py; this engine carries its own internal __version__).

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
__version__ = "1.0.9.5"  # v1.0.9.5: ABE 3.0 — 5 consistency improvements + HBS merge

# v1.2.7.0: DV types whose item columns are JOINTLY constrained (a permutation;
# or an allocation summing to a fixed total). Their columns must be exempted from
# any downstream per-item mutation (alpha re-correlation, anti-straight-line
# jitter, bounds-clipping) that would silently break the joint constraint.
_JOINT_DV_TYPES = frozenset({
    "rank_order", "ranking", "rank order",
    "constant_sum", "constant sum", "constantsum",
})

# v1.3.0.4: fewest response options at which a run of identical answers says anything
# about the respondent. Below it, chance agreement is of a different order: about 77% of
# the rows of a 3-item binary block are identical before anything touches them, against
# the 19% that the registry measured on 5- to 9-point instruments. Both straight-line
# passes (the consistency audit and the registry-calibrated identical-answer pass) are
# gated on it, because pulling such a block toward a Likert-measured share rewrites
# honest answers and takes the requested effect with it.
_MIN_OPTIONS_FOR_STRAIGHTLINE_LOGIC = 5
# Numeric-DV money/count classification cues are compiled once, just after the
# `import re` below (see _MONEY_CUE_RE / _COUNT_CUE_RE / _RATING_CTX_RE).

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
Participants are assigned to behavioral personas. Base weights are listed below;
study-specific reweighting then spreads them across domain personas, so the
realised mix differs per study:

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

4. VALIDATION STATUS
--------------------
The generator has NOT been validated against real participant data. What the test
suite checks today:
- Responses stay inside each scale's bounds; the same seed reproduces the same data.
- A configured Cohen's d is recovered on the scale mean within roughly +/-12%
  (tests/test_effect_size_recovery.py), including 0/null effects.
- Reverse-keyed items correlate negatively with their scale before recoding.
Distributional realism (marginals, scale reliability, cross-scale correlations,
timing) is a design goal that is still being measured, not an established property.
Effects that are not configured explicitly are inferred from condition labels and
are exploratory only.

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
import html
import os
import random
import re
import time

import numpy as np
import pandas as pd

# v1.2.8.4 (Codex P2): the process-wide _GLOBAL_RNG_LOCK was REMOVED. The engine no
# longer seeds the global np.random/random — every draw uses a per-instance RNG
# (local np.random.RandomState seeded from self.seed/participant_seed for numerics;
# HBSValidator(seed=...) and LLMResponseGenerator._rng for the former global
# consumers). With no shared global RNG state there is nothing to serialize, so
# concurrent Streamlit sessions now run fully in parallel (no run-wide lock held
# across LLM network I/O) while staying byte-identical for the same seed.

# v1.2.7.4: numeric-DV money/count classification cues, compiled ONCE. Letter-
# boundary matching ((?<![a-z])...(?![a-z])) so cues fire on underscore-normalized
# names ("wtp dollars") but never inside other words ('times'∉"sometimes",
# 'cost'∉"costly", 'bid'∉"bidding", 'invest'∉"investment attitude" — the last is
# also caught by the rating-context guard).
_LB = lambda body: r'(?<![a-z])(?:' + body + r')(?![a-z])'
_MONEY_CUE_RE = re.compile(
    _LB(r'wtp|willing to pay|willingness|prices?|priced|pay|paid|paying|'
        r'spend|spent|spending|donat[a-z]*|bids?|costs?|amounts?|dollars?|'
        r'contribut[a-z]*|invest|invests|invested|investment|budget|'
        r'wages?|salary|salaries') + r'|\$', re.IGNORECASE)
_COUNT_CUE_RE = re.compile(
    _LB(r'number of|how many|count|times|frequency|'
        r'per (?:week|day|month|year)|visits?|purchases?|how often'), re.IGNORECASE)
# Rating/attitude contexts are NOT money/count even if a cue word appears.
_RATING_CTX_RE = re.compile(
    _LB(r'agree|disagree|satisf[a-z]*|attitude|opinion|rate|rating|extent|'
        r'how much do you|likelihood|likely|confiden[a-z]*|important|'
        r'favorab[a-z]*|to what extent'), re.IGNORECASE)

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
        resolve_game_calibration_key,
        detect_game_type,
        looks_like_game_decision,
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

# v1.2.9.1: Empirical realism layer — provenance for the knowledge base, a
# content-based construct matcher, verified distribution shapes, and the
# real-data benchmark. Each import is guarded: these are additive, and the app
# must still load if any one of them is unavailable.
try:
    from .item_realism import (
        decouple_block,
        match_straightlining,
        target_r_from_alpha,
        match_item_dispersion,
        DEFAULT_STRAIGHTLINE_SHARE,
    )
    HAS_ITEM_REALISM = True
except Exception:
    HAS_ITEM_REALISM = False

try:
    from . import empirical_registry as _empirical_registry
    from . import design_signature as _design_signature
    HAS_EMPIRICAL_REGISTRY = True
except Exception:
    HAS_EMPIRICAL_REGISTRY = False

try:
    from . import construct_matcher as _construct_matcher
    HAS_CONSTRUCT_MATCHER = True
except Exception:
    HAS_CONSTRUCT_MATCHER = False

try:
    from . import literature_effects as _literature_effects
    HAS_LITERATURE_EFFECTS = True
except Exception:
    HAS_LITERATURE_EFFECTS = False

try:
    from . import paradigm_coverage as _paradigm_coverage
except Exception:  # the extra paradigms are optional: without them the older entries work as before
    _paradigm_coverage = None

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

# Import Adaptive Behavioral Engine (v2 narrative wrapper) — narrative-enhanced wrapper
try:
    from .adaptive_behavioral_engine_v2 import AdaptiveBehavioralEngineV2
    HAS_ABE_V2 = True
except ImportError:
    HAS_ABE_V2 = False

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

# ABE 3.0: Import HBS sub-modules for integrated post-processing
# These are optional — ABE 3.0 works without them but gains census-weighted
# demographics, stylometric fingerprinting, validation, and error calibration.
try:
    from .hbs_participant_state import HBSParticipantFactory
    HAS_HBS_DEMOGRAPHICS = True
except ImportError:
    HAS_HBS_DEMOGRAPHICS = False

try:
    from .hbs_stylometric_engine import HBSStylometricEngine
    HAS_HBS_STYLOMETRIC = True
except ImportError:
    HAS_HBS_STYLOMETRIC = False

try:
    from .hbs_validator import HBSValidator
    HAS_HBS_VALIDATOR = True
except ImportError:
    HAS_HBS_VALIDATOR = False

try:
    from .hbs_question_classifier import HBSQuestionClassifier
    HAS_HBS_CLASSIFIER = True
except ImportError:
    HAS_HBS_CLASSIFIER = False

try:
    from .hbs_error_calibrator import HBSErrorCalibrator
    HAS_HBS_ERROR_CAL = True
except ImportError:
    HAS_HBS_ERROR_CAL = False

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


# v1.2.6.4 PERFORMANCE: Cache compiled regex patterns. _word_in/_stem_in are
# called millions of times during large-N generation; re.search recompiles the
# pattern on every call (Python's internal cache is small and thrashes when many
# distinct keywords are used). An explicit module-level cache keyed on the raw
# keyword eliminates repeated compilation — the dominant cost in profiling.
_WORD_PATTERN_CACHE: Dict[str, "re.Pattern"] = {}
_STEM_PATTERN_CACHE: Dict[str, "re.Pattern"] = {}


def _kw_hit(keyword: str, text: str) -> bool:
    """Keyword match for condition labels that respects word boundaries: short keywords
    ("ai", "low") must be whole words; longer ones may be word prefixes ("sustainab" ->
    "sustainable"). Plain substring tests matched "ai" in "wait" and "low" in "follow-up",
    which silently changed response styles for unrelated conditions.

    v1.2.9.1: an underscore separates words here ("High_Threat" == "High Threat"). The regex word
    character class includes "_", so snake_case labels ("No_AI", "Loss_Frame") matched nothing and
    lost every name-based modifier."""
    keyword = keyword.replace("_", " ")
    text = text.replace("_", " ")
    return _word_in(keyword, text) if len(keyword) <= 4 else _stem_in(keyword, text)


def _condition_label(c: Any) -> str:
    """Display name of one condition, whether it arrives as a string or as a dict such as
    {"name": "A"} (which used to be stringified into the CONDITION column)."""
    if isinstance(c, dict):
        for key in ("name", "label", "condition", "title", "value"):
            v = c.get(key)
            if v not in (None, ""):
                return str(v)
        return ""
    return str(c)


def _word_in(keyword: str, text: str) -> bool:
    """Check if keyword appears in text as a whole word (word-boundary matching).

    Prevents false positives like 'ai' matching 'wait', 'gain' matching 'bargain',
    'own' matching 'brown', 'get' matching 'budget', etc.

    The boundary is "no word character directly before or after the match" rather
    than the regex ``\\b``. For a keyword that starts and ends with a word character
    the two are identical; they differ for keywords that begin or end with
    punctuation, where ``\\b`` demands a word character on the OUTER side and so never
    matched a label such as "Norm message (Control)", "80%" or "$10".

    v1.0.1.3: Added to fix substring false-positive keyword matching throughout
    the semantic effect engine.
    v1.2.6.4: Compiled-pattern cache for performance.
    v1.2.9.1: Lookaround boundaries, so labels that start or end with punctuation match.
    """
    _pat = _WORD_PATTERN_CACHE.get(keyword)
    if _pat is None:
        _pat = re.compile(r'(?<!\w)' + re.escape(keyword) + r'(?!\w)')
        _WORD_PATTERN_CACHE[keyword] = _pat
    return bool(_pat.search(text))


def _stem_in(stem: str, text: str) -> bool:
    """Check if stem appears at a word boundary start in text.

    For intentional prefix matches like 'automat' -> 'automated'/'automation',
    'reciproc' -> 'reciprocity'/'reciprocal', 'promot' -> 'promote'/'promotion'.
    v1.2.6.4: Compiled-pattern cache for performance.
    v1.2.9.1: "No word character before" instead of ``\\b``, so a stem that starts with
    punctuation ("$10", "(high") can match (identical for stems that start with a letter).
    """
    _pat = _STEM_PATTERN_CACHE.get(stem)
    if _pat is None:
        _pat = re.compile(r'(?<!\w)' + re.escape(stem))
        _STEM_PATTERN_CACHE[stem] = _pat
    return bool(_pat.search(text))


def _any_word_in(keywords: list, text: str) -> bool:
    """Check if any keyword from the list appears as a whole word in text."""
    return any(_word_in(kw, text) for kw in keywords)


# v1.2.9.1: matching of user-specified effects (EffectSizeSpec) against the generated variables
# and conditions. Compiled once, like the word/stem pattern caches above.
_VAR_TOKEN_SPLIT = re.compile(r"[\W_]+")
_WHITESPACE_RUN = re.compile(r"\s+")


def _var_tokens(name: Any) -> Tuple[str, ...]:
    """Lower-cased alphanumeric tokens of a variable name, so "Perceived_Quality",
    "Perceived Quality" and "perceived-quality (scale)" compare on the same words."""
    return tuple(t for t in _VAR_TOKEN_SPLIT.split(str(name if name is not None else "").lower()) if t)


def _tokens_contain(tokens: Tuple[str, ...], run: Tuple[str, ...]) -> bool:
    """True when ``run`` occurs as a contiguous run of WHOLE tokens inside ``tokens``
    ("trust" in "trust score", but not in "distrust")."""
    n = len(run)
    return n > 0 and any(tokens[i:i + n] == run for i in range(len(tokens) - n + 1))


def _label_norm(label: Any) -> str:
    """A condition or level label in comparison form: non-breaking spaces and runs of
    whitespace become one space, outer whitespace is dropped, lower case."""
    return _WHITESPACE_RUN.sub(" ", str(label if label is not None else "").replace("\xa0", " ")).strip().lower()


def _spec_get(effect: Any, key: str, default: Any = "") -> Any:
    """Read a field of an EffectSizeSpec object or of a plain dict (API callers, tests)."""
    if isinstance(effect, dict):
        return effect.get(key, default)
    return getattr(effect, key, default)


def _spec_side(effect: Any, condition_norm: str) -> int:
    """Which side of an effect spec a condition is on: +1 (the higher-scoring level), -1 (the
    lower-scoring level) or 0 (neither). ``condition_norm`` is a `_label_norm` string. When both
    levels occur in the label (a "no AI" condition contains "ai"), the longer, more specific level wins."""
    level_high = _label_norm(_spec_get(effect, "level_high", ""))
    level_low = _label_norm(_spec_get(effect, "level_low", ""))
    is_high = bool(level_high and _word_in(level_high, condition_norm))
    is_low = bool(level_low and _word_in(level_low, condition_norm))
    if is_high and is_low:
        if len(level_high) >= len(level_low):
            is_low = False
        else:
            is_high = False
    return 1 if is_high else (-1 if is_low else 0)


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


_SCRIPT_CTRL_RE = re.compile(r"[\x00-\x1f\x7f\x85\u2028\u2029]+")
_SCRIPT_NAME_RE = re.compile(r"[A-Za-z0-9_.]+")


def _script_text(value: Any) -> str:
    """Text for a comment or a quoted literal in an exported analysis script, as one physical line.

    Study titles, scale names and condition labels are typed by users, and the scripts are run by the
    people who receive the ZIP: a line break in such text would start a new statement."""
    return _SCRIPT_CTRL_RE.sub(" ", str(value if value is not None else "")).strip()


def _script_name_ok(name: Any) -> bool:
    """True for names that are safe to write unquoted or inside quotes in any exported script language."""
    return bool(_SCRIPT_NAME_RE.fullmatch(str(name)))


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
                    # v1.2.5.3: Preserve DV description and scale anchors for simulation context
                    "dv_description": str(scale.get("dv_description", "")),
                    "scale_anchors": scale.get("scale_anchors", {}),
                    # v1.2.7.3: preserve this DV's own question text so numeric-realism
                    # classification uses DV-specific cues, not study-level text.
                    "question_text": str(scale.get("question_text", "")),
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

            # v1.2.7.5: a LIST-valued items/num_items defines the item NAMES and
            # count. Previously a list fell through int() and defaulted to 5 with
            # empty item_names — e.g. items=["A","B","C"] wrongly became 5 items.
            item_names = list(scale.get("item_names") or [])
            raw_items = scale.get("num_items")
            if raw_items is None:
                raw_items = scale.get("items")  # QSF detection compatibility
            if isinstance(raw_items, (list, tuple)):
                item_names = [str(x) for x in raw_items if str(x).strip()]
                n_items = len(item_names)
            elif raw_items is None:
                n_items = len(item_names) if item_names else 5
            else:
                try:
                    n_items = int(raw_items)
                except (ValueError, TypeError):
                    n_items = len(item_names) if item_names else 5
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
                    "item_names": item_names,
                    # v1.4.0: Preserve scale type for downstream use
                    "type": str(scale.get("type", "matrix")),
                    # v1.2.5.3: Preserve DV description and scale anchors
                    "dv_description": str(scale.get("dv_description", "")),
                    "scale_anchors": scale.get("scale_anchors", {}),
                    # v1.2.7.3: preserve this DV's own question text (see above).
                    "question_text": str(scale.get("question_text", "")),
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


_NUMERIC_DV_TYPES = frozenset({"numeric", "numeric_input", "slider", "single_item"})


def _drop_oe_duplicating_dvs(
    open_ended: List[Dict[str, Any]], scales: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Remove text boxes that are also detected numeric DVs.

    A text-entry question that is a numeric outcome (a score, an amount) is parsed twice: as a
    scale and as an open-ended question with the same name. The DV columns already hold its
    numbers, so the second "open-ended" column only duplicated it, often with prose such as
    "i really loved how the survey was designed" in a column called Pre-return total score.
    Returns the kept questions and the names that were dropped.
    """
    def _key(x: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(x or "").lower()).strip("_")

    dv_keys = set()
    for sc in scales or []:
        if str(sc.get("type", "")).lower() in _NUMERIC_DV_TYPES:
            dv_keys.update(k for k in (_key(sc.get("variable_name")), _key(sc.get("name"))) if k)
    kept: List[Dict[str, Any]] = []
    dropped: List[str] = []
    for q in open_ended:
        if _key(q.get("variable_name") or q.get("name")) in dv_keys:
            dropped.append(str(q.get("variable_name") or q.get("name")))
        else:
            kept.append(q)
    return kept, dropped


def _normalize_open_ended(open_ended: Optional[List[Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in open_ended or []:
        if isinstance(item, str):
            name = item.strip()
            if name:
                normalized.append({"name": name, "type": "text", "question_text": name})
            continue
        if isinstance(item, dict):
            # v1.2.7.5: accept name from any of these keys (QSF/builder paths use
            # different ones). Previously only name/question_id were honored, so a
            # spec like {"variable_name": "explain", ...} was silently DROPPED.
            name = str(
                item.get("name")
                or item.get("variable_name")
                or item.get("export_tag")
                or item.get("question_id")
                or ""
            ).strip()
            if name:
                # Preserve question_text, display_logic, and condition info for survey flow
                normalized_item = dict(item)
                # Ensure name is set (for column naming)
                normalized_item["name"] = name
                # Preserve the chosen variable name for export/column naming
                normalized_item.setdefault("variable_name", name)
                # Ensure question_text is set for unique response generation
                if not normalized_item.get("question_text"):
                    normalized_item["question_text"] = name
                normalized.append(normalized_item)
    return normalized


# =============================================================================
# v1.2.9.1: Question-text cleaning and numeric text-box answers
# =============================================================================
# QSF question text arrives with HTML ("&nbsp;", "&quot;", "<br>"), Qualtrics
# placeholder text ("Click to write the question text") and bare variable ids
# ("Q104"). Left alone these leaked into generated answers ("...another study&#39;s
# participant...", "my take on Click to write the question text is..."). Numeric text
# boxes ("How many tickets...? (enter a number)", "What is your year of birth?") were
# answered with essays. These helpers are pure functions so they can be unit-tested.

_PLACEHOLDER_QUESTION_RE = re.compile(
    r"click to write (?:the )?question text|^\s*(?:enter|type) (?:your )?question (?:text )?here\s*\.?\s*$",
    re.IGNORECASE,
)
_BARE_VARIABLE_RE = re.compile(r"^(?:q|qid)?\s*\d+(?:[._]\d+)*(?:_text)?$", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^<>]*>")
# CSS pasted into a question's text ("#QID154-7-label {display: inline-block; width: 5%;}"). The
# selector must start a token (so the "." of "U.S." does not) and the block must hold a declaration
# ("prop: value") and no piped-text "$": the old pattern swallowed everything from the first "." or "#"
# to the next "{...}", so "The U.S. government gave ${e://Field/amount} to you" became "The U to you".
_CSS_BLOCK_RE = re.compile(
    r"(?<![\w$])[#.][\w\-]+(?:[ \t]*[,>+~][ \t]*[#.]?[\w\-]+)*[ \t]*\{[^{}$]*:[^{}$]*\}")
# <script>/<style> elements: their content is code, not question wording (dropped before tag stripping)
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
# Qualtrics piped text: ${e://Field/Name} (the "$" is sometimes already stripped)
_PIPED_TEXT_RE = re.compile(r"\$?\{[^{}]*\}")
_HTML_BREAK_TAG_RE = re.compile(r"</?(?:br|p|div|li|ul|ol|tr|td|th|table|h[1-6])\b[^<>]*>", re.IGNORECASE)
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:!?])")

# Newest survey year used when a question asks for a birth year. Fixed (not "today") so the
# same seed gives the same dataset no matter when it is run.
_REFERENCE_SURVEY_YEAR = 2025


def _clean_question_text(text: Any) -> str:
    """Return question text that is safe to use in prompts and templates.

    Decodes HTML entities, removes HTML tags, collapses whitespace, and returns an empty
    string for Qualtrics placeholder text or a bare variable id such as "Q104". Qualtrics
    piped-text markers (``${e://Field/X}``) are left untouched; they are substituted later.
    """
    if text is None:
        return ""
    t = html.unescape(html.unescape(str(text)))
    t = _SCRIPT_STYLE_RE.sub(" ", t)    # code inside <script>/<style> is not wording
    t = _HTML_BREAK_TAG_RE.sub(" ", t)  # line/paragraph breaks separate words
    t = _HTML_TAG_RE.sub("", t)         # inline tags (<b>, <span>) vanish without adding a gap
    t = _CSS_BLOCK_RE.sub(" ", t)       # stylesheet rules pasted into the question text
    t = t.replace("\xa0", " ").replace("\u200b", "")
    t = re.sub(r"\s+", " ", t).strip()
    t = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", t)
    if not t or _PLACEHOLDER_QUESTION_RE.search(t) or _BARE_VARIABLE_RE.match(t):
        return ""
    return t


_NUMERIC_EXCLUDE_RE = re.compile(
    r"\b(why|explain|describe|reasons?|in your own words|comments?|feedback|opinions?|thoughts?|"
    r"elaborate|tell us|suggestions?|what do you think|how do you feel|justify|briefly)\b",
    re.IGNORECASE,
)
_YEAR_OF_BIRTH_RE = re.compile(
    r"year of birth|birth ?year|year (?:were|are) you born|born in what year|what year .{0,20}born",
    re.IGNORECASE,
)
_AGE_RE = re.compile(r"\bhow old\b|\byour age\b|\bage in years\b|^\s*age\s*[:?]?\s*$", re.IGNORECASE)
_PERCENT_RE = re.compile(r"percent|percentage|%", re.IGNORECASE)
_MONEY_RE = re.compile(
    r"\$|€|£|\bdollars?\b|\busd\b|\beuros?\b|\bcents?\b|\bsalary\b|\bincome\b|\bwage\b|\bwtp\b|"
    r"willing(?:ness)? to pay|\bprice\b|\bdonat\w*\b|\bearn\w*\b",
    re.IGNORECASE,
)
_COUNT_ASK_RE = re.compile(
    r"how many|how much|number of|enter a number|enter number|enter a numeric|numerical|numeric answer|"
    r"in numbers|\(number\)|\(numeric\)|enter an? (?:integer|amount|value)",
    re.IGNORECASE,
)
_NUM = r"(-?\d+(?:\.\d+)?)"
_RANGE_PATTERNS = [
    re.compile(r"between\s+[$€£]?" + _NUM + r"\s*(?:and|to|-|–)\s*[$€£]?" + _NUM, re.IGNORECASE),
    re.compile(r"from\s+[$€£]?" + _NUM + r"\s+to\s+[$€£]?" + _NUM, re.IGNORECASE),
    re.compile(r"\(\s*[$€£]?" + _NUM + r"\s*(?:-|–|to)\s*[$€£]?" + _NUM + r"\s*\)", re.IGNORECASE),
    re.compile(r"\b" + _NUM + r"\s*(?:-|–|to)\s*" + _NUM + r"\b", re.IGNORECASE),
]
_UPPER_ONLY_RE = re.compile(r"(?:up to|at most|maximum of|max of|out of)\s+[$€£]?(\d+(?:\.\d+)?)", re.IGNORECASE)


_MTURK_ID_RE = re.compile(
    r"\bworker\s*id\b|\bworkerid\b|\bmturk\b[^.?!]{0,40}\b(?:id|code)\b|\b(?:id|code)\b[^.?!]{0,30}\bmturk\b|"
    r"mechanical turk[^.?!]{0,40}\bid\b",
    re.IGNORECASE,
)
_PROLIFIC_ID_RE = re.compile(r"\bprolific\b", re.IGNORECASE)
_PARTICIPANT_ID_RE = re.compile(
    r"\b(?:participant|respondent|subject|student|sona|survey)\s*(?:id|number|code)\b|\byour id\b|"
    r"\b(?:code|id number)\b[^.?!]{0,50}\b(?:given|provided|received|assigned|sent)\b",
    re.IGNORECASE,
)


def _parse_numeric_range(text: str) -> Optional[Tuple[float, float]]:
    """Find an explicit numeric range such as "(0-10)", "between 0 and 100" or "out of 20"."""
    for pat in _RANGE_PATTERNS:
        m = pat.search(text)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            if lo < hi <= 1e7:
                return lo, hi
    m = _UPPER_ONLY_RE.search(text)
    if m:
        hi = float(m.group(1))
        if 0 < hi <= 1e7:
            return 0.0, hi
    return None


def _infer_numeric_answer_spec(
    question_text: Any, variable_name: Any = "", question: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Decide whether a text box expects a number, and describe it. ``None`` means "free text".

    Uses Qualtrics' own validation (ContentType ValidNumber/ValidZip and its Min/Max) when
    the parser supplied it, otherwise conservative wording cues. Questions that also ask the
    participant to explain ("...and why?") stay free text.
    """
    q = question if isinstance(question, dict) else {}
    text = _clean_question_text(question_text)
    ctype = str(q.get("content_type") or "").lower()
    lo_hi: Optional[Tuple[float, float]] = None
    hard_lo = hard_hi = None   # declared bounds, enforced on every draw (also when only one side is declared)
    try:
        nmin, nmax = q.get("number_min"), q.get("number_max")
        hard_lo = float(nmin) if nmin not in (None, "") else None
        hard_hi = float(nmax) if nmax not in (None, "") else None
        if hard_lo is not None and hard_hi is not None and hard_lo < hard_hi:
            lo_hi = (hard_lo, hard_hi)
        elif hard_lo is None and hard_hi is not None and 0 < hard_hi < float("inf"):
            # Max only (Min left blank): what these boxes ask for (counts, amounts, percentages, ages, years)
            # is never negative, so the window is 0..Max. Without this a declared "at most 2" was ignored
            # and the draw came from the generic count distribution (0-13), or from a "(0-100)" in the text.
            lo_hi = (0.0, hard_hi)
    except (TypeError, ValueError):
        lo_hi = None
        hard_lo = hard_hi = None
    num_decimals = q.get("number_decimals")

    if ctype == "validzip":
        return {"kind": "zip"}
    declared_number = ctype in ("validnumber", "validdecimal", "validinteger")
    t = _PIPED_TEXT_RE.sub(" ", text).lower()  # "${e://Field/Random%20ID}" must not read as "%"
    # Crowd-worker / participant ID boxes hold an ID, not prose. Only short prompts that do not
    # ask for an explanation count ("Did you do this on MTurk? Please explain" stays free text).
    if t and len(t) <= 240 and not _NUMERIC_EXCLUDE_RE.search(t):
        if _MTURK_ID_RE.search(t):
            return {"kind": "mturk_id"}
        if _PROLIFIC_ID_RE.search(t) and re.search(r"\bid\b", t):
            return {"kind": "prolific_id"}
        if _PARTICIPANT_ID_RE.search(t):
            return {"kind": "participant_id"}
    if not declared_number:
        if not t or _NUMERIC_EXCLUDE_RE.search(t):
            return None
    if _YEAR_OF_BIRTH_RE.search(t):
        return {"kind": "year_of_birth", "lo": lo_hi[0] if lo_hi else None, "hi": lo_hi[1] if lo_hi else None}
    if _AGE_RE.search(t) or str(variable_name or "").strip().lower() == "age":
        return {"kind": "age", "lo": lo_hi[0] if lo_hi else None, "hi": lo_hi[1] if lo_hi else None}
    if not declared_number and not (_COUNT_ASK_RE.search(t) or _PERCENT_RE.search(t)
                                    or (_MONEY_RE.search(t) and re.search(r"how much|amount|enter|offer|bid|pay|give|spend", t))):
        return None
    if lo_hi is None:
        lo_hi = _parse_numeric_range(text)
    kind = "percent" if _PERCENT_RE.search(t) else ("money" if _MONEY_RE.search(t) else "count")
    if num_decimals not in (None, ""):
        decimals = int(num_decimals) > 0           # the survey's own NumDecimals setting wins over wording
    else:
        decimals = bool(re.search(r"decimal|cents|\d\.\d", t))
    return {"kind": kind, "lo": lo_hi[0] if lo_hi else None, "hi": lo_hi[1] if lo_hi else None,
            "decimals": decimals, "hard_lo": hard_lo, "hard_hi": hard_hi,
            "ndec": int(num_decimals) if num_decimals not in (None, "") and int(num_decimals) > 0 else 2}


def _format_numeric(value: float, spec: Dict[str, Any], lo: Optional[float] = None, hi: Optional[float] = None) -> str:
    """Format a drawn number so it stays inside every declared bound (integers round INTO the range)."""
    import math
    lo = spec.get("hard_lo") if spec.get("hard_lo") is not None else lo
    hi = spec.get("hard_hi") if spec.get("hard_hi") is not None else hi
    decimals = bool(spec.get("decimals"))
    if not decimals and lo is not None and hi is not None and math.ceil(lo) > math.floor(hi):
        decimals = True                                  # no integer fits (e.g. 0.2-0.8): use decimals
    if lo is not None:
        value = max(value, lo)
    if hi is not None:
        value = min(value, hi)
    if decimals:
        nd = int(spec.get("ndec", 2))
        return f"{value:.{nd}f}"
    iv = int(round(value))
    if lo is not None and iv < lo:
        iv = int(math.ceil(lo))
    if hi is not None and iv > hi:
        iv = int(math.floor(hi))
    return str(iv)


def _draw_numeric_answer(spec: Dict[str, Any], rng: "np.random.RandomState") -> str:
    """Draw one plausible numeric answer. People favour focal values (endpoints, the midpoint,
    round numbers), so a share of answers snap to those instead of being uniform."""
    kind = spec.get("kind")
    if kind == "zip":
        return f"{int(rng.randint(10000, 99999)):05d}"
    if kind == "mturk_id":  # Amazon worker IDs: "A" plus 12-13 upper-case letters and digits
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return "A" + "".join(alphabet[int(i)] for i in rng.randint(0, len(alphabet), size=int(rng.choice([12, 13]))))
    if kind == "prolific_id":  # Prolific IDs: 24 hexadecimal characters
        return "".join("0123456789abcdef"[int(i)] for i in rng.randint(0, 16, size=24))
    if kind == "participant_id":
        return str(int(rng.randint(100000, 999999)))
    if kind in ("age", "year_of_birth"):
        lo, hi = spec.get("lo"), spec.get("hi")

        def _one() -> int:
            for _try in range(20):                      # truncated (re-drawn) normal: no spike at the floor
                _a = int(round(rng.normal(37, 13)))
                if 18 <= _a <= 80:
                    break
            else:
                _a = 35
            return _a if kind == "age" else _REFERENCE_SURVEY_YEAR - _a

        if lo is None or hi is None:
            return str(_one())
        # The survey declares a validation range (e.g. ages 65-90): stay inside it. Draw from the
        # usual distribution and keep the draw if it fits; otherwise fall back to the window itself.
        lo, hi = (float(lo), float(hi)) if float(lo) <= float(hi) else (float(hi), float(lo))
        for _ in range(12):
            value = _one()
            if lo <= value <= hi:
                return str(value)
        return str(int(round(rng.uniform(lo, hi))))
    lo, hi = spec.get("lo"), spec.get("hi")
    decimals = bool(spec.get("decimals"))
    if lo is None or hi is None:
        if kind == "percent":
            lo, hi = 0.0, 100.0
        elif kind == "money":
            value = float(np.clip(np.exp(rng.normal(np.log(20.0), 1.0)), 0, 1000))
            if rng.random() < 0.5:  # round-number preference for money
                value = float(min([5, 10, 20, 25, 50, 100, 200, 500], key=lambda v: abs(v - value)))
            return _format_numeric(value, spec)
        else:
            return _format_numeric(float(np.clip(rng.negative_binomial(2, 0.4), 0, 30)), spec)
    span = hi - lo
    r = rng.random()
    if r < 0.10:
        value = lo
    elif r < 0.20:
        value = hi
    elif r < 0.32:
        value = lo + span / 2.0
    elif r < 0.45:
        step = 10.0 if span >= 50 else (5.0 if span >= 15 else 1.0)
        value = lo + round(rng.random() * span / step) * step
    else:
        value = lo + float(rng.beta(2.0, 2.0)) * span
    value = float(np.clip(value, lo, hi))
    return _format_numeric(value, spec, lo, hi)


# Public names for callers outside this module (the app's data preview uses them). Other
# modules must not import the underscore versions.
clean_question_text = _clean_question_text
infer_numeric_answer_spec = _infer_numeric_answer_spec
draw_numeric_answer = _draw_numeric_answer


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
    seed: Optional[int] = None,
    reverse_items: Optional[Any] = None,
) -> np.ndarray:
    """Inject inter-item correlation into independently generated scale items.

    Reverse-keyed items (``reverse_items``, 1-indexed) are recoded to the
    construct's direction before blending and flipped back afterwards. Without
    this, blending toward the RAW row mean forced reverse-keyed items to correlate
    POSITIVELY with the rest of the scale, so a researcher who recodes them (as
    anyone would) got a negative Cronbach's alpha.

    Uses a mixing approach: blend each item with a common factor (the
    participant's row mean) to achieve the target Cronbach's alpha, then
    round back to integer scale values while preserving per-item means/SDs.

    v1.4.11: Called after independent per-item generation to add realistic
    internal consistency without losing condition effects or persona variation.

    v1.2.8.1: per-item mixing weights so inter-item correlations are
    HETEROGENEOUS (real scales: some items load on the construct more strongly
    than others — a uniform weight makes every pair correlate ~identically, a
    structural "looks generated" tell flagged by Xie et al. 2026). Because this
    operates column-wise over the WHOLE matrix, the per-item weights are
    scale-stable (identical for every participant), so they produce genuine
    heterogeneity in the OBSERVED correlations rather than averaging out. The
    weights are centered on the uniform value, so the MEAN inter-item correlation
    — hence Cronbach's alpha — is preserved; per-item mean/SD restoration below
    keeps marginals (and condition effects) intact. Deterministic from `seed`.
    """
    n, k = item_matrix.shape
    if k <= 1 or n <= 1:
        return item_matrix

    _rev_idx = sorted({int(r) - 1 for r in (reverse_items or [])
                       if str(r).lstrip("-").isdigit() and 1 <= int(r) <= k})
    if _rev_idx:
        _flip = float(scale_min) + float(scale_max)
        _recoded = np.asarray(item_matrix, dtype=float).copy()
        _recoded[:, _rev_idx] = _flip - _recoded[:, _rev_idx]
        _out = _inject_inter_item_correlation(_recoded, target_alpha, scale_min, scale_max, seed=seed)
        _out = np.asarray(_out, dtype=float).copy()
        _out[:, _rev_idx] = _flip - _out[:, _rev_idx]
        return np.clip(np.round(_out), scale_min, scale_max).astype(int)

    # Target average inter-item correlation from Spearman-Brown
    denom = k - target_alpha * (k - 1)
    r_bar = target_alpha / denom if abs(denom) > 1e-9 else 0.5
    r_bar = float(np.clip(r_bar, 0.1, 0.9))

    # Mixing weight: x_new = w * common + (1-w) * x_old
    # Correlation between items ≈ w², so w = sqrt(r_bar)
    w = float(np.sqrt(r_bar))

    # v1.2.8.1: spread per-item weights ±~22% around w (corr(i,j) ≈ w_i·w_j), so
    # some item pairs correlate more than others. Centered on w → mean r ≈ r_bar.
    if seed is not None and k >= 3:
        _wr = np.random.RandomState(int(seed) & 0x7FFFFFFF)
        _wj = w * (1.0 + _wr.uniform(-0.22, 0.22, size=k))
        _wj = np.clip(_wj, 0.12, 0.97)
    else:
        _wj = np.full(k, w)

    # Common factor = row mean
    row_means = item_matrix.mean(axis=1, keepdims=True)

    # Blend each item with the common factor using its OWN weight
    mixed = _wj[np.newaxis, :] * row_means + (1.0 - _wj)[np.newaxis, :] * item_matrix

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


def _attenuate_inter_item_correlation(
    item_matrix: np.ndarray,
    current_alpha: float,
    target_alpha: float,
    scale_min: int,
    scale_max: int,
    seed: int,
) -> np.ndarray:
    """Lower an implausibly high Cronbach's alpha by adding item-specific noise.

    Items generated from a strong shared person tendency can reach alpha ~0.95+,
    above what real multi-item attitude scales show (typically 0.80-0.92) and a
    "too clean" tell. Classical test theory: adding independent noise with
    variance lam * var(item) scales the inter-item correlation by 1/(1+lam), so
    lam = r_now / r_target - 1. Each item is then rescaled by 1/sqrt(1+lam) around
    its own mean, which preserves per-item means and SDs. ``item_matrix`` must be in
    the CONSTRUCT direction (reverse items already recoded).
    """
    n, k = item_matrix.shape
    if k < 3 or n < 10 or current_alpha <= target_alpha:
        return item_matrix

    def _r_bar(alpha: float) -> float:
        return alpha / (k - alpha * (k - 1))

    r_now, r_tgt = _r_bar(min(current_alpha, 0.995)), _r_bar(target_alpha)
    if r_tgt <= 0 or r_now <= r_tgt:
        return item_matrix
    lam = r_now / r_tgt - 1.0
    rng = np.random.RandomState(int(seed) & 0x7FFFFFFF)
    X = np.asarray(item_matrix, dtype=float)
    mu = X.mean(axis=0, keepdims=True)
    C = X - mu
    noise = rng.normal(0.0, 1.0, size=X.shape) * np.sqrt(lam) * C.std(axis=0, keepdims=True)
    out = mu + (C + noise) / np.sqrt(1.0 + lam)
    return np.clip(np.round(out), scale_min, scale_max).astype(int)


# ---------------------------------------------------------------------------
# Economic-game outcome distributions (knowledge-base driven)
# ---------------------------------------------------------------------------
_GAME_QFN_CACHE: Dict[Any, Any] = {}


def _parse_subpop_band(name: str) -> Optional[Tuple[float, float]]:
    """Map a knowledge-base subpopulation name to an allocation band (proportions).

    Understands the numeric naming used for dictator/ultimatum types:
    'pure_selfish_zero' -> (0, 0); 'low_giver_1_20' -> (.01, .20);
    'fair_split_50' -> (.5, .5); 'generous_51_plus' -> (.51, .75);
    'low_offer_below_25' -> (0, .25); 'hyper_fair_above_50' -> (.5, .65).
    Returns None when the name carries no parseable range.
    """
    n = str(name).lower()
    if "zero" in n or n.endswith("selfish"):
        return (0.0, 0.0)
    m = re.search(r"(\d+)_(\d+)$", n)
    if m:
        return (int(m.group(1)) / 100.0, int(m.group(2)) / 100.0)
    m = re.search(r"(\d+)_plus$", n)
    if m:
        lo = int(m.group(1)) / 100.0
        return (lo, min(1.0, lo + 0.30))
    m = re.search(r"above_(\d+)$", n)
    if m:
        lo = int(m.group(1)) / 100.0
        return (lo, min(1.0, lo + 0.15))
    m = re.search(r"below_(\d+)$", n)
    if m:
        return (0.0, int(m.group(1)) / 100.0)
    m = re.search(r"_(\d+)$", n)
    if m:
        v = int(m.group(1)) / 100.0
        return (v, v)
    return None


def _game_quantile_fn(dist: Dict[str, Any]):
    """Return q in (0,1) -> allocation proportion in [0,1] for a game distribution.

    Subpopulation mixtures (zero spike / fair-split spike / bands) are used when
    every subpopulation name is parseable; otherwise a Beta matched to the
    published mean/SD (shape-agnostic). Returns None if neither is feasible.
    Deterministic (seeded empirical grid), numpy-only.
    """
    key = (str(dist.get("game")), str(dist.get("variant")))
    if key in _GAME_QFN_CACHE:
        return _GAME_QFN_CACHE[key]
    fn = None
    subpops = dist.get("subpops") or {}
    bands = []
    if subpops:
        for name, w in subpops.items():
            b = _parse_subpop_band(name)
            if b is None or w <= 0:
                bands = []
                break
            bands.append((b[0], b[1], float(w)))
    if bands:
        bands.sort(key=lambda t: (t[0], t[1]))
        total = sum(b[2] for b in bands)
        cum, edges = 0.0, []
        for lo, hi, w in bands:
            edges.append((cum / total, (cum + w) / total, lo, hi))
            cum += w

        _edges = tuple(edges)

        def _eval(q: float, gamma: float) -> float:
            q = min(max(q, 0.0), 1.0 - 1e-12)
            for c0, c1, lo, hi in _edges:
                if q < c1:
                    if hi <= lo:
                        return lo
                    t = (q - c0) / max(c1 - c0, 1e-12)
                    return lo + (t ** gamma) * (hi - lo)
            return _edges[-1][3]

        # The published mean and the subpopulation shares are not always mutually
        # consistent (dictator: shares imply ~0.23, Engel's mean is 0.28). Tilt mass
        # inside the continuous bands (one exponent, solved by bisection) so the
        # mixture hits the published mean while spikes and shares stay intact.
        _target = float(dist.get("mean", 0.0))
        _grid = np.linspace(0.0005, 0.9995, 2000)
        _lo_g, _hi_g = 0.25, 1.0
        gamma = 1.0
        if _target > 0:
            if np.mean([_eval(q, 1.0) for q in _grid]) < _target:
                for _ in range(30):
                    gamma = 0.5 * (_lo_g + _hi_g)
                    if np.mean([_eval(q, gamma) for q in _grid]) < _target:
                        _hi_g = gamma
                    else:
                        _lo_g = gamma
                gamma = 0.5 * (_lo_g + _hi_g)

        def fn(q: float, _g=gamma) -> float:
            return _eval(q, _g)
    else:
        m, sd = float(dist.get("mean", 0.5)), float(dist.get("sd", 0.2))
        if 0.04 < m < 0.96 and sd > 0.01:
            sd = min(sd, 0.95 * float(np.sqrt(m * (1.0 - m))))
            nu = m * (1.0 - m) / (sd * sd) - 1.0
            if nu > 0.2:
                grid = np.sort(np.random.RandomState(12345).beta(m * nu, (1.0 - m) * nu, 20001))
                qs = np.linspace(0.0, 1.0, grid.size)

                def fn(q: float, _g=grid, _q=qs) -> float:
                    return float(np.interp(q, _q, _g))
    _GAME_QFN_CACHE[key] = fn
    return fn


# Population mean of the persona x condition interaction multiplier in
# _generate_scale_response (measured empirically; see tests/test_effect_size_recovery.py).
_INTERACTION_MULTIPLIER_POP_MEAN = 1.12
# Latent-shift gain for game DVs (calibrated so recovered d on bounded, zero-inflated
# allocations tracks the configured d; see tests/test_effect_size_recovery.py).
_GAME_Z_GAIN = 0.9

# Share of a sample's chance (between-arm) variance that the game model's stratified draw lacks.
_GAME_MISSING_CHANCE_VAR = 0.7

# v1.3.0.6: games whose stored mean is the SHARE OF PARTICIPANTS choosing the "1" option
# (cooperate, stag, volunteer, enter, ...), so a two-option column is drawn at that rate.
_BINARY_RATE_GAME_TYPES = frozenset({
    "prisoners_dilemma", "stag_hunt", "volunteer_dilemma", "market_entry", "chicken",
    "battle_of_sexes",
})


def _binary_game_rate(dist: Optional[Dict[str, Any]]) -> Optional[float]:
    """Published choice rate of a two-option game outcome, or None when the entry is not one.

    A two-option column of a prisoner's dilemma / stag hunt / volunteer's dilemma / market entry /
    chicken / battle of the sexes game takes the stored mean as P(option 1); the binary trust entry
    does the same. Allocation games (dictator, ultimatum, ...) have no such reading.
    """
    if not dist:
        return None
    game, variant = str(dist.get("game")), str(dist.get("variant"))
    if game in _BINARY_RATE_GAME_TYPES or (game == "trust" and variant == "binary"):
        try:
            return float(min(0.98, max(0.02, float(dist.get("mean")))))
        except (TypeError, ValueError):
            return None
    return None
# Observed scale-score correlation produced by the pipeline for a latent correlation t:
#   r_obs ~= _XCORR_FLOOR + _XCORR_SLOPE * t
# The floor is common-method variance between unrelated scales (Podsakoff et al. 2003:
# r ~0.10-0.20); the slope is attenuation from imperfect scale reliability (alpha ~0.85)
# plus within-person noise. Fitted on a grid of targets (-0.6..+0.8, 4-item scales).
_XCORR_FLOOR = 0.045
_XCORR_SLOPE = 0.76
_XCORR_SLOPE_NEG = 0.68   # damped on the negative side (shared tendency/g-factor add positive covariance)


def _calibrate_latent_correlation(corr: Any) -> Any:
    """Invert the pipeline's attenuation so configured correlations are reproduced.

    Configured/inferred cross-DV correlations are treated as the OBSERVED correlation
    between scale scores. The latent matrix handed to the generator is
    t = (r - floor) / slope for every off-diagonal, clipped, then repaired to the
    nearest positive-definite correlation matrix (eigenvalue floor + rescale).
    """
    C = np.asarray(corr, dtype=float).copy()
    k = C.shape[0]
    if C.ndim != 2 or C.shape[0] != C.shape[1] or k < 2:
        return corr
    T = np.clip(np.where(C >= _XCORR_FLOOR, (C - _XCORR_FLOOR) / _XCORR_SLOPE,
                        (C - _XCORR_FLOOR) / _XCORR_SLOPE_NEG), -0.95, 0.95)
    np.fill_diagonal(T, 1.0)
    T = (T + T.T) / 2.0
    w, V = np.linalg.eigh(T)
    if w.min() < 1e-3:
        w = np.clip(w, 1e-3, None)
        T = V @ np.diag(w) @ V.T
        d = np.sqrt(np.diag(T))
        T = T / np.outer(d, d)
        np.fill_diagonal(T, 1.0)
    return T


# Cross-scale coupling knobs (calibrated against a grid of target correlations;
# see tests/test_effect_size_recovery.py::test_cross_scale_correlation_*).
_SHARED_TENDENCY_WEIGHT = 0.10   # weight of the persona-wide response tendency in each scale's base
_INDEP_TENDENCY_SD = 0.06        # SD of the per-scale independent tendency draw
_LATENT_WEIGHT_MULT = 1.6        # multiplier on the correlated-latent weight. It triples the SD of a scale composite, so
#                                  a user effect on a scale that carries it is built into the finished item responses
#                                  (_apply_user_effect_to_scale), not into the tendency: the shift calibrated for a lone
#                                  scale (_explicit_effect_scale) reaches only ~0.4 of the requested d next to it.
_G_FACTOR_MULT = 0.5             # multiplier on the common-method g-factor strength
_COHERENCE_MULT = 0.3            # multiplier on the running-mean cross-DV coherence pull
_INERTIA_MULT = 0.2              # multiplier on the recent-item anchoring pull (Schwarz & Strack)

# Literature anchoring of automatic effects ---------------------------------------
# When the study text names a paradigm that has a meta-analytic estimate in
# META_ANALYTIC_DB (anchoring, default effects, scarcity, ...), the coarse domain
# multiplier is replaced by one derived from that estimate, so an uncalibrated
# design gets the published magnitude rather than a generic domain guess.
_META_GENERIC_TOKENS = frozenset({
    "meta", "effect", "effects", "general", "expanded", "extended", "and", "the", "for", "vs",
    "in", "of", "to",
})
# Reference nominal d produced by a multiplier of 1.0 (measured on the valence
# contrast, see tests/test_effect_size_recovery.py::test_meta_anchored_effect_magnitude).
_META_REFERENCE_D = 0.6
_META_INDEX_CACHE: Optional[List[Tuple[str, float, Tuple[str, ...]]]] = None


# Single-word paradigm names distinctive enough to trigger on their own; every other
# single-token key needs one of its explicit phrase aliases below.
_META_SINGLE_TOKEN_OK = frozenset({
    "anchoring", "bystander", "inoculation", "endowment", "deindividuation", "spotlight",
    "placebo", "interleaving", "retargeting", "representativeness", "decoy", "denomination",
    "psychotherapy", "scarcity",
})
_META_ALIASES: Dict[str, Tuple[str, ...]] = {
    "default_effect": ("default effect", "default option", "opt out", "opt-out", "opt in", "opt-in",
                       "default enrollment"),
    "framing_general_meta": ("framing effect", "message framing", "gain frame", "loss frame",
                             "gain framing", "loss framing"),
    "testing_effect_meta": ("testing effect", "retrieval practice"),
}
if _paradigm_coverage is not None:
    # v1.3.0.5: phrases for entries a title or condition label could not reach before, and for the
    # paradigms added in `paradigm_coverage`. Matching is whole-phrase (see _meta_index).
    for _k, _als in _paradigm_coverage.ENTRY_ALIASES.items():
        _META_ALIASES[_k] = tuple(dict.fromkeys(_META_ALIASES.get(_k, ()) + tuple(_als)))


def _meta_index() -> List[Tuple[str, float, Tuple[Any, ...]]]:
    """Build (key, effect_d, patterns) for contrast-type META_ANALYTIC_DB entries.

    Multi-token names must occur in order within a short window (adjacent-ish), so
    unrelated words scattered through a study text cannot assemble a paradigm.
    """
    global _META_INDEX_CACHE
    if _META_INDEX_CACHE is not None:
        return _META_INDEX_CACHE
    out: List[Tuple[str, float, Tuple[Any, ...]]] = []
    # Entries added in v1.3.0.5 are recalled, not source-checked: their magnitude is damped by their
    # verification tier (as the literature fallback already does), so they push the data less hard
    # than a sourced value. Older entries keep their published magnitude.
    _damped = _paradigm_coverage.RULE_ONLY_KEYS if _paradigm_coverage is not None else frozenset()
    if HAS_KNOWLEDGE_BASE:
        for key, entry in META_ANALYTIC_DB.items():
            _signed = float(getattr(entry, "effect_d", 0.0) or 0.0)
            d = abs(_signed)
            if d < 0.05 or "game" in key or "auction" in key or "taking" in key:
                continue  # baselines/games are handled by GAME_CALIBRATIONS
            if key in _damped:
                if _signed < 0:
                    continue  # a harm-type effect needs its sign from the label rule, not the study text
                if HAS_EMPIRICAL_REGISTRY:
                    d *= float(_empirical_registry.confidence_weight("meta", key))
            toks = [t for t in re.split(r"[^a-z]+", key.lower()) if t and t not in _META_GENERIC_TOKENS]
            _als = _META_ALIASES.get(key, ())
            if not toks or not (any(len(t) >= 5 for t in toks) or _als):
                continue
            pats: List[Any] = []
            if len(toks) == 1:
                if toks[0] in _META_SINGLE_TOKEN_OK:
                    pats.append(re.compile(r"\b" + re.escape(toks[0][:max(5, len(toks[0]) - 2)]) + r"\w*"))
            elif any(len(t) >= 5 for t in toks) and key not in _damped:
                gap = r"[\W_]+(?:\w+[\W_]+){0,2}"
                pats.append(re.compile(gap.join(r"\b" + re.escape(t[:max(5, len(t) - 2)]) + r"\w*" for t in toks)))
            for al in _als:
                # "-" and "_" read as spaces on both sides: "loss_frame" and "opt-out" match "loss frame"/"opt out"
                pats.append(re.compile(r"\b" + re.escape(re.sub(r"[_\-]+", " ", al)) + r"\b"))
            if pats:
                out.append((key, d, tuple(pats)))
    _META_INDEX_CACHE = out
    return out


def _match_meta_entry(text: str) -> Optional[Tuple[Tuple[str, ...], float]]:
    """Return ``(knowledge-base keys, published |d|)`` for the paradigm named in ``text``, or None.

    The paradigm with the most specific (longest) match wins. When several
    different paradigms match equally well and disagree by more than 0.15 the
    text is ambiguous and no anchoring is applied. Equally good, agreeing
    matches are averaged and all their keys are returned.

    "_" and "-" read as spaces ("loss_frame", "foot-in-the-door"), so snake_case condition
    labels reach the same phrases as prose does.
    """
    text = re.sub(r"[_\-]+", " ", str(text).lower())
    hits: List[Tuple[int, str, float]] = []
    for _key, d, pats in _meta_index():
        best = 0
        for pat in pats:
            m = pat.search(text)
            if m:
                best = max(best, len(m.group(0)))
        if best:
            hits.append((best, _key, d))
    if not hits:
        return None
    top = max(h[0] for h in hits)
    group = [(k, d) for n, k, d in hits if n >= top * 0.999]
    if _paradigm_coverage is not None and len(group) > 1:
        # a more specific reading of the same words beats the older, broader one unless the
        # text carries the older paradigm's own vocabulary (see paradigm_coverage.TIE_PREFER)
        for _pref, _other, _guard in _paradigm_coverage.TIE_PREFER:
            _gk = {k for k, _d in group}
            if _pref in _gk and _other in _gk and not re.search(_guard, text):
                group = [(k, d) for k, d in group if k != _other]
    ds = [d for _k, d in group]
    if max(ds) - min(ds) > 0.15:
        return None
    return tuple(sorted(k for k, _d in group)), float(sum(ds) / len(ds))


def _match_meta_effect(text: str) -> Optional[float]:
    """Return the PUBLISHED meta-analytic |d| for the paradigm named in ``text``, or None.

    This is the knowledge-base value, before the replication shrinkage that the
    engine applies to inferred effects (see ``_shrink_inferred_meta_effect``).
    """
    hit = _match_meta_entry(text)
    return None if hit is None else hit[1]


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
        # v1.2.3.0: Adaptive Behavioral Engine — narrative-enhanced wrapper (used by ABE 3.0)
        use_abe_v2: bool = False,
        # v1.2.2.8: Per-question LLM OE cap for free-tier protection.
        # When > 0, limits LLM-generated OE responses to this many participants
        # per question; remaining participants get template fallback automatically.
        # This keeps the full sample size N for numeric scales.
        free_llm_oe_cap: int = 0,
        # v1.2.9.1: True (default) keeps the long-standing behaviour of inferring condition
        # effects from condition names. False builds in ONLY the effects you specify, so every
        # other contrast is a true null.
        auto_effects: bool = True,
    ):
        self.progress_callback = progress_callback
        self.auto_effects = bool(auto_effects)
        self.use_socsim_experimental = bool(use_socsim_experimental)
        self.use_abe_v2 = bool(use_abe_v2)
        self.study_title = str(study_title or "").strip()
        self.study_description = str(study_description or "").strip()
        self.sample_size = int(sample_size)
        # Normalize condition names: strip whitespace AND non-breaking spaces (\xa0)
        self.conditions = [
            _condition_label(c).replace('\xa0', ' ').strip()
            for c in (conditions or [])
            if _condition_label(c).replace('\xa0', ' ').strip()
        ]
        if not self.conditions:
            # v1.1.1.5: Use logger instead of self._log() here — validation_log
            # is not yet initialized at this point in __init__.
            logger.warning("No conditions specified — defaulting to single 'Condition A'")
            self.conditions = ["Condition A"]
        self.factors = _normalize_factors(factors, self.conditions)
        self.scales = _normalize_scales(scales)
        # v1.2.5.3: Build DV description lookup for condition effect intelligence
        # v1.2.5.5: Store BOTH space and underscore variants so lookup always matches
        # (variable names may arrive as "trust scale" or "trust_scale" depending on path)
        self._dv_descriptions: Dict[str, str] = {}
        for _sc in self.scales:
            _var = str(_sc.get("variable_name", _sc.get("name", ""))).lower().strip()
            _desc = str(_sc.get("dv_description", ""))
            if _var and _desc:
                self._dv_descriptions[_var] = _desc
                # Store underscore variant for cleaned column names
                _var_underscore = _var.replace(" ", "_").replace("-", "_")
                if _var_underscore != _var:
                    self._dv_descriptions[_var_underscore] = _desc
                # Store space variant for original names
                _var_space = _var.replace("_", " ")
                if _var_space != _var:
                    self._dv_descriptions[_var_space] = _desc
        self.additional_vars = additional_vars or []
        self.demographics = demographics or {}
        self.attention_rate = float(attention_rate)
        self.random_responder_rate = float(random_responder_rate)
        self.effect_sizes = effect_sizes or []
        # v1.2.9.1: user effects built into finished scales (see _apply_user_effect_to_scale)
        self._latent_dv_names: Set[str] = set()
        self._deferred_effect_vars: Set[str] = set()
        self._deferred_effect_log: List[Dict[str, Any]] = []
        self._effect_build_errors: List[str] = []
        self._reversal_ok_arr: Optional[Tuple[str, np.ndarray]] = None
        self.exclusion_criteria = exclusion_criteria or ExclusionCriteria()
        self.open_ended_questions = _normalize_open_ended(open_ended_questions)
        # v1.2.9.1: a text box that is also a detected numeric DV already has its numbers in the
        # DV columns; do not add a second, prose-filled column with the same name.
        self.open_ended_questions, self._oe_dropped_as_dv_duplicates = _drop_oe_duplicating_dvs(
            self.open_ended_questions, self.scales)
        # Columns whose answers are structured (ages, IDs, ZIP codes, counts, demographics), not prose:
        # the stylometric, validation and tidy passes must leave them exactly as generated.
        self._structured_oe_columns: Set[str] = set()
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
        self.free_llm_oe_cap = max(0, int(free_llm_oe_cap))
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

        # Deterministic run id: the same seed must give a byte-identical dataset, so the
        # id carries no wall-clock time. The generation timestamp lives in metadata only.
        self.run_id = f"{self.mode.upper()}_S{self.seed:010d}"

        # v1.2.7.5: Do NOT seed the GLOBAL np.random / random here. All generation
        # uses per-call seeded RandomState/random.Random(self.seed + ...) instances,
        # so global seeding was both redundant and a cross-session side effect in
        # multi-user Streamlit (one run reseeding global state used by another).
        # Verified: output is identical with the global state arbitrarily perturbed.

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
        all_domains = list(dict.fromkeys(self.detected_domains + condition_domains))
        self.detected_domains = all_domains if all_domains else self.detected_domains

        # v1.3.6: Also merge in explicitly provided persona_domains from builder
        _explicit_persona_domains = self.study_context.get("persona_domains", [])
        if _explicit_persona_domains and isinstance(_explicit_persona_domains, list):
            _merged = list(dict.fromkeys(self.detected_domains + _explicit_persona_domains))
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
                # v1.2.3.5: Use logger instead of self._log() — validation_log
                # is not yet initialized at this point in __init__.
                logger.info("Persona pool expanded via adjacent domains: "
                            "%d → %d domain-specific personas",
                            len(_domain_specific),
                            len({n: p for n, p in _expanded_personas.items()
                                 if p.category != 'response_style'}))

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

        # v1.2.3.5: validation_log MUST be initialized before any _log() calls.
        # The ABE v2 init block below uses _log(), so this must come first.
        self.column_info: List[Tuple[str, str]] = []
        # v1.2.9.1: what the empirical-realism passes did, for the audit page and tests
        self._item_realism_log: List[Dict[str, Any]] = []
        # Which registry calibrations actually changed a number in THIS run, so the
        # app can tell the user how much of their dataset rests on measured evidence
        # rather than on unverified literature.
        self.registry_ledger = (
            _empirical_registry.RunLedger() if HAS_EMPIRICAL_REGISTRY else None
        )
        self.validation_log: List[str] = []
        self._scale_generation_log: List[Dict[str, Any]] = []

        # Initialize comprehensive response generator if available
        # v1.2.3.0: When use_abe_v2 is True, use AdaptiveBehavioralEngineV2
        # as drop-in replacement (same generate() interface, enhanced narrative output).
        self.comprehensive_generator = None
        if self.use_abe_v2 and HAS_ABE_V2:
            self.comprehensive_generator = AdaptiveBehavioralEngineV2(seed=self.seed)
            if self.study_context:
                self.comprehensive_generator.set_study_context(self.study_context)
            self._log("Using Adaptive Behavioral Engine 3.0 (narrative-enhanced)")
        elif HAS_RESPONSE_LIBRARY:
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

    def llm_attempts_allowed(self) -> bool:
        """Whether this run may call the LLM at all.

        ``allow_template_fallback`` means two different things to the two
        callers that set it, and conflating them caused a silent regression:

        * "Template Engine" / ABE set it to mean *do not use the LLM*.
        * "Built-in AI" sets it to mean *fall back gracefully when the LLM is
          unavailable* — it still wants the LLM tried first.

        ``free_llm_oe_cap > 0`` is what distinguishes the second case: it is an
        upper bound on LLM-generated open-ended responses, so a caller that
        wants the LLM attempted with graceful fallback sets a positive cap.
        Both of the engine's LLM gates (pool prefill and per-participant
        generation) must agree, so they both read this one predicate.
        """
        return (not self.allow_template_fallback) or self.free_llm_oe_cap > 0

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
                if dropout_item >= total_items:
                    continue  # would blank nothing: the participant finishes, so it is not a dropout
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
                    # Keep composites consistent with generate(): recode reverse-coded
                    # items before averaging (see composite construction there).
                    _rev, _flip = set(), 0.0
                    for _le in (getattr(self, "_scale_generation_log", None) or []):
                        _cols = _le.get("columns_generated") or []
                        if _cols and _cols[0].rsplit("_", 1)[0] == prefix:
                            _rev = set(_le.get("reverse_items") or [])
                            _flip = float(_le.get("scale_min", 0)) + float(_le.get("scale_max", 0))
                            break
                    for i in range(n):
                        item_vals = []
                        for ic in item_cols:
                            v = data[ic][i]
                            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                                _idx = ic.rsplit("_", 1)[-1]
                                _is_rev = _idx.isdigit() and int(_idx) in _rev
                                item_vals.append(_flip - float(v) if _is_rev else float(v))
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
                    # v1.2.7.5: list-valued items/num_items define the item names + count
                    item_names = list(scale.get("item_names") or [])
                    raw_items = scale.get("num_items")
                    if raw_items is None:
                        raw_items = scale.get("items")  # QSF detection key
                    if isinstance(raw_items, (list, tuple)):
                        item_names = [str(x) for x in raw_items if str(x).strip()]
                        n_items = len(item_names)
                    else:
                        n_items = safe_int(raw_items, len(item_names) if item_names else 5)
                    n_items = max(1, n_items)
                    normalized.append({
                        "name": name,
                        "variable_name": safe_str(scale.get("variable_name"), name.replace(" ", "_")),
                        "num_items": n_items,
                        "scale_points": pts,
                        "reverse_items": list(scale.get("reverse_items") or []),
                        "item_names": item_names,
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

    def _is_explicit_condition(self, condition: str) -> bool:
        """True when a user-specified effect names this condition, so name-based extras
        (trait shifts) must not pile on top of the effect the user asked for."""
        cache = getattr(self, "_explicit_cond_cache", None)
        if cache is None:
            cache = {}
            self._explicit_cond_cache = cache
        key = str(condition)
        if key in cache:
            return cache[key]
        cond_l = _label_norm(key)
        hit = False
        for effect in getattr(self, "effect_sizes", None) or []:
            for attr in ("level_high", "level_low"):
                lvl = _label_norm(_spec_get(effect, attr, ""))
                if lvl and _word_in(lvl, cond_l):
                    hit = True
        cache[key] = hit
        return hit

    # ------------------------------------------------------------------
    # v1.2.9.1: which generated variable does an effect spec belong to?
    # ------------------------------------------------------------------
    def _variable_alias_map(self) -> Dict[str, Set[Tuple[str, ...]]]:
        """Per generated variable (its cleaned column prefix), the token tuples of every name it
        answers to: display name, variable name and column prefix. Built once."""
        amap = getattr(self, "_var_alias_cache", None)
        if amap is None:
            amap = {}
            for sc in getattr(self, "scales", None) or []:
                name = str(sc.get("name", "")).strip()
                var_name = str(sc.get("variable_name", "")).strip()
                col = _clean_column_name(var_name or name)
                aliases = {_var_tokens(x) for x in (name, var_name, col) if x}
                aliases.discard(())
                amap.setdefault(col, set()).update(aliases)
            self._var_alias_cache = amap
        return amap

    def _spec_applies_to_variable(self, spec_variable: Any, variable: str) -> bool:
        """True when an effect spec's variable names the generated variable ``variable``.

        Names are compared as lists of whole words, so "Perceived Quality" matches the column
        "Perceived_Quality" and "Trust" matches "Trust_Score", but "Trust" no longer matches
        "Distrust" and "DV1" no longer matches "DV10" (the old test was a substring test). A scale
        that carries exactly the spec's name owns the spec; the word-run fallback is used only
        when no scale has that name. A blank variable names nothing.
        """
        spec_tokens = _var_tokens(spec_variable)
        if not spec_tokens:
            return False
        amap = self._variable_alias_map()
        mine = amap.get(str(variable)) or ({_var_tokens(variable)} - {()})
        if spec_tokens in mine:
            return True
        for col, aliases in amap.items():
            if col != str(variable) and spec_tokens in aliases:
                return False
        return any(_tokens_contain(alias, spec_tokens) for alias in mine)

    def _variable_has_user_spec(self, variable: str) -> bool:
        """True when any user-specified effect targets this variable (memoised)."""
        cache = getattr(self, "_var_spec_cache", None)
        if cache is None:
            cache = {}
            self._var_spec_cache = cache
        key = str(variable)
        if key not in cache:
            cache[key] = any(
                self._spec_applies_to_variable(_spec_get(e, "variable", ""), key)
                for e in (getattr(self, "effect_sizes", None) or [])
            )
        return cache[key]

    def _condition_name_may_shape(self, variable: str) -> bool:
        """Whether the condition NAME may shift this variable's baseline or response style.

        Names do so only through the inferred effects: never with ``auto_effects=False`` (the true
        null) and never for a variable the user gave an effect, where every condition, named in the
        spec or not, must start from the same baseline so the requested d is the only difference.
        """
        return bool(getattr(self, "auto_effects", True)) and not self._variable_has_user_spec(variable)

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

        v1.2.6.4 PERFORMANCE: This method is deterministic for a given
        (condition, variable) pair within a single run — effect_sizes,
        conditions, and study context do not change during generation. It is
        called once per participant per scale item (N × items times), each
        doing heavy regex/string matching. Memoize the result so the expensive
        computation runs once per unique (condition, variable) pair instead of
        tens of thousands of times. This is the single largest hot-path cost
        for large-N simulations.
        """
        # v1.2.6.4: Memoization cache keyed on (condition, variable)
        _cache = getattr(self, "_effect_cache", None)
        if _cache is None:
            _cache = {}
            self._effect_cache = _cache
        _cache_key = (str(condition), str(variable), getattr(self, "_scale_effect_meta", {}).get(str(variable)))
        if _cache_key in _cache:
            return _cache[_cache_key]
        _result = self._compute_effect_for_condition(condition, variable)
        _cache[_cache_key] = _result
        return _result

    # Effective between-item correlation of the simulated response pipeline used
    # to relate single-item d to scale-mean d (fitted on a 5/7/11-pt, k=1..8 grid).
    _EFFECT_ITEM_RHO = 0.20

    def _explicit_effect_scale(self, variable: str) -> float:
        """Multiplier that makes a configured Cohen's d refer to the scale MEAN.

        Averaging k items shrinks the within-condition SD by sqrt((1+(k-1)rho)/k)
        while the condition gap is unchanged, so composite d exceeds item d by
        sqrt(k/(1+(k-1)rho)). Dividing the shift by that factor keeps the
        recovered composite d on target. Very wide numeric scales (>= 50 points,
        e.g. 0-100 sliders) recover ~10% low, so they get a 1.10 boost.
        Returns 1.0 when the scale geometry is unknown (e.g. direct callers).

        This calibration describes a LONE scale. With several scales in the design the cross-scale
        latent term enlarges the composite's SD about threefold, and the effect is then applied to the
        finished responses instead (_apply_user_effect_to_scale), so it does not use this factor.
        """
        meta = getattr(self, "_scale_effect_meta", {}).get(str(variable))
        if not meta:
            return 1.0
        k, smin, smax = meta
        k = max(1, int(k))
        rho = self._EFFECT_ITEM_RHO
        factor = 1.0 / float(np.sqrt(k / (1.0 + (k - 1) * rho)))
        if (smax - smin) >= 50:
            factor *= 1.10
        # v1.2.9.1 empirical corrections, measured over 12 independent seeds per cell
        # (N=1,200, d=0.5 and 0.8 behave the same). The rho-based factor over-corrects as
        # items are added, because the pipeline's inter-item correlation falls with k:
        # realised/requested was 1.00 (k=1-3), 1.07 (5), 1.16 (8), 1.19 (12), 1.24 (15),
        # 1.25 (20). It under-recovers on 2- and 3-point scales (0.75 and 0.93 of the request).
        if k > 3:
            factor /= 1.0 + 0.14 * float(np.log(min(k, 30) / 3.0))
        points = int(smax - smin) + 1
        if points <= 2:
            factor *= 1.30
        elif points == 3:
            factor *= 1.07
        return factor

    @staticmethod
    def _combine_matched_effects(matched: List[Tuple[str, "frozenset", float]]) -> float:
        """Combine the shifts of every effect spec that names one condition.

        Specs that describe the SAME factor are alternatives for one dimension (two contrasts against
        a shared control, a duplicated spec) and are averaged, so they do not stack. Specs on
        DIFFERENT factors are the main effects of a factorial design and are added: a cell that is
        high on A (d=0.5) and high on B (d=0.5) sits 0.5 + 0.5 above the cell that is low on both,
        so each marginal contrast keeps its requested d (averaging halved both, d=0.5 -> 0.25-0.28).
        Two specs are "the same factor" when their factor labels match or they share a level label.
        """
        groups: List[Tuple[Set[str], Set[str], List[float]]] = []
        for factor, levels, value in matched:
            joined = [g for g in groups if factor in g[0] or (set(levels) & g[1])]
            merged: Tuple[Set[str], Set[str], List[float]] = (
                {factor}.union(*[g[0] for g in joined]),
                set(levels).union(*[g[1] for g in joined]),
                [value] + [v for g in joined for v in g[2]],
            )
            groups = [g for g in groups if all(g is not j for j in joined)] + [merged]
        return float(sum(sum(g[2]) / len(g[2]) for g in groups))

    # Normalised per-side shift for one Cohen's d (see _compute_effect_for_condition); shared with the inferred-effect helper
    _EFFECT_D_TO_NORMALIZED = 0.109

    def _inferred_effect_value(self, condition: str, variable: str) -> float:
        """Normalised mean shift inferred from the condition names: the keyword rules first, then, when they find
        nothing, a strict content match against the literature table (never for the reference arm).

        Used when the user specified no effect for this variable and inference is on; the caller records
        where the value came from."""
        COHENS_D_TO_NORMALIZED = self._EFFECT_D_TO_NORMALIZED
        _effect_scale = self._explicit_effect_scale(variable)
        _auto = self._get_automatic_condition_effect(condition, variable)
        # "Nothing matched" does not come back as exactly zero: measured on
        # unmatched labels, the keyword pipeline returns noise-level values around
        # +/-0.004 normalized, i.e. Cohen's d near 0.01. Anything below d = 0.05 is
        # indistinguishable from no manipulation at all, so that is the trigger.
        _NEGLIGIBLE = 0.05 * COHENS_D_TO_NORMALIZED

        # v1.3.0.5 — paradigms with a curated label phrase ("Mortality salience", "Ostracized",
        # "Gamified", "Graphic warning", ...) take their recalled literature magnitude AND sign even
        # where the keyword rules guessed something: those rules know nothing of these paradigms
        # beyond a generic valence, and several would have signed an exclusion or a disclosure as a
        # benefit. Never for the reference arm, and never for economic-game designs, whose
        # calibrations are owned by the game models.
        if (HAS_LITERATURE_EFFECTS and _paradigm_coverage is not None
                and self._is_control_arm(condition) and self._design_has_curated_arm(variable)):
            # the reference arm of a design built on a curated paradigm is the zero point: it must not
            # keep a keyword residual that the other arm's label happens to induce
            return 0.0
        if (HAS_LITERATURE_EFFECTS and _paradigm_coverage is not None
                and not self._is_control_arm(condition) and not self._is_economic_game_context(variable)):
            _curated = None
            try:
                _curated = _literature_effects.lookup_curated(
                    str(condition), rng=self._stable_rng("literature-effect", str(condition), str(variable)))
            except Exception:
                _curated = None
            if _curated is not None:
                return self._literature_match_to_shift(condition, variable, _curated, "curated paradigm phrase")

        if abs(_auto) >= _NEGLIGIBLE or not HAS_LITERATURE_EFFECTS:
            return _auto * _effect_scale

        # v1.2.9.1 — LAST RESORT. The keyword rules above are a closed set: an
        # uploaded study whose conditions are called `descriptive_norm_high` /
        # `descriptive_norm_low` or `cognitive_dissonance_induced` matches none of
        # them, and the condition then does NOTHING — the user's design silently
        # becomes a null design, which is the worst failure a simulator can have.
        # So when the rules find no effect at all, match the condition against the
        # meta-analytic table on content instead. The match is deliberately strict
        # (two shared content words, one of which must come from the condition
        # label) because a wrong effect is worse than none: an unmatched condition
        # falls back to zero, which is explicit and inspectable. The returned value
        # is passed through empirical_registry.adjust_effect(), so an entry whose
        # numbers were never checked against a source pushes the data less hard.
        #
        # The reference arm never takes a literature effect. The match is made on
        # content, and a control label usually repeats the paradigm it is the
        # control FOR -- `cognitive_dissonance_control` shares every content word
        # with `cognitive_dissonance_induced`. Both would match the same entry and
        # both would be shifted by the same published d, which leaves no contrast
        # at all: the fallback meant to rescue a null design would have recreated
        # one. A control arm is the zero point, exactly as it is on the explicit
        # path above.
        if self._is_control_arm(condition):
            return _auto * _effect_scale
        try:
            _lit = _literature_effects.lookup(
                condition=str(condition),
                variable=str(variable),
                study_context=f"{self.study_title or ''} {self.study_description or ''}",
                rng=self._stable_rng("literature-effect", str(condition), str(variable)),
                **({"policy": self._INFERRED_EFFECT_POLICY} if self._INFERRED_EFFECT_POLICY is not None else {}),
            )
        except Exception:
            return _auto * _effect_scale
        if _lit is None:
            return _auto * _effect_scale
        return self._literature_match_to_shift(condition, variable, _lit, "no keyword rule matched")

    def _design_has_curated_arm(self, variable: str) -> bool:
        """Whether some non-reference arm of this design names a curated paradigm (see `lookup_curated`)."""
        if self._is_economic_game_context(variable):
            return False
        cache = getattr(self, "_curated_arm_cache", None)
        if cache is None:
            cache = self._curated_arm_cache = {}
        if "any" not in cache:
            found = False
            for c in (self.conditions or []):
                if self._is_control_arm(str(c)):
                    continue
                try:
                    if _literature_effects.lookup_curated(str(c)) is not None:
                        found = True
                        break
                except Exception:
                    continue
            cache["any"] = found
        return bool(cache["any"])

    def _is_economic_game_context(self, variable: str) -> bool:
        """Whether the DV, the study text or the condition names look like an economic game."""
        _ctx = " ".join([
            " ".join(str(c).lower() for c in (self.conditions or [])),
            str(self.study_title or "").lower(), str(self.study_description or "").lower(),
            str(variable).lower(), str(self._dv_descriptions.get(str(variable).lower(), "")).lower(),
        ])
        return _paradigm_coverage is None or _paradigm_coverage.is_economic_game_text(_ctx)

    def _literature_match_to_shift(self, condition: str, variable: str, _lit: Any, why: str) -> float:
        """Normalised shift for one arm from a literature match, with its provenance logged."""
        _lit_d = float(_lit.effect_d)
        if getattr(_lit, "polarity_aware", False):
            # A beneficial (or harmful) manipulation moves a positive construct one way and a
            # symptom-type construct (distress, prejudice, use, ...) the other.
            _dv_text = (str(variable).replace("_", " ") + " "
                        + str(self._dv_descriptions.get(str(variable).lower(), ""))).lower()
            if self._NEGATIVE_DV_RE.search(_dv_text) or (
                    _paradigm_coverage is not None and _paradigm_coverage.dv_is_negative(_dv_text)):
                _lit_d = -_lit_d
        _normalized = _lit_d * self._EFFECT_D_TO_NORMALIZED * self._explicit_effect_scale(variable)
        if getattr(_lit, "rule", "") == "label_phrase" and _paradigm_coverage is not None:
            # a curated paradigm contrasts with a zero-point reference arm: apply it in the explicit
            # currency (gap = 2 x 0.109 x d), as the study-level anchor does, so d is what is realised
            _normalized *= _paradigm_coverage.CURATED_GAP_FACTOR
        self._log(
            f"{why}: condition '{condition}' for '{variable}'; "
            f"used literature entry '{_lit.key}' ({_lit.source}, published "
            f"d={_lit.published_d}, applied d={_lit_d:.3f}, "
            f"verification={_lit.status})"
        )
        if not hasattr(self, "_literature_effect_log"):
            self._literature_effect_log = []
        self._literature_effect_log.append(
            dict(condition=str(condition), variable=str(variable), **_lit.as_dict())
        )
        if not hasattr(self, "_inferred_effect_log"):
            self._inferred_effect_log = []
        self._inferred_effect_log.append({
            "path": "literature_fallback", "condition": str(condition), "variable": str(variable),
            "key": _lit.key, "published_d": round(float(_lit.published_d), 4),
            "shrinkage_factor": round(float(getattr(_lit, "shrinkage", 1.0)), 4),
            "applied_d": round(float(_lit.effect_d), 4), "verification": _lit.status,
        })
        return _normalized

    def _compute_effect_for_condition(self, condition: str, variable: str) -> float:
        """v1.2.6.4: Uncached implementation of effect computation (see
        _get_effect_for_condition for memoization wrapper and docs)."""
        # Cohen's d is defined on the GAP between the two levels, in units of the
        # within-condition SD. The gap is applied symmetrically (+d/2 / -d/2), and
        # the empirical end-to-end gain of the response pipeline is calibrated so
        # that the recovered d on a SINGLE ITEM equals the configured d:
        #   0.125 = (per-side shift in scale-range units) / d, fitted on 7pt/5pt/11pt
        #   single-item scales (recovered/target = 1.00 +/- 0.05, tests/
        #   test_effect_size_recovery.py). Multi-item scales are then corrected by
        #   _explicit_effect_scale() so d refers to the scale MEAN.
        # Earlier values (0.30-0.40 per side) ignored the two-sided application and
        # the real SD, inflating observed d ~4x (d=0.5 -> ~2.1).
        COHENS_D_TO_NORMALIZED = self._EFFECT_D_TO_NORMALIZED

        # Check explicit effect size specifications -- accumulate ALL matching effects
        # for factorial designs where multiple effect specs may apply to one condition.
        # Each entry is (factor label, the spec's two levels, shift) so effects on different
        # factors can be added (see _combine_matched_effects).
        matched_effects: list = []
        _variable_has_spec = False  # any explicit effect spec targets this variable
        condition_lower = _label_norm(condition)

        for effect in self.effect_sizes:
            # v1.1.1.5: Support both EffectSizeSpec objects AND plain dicts.
            # The app always passes EffectSizeSpec, but the engine should be
            # robust against dicts from tests, API callers, or legacy code.
            _eget = _spec_get

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

            # Check if this effect spec matches the current variable. Variable names reach the
            # generator in their column form ("Perceived_Quality") while users type display names
            # ("Perceived Quality"); they are compared as lists of whole words, so an explicit
            # effect is never silently dropped (it would be replaced by keyword-derived automatic
            # effects) and never leaks onto a variable that merely contains the name ("Distrust").
            variable_matches = self._spec_applies_to_variable(_eget(effect, 'variable', ''), variable)

            if variable_matches:
                _variable_has_spec = True
                direction = str(_eget(effect, 'direction', 'positive')).lower().strip()

                # v1.0.1.3: word-boundary matching prevents false positives (level "ai" vs
                # condition "wait"); v1.2.9.1: labels may start or end with punctuation
                # ("Norm message (Control)", "80%"). When both levels occur in the label the
                # longer, more specific one wins ("no ai" contains "ai").
                side = _spec_side(effect, condition_lower)
                _spec_key = (
                    _label_norm(_eget(effect, 'factor', '')),
                    frozenset({_label_norm(_eget(effect, 'level_high', '')), _label_norm(_eget(effect, 'level_low', ''))} - {""}),
                )
                if side > 0:
                    d = cohens_d if direction == "positive" else -cohens_d
                    matched_effects.append((_spec_key[0], _spec_key[1], d * COHENS_D_TO_NORMALIZED))
                elif side < 0:
                    d = -cohens_d if direction == "positive" else cohens_d
                    matched_effects.append((_spec_key[0], _spec_key[1], d * COHENS_D_TO_NORMALIZED))

        # v1.2.9.1: remember where each (condition, variable) effect came from so the
        # metadata can say whether a contrast was requested ("user"), inferred from the
        # condition names ("inferred") or absent ("none"). `unit` is the normalised
        # shift that corresponds to one Cohen's d for this variable.
        _applied = getattr(self, "_applied_effects", None)
        if _applied is None:
            _applied = {}
            self._applied_effects = _applied
        _unit = COHENS_D_TO_NORMALIZED * self._explicit_effect_scale(variable)

        if matched_effects:
            # Effects that describe the same factor (same factor label, or a shared level such as a
            # common control) are averaged so they do not stack; effects on different factors are
            # main effects of a factorial design and add up.
            _value = self._combine_matched_effects(matched_effects) * self._explicit_effect_scale(variable)
            _applied[(str(condition), str(variable))] = {"source": "user", "offset": float(_value), "unit": _unit}
            return _value

        # The user configured effects for this variable but none involves this
        # condition (e.g. a Control group): it is the reference level. Do not add
        # keyword-derived automatic effects on top of an explicit design.
        if _variable_has_spec:
            _applied[(str(condition), str(variable))] = {"source": "user", "offset": 0.0, "unit": _unit}
            return 0.0

        # True null: with inferred effects switched off only user-specified effects
        # are built into the data.
        if not getattr(self, "auto_effects", True):
            _applied[(str(condition), str(variable))] = {"source": "none", "offset": 0.0, "unit": _unit}
            return 0.0

        # AUTO-GENERATE effect if no explicit specification
        # This ensures conditions ALWAYS produce different means.
        # Automatic effects are expressed in the same normalised-shift currency as
        # explicit ones (nominal d = gap / 0.25 of range), so they get the same
        # item-count correction: otherwise a 4+ item composite shows d ~1.3-2x the
        # literature value the keyword rule encodes (valence 1.3 vs ~0.6).
        _value = self._inferred_effect_value(condition, variable)
        _applied[(str(condition), str(variable))] = {"source": "inferred", "offset": float(_value), "unit": _unit}
        return _value

    def _get_automatic_condition_effect(self, condition: str, variable: str, _raw: bool = False) -> float:
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
        # v1.2.9.1: "_" separates words in a label ("Loss_Frame" reads as "loss frame"); as a regex
        # word character it hid every keyword in snake_case labels from the inferred effects.
        condition_lower = str(condition).lower().strip().replace("_", " ")
        variable_lower = str(variable).lower().strip()

        # Build study context string from all conditions + title for relational parsing
        _all_conds_text = " ".join(c.lower() for c in self.conditions) if self.conditions else ""
        _study_text = (self.study_title or "").lower() + " " + (self.study_description or "").lower()
        # v1.0.5.1: Include condition descriptions in full context for better semantic matching
        _cond_desc_text = ""
        if self.condition_descriptions:
            _cond_desc_text = " ".join(str(d).lower() for d in self.condition_descriptions.values() if d)
        # v1.2.5.3: Include DV description for richer semantic context
        _dv_desc = self._dv_descriptions.get(variable_lower, "")
        _full_context = _all_conds_text + " " + _study_text + " " + variable_lower + " " + _cond_desc_text + " " + _dv_desc.lower()

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

            # If the condition pairs BOTH a positive AND a negative attitude
            # (e.g., "trump lover vs trump hater") it describes a MIXED/OUTGROUP
            # pairing. It must be OPPOSITE valences — two same-valence words such
            # as "supporter and fan" are INGROUP and are handled by the elif below.
            # (Previously this used a count of >=2 attitude words, which wrongly
            # flagged same-valence conditions like "pro-Trump fan" as outgroup.)
            if _has_lover and _has_hater:
                # Condition pairs a positive and a negative attitude →
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

        # Valence keyword banks — defined UNCONDITIONALLY so the factorial-design
        # parsing block further below can reference them even when STEP 1 is
        # skipped because the condition was already handled by relational parsing.
        # Without this, a condition that is BOTH relational and factorial (e.g. an
        # MGP-Trump "ingroup × high-stakes" cell) raised UnboundLocalError.
        # NOTE: 'lover', 'hater' removed to prevent false positives in political
        # identity conditions (handled in STEP 0).
        positive_keywords = [
            'friend', 'positive', 'high', 'good', 'best', 'strong',
            'success', 'win', 'gain', 'benefit', 'reward', 'pleasant',
            'like', 'love', 'favor', 'approve', 'support', 'prosocial',
            'cooperative', 'trust', 'warm', 'kind', 'helpful', 'generous',
            'optimistic', 'confident', 'empowered', 'satisfied'
        ]
        negative_keywords = [
            'enemy', 'negative', 'low', 'bad', 'worst', 'weak',
            'failure', 'lose', 'loss', 'cost', 'punish', 'unpleasant',
            'dislike', 'hate', 'oppose', 'disapprove', 'reject', 'antisocial',
            'competitive', 'distrust', 'cold', 'hostile', 'harmful', 'selfish',
            'pessimistic', 'anxious', 'threatened', 'dissatisfied'
        ]

        if not _handled_by_relational:
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
        #   3. Cap total STEP 2 effect to ±0.50 to prevent runaway stacking
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
            # v1.2.7.0: new domains 39-43
            39: {'social_psychology', 'health_psychology'},                       # emotion induction/regulation
            40: {'media_communication', 'accuracy_misinformation', 'political_psychology'},  # misinformation/truth
            41: {'social_psychology'},                                           # aggression/provocation
            42: {'behavioral_economics', 'organizational_behavior'},             # negotiation/bargaining
            43: {'behavioral_economics', 'moral_psychology', 'social_psychology'},  # charitable giving
        }

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
        # DOMAIN 39: EMOTION INDUCTION & REGULATION (v1.2.7.0)
        # Effect on affect/evaluation DVs (positive = more favorable affect).
        # Regulation: Webb, Miles & Sheeran (2012, Psych Bulletin) meta —
        # reappraisal down-regulates negative affect (d≈0.45); suppression is
        # ineffective/costly. Discrete inductions: Lerner & Keltner (2001);
        # Lerner et al. (2004). Disgust→judgment contested (Landy & Goodwin 2015)
        # → kept small.
        # =====================================================================
        if _any_word_in(['reappraisal', 'reappraise', 'cognitive reappraisal',
                         'reframing'], condition_lower):
            semantic_effect += 0.20
        elif _any_word_in(['suppress emotion', 'emotion suppression',
                           'hide your feelings', 'conceal emotion'], condition_lower):
            semantic_effect -= 0.12
        if _any_word_in(['anger induction', 'angry mood', 'sadness induction',
                         'sad film', 'grief induction', 'fear induction',
                         'anxiety induction', 'disgust induction'], condition_lower):
            semantic_effect -= 0.15
        elif _any_word_in(['happy mood induction', 'positive mood induction',
                           'amusing film', 'joy induction'], condition_lower):
            semantic_effect += 0.15

        # =====================================================================
        # DOMAIN 40: MISINFORMATION & TRUTH JUDGMENT (v1.2.7.0)
        # Effect on belief-accuracy / sharing-discernment DVs (positive = more
        # accurate). Prebunking/inoculation: Roozenbeek et al. (2022, Sci. Adv.,
        # d≈0.40). Correction/fact-check: Walter & Murphy (2018) meta (partial,
        # continued-influence). Accuracy nudges: Pennycook et al. (2021, Nature) —
        # small/contested. Illusory truth (repetition): Hassan & Barber (2021).
        # =====================================================================
        if _any_word_in(['prebunk', 'prebunking', 'inoculation', 'forewarning'],
                        condition_lower):
            semantic_effect += 0.18
        elif _any_word_in(['correction', 'debunk', 'fact-check', 'fact check',
                           'corrective'], condition_lower):
            semantic_effect += 0.12
        elif _any_word_in(['accuracy prompt', 'accuracy nudge', 'consider accuracy'],
                          condition_lower):
            semantic_effect += 0.10  # small/contested
        elif _any_word_in(['repeated claim', 'illusory truth', 'familiar claim'],
                          condition_lower):
            semantic_effect -= 0.12  # inflated believability of repeated falsehoods

        # =====================================================================
        # DOMAIN 41: AGGRESSION & PROVOCATION (v1.2.7.0)
        # Effect on aggression/hostility DVs (positive = more aggression).
        # Provocation is the strongest moderator (Bettencourt & Miller 1996 meta).
        # Violent media (Anderson et al. 2010) and weapons priming (Benjamin et
        # al. 2018) are real but small/contested → kept small.
        # =====================================================================
        if _any_word_in(['provocation', 'provoked', 'insulted', 'frustration induction',
                         'taunt'], condition_lower):
            semantic_effect += 0.22
        elif _any_word_in(['violent media', 'violent video game', 'violent film'],
                          condition_lower):
            semantic_effect += 0.10  # small/contested
        elif _any_word_in(['weapon prime', 'weapons prime', 'weapon present'],
                          condition_lower):
            semantic_effect += 0.08  # small (Benjamin 2018)
        elif _any_word_in(['de-escalation', 'cooperation prime', 'conciliatory'],
                          condition_lower):
            semantic_effect -= 0.12

        # =====================================================================
        # DOMAIN 42: NEGOTIATION & BARGAINING (v1.2.7.0)
        # Effect on negotiation-outcome DVs (positive = better outcome for the
        # focal party). First-offer anchoring (Galinsky & Mussweiler 2001 —
        # robust, large). Integrative vs distributive framing (Pruitt 1981).
        # =====================================================================
        if _any_word_in(['first offer', 'opening offer', 'high anchor offer',
                         'aggressive first offer'], condition_lower):
            semantic_effect += 0.22
        elif _any_word_in(['integrative', 'win-win', 'interest-based',
                           'value creation'], condition_lower):
            semantic_effect += 0.18
        elif _any_word_in(['distributive', 'zero-sum', 'positional',
                           'competitive negotiation'], condition_lower):
            semantic_effect -= 0.12

        # =====================================================================
        # DOMAIN 43: CHARITABLE GIVING (v1.2.7.0)
        # Effect on donation/giving DVs (positive = more giving). Matching
        # (Karlan & List 2007). Identifiable victim (Small, Loewenstein & Slovic
        # 2007). Overhead aversion (Gneezy, Keenan & Gneezy 2014). Social
        # information (Frey & Meier 2004). One dominant cue fires (elif) to avoid
        # over-stacking; the ±0.50 cap bounds any residual.
        # =====================================================================
        if _any_word_in(['matching donation', 'matched donation', 'donation match',
                         'matching gift'], condition_lower):
            semantic_effect += 0.15
        elif _any_word_in(['identifiable victim', 'named beneficiary',
                           'identified victim'], condition_lower):
            semantic_effect += 0.12
        elif _any_word_in(['others donated', 'social information about giving',
                           'most people donate'], condition_lower):
            semantic_effect += 0.12
        elif _any_word_in(['low overhead', 'no overhead', 'efficient charity',
                           'overhead covered'], condition_lower):
            semantic_effect += 0.10

        # =====================================================================
        # v1.0.4.6: DOMAIN-AWARE EFFECT STACKING GUARD
        #
        # After all STEP 2 domains have been checked, apply two safeguards:
        # 1. If total STEP 2 contribution is large AND came from domains not
        #    in self.detected_domains, attenuate by 0.5× (less likely relevant)
        # 2. Cap total STEP 2 semantic_effect to ±0.50 to prevent runaway stacking
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

        # Literature anchoring: a named paradigm with a meta-analytic estimate fixes the
        # size of the design's main contrast (relational/economic-game designs keep their
        # own calibrated scaling).
        if not _raw and not _handled_by_relational and not _is_economic_game_dv:
            _meta_hit = _match_meta_entry(_study_text + " " + _all_conds_text + " " + _cond_desc_text)
            if _meta_hit is not None:
                # v1.3.0.5: the published d is shrunk toward the replication effect (and given
                # its between-study draw) before it sizes the contrast.
                _meta_d = self._shrink_inferred_meta_effect(_meta_hit[0], _meta_hit[1], variable)
                return self._meta_anchored_effect(condition, variable, _meta_d)

        # Apply Cohen's d scaling with domain-aware multiplier
        return semantic_effect * default_d * COHENS_D_TO_NORMALIZED * _domain_d_multiplier

    # Tokens marking the reference arm of a control-vs-treatment design.
    _CONTROL_ARM_WORDS = ("control", "baseline", "placebo", "waitlist", "wait-list", "wait list",
                          "no treatment", "no intervention", "neutral", "comparison", "usual",
                          "standard", "untreated", "none")
    # DV-name tokens for constructs a beneficial treatment REDUCES.
    _NEGATIVE_DV_RE = re.compile(
        r"\b(distress|anxi|depress|stress|symptom|pain\b|burnout|prejudice|biased?\b|aggress|conflict|"
        r"exhaust|bully|turnover|lonel|fear\b|risk behavio|misinformation|false belief|cheat|dishonest|"
        r"use\b|usage|consumption|waste|smok|emission|intake|absentee|errors?\b|craving|relapse|"
        r"discrimination|stigma|hostil|rumination|worry|guilt|shame)"
    )

    # A glued label ("ControlGroup", "NoTreatment", "Control1") is read as separate words: a change from
    # lower to upper case, or between a letter and a digit, starts a new word.
    _LABEL_WORD_BREAK_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")

    def _is_control_arm(self, condition: str) -> bool:
        """Whether ``condition`` names the reference arm of the design.

        A reference-arm word has to be a whole word of the label (``_word_in``): a substring test took
        "Unusual outcome", "Standardized message" and "Uncontrolled spending" for control arms, which
        skipped their literature effect. As in ``_kw_hit`` an underscore separates words
        ("cognitive_dissonance_control"); a plural counts ("Healthy controls"), and so does a glued
        label ("ControlGroup", "NoTreatment", "Control1").
        """
        _c = _label_norm(self._LABEL_WORD_BREAK_RE.sub(" ", str(condition or "")).replace("_", " "))
        return any(_word_in(w, _c) or _word_in(w + "s", _c) for w in self._CONTROL_ARM_WORDS)

    def _stable_rng(self, *parts: str) -> random.Random:
        """A Random seeded only by this run's seed and ``parts``.

        Anything drawn from it is reproducible: the same simulation seed and the
        same (condition, variable) give the same draw, in this process and in the
        next one. ``random.Random()`` with no argument seeds from the OS, so two
        engines built with the same seed would export different numbers -- the one
        thing a seeded simulator must never do. Python's ``hash()`` is salted per
        process and is no good here either, hence the digest.
        """
        _key = "|".join(str(p) for p in parts).encode("utf-8", "replace")
        _digest = hashlib.sha256(_key).digest()[:8]
        return random.Random(int(self.seed) ^ int.from_bytes(_digest, "big"))

    #: Policy for effects the tool infers (None = the registry default: replication-adjusted
    #: with the recalled shrinkage and heterogeneity draw). Set an
    #: ``empirical_registry.EffectPolicy(mode="as_published", heterogeneity_draw=False)`` to
    #: reproduce raw published d. Never consulted for a user-specified effect.
    _INFERRED_EFFECT_POLICY: Any = None

    def _shrink_inferred_meta_effect(self, keys: Tuple[str, ...], published_d: float, variable: str) -> float:
        """Replication-adjusted size for a paradigm-anchored (inferred) effect.

        Same ``adjust_effect`` the literature fallback uses, minus the tier weighting (the anchor
        never had it, and the paradigm match is not a per-entry verification claim). The
        heterogeneity draw is seeded by (paradigm, variable) only, so every arm of one contrast
        sees the same effect and a rerun with the same seed gives the same number; there is no
        per-participant randomness here, so memoisation stays valid.
        """
        published_d = float(published_d)
        if not HAS_EMPIRICAL_REGISTRY:
            return published_d
        cache = getattr(self, "_meta_shrink_cache", None)
        if cache is None:
            cache = self._meta_shrink_cache = {}
        ck = (tuple(keys), str(variable), round(published_d, 6))
        if ck in cache:
            return cache[ck]
        taus = [float(getattr(META_ANALYTIC_DB.get(k), "heterogeneity_tau", 0.0) or 0.0) for k in keys] if HAS_KNOWLEDGE_BASE else []
        taus = [t for t in taus if t > 0]
        tau = float(sum(taus) / len(taus)) if taus else None
        pol = self._INFERRED_EFFECT_POLICY
        applied = float(_empirical_registry.adjust_effect(
            published_d, kind="meta", key="", policy=pol,
            rng=self._stable_rng("meta-anchor", "|".join(keys), str(variable)), tau=tau))
        cache[ck] = applied
        if not hasattr(self, "_inferred_effect_log"):
            self._inferred_effect_log = []
        self._inferred_effect_log.append({
            "path": "paradigm_anchor", "variable": str(variable), "key": "|".join(keys),
            "published_d": round(published_d, 4),
            "shrinkage_factor": round(float(_empirical_registry.policy_factor(pol)), 4),
            "applied_d": round(applied, 4),
            "verification": _empirical_registry.shrinkage_tier(),
        })
        self._log(f"Paradigm anchor '{'|'.join(keys)}' for '{variable}': published d={published_d:.3f}, "
                  f"applied d={applied:.3f}")
        return applied

    def _meta_anchored_effect(self, condition: str, variable: str, meta_d: float) -> float:
        """Effect for ``condition`` when the study names a paradigm with a published estimate.

        The semantic keyword machinery decides WHO is higher; the literature decides
        HOW MUCH: the largest between-condition contrast is rescaled to ``meta_d``.
        When the keywords carry no usable contrast (e.g. "Self-affirmation" vs
        "Control"), the reference arm is the zero point and every other arm moves by
        the published effect, in the direction implied by the DV (benefit raises
        positive constructs and lowers symptom-type constructs).
        """
        cache = getattr(self, "_meta_anchor_cache", None)
        if cache is None:
            cache = self._meta_anchor_cache = {}
        key = (str(variable), round(float(meta_d), 4), tuple(str(c) for c in (self.conditions or [])))
        table = cache.get(key)
        if table is None:
            unit = 2.0 * 0.109 * float(meta_d)  # same currency as explicit specs: gap = 2*0.109*d
            conds = [str(c) for c in (self.conditions or [])]
            raw = {c: self._get_automatic_condition_effect(c, variable, _raw=True) for c in conds}
            is_ctrl = {c: (self._is_control_arm(c)
                           or bool(re.search(r"\b(no|without|absent|not)\b", c.lower()))) for c in conds}
            gap = (max(raw.values()) - min(raw.values())) if raw else 0.0
            table = {c: 0.0 for c in conds}
            # keyword valence is a weak signal against a reference arm: require a larger
            # semantic contrast there before trusting it over the DV-polarity rule
            if gap >= (0.06 if any(is_ctrl.values()) else 0.03):
                ctrl = [c for c in conds if is_ctrl[c]]
                centre = float(np.mean([raw[c] for c in ctrl])) if ctrl else (max(raw.values()) + min(raw.values())) / 2.0
                table = {c: (raw[c] - centre) / gap * unit for c in conds}
            elif any(is_ctrl.values()) and not all(is_ctrl.values()):
                _dv = (str(variable).replace("_", " ") + " " + str(self._dv_descriptions.get(str(variable).lower(), ""))).lower()
                sign = -1.0 if bool(self._NEGATIVE_DV_RE.search(_dv)) else 1.0
                table = {c: (0.0 if is_ctrl[c] else sign * unit) for c in conds}
            else:
                # No reference arm: order the arms by dose words (many/few, high/low, ...).
                hi = {c: bool(re.search(r"\b(many|more|high|higher|large|strong|major|most|numerous|majority)\b", c.lower())) for c in conds}
                lo = {c: bool(re.search(r"\b(few|fewer|less|low|lower|small|weak|minor|least|minority)\b", c.lower())) for c in conds}
                if any(hi.values()) and any(lo.values()):
                    _dv = (str(variable).replace("_", " ") + " " + str(self._dv_descriptions.get(str(variable).lower(), ""))).lower()
                    sign = -1.0 if bool(self._NEGATIVE_DV_RE.search(_dv)) else 1.0
                    table = {c: sign * unit / 2.0 * (1.0 if hi[c] and not lo[c] else -1.0 if lo[c] and not hi[c] else 0.0)
                             for c in conds}
                else:
                    table = dict(raw)  # nothing orders the arms: keep the generic (unanchored) effects
            cache[key] = table
        return float(table.get(str(condition), 0.0))

    def _get_condition_trait_modifier(self, condition: str) -> Dict[str, float]:
        """Memoised per condition: the result depends only on the study and the condition
        label, and this is called once per participant per item (N x items times)."""
        cache = getattr(self, "_cond_modifier_cache", None)
        if cache is None:
            cache = {}
            self._cond_modifier_cache = cache
        key = str(condition)
        if key not in cache:
            cache[key] = self._compute_condition_trait_modifier(condition)
        return dict(cache[key])

    def _compute_condition_trait_modifier(self, condition: str) -> Dict[str, float]:
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

        # v1.2.9.1: everything below depends on the condition NAME, so it creates differences
        # between conditions. With inferred effects off, or when the user specified an effect for
        # this condition, only the study-level priming above (identical for every condition)
        # is applied.
        if not getattr(self, "auto_effects", True) or self._is_explicit_condition(condition):
            return modifiers

        # AI-related conditions affect engagement and trust
        if _kw_hit('ai', condition_lower) and not _kw_hit('no ai', condition_lower):
            modifiers['engagement'] = -0.05  # Slightly less engaged with AI
            modifiers['response_consistency'] = 0.03  # Slightly more consistent
        elif _kw_hit('no ai', condition_lower) or _kw_hit('human', condition_lower):
            modifiers['engagement'] = 0.05  # More engaged with human
            modifiers['response_consistency'] = -0.02

        # Hedonic vs utilitarian products affect response style
        if _kw_hit('hedonic', condition_lower) or _kw_hit('experiential', condition_lower):
            modifiers['extremity'] = 0.08  # More extreme responses to hedonic
            modifiers['scale_use_breadth'] = 0.05
        elif _kw_hit('utilitarian', condition_lower) or _kw_hit('functional', condition_lower):
            modifiers['extremity'] = -0.05  # More moderate for utilitarian
            modifiers['scale_use_breadth'] = -0.03

        # High/low manipulations
        if _kw_hit('high', condition_lower):
            modifiers['acquiescence'] = 0.05  # Slight positive bias
        elif _kw_hit('low', condition_lower):
            modifiers['acquiescence'] = -0.05  # Slight negative bias

        # Treatment vs control
        if _kw_hit('treatment', condition_lower):
            modifiers['attention_level'] = 0.03  # Slightly more attentive
        elif _kw_hit('control', condition_lower):
            modifiers['attention_level'] = -0.02

        # v1.0.4.2: Political identity / intergroup conditions
        # When political identity is made salient, responses become more extreme
        # and variance increases (Iyengar & Westwood, 2015)
        _political_terms = ['trump', 'biden', 'political', 'partisan', 'republican',
                            'democrat', 'liberal', 'conservative']
        _is_political = any(_kw_hit(kw, condition_lower) for kw in _political_terms)
        if _is_political:
            modifiers['extremity'] = 0.12  # More polarized responses
            modifiers['response_consistency'] = 0.08  # More consistent within-person
            # Outgroup conditions: more negative emotional valence
            _outgroup_markers = ['hater', 'opponent', 'outgroup', 'other party',
                                 'opposing', 'different', 'anti']
            if any(_kw_hit(kw, condition_lower) for kw in _outgroup_markers):
                modifiers['acquiescence'] = -0.10  # Negative bias in outgroup evaluations
                modifiers['extremity'] = 0.15  # Even more extreme for outgroup

        # v1.0.4.2: Economic game conditions — intergroup matching
        # When participants play economic games with identified partners,
        # the partner's group membership strongly affects behavior
        _econ_game = any(_kw_hit(kw, condition_lower) for kw in
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
        if any(_kw_hit(kw, condition_lower) for kw in ['fear appeal', 'health threat',
               'disease risk', 'high threat', 'severe illness']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.08
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.10
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.06
        elif any(_kw_hit(kw, condition_lower) for kw in ['low threat', 'safe', 'healthy',
                 'prevention', 'wellness']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) - 0.03
            modifiers['extremity'] = modifiers.get('extremity', 0) - 0.05

        # --- Self-efficacy conditions ---
        # Bandura (1997): High self-efficacy → more confident, consistent responding
        if any(_kw_hit(kw, condition_lower) for kw in ['high efficacy', 'empowered',
               'capable', 'confident']):
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) + 0.06
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.05
        elif any(_kw_hit(kw, condition_lower) for kw in ['low efficacy', 'helpless',
                 'incapable', 'doubtful']):
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) - 0.08
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) + 0.06

        # --- Environmental/Sustainability conditions ---
        # Campbell & Kay (2014): Environmental messages trigger identity-protective
        # cognition — high engagement, polarized extremity
        if any(_kw_hit(kw, condition_lower) for kw in ['environment', 'climate', 'sustainab',
               'green', 'carbon', 'eco-friendly']):
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.08
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.04
        elif any(_kw_hit(kw, condition_lower) for kw in ['pollut', 'wasteful', 'unsustainable',
                 'carbon intensive']):
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.10
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) - 0.06

        # --- Moral/Ethical conditions ---
        # Haidt (2001): Moral judgments are emotion-driven, produce extreme responses
        # Greene et al. (2001): Personal moral dilemmas increase emotional engagement
        if any(_kw_hit(kw, condition_lower) for kw in ['moral', 'ethical', 'immoral',
               'unethical', 'trolley', 'dilemma']):
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.12
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.06
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) + 0.05

        # --- Authority/Credibility conditions ---
        # Milgram (1963): Authority increases compliance and acquiescence
        # Hovland & Weiss (1951): Source credibility amplifies persuasion
        if any(_kw_hit(kw, condition_lower) for kw in ['expert', 'authority', 'doctor',
               'professor', 'credible source', 'scientist']):
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) + 0.08
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) + 0.04
        elif any(_kw_hit(kw, condition_lower) for kw in ['non-expert', 'layperson', 'peer',
                 'low credibility', 'unknown source']):
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) - 0.06
            modifiers['extremity'] = modifiers.get('extremity', 0) - 0.04

        # --- Scarcity/Urgency conditions ---
        # Cialdini (2001): Scarcity increases arousal and extremity of evaluations
        # Worchel et al. (1975): Scarce items rated higher, more emotionally
        if any(_kw_hit(kw, condition_lower) for kw in ['scarce', 'limited', 'exclusive',
               'last chance', 'urgent', 'deadline']):
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.10
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.05
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.04

        # --- Social presence/Observation conditions ---
        # Zajonc (1965): Social facilitation — presence amplifies dominant responses
        # Bond & Titus (1983 meta): Audience effects on performance
        if any(_kw_hit(kw, condition_lower) for kw in ['observed', 'watched', 'public',
               'social presence', 'audience', 'with others']):
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) + 0.10
            modifiers['extremity'] = modifiers.get('extremity', 0) - 0.05
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) + 0.04
        elif any(_kw_hit(kw, condition_lower) for kw in ['anonymous', 'private', 'alone',
                 'unobserved', 'confidential']):
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) - 0.08
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.04

        # --- Loss/Gain framing conditions ---
        # Tversky & Kahneman (1981): Loss frame increases attention, risk-seeking
        # Levin et al. (2002): Framing effects on risk perception
        if any(_kw_hit(kw, condition_lower) for kw in ['loss frame', 'lose', 'forfeit',
               'penalty', 'risk of losing']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.06
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.08
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.04
        elif any(_kw_hit(kw, condition_lower) for kw in ['gain frame', 'earn', 'save',
                 'benefit', 'reward']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.03
            modifiers['extremity'] = modifiers.get('extremity', 0) - 0.03

        # --- Emotional induction conditions ---
        # Lerner & Keltner (2001): Anger → risk-seeking, certainty appraisals
        # Schwarz & Clore (1983): Mood-as-information
        if any(_kw_hit(kw, condition_lower) for kw in ['anger', 'angry', 'outrage',
               'frustrated', 'hostile']):
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.14
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) - 0.08
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.06
        elif any(_kw_hit(kw, condition_lower) for kw in ['sad', 'sadness', 'melanchol',
                 'grief', 'lonely']):
            modifiers['extremity'] = modifiers.get('extremity', 0) - 0.06
            modifiers['engagement'] = modifiers.get('engagement', 0) - 0.05
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) - 0.04
        elif any(_kw_hit(kw, condition_lower) for kw in ['happy', 'joy', 'elated',
                 'positive mood', 'cheerful']):
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) + 0.06
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.04

        # --- Cognitive load conditions ---
        # Sweller (1988): Cognitive load reduces processing depth → satisficing
        # Gilbert et al. (1988): Load increases reliance on heuristics
        if any(_kw_hit(kw, condition_lower) for kw in ['cognitive load', 'high load',
               'dual task', 'multitask', 'distract']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) - 0.10
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) - 0.08
            modifiers['extremity'] = modifiers.get('extremity', 0) - 0.05
        elif any(_kw_hit(kw, condition_lower) for kw in ['no load', 'low load', 'focused',
                 'undistracted']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.04
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) + 0.03

        # --- Time pressure conditions ---
        # Dror et al. (1999): Time pressure reduces accuracy, increases satisficing
        # Maule & Edland (1997): Deadline stress → more extreme, less careful
        if any(_kw_hit(kw, condition_lower) for kw in ['time pressure', 'deadline',
               'hurry', 'limited time', 'timed']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) - 0.08
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.06
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) - 0.06

        # --- Gender/Stereotype conditions ---
        # Steele & Aronson (1995): Stereotype threat increases anxiety, reduces performance
        # Schmader et al. (2008): Working memory interference under threat
        if any(_kw_hit(kw, condition_lower) for kw in ['stereotype threat', 'gender salient',
               'race salient', 'diagnostic test']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) - 0.06
            modifiers['extremity'] = modifiers.get('extremity', 0) - 0.04
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.04

        # --- Nostalgia/Memory conditions ---
        # Wildschut et al. (2006): Nostalgia increases positive affect, social connectedness
        # Mitchell et al. (1997): Rosy retrospection inflates positive recall
        if any(_kw_hit(kw, condition_lower) for kw in ['nostalgia', 'remember', 'childhood',
               'past experience', 'memory']):
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) + 0.06
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.05

        # ================================================================
        # v1.0.4.4: Additional domain-condition interaction patterns
        # ================================================================

        # --- Power/Hierarchy conditions ---
        # Keltner et al. (2003): Power increases approach, reduces inhibition
        # Galinsky et al. (2003): Power priming increases risk-taking
        if any(_kw_hit(kw, condition_lower) for kw in ['high power', 'power prime', 'boss',
               'leader role', 'in charge', 'manager']):
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.10
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) - 0.06
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) - 0.05
        elif any(_kw_hit(kw, condition_lower) for kw in ['low power', 'subordinate',
                 'employee role', 'follower', 'powerless']):
            modifiers['extremity'] = modifiers.get('extremity', 0) - 0.08
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) + 0.06
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) + 0.08

        # --- Competition conditions ---
        # Deutsch (1949): Competition decreases cooperation, increases defensiveness
        if any(_kw_hit(kw, condition_lower) for kw in ['competi', 'rival', 'contest',
               'tournament', 'winner', 'ranking']):
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.08
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.06

        # --- Mindfulness/Reflection conditions ---
        # Brown & Ryan (2003): Mindfulness reduces reactivity, increases presence
        if any(_kw_hit(kw, condition_lower) for kw in ['mindful', 'meditation', 'reflective',
               'contemplat', 'breathing exercise']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.08
            modifiers['extremity'] = modifiers.get('extremity', 0) - 0.08
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) + 0.05

        # --- Accountability conditions ---
        # Lerner & Tetlock (1999): Accountability increases accuracy motivation
        if any(_kw_hit(kw, condition_lower) for kw in ['accountable', 'justify decision',
               'explain to', 'audience', 'evaluated by']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.06
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) + 0.05
            modifiers['extremity'] = modifiers.get('extremity', 0) - 0.06

        # --- Goal-setting conditions ---
        # Locke & Latham (2002): Specific difficult goals increase effort
        if any(_kw_hit(kw, condition_lower) for kw in ['specific goal', 'challenging goal',
               'performance target', 'achievement goal']):
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.08
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) + 0.05

        # --- Depletion/Fatigue conditions ---
        # Baumeister et al. (1998): Ego depletion reduces self-regulation
        # (Though replication debates exist, fatigue effects are robust)
        if any(_kw_hit(kw, condition_lower) for kw in ['depleted', 'fatigued', 'exhausted',
               'ego depletion', 'self-control depletion']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) - 0.10
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) - 0.06
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.05

        # --- Mortality salience conditions ---
        # Greenberg et al. (1990): Terror Management Theory
        # Mortality reminders increase worldview defense, self-esteem striving
        if any(_kw_hit(kw, condition_lower) for kw in ['mortality salien', 'death remind',
               'think about death', 'mortality', 'funeral']):
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.12
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.06
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) + 0.05

        # --- v1.0.4.9: Narrative transportation conditions ---
        # Green & Brock (2000): Transportation reduces counterarguing
        if any(_kw_hit(kw, condition_lower) for kw in ['narrative', 'story', 'transported',
               'immersed', 'fictional scenario']):
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.08
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.05
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) + 0.04

        # --- v1.0.4.9: Social comparison conditions ---
        # Festinger (1954): Social comparison affects self-evaluation
        if any(_kw_hit(kw, condition_lower) for kw in ['upward comparison', 'better than',
               'outperformed', 'social comparison']):
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.08
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) + 0.06
        elif any(_kw_hit(kw, condition_lower) for kw in ['downward comparison', 'worse than',
                 'outperforming']):
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.04

        # --- v1.0.4.9: Gratitude/positive intervention conditions ---
        # Emmons & McCullough (2003): Gratitude increases positive affect
        if any(_kw_hit(kw, condition_lower) for kw in ['gratitude', 'thankful', 'count blessings',
               'three good things', 'best possible self']):
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) + 0.05
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.04

        # --- v1.0.4.9: Moral threat/cleansing conditions ---
        # Sachdeva et al. (2009): Moral self-regulation
        if any(_kw_hit(kw, condition_lower) for kw in ['moral threat', 'guilt', 'transgression',
               'sacred value', 'taboo']):
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.10
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) + 0.08
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.06

        # --- v1.0.4.9: Digital distraction conditions ---
        # Ward et al. (2017): Phone presence reduces cognitive capacity
        if any(_kw_hit(kw, condition_lower) for kw in ['phone present', 'notification',
               'multitask', 'distract', 'interrupted']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) - 0.08
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) - 0.05
        elif any(_kw_hit(kw, condition_lower) for kw in ['no phone', 'focus mode',
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
        if any(_kw_hit(kw, condition_lower) for kw in ['nostalgia induct', 'nostalgic',
               'recall a fond memory', 'sentimental', 'good old days']):
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.06
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.05
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) + 0.04

        # ── 2. Self-Affirmation (Steele, 1988; Cohen & Sherman, 2014) ──
        # Self-affirmation reduces defensiveness and identity threat, leading to
        # more open, less socially desirable responding with greater consistency.
        if any(_kw_hit(kw, condition_lower) for kw in ['self-affirm', 'self affirm',
               'values affirmation', 'affirmed', 'wrote about values',
               'personal strengths']):
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) - 0.06
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.04
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) + 0.04

        # ── 3. Mindfulness / Present-Moment Focus (Brown & Ryan, 2003; Arch & Craske, 2006) ──
        # Mindfulness increases attention and deliberate responding while reducing
        # reactive extremity. Enhances consistency through careful item processing.
        if any(_kw_hit(kw, condition_lower) for kw in ['present-moment', 'present moment',
               'body scan', 'mindful attention', 'focused awareness',
               'mindfulness induction']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.10
            modifiers['extremity'] = modifiers.get('extremity', 0) - 0.06
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) + 0.06

        # ── 4. Gratitude Induction (Emmons & McCullough, 2003; Wood et al., 2010) ──
        # Gratitude elevates positive mood, increasing acquiescence and engagement.
        # Also produces slightly more extreme positive evaluations.
        if any(_kw_hit(kw, condition_lower) for kw in ['gratitude induct', 'gratitude journal',
               'grateful', 'appreciation', 'counting blessings',
               'grateful reflection']):
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) + 0.06
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.05
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.04

        # ── 5. Power Priming — High Power (Galinsky et al., 2003; Anderson & Berdahl, 2002) ──
        # High power increases approach motivation, risk-taking, and action orientation.
        # Reduces social desirability concerns and boosts engagement.
        if any(_kw_hit(kw, condition_lower) for kw in ['power priming', 'high status',
               'recall a time you had power', 'dominant role', 'authority role',
               'elevated status']):
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.10
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) - 0.06
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.04

        # ── 6. Power Priming — Low Power (Keltner et al., 2003; Anderson & Galinsky, 2006) ──
        # Low power increases inhibition, conformity, and social monitoring.
        # Reduces extremity and increases social desirability and vigilant attention.
        if any(_kw_hit(kw, condition_lower) for kw in ['low status', 'subordinate role',
               'recall a time someone had power over', 'submissive', 'deferential',
               'disempowered']):
            modifiers['extremity'] = modifiers.get('extremity', 0) - 0.06
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) + 0.08
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.04

        # ── 7. Cognitive Load — Dual Task (Sweller, 1988; Gilbert et al., 1988) ──
        # Heavy cognitive load impairs processing capacity, reducing attention and
        # consistency. Paradoxically increases extremity through reliance on heuristics.
        if any(_kw_hit(kw, condition_lower) for kw in ['dual task', 'memorize number',
               'concurrent task', 'working memory load', 'remember digits',
               'count backwards']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) - 0.12
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) - 0.10
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.04

        # ── 8. Mortality Salience (Greenberg et al., 1990; Burke et al., 2010 meta) ──
        # Terror Management Theory: death awareness triggers worldview defense,
        # producing more extreme, engaged, and consistent value-congruent responding.
        if any(_kw_hit(kw, condition_lower) for kw in ['death prime', 'mortality prime',
               'write about own death', 'life is short', 'impermanence',
               'end of life']):
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.14
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.08
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) + 0.06

        # ── 9. Sleep Deprivation / Fatigue (Lim & Dinges, 2010; Killgore, 2010) ──
        # Sleep deprivation impairs executive function, reducing sustained attention
        # and response consistency. Increases extremity via reduced inhibition.
        if any(_kw_hit(kw, condition_lower) for kw in ['sleep depriv', 'sleep restrict',
               'fatigued participant', 'tired', 'insufficient sleep',
               'sleep loss', 'no sleep']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) - 0.12
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) - 0.08
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.06

        # ── 10. Nature Exposure / Green Space (Kaplan, 1995; Berman et al., 2008) ──
        # Attention Restoration Theory: exposure to natural environments restores
        # directed attention, reduces mental fatigue, and promotes calmer responding.
        if any(_kw_hit(kw, condition_lower) for kw in ['nature exposure', 'nature walk',
               'green space', 'outdoor', 'park scene', 'forest',
               'natural environment', 'nature image']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.06
            modifiers['extremity'] = modifiers.get('extremity', 0) - 0.04
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.04

        # ── 11. Social Exclusion / Ostracism (Williams, 2007; Baumeister et al., 2005) ──
        # Ostracism threatens fundamental needs (belonging, self-esteem, control, meaning).
        # Produces more extreme responses, higher engagement, but reduced acquiescence
        # as excluded individuals resist conforming to group norms.
        if any(_kw_hit(kw, condition_lower) for kw in ['social exclusion', 'ostracism',
               'ostracized', 'excluded', 'cyberball exclusion', 'rejected',
               'left out', 'ignored by group']):
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.10
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.06
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) - 0.08

        # ── 12. Warmth / Cold Priming (Williams & Bargh, 2008; IJzerman & Semin, 2009) ──
        # Physical warmth primes social warmth — increased acquiescence and engagement.
        # Physical cold primes social coldness — decreased acquiescence and engagement.
        if any(_kw_hit(kw, condition_lower) for kw in ['warm cup', 'warm drink', 'warm prime',
               'physical warmth', 'warm condition', 'heated room',
               'warm temperature']):
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) + 0.06
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.04
        elif any(_kw_hit(kw, condition_lower) for kw in ['cold cup', 'cold drink', 'cold prime',
                 'physical cold', 'cold condition', 'cold temperature',
                 'ice']):
            modifiers['acquiescence'] = modifiers.get('acquiescence', 0) - 0.06
            modifiers['engagement'] = modifiers.get('engagement', 0) - 0.04

        # ── 13. Scarcity Priming (Shah et al., 2012; Mullainathan & Shafir, 2013) ──
        # Scarcity captures attention (tunneling effect), increases engagement,
        # and produces more extreme evaluations of scarce resources.
        if any(_kw_hit(kw, condition_lower) for kw in ['scarcity prime', 'resource scarce',
               'financial scarcity', 'scarcity mindset', 'not enough',
               'running out', 'shortage']):
            modifiers['attention_level'] = modifiers.get('attention_level', 0) + 0.08
            modifiers['extremity'] = modifiers.get('extremity', 0) + 0.06
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.04

        # ── 14. Autonomy Support (Deci & Ryan, 2000; Ryan & Deci, 2017) ──
        # Self-Determination Theory: autonomy support satisfies the need for autonomy,
        # increasing intrinsic motivation, engagement, and consistent responding
        # while reducing impression management.
        if any(_kw_hit(kw, condition_lower) for kw in ['autonomy support', 'autonomous',
               'free choice', 'self-determined', 'your decision',
               'choose freely', 'volitional']):
            modifiers['engagement'] = modifiers.get('engagement', 0) + 0.08
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) + 0.06
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) - 0.04

        # ── 15. Autonomy Thwarting / Controlling (Deci & Ryan, 2000; Vansteenkiste & Ryan, 2013) ──
        # Controlling contexts undermine intrinsic motivation, reducing engagement
        # and consistency while increasing social desirability (conformity pressure).
        if any(_kw_hit(kw, condition_lower) for kw in ['autonomy thwart', 'controlling',
               'forced choice', 'no choice', 'mandated', 'required to',
               'must comply', 'coerced']):
            modifiers['engagement'] = modifiers.get('engagement', 0) - 0.06
            modifiers['response_consistency'] = modifiers.get('response_consistency', 0) - 0.04
            modifiers['social_desirability'] = modifiers.get('social_desirability', 0) + 0.06

        return modifiers

    def _survey_wording_for(self, variable_name: str) -> Tuple[str, str]:
        """(question_text, item_text) for a generated column, or two empty strings.

        Columns of an uploaded survey are often bare identifiers -- `Q17`, `DV_3` --
        and a content matcher given only the identifier can never recognise the
        construct, which is precisely the case this fallback exists for. The
        wording is already on the scale the column came from, so index it once and
        strip the trailing item number to get back to the scale.
        """
        _idx = getattr(self, "_wording_index", None)
        if _idx is None:
            _idx = {}
            for _sc in (getattr(self, "scales", None) or []):
                try:
                    _q = str(_sc.get("question_text", "") or "")
                    _d = str(_sc.get("dv_description", "") or "")
                    _keys = [_sc.get("variable_name"), _sc.get("name")]
                    _keys.extend(_sc.get("item_names", []) or [])
                    for _k in _keys:
                        _k = str(_k or "").strip().lower()
                        if _k and _k not in _idx:
                            _idx[_k] = (_q, _d)
                except Exception:
                    continue
            self._wording_index = _idx
        _v = str(variable_name or "").strip().lower()
        if _v in _idx:
            return _idx[_v]
        _stem_name = re.sub(r"[_\-\s]*\d+$", "", _v)
        return _idx.get(_stem_name, ("", ""))

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
        # v1.3.0.6: "sent" is a whole word ("Amount_Sent"), not the middle of "consent"/"presentation".
        _var_tokens = set(re.split(r"[^a-z0-9]+", var_lower))
        _is_econ_game = any(kw in var_lower for kw in _econ_kws if kw != 'sent') or (
            'sent' in _var_tokens) or any(
            kw in condition_lower for kw in ['dictator', 'trust game', 'ultimatum',
                                              'public good', 'prisoner'])
        # v1.3.0.6: games the allocation keywords never matched (stag hunt, common pool, beauty
        # contest, auctions, centipede, ...). A game must be NAMED as a whole phrase (variable name,
        # title, description or condition label); a variable named for the game is its outcome, any
        # other variable counts only when it reads like a decision (choice, bid, guess, harvest ...),
        # so a Likert "Trust_in_Government" in a study that mentions a trust game stays a scale.
        _kb_key_resolved = None
        if HAS_KNOWLEDGE_BASE:
            try:
                _kb_key_resolved = resolve_game_calibration_key(
                    variable_name, self.study_title or "", self.study_description or "", condition_lower)
                if _kb_key_resolved and not _is_econ_game:
                    _gsrc = detect_game_type(variable_name, self.study_title or "",
                                             self.study_description or "", condition_lower)[1]
                    _is_econ_game = _gsrc == "variable" or looks_like_game_decision(variable_name)
            except Exception as _gerr:  # never let game resolution take a run down
                self._log(f"Game resolution skipped: {_gerr}")
                _kb_key_resolved = None
        if not _is_econ_game and 'prisoner' in (
                (self.study_title or "") + " " + (self.study_description or "")).lower():
            # v1.3.0.5: a prisoner's-dilemma DV is usually named "cooperate"/"defect",
            # which no allocation keyword matches ("cooperat" alone is too broad: it
            # also names Likert cooperation scales). Require the game in the study text.
            _is_econ_game = any(kw in var_lower for kw in ('cooperat', 'defect'))
        if _is_econ_game:
            # Detect specific game type for precise calibration
            _full_ctx = var_lower + " " + condition_lower + " " + (
                self.study_title or "").lower() + " " + (self.study_description or "").lower()

            # v1.0.8.7: Try structured knowledge base FIRST for game calibrations
            if HAS_KNOWLEDGE_BASE:
                _kb_game = GAME_CALIBRATIONS.get(_kb_key_resolved) if _kb_key_resolved else None
                # v1.3.0.5: the loop keys are underscored, but study text says
                # "public goods game" / "prisoner's dilemma", so those two games never
                # matched and fell through to the generic branch (a one-shot public
                # goods game came out at a 64% mean contribution, a prisoner's dilemma at
                # 70% cooperation, against 40% and 47% in the knowledge base).
                _gt_aliases = {
                    'public_good': ('public good', 'public-good', 'voluntary contribution'),
                    'prisoner': ("prisoner's dilemma", 'prisoners dilemma', 'prisoners\' dilemma',
                                 'prisoner dilemma'),
                }
                # v1.3.0.6: the resolver above names the game by whole phrase; this substring loop
                # remains the fallback for study text it does not recognise (e.g. "trust" alone).
                for _gt in ([] if _kb_game else
                            ['dictator', 'trust', 'ultimatum', 'public_good',
                             'prisoner', 'auction', 'bargain', 'gift_exchange',
                             'stag_hunt', 'common_pool', 'holt_laury',
                             'beauty_contest', 'die_roll', 'bribery']):
                    if _gt in _full_ctx or any(a in _full_ctx for a in _gt_aliases.get(_gt, ())):
                        _variant = 'standard'
                        if _gt == 'dictator' and any(kw in _full_ctx for kw in ['tak', 'steal', 'negative']):
                            _variant = 'taking'
                        elif _gt == 'dictator' and any(kw in _full_ctx for kw in ['third party', 'punishment']):
                            _variant = 'third_party_punishment'
                        elif _gt == 'public_good' and 'punish' in _full_ctx:
                            _variant = 'punishment'
                        _gt_clean = _gt.replace('_good', '_goods')
                        if _gt == 'prisoner':
                            _gt_clean = 'prisoners_dilemma'
                        _kb_game = get_game_calibration(_gt_clean, _variant)
                        if _kb_game is None:
                            _kb_game = get_game_calibration(_gt, _variant)
                        break
                if _kb_game:
                    # Use structured calibration: convert mean_proportion to adjustment
                    # mean_proportion is 0-1 scale, default midpoint is 0.5
                    # (a mean above 1, e.g. second-price overbidding at 1.05 of value, is a
                    # ratio rather than a proportion: clamp it so the tendency stays on the scale)
                    _kb_mean = min(0.95, max(0.05, float(_kb_game.mean_proportion)))
                    calibration['mean_adjustment'] = _kb_mean - 0.50
                    calibration['variance_adjustment'] = max(0.08, _kb_game.sd_proportion * 0.8)
                    calibration['positivity_bias'] = -0.05 if _kb_game.mean_proportion < 0.40 else 0.0
                    calibration['_game_variant'] = f"{_kb_game.game_type}_{_kb_game.variant}"
                    calibration['_kb_source'] = _kb_game.source
                    # Full empirical distribution (shape, subpopulation shares) so the
                    # generator can reproduce the real outcome distribution, not just
                    # shift a tendency toward the published mean.
                    calibration['_kb_dist'] = {
                        'game': _kb_game.game_type, 'variant': _kb_game.variant,
                        'mean': _kb_game.mean_proportion, 'sd': _kb_game.sd_proportion,
                        'subpops': dict(_kb_game.subpopulations or {}),
                    }
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
                'npi': 'narcissism_npi',
                # v1.2.7.0: wire the dormant Dark Triad norms (SD3; Jones & Paulhus 2014)
                # so Machiavellianism/psychopathy DVs get their (right-skewed, low-mean)
                # baseline instead of a generic midpoint.
                'machiavellian': 'dark_triad_machiavellianism', 'machiavell': 'dark_triad_machiavellianism',
                'psychopath': 'dark_triad_psychopathy', 'dark_triad': 'dark_triad_machiavellianism',
                'dark triad': 'dark_triad_machiavellianism', 'sd3': 'dark_triad_machiavellianism',
                'attachment_anx': 'attachment_anxiety_ecr',
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

        # ===== CONDITION-BASED ADJUSTMENTS: intentionally NONE =====
        # Condition effects belong exclusively to the effect pipeline
        # (_get_effect_for_condition: user-specified d, or the literature-grounded
        # automatic rules). This calibration used to add +/-0.03 (and -0.02 for any
        # condition containing the substring 'ai') from bare substring checks on the
        # condition name, a second, uncontrolled condition effect: d=0 between
        # "High" and "Low" conditions still showed d ~0.24, a configured d got an
        # unrequested boost, and names like "Paid"/"Fair"/"Maintain" matched 'ai'.

        # v1.2.9.9: LAST RESORT, and it has to be last.
        #
        # The substring map near the top reaches about 40 of the 201 published
        # norms, and only when the variable name happens to contain the mapped
        # fragment. An uploaded survey calling its items `PSS4_1`, `stress_total`,
        # `bfi_extra_1` or `panas_pos_4` matched nothing, so the published norm went
        # unused. Content-based matching over the whole table closes that gap:
        # instrument acronyms are matched exactly, otherwise by IDF-weighted token
        # overlap, and an ambiguous match is treated as no match so a wrong norm is
        # never applied.
        #
        # It runs AFTER every keyword branch, not before them. Placed earlier it
        # returned first and REPLACED a richer keyword calibration with a thinner
        # one — on an 8-item attitude scale that dropped the variance the attitude
        # branch would have added and pushed a configured d of 0.50 up to 0.68,
        # because d is gap over SD and the SD had been quietly narrowed. A
        # last-resort rule that pre-empts the rules it is a fallback for is not a
        # fallback. It now fires only where nothing else did.
        _already_calibrated = (
            calibration.get('_kb_source')
            or calibration['mean_adjustment'] != 0.0
            or calibration['variance_adjustment'] != 0.0
            or calibration['positivity_bias'] != 0.0
        )
        if HAS_CONSTRUCT_MATCHER and not _already_calibrated:
            try:
                _wording = self._survey_wording_for(variable_name)
                _m = _construct_matcher.match(
                    variable_name=variable_name,
                    question_text=_wording[0],
                    item_text=_wording[1],
                )
            except Exception:
                _m = None
            if _m is not None:
                _norm = get_construct_norm(_m.key, target_scale_points=7)
                if _norm:
                    _dev = (_norm['mean'] - 4.0) / 3.0
                    calibration['mean_adjustment'] = _dev * 0.15
                    calibration['positivity_bias'] = max(-0.10, min(0.12, _dev * 0.10))
                    if _norm.get('skewness', 0) > 0.3:
                        calibration['variance_adjustment'] += 0.06
                    elif _norm.get('skewness', 0) < -0.3:
                        calibration['variance_adjustment'] -= 0.02
                    calibration['_kb_source'] = (
                        f"ConstructNorm: {_m.key} (matched on "
                        f"{'acronym' if _m.via_acronym else 'content'})"
                    )

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
        # v1.2.9.1: a game word in a condition LABEL ("Dictator game", "Trust game") used to pick a
        # per-label game baseline, so labels alone moved the mean by ~1.9 d even with inferred effects
        # off or an explicit d. The label may inform the baseline only while inferred effects are on
        # and this variable has no user-specified effect.
        _name_shapes_response = self._condition_name_may_shape(variable_name)
        domain_calibration = self._get_domain_response_calibration(
            variable_name, condition if _name_shapes_response else "")

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

        # v1.2.6.5: Per-scale diversity noise (Barrie & Cerina 2026 correction)
        # A single response_tendency driving all scales makes attitudes
        # co-move too tightly (over-constraint). Real people evaluate different
        # constructs partially independently. Add per-scale noise drawn from
        # a distribution seeded on (participant, variable) so it's reproducible
        # but varies across scales for the same person.
        _scale_noise_seed = _stable_int_hash(f"{participant_seed}_{variable_name}") % (2**31)
        _scale_noise_rng = np.random.RandomState(_scale_noise_seed)
        # v1.2.6.6: Construct-specific tendency replacement. Instead of adding
        # noise to a shared tendency, generate a PARTIALLY INDEPENDENT base
        # tendency for each construct. The shared component (from persona) is
        # weighted against an independent per-construct draw. Weight chosen so
        # between-scale r lands in the realistic 0.10-0.25 range per Barrie &
        # Cerina (2026) and real GSS data (PVE1 ≈ 0.07-0.10).
        #
        # Model: tendency_for_scale = w*shared + (1-w)*independent
        # where w controls coupling. At w=1.0 all scales move together (old
        # behavior). At w=0.0 they are fully independent. Target: w ≈ 0.35
        # to match real human data where demographics/traits explain ~15-25%
        # of cross-scale variance.
        _SHARED_WEIGHT = _SHARED_TENDENCY_WEIGHT
        _independent_tendency = _scale_noise_rng.normal(0.58, _INDEP_TENDENCY_SD)
        _independent_tendency = float(np.clip(_independent_tendency, 0.10, 0.90))
        _secondary_z = traits.get('_secondary_diversity_z', 0.0)
        _independent_tendency += _secondary_z * 0.06
        base_tendency = _SHARED_WEIGHT * base_tendency + (1.0 - _SHARED_WEIGHT) * _independent_tendency

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
        # v1.2.9.1: a user effect on a scale that is generated next to other scales is built into the
        # finished item responses (_apply_user_effect_to_scale); the generator sees no condition shift.
        # The lookup above still runs so the metadata records the intended effect.
        if condition_effect != 0.0 and variable_name in getattr(self, "_deferred_effect_vars", ()):
            condition_effect = 0.0

        # v1.2.9.9: a condition effect is specified in Cohen's d — a GAP DIVIDED BY
        # AN SD — so it has to travel with whatever SD this variable ends up with.
        # The domain calibration below widens or narrows the within-person SD by
        # `variance_adjustment` (an intention scale gets +0.05, a moral-identity
        # scale -0.02), and the effect was being added as a fixed shift regardless.
        # The recovered d therefore moved whenever the calibration did, in the
        # opposite direction and for no substantive reason: the same configured
        # d = 0.50 came back as 0.50 or 0.68 depending only on which keyword branch
        # the variable's NAME happened to hit. Scaling the shift by the same factor
        # makes the recovered effect invariant to the calibration, which is what
        # "configured d" has to mean if it means anything.
        _var_adj = float(domain_calibration.get('variance_adjustment', 0.0) or 0.0)
        if _var_adj:
            condition_effect *= (1.0 + _var_adj)

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
            # The persona factors above are NOT mean-1 in the simulated population
            # (measured mean ~1.12 with default persona mix: processing depth alone
            # averages ~1.1). Dividing by the population mean keeps the heterogeneity
            # (SD ~0.13) but stops it from inflating the average effect.
            _interaction_multiplier /= _INTERACTION_MULTIPLIER_POP_MEAN

            condition_effect *= _interaction_multiplier

        # =====================================================================
        # STEP 4-GAME: Economic-game DVs are drawn from the published outcome
        # distribution (zero spike, 50/50 spike, bands; Engel 2011 etc.) instead
        # of a Likert-style normal around a shifted tendency. A person-level latent
        # (prosociality traits + noise) fixes WHERE in that distribution this
        # participant sits; the condition effect shifts that latent, so group
        # differences and discrimination effects still operate. Gated to
        # allocation-sized, unipolar scales with a recognised KB game.
        # =====================================================================
        _kb_dist = domain_calibration.get('_kb_dist')
        _bin_rate = _binary_game_rate(_kb_dist) if scale_range == 1 else None
        if (_bin_rate is not None and not is_reverse and not _scale_geom['is_bipolar']
                and domain_calibration.get('_game_variant') not in ('dictator_taking', 'dictator_third_party')):
            # v1.3.0.6: a two-option game outcome is drawn at the published choice rate (prisoner's
            # dilemma 47%, Sally 1995; Dal Bo & Frechette 2018), not at the Likert tendency (~58%).
            # The same person latent as the allocation route decides who chooses "1"; the condition
            # effect moves that latent on the probit scale, sized so a requested Cohen's d on the 0/1
            # column is recovered: gap_z = d * sqrt(p(1-p)) / phi(Phi^-1(p)).
            from statistics import NormalDist
            _nd = NormalDist()
            _grng = np.random.RandomState((participant_seed * 7919 + 13) % (2**31))
            _coop = _safe_trait_value(modified_traits.get("cooperation_tendency"), 0.5)
            _emp = _safe_trait_value(modified_traits.get("empathy"), 0.5)
            _zt = float(np.clip(((_coop - 0.5) + (_emp - 0.5)) / 2.0 / 0.2, -2.5, 2.5))
            _w = 0.35
            _z = _w * _zt + float(np.sqrt(1.0 - _w * _w)) * float(_grng.normal())
            _zp = _nd.inv_cdf(_bin_rate)
            _kappa = float(np.sqrt(_bin_rate * (1.0 - _bin_rate)) / _nd.pdf(_zp))
            _unit = self._EFFECT_D_TO_NORMALIZED * self._explicit_effect_scale(variable_name)
            # (undo the variance widening applied to every effect above: d is a gap over an SD, and
            # here the SD is the Bernoulli SD, which the kappa term already accounts for)
            _va = 1.0 + float(domain_calibration.get('variance_adjustment', 0.0) or 0.0)
            _side_d = condition_effect / (_va * 2.0 * _unit) if _unit > 0 else 0.0   # +/- d/2 per arm
            return int(scale_max if (_z + _zp + _side_d * _kappa) > 0.0 else scale_min)

        if (_kb_dist and not is_reverse and scale_range >= 10 and not _scale_geom['is_bipolar']
                and domain_calibration.get('_game_variant') not in ('dictator_taking', 'dictator_third_party')):
            _qfn = _game_quantile_fn(_kb_dist)
            if _qfn is not None:
                import math
                _grng = np.random.RandomState((participant_seed * 7919 + 13) % (2**31))
                _coop = _safe_trait_value(modified_traits.get("cooperation_tendency"), 0.5)
                _emp = _safe_trait_value(modified_traits.get("empathy"), 0.5)
                _zt = float(np.clip(((_coop - 0.5) + (_emp - 0.5)) / 2.0 / 0.2, -2.5, 2.5))
                _w = 0.35
                _z = _w * _zt + math.sqrt(1.0 - _w * _w) * float(_grng.normal())
                _z_shift = condition_effect / (self._explicit_effect_scale(variable_name) * 0.25)
                _q = 0.5 * (1.0 + math.erf((_z + _z_shift * _GAME_Z_GAIN) / math.sqrt(2.0)))
                _val = scale_min + _qfn(_q) * scale_range
                return int(max(scale_min, min(scale_max, int(round(_val)))))

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
            _latent_weight = float(np.clip(_latent_weight, 0.08, 0.22)) * _LATENT_WEIGHT_MULT
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
            _g_strength = traits.get("_g_factor_strength", 0.12) * _G_FACTOR_MULT
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

            # v1.2.6.5: Add per-scale jitter to g-factor loading to prevent
            # all constructs from moving in lockstep. Real CMV varies within
            # a person across different constructs (Barrie & Cerina 2026).
            _g_jitter = rng.normal(0, _g_loading * 0.3)
            _g_effect = _g_factor_z * _g_strength * (_g_loading + _g_jitter)
            adjusted_tendency = float(np.clip(
                adjusted_tendency + _g_effect, _bound_low, _bound_high
            ))

        # =====================================================================
        # ABE 3.0 STEP 4c-ii: 3D Latent Attitude Vector
        # Applies construct-specific loadings from the 3D latent vector
        # (evaluative valence, arousal/engagement, approach-avoidance).
        # =====================================================================
        # v1.2.6.6: Attenuate latent attitude vector to prevent over-coupling.
        # The construct-independent tendency model (STEP 3) already handles
        # between-scale diversity; this vector should add subtle coherence,
        # not dominate. Loadings scaled by 0.4 to avoid stacking too many
        # coupling mechanisms (g-factor + 3D vector + shared tendency).
        _lat_valence = traits.get("_latent_valence", 0.0)
        _lat_arousal = traits.get("_latent_arousal", 0.0)
        _lat_approach = traits.get("_latent_approach", 0.0)
        _LATENT_ATTENUATION = 0.4
        if any(abs(v) > 0.001 for v in [_lat_valence, _lat_arousal, _lat_approach]):
            _v_load, _a_load, _p_load = self._get_latent_attitude_loading(variable_name)
            _latent_shift = (
                _lat_valence * _v_load
                + _lat_arousal * _a_load
                + _lat_approach * _p_load
            ) * _LATENT_ATTENUATION
            adjusted_tendency = float(np.clip(
                adjusted_tendency + _latent_shift, _bound_low, _bound_high
            ))

        # =====================================================================
        # ABE 3.0 STEP 4c-iii: Survey-Level Fatigue Drift
        # Modulates response tendency based on progress through the full survey.
        # =====================================================================
        _global_idx = getattr(self, '_global_item_counter', 0)
        _global_total = getattr(self, '_total_global_items', 1)
        _fatigue_deltas = self._compute_survey_fatigue(traits, _global_idx, _global_total)
        if _fatigue_deltas:
            adjusted_tendency += _fatigue_deltas.get("_fatigue_tendency_shift", 0.0)
            adjusted_tendency = float(np.clip(adjusted_tendency, _bound_low, _bound_high))

        # =====================================================================
        # ABE 3.0 STEP 4c-iv: Response Pattern Inertia (Anchoring)
        # Schwarz & Strack (1991): responses anchored to recent similar items.
        # =====================================================================
        _p_idx_inertia = traits.get('_participant_idx', -1)
        if isinstance(_p_idx_inertia, (int, float)) and int(_p_idx_inertia) >= 0:
            _inertia_pull = self._compute_response_inertia(
                traits, variable_name, int(_p_idx_inertia),
                adjusted_tendency, scale_min, scale_max,
            )
            adjusted_tendency = float(np.clip(
                adjusted_tendency + _inertia_pull * _INERTIA_MULT, _bound_low, _bound_high
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
                    _coherence_weight = float(np.clip(_coherence_weight, 0.02, 0.10)) * _COHERENCE_MULT
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

            # v1.2.9.1: a deferred user effect must follow the same direction (see _apply_user_effect_to_scale)
            _rok = getattr(self, "_reversal_ok_arr", None)
            if _rok is not None and _rok[0] == variable_name and _p_idx is not None:
                _rok[1][_p_idx, int(getattr(self, "_current_item_position", 1)) - 1] = _correctly_reversed

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

        # ABE 3.0: Apply fatigue variance inflation (Improvement #1)
        _fat_var_infl = _fatigue_deltas.get("_fatigue_variance_inflation", 1.0) if _fatigue_deltas else 1.0
        sd *= _fat_var_infl

        # Generate response from normal distribution
        response = float(rng.normal(center, sd))

        # =====================================================================
        # STEP 7: Apply extreme response style (Greenleaf, 1992)
        # ERS respondents use endpoints 2-3x more than modal
        # =====================================================================
        extremity = _safe_trait_value(modified_traits.get("extremity"), 0.18)
        # Apply scale-type extremity boost
        extremity += scale_calibration['extremity_boost']
        # ABE 3.0: Fatigue reduces extremity (less cognitive effort → fewer endpoints)
        extremity += _fatigue_deltas.get("_fatigue_extremity_reduction", 0.0) if _fatigue_deltas else 0.0
        extremity = float(np.clip(extremity, 0.0, 0.95))
        if rng.random() < extremity * 0.45:  # Calibrated to produce ~15-20% endpoints for ERS
            # Use proportional noise near endpoints (scales to range)
            endpoint_noise = max(0.5, scale_range * 0.02)  # 2% of range, min 0.5
            # Extreme responders snap to an endpoint only when the item is at least
            # moderately favourable/unfavourable to them (>= 15% of the range beyond
            # the midpoint). Snapping EVERY response past the midpoint turned slight
            # leaners into 7s and produced a ceiling spike taller than the 6 bin.
            _mid = (scale_min + scale_max) / 2.0
            _ers_margin = 0.15 * scale_range
            if response > _mid + _ers_margin:
                response = scale_max - float(rng.uniform(0, endpoint_noise))
            elif response < _mid - _ers_margin:
                response = scale_min + float(rng.uniform(0, endpoint_noise))

        # =====================================================================
        # STEP 8: Apply acquiescence bias (Billiet & McClendon, 2000)
        # High acquiescers: +0.5-1.0 point inflation on agreement items
        # =====================================================================
        acquiescence = _safe_trait_value(modified_traits.get("acquiescence"), 0.50)
        # ABE 3.0: Fatigue increases acquiescence (satisficing; Krosnick 1991)
        acquiescence += _fatigue_deltas.get("_fatigue_acquiescence_boost", 0.0) if _fatigue_deltas else 0.0
        acquiescence = float(np.clip(acquiescence, 0.0, 1.0))
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

            # Also check condition context for sensitivity (inferred effects only: with them off,
            # or an explicit effect on this variable, the label must not change the response style)
            _cond_lower = condition.lower() if (condition and _name_shapes_response) else ""
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

    # ==================================================================
    # ABE 3.0: CONSISTENCY IMPROVEMENT #1 — Survey-Level Fatigue Drift
    # Galesic & Bosnjak (2009); Herzog & Bachman (1981)
    # Traits evolve as the participant progresses through the FULL survey
    # (across scales, not just within a single scale).
    # ==================================================================

    def _compute_survey_fatigue(
        self,
        traits: Dict[str, float],
        global_item_index: int,
        total_global_items: int,
    ) -> Dict[str, float]:
        """Return per-trait fatigue adjustments based on survey progress.

        Returns a dict of trait deltas (add to base traits) that model:
        - Mean regression toward midpoint (response_tendency → 0.50)
        - Increased acquiescence (satisficing)
        - Decreased extremity (less cognitive effort)
        - Increased within-person variance
        """
        if total_global_items < 3:
            return {}

        progress = global_item_index / max(1, total_global_items)
        # Fatigue kicks in after ~40% of the survey, ramps up
        fatigue_strength = max(0.0, (progress - 0.40) / 0.60)  # 0 at 40%, 1 at 100%

        # Persona resistance: engaged/attentive resist fatigue
        attention = _safe_trait_value(traits.get("attention_level"), 0.7)
        engagement = _safe_trait_value(traits.get("engagement"), 0.65)
        resistance = 0.3 + attention * 0.35 + engagement * 0.35  # 0.3-1.0
        resistance = float(np.clip(resistance, 0.3, 1.0))

        # Effective fatigue after resistance
        eff_fatigue = fatigue_strength * (1.0 - resistance * 0.7)  # max ~0.79 for very low attention
        eff_fatigue = float(np.clip(eff_fatigue, 0.0, 0.60))

        deltas: Dict[str, float] = {}

        # Mean regression: response_tendency drifts toward 0.50
        current_tendency = _safe_trait_value(traits.get("response_tendency"), 0.58)
        deltas["_fatigue_tendency_shift"] = (0.50 - current_tendency) * eff_fatigue * 0.15

        # Acquiescence increases (satisficing; Krosnick 1991)
        deltas["_fatigue_acquiescence_boost"] = eff_fatigue * 0.06

        # Extremity decreases (less effort to discriminate)
        deltas["_fatigue_extremity_reduction"] = -eff_fatigue * 0.05

        # Variance inflation factor (multiply within-person SD by this)
        deltas["_fatigue_variance_inflation"] = 1.0 + eff_fatigue * 0.12

        return deltas

    # ==================================================================
    # ABE 3.0: CONSISTENCY IMPROVEMENT #2 — Demographic → Style Coupling
    # Krosnick (1991); Greenleaf (1992); Meisenberg & Williams (2008)
    # Demographics modulate response style traits with small effects.
    # ==================================================================

    def _apply_demographic_trait_modulation(
        self,
        traits: Dict[str, float],
        participant_id: int,
        demographics_df: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Modulate response style traits based on demographic variables.

        Small effects (max ±0.08) so personas remain the primary driver.
        Returns modified traits dict.
        """
        if demographics_df is None or len(demographics_df) == 0:
            return traits

        if participant_id >= len(demographics_df):
            return traits

        row = demographics_df.iloc[participant_id]
        age = float(row.get("Age", 35)) if "Age" in demographics_df.columns else 35.0
        # Encode education as 0-4 ordinal if available
        _edu_col = None
        for col in demographics_df.columns:
            if "education" in col.lower() or "edu" in col.lower():
                _edu_col = col
                break

        # Age effects (Krosnick 1991: older → more acquiescent, more midpoint)
        # Normalize age to 0-1 range (18=0, 80=1)
        age_norm = float(np.clip((age - 18) / 62.0, 0.0, 1.0))
        traits = dict(traits)  # Copy to avoid mutating original

        # Older → slightly more acquiescent (+0.04 at age 80)
        acq = _safe_trait_value(traits.get("acquiescence"), 0.50)
        traits["acquiescence"] = float(np.clip(acq + (age_norm - 0.3) * 0.06, 0.0, 1.0))

        # Older → slightly less extreme (more midpoint usage)
        ext = _safe_trait_value(traits.get("extremity"), 0.30)
        traits["extremity"] = float(np.clip(ext - (age_norm - 0.3) * 0.04, 0.0, 1.0))

        # Education effects (Meisenberg & Williams 2008)
        if _edu_col is not None:
            edu_val = row[_edu_col]
            # Try to parse education level to ordinal
            edu_ord = 2.0  # default mid
            if isinstance(edu_val, (int, float)) and not np.isnan(edu_val):
                edu_ord = float(np.clip(edu_val, 0, 4))
            elif isinstance(edu_val, str):
                edu_lower = edu_val.lower()
                if any(kw in edu_lower for kw in ["grad", "master", "phd", "doctor"]):
                    edu_ord = 4.0
                elif "bachelor" in edu_lower:
                    edu_ord = 3.0
                elif "some college" in edu_lower or "associate" in edu_lower:
                    edu_ord = 2.0
                elif "high school" in edu_lower:
                    edu_ord = 1.0
                elif "less" in edu_lower:
                    edu_ord = 0.0

            edu_norm = edu_ord / 4.0  # 0-1

            # Higher education → less acquiescence, less extreme responding
            acq2 = _safe_trait_value(traits.get("acquiescence"), 0.50)
            traits["acquiescence"] = float(np.clip(acq2 - (edu_norm - 0.5) * 0.06, 0.0, 1.0))
            ext2 = _safe_trait_value(traits.get("extremity"), 0.30)
            traits["extremity"] = float(np.clip(ext2 - (edu_norm - 0.5) * 0.04, 0.0, 1.0))
            # Higher education → more engagement
            eng = _safe_trait_value(traits.get("engagement"), 0.65)
            traits["engagement"] = float(np.clip(eng + (edu_norm - 0.5) * 0.06, 0.0, 1.0))

        return traits

    # ==================================================================
    # ABE 3.0: CONSISTENCY IMPROVEMENT #3 — 3D Latent Attitude Vector
    # Extends scalar g-factor to 3 correlated dimensions:
    #   (1) Evaluative valence (positive/negative general tendency)
    #   (2) Engagement/arousal (action orientation)
    #   (3) Approach-avoidance (risk seeking vs avoiding)
    # Correlated via Cholesky decomposition within-person.
    # ==================================================================

    def _generate_latent_attitude_vector(
        self,
        traits: Dict[str, float],
        participant_seed: int,
    ) -> Dict[str, float]:
        """Generate a 3D latent attitude vector for within-person coherence.

        Returns dict with keys: _latent_valence, _latent_arousal, _latent_approach.
        """
        rng = np.random.RandomState(participant_seed)

        # Inter-dimension correlations (within-person)
        # Valence-Arousal r≈0.35, Valence-Approach r≈0.40, Arousal-Approach r≈0.25
        corr_matrix = np.array([
            [1.00, 0.35, 0.40],
            [0.35, 1.00, 0.25],
            [0.40, 0.25, 1.00],
        ])
        try:
            L = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            L = np.eye(3)

        z = rng.standard_normal(3)
        correlated = L @ z

        # Scale by participant consistency — high consistency → stronger latent structure
        consistency = _safe_trait_value(traits.get("response_consistency"), 0.60)
        strength = 0.10 + consistency * 0.20  # Range: 0.10-0.30

        return {
            "_latent_valence": float(correlated[0]) * strength,
            "_latent_arousal": float(correlated[1]) * strength,
            "_latent_approach": float(correlated[2]) * strength,
        }

    def _get_latent_attitude_loading(
        self,
        variable_name: str,
    ) -> Tuple[float, float, float]:
        """Return (valence_loading, arousal_loading, approach_loading) for a DV.

        Construct-type-specific loadings determine how much each latent dimension
        influences responses on this particular variable.
        """
        var_lower = variable_name.lower()

        # Default moderate valence loading, low arousal/approach
        v_load, a_load, p_load = 0.15, 0.05, 0.05

        # Attitudes/evaluations: HIGH valence loading
        if any(kw in var_lower for kw in ['attitude', 'evaluation', 'opinion',
                'perception', 'judgment', 'satisf', 'happy', 'enjoy']):
            v_load, a_load, p_load = 0.30, 0.08, 0.05

        # Trust/credibility: HIGH valence + moderate approach
        elif any(kw in var_lower for kw in ['trust', 'credib', 'reliab',
                  'integrity', 'competenc']):
            v_load, a_load, p_load = 0.25, 0.05, 0.15

        # Behavioral intentions: moderate valence + HIGH arousal
        elif any(kw in var_lower for kw in ['intention', 'willing', 'plan',
                  'expect', 'intend', 'wtp', 'buy']):
            v_load, a_load, p_load = 0.15, 0.25, 0.15

        # Risk/threat: LOW valence, moderate arousal, HIGH approach-avoidance
        elif any(kw in var_lower for kw in ['risk', 'danger', 'threat',
                  'fear', 'anxiety', 'worry']):
            v_load, a_load, p_load = 0.10, 0.15, 0.30

        # Economic games: moderate all
        elif any(kw in var_lower for kw in ['dictator', 'trust_game', 'ultimatum',
                  'public_good', 'allocation', 'offer']):
            v_load, a_load, p_load = 0.15, 0.10, 0.20

        # Emotional/affect: HIGH valence + HIGH arousal
        elif any(kw in var_lower for kw in ['emotion', 'affect', 'mood',
                  'feeling', 'anger', 'joy']):
            v_load, a_load, p_load = 0.30, 0.25, 0.08

        return v_load, a_load, p_load

    # ==================================================================
    # ABE 3.0: CONSISTENCY IMPROVEMENT #4 — Response Pattern Inertia
    # Schwarz & Strack (1991); Tourangeau et al. (2000)
    # Responses to item N are anchored by the most recent similar item.
    # ==================================================================

    def _compute_response_inertia(
        self,
        traits: Dict[str, float],
        variable_name: str,
        participant_idx: int,
        current_tendency: float,
        scale_min: int,
        scale_max: int,
    ) -> float:
        """Compute anchoring pull from recent similar items.

        Returns an additive shift to apply to current_tendency (in 0-1 normalized space).
        """
        if not hasattr(self, '_response_inertia_memory'):
            return 0.0

        memory = self._response_inertia_memory.get(participant_idx, [])
        if len(memory) < 1:
            return 0.0

        consistency = _safe_trait_value(traits.get("response_consistency"), 0.60)
        # Higher consistency → stronger anchoring
        inertia_weight = 0.06 + consistency * 0.12  # Range: 0.06-0.18
        inertia_weight = float(np.clip(inertia_weight, 0.04, 0.20))

        var_lower = variable_name.lower()

        # Find most recent entry and compute similarity-weighted pull
        total_pull = 0.0
        total_weight = 0.0
        # Look at last 3 items (recency decay)
        recent = memory[-3:]
        for idx, (prev_var, prev_normalized_val) in enumerate(recent):
            recency = (idx + 1) / len(recent)  # More recent = higher weight

            # Compute semantic similarity (crude but effective)
            prev_lower = prev_var.lower()
            if prev_lower == var_lower:
                similarity = 0.9  # Same scale
            elif any(kw in prev_lower and kw in var_lower for kw in
                     var_lower.split('_') if len(kw) > 3):
                similarity = 0.5  # Shared meaningful keyword
            else:
                similarity = 0.15  # Adjacent items (position proximity)

            w = recency * similarity * inertia_weight
            total_pull += w * (prev_normalized_val - current_tendency)
            total_weight += w

        if total_weight > 0:
            return float(np.clip(total_pull, -0.08, 0.08))
        return 0.0

    # ==================================================================
    # ABE 3.0: CONSISTENCY IMPROVEMENT #5 — Post-Generation Audit & Repair
    # Validates achieved within-person consistency and repairs violations.
    # ==================================================================

    def _refresh_scale_composites(self, df: pd.DataFrame, scale_log: List[Dict[str, Any]]) -> None:
        """Recompute each ``<Scale>_mean`` from its item columns (reverse-recoded, NaN-aware)."""
        for entry in scale_log or []:
            cols = [c for c in (entry.get("columns_generated") or []) if c in df.columns]
            if not cols:
                continue
            mean_col = f"{cols[0].rsplit('_', 1)[0]}_mean"
            if mean_col not in df.columns:
                continue
            rev = set(entry.get("reverse_items") or [])
            flip = float(entry.get("scale_min", 1)) + float(entry.get("scale_max", 7))
            vals = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
            for j, col in enumerate(entry.get("columns_generated") or []):
                if (j + 1) in rev and col in cols:
                    k = cols.index(col)
                    vals[:, k] = flip - vals[:, k]
            answered = ~np.isnan(vals)
            counts = answered.sum(axis=1)
            sums = np.where(answered, vals, 0.0).sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
            df[mean_col] = np.round(means, 2)

    def _audit_individual_consistency(
        self,
        df: pd.DataFrame,
        data: Dict[str, list],
        all_traits: List[Dict[str, float]],
        scale_generation_log: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Post-generation consistency audit with targeted repair.

        Returns audit report dict with pass/fail per check and repair actions taken.
        """
        n = len(df)
        audit_report: Dict[str, Any] = {
            "per_scale_alpha": {},
            "reverse_item_violations": 0,
            "repairs_performed": 0,
            "total_checks": 0,
        }

        if n < 5:
            audit_report["skipped"] = "Sample too small for consistency audit"
            return audit_report

        # --- CHECK 1: Per-scale Cronbach's alpha ---
        for log_entry in scale_generation_log:
            cols = log_entry.get("columns_generated", [])
            if len(cols) < 3:
                continue  # Need ≥3 items for alpha
            if str(log_entry.get("type", "")).lower() in _JOINT_DV_TYPES:
                continue  # v1.2.7.0: rank-order/constant-sum are jointly constrained;
                          # re-injecting inter-item correlation would break the constraint

            audit_report["total_checks"] += 1
            try:
                # Build item matrix
                item_matrix = np.array(
                    [df[c].dropna().values for c in cols if c in df.columns],
                    dtype=float
                )
                if item_matrix.shape[0] < 3 or item_matrix.shape[1] < 5:
                    continue
                # Judge reliability in the construct direction (recode reverse items).
                _audit_rev = [r - 1 for r in (log_entry.get("reverse_items") or [])
                              if 1 <= r <= item_matrix.shape[0]]
                if _audit_rev:
                    item_matrix[_audit_rev, :] = (
                        float(log_entry.get("scale_min", 1)) + float(log_entry.get("scale_max", 7))
                        - item_matrix[_audit_rev, :]
                    )

                # Compute Cronbach's alpha
                k = item_matrix.shape[0]
                item_vars = np.var(item_matrix, axis=1, ddof=1)
                total_var = np.var(np.sum(item_matrix, axis=0), ddof=1)
                if total_var > 0:
                    alpha = (k / (k - 1)) * (1 - np.sum(item_vars) / total_var)
                else:
                    alpha = 0.0

                scale_name = log_entry.get("name", "unknown")
                audit_report["per_scale_alpha"][scale_name] = round(float(alpha), 3)

                # REPAIR: If alpha < 0.50, re-inject correlation (max 1 repair pass)
                if alpha < 0.50 and len(cols) >= 3:
                    scale_min = log_entry.get("scale_min", 1)
                    scale_max = log_entry.get("scale_max", 7)
                    target_alpha = 0.75  # Conservative target for repair
                    try:
                        _item_mat = np.array(
                            [data[c] for c in cols], dtype=float
                        ).T
                        # v1.2.8.1: scale-stable per-item loading heterogeneity.
                        _iic_seed = _stable_int_hash(f"{scale_name}|iic_loadings")
                        _repaired = _inject_inter_item_correlation(
                            _item_mat, target_alpha, scale_min, scale_max,
                            seed=_iic_seed, reverse_items=log_entry.get("reverse_items"),
                        )
                        for j, c in enumerate(cols):
                            data[c] = _repaired[:, j].tolist()
                            if c in df.columns:
                                df[c] = _repaired[:, j]
                        # Items changed -> refresh this scale's composite (reverse-aware)
                        _mcol = f"{cols[0].rsplit('_', 1)[0]}_mean"
                        if _mcol in df.columns:
                            _rv0 = [r - 1 for r in (log_entry.get("reverse_items") or []) if 1 <= r <= len(cols)]
                            _sc = _repaired.astype(float).copy()
                            if _rv0:
                                _sc[:, _rv0] = (float(scale_min) + float(scale_max)) - _sc[:, _rv0]
                            _newmeans = np.round(_sc.mean(axis=1), 2)
                            df[_mcol] = _newmeans
                            if _mcol in data:
                                data[_mcol] = _newmeans.tolist()
                        audit_report["repairs_performed"] += 1
                        self._log(f"AUDIT REPAIR: Re-correlated '{scale_name}' "
                                  f"(alpha {alpha:.2f} → target {target_alpha:.2f})")
                    except Exception as _repair_err:
                        self._log(f"AUDIT REPAIR FAILED for '{scale_name}': {_repair_err}")

            except Exception as _alpha_err:
                self._log(f"AUDIT: Alpha computation failed for {log_entry.get('name', '?')}: {_alpha_err}")

        # --- CHECK 2: Reverse-item sign violations ---
        audit_report["total_checks"] += 1
        for log_entry in scale_generation_log:
            cols = log_entry.get("columns_generated", [])
            if len(cols) < 2:
                continue
            # Check if any reverse items exist for this scale
            # (We can only detect this from the log entry)
            # Simple heuristic: if any column has "_R" suffix or the scale had reverse_items
            # For now, skip — the within-generation reverse tracking handles this

        # --- CHECK 3: Extreme individual-level inconsistency ---
        # Flag participants whose within-person SD across all items is suspiciously
        # high OR low relative to their persona
        audit_report["total_checks"] += 1
        all_scale_cols = []
        for log_entry in scale_generation_log:
            if str(log_entry.get("type", "")).lower() in _JOINT_DV_TYPES:
                continue  # v1.2.7.0: exclude rank-order/constant-sum from anti-straight-line
                          # jitter — ±1 noise on a near-uniform allocation breaks the sum
            all_scale_cols.extend(log_entry.get("columns_generated", []))

        if len(all_scale_cols) >= 3:
            existing_cols = [c for c in all_scale_cols if c in df.columns]
            if len(existing_cols) >= 3:
                # v1.2.8.1 (bugfix): map EACH column to its OWN scale's bounds.
                # The old code read scale_min/scale_max from `log_entry`, a LEAKED
                # loop variable holding the LAST scale (and defaulting scale_max to
                # 7), so jittering a 2-point item produced 2+1=3 — a scale-bounds
                # violation. With a per-column map every item is clipped to its own
                # range, and multi-scale surveys no longer cross-contaminate bounds.
                _col_bounds: Dict[str, Tuple[int, int]] = {}
                for _le in scale_generation_log:
                    _bmin = int(_le.get("scale_min", 1))
                    _bmax = int(_le.get("scale_max", _le.get("scale_points", 7)))
                    for _bc in _le.get("columns_generated", []):
                        _col_bounds[_bc] = (_bmin, _bmax)
                # v1.2.9.1: straight-lining is only suspicious when chance agreement is low,
                # i.e. across at least five items with five or more response options. On a
                # binary/3-point scale, or with only three items, most people legitimately
                # give the same answer to every item, and "repairing" them randomised the
                # data (and erased the condition effect).
                #
                # v1.3.0.4: the item-count half of that rule is relaxed for ONE case. The
                # identical-answer pass further down puts the registry's share of constant
                # rows back into every block of three or more items, and it was calibrated
                # on data this audit had already cleaned: the audit strips them (clipped
                # +/-1 jitter, which also drains the endpoint bins), the pass restores the
                # measured share. With five or more scale items in the survey nothing
                # changes. With 3 or 4 items, the whole survey of a short single-scale
                # design, switching the audit off left 4.6% of respondents identical before
                # the pass instead of 0.6% (8 seeds, N = 3,000), so the pass converted fewer,
                # differently chosen rows and the top bin ended 2.5 points above the next
                # one (P(7) - P(6) = +0.025 against -0.010 on main; the ceiling-spike guard
                # failed on 3 of 8 seeds). The audit therefore also runs when the survey has
                # a block of three or more items, which is exactly where that pass follows.
                # Single-item DVs and two-item scales get no such pass, so identical rows
                # across them stay as generated.
                _min_points = (min(hi - lo + 1 for lo, hi in _col_bounds.values())
                               if _col_bounds else 0)
                _widest_block = max(
                    (len(_le.get("columns_generated") or []) for _le in scale_generation_log
                     if str(_le.get("type", "")).lower() not in _JOINT_DV_TYPES),
                    default=0,
                )
                _check_straightlining = (
                    _min_points >= _MIN_OPTIONS_FOR_STRAIGHTLINE_LOGIC
                    and (len(existing_cols) >= 5 or _widest_block >= 3)
                )
                for i in range(n if _check_straightlining else 0):
                    vals = [float(df.iloc[i][c]) for c in existing_cols
                            if pd.notna(df.iloc[i][c])]
                    if len(vals) < 3:
                        continue
                    within_sd = float(np.std(vals))
                    # Flag if SD is essentially 0 (straight-liner) but persona is not Careless
                    if within_sd < 0.15 and i < len(all_traits):
                        attn = _safe_trait_value(all_traits[i].get("attention_level"), 0.7)
                        if attn > 0.5:
                            # This participant shouldn't be straight-lining
                            # Mild repair: add small noise to 2-3 items
                            # Only jitter cells that were actually answered: missing
                            # data is applied before this audit, and int(NaN) used to
                            # crash the whole run for studies with missingness enabled.
                            _answered_cols = [c for c in existing_cols if pd.notna(df.at[i, c])]
                            _items_to_jitter = min(3, len(_answered_cols))
                            if _items_to_jitter == 0:
                                continue
                            _rng = np.random.RandomState((self.seed + i * 31) % (2**31))
                            _jitter_cols = _rng.choice(_answered_cols, _items_to_jitter, replace=False)
                            for jc in _jitter_cols:
                                _old_val = int(df.at[i, jc]) if i in df.index else int(data[jc][i])
                                _noise = int(_rng.choice([-1, 1]))
                                _s_min, _s_max = _col_bounds.get(jc, (1, 7))
                                _new_val = max(_s_min, min(_s_max, _old_val + _noise))
                                if jc in df.columns:
                                    df.at[i, jc] = _new_val
                                data[jc][i] = _new_val
                            audit_report["repairs_performed"] += 1
                            # Keep each scale composite consistent with its (now
                            # edited) items: recode reverse items, skip missing cells.
                            for _le in scale_generation_log:
                                _lc = _le.get("columns_generated") or []
                                _mc = f"{_lc[0].rsplit('_', 1)[0]}_mean" if _lc else ""
                                if len(_lc) < 2 or _mc not in df.columns or not any(c in _jitter_cols for c in _lc):
                                    continue
                                _rv = set(_le.get("reverse_items") or [])
                                _fl = float(_le.get("scale_min", 0)) + float(_le.get("scale_max", 0))
                                _vs = [(_fl - float(df.at[i, c])) if (j + 1) in _rv else float(df.at[i, c])
                                       for j, c in enumerate(_lc) if c in df.columns and pd.notna(df.at[i, c])]
                                if _vs:
                                    _newmean = round(float(np.mean(_vs)), 2)
                                    df.at[i, _mc] = _newmean
                                    if _mc in data:
                                        data[_mc][i] = _newmean

        return audit_report

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

    @staticmethod
    def _readable_topic(text: Any) -> str:
        """``text`` when it reads like words (two or more real words, no underscores, few digits), else ''.

        Survey titles such as "BDS5010_G12" or "Survey_v2_FINAL" are identifiers: written into an answer
        ("thoughts about BDS5010_G12 ...") they are an artifact, not a topic."""
        t = re.sub(r"\s+", " ", str(text or "")).strip()
        if not t or "_" in t:
            return ""
        if len(re.findall(r"[A-Za-z]{3,}", t)) < 2 or sum(c.isdigit() for c in t) > 0.15 * len(t):
            return ""
        return t

    def _readable_study_topic(self) -> str:
        """The study title if it reads like words, else the start of the description, else ''."""
        title = self._readable_topic(self.study_title)
        if title:
            return title
        first_sentence = re.split(r"(?<=[.!?])\s", str(self.study_description or "").strip(), maxsplit=1)[0]
        return self._readable_topic(first_sentence[:120])

    def _build_enriched_question_text(
        self, question_text: str, question_context: str, condition: str
    ) -> str:
        """Build the enriched question text used for BOTH the LLM prompt and the
        response-pool cache key.

        v1.2.7.9 (H1 fix): The prefill path and the per-participant path used to
        build this string with two separately-maintained code blocks that had
        drifted: the per-participant block appended ``\\nCondition:`` and
        ``\\nAdditional context:`` while the prefill block did not. Because the
        pool key is ``md5(question_text[:200] | condition | sentiment)``, the two
        strings hashed differently whenever the divergence fell inside the first
        200 chars (typical for short variable-name questions with brief context),
        so every per-participant pool draw MISSED the prefilled pool — silently
        defeating the prefill budget (CLAUDE.md anti-pattern #29) and forcing an
        expensive on-demand call per participant. Routing both paths through this
        single helper guarantees byte-identical keys so the pool is actually hit.
        """
        import re as _re
        # v1.2.9.1: decode HTML entities, drop tags/placeholder text/bare variable ids. When nothing
        # usable remains, fall back to the study topic so answers still stay on topic.
        _qt = _clean_question_text(question_text)
        if not _qt:
            _topic_fb = self._readable_study_topic() or "the questions asked"
            _qt = f"Please share your thoughts about {_topic_fb}"
        _ctx = str(question_context or "").strip()
        if _ctx:
            _humanized = (_re.sub(r'[_\-]+', ' ', _qt).strip()
                          if _qt and " " not in _qt.strip() else _qt)
            _study_topic = self._readable_study_topic()
            _out = f"Question: {_humanized}\nContext: {_ctx}"
            if _study_topic:
                _out += f"\nStudy topic: {_study_topic}"
            # Include condition for tighter prompt grounding (it is ALSO a separate
            # pool-key component, but must be embedded identically in both paths).
            if condition:
                _out += f"\nCondition: {condition}"
            _add_ctx = self.study_context.get("additional_context", "")
            if _add_ctx:
                _out += f"\nAdditional context: {_add_ctx[:200]}"
            return _out
        if _qt and " " not in _qt.strip():
            # Variable-name-looking question with no context: build a richer prompt
            # from study context. (No condition embedded here — mirrors prior
            # behavior; condition still varies the pool key as a separate field.)
            _humanized = _re.sub(r'[_\-]+', ' ', _qt).strip()
            _study_topic = self._readable_study_topic()
            if _study_topic:
                return (f"In the context of a study about {_study_topic}, "
                        f"please share your thoughts on: {_humanized}")
            return f"Please share your thoughts on: {_humanized}"
        return _qt

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
        _original_question_text = _clean_question_text(question_text)  # v1.0.4.7: preserve before context embedding (v1.2.9.1: HTML-clean)
        context_type = str(question_spec.get("context_type", "general"))
        question_context = str(question_spec.get("question_context", "")).strip()

        # v1.0.1.2 / v1.2.7.9: Enrich the prompt with user-provided context,
        # study topic, condition, and additional context. This is critical for
        # questions like "explain_feel_donald" where the variable name alone
        # doesn't convey what's really being asked. Built via the SHARED helper so
        # the resulting string is byte-identical to the prefill path's pool key
        # (H1 fix — see _build_enriched_question_text).
        question_text = self._build_enriched_question_text(
            question_text, question_context, condition
        )

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
        # v1.2.2.9: EXCEPTION — when free_llm_oe_cap > 0, the user chose "Proceed
        # (AI for 100, template for rest)".  allow_template_fallback is True (needed
        # for graceful fallback after cap), but we MUST still try LLM for the first
        # 100 participants.  Once LLM is force-disabled (cap reached), skip directly
        # to comprehensive_generator for correct source tracking and efficiency.
        # v1.4.9: Try LLM generator first (question-specific, persona-aligned)
        _llm_force_off = getattr(self.llm_generator, '_force_disabled', False) if self.llm_generator else True
        # v1.2.7.7: also skip the LLM for this participant when the free tier is
        # throttled/exhausted right now (fast bail-out, recoverable) — avoids an
        # on-demand call per participant when providers are all rate-limited.
        _llm_throttled_now = getattr(self.llm_generator, 'free_tier_exhausted_now', False) if self.llm_generator else False
        _should_try_llm = (
            self.llm_generator is not None
            and not _llm_force_off
            and not _llm_throttled_now
            and self.llm_attempts_allowed()
        )
        if _should_try_llm:
            try:
                # v1.2.2.9: Track fallback count to distinguish real AI responses
                # from template responses returned through the LLM generator's
                # internal fallback path.  Without this, template fallbacks in cap
                # mode are labeled "AI", wasting the 100-response cap budget on
                # non-AI text and corrupting the _Generation_Source column.
                _fb_before = getattr(self.llm_generator, '_fallback_count', 0)
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
                    question_intent=_early_intent,
                    question_context=question_context,
                )
                if resp and resp.strip():
                    _fb_after = getattr(self.llm_generator, '_fallback_count', 0)
                    self._last_oe_source = "AI" if _fb_after == _fb_before else "Template"
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
        # v1.2.8.8: duplicate keys removed (corruption/memory/family/conspiracy/belief were
        # defined twice; Python kept the LAST value). Kept the FIRST (more specific,
        # natural-language) phrasing: 'conspiracy_theory' already covers the 'alternative
        # explanations' wording, and the narrative-domain 'memory'/'family'/'belief' rewordings
        # were less natural topic descriptions.
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
            'paranormal': 'paranormal beliefs and supernatural attitudes',
            # v1.0.8.3: Expanded for narrative/creative/disclosure question types
            'secret': 'personal secrets and self-disclosure',
            'disclosure': 'personal disclosure and private information sharing',
            'confession': 'confessions and personal admissions',
            'narrative': 'personal narratives and life stories',
            'anecdote': 'personal anecdotes and memorable experiences',
            'story': 'personal stories and lived experiences',
            'theory': 'personal theories and explanatory beliefs',
            'opinion': 'personal opinions and value judgments',
            'experience': 'personal experiences and life events',
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
        # v1.2.0.9: Use user-specified age bounds for clipping instead of hardcoded 18-70
        _age_clip_min = _safe_numeric(self.demographics.get("age_min", 18), default=18.0)
        _age_clip_max = _safe_numeric(self.demographics.get("age_max", 80), default=80.0)
        ages = rng.normal(age_mean, age_sd, int(n))
        ages = np.clip(ages, _age_clip_min, _age_clip_max).astype(int)

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

        df_demo = pd.DataFrame({"Age": ages, "Gender": genders})

        # v1.2.0.4: Custom demographic variables (political orientation, education, etc.)
        custom_demos = self.demographics.get("custom_demographics", [])
        for demo_spec in custom_demos:
            if not isinstance(demo_spec, dict):
                continue
            col_name = str(demo_spec.get("name", "")).strip()
            if not col_name:
                continue
            demo_type = str(demo_spec.get("demo_type", "categorical")).lower()
            options = demo_spec.get("options", [])
            weights = demo_spec.get("weights", [])

            if demo_type == "categorical" and options:
                # Categorical variable (e.g., political orientation, education, ethnicity)
                n_opts = len(options)
                if weights and len(weights) == n_opts:
                    _w = np.array(weights, dtype=float)
                    _w = np.clip(_w, 0, None)
                    _wsum = _w.sum()
                    probs = (_w / _wsum).tolist() if _wsum > 0 else [1.0 / n_opts] * n_opts
                else:
                    probs = [1.0 / n_opts] * n_opts
                df_demo[col_name] = rng.choice(options, size=int(n), p=probs)
            elif demo_type == "numeric":
                # Numeric demographic (e.g., income, years of education)
                d_mean = float(demo_spec.get("mean", 50))
                d_sd = float(demo_spec.get("sd", 15))
                d_min = float(demo_spec.get("min", 0))
                d_max = float(demo_spec.get("max", 100))
                vals = rng.normal(d_mean, max(0.1, d_sd), int(n))
                vals = np.clip(vals, d_min, d_max)
                if demo_spec.get("integer", True):
                    vals = vals.astype(int)
                df_demo[col_name] = vals
            elif demo_type == "ordinal" and options:
                # Ordinal variable with roughly normal distribution across levels
                n_opts = len(options)
                # Centre-weighted distribution for ordinal levels
                _x = np.arange(n_opts, dtype=float)
                _centre = (n_opts - 1) / 2.0
                _spread = max(1.0, n_opts / 3.0)
                _raw = np.exp(-0.5 * ((_x - _centre) / _spread) ** 2)
                probs = (_raw / _raw.sum()).tolist()
                df_demo[col_name] = rng.choice(options, size=int(n), p=probs)

        return df_demo

    def _adjust_demographics_for_personas(
        self,
        data: Dict[str, list],
        assigned_personas: List[str],
        all_traits: List[Dict[str, float]],
        n: int,
    ) -> None:
        """Adjust custom demographic values for persona consistency.

        v1.2.0.5: Post-hoc adjustment that swaps demographic values between
        participants to create realistic persona↔demographic correlations
        WITHOUT changing the marginal distribution of any demographic column.

        Strategy: for each ordinal/categorical demographic, score how "fitting"
        each option is for each persona, then probabilistically swap values
        between participants so that better-fitting values end up with the
        matching personas.  Because we only swap (never create new values),
        the marginal distribution is exactly preserved.

        Grounded in:
        - Weijters et al. (2010): acquiescence correlates with lower education
        - Yan & Tourangeau (2008): response speed correlates with age
        - Greenleaf (1992): extremity associated with certain demographics
        - Iyengar & Westwood (2015): political identity shapes economic behavior
        """
        custom_demos = self.demographics.get("custom_demographics", [])
        if not custom_demos:
            return

        rng = np.random.RandomState(self.seed + 2000)

        # Persona→demographic affinity rules.  Maps (persona_keyword, demo_keyword)
        # to a direction: +1 = skew toward high end of ordinal / last options,
        # -1 = skew toward low end / first options.  Magnitude 0.0-1.0.
        _PERSONA_DEMO_AFFINITIES = {
            # Political
            ("partisan", "political"): 0.7,    # Partisan Ideologue → extreme political orientation
            ("partisan", "party"): 0.7,
            ("egalitarian", "political"): -0.3, # Egalitarian → liberal-leaning
            ("conformist", "political"): 0.0,   # Conformist → moderate
            ("individualist", "political"): 0.3,# Individualist → conservative-leaning
            # Education
            ("deep_learner", "education"): 0.6, # Deep Learner → higher education
            ("surface_learner", "education"): -0.3,
            ("engaged", "education"): 0.4,      # Engaged Responder → higher education (Krosnick)
            ("careless", "education"): -0.5,    # Careless → lower education (Weijters et al.)
            ("satisficer", "education"): -0.2,
            # Age-related
            ("tech_enthusiast", "age"): -0.5,   # Tech personas → younger
            ("digital_native", "age"): -0.6,
            ("gen_z", "age"): -0.8,
            ("boomer", "age"): 0.8,
            ("traditional", "age"): 0.5,
            # Income
            ("deal_seeker", "income"): -0.3,    # Deal Seeker → lower income
            ("hedonic", "income"): 0.3,         # Hedonic Consumer → higher income
            ("conscious_consumer", "income"): 0.2,
            # Religion
            ("moral_absolutist", "religio"): 0.5,
            ("conformist", "religio"): 0.3,
        }

        for demo_spec in custom_demos:
            if not isinstance(demo_spec, dict):
                continue
            col_name = str(demo_spec.get("name", "")).strip()
            if not col_name or col_name not in data:
                continue
            demo_type = str(demo_spec.get("demo_type", "categorical")).lower()
            options = demo_spec.get("options", [])
            if demo_type not in ("ordinal", "categorical") or len(options) < 2:
                continue

            col_lower = col_name.lower()
            values = list(data[col_name])
            n_opts = len(options)

            # Compute a "preference score" per participant: how much they should
            # skew toward high vs low options based on their persona.
            pref_scores = np.zeros(n, dtype=float)
            for i in range(n):
                persona_lower = assigned_personas[i].lower().replace(" ", "_")
                traits = all_traits[i]
                score = 0.0
                for (p_key, d_key), affinity in _PERSONA_DEMO_AFFINITIES.items():
                    if p_key in persona_lower and d_key in col_lower:
                        score += affinity
                # Also use trait-based soft coupling:
                # extremity → more extreme ordinal values
                extremity = traits.get("extremity", 0.5)
                if extremity > 0.7 and "political" in col_lower:
                    score += (extremity - 0.5) * 0.8  # push toward extremes

                # acquiescence + low attention → lower education affinity (Weijters)
                if "education" in col_lower:
                    acq = traits.get("acquiescence", 0.5)
                    att = traits.get("attention_level", 0.7)
                    score += (att - 0.5) * 0.5  # higher attention → higher education
                    score -= (acq - 0.5) * 0.3  # higher acquiescence → lower education

                pref_scores[i] = float(np.clip(score, -1.5, 1.5))

            # Convert current values to ordinal indices
            option_to_idx = {str(o): j for j, o in enumerate(options)}
            current_indices = np.array([option_to_idx.get(str(v), n_opts // 2) for v in values])

            # Swap-sort: participants with higher pref_scores should tend to have
            # higher ordinal indices.  Do probabilistic swaps to create correlation
            # while preserving the exact marginal distribution.
            max_abs_score = max(0.01, float(np.max(np.abs(pref_scores))))
            # Number of swap passes proportional to the strength of correlations
            n_swaps = int(n * 0.3 * (max_abs_score / 1.5))
            n_swaps = max(0, min(n_swaps, n * 2))

            for _ in range(n_swaps):
                # Pick two random participants
                i1, i2 = rng.choice(n, size=2, replace=False)
                idx1, idx2 = current_indices[i1], current_indices[i2]
                if idx1 == idx2:
                    continue
                # Participant with higher pref_score should have higher index
                if pref_scores[i1] > pref_scores[i2] and idx1 < idx2:
                    # i1 wants higher but has lower → swap with probability
                    swap_prob = min(0.8, abs(pref_scores[i1] - pref_scores[i2]) / 2.0)
                    if rng.random() < swap_prob:
                        current_indices[i1], current_indices[i2] = idx2, idx1
                elif pref_scores[i1] < pref_scores[i2] and idx1 > idx2:
                    swap_prob = min(0.8, abs(pref_scores[i1] - pref_scores[i2]) / 2.0)
                    if rng.random() < swap_prob:
                        current_indices[i1], current_indices[i2] = idx2, idx1

            # Write back adjusted values
            for i in range(n):
                data[col_name][i] = options[int(current_indices[i])]

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

    # ------------------------------------------------------------------
    # v1.2.7.x DV-type-aware post-processing helpers (called from generate())
    # ------------------------------------------------------------------
    def _apply_joint_dv_structure(self, data, scale_log, n):
        """Convert independently-generated rank-order / constant-sum items into
        their correct JOINT structure (a valid 1..k permutation; or an allocation
        summing exactly to the total). The per-item values already carry condition
        effects + persona variation, so we only reshape — we don't re-roll.

        Returns the set of column-prefixes whose composite ``_mean`` is meaningless
        (so generate() can skip it). Also records ``self._typed_dv_columns`` so the
        consistency audit / bounds-clipping skip these jointly-constrained columns.
        Every non-joint DV type is left byte-identical (gated on ``_JOINT_DV_TYPES``).
        """
        typed_dv_scales: set = set()
        self._typed_dv_columns = set()
        for log_entry in scale_log:
            dv_type = log_entry.get("type", "")
            icols = log_entry.get("columns_generated", [])
            k = len(icols)
            if k < 2 or dv_type not in _JOINT_DV_TYPES:
                continue
            if any(c not in data or len(data[c]) < n for c in icols):
                continue
            self._typed_dv_columns.update(icols)
            try:
                M = np.array([data[c] for c in icols], dtype=float).T  # (n, k)
                smin = float(log_entry.get("scale_min", 1))
                smax = float(log_entry.get("scale_max", k))
                prefix = icols[0].rsplit("_", 1)[0]
                if dv_type in ("rank_order", "ranking", "rank order"):
                    # Latent preference = generated value + small STABLE per-item base
                    # → argsort to a valid 1..k permutation (rank 1 = most preferred).
                    # Plackett-Luce/Thurstonian flavour: systematic item preference +
                    # idiosyncratic per-participant variation.
                    base = np.array([(_stable_int_hash(c) % 997) / 997.0 for c in icols])
                    util = M + base[None, :] * max(1.0, (smax - smin)) * 0.20
                    order = np.argsort(-util, axis=1, kind="stable")  # most→least
                    ranks = np.empty((n, k), dtype=int)
                    rows = np.arange(n)
                    for pos in range(k):
                        ranks[rows, order[:, pos]] = pos + 1
                    for j, c in enumerate(icols):
                        data[c] = ranks[:, j].tolist()
                else:  # constant-sum: renormalize each row to sum EXACTLY to total
                    total = int(round(smax)) if smax and smax >= k else 100
                    W = np.clip(M - smin + 0.5, 0.01, None)  # values as positive weights
                    alloc = W / W.sum(axis=1, keepdims=True) * total
                    floor = np.floor(alloc).astype(int)
                    for ri in range(n):  # largest-remainder rounding to hit total exactly
                        resid = total - int(floor[ri].sum())
                        if resid != 0:
                            frac_order = np.argsort(-(alloc[ri] - floor[ri]))
                            for t in range(abs(resid)):
                                floor[ri, frac_order[t % k]] += 1 if resid > 0 else -1
                            np.clip(floor[ri], 0, total, out=floor[ri])
                    for j, c in enumerate(icols):
                        data[c] = floor[:, j].tolist()
                typed_dv_scales.add(prefix)
                self._log(f"Applied {dv_type} joint structure to '{log_entry.get('name')}' ({k} items)")
            except Exception as err:
                self._log(f"WARNING: joint-DV transform failed for '{log_entry.get('name')}': {err}")
        return typed_dv_scales

    @staticmethod
    def _numeric_dv_kind(log_entry):
        """Classify a numeric DV as 'money', 'count', or None using DV-SPECIFIC text
        ONLY (name/columns/question_text/description/anchors/item_names) — never the
        study title/description. Rating/attitude items are never money/count even if
        a cue word appears. Returns 'money' | 'count' | None."""
        if str(log_entry.get("type", "")).lower() not in ("numeric", "numeric_input"):
            return None
        ctx = " ".join([
            str(log_entry.get("name", "")),
            " ".join(log_entry.get("columns_generated", []) or []),
            str(log_entry.get("question_text", "")),
            str(log_entry.get("dv_description", "")),
            " ".join(str(x) for x in (log_entry.get("item_names") or [])),
            " ".join(str(v) for v in (log_entry.get("scale_anchors") or {}).values()),
        ]).replace("_", " ").replace("-", " ")  # expose cues at letter boundaries
        if _RATING_CTX_RE.search(ctx):
            return None
        if _MONEY_CUE_RE.search(ctx):
            return "money"
        if _COUNT_CUE_RE.search(ctx):
            return "count"
        return None

    def _apply_numeric_distribution_realism(self, data, scale_log):
        """Reshape money/WTP/count numeric DVs to a realistic RIGHT-SKEWED marginal
        (money: log-normal + ~12% floor spike; count: gamma, mode low). Uses
        rank-assignment so the condition/persona ordering — hence treatment effects —
        is preserved. Generic numeric DVs (age, temperature, …) are left untouched."""
        for log_entry in scale_log:
            kind = self._numeric_dv_kind(log_entry)
            if kind is None:
                continue
            icols = log_entry.get("columns_generated", [])
            if not icols or any(c not in data or len(data[c]) < 8 for c in icols):
                continue  # need enough rows for a stable marginal
            smin = float(log_entry.get("scale_min", 0))
            smax = float(log_entry.get("scale_max", 100))
            span = smax - smin
            if span <= 0:
                continue
            try:
                for c in icols:
                    cur = np.asarray(data[c], dtype=float)
                    nn = len(cur)
                    rng = np.random.RandomState((self.seed + _stable_int_hash(c)) % (2**31))
                    if kind == "money":
                        # log-normal scaled to a target MEDIAN (~30% of span) so the
                        # long upper tail survives; then a floor spike (~12% at min).
                        samp = np.sort(rng.lognormal(mean=0.0, sigma=0.9, size=nn))
                        target = smin + samp * (span * 0.30)
                        n_floor = int(round(nn * 0.12))
                        if n_floor > 0:
                            target[:n_floor] = smin
                    else:  # count/frequency → right-skewed, mode low
                        samp = np.sort(rng.gamma(shape=1.6, scale=1.0, size=nn))
                        target = smin + samp * (span * 0.18)
                    # rank-assignment: smallest current value → smallest target value.
                    ranks = np.argsort(np.argsort(cur, kind="stable"))
                    data[c] = np.clip(np.round(target[ranks]), smin, smax).astype(int).tolist()
                self._log(f"Applied {kind} right-skew realism to numeric DV '{log_entry.get('name')}'")
            except Exception as err:
                self._log(f"WARNING: numeric realism transform failed for '{log_entry.get('name')}': {err}")

    def _apply_identical_answer_realism(self, data, scale_log, frame=None) -> List[str]:
        """Bring the share of identical-answer respondents up to the real level.

        In real item-level data (see utils/reference_profiles.json), 5.2% of
        respondents give the SAME answer to every item of a 5-item block once the
        items are aligned to the construct direction — people sitting at the
        ceiling of a construct, answering "6,6,6,6,6". A continuous latent trait
        plus noise almost never produces five identical integers: before this pass
        the simulator produced 0.2%, a 25-fold shortfall that any careless-
        responding screen over the output would show up.

        Only direction-aligned Likert-style blocks of 3+ items are touched, and a
        block that already has enough constant rows is left exactly as it was.

        `frame` is the authoritative source when given. By the point this runs,
        later passes have written corrections into the DataFrame that never went
        back into `data`, so reading `data` would resurrect pre-correction values:
        doing that measured as condition means of 4.31/4.64 against the 3.00/4.00
        the calibrated frame actually held.
        """
        changed: List[str] = []
        if not HAS_ITEM_REALISM:
            return changed
        for log_entry in scale_log:
            icols = log_entry.get("columns_generated") or []
            if len(icols) < 3:
                continue
            if any(c in getattr(self, "_typed_dv_columns", set()) for c in icols):
                continue        # rank-order / constant-sum blocks have their own shape
            def _source(col):
                if frame is not None and col in getattr(frame, "columns", []):
                    return list(frame[col].tolist())
                return list(data.get(col, []))

            if any(len(_source(c)) < 20 for c in icols):
                continue
            try:
                smin = float(log_entry.get("scale_min", 1))
                smax = float(log_entry.get("scale_max", 5))
                if smax - smin <= 0 or smax - smin > 10:
                    continue    # wide/continuous DVs are shaped elsewhere
                # v1.3.0.4 — the registry's rates were measured on 5- to 9-point
                # instruments. On a binary, 3- or 4-point block chance agreement is far
                # higher (about 77% of the rows of a 3-item binary block are identical
                # as generated, against 19% in the registry for three items), so pulling
                # the share down to the registry value rewrote honest answers: a
                # requested d of 0.8 came back as 0.61 on a 3-item binary scale (0.77
                # without this pass). A declined block is left as it was, the same rule
                # the registry applies outside the designs it was measured on, and the
                # consistency audit is gated on the same number of options.
                if smax - smin + 1 < _MIN_OPTIONS_FOR_STRAIGHTLINE_LOGIC:
                    continue
                _rng = random.Random(self.seed + _stable_int_hash(str(log_entry.get("name", ""))))
                cols = [_source(c) for c in icols]
                if any(any(v != v for v in col) for col in cols):
                    continue      # missing data present: leave the block alone
                # v1.2.9.4 — the target is width- and keying-conditional, not one
                # global share. Measured over every contiguous window of four
                # published instruments, the rate falls from 7.1% at 3 items to
                # 0.17% at 10, and a same-keyed block runs 3-6x a mixed-keyed one
                # of the same width (nothing contradicts a run of identical answers
                # when every item points the same way). A single 5.2% constant is
                # therefore roughly right only for a direction-aligned 5-item block
                # and wrong by an order of magnitude at either end of that range.
                # The registry declines outside the widths it was measured at, and
                # a declined lookup keeps the previous constant.
                #
                # v1.2.9.9 — the per-width entries were measured at k=3..10
                # (k=3..9 same-keyed). Past that the lookup declines, and falling
                # back to the 5.2% default would be worse than doing nothing: the
                # measured rate is 0.169% at k=10 and 0.124% across a 22-50 item
                # instrument, so a 22-item Big Five block would have about 40x too
                # many respondents collapsed to a single value -- and because this
                # pass now corrects downward as well as up, it would actively
                # CREATE them, distorting composites, alpha and every
                # careless-response flag computed from the block. So: the width
                # entry, else the full-instrument entry, else leave the block
                # alone. A declined lookup changes nothing, which is the rule the
                # rest of the registry follows.
                _target_share = None
                _sl_hit = None
                if HAS_EMPIRICAL_REGISTRY:
                    _rev = log_entry.get("reverse_items") or []
                    _keying = "mixed" if _rev else "same"
                    _sl_sig = _design_signature.for_block(
                        scale_min=smin, scale_max=smax, n_items=len(icols),
                        keying=_keying,
                    )
                    _sl_hit = _empirical_registry.lookup_best(
                        f"item.likert.{_keying}.k{len(icols)}",
                        "straightlined_share", _sl_sig)
                    if _sl_hit is None:
                        _sl_hit = _empirical_registry.lookup_best(
                            "item.likert5.full_instrument",
                            "straightlined_share", _sl_sig)
                    if _sl_hit is not None:
                        _target_share = float(_sl_hit.value)
                elif len(icols) <= 10:
                    # No registry at all: the old global constant, which was
                    # roughly right for a block of this width and is the previous
                    # behaviour. Beyond that width it was never right, so skip.
                    _target_share = DEFAULT_STRAIGHTLINE_SHARE
                if _target_share is None:
                    continue
                new_cols, report = match_straightlining(
                    cols, smin, smax,
                    target_share=_target_share, rng=_rng,
                )
                if report.get("applied"):
                    # Keep integer columns integral: these passes work in floats,
                    # and letting that leak would turn an exported "4" into "4.0"
                    # for every Likert item in the CSV.
                    for j, c in enumerate(icols):
                        vals = [int(round(v)) for v in new_cols[j]]
                        data[c] = vals
                        if frame is not None and c in getattr(frame, "columns", []):
                            frame[c] = vals
                    changed.extend(icols)
                    _src = (f"{_sl_hit.entry_id}, {_sl_hit.tier}" if _sl_hit is not None
                            else "default 5-item reference")
                    self._log(
                        f"Identical-answer realism for '{log_entry.get('name')}': "
                        f"{report.get('share_before')} -> {report.get('share_after')} "
                        f"(target {_target_share:.4f} from {_src})"
                    )
                    if _sl_hit is not None and self.registry_ledger is not None:
                        self.registry_ledger.record_lookup(
                            _sl_hit, f"scale '{log_entry.get('name')}' straight-lining")
                    self._item_realism_log.append(
                        dict(report, scale=log_entry.get("name"),
                             stage="identical_answers",
                             entry_id=(_sl_hit.entry_id if _sl_hit else ""))
                    )
            except Exception as err:
                self._log(
                    f"WARNING: identical-answer realism failed for "
                    f"'{log_entry.get('name')}': {err}"
                )
        return changed

    def generate(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        # v1.2.8.4 (Codex P2 — de-serialize multi-user runs): generation no longer
        # seeds the process-GLOBAL np.random/random, so it no longer needs a
        # process-wide lock held across the WHOLE run (including LLM network I/O).
        # Reproducibility is now fully PER-INSTANCE: every numeric draw uses a local
        # np.random.RandomState seeded from self.seed/participant_seed (23 sites);
        # the only previously-global consumers in the generation path —
        # HBSValidator and the LLM backoff jitter — were migrated to per-instance
        # RNGs (HBSValidator(seed=self.seed); LLMResponseGenerator._rng). Two
        # concurrent Streamlit sessions therefore run fully in parallel with
        # identical same-seed output and zero cross-session interference. Verified by
        # the cross-process determinism battery + a concurrent-generation test.
        return self._generate_body()

    def _generate_body(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        n = self.sample_size
        data: Dict[str, Any] = {}

        # Reset text generator's used responses for fresh dataset
        self.text_generator.reset_used_responses()
        # v1.2.7.9 (M2 fix): Also reset the PRIMARY non-LLM OE generator's dedup
        # state. Previously only text_generator (the fallback) was reset, so a
        # REUSED engine produced different OE text on the same seed because
        # comprehensive_generator's _used_responses/_used_sentences carried over.
        # The Streamlit UI rebuilds the engine each run (so this was latent), but
        # the reproducibility contract must hold for SDK/batch reuse too.
        if self.comprehensive_generator is not None and hasattr(self.comprehensive_generator, 'reset'):
            try:
                self.comprehensive_generator.reset()
            except Exception:
                pass

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
        # v1.2.0.9: Respect include flags — only add Age/Gender to output if user wants them.
        # Demographics are always generated internally for persona assignment.
        _include_age_col = self.demographics.get("include_age_column", True)
        _include_gender_col = self.demographics.get("include_gender_column", True)
        if _include_age_col:
            data["Age"] = demographics_df["Age"].tolist()
            _age_clip_min = self.demographics.get("age_min", 18)
            _age_clip_max = self.demographics.get("age_max", 80)
            self.column_info.append(
                ("Age", f"Participant age in years ({_age_clip_min}-{_age_clip_max}, mean ~ {self.demographics.get('age_mean', 35)})")
            )
        if _include_gender_col:
            data["Gender"] = demographics_df["Gender"].tolist()
            self.column_info.append(
                ("Gender", "Participant gender: Male, Female, Non-binary, or Prefer not to say")
            )
        # v1.2.0.4: Add custom demographic columns to the output
        for _demo_col in demographics_df.columns:
            if _demo_col not in ("Age", "Gender"):
                data[_demo_col] = demographics_df[_demo_col].tolist()
                _custom_demos = self.demographics.get("custom_demographics", [])
                _demo_desc = next(
                    (d.get("description", f"Custom demographic variable: {_demo_col}")
                     for d in _custom_demos if isinstance(d, dict) and d.get("name") == _demo_col),
                    f"Custom demographic variable: {_demo_col}"
                )
                self.column_info.append((_demo_col, _demo_desc))

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

        # v1.2.0.5: Adjust custom demographics to be consistent with personas.
        # After personas are assigned, modulate demographic values so that e.g.
        # "Partisan Ideologue" skews toward extreme political orientation,
        # "Tech Enthusiast" skews younger, "Deep Learner" skews higher education.
        # This preserves the user-specified MARGINAL distributions while adding
        # realistic persona↔demographic correlations.
        self._adjust_demographics_for_personas(data, assigned_personas, all_traits, n)

        # =================================================================
        # ABE 3.0: CONSISTENCY IMPROVEMENT #2 — Demographic → Style Coupling
        # Modulate traits based on demographics (small effects ±0.08)
        # =================================================================
        for i in range(n):
            all_traits[i] = self._apply_demographic_trait_modulation(
                all_traits[i], i, demographics_df,
            )

        # =================================================================
        # ABE 3.0: CONSISTENCY IMPROVEMENT #3 — 3D Latent Attitude Vector
        # Generate correlated 3D latent dimensions per participant
        # =================================================================
        for i in range(n):
            _latent_seed = (self.seed + i * 7723) % (2**31)
            _latent_vec = self._generate_latent_attitude_vector(all_traits[i], _latent_seed)
            all_traits[i].update(_latent_vec)

        # ABE 3.0: Initialize response inertia memory (Improvement #4)
        self._response_inertia_memory: Dict[int, list] = {i: [] for i in range(n)}

        # ABE 3.0: Track global item index for survey-level fatigue (Improvement #1)
        self._global_item_counter = 0
        self._total_global_items = sum(
            max(1, int(float(s.get("num_items", 1)))) for s in self.scales
        )

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

        self._latent_dv_names = set()
        self._deferred_effect_vars = set()
        self._deferred_effect_log = []
        self._effect_build_errors = []
        self._reversal_ok_arr = None
        if _corr_matrix is not None and len(_scale_names) > 1:
            try:
                _latent_scores = generate_latent_scores(n, _calibrate_latent_correlation(_corr_matrix), self.seed)
                # Store latent z-scores in each participant's traits
                for i in range(n):
                    all_traits[i]["_latent_dvs"] = {
                        _scale_names[j]: float(_latent_scores[i, j])
                        for j in range(min(len(_scale_names), _latent_scores.shape[1]))
                    }
                # the scales whose responses carry the cross-scale latent term (matched by name)
                self._latent_dv_names = set(_scale_names[:_latent_scores.shape[1]])
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
                "reverse_items": sorted(i for i in reverse_items if 1 <= i <= num_items),
                "type": str(scale.get("type", "")).lower(),
                "question_text": str(scale.get("question_text", "")),
                "dv_description": str(scale.get("dv_description", "")),
                # v1.2.7.4: store anchors + item names so numeric-realism
                # classification can actually read them (previously read but never
                # stored — a money cue living only in an anchor label was missed).
                "scale_anchors": scale.get("scale_anchors", {}) or {},
                "item_names": scale.get("item_names", []) or [],
                "columns_generated": [],
            })

            if not hasattr(self, "_scale_effect_meta"):
                self._scale_effect_meta = {}
            self._scale_effect_meta[str(scale_name)] = (num_items, scale_min, scale_max)

            # v1.2.9.1: next to other scales every response carries a person-level latent term that
            # triples the composite's SD, so a shift calibrated for a lone scale reaches only ~0.4 of
            # the requested d. A user effect on such a scale is therefore built into the finished item
            # responses, in units of the realised within-condition SD (_apply_user_effect_to_scale).
            if self._defer_user_effect_for_scale(scale_name, scale_min, scale_max, bool(reverse_items)):
                self._deferred_effect_vars.add(scale_name)
                self._reversal_ok_arr = (scale_name, np.ones((n, num_items), dtype=bool))
            else:
                self._reversal_ok_arr = None

            for item_num in range(1, num_items + 1):
                col_name = f"{scale_name}_{item_num}"
                is_reverse = item_num in reverse_items

                # v1.2.0.1: Report progress per item within scale so the UI timer
                # keeps moving.  Without this, a single scale with many items
                # leaves the progress stale for minutes.
                _report_progress("scales", scale_idx, _total_scales)

                item_values: List[int] = []
                col_hash = _stable_int_hash(col_name)
                # v1.2.0.1: Calculate callback interval for per-participant updates
                # during scale generation.  Fire every ~5% of participants (min 1,
                # max every 20) so the elapsed timer keeps moving.
                _scale_cb_interval = max(1, min(20, n // 20))
                for i in range(n):
                    # v1.2.0.1: Fire "generating" callback periodically within
                    # the scale inner loop.  This drives the participant counter
                    # and elapsed timer in the UI during long scale generation.
                    if i % _scale_cb_interval == 0:
                        _report_progress("generating", i, n)
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

                    # ABE 3.0: Update response inertia memory (Improvement #4)
                    if hasattr(self, '_response_inertia_memory') and i in self._response_inertia_memory:
                        self._response_inertia_memory[i].append(
                            (scale_name, _normalized_val)
                        )
                        # Keep only last 5 items per participant
                        if len(self._response_inertia_memory[i]) > 5:
                            self._response_inertia_memory[i] = self._response_inertia_memory[i][-5:]

                # ABE 3.0: Increment global item counter for survey-level fatigue
                self._global_item_counter = getattr(self, '_global_item_counter', 0) + 1

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
                # v1.2.6.6: Check existing alpha BEFORE injecting correlation.
                # Items already share condition effects + traits + per-scale
                # tendency, so they may already exceed the target. Only inject
                # additional correlation if current alpha is below target.
                # Target reliability: honour an explicit value (e.g. 0.85 from the
                # scale builder); otherwise draw a realistic per-scale alpha
                # (0.80-0.90, seeded by scale name so scales differ but runs are
                # reproducible) instead of one fixed 0.75 for every scale.
                _rel_user = scale.get("reliability")
                try:
                    target_alpha = float(_rel_user) if _rel_user is not None else None
                except (TypeError, ValueError):
                    target_alpha = None
                if target_alpha is None or not (0.3 <= target_alpha <= 0.99):
                    target_alpha = float(np.random.RandomState(
                        _stable_int_hash(f"{scale_name}|target_alpha") & 0x7FFFFFFF).uniform(0.80, 0.90))
                item_col_names = [f"{scale_name}_{j+1}" for j in range(num_items)]
                try:
                    _rev_idx0 = [r - 1 for r in sorted(reverse_items) if 1 <= r <= num_items]

                    def _construct_matrix(item_col_names=item_col_names, _rev_idx0=_rev_idx0,
                                          scale_min=scale_min, scale_max=scale_max) -> np.ndarray:
                        # Alpha/correlation must be judged in the CONSTRUCT direction.
                        _m = np.array([data[c] for c in item_col_names], dtype=float).T
                        if _rev_idx0:
                            _m[:, _rev_idx0] = (scale_min + scale_max) - _m[:, _rev_idx0]
                        return _m

                    def _std_alpha(_m: np.ndarray, num_items=num_items) -> float:
                        _c = np.corrcoef(_m.T)
                        _rb = float(np.mean(_c[np.triu_indices_from(_c, k=1)]))
                        return (num_items * _rb) / (1 + (num_items - 1) * _rb) if _rb > 0 else 0.0

                    _item_matrix = np.array([data[c] for c in item_col_names], dtype=float).T
                    _existing_alpha = _std_alpha(_construct_matrix())
                    if _existing_alpha < target_alpha:
                        # v1.2.8.1: scale-stable seed -> per-item loading heterogeneity.
                        _iic_seed = _stable_int_hash(f"{scale_name}|iic_loadings")
                        _correlated = _inject_inter_item_correlation(
                            _item_matrix, target_alpha, scale_min, scale_max,
                            seed=_iic_seed, reverse_items=sorted(reverse_items),
                        )
                        for j, c in enumerate(item_col_names):
                            data[c] = _correlated[:, j].tolist()
                        self._log(f"Injected inter-item correlation for '{scale_name_raw}' (existing alpha={_existing_alpha:.2f} -> target={target_alpha:.2f})")
                    # The injection assumes independent items, so on items that already
                    # share variance it OVERSHOOTS (alpha 0.73 -> 0.95 with target 0.75).
                    # Bring any alpha well above target back to it (item-specific noise).
                    _alpha_now = _std_alpha(_construct_matrix())
                    if _alpha_now > target_alpha + 0.04:
                        _cd_new = _attenuate_inter_item_correlation(
                            _construct_matrix(), _alpha_now, target_alpha, scale_min, scale_max,
                            seed=_stable_int_hash(f"{scale_name}|alpha_noise"),
                        ).astype(float)
                        if _rev_idx0:
                            _cd_new[:, _rev_idx0] = (scale_min + scale_max) - _cd_new[:, _rev_idx0]
                        for j, c in enumerate(item_col_names):
                            data[c] = _cd_new[:, j].astype(int).tolist()
                        self._log(f"Attenuated inter-item correlation for '{scale_name_raw}' (alpha={_alpha_now:.2f} -> ~{target_alpha:.2f})")
                except Exception as _corr_err:
                    self._log(f"WARNING: Could not inject correlation for '{scale_name_raw}': {_corr_err}")

                # v1.2.9.1 — VERIFY the reliability that was actually achieved.
                # Both paths above can leave a block MORE internally consistent
                # than the target: injection overshoots (measured 0.58 -> 0.88
                # while aiming at 0.75 on a 5-item block), and a block that
                # started above target was previously left untouched. Benchmarked
                # against real item-level data (utils/reference_profiles.json),
                # that overshoot is the clearest signature of synthetic survey
                # data: mean inter-item r 0.60 and alpha 0.88 where real 5-item
                # blocks sit at 0.36 and 0.73, with within-person SD 0.72 against
                # 1.02 and 0.2% of respondents giving identical answers against
                # 5.2%. Scaling the item-specific variance up fixes all of them
                # at once, and leaves every participant's rank on the construct —
                # so every condition effect — where it was.
                if HAS_ITEM_REALISM and num_items >= 3:
                    try:
                        _target_r = target_r_from_alpha(target_alpha, num_items)
                        _cols = [list(data[c]) for c in item_col_names
                                 if c in data]
                        if len(_cols) == num_items:
                            _new_cols, _dc_report = decouple_block(
                                _cols, scale_min, scale_max,
                                condition_labels=data.get("CONDITION"),
                                target_r=_target_r,
                                preserve_effect=True,
                            )
                            if _dc_report.applied:
                                for j, c in enumerate(item_col_names):
                                    data[c] = [int(round(v)) for v in _new_cols[j]]
                                self._log(
                                    f"Reliability correction for '{scale_name_raw}' "
                                    f"(target alpha {target_alpha:.2f} -> r {_target_r:.3f}): "
                                    f"{_dc_report.reason}"
                                )
                                self._item_realism_log.append(
                                    dict(scale=scale_name_raw, stage="decouple",
                                         **_dc_report.as_dict())
                                )
                    except Exception as _dc_err:
                        self._log(
                            f"WARNING: reliability correction failed for "
                            f"'{scale_name_raw}': {_dc_err}"
                        )

                # v1.2.9.3 — MARGINAL SHAPE. The reliability pass above fixes how
                # items relate to each other; this one fixes what a single item
                # looks like on its own. Simulated items come out far too tame:
                # measured across 17 blocks of four published instruments (48,431
                # respondents), real item SD is 0.295 of the scale span with only
                # a 0.019 spread across 5-point and 9-point scales alike, and 34%
                # of all responses sit on an endpoint. A discretised normal puts
                # almost nothing on the endpoints and is peaked where real data is
                # flat (measured excess kurtosis -0.51).
                #
                # The fix rank-transports each item onto a maximum-entropy
                # distribution carrying the item's own mean and the measured
                # dispersion, so every participant keeps their position and the
                # manipulation, the persona structure and the inter-item
                # correlation all survive; only the marginal changes. The target
                # is widened by whatever between-condition variance the column
                # already holds, so a strong manipulation is not squeezed back
                # toward a single-group spread.
                #
                # Gated on the registry: the benchmark declines on any design it
                # was not measured under, and a declined lookup leaves the block
                # exactly as the engine built it.
                if HAS_ITEM_REALISM and HAS_EMPIRICAL_REGISTRY and num_items >= 3:
                    try:
                        _sig = _design_signature.for_block(
                            scale_min=scale_min, scale_max=scale_max,
                            n_items=num_items,
                            design_type=getattr(self, "design_type", None),
                            n_conditions=len(self.conditions or []),
                        )
                        _hit = _empirical_registry.lookup_best(
                            "item.likert.any", "item_sd_fraction_of_span", _sig)
                        _cols = [list(data[c]) for c in item_col_names if c in data]
                        if _hit is not None and len(_cols) == num_items:
                            _md_cols, _md_rep = match_item_dispersion(
                                _cols, int(scale_min), int(scale_max),
                                condition_labels=data.get("CONDITION"),
                                sd_fraction=float(_hit.value),
                                rng=random.Random(int(self.seed) + 0x5D15),
                            )
                            if _md_rep.adjusted_items and not _md_rep.skipped:
                                for j, c in enumerate(item_col_names):
                                    data[c] = [int(v) for v in _md_cols[j]]
                                if self.registry_ledger is not None:
                                    self.registry_ledger.record_lookup(
                                        _hit, f"scale '{scale_name_raw}' item marginals")
                                self._log(
                                    f"Marginal shape for '{scale_name_raw}': item SD "
                                    f"{_md_rep.sd_before:.2f} -> {_md_rep.sd_after:.2f} "
                                    f"(target {_md_rep.target_sd:.2f}), endpoint share "
                                    f"{_md_rep.endpoint_before:.3f} -> "
                                    f"{_md_rep.endpoint_after:.3f} "
                                    f"[{_hit.entry_id}, {_hit.tier}]"
                                )
                                self._item_realism_log.append(
                                    dict(scale=scale_name_raw, stage="marginal",
                                         entry_id=_hit.entry_id, tier=_hit.tier,
                                         **vars(_md_rep))
                                )
                    except Exception as _md_err:
                        self._log(
                            f"WARNING: marginal shape pass failed for "
                            f"'{scale_name_raw}': {_md_err}"
                        )
            # v1.2.9.1: build a deferred user effect into the finished items of this scale
            if scale_name in self._deferred_effect_vars:
                try:
                    _ue_log = self._apply_user_effect_to_scale(
                        data, scale_name, [f"{scale_name}_{j + 1}" for j in range(num_items)],
                        reverse_items, scale_min, scale_max, conditions)
                    if _ue_log:
                        self._deferred_effect_log.append(_ue_log)
                except Exception as _ue_err:  # the requested effect would be missing: say so loudly
                    logger.warning("User effect on '%s' could not be built into the data: %s", scale_name, _ue_err)
                    self._log(f"WARNING: user effect on '{scale_name}' could not be built in: {_ue_err}")
                    self._effect_build_errors.append(f"The expected effect on '{scale_name}' could not be built into the data ({_ue_err}).")
            self._reversal_ok_arr = None

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

        # v1.2.7.x: DV-type-aware post-processing. The per-item loop above emits
        # INDEPENDENT values (correct for Likert/slider/matrix/numbered/single-item).
        # These two helpers fix the cases that need joint structure or a realistic
        # marginal; every other DV type is left byte-identical. Extracted into named
        # methods (v1.2.7.4) to keep generate() readable.
        _typed_dv_scales = self._apply_joint_dv_structure(data, _scale_generation_log, n)
        self._apply_numeric_distribution_realism(data, _scale_generation_log)

        # =====================================================================
        # v1.4.3: COMPOSITE MEAN COLUMNS FOR MULTI-ITEM SCALES
        # For each scale with > 1 item, compute a _mean column as the row-wise
        # average across all items. This is the standard composite score used
        # in behavioral science analysis.
        # =====================================================================
        for log_entry in _scale_generation_log:
            item_cols = log_entry["columns_generated"]
            if item_cols and item_cols[0].rsplit("_", 1)[0] in _typed_dv_scales:
                continue  # rank-order/constant-sum: a row-mean is not a meaningful composite
            if len(item_cols) > 1:
                # v1.2.0.8: Validate all columns exist and have n rows before computing mean.
                # If generation was interrupted, some columns may be missing or incomplete.
                if any(col not in data for col in item_cols):
                    _missing = [c for c in item_cols if c not in data]
                    self._log(f"WARNING: Skipping composite mean — missing columns: {_missing}")
                    continue
                if any(len(data[col]) < n for col in item_cols):
                    self._log(f"WARNING: Skipping composite mean — some columns have < {n} rows")
                    continue
                # Compute row-wise mean across all items for this scale
                # Reverse-coded items are exported RAW (as in a real Qualtrics
                # export) but the composite is a SCORED scale: recode them
                # (min + max - x) before averaging, otherwise items pointing in
                # opposite directions cancel and the composite stops measuring
                # the construct (and under-recovers configured effects).
                _rev = set(log_entry.get("reverse_items") or [])
                _flip = log_entry["scale_min"] + log_entry["scale_max"]
                mean_values: List[float] = []
                for i in range(n):
                    item_sum = sum(
                        (_flip - data[col][i]) if (j + 1) in _rev else data[col][i]
                        for j, col in enumerate(item_cols)
                    )
                    mean_values.append(round(item_sum / len(item_cols), 2))
                # Derive clean composite column name from the first item column
                # e.g., "Trust_1" -> "Trust_mean"
                _prefix = item_cols[0].rsplit("_", 1)[0]
                mean_col_name = f"{_prefix}_mean"
                data[mean_col_name] = mean_values
                scale_raw_name = log_entry["name"]
                self.column_info.append(
                    (mean_col_name, f"{scale_raw_name} composite mean ({log_entry['scale_min']}-{log_entry['scale_max']})"
                     + (f"; reverse-coded items {sorted(_rev)} recoded before averaging" if _rev else ""))
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
        # v1.2.2.9: EXCEPTION — when free_llm_oe_cap > 0, the user chose "Proceed
        # (AI for 100, template for rest)".  We MUST prefill so the first 100
        # participants have pool responses ready.
        if (self.llm_generator and self.open_ended_questions
                and self.llm_attempts_allowed()):
            try:
                self.llm_generator.reset_providers()
                self._log("LLM providers reset before prefill (clean state)")
            except Exception as _rst_err:
                logger.warning("LLM provider reset before prefill failed: %s", _rst_err)
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
                    # v1.2.9.1: numeric text boxes are answered with numbers, not LLM text.
                    if _infer_numeric_answer_spec(oq.get("question_text", ""), oq.get("name", ""), oq) is not None:
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
                    # v1.2.7.7: Fast bail-out — if the free tier is saturated right now
                    # (several consecutive all-throttled batches), don't burn the rest of
                    # the prefill budget; switch to templates so the user isn't left
                    # waiting. (Recoverable signal; distinct from the permanent kill switch.)
                    if getattr(self.llm_generator, "free_tier_exhausted_now", False):
                        self._log("LLM prefill: free tier throttled/exhausted right now — "
                                  "switching to template fallback to avoid a long wait")
                        _prefill_timed_out = True
                        break
                    _q_raw = str(oq.get("question_text", oq.get("name", "")))
                    _q_ctx = str(oq.get("question_context", "")).strip()
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
                            # v1.2.7.9 (H1 fix): Build the enriched text via the SHARED
                            # helper, per-condition, so the prefill pool key is
                            # byte-identical to the per-participant draw key. Previously
                            # this path omitted the \nCondition:/\nAdditional context:
                            # suffixes the runtime path adds, so every draw missed.
                            _q_text = self._build_enriched_question_text(_q_raw, _q_ctx, _cond)
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
            # v1.2.2.8: Per-question LLM OE cap counter — tracks how many
            # participants in THIS question got an LLM-generated response.
            # When it reaches free_llm_oe_cap (e.g. 100), LLM is disabled
            # and remaining participants get template fallback.
            _llm_oe_cap_count = 0
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
            # v1.2.2.8: Do NOT re-enable if disabled by free_llm_oe_cap — the cap
            # is cumulative across ALL questions, not per-question.
            _disabled_by_cap = getattr(self, '_llm_disabled_by_oe_cap', False)
            if (self.llm_generator
                    and getattr(self.llm_generator, '_force_disabled', False)
                    and not _disabled_by_cap):
                self._log(f"OE Q{_oe_idx+1}/{_total_oe}: Re-enabling LLM (was force-disabled by previous question)")
                self.llm_generator._force_disabled = False
                self.llm_generator._api_available = True
            # v1.2.7.9 (M1 fix): Give each NEW OE question a fresh transient-throttle
            # budget. free_tier_exhausted_now latches once _consecutive_transient_batches
            # hits the threshold and — because every caller then skips _generate_batch —
            # it can never reset itself mid-run. Clearing it per-question lets a free
            # tier that recovered between questions be used again, while the cumulative
            # OE cap (_disabled_by_cap) still wins so we never over-spend the budget.
            if (self.llm_generator
                    and not _disabled_by_cap
                    and getattr(self.llm_generator, '_consecutive_transient_batches', 0)):
                self.llm_generator._consecutive_transient_batches = 0
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
                    elif any(kw in _demo_search_text for kw in ("zip", "postal", "postcode")):
                        _demo_responses.append(f"{int(_d_rng.randint(10000, 99999)):05d}")
                    elif any(kw in _demo_search_text for kw in ("state", "location", "country", "city")):
                        _us_states = ["California", "Texas", "Florida", "New York",
                                      "Pennsylvania", "Illinois", "Ohio", "Georgia",
                                      "North Carolina", "Michigan", "Other"]
                        _demo_responses.append(_d_rng.choice(_us_states))
                    else:
                        # Generic demographic — generate short factual answers
                        _demo_responses.append(str(int(np.clip(_d_rng.normal(40, 15), 1, 99))))
                data[col_name] = _demo_responses
                self._structured_oe_columns.add(col_name)
                self._log(f"Generated demographic data for '{col_name}' ({n} values)")
                continue  # Skip normal OE text generation

            # v1.2.9.1: numeric text boxes ("How many tickets...? (enter a number)", "year of birth")
            # get numbers, honoring Qualtrics validation ranges and survey-flow visibility. They used
            # to be answered with essays.
            try:
                _numeric_spec = _infer_numeric_answer_spec(q_text, col_name, q)
            except Exception as _num_err:
                logger.warning("Numeric-answer detection failed for '%s': %s", col_name, _num_err)
                _numeric_spec = None
            if _numeric_spec is not None:
                _num_responses: List[str] = []
                for _ni in range(n):
                    _report_progress("generating", _ni, n)
                    if not self.survey_flow_handler.is_question_visible(col_name, conditions.iloc[_ni]):
                        _num_responses.append("")
                        continue
                    _n_rng = np.random.RandomState((self.seed + _ni * 100 + col_hash) % (2**31))
                    _num_responses.append(_draw_numeric_answer(_numeric_spec, _n_rng))
                data[col_name] = _num_responses
                self._structured_oe_columns.add(col_name)
                self._log(f"Generated numeric answers for '{col_name}' ({_numeric_spec.get('kind')}, {n} values)")
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
                            except Exception as _dp_err:
                                logger.warning("disable_permanently() failed on budget exceed: %s", _dp_err)

                participant_condition = conditions.iloc[i]

                # Check if this participant's condition allows them to see this question
                if not self.survey_flow_handler.is_question_visible(col_name, participant_condition):
                    # Participant wouldn't see this question - leave blank (NA)
                    responses.append("")
                    # v1.2.2.3: Must also append to _sources_for_col to keep lists aligned.
                    # Without this, len(responses) != len(_sources_for_col), corrupting
                    # the _Generation_Source column downstream (wrong participant mapping).
                    _sources_for_col.append("N/A")
                    continue

                # Generate unique seed using participant, question, and question text hash
                # v1.0.0 CRITICAL FIX: Use stable hash instead of Python's hash()
                # v1.3.3: Use MD5 of full text to avoid collisions on similar questions
                _q_hash_input = (q_text + col_name).encode('utf-8', errors='replace')
                q_text_hash = int(hashlib.md5(_q_hash_input).hexdigest()[:8], 16)
                p_seed = (self.seed + i * 100 + col_hash + q_text_hash) % (2**31)
                persona_name = assigned_personas[i] if i < len(assigned_personas) else "engaged"
                # v1.2.0.8: Safe lookup with empty-pool guard. next(iter({})) raises StopIteration.
                persona = self.available_personas.get(persona_name)
                if persona is None:
                    persona = next(iter(self.available_personas.values()), None) if self.available_personas else None
                if persona is None:
                    from .persona_library import Persona
                    persona = Persona(name="default", description="Default responder", weight=1.0, traits={})
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
                        except Exception as _dp_err2:
                            logger.warning("disable_permanently() failed on slow participants: %s", _dp_err2)
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

                # v1.2.2.8: Free-tier LLM OE cap enforcement.
                # After N participants have gotten LLM responses for this question,
                # disable LLM so remaining participants get template fallback.
                # This protects free API tokens while keeping the full sample size.
                if _participant_source == "AI":
                    _llm_oe_cap_count += 1
                if (self.free_llm_oe_cap > 0
                        and _llm_oe_cap_count >= self.free_llm_oe_cap
                        and not _oe_budget_exceeded
                        and self.llm_generator
                        and not getattr(self.llm_generator, '_force_disabled', False)):
                    self._log(
                        f"Free LLM OE cap reached: {_llm_oe_cap_count}/{self.free_llm_oe_cap} "
                        f"AI responses for question '{col_name}' — "
                        f"remaining {n - i - 1} participants will use template fallback"
                    )
                    try:
                        self.llm_generator.disable_permanently(
                            f"Free LLM OE cap ({self.free_llm_oe_cap}) reached"
                        )
                    except Exception as _cap_err:
                        logger.warning("disable_permanently() failed on OE cap: %s", _cap_err)
                    # Mark that the cap (not budget/timeout) caused the disable,
                    # so between-question re-enable logic won't undo it.
                    self._llm_disabled_by_oe_cap = True
                    _oe_budget_exceeded = True
                    # Notify UI via progress callback
                    _report_progress(
                        "oe_cap_reached", _llm_oe_cap_count, self.free_llm_oe_cap
                    )

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
                except Exception as _stats_err:
                    logger.debug("Failed to retrieve LLM stats for budget log: %s", _stats_err)
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
                        # v1.2.2.4: Skip "N/A" entries (invisible questions) so they
                        # don't inflate Template count and corrupt the source label.
                        _src_val = _src_list[_pi]
                        if _src_val == "N/A":
                            continue
                        _total_count += 1
                        if _src_val == "AI":
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
        # v1.2.2.3: Use _completed_oe_columns (actual names after any renames) instead
        # of re-deriving names from questions.  Previous code missed renamed columns
        # (e.g., "Age" → "OE_Age") causing duplicate detection to silently skip them.
        open_ended_cols = []
        for _oec in _completed_oe_columns:
            if (_oec in data and isinstance(data[_oec], list)
                    and len(data[_oec]) > 0 and isinstance(data[_oec][0], str)):
                open_ended_cols.append(_oec)
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
                    # v1.2.7.5: _stable_int_hash (not salted built-in hash()) so the
                    # chosen modifier is reproducible across processes for a given seed.
                    modifier_idx = (i + _stable_int_hash(dup_col) % 100) % len(modifiers)
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
                if col in getattr(self, "_typed_dv_columns", set()):
                    continue  # v1.2.7.0: joint-constrained DV (rank-order/constant-sum) —
                              # per-cell clipping would break the permutation/total
                col_min = issue["expected_min"]
                col_max = issue["expected_max"]
                # Auto-correct out-of-bounds values (preserve NaN from missing data)
                col_series = df[col]
                mask = col_series.notna()
                if mask.any():
                    df.loc[mask, col] = col_series[mask].clip(lower=col_min, upper=col_max).astype(int)
                self._log(f"  Corrected {col}: clipped to [{col_min}, {col_max}]")

        # =================================================================
        # ABE 3.0: CONSISTENCY IMPROVEMENT #5 — Post-Generation Audit & Repair
        # Validates achieved within-person consistency and repairs violations.
        # =================================================================
        _consistency_audit = self._audit_individual_consistency(
            df, data, all_traits, _scale_generation_log,
        )
        if _consistency_audit.get("repairs_performed", 0) > 0:
            self._log(
                f"ABE 3.0 consistency audit: {_consistency_audit['repairs_performed']} "
                f"repair(s) performed across {_consistency_audit['total_checks']} checks"
            )
            # Re-create DataFrame with repaired data
            df = pd.DataFrame(data)

        # v1.2.9.1 — identical-answer realism, deliberately LAST.
        # Generation leaves about 11% of respondents answering identically across
        # a 5-item block; the consistency audit above then repairs nearly all of
        # them away, leaving 0.3%. Real data sits in between: 5.2% of respondents
        # in the reference sample give the same answer to every direction-aligned
        # item of a 5-item block (utils/reference_profiles.json), because people at
        # the ceiling of a construct genuinely answer "6,6,6,6,6". Restoring that
        # share has to happen after every pass that would undo it.
        # Only the touched columns are written back. Rebuilding the frame from
        # `data` here would silently discard every df-level correction made above
        # (the range clipping, for one) — which measured as a drop in the observed
        # treatment effect from d=1.41 to d=0.34 before this was caught.
        self._apply_identical_answer_realism(data, _scale_generation_log, frame=df)

        # Final consistency pass: every <Scale>_mean must equal the mean of the delivered
        # (reverse-recoded, missing-aware) items, whatever the steps above did to them.
        # Runs after the identical-answer pass, which rewrites items.
        self._refresh_scale_composites(df, _scale_generation_log)

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
                    "level_high": e.get("level_high", "") if isinstance(e, dict) else getattr(e, "level_high", ""),
                    "level_low": e.get("level_low", "") if isinstance(e, dict) else getattr(e, "level_low", ""),
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
            # ABE 3.0: Individual-level consistency audit results
            "consistency_audit": _consistency_audit,
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
            # v1.2.9.1: free-text boxes that repeated a numeric DV already in the data
            "open_ended_duplicates_of_numeric_dvs": list(
                getattr(self, "_oe_dropped_as_dv_duplicates", []) or []
            ),
            # v1.4.6: LLM response generation stats
            # v1.0.6.1: Guard against .stats being None
            "llm_response_stats": (getattr(self.llm_generator, 'stats', None) or {"llm_calls": 0, "fallback_uses": 0}) if self.llm_generator else {"llm_calls": 0, "fallback_uses": 0},
            "llm_init_error": self.llm_init_error,
            # v1.1.1.4: OE generation budget tracking for transparent user reporting
            "oe_budget_exceeded": getattr(self, '_oe_budget_exceeded', False),
            "oe_budget_switched_count": getattr(self, '_oe_budget_switched_count', 0),
            # v1.2.2.8: Whether the free LLM OE cap was the cause of the fallback
            "free_llm_oe_cap_reached": getattr(self, '_llm_disabled_by_oe_cap', False),
            "free_llm_oe_cap": self.free_llm_oe_cap,
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
                    # v1.2.0.8: Null-check socsim_meta — run_socsim_enrichment() may return None
                    if isinstance(socsim_meta, dict):
                        metadata["socsim"] = socsim_meta
                        self._log(f"SocSim enrichment complete: {len(socsim_meta.get('enriched_dvs', []))} DVs enriched")
                        try:
                            self._reapply_user_effects_after_game_model(df, socsim_meta)
                        except Exception as _reapply_err:
                            self._log(f"SocSim: could not re-apply requested effects: {_reapply_err}")
                    else:
                        metadata["socsim"] = {"socsim_used": False, "error": "Invalid return from enrichment"}
                        self._log("SocSim enrichment returned invalid metadata")
                else:
                    self._log("SocSim: No game-theory DVs detected — running standard simulation only")
                    metadata["socsim"] = {"socsim_used": False, "reason": "no_game_dvs_detected"}
            except Exception as e:
                self._log(f"SocSim enrichment failed (non-fatal): {e}")
                metadata["socsim"] = {"socsim_used": False, "error": str(e)}

        # =================================================================
        # ABE 3.0: Integrated HBS Post-Processing Pipeline
        # Census demographics, stylometric fingerprinting, validation.
        # =================================================================
        _abe3_start = time.time() if 'time' in dir() else 0
        try:
            import time as _time_mod
            _abe3_start = _time_mod.time()
        except Exception:
            _abe3_start = 0

        # Step A: Census-weighted demographics enrichment
        _hbs_participant_states = []
        if HAS_HBS_DEMOGRAPHICS:
            try:
                _hbs_factory = HBSParticipantFactory(seed=self.seed)
                _domain = self.detected_domains[0] if self.detected_domains else ""
                _hbs_participant_states = _hbs_factory.create_batch(
                    n=n, conditions=self.conditions, domain=_domain,
                )
                if _hbs_participant_states:
                    _n_states = len(_hbs_participant_states)
                    _get_st = lambda idx: _hbs_participant_states[idx % _n_states]
                    _hbs_demo_cols = {
                        "ABE3_Education": [_get_st(i).education_level for i in range(n)],
                        "ABE3_Income": [_get_st(i).income_bracket for i in range(n)],
                        "ABE3_PartyID": [_get_st(i).party_id for i in range(n)],
                        "ABE3_Ideology": [round(_get_st(i).ideology, 2) for i in range(n)],
                        "ABE3_State": [_get_st(i).state for i in range(n)],
                        "ABE3_Region": [_get_st(i).region for i in range(n)],
                        "ABE3_ResponseStyle": [_get_st(i).response_style for i in range(n)],
                    }
                    for col_name, col_data in _hbs_demo_cols.items():
                        if col_name not in df.columns:
                            df[col_name] = col_data
                    self.column_info.extend([
                        ("ABE3_Education", "Census-weighted education level"),
                        ("ABE3_Income", "Census-weighted income bracket"),
                        ("ABE3_PartyID", "7-point party identification (ANES)"),
                        ("ABE3_Ideology", "Ideology score (-3.0 liberal to +3.0 conservative)"),
                        ("ABE3_State", "U.S. state (2-letter code)"),
                        ("ABE3_Region", "U.S. region (Northeast/South/Midwest/West)"),
                        ("ABE3_ResponseStyle", "Response style (Krosnick taxonomy)"),
                    ])
                    self._log(f"ABE 3.0: Enriched with 7 census-weighted demographic columns")
            except Exception as _demo_err:
                self._log(f"ABE 3.0: Census demographics enrichment skipped: {_demo_err}")

        # Step B: Stylometric fingerprinting for OE columns
        if HAS_HBS_STYLOMETRIC and _hbs_participant_states:
            try:
                _stylo_engine = HBSStylometricEngine()
                # v1.2.5.1: Use centralized OE detection with known OE names
                from utils import detect_oe_columns as _detect_oe
                _known_oe_names = set()
                for oeq in self.open_ended_questions:
                    if isinstance(oeq, dict):
                        _known_oe_names.add(oeq.get("variable_name", ""))
                        _known_oe_names.add(oeq.get("name", ""))
                _known_oe_names.discard("")
                _oe_cols = _detect_oe(df, known_oe_names=_known_oe_names or None)
                _oe_cols = [c for c in _oe_cols if c not in self._structured_oe_columns]
                if _oe_cols:
                    _applied = 0
                    _n_states = len(_hbs_participant_states)
                    for i in range(n):
                        _state = _hbs_participant_states[i % _n_states]
                        _fp_rng = random.Random(self.seed + i * 13)
                        _fp = _stylo_engine.build_fingerprint(_state, _fp_rng)
                        for col in _oe_cols:
                            _text = df.at[i, col] if i in df.index else None
                            if isinstance(_text, str) and len(_text) > 10:
                                _new_text = _stylo_engine.apply_fingerprint(
                                    _text, _fp, _fp_rng,
                                )
                                df.at[i, col] = _new_text
                                _applied += 1
                    self._log(f"ABE 3.0: Applied stylometric fingerprints ({_applied} cells)")
            except Exception as _stylo_err:
                self._log(f"ABE 3.0: Stylometric fingerprinting skipped: {_stylo_err}")

        # Step C: Adversarial self-validation + auto-correction
        _validation_report: Dict[str, Any] = {}
        if HAS_HBS_VALIDATOR:
            try:
                # v1.2.8.4: seed the validator per-run so its perturbations are
                # reproducible WITHOUT seeding the process-global RNG (which let us
                # drop the run-wide _GLOBAL_RNG_LOCK that serialized concurrent users).
                _validator = HBSValidator(seed=self.seed, protected_columns=self._structured_oe_columns)
                df, _validation_report = _validator.validate_and_correct(df)
                self._log(f"ABE 3.0: Validation complete — {_validation_report.get('summary', 'ok')}")
            except Exception as _val_err:
                self._log(f"ABE 3.0: Validation skipped: {_val_err}")

        # Step C2 (v1.2.9.1): last mechanical tidy of every open-ended cell (spacing, a/an
        # agreement, misplaced fillers, cut-off endings) after the stylometric and validation
        # passes, which can both leave such artifacts behind.
        try:
            from utils import detect_oe_columns as _detect_oe_final
            from .text_cleanup import finalize_generated_text as _finalize_oe_text
            _known_final = set()
            for _oeq in self.open_ended_questions:
                if isinstance(_oeq, dict):
                    _known_final.add(str(_oeq.get("variable_name", "") or ""))
                    _known_final.add(str(_oeq.get("name", "") or ""))
            _known_final.discard("")
            for _col in _detect_oe_final(df, known_oe_names=_known_final or None):
                if _col in self._structured_oe_columns:
                    continue
                for _idx in df.index:
                    _val = df.at[_idx, _col]
                    if isinstance(_val, str) and len(_val) > 10:
                        _tidy = _finalize_oe_text(_val)
                        if _tidy != _val:
                            df.at[_idx, _col] = _tidy
        except Exception as _tidy_err:
            self._log(f"ABE 3.0: Final text tidy skipped: {_tidy_err}")

        # The validator may perturb item values: re-derive composites from the final items.
        self._refresh_scale_composites(df, getattr(self, "_scale_generation_log", None) or [])

        # Step D: Add ABE 3.0 metadata
        try:
            import time as _time_mod
            _abe3_elapsed = _time_mod.time() - _abe3_start
        except Exception:
            _abe3_elapsed = 0

        metadata["abe3_engine"] = {
            "enabled": True,
            "version": "3.0.0",
            "census_demographics_active": HAS_HBS_DEMOGRAPHICS and len(_hbs_participant_states) > 0,
            "stylometric_engine_active": HAS_HBS_STYLOMETRIC,
            "validator_active": HAS_HBS_VALIDATOR,
            "error_calibrator_active": False,  # module may import, but nothing in the generation path calls it
            "question_classifier_active": False,  # module may import, but nothing in the generation path calls it
            "validation_report": _validation_report,
            "abe3_processing_time_seconds": round(_abe3_elapsed, 2),
            "consistency_improvements": [
                "survey_fatigue_drift",
                "demographic_style_coupling",
                "3d_latent_attitude_vector",
                "response_pattern_inertia",
                "post_generation_audit_repair",
            ],
        }

        # Override generation method labels for ABE 3.0
        metadata["generation_method"] = "abe_v3"
        metadata["generation_method_label"] = "Adaptive Behavioral Engine 3.0"

        # Set generation source column.
        # v1.2.7.7 BUGFIX: do NOT clobber the per-participant AI/Template/Mixed
        # provenance that the OE loop already computed (it was being overwritten
        # with a hardcoded "Non-LLM" label for EVERY row — so LLM-generated open-
        # ended text was mislabeled as Non-LLM, making it look like the free LLM
        # never ran even when it produced every response). Numeric data always
        # comes from ABE 3.0; the open-ended TEXT source is what this column tracks.
        _existing_src = df["_Generation_Source"] if "_Generation_Source" in df.columns else None
        _has_real_provenance = (
            _existing_src is not None
            and _existing_src.astype(str).isin(["AI", "Template", "Mixed"]).any()
        )
        if _has_real_provenance:
            # Map the per-participant OE source to a clear label; keep ABE 3.0 as
            # the numeric engine, annotate the open-ended text source.
            _label_map = {
                "AI": "ABE 3.0 numeric + AI open-ended",
                "Mixed": "ABE 3.0 numeric + mixed AI/template open-ended",
                "Template": "Adaptive Behavioral Engine 3.0 (Non-LLM)",
                "N/A": "Adaptive Behavioral Engine 3.0 (Non-LLM)",
            }
            df["_Generation_Source"] = _existing_src.astype(str).map(
                lambda s: _label_map.get(s, "Adaptive Behavioral Engine 3.0 (Non-LLM)"))
        else:
            df["_Generation_Source"] = "Adaptive Behavioral Engine 3.0 (Non-LLM)"

        _report_progress("complete", n, n)
        # Final authoritative reconciliation: no routine that edits item values after
        # the composites were built (repairs, jitter, enrichment) may leave a scale
        # mean inconsistent with its own items in the exported data.
        try:
            self._reconcile_composites(df)
        except Exception as _rec_err:
            self._log(f"WARNING: composite reconciliation failed: {_rec_err}")

        # v1.2.9.1: the observed-effect summary was computed before the game-model
        # enrichment, the validator and the final reconciliation; recompute it from the
        # data that is actually returned and describe what effect was built in.
        try:
            metadata["effect_sizes_observed"] = self._compute_observed_effect_sizes(df)
            metadata["effect_sizes_applied"] = self._build_effects_applied(metadata["effect_sizes_observed"])
        except Exception as _eff_err:
            self._log(f"WARNING: final effect summary refresh skipped: {_eff_err}")

        return df, metadata

    def _effect_spec_diagnostics(self) -> Dict[str, Any]:
        """Check every user-specified effect against the generated variables and conditions.

        Returns ``{"specs": [...], "warnings": [...]}``. A spec whose variable is not generated, or
        whose levels name no condition, builds nothing into the data (the observed d then shows
        nothing but sampling noise); a spec where only one level names a condition moves one arm only,
        so the contrast is about half the requested d. Reporting both is the only protection against a
        requested effect that silently went missing. Each row carries ``matched`` (the effect reached
        at least one condition of a generated variable), ``status`` ("applied", "one_side_only",
        "levels_not_found" or "variable_not_found") and the variables and conditions it reached.
        """
        columns = list(self._variable_alias_map().keys())
        conditions = [(str(c), _label_norm(c)) for c in (self.conditions or [])]
        rows: List[Dict[str, Any]] = []
        warns: List[str] = []
        for effect in getattr(self, "effect_sizes", None) or []:
            variable = str(_spec_get(effect, "variable", "") or "")
            level_high = str(_spec_get(effect, "level_high", "") or "")
            level_low = str(_spec_get(effect, "level_low", "") or "")
            variables = [c for c in columns if self._spec_applies_to_variable(variable, c)]
            high_conditions = [c for c, norm in conditions if _spec_side(effect, norm) > 0]
            low_conditions = [c for c, norm in conditions if _spec_side(effect, norm) < 0]
            if not variables:
                status = "variable_not_found"
                warns.append(
                    f"Expected effect on '{variable}' was NOT applied: no generated variable has that name "
                    f"(variables: {', '.join(columns) if columns else 'none'})."
                )
            elif not high_conditions and not low_conditions:
                status = "levels_not_found"
                warns.append(
                    f"Expected effect on '{variable}' ('{level_high}' vs '{level_low}') was NOT applied: "
                    f"neither level matches a condition name (conditions: {', '.join(c for c, _ in conditions)})."
                )
            elif not high_conditions or not low_conditions:
                status = "one_side_only"
                missing = level_high if not high_conditions else level_low
                warns.append(
                    f"Expected effect on '{variable}' ('{level_high}' vs '{level_low}') was applied to one side only: "
                    f"'{missing}' matches no condition name, so the contrast against the other conditions is "
                    f"about half the requested d."
                )
            else:
                status = "applied"
            raw_d = _spec_get(effect, "cohens_d", None)
            try:
                d_value: Optional[float] = float(raw_d)
            except (TypeError, ValueError):
                d_value = None
            rows.append({
                "variable": variable, "level_high": level_high, "level_low": level_low, "cohens_d": d_value,
                "matched": status in ("applied", "one_side_only"), "status": status,
                "matched_variables": variables, "high_conditions": high_conditions, "low_conditions": low_conditions,
            })
        return {"specs": rows, "warnings": warns}

    def _check_generation_warnings(self, df: pd.DataFrame) -> List[str]:
        """Return any warnings about the generated data quality."""
        warnings: List[str] = []
        # v1.2.9.1: an expected effect that reached no variable or condition must not go missing silently
        warnings.extend(self._effect_spec_diagnostics()["warnings"])
        warnings.extend(getattr(self, "_effect_build_errors", []) or [])
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
                    # v1.2.9.1: missing cells are NaN; min/max/mean over them gave NaN, which
                    # Metadata.json wrote as the invalid JSON token "NaN" (49 of 150 designs).
                    col_values = df[col_name].dropna().tolist()
                    if not col_values:
                        continue
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
                _obs_min, _obs_max = min(all_values), max(all_values)
                # columns holding missing cells are float typed: keep whole numbers whole, as before
                scale_report["observed_min"] = int(_obs_min) if float(_obs_min).is_integer() else float(_obs_min)
                scale_report["observed_max"] = int(_obs_max) if float(_obs_max).is_integer() else float(_obs_max)
                scale_report["observed_mean"] = round(sum(all_values) / len(all_values), 2)

                # Flag if range utilization is suspiciously low for large scales
                if spec_points > 10 and utilization < 30 and len(all_values) >= 20:
                    scale_report["status"] = "LOW_RANGE_UTILIZATION"

            report.append(scale_report)

        return report

    def _reconcile_composites(self, df: pd.DataFrame) -> int:
        """Recompute every ``<scale>_mean`` composite from the final item columns.

        Reverse-keyed items are recoded (min + max - x) before averaging and missing
        cells are skipped. Returns the number of rows whose composite was corrected.
        """
        fixed = 0
        for le in (getattr(self, "_scale_generation_log", None) or []):
            cols = [c for c in (le.get("columns_generated") or []) if c in df.columns]
            if len(cols) < 2:
                continue
            mcol = f"{cols[0].rsplit('_', 1)[0]}_mean"
            if mcol not in df.columns:
                continue
            allcols = le.get("columns_generated") or []
            rev = set(le.get("reverse_items") or [])
            flip = float(le.get("scale_min", 0)) + float(le.get("scale_max", 0))
            M = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
            for j, c in enumerate(cols):
                if (allcols.index(c) + 1) in rev:
                    M[:, j] = flip - M[:, j]
            with np.errstate(all="ignore"):
                new = np.round(np.nanmean(M, axis=1), 2)
            old = pd.to_numeric(df[mcol], errors="coerce").to_numpy(dtype=float)
            diff = ~np.isclose(np.nan_to_num(old, nan=-999.0), np.nan_to_num(new, nan=-999.0), atol=0.011)
            if diff.any():
                df[mcol] = new
                fixed += int(diff.sum())
        if fixed:
            self._log(f"Reconciled {fixed} composite value(s) with their final item values")
        return fixed

    def _defer_user_effect_for_scale(self, scale_name: str, scale_min: int, scale_max: int, has_reverse: bool) -> bool:
        """Whether this scale's user-specified effect is built into the finished item responses.

        True for a scale that carries the cross-scale latent term (several scales in the design) and
        has a user effect. Not for knowledge-base economic-game outcomes: their generator shifts a latent
        quantile of the published outcome distribution (keeping its spikes at zero and at an even
        split) and never receives the latent term, so the ordinary route is exact for them.
        """
        if scale_name not in getattr(self, "_latent_dv_names", ()) or scale_max <= scale_min:
            return False
        if not self._variable_has_user_spec(scale_name):
            return False
        if not has_reverse and (scale_max - scale_min) >= 1:
            geometry = self._detect_scale_geometry(scale_min, scale_max, scale_name)
            calibration = self._get_domain_response_calibration(scale_name, "")
            _kd = calibration.get("_kb_dist")
            _route = ((scale_max - scale_min) >= 10) or (
                (scale_max - scale_min) == 1 and _binary_game_rate(_kd) is not None)   # v1.3.0.6: binary route
            if (_kd and _route and not geometry["is_bipolar"]
                    and calibration.get("_game_variant") not in ("dictator_taking", "dictator_third_party")):
                return False
        return True

    def _apply_user_effect_to_scale(
        self,
        data: Dict[str, list],
        scale_name: str,
        item_cols: List[str],
        reverse_items: Set[int],
        scale_min: int,
        scale_max: int,
        conditions: "pd.Series",
    ) -> Optional[Dict[str, Any]]:
        """Build a user-specified effect into a finished scale, in units of its own within-condition SD.

        Why: next to other scales every response carries the cross-scale latent term, which makes the
        composite's SD about 3x as large as for a lone scale and lets floor and ceiling swallow part of a
        tendency shift, so a shift calibrated for a lone scale produced only ~0.4 of the requested d (no
        matter how many scales, or how related). The requested d is a statement about the composite's
        SD, so the shift is applied to the generated responses and sized from that SD.

        How: the scale was generated without any condition shift. The pooled within-condition SD of its
        scored composite is measured; each condition's participants are then moved by their target
        (+/- d/2 SDs for the two arms of a spec, 0 for a reference condition) with randomised rounding,
        so answers stay integers and a fractional shift moves the mean by exactly that fraction.
        Floor and ceiling swallow part of the move, so each condition's multiplier is re-aimed until
        the composite MEAN moved by its target. Only that applied move is controlled, never the realised
        gap between arms: the arm differences already present by chance in the generated data remain, so
        the observed d keeps its ordinary sampling variability around the request.
        Reverse-keyed items move against the item direction, except for respondents who failed to
        reverse them (recorded while generating), who move with it: the attenuation that careless reverse
        responding causes in real data is kept, exactly as for a lone scale.

        Returns a small log row (also written to ``effect_sizes_applied``) or None when nothing was to
        be done.
        """
        applied = getattr(self, "_applied_effects", None) or {}
        targets: Dict[str, float] = {}
        for (cond, var), info in applied.items():
            if var != scale_name or info.get("source") != "user":
                continue
            unit = float(info.get("unit") or 0.0)
            if unit > 0:
                targets[str(cond)] = float(info.get("offset", 0.0)) / (2.0 * unit)
        targets = {c: t for c, t in targets.items() if abs(t) > 1e-9}
        if not targets:
            return None                                      # nothing was requested for this scale

        def skipped(reason: str) -> None:
            """The requested effect cannot be built in: say so instead of returning clean data."""
            self._effect_build_errors.append(f"The expected effect on '{scale_name}' was not applied: {reason}.")
            return None

        cols = [c for c in item_cols if c in data]
        n = len(conditions)
        if not cols or scale_max <= scale_min or any(len(data[c]) != n for c in cols):
            return skipped("the item columns are missing or inconsistent")
        if n < 2:
            return skipped("fewer than two participants")
        k = len(cols)
        lo, hi = float(scale_min), float(scale_max)
        flip = lo + hi
        X = np.array([data[c] for c in cols], dtype=float).T
        rev = np.array([(j + 1) in reverse_items for j in range(k)], dtype=bool)
        sign = np.where(rev, -1.0, 1.0)                      # direction of an item in the scored composite
        ok = np.ones((n, k), dtype=bool)
        rec = getattr(self, "_reversal_ok_arr", None)
        if rec is not None and rec[0] == scale_name and rec[1].shape == (n, k):
            ok = rec[1]
        move = np.ones((n, k))                                # raw-answer direction of a construct-direction shift
        if rev.any():
            move[:, rev] = np.where(ok[:, rev], -1.0, 1.0)

        def composite(M: np.ndarray) -> np.ndarray:
            return (M * sign + flip * rev).mean(axis=1)

        cond_arr = np.asarray(conditions.to_numpy() if hasattr(conditions, "to_numpy") else conditions, dtype=object)
        groups = {c: cond_arr == c for c in dict.fromkeys(cond_arr.tolist())}
        arms = {c: groups[c] for c in targets if c in groups and groups[c].any()}
        if not arms:
            return skipped("no participant was assigned to a condition named in the effect")

        def pooled_sd(comp: np.ndarray) -> float:
            num = den = 0.0
            for m in groups.values():
                if m.sum() > 1:
                    num += float(comp[m].var(ddof=1)) * (int(m.sum()) - 1)
                    den += int(m.sum()) - 1
            return float(np.sqrt(num / den)) if den > 0 and num > 0 else 0.0

        comp0 = composite(X)
        sd = pooled_sd(comp0)
        if sd <= 0:
            return skipped("the scale shows no variation within conditions")
        shrink = {c: float((sign * move)[m].mean()) for c, m in arms.items()}   # reverse-item failures
        u = np.random.RandomState((int(self.seed) + _stable_int_hash(f"{scale_name}|user_effect")) % (2**31)).random_sample((n, k))
        mult = {c: 1.0 for c in arms}
        result, moved, iterations = X, {}, 0
        for iterations in range(1, 15):
            result = X.copy()
            for c, m in arms.items():
                result[m] = np.clip(np.floor(X[m] + move[m] * (mult[c] * targets[c] * sd) + u[m]), lo, hi)
            comp = composite(result)
            sd = pooled_sd(comp) or sd
            moved = {c: float(comp[m].mean() - comp0[m].mean()) for c, m in arms.items()}
            wanted = {c: targets[c] * sd * shrink[c] for c in arms}
            if all(abs(moved[c] - wanted[c]) <= 0.004 * sd for c in arms):
                break
            for c in arms:
                if moved[c] * wanted[c] > 1e-12:
                    mult[c] = float(np.clip(mult[c] * wanted[c] / moved[c], 0.2, 8.0))
                elif abs(wanted[c]) > 1e-12:
                    mult[c] = float(np.clip(mult[c] * 2.0, 0.2, 8.0))
        for j, col in enumerate(cols):
            data[col] = result[:, j].astype(int).tolist()
        log: Dict[str, Any] = {
            "variable": scale_name, "method": "applied to the finished item responses",
            "targets_sd": {c: round(t, 4) for c, t in targets.items()},
            "within_condition_sd": round(float(sd), 4), "iterations": int(iterations),
            "multiplier": {c: round(v, 3) for c, v in mult.items()},
        }
        if rev.any():
            # share of the requested move that survives careless reverse-item answering, per condition
            log["reverse_key_attenuation"] = {c: round(v, 3) for c, v in shrink.items()}
        return log

    def _reapply_user_effects_after_game_model(self, df: pd.DataFrame, socsim_meta: Dict[str, Any]) -> None:
        """Restore the effect the user specified on a DV that the game model overwrote.

        The behavioral-economics game model replaces the item columns of a game DV with its
        own output, which knows nothing about the requested Cohen's d (observed d came out
        near 0 for a requested 0.5). For every enriched DV that carries a user-specified
        effect, move each condition's mean to the target pattern (+/- d/2 within-condition
        SDs around the DV's overall mean, so two arms differ by d) and re-check after
        rounding and clipping at the scale bounds. DVs without a requested effect keep the
        game model's own condition differences.
        """
        enriched = {str(e.get("variable", "")) for e in (socsim_meta or {}).get("enriched_dvs", [])}
        if not enriched or "CONDITION" not in df.columns:
            return
        applied = getattr(self, "_applied_effects", None) or {}
        done: List[Dict[str, Any]] = []
        for entry in getattr(self, "_scale_generation_log", None) or []:
            all_cols = entry.get("columns_generated") or []
            cols = [c for c in all_cols if c in df.columns]
            if not cols:
                continue
            sc = next((x for x in self.scales if (str(x.get("name", "Scale")).strip() or "Scale") == entry.get("name")), {})
            if not ({str(sc.get("variable_name", "")), str(sc.get("name", ""))} & enriched):
                continue
            prefix = all_cols[0].rsplit("_", 1)[0]
            # per-condition target offset in units of "d" (arms sit at +/- d/2)
            targets_d: Dict[str, float] = {}
            for (cond, var), info in applied.items():
                if var != prefix or info.get("source") != "user":
                    continue
                unit = float(info.get("unit") or 0.0)
                if unit > 0:
                    targets_d[cond] = float(info["offset"]) / (2.0 * unit)
            if not any(abs(v) > 1e-9 for v in targets_d.values()):
                continue
            block = df[cols].apply(pd.to_numeric, errors="coerce")
            comp = block.mean(axis=1)
            groups = [g.dropna() for _, g in comp.groupby(df["CONDITION"])]
            groups = [g for g in groups if len(g) > 1]
            if not groups:
                continue
            pooled_var = sum(float(g.var()) * (len(g) - 1) for g in groups) / max(1, sum(len(g) - 1 for g in groups))
            sd_w = float(np.sqrt(pooled_var)) if pooled_var > 0 else 0.0
            if sd_w <= 0:
                continue
            lo, hi = float(entry["scale_min"]), float(entry["scale_max"])
            if not any((df["CONDITION"] == cond).any() for cond in targets_d):
                continue
            # Every condition takes part: the requested arms move by their target, and each
            # condition (reference included) gets the chance error of an independent sample.
            masks = {cond: (df["CONDITION"] == cond).to_numpy() for cond in df["CONDITION"].dropna().unique()}
            masks = {cond: m for cond, m in masks.items() if m.sum() > 1}
            aim_d = {cond: float(targets_d.get(cond, 0.0)) for cond in masks}
            # The game model draws each condition's participants by stratified latent class, so its
            # arms differ by chance only ~30% as much as independent samples do (measured on the
            # null gap: SD 0.044 against sqrt(1/n1 + 1/n2) = 0.082 at N = 600). A real sample
            # carries the full sampling error, so the missing ~70% of its variance is added as a
            # seeded draw per condition mean, in within-condition SD units (var = 0.7 / n).
            _chance = np.random.RandomState((int(self.seed) + _stable_int_hash(prefix + "|chance")) % (2**31))
            for cond in sorted(masks, key=str):
                aim_d[cond] += float(_chance.normal(0.0, np.sqrt(_GAME_MISSING_CHANCE_VAR / int(masks[cond].sum()))))
            # One uniform draw per participant, shared by the item columns, drives randomised
            # rounding: the answers are integers, so a shift smaller than half a point would
            # otherwise round away entirely (or jump a whole point), while randomised rounding
            # moves the arm's mean by exactly the intended amount.
            rng = np.random.RandomState((int(self.seed) + _stable_int_hash(prefix)) % (2**31))
            u = rng.random_sample(len(df))
            values = block.to_numpy(dtype=float)
            groups_idx = [m for m in masks.values()]
            shift = {cond: 0.0 for cond in masks}
            sd_now = sd_w
            result = values.copy()
            # v1.3.0.6: aim at the MOVE, not at the realised gap. Each arm's mean is moved by its
            # target (+/- d/2 within-condition SDs) from wherever the generated data put it, so the
            # chance difference between arms that any real sample has is kept and the observed d
            # varies around the request (SD ~ sqrt(1/n1 + 1/n2)) instead of landing on it every
            # time (it used to vary by ~0.01 between seeds, against ~0.08 at N = 600). Averaged over
            # samples the gap is still the requested one. Same rule as _apply_user_effect_to_scale.
            base_mean = {cond: float(np.nanmean(np.nanmean(values[m], axis=1))) for cond, m in masks.items()}
            for _ in range(8):  # clipping at the bounds eats part of the shift; re-aim
                result = values.copy()
                for cond, m in masks.items():
                    result[m] = np.clip(np.floor(values[m] + shift[cond] + u[m][:, None]), lo, hi)
                new_comp = np.nanmean(result, axis=1)
                gaps = {}
                for cond, m in masks.items():
                    gaps[cond] = aim_d[cond] * sd_now - (float(np.nanmean(new_comp[m])) - base_mean[cond])
                var_parts = [(float(np.nanvar(new_comp[m], ddof=1)), int(m.sum()) - 1) for m in groups_idx if m.sum() > 1]
                if var_parts:
                    sd_now = float(np.sqrt(sum(v * k for v, k in var_parts) / max(1, sum(k for _, k in var_parts)))) or sd_now
                if max(abs(g) for g in gaps.values()) < 0.01 * max(sd_now, 1e-9):
                    break
                for cond in masks:
                    shift[cond] += gaps[cond]
            for cond, m in masks.items():
                shifted = pd.DataFrame(result[m], index=df.index[m], columns=cols)
                df.loc[m, cols] = shifted if shifted.isna().any().any() else shifted.astype(int)
            done.append({"variable": prefix, "conditions": {c: round(float(v), 3) for c, v in targets_d.items()}})
        if done:
            socsim_meta["user_effects_reapplied"] = done
            self._log(f"SocSim: re-applied user-specified effects to {len(done)} game DV(s)")

    def _build_effects_applied(self, observed_effects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Describe, per DV and pair of conditions, the effect that was built into the data.

        ``source`` is "user" (an effect you specified; calibrated so the observed Cohen's d on
        the scale mean lands near ``intended_d``), "inferred" (a heuristic difference derived
        from the condition names; NOT calibrated, so no intended d is given) or "none"
        (inferred effects switched off). ``observed_d`` is the effect actually present in
        this sample.
        """
        import itertools

        applied = getattr(self, "_applied_effects", {}) or {}
        inferred_log = list(getattr(self, "_inferred_effect_log", []) or [])
        by_var: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for (cond, var), info in applied.items():
            by_var.setdefault(var, {})[cond] = info
        observed_by_key = {}
        for o in observed_effects or []:
            observed_by_key[(o.get("variable"), o.get("condition_1"), o.get("condition_2"))] = o.get("cohens_d")
        rows: List[Dict[str, Any]] = []
        for var, per_cond in by_var.items():
            conds = [c for c in self.conditions if c in per_cond]
            pairs = list(itertools.combinations(conds, 2))
            if len(pairs) > 15:
                pairs = [(conds[0], c) for c in conds[1:]]
            for c1, c2 in pairs:
                i1, i2 = per_cond[c1], per_cond[c2]
                sources = {i1["source"], i2["source"]}
                if sources == {"user"}:
                    source = "user"
                elif sources == {"none"}:
                    source = "none"
                elif sources <= {"inferred", "none"}:
                    source = "inferred"
                else:
                    source = "mixed"
                # the two arms of a requested effect sit at +/- d * unit, so their gap is
                # 2 * d * unit in normalised-shift units
                unit = float(i1.get("unit") or i2.get("unit") or 0.0)
                intended = (i1["offset"] - i2["offset"]) / (2.0 * unit) if source in ("user", "none") and unit > 0 else None
                observed = None
                for col in (f"{var}_mean", f"{var}_1"):  # single-item scales have no mean column
                    observed = observed_by_key.get((col, c1, c2))
                    if observed is None:
                        reverse = observed_by_key.get((col, c2, c1))
                        observed = -reverse if reverse is not None else None
                    if observed is not None:
                        break
                row = {
                    "variable": var, "condition_1": c1, "condition_2": c2, "source": source,
                    "intended_d": None if intended is None else round(float(intended), 3),
                    "observed_d": None if observed is None else round(float(observed), 3),
                }
                if source in ("inferred", "mixed"):
                    # v1.3.0.5: published d, the replication shrinkage and the d that sized the contrast
                    for ent in inferred_log:
                        if ent["variable"] == var and (ent["path"] == "paradigm_anchor"
                                                       or ent.get("condition") in (c1, c2)):
                            row.update(published_d=ent["published_d"],
                                       shrinkage_factor=ent["shrinkage_factor"],
                                       applied_d=ent["applied_d"], inferred_from=ent["key"])
                            break
                rows.append(row)
        return {
            "inferred_effects_enabled": bool(getattr(self, "auto_effects", True)),
            "contrasts": rows,
            # v1.3.0.5: how literature-inferred effects were turned into targets
            "inferred_effect_policy": self._inferred_policy_summary(),
            "inferred_effect_sources": inferred_log,
            # v1.2.9.1: one entry per effect you specified: did it reach a variable and a condition?
            "specs": self._effect_spec_diagnostics()["specs"],
            # scales whose effect was built into the finished item responses (several scales in the design)
            "applied_after_generation": list(getattr(self, "_deferred_effect_log", []) or []),
            "note": ("Each contrast is condition_1 minus condition_2, in the order of the conditions. "
                     "intended_d is given only for effects you specified. Inferred effects are a "
                     "heuristic read of the condition names and are not calibrated to a target d."),
        }

    def _inferred_policy_summary(self) -> Dict[str, Any]:
        """The replication policy applied to inferred (never to user-specified) effects."""
        if not HAS_EMPIRICAL_REGISTRY:
            return {"mode": "as_published", "shrinkage_factor": 1.0}
        pol = self._INFERRED_EFFECT_POLICY or _empirical_registry.EffectPolicy()
        tier = _empirical_registry.shrinkage_tier()
        return {
            "mode": pol.mode,
            "shrinkage_factor": round(float(_empirical_registry.policy_factor(pol)), 4),
            "default_tau": round(float(pol.default_tau or _empirical_registry.default_tau()), 4),
            "heterogeneity_draw": bool(pol.heterogeneity_draw),
            "min_retained": float(pol.min_retained),
            "evidence_tier": tier,
            "source_verified": bool(_empirical_registry.shrinkage_verified()),
            "applies_to": "inferred effects only (paradigm anchoring and literature fallback); "
                          "never to effects you specify, the true null, or economic-game baselines",
        }

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
        # Qualtrics-style delivery: Simulated_Data.csv holds the participant-facing
        # columns plus Qualtrics metadata; internal columns live in the diagnostics sidecar.
        try:
            from .qualtrics_export import (
                QUALTRICS_METADATA_COLUMNS as _qx_meta_cols,
                QUALTRICS_COLUMN_DESCRIPTIONS as _qx_meta_desc,
                split_columns as _qx_split,
                protected_columns as _qx_protected,
            )
            _qx_ok = True
        except ImportError:
            _qx_ok = False
            _qx_meta_cols, _qx_meta_desc = [], {}
            _qx_split = None

        _all_names = [c for c, _ in self.column_info]
        if _qx_ok:
            _facing_names, _internal_names = _qx_split(_all_names, _qx_protected({
                "scale_generation_log": getattr(self, "_scale_generation_log", None),
                "open_ended_questions": self.open_ended_questions,
            }))
        else:
            _facing_names, _internal_names = _all_names, []
        _facing_set = set(_facing_names)

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
        ]

        if _qx_ok:
            lines.extend([
                "-" * 70,
                "DELIVERED FILES",
                "-" * 70,
                "",
                "Simulated_Data.csv",
                "    The dataset, laid out like a Qualtrics export (one header row). It starts",
                "    with the Qualtrics metadata columns, followed by the survey columns.",
                "    Analysis scripts in this folder read this file.",
                "Simulated_Data_Qualtrics_Raw.csv",
                "    Same rows and columns with the three-row Qualtrics header: column names,",
                "    question text, and an ImportId row. Use it to practice the usual step of",
                "    deleting rows 2-3 before analysis.",
                "Simulation_Diagnostics.csv",
                "    Simulator bookkeeping for each response, keyed by ResponseId (same row order).",
                "    A real survey export has none of these columns. It is a separate file so the",
                "    main dataset looks like data collected from real respondents.",
                "If Simulation_Diagnostics.csv is not in the ZIP, the Qualtrics-format build was",
                "unavailable for this run and Simulated_Data.csv is the unprocessed simulator table.",
                "",
                "-" * 70,
                "QUALTRICS METADATA COLUMNS (Simulated_Data.csv)",
                "-" * 70,
                "",
            ])
            for _mc in _qx_meta_cols:
                lines.append(_mc)
                lines.append(f"    {_qx_meta_desc.get(_mc, '')}")
                lines.append("")
            lines.extend([
                "Rows are sorted by StartDate. Start times are spread over one to four days and",
                "are generated from the random seed; they are not the time you ran the tool.",
                "Responses that stop before the end of the survey have Finished = 0, a Progress",
                "below 100, blank cells after the point where they stopped, and a shorter Duration.",
                "",
                "-" * 70,
                "SURVEY COLUMNS (Simulated_Data.csv)",
                "-" * 70,
                "",
            ])
            _survey_cols = [(c, d) for c, d in self.column_info if c in _facing_set]
        else:
            lines.extend(["-" * 70, "VARIABLE DESCRIPTIONS", "-" * 70, ""])
            _survey_cols = list(self.column_info)

        for col_name, description in _survey_cols:
            lines.append(f"{col_name}")
            lines.append(f"    {description}")
            lines.append("")

        if _qx_ok:
            lines.extend([
                "-" * 70,
                "DIAGNOSTICS COLUMNS (Simulation_Diagnostics.csv)",
                "-" * 70,
                "",
                "ResponseId",
                "    Links each row to the same row of Simulated_Data.csv (merge on this column).",
                "",
            ])
            for col_name, description in self.column_info:
                if col_name in _facing_set:
                    continue
                if col_name.endswith("_mean"):
                    description = f"{description} (researcher composite; not in Simulated_Data.csv, compute it from the item columns)"
                lines.append(f"{col_name}")
                lines.append(f"    {description}")
                lines.append("")
            lines.extend([
                "Exclude_Recommended is the simulator's own flag. In a real analysis you would",
                "apply your preregistered exclusion rules to Duration (in seconds), Finished and",
                "the attention-check columns of Simulated_Data.csv instead.",
                "",
            ])

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

    # ------------------------------------------------------------------
    # Analysis-script helpers (R / Python / Julia / SPSS / Stata)
    # ------------------------------------------------------------------
    _EXCLUSION_NOTE = (
        "Real analyses apply preregistered exclusion rules to the Duration (in seconds), "
        "Finished and attention-check columns of Simulated_Data.csv. Exclude_Recommended is the "
        "simulator's own flag; it is stored in Simulation_Diagnostics.csv and joined on ResponseId "
        "for this optional step only."
    )

    def _export_script_scales(self, df: Optional[pd.DataFrame], lowercase: bool = False) -> List[Dict[str, Any]]:
        """Scale specs for analysis scripts, restricted to item columns present in the delivered CSV.

        When ``df`` is given, only items that exist as columns are referenced, so a script never
        names a column the file does not contain. Composites are computed by the script itself
        (the delivered CSV has no ``<Scale>_mean`` columns).
        """
        cols = set(df.columns) if df is not None and hasattr(df, "columns") else None
        out: List[Dict[str, Any]] = []
        _skipped: List[str] = []
        _log = list(getattr(self, "_scale_generation_log", None) or [])
        _used: Set[int] = set()
        for scale in self.scales:
            raw = str(scale.get("name", "Scale")).strip() or "Scale"
            name = _clean_column_name(raw)
            num_items = _safe_numeric(scale.get("num_items", 5), default=5, as_int=True)
            points = _safe_numeric(scale.get("scale_points", 7), default=7, as_int=True)
            reverse = _safe_parse_reverse_items(scale.get("reverse_items", []))
            # Reverse coding flips around scale_min + scale_max (same rule as the engine),
            # which equals points + 1 only for 1-based scales.
            _smin = _safe_numeric(scale.get("scale_min", 1), default=1, as_int=True)
            _smax = _safe_numeric(scale.get("scale_max", points), default=points, as_int=True)
            flip = _smin + _smax
            items = [f"{name}_{i}" for i in range(1, num_items + 1)]
            # The generator records the columns it actually wrote (their prefix comes from
            # variable_name and is de-duplicated), so prefer that over rebuilding from `name`.
            _k = next((k for k, e in enumerate(_log)
                       if k not in _used and str(e.get("name", "")).strip() == raw), None)
            _entry = _log[_k] if _k is not None else None
            if _entry is not None:
                _used.add(_k)
                _gen_cols = [str(c) for c in (_entry.get("columns_generated") or [])]
                if _gen_cols:
                    name = _gen_cols[0].rsplit("_", 1)[0]
                    items = _gen_cols
                    reverse = _safe_parse_reverse_items(_entry.get("reverse_items", reverse))
            if cols is not None:
                items = [it for it in items if it in cols]
                if not items:
                    continue
            rev = sorted(r for r in reverse if f"{name}_{r}" in items)
            if lowercase:
                name = name.lower()
                items = [it.lower() for it in items]
            # Composite = mean of the items AFTER reverse coding, so scripts must average
            # the recoded `_R` (Stata: `_r`) columns for reverse-keyed items.
            _sfx = "_r" if lowercase else "_R"
            _rev_cols = {f"{name}_{r}" for r in rev}
            composite_items = [(f"{it}{_sfx}" if it in _rev_cols else it) for it in items]
            if not (_script_name_ok(name) and all(_script_name_ok(it) for it in items)):
                # Names with quotes, brackets, spaces or line breaks cannot be written into code safely.
                _skipped.append(_script_text(raw)[:60])
                continue
            out.append({"raw": _script_text(raw), "name": name, "items": items, "reverse": rev,
                        "points": points, "flip": flip, "composite_items": composite_items})
        self._export_skipped_scales = _skipped
        return out

    def _export_skipped_note(self, prefix: str, suffix: str = "") -> List[str]:
        """Comment lines naming scales left out of an exported script because their names are unsafe in code."""
        skipped = list(getattr(self, "_export_skipped_scales", None) or [])
        if not skipped:
            return []
        names = ", ".join(skipped[:5]) + (" ..." if len(skipped) > 5 else "")
        return [f"{prefix} Not scripted (the name has characters that cannot be written safely into code): {names}{suffix}", ""]

    def _export_has_gender(self, df: Optional[pd.DataFrame]) -> bool:
        if df is not None and hasattr(df, "columns"):
            return "Gender" in df.columns
        return bool(self.demographics.get("include_gender_column", True))

    def generate_r_export(self, df: pd.DataFrame) -> str:
        """
        Generate R data-preparation script for Simulated_Data.csv.

        Composites are computed from the item columns; Simulation_Diagnostics.csv is
        joined on ResponseId for the optional exclusion step.
        """
        def _r_quote(x: str) -> str:
            x = _script_text(x).replace("\\", "\\\\").replace('"', '\\"')
            return f'"{x}"'

        condition_levels = ", ".join([_r_quote(c) for c in self.conditions])

        lines: List[str] = [
            "# ============================================================",
            f"# R Data Preparation Script - {_script_text(self.study_title)}",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Run ID: {_script_text(self.run_id)}",
            "# ============================================================",
            "",
            "# Load packages",
            "suppressPackageStartupMessages({",
            "  library(readr)",
            "  library(dplyr)",
            "})",
            "",
            "# Load the data (Qualtrics-style export; one header row)",
            'data <- read_csv("Simulated_Data.csv", show_col_types = FALSE)',
            "",
            "# Convert CONDITION to factor with proper levels",
            f"data$CONDITION <- factor(data$CONDITION, levels = c({condition_levels}))",
            "",
        ]

        if self._export_has_gender(df):
            lines.extend([
                "# Gender is already labeled as strings (Male, Female, Non-binary, Prefer not to say)",
                'data$Gender <- factor(data$Gender)',
                "",
            ])

        for sc in self._export_script_scales(df):
            if sc["reverse"]:
                lines.append(f"# {sc['raw']} - reverse code items {sc['reverse']}")
                for r_item in sc["reverse"]:
                    item_name = f"{sc['name']}_{r_item}"
                    lines.append(f"data${item_name}_R <- {sc['flip']} - data${item_name}")
                lines.append("")

            lines.append(f"# Create {sc['raw']} composite from the item columns")
            item_list = ", ".join([f"data${item}" for item in sc["composite_items"]])
            lines.append(f"data${sc['name']}_composite <- rowMeans(cbind({item_list}), na.rm = TRUE)")
            lines.append("")
        lines.extend(self._export_skipped_note("#"))

        lines.extend(
            [
                "# Optional exclusion step",
                f"# {self._EXCLUSION_NOTE}",
                'if (file.exists("Simulation_Diagnostics.csv")) {',
                '  diagnostics <- read_csv("Simulation_Diagnostics.csv", show_col_types = FALSE)',
                '  data <- left_join(data, diagnostics[, c("ResponseId", "Exclude_Recommended")], by = "ResponseId")',
                "  data_clean <- data[!is.na(data$Exclude_Recommended) & data$Exclude_Recommended == 0, ]",
                "} else {",
                '  message("Simulation_Diagnostics.csv not found; no exclusions applied.")',
                "  data_clean <- data",
                "}",
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
        Generate Python (pandas) data-preparation script for Simulated_Data.csv.

        Composites are computed from the item columns; Simulation_Diagnostics.csv is
        merged on ResponseId for the optional exclusion step.
        """
        def _py_quote(x: str) -> str:
            x = _script_text(x).replace("\\", "\\\\").replace("'", "\\'")
            return f"'{x}'"

        condition_levels = ", ".join([_py_quote(c) for c in self.conditions])

        lines: List[str] = [
            "# ============================================================",
            f"# Python Data Preparation Script - {_script_text(self.study_title)}",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Run ID: {_script_text(self.run_id)}",
            "# ============================================================",
            "",
            "import os",
            "import pandas as pd",
            "import numpy as np",
            "",
            "# Load the data (Qualtrics-style export; one header row)",
            "data = pd.read_csv('Simulated_Data.csv')",
            "",
            "# Convert CONDITION to categorical with proper order",
            f"condition_order = [{condition_levels}]",
            "data['CONDITION'] = pd.Categorical(data['CONDITION'], categories=condition_order, ordered=True)",
            "",
        ]

        if self._export_has_gender(df):
            lines.extend([
                "# Gender is already labeled as strings (Male, Female, Non-binary, Prefer not to say)",
                "data['Gender'] = pd.Categorical(data['Gender'])",
                "",
            ])

        scales = self._export_script_scales(df)
        for sc in scales:
            if sc["reverse"]:
                lines.append(f"# {sc['raw']} - reverse code items {sc['reverse']}")
                for r_item in sc["reverse"]:
                    item_name = f"{sc['name']}_{r_item}"
                    lines.append(f"data['{item_name}_R'] = {sc['flip']} - data['{item_name}']")
                lines.append("")

            lines.append(f"# Create {sc['raw']} composite from the item columns")
            item_list = ", ".join([f"'{item}'" for item in sc["composite_items"]])
            lines.append(f"data['{sc['name']}_composite'] = data[[{item_list}]].mean(axis=1)")
            lines.append("")
        lines.extend(self._export_skipped_note("#"))

        lines.extend([
            "# Optional exclusion step",
            f"# {self._EXCLUSION_NOTE}",
            "if os.path.exists('Simulation_Diagnostics.csv'):",
            "    diagnostics = pd.read_csv('Simulation_Diagnostics.csv')",
            "    data = data.merge(diagnostics[['ResponseId', 'Exclude_Recommended']], on='ResponseId', how='left')",
            "    data_clean = data[data['Exclude_Recommended'] == 0].copy()",
            "else:",
            "    print('Simulation_Diagnostics.csv not found; no exclusions applied.')",
            "    data_clean = data.copy()",
            "",
            "print(f'Total N: {len(data)}')",
            "print(f'Clean N: {len(data_clean)}')",
            "",
            "# Ready for analysis",
        ])
        if scales:
            lines.append(f"# Example: data_clean.groupby('CONDITION')['{scales[0]['name']}_composite'].mean()")

        return "\n".join(lines)

    def generate_julia_export(self, df: pd.DataFrame) -> str:
        """
        Generate Julia (DataFrames.jl) data-preparation script for Simulated_Data.csv.

        Composites are computed from the item columns; Simulation_Diagnostics.csv is
        joined on ResponseId for the optional exclusion step.
        """
        def _jl_quote(x: str) -> str:
            x = _script_text(x).replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")
            return f'"{x}"'

        condition_levels = ", ".join([_jl_quote(c) for c in self.conditions])

        lines: List[str] = [
            "# ============================================================",
            f"# Julia Data Preparation Script - {_script_text(self.study_title)}",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Run ID: {_script_text(self.run_id)}",
            "# ============================================================",
            "",
            "using CSV",
            "using DataFrames",
            "using CategoricalArrays",
            "using Statistics",
            "",
            "# Load the data (Qualtrics-style export; one header row)",
            'data = CSV.read("Simulated_Data.csv", DataFrame)',
            "",
            "# Convert CONDITION to categorical with proper order",
            f"condition_levels = [{condition_levels}]",
            "data.CONDITION = categorical(data.CONDITION, levels=condition_levels, ordered=true)",
            "",
        ]

        if self._export_has_gender(df):
            lines.extend([
                "# Gender is already labeled as strings (Male, Female, Non-binary, Prefer not to say)",
                "data.Gender = categorical(data.Gender)",
                "",
            ])

        scales = self._export_script_scales(df)
        for sc in scales:
            if sc["reverse"]:
                lines.append(f"# {sc['raw']} - reverse code items {sc['reverse']}")
                for r_item in sc["reverse"]:
                    item_name = f"{sc['name']}_{r_item}"
                    lines.append(f'data.{item_name}_R = {sc["flip"]} .- data.{item_name}')
                lines.append("")

            lines.append(f"# Create {sc['raw']} composite from the item columns (missing values skipped)")
            item_syms = ", ".join([f":{item}" for item in sc["composite_items"]])
            lines.append(
                f"data.{sc['name']}_composite = [isempty(collect(skipmissing(collect(r)))) ? missing : "
                f"mean(skipmissing(collect(r))) for r in eachrow(data[:, [{item_syms}]])]"
            )
            lines.append("")
        lines.extend(self._export_skipped_note("#"))

        lines.extend([
            "# Optional exclusion step",
            f"# {self._EXCLUSION_NOTE}",
            'if isfile("Simulation_Diagnostics.csv")',
            '    diagnostics = CSV.read("Simulation_Diagnostics.csv", DataFrame)',
            "    data = leftjoin(data, select(diagnostics, [:ResponseId, :Exclude_Recommended]), on = :ResponseId)",
            "    data_clean = filter(row -> coalesce(row.Exclude_Recommended, 1) == 0, data)",
            "else",
            '    println("Simulation_Diagnostics.csv not found; no exclusions applied.")',
            "    data_clean = copy(data)",
            "end",
            "",
            'println("Total N: ", nrow(data))',
            'println("Clean N: ", nrow(data_clean))',
            "",
            "# Ready for analysis",
        ])
        if scales:
            lines.append(
                f"# Example: combine(groupby(data_clean, :CONDITION), :{scales[0]['name']}_composite => x -> mean(skipmissing(x)))"
            )

        return "\n".join(lines)

    def generate_spss_export(self, df: pd.DataFrame) -> str:
        """
        Generate SPSS syntax for data preparation of Simulated_Data.csv.

        Composites are computed from the item columns; Simulation_Diagnostics.csv is
        matched on ResponseId for the optional exclusion step.
        """
        lines: List[str] = [
            "* ============================================================.",
            f"* SPSS Data Preparation Syntax - {_script_text(self.study_title)}.",
            f"* Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.",
            f"* Run ID: {_script_text(self.run_id)}.",
            "* ============================================================.",
            "",
            "* Load the data first using:",
            "*   File > Import Data > CSV Data...",
            "*   Select 'Simulated_Data.csv' (one header row; text columns as strings).",
            "",
            "DATASET NAME data WINDOW=FRONT.",
            "",
            "* CONDITION is a string column; create a numeric version with value labels.",
            "AUTORECODE VARIABLES=CONDITION /INTO CONDITION_num /PRINT.",
            "",
        ]

        if self._export_has_gender(df):
            lines.extend([
                "* Gender is already labeled as strings (Male, Female, Non-binary, Prefer not to say).",
                "",
            ])

        for sc in self._export_script_scales(df):
            if sc["reverse"]:
                lines.append(f"* {sc['raw']} - reverse code items {sc['reverse']}.")
                for r_item in sc["reverse"]:
                    item_name = f"{sc['name']}_{r_item}"
                    lines.append(f"COMPUTE {item_name}_R = {sc['flip']} - {item_name}.")
                lines.append("EXECUTE.")
                lines.append("")

            lines.append(f"* Create {sc['raw']} composite from the item columns.")
            lines.append(f"COMPUTE {sc['name']}_composite = MEAN({' '.join(sc['composite_items'])}).")
            lines.append("EXECUTE.")
            lines.append("")
        lines.extend(self._export_skipped_note("*", "."))

        lines.extend([
            "* Optional exclusion step.",
            f"* {self._EXCLUSION_NOTE}",
            "* Import 'Simulation_Diagnostics.csv' the same way (File > Import Data > CSV Data...), then run:",
            "DATASET NAME diag WINDOW=FRONT.",
            "DATASET ACTIVATE diag.",
            "SORT CASES BY ResponseId (A).",
            "DATASET ACTIVATE data.",
            "SORT CASES BY ResponseId (A).",
            "MATCH FILES /FILE=* /TABLE=diag /BY ResponseId.",
            "EXECUTE.",
            "",
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
        Generate Stata .do file for data preparation of Simulated_Data.csv.

        Composites are computed from the item columns; Simulation_Diagnostics.csv is
        merged on responseid (Stata lower-cases imported names) for the optional
        exclusion step.
        """
        def _stata_quote(x: str) -> str:
            # Stata expands `macros' and $globals inside double quotes, so neither may survive in a label
            x = _script_text(x).replace('"', "'").replace("`", "'").replace("$", "")
            return f'"{x}"'

        lines: List[str] = [
            "// ============================================================",
            f"// Stata Data Preparation Do-File - {_script_text(self.study_title)}",
            f"// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"// Run ID: {_script_text(self.run_id)}",
            "// ============================================================",
            "",
            "// Load the data (Qualtrics-style export; one header row)",
            "// Note: import delimited lower-cases variable names (CONDITION -> condition).",
            'import delimited "Simulated_Data.csv", clear varnames(1)',
            "",
            "// Label the CONDITION variable",
        ]

        for i, c in enumerate(self.conditions):
            lines.append(f'label define condition_lbl {i+1} {_stata_quote(c)}, add')
        lines.extend([
            "encode condition, gen(condition_num) label(condition_lbl)",
            "",
        ])

        if self._export_has_gender(df):
            lines.extend([
                "// Gender is already labeled as strings (Male, Female, Non-binary, Prefer not to say)",
                "// No numeric encoding needed",
                "",
            ])

        for sc in self._export_script_scales(df, lowercase=True):
            if sc["reverse"]:
                lines.append(f"// {sc['raw']} - reverse code items {sc['reverse']}")
                for r_item in sc["reverse"]:
                    item_name = f"{sc['name']}_{r_item}"
                    lines.append(f"gen {item_name}_r = {sc['flip']} - {item_name}")
                lines.append("")

            lines.append(f"// Create {sc['raw']} composite from the item columns")
            lines.append(f"egen {sc['name']}_composite = rowmean({' '.join(sc['composite_items'])})")
            lines.append("")
        lines.extend(self._export_skipped_note("//"))

        lines.extend([
            "// Optional exclusion step",
            f"// {self._EXCLUSION_NOTE}",
            'capture confirm file "Simulation_Diagnostics.csv"',
            "if _rc == 0 {",
            "    preserve",
            '    import delimited "Simulation_Diagnostics.csv", clear varnames(1)',
            "    keep responseid exclude_recommended",
            "    tempfile diag",
            "    save `diag'",
            "    restore",
            "    merge 1:1 responseid using `diag', keep(master match) nogenerate",
            "} else {",
            '    display "Simulation_Diagnostics.csv not found; no exclusions applied."',
            "    gen exclude_recommended = 0",
            "}",
            "",
            'display "Total N: " _N',
            "preserve",
            "keep if exclude_recommended == 0",
            'display "Clean N: " _N',
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

        This produces a methods paragraph that documents how the synthetic data
        were generated. It is not a description of empirical data collection.

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
        effect_info = "not configured; inferred from condition labels (exploratory)"
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

Data were generated by a persona-based simulation engine whose response-style
and domain parameters are informed by published survey-methodology research.
The data are synthetic.

Sample and Design: N = {self.sample_size} synthetic participants were randomly
assigned to {n_conditions} experimental condition{'s' if n_conditions > 1 else ''}.
Responses were generated for {n_scales} scale{'s' if n_scales > 1 else ''} measuring
dependent variables relevant to the study context.

Response Generation: Each response was generated through a multi-step process:
(1) Each participant was assigned a persona (response-style and domain personas,
after study-specific reweighting) informed by Krosnick (1991) satisficing theory.
(2) Domain-specific calibrations adjusted response means to match published norms
(Oliver, 1980; Slovic, 1987; Mayer et al., 1995).
(3) Condition effects were applied using standardized effect sizes ({effect_info})
with Cohen's d methodology (Cohen, 1988).
(4) Response styles were simulated based on Greenleaf (1992) for extreme responding
and Billiet & McClendon (2000) for acquiescence bias.

Validation: The generator has not been validated against real participant data.
Responses are bounded to each scale, runs are reproducible from the seed, and a
configured effect size is recovered on the scale mean within roughly 12%. These
data are synthetic and must not be reported as empirical findings.

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
