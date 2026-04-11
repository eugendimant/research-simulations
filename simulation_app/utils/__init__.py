# Utils package for Behavioral Experiment Simulation Tool
"""
Utility modules for the Behavioral Experiment Simulation Tool.

Version: 1.2.6.0 - Comprehension checks, mediator engine fix

Changes (v1.0.0 - 20 Iterations of Comprehensive Improvements):
    === ENHANCED SCALE/MATRIX DETECTION ===
    - NEW: Semantic scale type detection (satisfaction, trust, intention, etc.)
    - NEW: Well-known scale recognition (Big Five, PANAS, SWLS, PSS, etc.)
    - NEW: Reverse-coded item detection
    - NEW: Scale anchor extraction
    - NEW: Scale quality scoring with warnings/recommendations
    - ENHANCED: Multi-dimensional scale detection with domain mapping

    === LIVE DATA PREVIEW ===
    - NEW: 5-row preview before full generation
    - NEW: Preview shows exact output format
    - NEW: Column type indicators in preview

    === CONDITIONAL/SKIP LOGIC AWARENESS ===
    - NEW: Full DisplayLogic parsing from QSF
    - NEW: SkipLogic parsing with destination tracking
    - NEW: Question dependency graph building
    - NEW: Conditional branching detection
    - NEW: Skip logic awareness in data generation

    === DIFFICULTY LEVELS FOR DATA QUALITY ===
    - NEW: 4 difficulty levels (Easy, Medium, Hard, Expert)
    - NEW: Difficulty impacts numeric response patterns
    - NEW: Difficulty impacts open-text response quality and complexity
    - NEW: Difficulty-aware careless responder rates

    === MEDIATION VARIABLE SUPPORT ===
    - NEW: Automatic mediator variable detection
    - NEW: Mediator configuration UI
    - NEW: Mediation model data generation
    - NEW: Path coefficient simulation

    === PRE-REGISTRATION CONSISTENCY CHECKER ===
    - NEW: OSF pre-registration format parsing
    - NEW: AEA pre-registration format parsing
    - NEW: AsPredicted format parsing
    - NEW: Consistency checker comparing pre-reg to design
    - NEW: Warnings for deviations from pre-registration

Previous (v2.4.5 - 5 Iterations of Improvements):
    - Enhanced DV detection, cultural personas, new domains, export formats

Modules:
    - qsf_parser: Parse Qualtrics Survey Format (.qsf) files
    - qsf_preview: Interactive QSF preview with 200+ exclusion patterns
    - simulation_engine: Core simulation logic with persona-based generation
    - enhanced_simulation_engine: Advanced simulation with effect sizes and personas
    - persona_library: Comprehensive behavioral persona library (50+ archetypes)
    - response_library: 225+ domain-specific response templates
    - text_generator: Free open-ended text response generation (40 question types)
    - condition_identifier: Enhanced condition identification (30 variable roles)
    - instructor_report: Comprehensive instructor-only analysis reports
    - schema_validator: Data validation with 10+ quality checks
    - pdf_generator: Generate tamper-proof audit log PDFs
    - group_management: Student group registration and usage tracking
    - svg_charts: Pure SVG chart generators (guaranteed visualizations)
    - github_qsf_collector: Auto-upload QSF files to GitHub for collection
"""

# Package version - should match all module versions
__version__ = "1.2.6.0"


# =============================================================================
# CENTRALIZED OE COLUMN DETECTION (v1.2.5.1)
# =============================================================================
# This is the SINGLE SOURCE OF TRUTH for determining which DataFrame columns
# contain open-ended text responses vs metadata/condition/demographic columns.
# ALL modules MUST use this function instead of rolling their own detection.
# This prevents the CONDITION corruption bug where stylometric fingerprinting
# or uniqueness correction treats condition labels as OE text.
# =============================================================================

# Columns that are NEVER open-ended text, regardless of dtype or content length
_PROTECTED_COLUMNS = frozenset({
    "PARTICIPANT_ID", "RUN_ID", "SIMULATION_MODE", "SIMULATION_SEED",
    "CONDITION", "Age", "Gender", "Attention_Check_1",
    "Completion_Time_Seconds", "Attention_Pass_Rate", "Max_Straight_Line",
    "Flag_Speed", "Flag_Attention", "Flag_StraightLine", "Exclude_Recommended",
    "_Generation_Source", "Mean_Item_RT_ms", "Total_Scale_RT_ms",
})

# Prefixes that indicate non-OE columns
_PROTECTED_PREFIXES = (
    "ABE3_", "HBS_", "_", "Flag_", "Exclude_", "Attention_",
    "Hedonic_", "Utilitarian_",
)


def detect_oe_columns(df, known_oe_names=None):
    """Centralized open-ended column detection.

    Args:
        df: pandas DataFrame with generated data
        known_oe_names: Optional set/list of known OE variable names from engine config.
            When provided, ONLY these columns are returned (safest mode).
            When None, falls back to heuristic detection with strict guards.

    Returns:
        List of column names that are open-ended text response columns.

    This function is the ONLY approved way to detect OE columns.
    It guarantees that CONDITION, demographics, and metadata columns are
    NEVER misidentified as OE text, regardless of their content or length.
    """
    import re as _re

    # If we know the exact OE variable names, use them exclusively
    if known_oe_names:
        _names = set(known_oe_names) if not isinstance(known_oe_names, set) else known_oe_names
        _names.discard("")
        if _names:
            return [col for col in df.columns if col in _names]

    # Fallback: heuristic detection with strict safety guards
    oe_cols = []
    for col in df.columns:
        # Guard 1: Explicit exclusion by name
        if col in _PROTECTED_COLUMNS:
            continue

        # Guard 2: Explicit exclusion by prefix
        if any(col.startswith(p) for p in _PROTECTED_PREFIXES):
            continue

        # Guard 3: Skip numeric columns
        _dtype_str = str(df[col].dtype)
        if _dtype_str in ('int64', 'float64', 'int32', 'float32', 'int', 'float'):
            continue

        # Guard 4: Must be a text dtype
        _is_text = (df[col].dtype == object
                    or _dtype_str in ('string', 'str', 'String'))
        if not _is_text:
            continue

        # Guard 5: Cardinality check — categorical columns (like CONDITION)
        # have very low cardinality relative to row count
        _non_null = df[col].dropna()
        if len(_non_null) == 0:
            continue
        _n_unique = _non_null.nunique()
        _n_total = len(_non_null)
        if _n_total >= 10 and _n_unique / _n_total < 0.10:
            continue  # <10% unique → categorical, not OE text

        # Guard 6: Must have substantial text length (>20 chars avg)
        _sample = _non_null.head(20)
        _avg_len = _sample.astype(str).str.len().mean()
        if _avg_len <= 20:
            continue

        # Guard 7: Skip columns whose name looks like a scale item (ends with _N)
        if _re.match(r'^.+_\d+$', col) and _avg_len < 50:
            continue

        oe_cols.append(col)

    return oe_cols

from .qsf_parser import parse_qsf_file, extract_survey_structure, generate_qsf_summary
from .simulation_engine import SimulationEngine
from .enhanced_simulation_engine import (
    EnhancedSimulationEngine,
    EffectSizeSpec,
    ExclusionCriteria,
    LLMExhaustedMidGeneration,
)
from .persona_library import (
    PersonaLibrary,
    Persona,
    TextResponseGenerator,
    StimulusEvaluationHandler
)
from .pdf_generator import generate_audit_log_pdf
from .schema_validator import (
    validate_schema,
    generate_schema_summary,
    check_data_quality,
    validate_scale_response_ranges,
    check_condition_allocation_balance,
    analyze_missing_data_patterns,
    detect_extreme_values,
    generate_validation_report,
)
from .qsf_preview import (
    QSFPreviewParser,
    QSFPreviewResult,
    QSFCorrections
)
from .instructor_report import (
    InstructorReportGenerator,
    ComprehensiveInstructorReport,
)
from .group_management import (
    GroupManager,
    APIKeyManager,
    RegisteredGroup,
    create_sample_groups_file
)
from .condition_identifier import (
    EnhancedConditionIdentifier,
    DesignAnalysisResult,
    IdentifiedCondition,
    IdentifiedFactor,
    IdentifiedScale,
    IdentifiedVariable,
    VariableRole,
    RandomizationLevel,
    RandomizationInfo,
    analyze_qsf_design,
)
from .text_generator import (
    OpenEndedTextGenerator,
    PersonaTextTraits,
    ResponseSentiment,
    ResponseStyle,
    MarkovChainGenerator,
    create_text_generator,
)
from .svg_charts import (
    create_bar_chart_svg,
    create_distribution_svg,
    create_histogram_svg,
    create_means_comparison_svg,
    create_effect_size_svg,
    create_summary_table_svg,
)
from .github_qsf_collector import (
    collect_qsf_async,
    collect_qsf_sync,
    is_collection_enabled,
    get_collection_status,  # v1.2.0: Added detailed status function
)
from .survey_builder import (
    SurveyDescriptionParser,
    ParsedDesign,
    ParsedCondition,
    ParsedScale,
    ParsedOpenEnded,
    generate_qsf_from_design,
)
from .scientific_knowledge_base import (
    get_knowledge_base_summary,
    get_meta_analytic_effect,
    get_game_calibration,
    get_construct_norm,
    get_cultural_adjustment,
    META_ANALYTIC_DB,
    GAME_CALIBRATIONS,
    CONSTRUCT_NORMS,
)

__all__ = [
    # Original exports
    'parse_qsf_file',
    'extract_survey_structure',
    'generate_qsf_summary',
    'SimulationEngine',
    'generate_audit_log_pdf',
    'validate_schema',
    'generate_schema_summary',
    'check_data_quality',
    'validate_scale_response_ranges',
    'check_condition_allocation_balance',
    'analyze_missing_data_patterns',
    'detect_extreme_values',
    'generate_validation_report',
    # Enhanced simulation
    'EnhancedSimulationEngine',
    'EffectSizeSpec',
    'ExclusionCriteria',
    # Persona library
    'PersonaLibrary',
    'Persona',
    'TextResponseGenerator',
    'StimulusEvaluationHandler',
    # QSF preview
    'QSFPreviewParser',
    'QSFPreviewResult',
    'QSFCorrections',
    # Instructor reports
    'InstructorReportGenerator',
    'ComprehensiveInstructorReport',
    # Group management
    'GroupManager',
    'APIKeyManager',
    'RegisteredGroup',
    'create_sample_groups_file',
    # Condition identifier
    'EnhancedConditionIdentifier',
    'DesignAnalysisResult',
    'IdentifiedCondition',
    'IdentifiedFactor',
    'IdentifiedScale',
    'IdentifiedVariable',
    'VariableRole',
    'RandomizationLevel',
    'RandomizationInfo',
    'analyze_qsf_design',
    # Text generator
    'OpenEndedTextGenerator',
    'PersonaTextTraits',
    'ResponseSentiment',
    'ResponseStyle',
    'MarkovChainGenerator',
    'create_text_generator',
    # SVG charts (guaranteed visualization fallbacks)
    'create_bar_chart_svg',
    'create_distribution_svg',
    'create_histogram_svg',
    'create_means_comparison_svg',
    'create_effect_size_svg',
    'create_summary_table_svg',
    # GitHub QSF collection
    'collect_qsf_async',
    'collect_qsf_sync',
    'is_collection_enabled',
    'get_collection_status',
    # Survey builder
    'SurveyDescriptionParser',
    'ParsedDesign',
    'ParsedCondition',
    'ParsedScale',
    'ParsedOpenEnded',
    'generate_qsf_from_design',
]
