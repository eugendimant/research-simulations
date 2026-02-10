# Utils package for Behavioral Experiment Simulation Tool
"""
Utility modules for the Behavioral Experiment Simulation Tool.

Version: 1.8.8.1 - Correlation validation, NaN-safe post-generation validation

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
__version__ = "1.8.8.1"

from .qsf_parser import parse_qsf_file, extract_survey_structure, generate_qsf_summary
from .simulation_engine import SimulationEngine
from .enhanced_simulation_engine import (
    EnhancedSimulationEngine,
    EffectSizeSpec,
    ExclusionCriteria
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
