# Utils package for Behavioral Experiment Simulation Tool
"""
Utility modules for the Behavioral Experiment Simulation Tool.

Version: 2.4.5
Changes (v2.4.5 - 5 Iterations of Improvements):
    - NEW: Enhanced DV detection (rank-order, best-worst, paired comparison, hot spot scales)
    - NEW: 6 cultural response style personas (East Asian, Latin, Nordic, Middle Eastern)
    - NEW: 2 generational personas (Gen Z, Baby Boomer)
    - NEW: 33 new research domains (AI alignment, climate action, health disparities, genomics, etc.)
    - NEW: 10 new domain templates for emerging research areas
    - NEW: Python/pandas export script (Python_Prepare_Data.py)
    - NEW: Julia/DataFrames export script (Julia_Prepare_Data.jl)
    - NEW: SPSS syntax export (SPSS_Prepare_Data.sps)
    - NEW: Stata do-file export (Stata_Prepare_Data.do)
    - NEW: Design Preview panel in Step 3 with configuration summary
    - ENHANCED: Progress indicators with percentage and status emojis
    - ENHANCED: Design type auto-detection (2×2, 2×3, 3×3 factorial)

Previous (v2.4.4 - UI/UX Improvements + Instructor Report):
    - FIX: Step 3 now scrolls to top (section 1) when navigating
    - NEW: Open-ended question verification step in Step 3 UI (Section 5)
    - NEW: User can add/remove open-ended questions like conditions/DVs
    - NEW: Confirmed open-ended questions passed to simulation engine
    - NEW: Instructor report Section 6: Open-Ended Questions Summary
    - NEW: Instructor report Section 7: Effect Size Quality Assessment
    - NEW: Instructor report Section 8: Condition Balance Analysis
    - ENHANCED: Design Summary now shows open-ended question count
    - ENHANCED: 5 scroll-to-top strategies for Streamlit compatibility

Previous (v2.4.3):
    - QSF training on 6 files, 15+ selector types, slider config, validation details

Previous (v2.4.2):
    - Enhanced detection from QSF analysis, FORM fields, ForceResponse tracking

Previous (v2.4.1):
    - 100+ manipulation types grounded in 75+ published sources

Previous (v2.3.0):
    - COMPREHENSIVE: All manipulation types grounded in published literature

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
"""

# Package version - should match all module versions
__version__ = "2.4.5"

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
from .schema_validator import validate_schema, generate_schema_summary, check_data_quality
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
]
