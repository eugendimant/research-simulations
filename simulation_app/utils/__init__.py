# Utils package for Behavioral Experiment Simulation Tool
"""
Utility modules for the Behavioral Experiment Simulation Tool.

Version: 2.1.5
Changes:
    - Added ComprehensiveInstructorReport for detailed instructor-only analysis
    - Comprehensive report includes: statistical tests, effect sizes, hypothesis checks
    - Instructor email now receives detailed analysis; students get simpler report
    - Fixed extra DV generation issue with scale deduplication
    - Improved scale point detection and handling (preserve QSF values, track source)
    - Added comprehensive persona transparency section to instructor report
    - Added scale source tracking (QSF detected vs default) in reports
    - Fixed nested flow list handling in QSF parsers
    - Added forced module reload in app.py to fix Streamlit caching issues

Modules:
    - qsf_parser: Parse Qualtrics Survey Format (.qsf) files
    - simulation_engine: Core simulation logic with persona-based generation
    - enhanced_simulation_engine: Advanced simulation with effect sizes and personas
    - persona_library: Comprehensive behavioral persona library
    - pdf_generator: Generate tamper-proof audit log PDFs
    - schema_validator: Validate and summarize generated data schemas
    - qsf_preview: Interactive QSF preview with error logging
    - instructor_report: Generate instructor-only analysis reports
    - group_management: Student group registration and usage tracking
    - condition_identifier: Enhanced condition and variable identification
    - text_generator: Free open-ended text response generation
"""

# Package version - should match all module versions
__version__ = "2.1.5"

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
]
