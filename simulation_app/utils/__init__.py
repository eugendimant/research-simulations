# Utils package for Behavioral Experiment Simulation Tool
"""
Utility modules for the Behavioral Experiment Simulation Tool.

Version: 2.4.0
Changes (v2.4.0 - EXPANDED - 100+ Manipulation Types from 75+ Sources):
    - EXPANDED: 100+ manipulation types grounded in 75+ published sources
    - Following Westwood (PNAS 2025) for realistic human simulation
    - 16 research domains with theory-based effect directions:
        1-8: AI/Tech, Consumer, Social Psych, Behavioral Econ, Game Theory, Health, Org, Political
        9. Cognitive/Decision: Choice overload (Iyengar), Sunk cost (Staw), CLT (Trope/Liberman)
        10. Communication: Source credibility (Hovland), Inoculation (McGuire), Narratives (Allen)
        11. Learning/Memory: Testing effect (Roediger), Spacing effect (Cepeda meta)
        12. Social Identity: Contact hypothesis (Pettigrew meta), Common identity (Gaertner)
        13. Motivation: Implementation intentions (Gollwitzer meta d=0.65), Mindset (Dweck)
        14. Environmental: Temperature (Anderson), Nature (Berman), Crowding (Baum)
        15. Embodiment: Facial feedback (Strack/Coles replication), Power posing (contested)
        16. Temporal: Time pressure (Dror), Circadian effects (Sievertsen)
    - Feedback/bug report system added to all pages
    - QSF files now stored in example_files folder

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
__version__ = "2.4.0"

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
