# Utils package for BDS5010 Simulation Tool
"""
Utility modules for the BDS5010 Behavioral Experiment Simulation Tool.

Modules:
    - qsf_parser: Parse Qualtrics Survey Format (.qsf) files
    - simulation_engine: Core simulation logic with persona-based generation
    - pdf_generator: Generate tamper-proof audit log PDFs
    - schema_validator: Validate and summarize generated data schemas
"""

from .qsf_parser import parse_qsf_file, extract_survey_structure, generate_qsf_summary
from .simulation_engine import SimulationEngine
from .pdf_generator import generate_audit_log_pdf
from .schema_validator import validate_schema, generate_schema_summary, check_data_quality

__all__ = [
    'parse_qsf_file',
    'extract_survey_structure',
    'generate_qsf_summary',
    'SimulationEngine',
    'generate_audit_log_pdf',
    'validate_schema',
    'generate_schema_summary',
    'check_data_quality'
]
