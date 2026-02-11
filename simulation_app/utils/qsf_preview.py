"""
Interactive QSF Preview with Error Logging
==========================================
Module for parsing, previewing, and validating Qualtrics Survey Format (QSF) files
with comprehensive error detection and logging for instructor review.

Key Features:
- Parses QSF JSON structure with robust error handling
- Detects 45+ randomization patterns across 6 categories
- Extracts experimental conditions from flow, blocks, and embedded data
- Identifies scales, attention checks, and manipulation checks
- Provides detailed logging for debugging and review
- Supports multiple QSF format variants (old and new Qualtrics exports)

Error Handling:
- Graceful degradation when optional fields are missing
- Detailed error messages with context for debugging
- Validation of critical structures before processing
- Recovery from malformed JSON sections

Supported QSF Formats:
- Qualtrics Research Suite exports
- Qualtrics Core XM exports
- Legacy Qualtrics format
- UTF-8 and UTF-16 encoded files
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Version identifier to help track deployed code
__version__ = "1.0.3.7"  # v1.0.3.7: Fix checklist logic, confirmation banners


# ============================================================================
# RANDOMIZATION PATTERN DEFINITIONS
# ============================================================================
# Qualtrics supports many different ways to implement randomization and
# experimental conditions. This module detects 15+ patterns:
#
# 1. BlockRandomizer (SubSet=1) - Standard between-subjects design
# 2. BlockRandomizer (SubSet>1) - Within-subjects/mixed designs
# 3. Randomizer - General randomizer element
# 4. EmbeddedData in Randomizer - Conditions set via EmbeddedData fields
# 5. Branch-based conditions - Using Branch elements with display logic
# 6. Nested randomizers - Factorial designs with multiple levels
# 7. Group randomizers - Randomization at group level
# 8. Question randomization - Randomizing question order within blocks
# 9. Quota-based conditions - Using Quotas for participant assignment
# 10. WebService conditions - External API-based assignment
# 11. RandomInteger/RandomNumber - Numeric randomization for conditions
# 12. Authenticator-based conditions - Based on authentication results
# 13. Reference Survey conditions - Conditions from other surveys
# 14. EndSurvey branches - Different endings based on conditions
# 15. Piped text conditions - Dynamic text based on randomized values
# ============================================================================


class LogLevel(Enum):
    """Log severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """A single log entry."""
    timestamp: str
    level: LogLevel
    category: str
    message: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'level': self.level.value,
            'category': self.category,
            'message': self.message,
            'details': self.details
        }


@dataclass
class QuestionInfo:
    """Parsed information about a survey question."""
    question_id: str
    question_text: str
    question_type: str
    block_name: str
    choices: List[str] = field(default_factory=list)
    scale_points: Optional[int] = None
    is_matrix: bool = False
    sub_questions: List[str] = field(default_factory=list)
    validation_issues: List[str] = field(default_factory=list)
    # Track text entry choices in MC questions (e.g., "Other: ____")
    text_entry_choices: List[Dict[str, str]] = field(default_factory=list)
    # Track if this question has ForceResponse validation
    force_response: bool = False
    # Track the export tag (variable name in data)
    export_tag: str = ""
    # Raw payload for advanced parsing
    raw_payload: Dict = field(default_factory=dict)
    # v2.4.2: Track selector type for precise detection (SL, ML, ESTB, FORM, etc.)
    selector: str = ""
    # v2.4.2: Track form fields for FORM questions (multiple text inputs)
    form_fields: List[Dict[str, str]] = field(default_factory=list)
    # v2.4.2: Track if this is a comprehension check (CustomValidation with expected answer)
    is_comprehension_check: bool = False
    # v2.4.2: Track expected answer for comprehension checks
    comprehension_expected: Optional[str] = None
    # v2.4.3: Enhanced validation tracking from QSF training
    min_chars: Optional[int] = None  # Minimum character requirement for text entry
    max_chars: Optional[int] = None  # Maximum character requirement
    validation_regex: Optional[str] = None  # Regex pattern for validation (email, etc.)
    number_min: Optional[float] = None  # Minimum value for number validation
    number_max: Optional[float] = None  # Maximum value for number validation
    content_type: Optional[str] = None  # ValidNumber, ValidZip, ValidEmail, etc.
    # v2.4.3: Slider configuration details
    slider_min: Optional[float] = None  # Slider minimum value
    slider_max: Optional[float] = None  # Slider maximum value
    slider_grid_lines: Optional[int] = None  # Number of grid lines
    slider_snap_to_grid: bool = False  # Whether slider snaps to grid
    slider_labels: Dict[str, str] = field(default_factory=dict)  # Position labels
    # v2.4.3: Skip/display logic tracking
    has_skip_logic: bool = False
    has_display_logic: bool = False
    # v2.4.3: Randomization tracking
    choice_randomization: bool = False  # Whether choices are randomized
    fixed_choices: List[str] = field(default_factory=list)  # Choices fixed at end
    # v1.0.0: Enhanced skip/display logic details
    skip_logic_details: Dict[str, Any] = field(default_factory=dict)  # Full skip logic definition
    display_logic_details: Dict[str, Any] = field(default_factory=dict)  # Full display logic definition
    depends_on_questions: List[str] = field(default_factory=list)  # Questions this depends on
    triggers_skip_to: List[str] = field(default_factory=list)  # Questions this can skip to
    # v1.0.0: Semantic scale detection
    scale_semantic_type: Optional[str] = None  # satisfaction, trust, intention, etc.
    is_reverse_coded: bool = False  # Whether this is a reverse-coded item
    scale_anchors: Dict[str, str] = field(default_factory=dict)  # {1: "Strongly disagree", 7: "Strongly agree"}
    # v1.0.0: Mediation variable detection
    is_potential_mediator: bool = False  # Based on position and type
    mediator_hints: List[str] = field(default_factory=list)  # Keywords suggesting mediation


@dataclass
class BlockInfo:
    """Parsed information about a survey block."""
    block_id: str
    block_name: str
    questions: List[QuestionInfo] = field(default_factory=list)
    is_randomizer: bool = False
    randomizer_type: Optional[str] = None
    block_type: str = "Standard"  # Standard, Default, Trash, etc.


@dataclass
class QSFPreviewResult:
    """Complete result of QSF parsing and validation."""
    success: bool
    survey_name: str
    total_questions: int
    total_blocks: int
    blocks: List[BlockInfo]
    detected_conditions: List[str]
    detected_scales: List[Dict[str, Any]]
    embedded_data: List[str]
    flow_elements: List[str]
    validation_errors: List[str]
    validation_warnings: List[str]
    log_entries: List[LogEntry]
    raw_structure: Dict[str, Any]
    open_ended_questions: List[str] = field(default_factory=list)
    attention_checks: List[str] = field(default_factory=list)
    randomizer_info: Dict[str, Any] = field(default_factory=dict)
    # Detailed open-ended info (includes question text, context, etc.)
    open_ended_details: List[Dict[str, Any]] = field(default_factory=list)
    # Study context extracted from questions and instructions
    study_context: Dict[str, Any] = field(default_factory=dict)
    # Embedded data conditions (for surveys using EmbeddedData for randomization)
    embedded_data_conditions: List[Dict[str, Any]] = field(default_factory=list)
    # v2.4.2: Questions with ForceResponse validation (MUST be filled)
    forced_response_questions: List[Dict[str, Any]] = field(default_factory=list)
    # v2.4.2: Comprehension checks with expected answers
    comprehension_checks: List[Dict[str, Any]] = field(default_factory=list)
    # v2.4.3: Slider questions with full configuration for accurate simulation
    slider_questions: List[Dict[str, Any]] = field(default_factory=list)
    # v2.4.3: Text entry questions with validation requirements
    text_entry_questions: List[Dict[str, Any]] = field(default_factory=list)
    # v1.0.0: Enhanced skip/display logic awareness
    skip_logic_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # question_id -> skip logic details
    display_logic_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # question_id -> display logic details
    question_dependencies: Dict[str, List[str]] = field(default_factory=dict)  # question_id -> list of dependent question_ids
    conditional_branches: List[Dict[str, Any]] = field(default_factory=list)  # All branch conditions in survey
    # v1.0.0: Mediation variable detection
    potential_mediators: List[Dict[str, Any]] = field(default_factory=list)  # Detected potential mediator variables
    # v1.0.0: Enhanced scale detection with semantic types
    scale_semantic_types: Dict[str, str] = field(default_factory=dict)  # scale_name -> semantic_type
    recognized_scales: List[Dict[str, Any]] = field(default_factory=list)  # Well-known scales (Big Five, PANAS, etc.)
    # v1.0.0: Scale validation and quality metrics
    scale_quality_scores: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # scale_name -> quality metrics
    # v1.0.0: Condition-to-question visibility mapping (CRITICAL for simulation)
    # Maps each condition to the set of questions that condition's participants would see
    condition_visibility_map: Dict[str, Dict[str, bool]] = field(default_factory=dict)  # condition -> {question_id: visible}
    # v1.0.0: Condition-to-block mapping
    condition_blocks: Dict[str, List[str]] = field(default_factory=dict)  # condition -> list of block_ids
    # v1.0.0: Block-to-questions mapping
    block_questions: Dict[str, List[str]] = field(default_factory=dict)  # block_id -> list of question_ids
    # v1.0.0: Questions that ALL conditions see (shared questions like demographics)
    shared_questions: List[str] = field(default_factory=list)  # question_ids visible to all conditions


class QSFPreviewParser:
    """
    Parser for Qualtrics Survey Format files with comprehensive validation.

    Features:
    - Extracts all questions, blocks, and survey flow
    - Detects experimental conditions from randomizers
    - Identifies multi-item scales
    - Validates survey structure
    - Generates detailed error logs
    """

    # Block names that are NEVER experimental conditions
    # These are common structural/admin block names in Qualtrics surveys
    # Comprehensive list of block names that are NEVER experimental conditions
    # These include all variations of trash, unused, default, and administrative blocks
    # EXPANDED TO 400+ PATTERNS for maximum exclusion coverage (v1.0.0 training on 210 QSF files)
    EXCLUDED_BLOCK_NAMES = {
        # ========== TRASH / UNUSED (CRITICAL - NEVER USE) ==========
        'trash', 'trash / unused questions', 'trash/unused questions',
        'trash questions', 'trash block', 'trashed', 'trashed questions',
        'unused', 'unused questions', 'unused block', 'unused items',
        'deleted', 'deleted questions', 'deleted block', 'deleted items',
        'archived', 'archived questions', 'archived block', 'archive',
        'old', 'old questions', 'old block', 'old items', 'outdated',
        'deprecated', 'deprecated questions', 'removed', 'removed questions',
        'do not use', 'dont use', "don't use", 'not in use', 'not used',
        'ignore', 'ignored', 'hidden', 'disabled', 'inactive', 'obsolete',
        'discarded', 'scrapped', 'backup', 'backup questions', 'spare',
        'temp', 'temporary', 'test', 'testing', 'draft', 'wip',
        '(only for pre test) feedback',  # v1.0.0: From QSF training

        # ========== GENERIC BLOCK NAMES ==========
        'block', 'block 1', 'block 2', 'block 3', 'block 4', 'block 5',
        'block 6', 'block 7', 'block 8', 'block 9', 'block 10',
        'block 11', 'block 12', 'block 13', 'block 14', 'block 15',
        'block 16', 'block 17', 'block 18', 'block 21', 'block 23',  # v1.0.0: From QSF training
        'block1', 'block2', 'block3', 'block4', 'block5',
        'block db', 'block ds', 'block sb', 'block ss',  # v1.0.0: From QSF training
        'default question block', 'default', 'standard', 'main', 'main block',
        'new block', 'untitled', 'unnamed', 'copy', 'duplicate', 'copy of',
        'blank', 'empty', 'placeholder', 'template', 'skeleton',
        'untitled group',  # v1.0.0: From QSF training

        # ========== INTRODUCTION / WELCOME ==========
        'intro', 'introduction', 'welcome', 'welcome screen', 'welcome message',
        'landing', 'landing page', 'start', 'beginning', 'overview',
        'study intro', 'study introduction', 'survey intro', 'survey introduction',
        'opening', 'opening screen', 'opening page', 'initial', 'initial screen',
        'preamble', 'preface', 'preliminary', 'pre-survey', 'pre survey',
        'opening pages', 'intro block', 'introduction block',  # v1.0.0: From QSF training
        'b1_intro', 'survey intro', 'begin study: information sheet and directions',
        'bds 5010: start', 'intro & consent', 'intro questions', 'start (everyone)',
        'intake', 'pre-experiment',  # v1.0.0: From QSF training

        # ========== INSTRUCTIONS ==========
        'instructions', 'general instructions', 'task instructions',
        'game instructions', 'study instructions', 'survey instructions',
        'directions', 'guidelines', 'rules', 'procedure', 'how to',
        'how-to', 'step by step', 'explanation', 'explainer', 'tutorial',
        'briefing', 'brief', 'orientation', 'introduction instructions',
        'task overview instructions', 'overall instructions', 'game explanation',  # v1.0.0
        'explanation of trust game', 'explanation of die-rolling task',
        'pgg instructions', 'dictator game instructions', 'game structure explanation',
        'experiment walkthrough', 'game start', 'game intro',  # v1.0.0: From QSF training
        'instruction and background', 'instructions_kw', 'instructions_nd', 'instructions_p',

        # ========== CONSENT ==========
        'consent', 'informed consent', 'consent form', 'agreement',
        'irb', 'eligibility', 'screening', 'qualification', 'terms',
        'age verification', 'age check', 'participant agreement',
        'privacy', 'privacy notice', 'data protection', 'gdpr',
        'terms and conditions', 'tos', 'legal', 'disclaimer',
        'participant consent', 'study consent', 'research consent',
        '1. consent form', 'captcha & consent form', 'constent',  # v1.0.0: From QSF training
        'informed consent block + quick demographic', 'intro_consent_block',

        # ========== QUALITY CONTROL ==========
        'captcha', 'bot check', 'recaptcha', 'verification', 'verify',
        'attention check', 'attention checks', 'quality check', 'quality control',
        'manipulation check', 'manipulation checks', 'mc', 'ac',
        'comprehension check', 'comprehension', 'understanding check',
        'instructed response', 'imc', 'trap question', 'screener', 'screen',
        'attention', 'trap', 'check question', 'vigilance', 'vigilance check',
        'data quality', 'response quality', 'validity check', 'validity',
        'attention check 1', 'attention check 2', 'captcha 1', 'captcha 2',  # v1.0.0
        'captcha verification', 'captcha + attention check', 'final attention check',
        'attention/comprehension check', 'attention checks', 'attention check 2 + quality',
        'attention check with current solution', 'attention question',
        'attention_check_1_block', 'comprehension_check_block', 'manipulation_check_block',
        'comprehension 1', 'comprehension 2', 'comprehension questions', 'comprehension checks',
        'b2_conceptcheck', 'check', 'checkquestions', 'crt',  # v1.0.0: From QSF training
        'eligibilitycheck1', 'eligibilitycheck2', 'elibilitycheck1',
        'knowledge check', 'validation', 'validation check',

        # ========== DEMOGRAPHICS ==========
        'demographics', 'demographic info', 'demographic information',
        'demographic questions', 'background', 'background info',
        'personal info', 'personal information', 'about you', 'about yourself',
        'profile', 'participant info', 'respondent info', 'covariates',
        'sociodemographics', 'sociodemographic', 'participant demographics',
        'basic info', 'basic information', 'personal details', 'your info',
        'age gender', 'gender age', 'bio', 'biographical',
        'demographic', 'demographic block', 'demographic question',  # v1.0.0
        'demographic survey questions', 'demographics 1', 'demographics final',
        'demographics & others', 'demographics  employment', 'demography questions',
        'demographic info and conclusion', 'demographic information question',
        'b4_demographics', 'demo_block', 'questions_age', 'questions_education',
        'questions_gender', 'questions_ratio', 'education', 'health',  # v1.0.0: From QSF training
        'bds 5010: financial background', 'income',

        # ========== END / DEBRIEF ==========
        'end', 'end of survey', 'end of games', 'ending', 'finish',
        'completion', 'complete', 'done', 'final', 'conclusion',
        'debrief', 'debriefing', 'debrief form', 'debriefing form',
        'thank you', 'thanks', 'thank you screen', 'thankyou', 'ty',
        'redirect', 'exit', 'goodbye', 'end message', 'closing',
        'wrap up', 'wrapup', 'wrap-up', 'final screen', 'last page',
        'survey end', 'study end', 'experiment end', 'game end',
        'post-survey', 'post survey', 'post-experiment', 'post experiment',
        'closing block', 'thank-you ', '7. respondent codes',  # v1.0.0
        'post-experimental questionnaire', 'post-exprimental questionnaire',
        'purpose/feedback', 'purpose of survey', 'study about', 'survey details',

        # ========== FEEDBACK ==========
        'feedback', 'comments', 'feedback on the survey', 'final feedback',
        'survey feedback', 'general feedback', 'final thoughts',
        'additional comments', 'other comments', 'open feedback',
        'closing feedback', 'participant feedback', 'user feedback',
        'suggestions', 'your thoughts', 'any thoughts', 'thoughts',
        'pilot feedback', 'pilot questions', 'comment',  # v1.0.0: From QSF training

        # ========== OPEN-ENDED ==========
        'open-ended', 'open ended', 'free response', 'free text',
        'open text', 'text entry', 'essay', 'written response',
        'write-in', 'write in', 'open response', 'free form',
        'open-ended questions', 'open',  # v1.0.0: From QSF training

        # ========== PRACTICE / TRAINING ==========
        'practice', 'practice trial', 'practice trials', 'practice round',
        'training', 'training trial', 'tutorial', 'warmup', 'warm up',
        'example', 'sample', 'demo', 'demonstration', 'dry run',
        'practice questions', 'practice block', 'training block',
        'trial run', 'test run', 'familiarization', 'practice session',
        'round 1 (practice)', 'sample question',  # v1.0.0: From QSF training

        # ========== STRUCTURAL (NON-CONDITION) ==========
        'game', 'task', 'main task', 'primary task', 'core task',
        'pairing', 'pairing prompt', 'pair', 'matching', 'match',
        'question', 'questions', 'items', 'measures', 'scales',
        'survey', 'questionnaire', 'assessment', 'test', 'exam',
        'section', 'part', 'module', 'component', 'segment',
        'questionaire', 'questionnaire ',  # v1.0.0: From QSF training (with typo)
        'b3_experiment', 'experiment', 'pilot',  # v1.0.0

        # ========== TIMING / PROGRESS ==========
        'timer', 'timing', 'duration', 'progress', 'progress bar',
        'page break', 'break', 'intermission', 'pause', 'wait',
        'loading', 'transition', 'next', 'continue', 'proceed',
        'transitionpage', 'waiting', 'pairing....',  # v1.0.0: From QSF training

        # ========== PAYMENT / COMPENSATION ==========
        'payment', 'compensation', 'reward', 'bonus', 'payment info',
        'mturk', 'prolific', 'completion code', 'code', 'survey code',
        'prolific code', 'mturk code', 'amazon', 'crowdsourcing',
        'payout', 'pay', 'incentive', 'raffle', 'lottery', 'prize',
        'compensation info', 'payment details', 'bonus info',
        'mturk id', 'mturk id input', 'mturk ids', 'mturkid',  # v1.0.0
        'participant id assignment', 'participant id input', 'worker',
        'random id', 'random id generation', 'randomid', 'romdom id',
        'respondent id generation', 'survey and mturk id', 'mturk participant',
        'bonus pay', 'payout1', 'payout2', 'payment/verification',
        'completion', 'id', 'survey id',  # v1.0.0: From QSF training

        # ========== RANDOMIZATION ARTIFACTS ==========
        'randomizer', 'randomization', 'random', 'assignment',
        'branch', 'branching', 'skip logic', 'display logic',
        'quota', 'quotas', 'quota check', 'embedded data',
        'flow', 'survey flow', 'logic', 'conditional',
        'random and workerid', 'condition assignment ', 'random number generator',

        # ========== ECONOMIC GAME STRUCTURAL (NOT CONDITIONS) ==========
        # These are structural elements of games, not experimental conditions
        'dictator game', 'trust game', 'public goods game', 'pgg',  # v1.0.0
        'dg', 'dg activity', 'dg check', 'pgg activity', 'pgg check', 'pgg intro',
        'dictator choice', 'dictator choice 2', 'dice roll', 'dice roll summary',
        'die-rolling task pg 2', 'dieresult', 'dieroll', 'earnings update',
        'economic game', 'game abc', 'game bca', 'game cab',
        'linear pgg', 'lottery game 2', 'power-to-take game 2',
        'result', 'reward structure', 'risk', 'take-or-give',
        'trading block', 'trading simulation block', 'trust_sat',
        'trustee decision', 'trustee decision 2', 'trustee feedback', 'trustee intro',
        'trustor feedback', 'trustor feedback 1', 'trustor feedback 2',
        'social value orientation', 'social value orientation question 1',
        'social value orientation question 2', 'social value orientation question 3',
        'social value orientation question 4', 'social value orientation question 5',
        'social value orientation question 6',  # v1.0.0: From QSF training

        # ========== ROUND/TRIAL STRUCTURAL (NOT CONDITIONS) ==========
        # Numbered rounds/trials are structural, not experimental conditions
        'round 1', 'round 2', 'round 3', 'round 4', 'round 5',  # v1.0.0
        'round 6', 'round 7', 'round 8', 'round 9', 'round 10',
        'round 11', 'round 12', 'round 13', 'round 14', 'round 15',
        'round 16', 'round 17', 'round 18', 'round 19', 'round 20',
        'round 1 game', 'round 2 game', 'round 2 questions',
        'round 3 - consequences', 'round 6 - consequences', 'round 12 - consequences',
        'convoround_1', 'convoround_2', 'convoround_3',
        'ego network 1', 'ego network 2', 'ego network 3', 'ego network follow up',

        # ========== BELIEF/NORM ELICITATION STRUCTURAL ==========
        # These are measurement blocks, not experimental conditions
        'elicitation of beliefs', 'elicitation of beliefs - announcement',  # v1.0.0
        'gather beliefs', 'beliefs', 'instructions beliefs',
        'norm elicitation (intro)', 'norm elicitation (dg)', 'norm elicitation (outro)',
        'norm elicitation (disclosure)', 'norm elicitation (handshaking)',

        # ========== VIGNETTE STRUCTURAL (NOT CONDITIONS) ==========
        'first scenario', 'scenario 2', 'scenario question',  # v1.0.0
        'vignette ', 'case file', 'new situation',

        # ========== MISCELLANEOUS STRUCTURAL ==========
        'ads', 'advertisement', 'collect email', 'device', 'use mobile',  # v1.0.0
        'networks', 'reference network question', 'own skills',
        'impressions', 'opinion', 'mediators', 'deviance credit',
        'moral foundations task', 'moral opposition question',
        'oneness scale', 'empathy questionnaire', 'fairness questions',
        'uncertainty questions', 'attitudes to lying', 'attitudes/norms',
        'attitudinal 1', 'attitudinal 2', 'awareness/media',
        'product page', 't-shirt picture', 'warm glow photo',
        'stimulus_warning', 'summary screen', 'calculation',
        'confirm wealth', 'hypothetical decision', 'investment scenario',
        'logistical questions', 'plan', 'prompt', 'revise',
        'relationship and household composition', 'tournament thoughts',
    }

    # Block type keywords that indicate non-condition blocks (case-insensitive)
    EXCLUDED_BLOCK_TYPES = {'Trash', 'Default', 'Standard', 'trash', 'default', 'standard'}

    # Patterns that definitively indicate a block should be excluded (v1.0.0: 50+ patterns from 210 QSF training)
    EXCLUDED_BLOCK_PATTERNS = [
        # ========== GENERIC BLOCK PATTERNS ==========
        r'^block\s*\d*$',  # "Block 1", "Block2", etc.
        r'^b\d+$',  # "B1", "B2", etc.
        r'^blk?\s*\d+$',  # "Blk1", "Bl 2", etc.
        r'^\d+$',  # Just numbers
        r'^\s*$',  # Empty or whitespace only
        r'^untitled',  # "Untitled" anything
        r'^unnamed',  # "Unnamed" anything
        r'^copy\s*(?:of\s*)?',  # "copy of X"
        r'\(copy\)',  # Contains "(copy)"
        r'_copy$',  # Ends with "_copy"

        # ========== TRASH / UNUSED PATTERNS ==========
        r'trash',  # Any block containing "trash"
        r'unused',  # Any block containing "unused"
        r'deleted?',  # "delete" or "deleted"
        r'archived?',  # "archive" or "archived"
        r'old\s*(?:questions?)?$',  # "old", "old questions"
        r'(?:do\s*)?not\s*use',  # "do not use", "not use"
        r'donotuse',  # "DONOTUSE" without spaces
        r'obsolete',  # Contains "obsolete"
        r'discarded',  # Contains "discarded"
        r'deprecated',  # Contains "deprecated"
        r'hidden',  # Contains "hidden"
        r'disabled',  # Contains "disabled"
        r'inactive',  # Contains "inactive"
        r'placeholder',  # Any block containing "placeholder"
        r'template',  # Any block containing "template"
        r'backup',  # Any block containing "backup"
        r'^\s*test\s*$',  # Just "test"
        r'^\s*draft\s*$',  # Just "draft"
        r'wip$',  # Ends with "wip"
        r'^temp\b',  # Starts with "temp"

        # ========== ROUND/TRIAL PATTERNS (v1.0.0) ==========
        r'^round\s*\d+',  # "Round 1", "Round 2", etc.
        r'^trial\s*\d+',  # "Trial 1", etc.
        r'^punisher\s*\d+$',  # "Punisher 1", "Punisher 99" - numbered iterations

        # ========== STRUCTURAL PATTERNS (v1.0.0) ==========
        r'^et_\d+$',  # "ET_1", "ET_2", etc.
        r'^y\d+$',  # "Y1", "Y2", etc. (numbered labels)
        r'^convoround_\d+$',  # "ConvoRound_1", etc.
        r'^explanation[a-z]_\d+',  # "ExplanationA_1", "ExplanationB_2", etc.
        r'^lt_',  # Lottery task blocks (LT_DieRoll, LT_Reporting, etc.)
        r'^runfunction_\d+$',  # "runFunction_1", etc.
        r'^wtp_\d+$',  # "WTP_1", "WTP_2" (willingness to pay structural)

        # ========== QUALITY CHECK PATTERNS (v1.0.0) ==========
        r'attention.?check',  # attention_check, attention check, attentioncheck
        r'manipulation.?check',  # manipulation_check, manipulation check
        r'comprehension.?check',  # comprehension_check, comprehension check
        r'eligibility.?check',  # eligibilitycheck, eligibility check
        r'^captcha',  # Starts with captcha
        r'^crt$',  # Cognitive Reflection Test (structural)
        r'^check$',  # Just "check"

        # ========== ADMIN/CONSENT PATTERNS (v1.0.0) ==========
        r'^consent',  # Starts with consent
        r'^informed\s*consent',  # "Informed Consent"
        r'^irb\b',  # IRB-related
        r'^\d+\.\s*consent',  # "1. Consent Form"

        # ========== DEMOGRAPHIC PATTERNS (v1.0.0) ==========
        r'^demo(?:graphic)?s?\b',  # demographics, demographic, demo
        r'^questions_(?:age|gender|education|ratio)',  # questions_age, etc.
        r'financial\s*background',  # Financial background questions

        # ========== INSTRUCTION PATTERNS (v1.0.0) ==========
        r'^instructions?$',  # Just "Instructions" or "Instruction"
        r'^instructions_',  # "instructions_kw", "instructions_nd", etc.
        r'^game\s*instruction',  # "Game Instructions"
        r'^pgg\s*instruction',  # "PGG Instructions"
        r'explanation$',  # Ends with "explanation"

        # ========== PAYMENT/ID PATTERNS (v1.0.0) ==========
        r'mturk',  # Contains "mturk"
        r'^prolific',  # Starts with "prolific"
        r'^random\s*id',  # "Random ID", "RandomID"
        r'worker\s*id',  # "Worker ID"
        r'^participant\s*id',  # "Participant ID"
        r'^survey\s*(?:and\s*)?mturk',  # "Survey and MTurk ID"
        r'completion\s*code',  # "Completion Code"
        r'^payout\d*$',  # "Payout", "Payout1", "Payout2"

        # ========== GAME STRUCTURAL PATTERNS (v1.0.0) ==========
        r'^dg\s*(?:activity|check)?$',  # "DG", "DG activity", "DG check"
        r'^pgg\s*(?:activity|check|intro)?$',  # "PGG", "PGG activity", etc.
        r'^dice\s*(?:roll|q\d|game)',  # "Dice Roll", "Dice Q1", "Dice Game"
        r'^game\s*(?:abc|bca|cab|start)$',  # Game ordering variants
        r'^trustee?\s*(?:decision|feedback|intro)',  # Trustee/Trustor blocks
        r'^trustor\s*(?:decision|feedback)',  # Trustor blocks
        r'^social\s*value\s*orientation',  # SVO blocks

        # ========== NETWORK/EGO PATTERNS (v1.0.0) ==========
        r'^ego\s*network\s*\d+',  # "Ego network 1", etc.

        # ========== VIGNETTE CHECK PATTERNS (v1.0.0) ==========
        r'^vignette_check\d+_scenario',  # "Vignette_Check1_Scenario1_Equity", etc.

        # ========== DEFAULT/STANDARD PATTERNS ==========
        r'default',  # Any block containing "default"
    ]

    # v1.0.0: Keywords that indicate a block IS a condition (positive indicators)
    CONDITION_POSITIVE_KEYWORDS = {
        'treatment', 'control', 'condition', 'group', 'arm',
        'experimental', 'intervention', 'manipulation',
        'high', 'low', 'positive', 'negative',
        'ai', 'human', 'robot', 'automated', 'manual',
        'hedonic', 'utilitarian', 'deontology', 'utilitarianism',
        'empirical', 'normative', 'injunctive', 'descriptive',
        'honest', 'dishonest', 'fair', 'unfair',
        'male', 'female', 'gender',
        'rich', 'poor', 'wealthy', 'scarce', 'abundant',
        'priming', 'prime', 'primed',
        'frame', 'framing', 'framed',
        'message', 'nudge', 'default',
        'ingroup', 'outgroup', 'ingroup', 'minimal group',
        'prosocial', 'antisocial', 'altruistic', 'selfish',
        'cooperative', 'competitive',
        'punishment', 'reward',
        'transparent', 'opaque', 'secret', 'public',
        'individual', 'collective', 'team', 'solo',
        'anchoring', 'anchor',
        'accurate', 'inaccurate',
        'norm', 'norms',
        'equity', 'inequity',
    }

    def __init__(self):
        self.log_entries: List[LogEntry] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def _log(
        self,
        level: LogLevel,
        category: str,
        message: str,
        details: Optional[Dict] = None
    ):
        """Add a log entry."""
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            category=category,
            message=message,
            details=details
        )
        self.log_entries.append(entry)

        if level == LogLevel.ERROR or level == LogLevel.CRITICAL:
            self.errors.append(f"[{category}] {message}")
        elif level == LogLevel.WARNING:
            self.warnings.append(f"[{category}] {message}")

    def _normalize_survey_elements(self, qsf_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalize SurveyElements to a list of dicts across QSF variants."""
        elements = qsf_data.get('SurveyElements', [])
        if isinstance(elements, dict):
            elements = list(elements.values())
        if not isinstance(elements, list):
            return []
        return [element for element in elements if isinstance(element, dict)]

    def _normalize_flow(self, flow: Any) -> List[Dict[str, Any]]:
        """Normalize Flow payloads to a list of dicts across QSF variants."""
        if isinstance(flow, dict):
            if 'Flow' in flow:
                flow = flow.get('Flow', [])
            else:
                flow = list(flow.values())
        if not isinstance(flow, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for item in flow:
            if isinstance(item, dict):
                normalized.append(item)
            elif isinstance(item, list):
                for sub_item in item:
                    if isinstance(sub_item, dict):
                        normalized.append(sub_item)
        return normalized

    def _extract_flow_payload(self, flow_data: Any) -> List[Dict[str, Any]]:
        """Extract a normalized flow list from any flow payload variant."""
        if flow_data is None:
            return []
        if isinstance(flow_data, dict):
            payload = flow_data.get('Payload', flow_data)
        else:
            payload = flow_data
        if isinstance(payload, dict) and 'Flow' in payload:
            payload = payload.get('Flow', [])
        return self._normalize_flow(payload)

    def parse(self, qsf_content: bytes) -> QSFPreviewResult:
        """
        Parse QSF file content and return structured preview.

        Args:
            qsf_content: Raw bytes of the QSF file

        Returns:
            QSFPreviewResult with all parsed information and validation results
        """
        self.log_entries = []
        self.errors = []
        self.warnings = []

        self._log(LogLevel.INFO, "PARSE_START", f"Beginning QSF file parsing (parser v{__version__})")

        # Attempt to parse JSON
        try:
            qsf_data = json.loads(qsf_content.decode('utf-8'))
            # Validate top-level structure is a dict
            if not isinstance(qsf_data, dict):
                raise ValueError(f"QSF top-level structure is {type(qsf_data).__name__}, expected dict")
            self._log(LogLevel.INFO, "JSON_PARSE", "Successfully parsed JSON structure")
        except json.JSONDecodeError as e:
            self._log(
                LogLevel.CRITICAL, "JSON_PARSE",
                f"Failed to parse QSF as JSON: {str(e)}",
                {'error_position': e.pos if hasattr(e, 'pos') else None}
            )
            return QSFPreviewResult(
                success=False,
                survey_name="Unknown",
                total_questions=0,
                total_blocks=0,
                blocks=[],
                detected_conditions=[],
                detected_scales=[],
                embedded_data=[],
                flow_elements=[],
                validation_errors=self.errors,
                validation_warnings=self.warnings,
                log_entries=self.log_entries,
                raw_structure={}
            )

        # Extract survey info
        survey_entry = qsf_data.get('SurveyEntry', {})
        survey_name = survey_entry.get('SurveyName', 'Unnamed Survey')
        self._log(LogLevel.INFO, "SURVEY_INFO", f"Survey name: {survey_name}")

        # Parse survey elements
        elements = self._normalize_survey_elements(qsf_data)
        self._log(LogLevel.INFO, "ELEMENTS", f"Found {len(elements)} survey elements")

        # Organize elements by type
        blocks_map = {}
        questions_map = {}
        flow_data = None
        embedded_data_fields = []

        for element in elements:
            elem_type = element.get('Element', '')

            if elem_type == 'BL':
                # Block definitions
                self._parse_blocks(element, blocks_map)

            elif elem_type == 'SQ':
                # Survey question
                self._parse_question(element, questions_map)

            elif elem_type == 'FL':
                # Survey flow
                flow_data = element
                self._log(LogLevel.INFO, "FLOW", "Found survey flow definition")

            elif elem_type == 'ED':
                # Embedded data
                self._parse_embedded_data(element, embedded_data_fields)

        # Map questions to blocks
        blocks = self._map_questions_to_blocks(blocks_map, questions_map)

        # Detect conditions from flow/randomizers
        detected_conditions = self._detect_conditions(flow_data, blocks)

        # Detect scales
        detected_scales = self._detect_scales(questions_map)

        # Parse flow elements
        flow_elements = self._parse_flow(flow_data)

        # Detect open-ended questions (basic list)
        open_ended = self._detect_open_ended(questions_map)

        # Detect open-ended questions with FULL DETAILS (question text, context, etc.)
        open_ended_details = self._extract_open_ended_details(questions_map, blocks)

        # Extract study context from all questions and instructions
        study_context = self._extract_study_context(survey_name, questions_map, blocks)

        # Detect attention checks
        attention_checks = self._detect_attention_checks(questions_map)

        # Analyze randomizer structure
        randomizer_info = self._analyze_randomizers(flow_data)

        # Detect embedded data conditions (for surveys using EmbeddedData for randomization)
        embedded_data_conditions = self._detect_embedded_data_conditions(flow_data)

        # v2.4.2: Extract forced response questions and comprehension checks
        forced_response_questions = self._extract_forced_response_questions(questions_map)
        comprehension_checks = self._extract_comprehension_checks(questions_map)

        # v2.4.3: Extract slider and text entry questions with full config
        slider_questions = self._extract_slider_questions(questions_map)
        text_entry_questions = self._extract_text_entry_questions(questions_map)

        # v1.0.0: Build condition visibility map (CRITICAL for simulation)
        visibility_result = self._build_condition_visibility_map(
            detected_conditions, blocks, questions_map, flow_data
        )

        # Validate structure
        self._validate_structure(blocks, questions_map, detected_conditions)

        self._log(
            LogLevel.INFO, "PARSE_COMPLETE",
            f"Parsing complete. {len(self.errors)} errors, {len(self.warnings)} warnings, {len(forced_response_questions)} forced Qs, {len(slider_questions)} sliders, {len(text_entry_questions)} text entries"
        )

        return QSFPreviewResult(
            success=len(self.errors) == 0 or all('CRITICAL' not in e for e in self.errors),
            survey_name=survey_name,
            total_questions=len(questions_map),
            total_blocks=len(blocks),
            blocks=blocks,
            detected_conditions=detected_conditions,
            detected_scales=detected_scales,
            embedded_data=embedded_data_fields,
            flow_elements=flow_elements,
            validation_errors=self.errors,
            validation_warnings=self.warnings,
            log_entries=self.log_entries,
            raw_structure={
                'survey_name': survey_name,
                'element_count': len(elements),
                'block_count': len(blocks_map),
                'question_count': len(questions_map)
            },
            open_ended_questions=open_ended,
            attention_checks=attention_checks,
            randomizer_info=randomizer_info,
            open_ended_details=open_ended_details,
            study_context=study_context,
            embedded_data_conditions=embedded_data_conditions,
            forced_response_questions=forced_response_questions,
            comprehension_checks=comprehension_checks,
            slider_questions=slider_questions,
            text_entry_questions=text_entry_questions,
            # v1.0.0: Condition visibility mapping (CRITICAL for simulation)
            condition_visibility_map=visibility_result.get('condition_visibility_map', {}),
            condition_blocks=visibility_result.get('condition_blocks', {}),
            block_questions=visibility_result.get('block_questions', {}),
            shared_questions=visibility_result.get('shared_questions', []),
        )

    def _parse_blocks(self, element: Dict, blocks_map: Dict):
        """Parse block definitions.

        Handles all QSF block payload formats:
        - List format (newer exports): [{"ID": "BL_123", "Description": "Block 1", ...}]
        - Dict with Blocks key: {"Blocks": [...]}
        - Dict format (older exports): {"0": {"ID": "BL_123", ...}, "1": {...}}
        - Dict keyed by block ID: {"BL_123": {"Description": "Block 1", ...}}
        """
        try:
            payload = element.get('Payload', {})
            if payload is None:
                payload = {}
                self._log(LogLevel.INFO, "BLOCK", "Block payload is null, using empty dict")

            # Handle nested Blocks key (some QSF exports wrap blocks in a dict)
            if isinstance(payload, dict) and isinstance(payload.get('Blocks'), (list, dict)):
                payload = payload.get('Blocks', payload)
                self._log(LogLevel.INFO, "BLOCK", "Extracted blocks from nested 'Blocks' key")

            # Handle list format (newer QSF exports - most common)
            if isinstance(payload, list):
                self._log(LogLevel.INFO, "BLOCK", f"Processing {len(payload)} blocks in list format")
                for block_data in payload:
                    if isinstance(block_data, dict):
                        block_id = block_data.get('ID', '')
                        if not block_id:
                            continue
                        block_name = block_data.get('Description', f'Block {block_id}')
                        block_type = block_data.get('Type', 'Standard')

                        # Handle BlockElements that could be None, list, or dict
                        block_elements = block_data.get('BlockElements')
                        if block_elements is None:
                            block_elements = []
                        elif isinstance(block_elements, dict):
                            block_elements = list(block_elements.values())
                        elif not isinstance(block_elements, list):
                            block_elements = []

                        blocks_map[block_id] = {
                            'name': block_name,
                            'type': block_type,
                            'elements': block_elements
                        }

                        self._log(
                            LogLevel.INFO, "BLOCK",
                            f"Found block: {block_name} (Type: {block_type})"
                        )

            # Handle dict format (older QSF exports)
            elif isinstance(payload, dict):
                self._log(LogLevel.INFO, "BLOCK", f"Processing {len(payload)} blocks in dict format")
                for dict_key, block_data in payload.items():
                    if isinstance(block_data, dict):
                        # Use the ID field if available, otherwise use the dict key
                        # This is important because flow elements reference blocks by ID field
                        block_id = block_data.get('ID', dict_key)
                        block_name = block_data.get('Description', f'Block {block_id}')
                        block_type = block_data.get('Type', 'Standard')

                        # Handle BlockElements that could be None, list, or dict
                        block_elements = block_data.get('BlockElements')
                        if block_elements is None:
                            block_elements = []
                        elif isinstance(block_elements, dict):
                            block_elements = list(block_elements.values())
                        elif not isinstance(block_elements, list):
                            block_elements = []

                        blocks_map[block_id] = {
                            'name': block_name,
                            'type': block_type,
                            'elements': block_elements
                        }
                        # Also store with dict key for compatibility
                        if dict_key != block_id:
                            blocks_map[dict_key] = blocks_map[block_id]

                        self._log(
                            LogLevel.INFO, "BLOCK",
                            f"Found block: {block_name} (ID: {block_id}, Type: {block_type})"
                        )
            else:
                self._log(
                    LogLevel.WARNING, "BLOCK_PARSE",
                    f"Unexpected payload type: {type(payload).__name__}, expected list or dict"
                )
        except Exception as e:
            self._log(LogLevel.WARNING, "BLOCK_PARSE", f"Error parsing blocks: {e}")

    def _parse_question(self, element: Dict, questions_map: Dict):
        """Parse a survey question with robust scale point detection."""
        try:
            payload = element.get('Payload', {})
            if payload is None:
                payload = {}
            q_id = payload.get('QuestionID', element.get('PrimaryAttribute', ''))
            if not q_id:
                return  # Skip questions without ID

            question_text = payload.get('QuestionText', '') or ''
            # Clean HTML
            question_text = re.sub(r'<[^>]+>', '', str(question_text))

            question_type = payload.get('QuestionType', 'Unknown') or 'Unknown'
            selector = payload.get('Selector', '') or ''

            # Determine question category
            category = self._categorize_question(question_type, selector)

            # Extract choices (for MC questions, these are the response options)
            choices = []
            choices_data = payload.get('Choices', {})
            if choices_data is None:
                choices_data = {}
            if isinstance(choices_data, dict):
                # Sort by choice ID to maintain order
                try:
                    sorted_choices = sorted(choices_data.items(), key=lambda x: self._safe_int_key(x[0]))
                    for _, choice_data in sorted_choices:
                        if isinstance(choice_data, dict):
                            choice_text = choice_data.get('Display', str(choice_data))
                            choices.append(choice_text)
                        else:
                            choices.append(str(choice_data))
                except Exception:
                    pass  # If sorting fails, skip choices
            elif isinstance(choices_data, list):
                for choice_data in choices_data:
                    if isinstance(choice_data, dict):
                        choices.append(choice_data.get('Display', str(choice_data)))
                    else:
                        choices.append(str(choice_data))

            # Check for matrix (scale) questions
            is_matrix = question_type == 'Matrix' or (question_type == 'MC' and selector in ['Likert', 'Bipolar'])

            # Extract sub-questions for matrix (these are the rows/items)
            # v1.2.0: Fixed - For Matrix, Choices are items (rows), Answers are scale (columns)
            sub_questions = []
            scale_anchors_parsed = {}  # v1.2.0: Extract scale anchors from Answers

            # For Matrix questions, Choices are the items (rows) and Answers are the scale (columns)
            if question_type == 'Matrix':
                # The "Choices" in Matrix are the items/statements (rows to rate)
                # Extract items from Choices for sub_questions
                if isinstance(choices_data, dict):
                    try:
                        sorted_choices = sorted(choices_data.items(), key=lambda x: self._safe_int_key(x[0]))
                        for choice_id, choice_data in sorted_choices:
                            if isinstance(choice_data, dict):
                                sub_questions.append(choice_data.get('Display', str(choice_data)))
                            else:
                                sub_questions.append(str(choice_data))
                    except Exception:
                        pass  # If sorting fails, use choices list directly
                    if not sub_questions:
                        sub_questions = choices.copy()  # Fallback to already-parsed choices

                # The "Answers" are the response scale options (columns)
                # Extract these into scale_anchors with their recode values
                answers_data = payload.get('Answers', {})
                recode_values = payload.get('RecodeValues', {})
                if answers_data is None:
                    answers_data = {}
                if isinstance(answers_data, dict):
                    try:
                        sorted_answers = sorted(answers_data.items(), key=lambda x: self._safe_int_key(x[0]))
                        for ans_id, ans_data in sorted_answers:
                            if isinstance(ans_data, dict):
                                label = ans_data.get('Display', '')
                                # Use recode value if available, otherwise use answer ID
                                recode_val = recode_values.get(str(ans_id), ans_id)
                                scale_anchors_parsed[str(recode_val)] = label
                            else:
                                recode_val = recode_values.get(str(ans_id), ans_id)
                                scale_anchors_parsed[str(recode_val)] = str(ans_data)
                    except Exception:
                        pass  # If sorting fails, skip scale anchors
            else:
                # For non-matrix questions with sub-questions (e.g., side-by-side)
                answers_data = payload.get('Answers', {})
                if answers_data is None:
                    answers_data = {}
                if isinstance(answers_data, dict):
                    try:
                        sorted_answers = sorted(answers_data.items(), key=lambda x: self._safe_int_key(x[0]))
                        for ans_id, ans_data in sorted_answers:
                            if isinstance(ans_data, dict):
                                sub_questions.append(ans_data.get('Display', ''))
                    except Exception:
                        pass  # If sorting fails, skip sub-questions
                # For MC questions, extract scale anchors from choices
                if choices and self._looks_like_scale_choices(choices):
                    recode_values = payload.get('RecodeValues', {})
                    if isinstance(choices_data, dict):
                        try:
                            sorted_choices = sorted(choices_data.items(), key=lambda x: self._safe_int_key(x[0]))
                            for choice_id, choice_data in sorted_choices:
                                label = choice_data.get('Display', str(choice_data)) if isinstance(choice_data, dict) else str(choice_data)
                                recode_val = recode_values.get(str(choice_id), choice_id)
                                scale_anchors_parsed[str(recode_val)] = label
                        except Exception:
                            pass
                    else:
                        # Fallback: number the choices
                        for i, choice in enumerate(choices, 1):
                            scale_anchors_parsed[str(i)] = choice

            # ROBUST SCALE POINT DETECTION
            scale_points = self._detect_scale_points(payload, question_type, selector, choices, sub_questions)

            # Get the data export tag (actual variable name in exported data)
            data_export_tag = payload.get('DataExportTag', q_id)

            # NEW: Detect text entry choices in MC questions (e.g., "Other: ____")
            text_entry_choices = []
            if isinstance(choices_data, dict):
                for choice_id, choice_data in choices_data.items():
                    if isinstance(choice_data, dict):
                        # Check if this choice has TextEntry enabled
                        if choice_data.get('TextEntry') or choice_data.get('TextEntrySize'):
                            choice_text = choice_data.get('Display', str(choice_data))
                            text_entry_choices.append({
                                'choice_id': str(choice_id),
                                'choice_text': choice_text,
                                'variable_name': f"{data_export_tag}_{choice_id}_TEXT"
                            })
                            self._log(
                                LogLevel.INFO, "TEXT_ENTRY_CHOICE",
                                f"Found text entry in choice {choice_id} of {q_id}: {choice_text[:30]}..."
                            )

            # Check for ForceResponse validation
            force_response = False
            validation = payload.get('Validation', {})
            if isinstance(validation, dict):
                settings = validation.get('Settings', {})
                if isinstance(settings, dict):
                    force_val = settings.get('ForceResponse')
                    force_response = force_val in ['ON', True, 'on', '1', 1]
                    # Also check ForceResponseType
                    force_type = settings.get('ForceResponseType')
                    if force_type in ['ON', 'Force', 'ForceAll']:
                        force_response = True

            # v2.4.2: Track selector for precise question type detection
            selector_str = selector if selector else ''

            # v2.4.2: Extract FORM fields (TE/FORM questions have multiple text inputs)
            form_fields = []
            if question_type == 'TE' and selector == 'FORM':
                for choice_id, choice_data in (choices_data.items() if isinstance(choices_data, dict) else []):
                    if isinstance(choice_data, dict):
                        field_label = choice_data.get('Display', '')
                        # Clean HTML from label
                        field_label = re.sub(r'<[^>]+>', '', str(field_label)).strip()
                        has_text_entry = choice_data.get('TextEntry') in ['on', 'ON', True, 'true', '1', 1]
                        if has_text_entry or selector == 'FORM':
                            form_fields.append({
                                'field_id': str(choice_id),
                                'label': field_label,
                                'variable_name': f"{data_export_tag}_{choice_id}" if data_export_tag else f"{q_id}_{choice_id}"
                            })
                if form_fields:
                    self._log(
                        LogLevel.INFO, "FORM_FIELDS",
                        f"Detected {len(form_fields)} FORM fields in {q_id}: {[f['label'][:30] for f in form_fields]}"
                    )

            # v2.4.2: Detect comprehension checks (CustomValidation with expected answer)
            is_comprehension_check = False
            comprehension_expected = None
            if isinstance(validation, dict):
                settings = validation.get('Settings', {})
                if isinstance(settings, dict):
                    custom_val = settings.get('CustomValidation', {})
                    if isinstance(custom_val, dict) and 'Logic' in custom_val:
                        is_comprehension_check = True
                        # Try to extract expected answer from logic
                        logic = custom_val.get('Logic', {})
                        if isinstance(logic, dict):
                            for _, condition_group in logic.items():
                                if isinstance(condition_group, dict):
                                    for _, condition in condition_group.items():
                                        if isinstance(condition, dict):
                                            right_operand = condition.get('RightOperand')
                                            if right_operand:
                                                comprehension_expected = str(right_operand)
                                                break
                        if is_comprehension_check:
                            self._log(
                                LogLevel.INFO, "COMPREHENSION_CHECK",
                                f"Detected comprehension check: {q_id} (expected: {comprehension_expected})"
                            )

            # v2.4.3: Extract enhanced validation details from QSF training
            min_chars = None
            max_chars = None
            validation_regex = None
            number_min = None
            number_max = None
            content_type = None
            if isinstance(validation, dict):
                settings = validation.get('Settings', {})
                if isinstance(settings, dict):
                    # MinChars/MaxChars for text entry
                    if 'MinChars' in settings:
                        try:
                            min_chars = int(settings['MinChars'])
                        except (ValueError, TypeError):
                            pass
                    if 'MaxChars' in settings:
                        try:
                            max_chars = int(settings['MaxChars'])
                        except (ValueError, TypeError):
                            pass
                    # Content type validation (ValidNumber, ValidZip, etc.)
                    content_type = settings.get('ContentType')
                    # Number range validation
                    if 'Min' in settings:
                        try:
                            number_min = float(settings['Min'])
                        except (ValueError, TypeError):
                            pass
                    if 'Max' in settings:
                        try:
                            number_max = float(settings['Max'])
                        except (ValueError, TypeError):
                            pass
                    # Regex validation pattern
                    custom_val = settings.get('CustomValidation', {})
                    if isinstance(custom_val, dict):
                        # Check for regex in custom validation message or logic
                        message = custom_val.get('Message', '')
                        # Ensure message is a string before calling .lower() (QSF can have dicts here)
                        if isinstance(message, str) and ('email' in message.lower() or 'valid' in message.lower()):
                            # Common email regex pattern
                            validation_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

            # v2.4.3: Extract slider configuration
            slider_min_val = None
            slider_max_val = None
            slider_grid_lines = None
            slider_snap_to_grid = False
            slider_labels = {}
            config = payload.get('Configuration', {})
            if isinstance(config, dict) and question_type == 'Slider':
                if 'CSSliderMin' in config:
                    try:
                        slider_min_val = float(config['CSSliderMin'])
                    except (ValueError, TypeError):
                        pass
                if 'CSSliderMax' in config:
                    try:
                        slider_max_val = float(config['CSSliderMax'])
                    except (ValueError, TypeError):
                        pass
                if 'GridLines' in config:
                    try:
                        slider_grid_lines = int(config['GridLines'])
                    except (ValueError, TypeError):
                        pass
                slider_snap_to_grid = config.get('SnapToGrid', False) in [True, 'true', 'True', 1, '1']
                # Extract slider labels
                labels = config.get('Labels', {})
                if isinstance(labels, dict):
                    slider_labels = {str(k): str(v) for k, v in labels.items()}

            # v2.4.3: Detect skip/display logic
            has_skip_logic = 'SkipLogic' in payload or 'BranchLogic' in payload
            has_display_logic = 'DisplayLogic' in payload

            # v2.4.3: Detect choice randomization
            choice_randomization = False
            fixed_choices = []
            randomization = payload.get('Randomization', {})
            if isinstance(randomization, dict):
                choice_randomization = randomization.get('Advanced', {}).get('Randomize', False)
                # Get fixed choices (usually "Other" or "No preference" at end)
                fixed_positions = randomization.get('Advanced', {}).get('FixedOrder', [])
                if fixed_positions and isinstance(choices_data, dict):
                    for pos in fixed_positions:
                        if str(pos) in choices_data:
                            choice_data = choices_data[str(pos)]
                            if isinstance(choice_data, dict):
                                fixed_choices.append(choice_data.get('Display', ''))

            questions_map[q_id] = QuestionInfo(
                question_id=q_id,
                question_text=question_text[:200] + ('...' if len(question_text) > 200 else ''),
                question_type=category,
                block_name='',  # Will be filled when mapping
                choices=choices,
                scale_points=scale_points,
                is_matrix=is_matrix,
                sub_questions=sub_questions,
                text_entry_choices=text_entry_choices,
                force_response=force_response,
                export_tag=data_export_tag,
                raw_payload=payload,
                selector=selector_str,
                form_fields=form_fields,
                is_comprehension_check=is_comprehension_check,
                comprehension_expected=comprehension_expected,
                min_chars=min_chars,
                max_chars=max_chars,
                validation_regex=validation_regex,
                number_min=number_min,
                number_max=number_max,
                content_type=content_type,
                slider_min=slider_min_val,
                slider_max=slider_max_val,
                slider_grid_lines=slider_grid_lines,
                slider_snap_to_grid=slider_snap_to_grid,
                slider_labels=slider_labels,
                has_skip_logic=has_skip_logic,
                has_display_logic=has_display_logic,
                choice_randomization=choice_randomization,
                fixed_choices=fixed_choices,
                scale_anchors=scale_anchors_parsed  # v1.2.0: Properly extract scale anchors
            )

            self._log(
                LogLevel.INFO, "QUESTION",
                f"Parsed question {q_id}: {category} (selector={selector_str}, scale_points={scale_points}, force={force_response}, min_chars={min_chars})",
                {'text_preview': question_text[:100], 'data_export_tag': data_export_tag}
            )
        except Exception as e:
            self._log(LogLevel.WARNING, "QUESTION_PARSE", f"Error parsing question: {e}")

    def _safe_int_key(self, key: Any) -> int:
        """Convert key to int for sorting, handling non-numeric keys."""
        try:
            return int(key)
        except (ValueError, TypeError):
            return 0

    def _detect_scale_points(
        self,
        payload: Dict,
        question_type: str,
        selector: str,
        choices: List[str],
        sub_questions: List[str]
    ) -> Optional[int]:
        """
        Robustly detect scale points from multiple QSF sources.

        Priority order:
        1. For Matrix: Answers dict (response scale)
        2. ColumnLabels (explicit scale definition)
        3. RecodeValues (numeric mapping implies scale)
        4. Choices length (fallback for MC)
        5. Common patterns in choice text (1-7, strongly disagree, etc.)
        """
        # Source 1: For Matrix questions, Answers contains the response scale
        if question_type == 'Matrix':
            answers_data = payload.get('Answers', {})
            if isinstance(answers_data, dict) and len(answers_data) >= 2:
                return len(answers_data)

        # Source 2: ColumnLabels (some matrix formats use this)
        column_labels = payload.get('ColumnLabels', {})
        if isinstance(column_labels, dict) and len(column_labels) >= 2:
            return len(column_labels)

        # Source 3: RecodeValues can indicate scale structure
        recode_values = payload.get('RecodeValues', {})
        if isinstance(recode_values, dict) and len(recode_values) >= 2:
            # Check if it's a numeric scale
            try:
                values = [int(v) for v in recode_values.values() if str(v).isdigit()]
                if values:
                    return max(values) - min(values) + 1 if max(values) != min(values) else len(recode_values)
            except (ValueError, TypeError):
                pass
            return len(recode_values)

        # Source 4: Choices for MC/single choice questions
        if choices and len(choices) >= 2:
            # Check if these look like scale labels
            if self._looks_like_scale_choices(choices):
                return len(choices)

        # Source 5: Check SubSelector for scale type hints
        sub_selector = payload.get('SubSelector', '')
        if sub_selector:
            # Common Qualtrics scale patterns
            scale_hints = {
                'SingleAnswer': 7,  # Default assumption
                'MultipleAnswer': None,  # Not a scale
                'DL': 7,  # Dropdown default
                'TX': None,  # Text
            }
            if sub_selector in scale_hints:
                return scale_hints[sub_selector]

        # Source 6: Look for scale hints in Configuration
        config = payload.get('Configuration', {})
        if isinstance(config, dict):
            slider_min = config.get('CSSliderMin')
            slider_max = config.get('CSSliderMax')
            if slider_min is not None and slider_max is not None:
                try:
                    slider_min = int(slider_min)
                    slider_max = int(slider_max)
                    if slider_max >= slider_min:
                        return slider_max - slider_min + 1
                except (TypeError, ValueError):
                    pass

            # Some surveys specify NumChoices
            num_choices = config.get('NumChoices')
            if num_choices and isinstance(num_choices, int):
                return num_choices

        # Fallback: Use choices length if it looks reasonable for a scale
        if choices and 2 <= len(choices) <= 11:
            return len(choices)

        return None

    def _looks_like_scale_choices(self, choices: List[str]) -> bool:
        """
        Check if choices look like a Likert-type scale.
        """
        if len(choices) < 2:
            return False

        # Check for numeric patterns (1, 2, 3... or 1-Strongly Disagree)
        numeric_count = 0
        for choice in choices:
            choice_clean = str(choice).strip().lower()
            # Check if starts with number
            if choice_clean and (choice_clean[0].isdigit() or
                                 choice_clean.startswith('-') or
                                 choice_clean.startswith('(')):
                numeric_count += 1

        if numeric_count >= len(choices) * 0.5:
            return True

        # Check for common scale anchors
        scale_keywords = [
            'strongly', 'agree', 'disagree', 'likely', 'unlikely',
            'satisfied', 'dissatisfied', 'good', 'bad', 'poor', 'excellent',
            'never', 'always', 'sometimes', 'often', 'rarely',
            'not at all', 'extremely', 'very', 'somewhat', 'slightly',
            'important', 'unimportant', 'confident', 'certain', 'uncertain'
        ]

        anchor_count = 0
        for choice in choices:
            choice_lower = choice.lower()
            if any(kw in choice_lower for kw in scale_keywords):
                anchor_count += 1

        return anchor_count >= 2

    # =========================================================================
    # v1.0.0: ENHANCED SCALE DETECTION - SEMANTIC TYPE CLASSIFICATION
    # =========================================================================

    # Well-known psychological scales with their patterns
    WELL_KNOWN_SCALES = {
        'big_five': {
            'patterns': [r'big\s*five', r'ocean', r'personality', r'extraversion', r'agreeableness',
                        r'conscientiousness', r'neuroticism', r'openness', r'bfi', r'ipip', r'neo'],
            'items': 5,  # Minimum expected items for short form
            'domains': ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness'],
            'scale_points': [5, 7],
        },
        'panas': {
            'patterns': [r'panas', r'positive\s*affect', r'negative\s*affect', r'emotional\s*state'],
            'items': 10,  # 10-item PANAS short form
            'domains': ['Positive Affect', 'Negative Affect'],
            'scale_points': [5, 7],
        },
        'rosenberg_self_esteem': {
            'patterns': [r'self\s*esteem', r'rosenberg', r'rse', r'self\s*worth'],
            'items': 10,
            'scale_points': [4, 5],
        },
        'satisfaction_with_life': {
            'patterns': [r'satisfaction\s*with\s*life', r'swls', r'life\s*satisfaction', r'diener'],
            'items': 5,
            'scale_points': [7],
        },
        'perceived_stress': {
            'patterns': [r'perceived\s*stress', r'pss', r'stress\s*scale'],
            'items': 10,  # PSS-10
            'scale_points': [5],
        },
        'trust_propensity': {
            'patterns': [r'trust\s*propensity', r'generalized\s*trust', r'interpersonal\s*trust'],
            'items': 5,
            'scale_points': [5, 7],
        },
        'risk_propensity': {
            'patterns': [r'risk\s*propensity', r'risk\s*taking', r'dospert', r'risk\s*attitude'],
            'items': 6,
            'scale_points': [5, 7],
        },
        'need_for_cognition': {
            'patterns': [r'need\s*for\s*cognition', r'nfc', r'thinking\s*preference'],
            'items': 18,
            'scale_points': [5, 9],
        },
        'social_desirability': {
            'patterns': [r'social\s*desirability', r'marlowe\s*crowne', r'impression\s*management'],
            'items': 10,
            'scale_points': [2, 5, 7],
        },
        'regulatory_focus': {
            'patterns': [r'regulatory\s*focus', r'promotion\s*focus', r'prevention\s*focus', r'rfq'],
            'items': 11,
            'scale_points': [5, 7],
        },
    }

    # Semantic scale type patterns
    SEMANTIC_TYPE_PATTERNS = {
        'satisfaction': [r'satisf', r'pleased', r'content(?:ment)?', r'happy\s*with'],
        'trust': [r'trust', r'reliabl', r'dependab', r'faith\s*in'],
        'intention': [r'intention', r'intend', r'likely\s*to', r'plan\s*to', r'will\s*i'],
        'attitude': [r'attitude', r'feel\s*about', r'opinion', r'view\s*of'],
        'preference': [r'prefer', r'like\s*better', r'rather', r'favorite'],
        'risk': [r'risk', r'danger', r'threat', r'hazard', r'uncertain'],
        'anxiety': [r'anxious', r'nervous', r'worried', r'fear', r'apprehension'],
        'efficacy': [r'efficac', r'capable', r'able\s*to', r'confident\s*in\s*ability'],
        'engagement': [r'engag', r'involv', r'participat', r'commit'],
        'motivation': [r'motivat', r'driven', r'desire', r'willing'],
        'fairness': [r'fair', r'just', r'equitab', r'impartial'],
        'identification': [r'identif', r'belong', r'part\s*of', r'member'],
        'quality': [r'quality', r'excellent', r'superior', r'well\s*made'],
        'willingness_to_pay': [r'willing\s*to\s*pay', r'wtp', r'pay\s*for', r'spend'],
        'purchase_intention': [r'buy', r'purchase', r'acquire', r'shopping'],
        'recommendation': [r'recommend', r'suggest', r'refer', r'tell\s*friends'],
    }

    # Mediation-related keywords
    MEDIATOR_KEYWORDS = [
        'perceived', 'feelings', 'thoughts', 'reaction', 'response',
        'interpretation', 'judgment', 'evaluation', 'assessment',
        'attribution', 'expectation', 'belief', 'attitude',
        'mechanism', 'process', 'mediating', 'underlying',
        'explanation', 'reason', 'why', 'how'
    ]

    def _detect_semantic_scale_type(self, question_text: str, variable_name: str, choices: List[str]) -> Optional[str]:
        """Detect the semantic type of a scale based on text content."""
        combined_text = f"{question_text} {variable_name} {' '.join(choices)}".lower()

        for sem_type, patterns in self.SEMANTIC_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, combined_text):
                    return sem_type
        return None

    def _detect_well_known_scale(self, questions: List[QuestionInfo],
                                  scale_name: str) -> Optional[Dict[str, Any]]:
        """Detect if a scale matches a well-known psychological scale."""
        combined_text = scale_name.lower()

        # Also check question texts
        for q in questions:
            combined_text += f" {q.question_text.lower()}"

        for scale_id, scale_info in self.WELL_KNOWN_SCALES.items():
            for pattern in scale_info['patterns']:
                if re.search(pattern, combined_text):
                    return {
                        'scale_id': scale_id,
                        'expected_items': scale_info['items'],
                        'expected_scale_points': scale_info['scale_points'],
                        'domains': scale_info.get('domains', []),
                        'match_confidence': 'high' if len([p for p in scale_info['patterns']
                                                          if re.search(p, combined_text)]) > 1 else 'medium'
                    }
        return None

    def _detect_reverse_coded_items(self, question_text: str, choices: List[str]) -> bool:
        """Detect if an item is likely reverse-coded."""
        reverse_indicators = [
            r'\(r\)', r'\(reversed\)', r'\(rev\)', r'_r$', r'_rev$',
            r'not\s+at\s+all', r'never', r'negative', r'bad', r'poor',
            r'disagree', r'unlikely', r'difficult', r'hard', r'impossible'
        ]

        text_lower = question_text.lower()
        for indicator in reverse_indicators:
            if re.search(indicator, text_lower):
                return True
        return False

    def _extract_scale_anchors(self, choices: List[str]) -> Dict[str, str]:
        """Extract scale anchors from choice list."""
        anchors = {}
        if not choices:
            return anchors

        # Assume ordered choices
        for i, choice in enumerate(choices, 1):
            anchors[str(i)] = str(choice).strip()
        return anchors

    def _extract_scale_range(self, scale_anchors: Dict[str, str], scale_points: Optional[int]) -> Tuple[int, int]:
        """
        Extract scale_min and scale_max from scale_anchors keys.

        v1.2.0: New method to properly detect scale ranges from QSF RecodeValues.

        The scale_anchors dict has keys that are the actual recode values (e.g., "1", "2", "7")
        which represent the numeric values in the exported data. This method extracts
        the minimum and maximum from these keys.

        Args:
            scale_anchors: Dictionary with recode values as keys and labels as values
            scale_points: Number of scale points (fallback if no anchors)

        Returns:
            Tuple of (scale_min, scale_max)
        """
        if scale_anchors:
            try:
                # Extract numeric keys from scale_anchors
                numeric_keys = []
                for key in scale_anchors.keys():
                    try:
                        numeric_keys.append(int(key))
                    except (ValueError, TypeError):
                        # Handle non-numeric keys (e.g., "NA", "DK")
                        pass
                if numeric_keys:
                    return min(numeric_keys), max(numeric_keys)
            except Exception:
                pass

        # Fallback to default 1 to scale_points range
        if scale_points is not None:
            return 1, scale_points

        # Ultimate fallback
        return 1, 7

    def _detect_potential_mediator(self, q_info: QuestionInfo, block_position: int,
                                    total_blocks: int) -> Tuple[bool, List[str]]:
        """Detect if a question is likely a potential mediator variable."""
        hints = []
        is_mediator = False

        # Check question text for mediator keywords
        text_lower = q_info.question_text.lower()
        matches = [kw for kw in self.MEDIATOR_KEYWORDS if kw in text_lower]
        if matches:
            hints.extend([f"Contains mediator keyword: {kw}" for kw in matches[:3]])

        # Mediators often appear between manipulation and DV (middle of survey)
        relative_position = block_position / max(1, total_blocks)
        if 0.3 < relative_position < 0.7:
            hints.append("Positioned between manipulation and outcome (typical mediator location)")

        # Scale types that are often mediators
        mediator_scale_types = ['perception', 'attitude', 'evaluation', 'interpretation', 'feeling']
        if any(t in text_lower for t in mediator_scale_types):
            hints.append("Scale type commonly used as mediator")

        is_mediator = len(hints) >= 2
        return is_mediator, hints

    def _compute_scale_quality_score(self, scale_info: Dict[str, Any],
                                      questions: List[QuestionInfo]) -> Dict[str, Any]:
        """Compute quality metrics for a detected scale."""
        quality = {
            'completeness': 1.0,
            'consistency': 1.0,
            'detection_confidence': 'high',
            'warnings': [],
            'recommendations': []
        }

        # Check item count
        items = scale_info.get('items', 1)
        if items < 3:
            quality['warnings'].append("Single or 2-item scale may have low reliability")
            quality['completeness'] *= 0.7

        # Check scale points consistency
        scale_points = scale_info.get('scale_points')
        if scale_points is None:
            quality['warnings'].append("Scale points not detected from QSF")
            quality['detection_confidence'] = 'medium'
        elif scale_points not in [5, 7, 9, 11]:
            quality['recommendations'].append(f"Unusual scale points ({scale_points}), verify correct")

        # Check for reverse-coded items if multi-item
        if items >= 3:
            has_reverse = any(q.is_reverse_coded for q in questions if hasattr(q, 'is_reverse_coded'))
            if not has_reverse:
                quality['recommendations'].append("Consider adding reverse-coded items to reduce acquiescence bias")

        return quality

    # =========================================================================
    # v1.0.0: SKIP LOGIC AND DISPLAY LOGIC PARSING
    # =========================================================================

    def _parse_display_logic(self, display_logic: Dict[str, Any]) -> Dict[str, Any]:
        """Parse DisplayLogic from QSF into structured format."""
        if not display_logic:
            return {}

        parsed = {
            'type': display_logic.get('Type', 'Unknown'),
            'conditions': [],
            'depends_on': [],
        }

        # Parse the logic conditions
        logic_conditions = display_logic.get('0', {})
        if isinstance(logic_conditions, dict):
            # Extract question dependencies
            choice_locator = logic_conditions.get('ChoiceLocator', '')
            if choice_locator:
                # Format: "q://QID123/SelectableChoice/1"
                match = re.search(r'q://([^/]+)/', choice_locator)
                if match:
                    parsed['depends_on'].append(match.group(1))

            question_id = logic_conditions.get('QuestionID', '')
            if question_id:
                parsed['depends_on'].append(question_id)

            parsed['conditions'].append({
                'logic_type': logic_conditions.get('LogicType', ''),
                'choice_locator': choice_locator,
                'operator': logic_conditions.get('Operator', ''),
                'question_id': question_id,
            })

        return parsed

    def _parse_skip_logic(self, skip_logic: Dict[str, Any]) -> Dict[str, Any]:
        """Parse SkipLogic from QSF into structured format."""
        if not skip_logic:
            return {}

        parsed = {
            'type': 'SkipLogic',
            'skip_to': None,
            'conditions': [],
        }

        # SkipLogic typically has SkipToDestination
        skip_dest = skip_logic.get('SkipToDestination', '')
        if skip_dest:
            parsed['skip_to'] = skip_dest

        # Parse conditions
        for key, value in skip_logic.items():
            if isinstance(value, dict) and 'LogicType' in value:
                parsed['conditions'].append({
                    'logic_type': value.get('LogicType', ''),
                    'choice': value.get('Choice', ''),
                    'question_id': value.get('QuestionID', ''),
                })

        return parsed

    def _build_question_dependency_graph(self, questions_map: Dict[str, QuestionInfo]) -> Dict[str, List[str]]:
        """Build a dependency graph from skip/display logic."""
        dependencies = {}

        for q_id, q_info in questions_map.items():
            deps = []

            # Add dependencies from display logic
            if q_info.display_logic_details:
                deps.extend(q_info.display_logic_details.get('depends_on', []))

            # Add dependencies from skip logic
            if q_info.skip_logic_details:
                for cond in q_info.skip_logic_details.get('conditions', []):
                    if cond.get('question_id'):
                        deps.append(cond['question_id'])

            if deps:
                dependencies[q_id] = list(set(deps))

        return dependencies

    def _categorize_question(self, q_type: str, selector: str) -> str:
        """Categorize question type with enhanced selector support.

        v2.4.3: Enhanced to handle more selector types from QSF analysis:
        - SAVR: Single Answer Vertical Rows (standard radio)
        - SAHR: Single Answer Horizontal Rows
        - DL: Dropdown Likert (matrix dropdown)
        - GRB: Graphic Block (images/stimuli)
        - TB: Text Block (instructions)
        - ESTB: Extended Single Text Box
        - SL: Single Line text
        - ML: Multi-Line text
        - FORM: Form with labeled rows
        """
        if q_type == 'Matrix':
            if selector == 'DL':
                return 'Matrix Dropdown'
            elif selector == 'Profile':
                return 'Matrix Profile'
            return 'Likert Scale Matrix'
        elif q_type == 'MC':
            if selector in ['Likert', 'SAVR']:
                return 'Single Choice (Radio)'
            elif selector == 'SAHR':
                return 'Single Choice (Horizontal)'
            elif selector in ['MAVR', 'MACOL']:
                return 'Multiple Choice'
            elif selector == 'DL':
                return 'Dropdown'
            else:
                return 'Multiple Choice'
        elif q_type == 'TE':
            if selector == 'ESTB':
                return 'Text Entry (Essay)'
            elif selector == 'ML':
                return 'Text Entry (Multi-line)'
            elif selector == 'FORM':
                return 'Text Entry (Form)'
            return 'Text Entry'
        elif q_type == 'Slider':
            return 'Slider'
        elif q_type == 'DB':
            if selector == 'GRB':
                return 'Graphic Block'
            return 'Descriptive Text'
        elif q_type == 'Timing':
            return 'Timing'
        elif q_type == 'Captcha':
            return 'Captcha'
        else:
            return f'{q_type} ({selector})'

    def _parse_embedded_data(self, element: Dict, fields: List[str]):
        """Parse embedded data fields."""
        payload = element.get('Payload', {})
        flow = payload.get('Flow', payload if isinstance(payload, list) else [])
        flow = self._normalize_flow(flow)

        for item in flow:
            field_name = item.get('EmbeddedData', [])
            if isinstance(field_name, list):
                for fd in field_name:
                    if isinstance(fd, dict):
                        fields.append(fd.get('Field', 'Unknown'))
            elif isinstance(field_name, str):
                fields.append(field_name)

        if fields:
            self._log(
                LogLevel.INFO, "EMBEDDED_DATA",
                f"Found {len(fields)} embedded data fields",
                {'fields': fields}
            )

    def _map_questions_to_blocks(
        self,
        blocks_map: Dict,
        questions_map: Dict
    ) -> List[BlockInfo]:
        """Map questions to their containing blocks."""
        blocks = []

        # Safety check - ensure blocks_map is a dict
        if not isinstance(blocks_map, dict):
            self._log(LogLevel.WARNING, "BLOCKS", f"blocks_map is {type(blocks_map).__name__}, expected dict")
            return blocks

        for block_id, block_data in blocks_map.items():
            if not isinstance(block_data, dict):
                continue

            block_questions = []

            block_elements = block_data.get('elements', [])
            if isinstance(block_elements, dict):
                block_elements = list(block_elements.values())
            elif not isinstance(block_elements, list):
                block_elements = []

            for elem in block_elements:
                q_id = ''
                if isinstance(elem, dict) and elem.get('Type') == 'Question':
                    q_id = elem.get('QuestionID', '') or elem.get('ID', '')
                elif isinstance(elem, str):
                    q_id = elem

                if q_id and q_id in questions_map:
                    q_info = questions_map[q_id]
                    q_info.block_name = block_data.get('name', '')
                    block_questions.append(q_info)

            block_name = block_data.get('name', f'Block {block_id}')
            block_type = block_data.get('type', 'Standard')
            is_randomizer = 'random' in block_name.lower()

            blocks.append(BlockInfo(
                block_id=block_id,
                block_name=block_name,
                questions=block_questions,
                is_randomizer=is_randomizer,
                block_type=block_type
            ))

        return blocks

    def _is_excluded_block_name(self, block_name: str, block_type: str = "") -> bool:
        """Check if a block name should be excluded from condition detection.

        This comprehensive check ensures trash, unused, and administrative blocks
        are NEVER considered as experimental conditions.

        v1.0.0: Enhanced with training from 210 QSF files.

        Args:
            block_name: The name/description of the block
            block_type: The block type field (e.g., 'Trash', 'Default', 'Standard')

        Returns:
            True if the block should be excluded, False otherwise
        """
        # Empty or whitespace-only names are always excluded
        if not block_name or not block_name.strip():
            return True

        normalized = block_name.lower().strip()

        # Check block type first (highest priority exclusion)
        if block_type:
            if block_type in self.EXCLUDED_BLOCK_TYPES:
                return True
            if block_type.lower() in {'trash', 'default', 'standard'}:
                return True

        # Check exact match against exclusion list
        if normalized in self.EXCLUDED_BLOCK_NAMES:
            return True

        # Check against exclusion patterns (catches variations)
        for pattern in self.EXCLUDED_BLOCK_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                return True

        # Additional checks for common variations
        # Catch "Block N" patterns more broadly
        if re.match(r'^block\s*\d*$', normalized):
            return True
        if re.match(r'^bl?k?\s*\d+$', normalized):  # B1, Blk1, etc.
            return True

        # Catch blocks that contain "trash" or "unused" anywhere
        if 'trash' in normalized or 'unused' in normalized:
            return True

        # Catch "(copy)" or "(duplicate)" suffixes that indicate duplicates
        # Note: "(new)" is kept as it's often used for revised condition blocks
        if re.search(r'\((?:copy|duplicate)\)', normalized):
            return True

        # v1.0.0: Catch "old" suffix patterns (e.g., "Coach Condition Old")
        if re.search(r'\bold\b', normalized):
            return True

        # v1.0.0: Catch numbered structural blocks that aren't conditions
        # e.g., "Punisher 45", "Y1", "ET_3"
        if re.match(r'^(?:punisher|y|et_?)\s*\d+$', normalized):
            return True

        return False

    def _is_trash_block_name(self, block_name: str) -> bool:
        """Check if a block name indicates trash/unused content.

        v1.0.0: Minimal check for use inside BlockRandomizers where blocks
        are conditions by definition. Only excludes obviously trash blocks.

        This is LESS aggressive than _is_excluded_block_name() because
        blocks inside a BlockRandomizer ARE conditions, even if they have
        generic names like "Block A" or "Block DB".

        Args:
            block_name: The name/description of the block

        Returns:
            True if the block is clearly trash/unused, False otherwise
        """
        if not block_name or not block_name.strip():
            return True

        normalized = block_name.lower().strip()

        # Only exclude blocks with explicit trash/unused indicators
        trash_indicators = [
            'trash', 'unused', 'delete', 'remove', 'deprecated',
            'archive', 'old version', 'do not use', 'donotuse',
            'ignore', 'hidden', 'disabled', 'inactive',
            'pilot', 'practice', 'training', 'calibration',
        ]

        for indicator in trash_indicators:
            if indicator in normalized:
                return True

        # Check for explicit trash patterns
        trash_patterns = [
            r'^\s*trash\s*',
            r'\btrash\b',
            r'\bunused\b',
            r'\bdelete\b',
            r'\bremove\b',
            r'\(delete\)',
            r'\(unused\)',
            r'\(old\)',
            r'_old$',
            r'_trash$',
            r'_unused$',
        ]

        for pattern in trash_patterns:
            if re.search(pattern, normalized):
                return True

        return False

    def _has_condition_keywords(self, block_name: str) -> bool:
        """Check if a block name contains positive condition indicators.

        v1.0.0: Added to improve condition detection by recognizing
        common experimental condition naming patterns.

        Args:
            block_name: The name/description of the block

        Returns:
            True if the block name contains condition keywords
        """
        if not block_name:
            return False

        normalized = block_name.lower()
        words = set(re.findall(r'\b\w+\b', normalized))

        # Check for positive condition keywords
        for keyword in self.CONDITION_POSITIVE_KEYWORDS:
            if keyword in words or keyword in normalized:
                return True

        # Check for condition-like patterns
        condition_patterns = [
            r'condition\s*\d+',  # "Condition 1", "Condition 2"
            r'treatment\s*\d*',  # "Treatment", "Treatment 1"
            r'group\s*\d+',  # "Group 1", "Group 2"
            r'arm\s*[a-z\d]+',  # "Arm A", "Arm 1"
            r'(?:high|low)\s+\w+',  # "High Risk", "Low Effort"
            r'\w+\s+(?:vs?\.?|×|x)\s+\w+',  # "AI vs Human", "Male × Female"
            r'(?:ai|human)[_\s/]',  # "AI/", "Human_", "AI "
            r'(?:positive|negative)\s+\w+',  # "Positive Frame"
            r'(?:empirical|normative)\s*\w*',  # "Empirical Norm"
            r'(?:control|treatment)\s*(?:group|condition)?',  # "Control Group"
        ]

        for pattern in condition_patterns:
            if re.search(pattern, normalized):
                return True

        return False

    def _score_condition_likelihood(self, block_name: str) -> float:
        """Score how likely a block name represents an experimental condition.

        v1.0.0: Added for improved condition detection ranking.

        Args:
            block_name: The name/description of the block

        Returns:
            Score from 0.0 (unlikely condition) to 1.0 (definitely condition)
        """
        if not block_name:
            return 0.0

        # Start with base score
        score = 0.0
        normalized = block_name.lower()

        # Strong positive indicators (+0.3 each)
        strong_positives = [
            r'\bcondition\s*\d+',
            r'\btreatment\s*\d*\b',
            r'\bcontrol\s*(?:group|condition)?\b',
            r'\b(?:ai|human)\s*[×x/]\s*(?:ai|human|\w+)',
            r'\b(?:high|low)\s*(?:ses|stakes|urgency|performance)',
        ]
        for pattern in strong_positives:
            if re.search(pattern, normalized):
                score += 0.3

        # Moderate positive indicators (+0.2 each)
        moderate_positives = [
            r'\b(?:empirical|normative|injunctive|descriptive)\b',
            r'\b(?:prosocial|antisocial)\b',
            r'\b(?:honest|dishonest)\b',
            r'\b(?:male|female)\s*(?:treatment|condition)?',
            r'\b(?:accurate|inaccurate)\s*(?:ai|human)',
            r'\b(?:positive|negative)\s*(?:frame|message|default)',
        ]
        for pattern in moderate_positives:
            if re.search(pattern, normalized):
                score += 0.2

        # Weak positive indicators (+0.1 each)
        weak_positives = [
            r'\b(?:hedonic|utilitarian)\b',
            r'\b(?:priming|prime|primed)\b',
            r'\b(?:frame|framing)\b',
            r'\b(?:nudge|default)\b',
            r'\bmessage\s*\d*\b',
        ]
        for pattern in weak_positives:
            if re.search(pattern, normalized):
                score += 0.1

        # Cap at 1.0
        return min(score, 1.0)

    def _detect_conditions(
        self,
        flow_data: Optional[Any],
        blocks: List[BlockInfo]
    ) -> List[str]:
        """Detect experimental conditions using comprehensive 15+ pattern detection.

        This method uses the RandomizationPatternDetector to identify conditions
        across all known Qualtrics randomization patterns:

        1. BlockRandomizer (SubSet=1) - Standard between-subjects design
        2. BlockRandomizer (SubSet>1) - Within-subjects/mixed designs
        3. Randomizer - General randomizer element
        4. EmbeddedData in Randomizer - Conditions set via EmbeddedData fields
        5. Branch-based conditions - Using Branch elements with display logic
        6. Nested randomizers - Factorial designs with multiple levels
        7. Group randomizers - Randomization at group level
        8. Question randomization - Randomizing question order within blocks
        9. Quota-based conditions - Using Quotas for participant assignment
        10. WebService conditions - External API-based assignment
        11. RandomInteger/RandomNumber - Numeric randomization for conditions
        12. Authenticator-based conditions - Based on authentication results
        13. Reference Survey conditions - Conditions from other surveys
        14. EndSurvey branches - Different endings based on conditions
        15. Piped text conditions - Dynamic text based on randomized values

        Blocks are EXCLUDED if:
        - Their name is in EXCLUDED_BLOCK_NAMES
        - Their block type is 'Trash' or 'Default'
        - They have generic names like 'Block 1', 'Block 2'
        """
        conditions: List[str] = []
        blocks_by_id = {block.block_id: block.block_name for block in blocks}

        # Use the comprehensive pattern detector
        if flow_data:
            def log_callback(message, details=None):
                self._log(LogLevel.INFO, "PATTERN_DETECT", message, details)

            detector = RandomizationPatternDetector(logger_callback=log_callback)
            # v1.0.0: Use _is_trash_block_name for blocks inside randomizers
            # because blocks inside BlockRandomizer ARE conditions by definition,
            # only obvious trash blocks should be excluded (not generic patterns)
            result = detector.detect_all_patterns(
                flow_data=flow_data,
                blocks_by_id=blocks_by_id,
                normalize_flow_func=self._normalize_flow,
                extract_payload_func=self._extract_flow_payload,
                excluded_func=self._is_trash_block_name,
            )

            # Get conditions from detector
            conditions = result.get('conditions', [])

            # Log detection results
            num_patterns = result.get('num_patterns', 0)
            design_type = result.get('design_type', 'unknown')
            is_factorial = result.get('is_factorial', False)

            if num_patterns > 0:
                self._log(
                    LogLevel.INFO, "CONDITIONS",
                    f"Detected {num_patterns} randomization patterns (design: {design_type}, factorial: {is_factorial})",
                    {'patterns': [p.get('type') for p in result.get('patterns', [])]}
                )

            # Store embedded conditions for later use
            if result.get('embedded_conditions'):
                self._log(
                    LogLevel.INFO, "EMBEDDED_CONDITIONS",
                    f"Found {len(result['embedded_conditions'])} embedded data conditions"
                )

        # v1.0.0: Filter out only truly trash blocks from conditions
        # Use less aggressive filter since conditions from randomizers are valid
        conditions = [c for c in conditions if not self._is_trash_block_name(c)]

        # v1.0.0: Safety heuristic - if condition count is very high (>30),
        # the parser likely picked up stimulus iterations / within-subjects
        # blocks rather than between-subjects conditions.  Apply aggressive
        # filter as secondary pass and keep only truly condition-like names.
        if len(conditions) > 30:
            aggressive_filtered = [
                c for c in conditions if not self._is_excluded_block_name(c)
            ]
            # Only use aggressive filter if it leaves at least 2 conditions
            if len(aggressive_filtered) >= 2:
                self._log(
                    LogLevel.INFO, "CONDITIONS",
                    f"Reduced {len(conditions)} detected conditions to "
                    f"{len(aggressive_filtered)} after aggressive filtering "
                    f"(likely stimulus iteration blocks removed)"
                )
                conditions = aggressive_filtered

        # Fallback: use block names that look like conditions (STRICT mode).
        # Only use if no conditions were found from pattern detection
        if not conditions:
            self._log(
                LogLevel.INFO, "CONDITIONS",
                "No conditions detected from patterns. Using fallback block name detection."
            )
            for block in blocks:
                desc = block.block_name.strip()
                # Skip excluded blocks
                if self._is_excluded_block_name(desc):
                    continue
                # Skip trash/default block types
                block_type = getattr(block, 'block_type', 'Standard')
                if block_type in self.EXCLUDED_BLOCK_TYPES:
                    continue
                # Only add if it strongly looks like a condition
                if self._looks_like_condition(desc):
                    self._add_condition(desc, conditions)

        conditions = self._dedupe_conditions(conditions)

        if not conditions:
            self._log(
                LogLevel.WARNING, "CONDITIONS",
                "No experimental conditions automatically detected. "
                "Please define conditions manually."
            )

        return conditions

    def _find_conditions_in_flow(self, flow: List, conditions: List[str], blocks_by_id: Dict[str, str]):
        """Recursively find conditions in flow structure.

        Key logic:
        - Only consider blocks inside Randomizer or BlockRandomizer elements
        - Require at least 2 blocks in a randomizer for conditions
        - Skip excluded block names
        - BlockRandomizer with SubSet=1 means between-subjects (each block = condition)
        """
        for item in flow:
            if isinstance(item, dict):
                flow_type = item.get('Type', '')

                if flow_type == 'Randomizer':
                    # Standard Randomizer element
                    sub_flow = item.get('Flow', [])
                    sub_flow = self._normalize_flow(sub_flow)

                    # Collect potential conditions from this randomizer
                    randomizer_conditions = []
                    for sub_item in sub_flow:
                        block_id = self._extract_block_id(sub_item)
                        if block_id:
                            block_name = blocks_by_id.get(block_id, block_id)
                            if not self._is_excluded_block_name(block_name):
                                randomizer_conditions.append(block_name)
                        if sub_item.get('Type') == 'Group':
                            group_name = sub_item.get('Description', '')
                            if group_name and not self._is_excluded_block_name(group_name):
                                randomizer_conditions.append(group_name)

                    # Only add if there are at least 2 conditions (real randomization)
                    if len(randomizer_conditions) >= 2:
                        for cond in randomizer_conditions:
                            self._add_condition(cond, conditions)
                        self._log(
                            LogLevel.INFO, "RANDOMIZER",
                            f"Found Randomizer with {len(randomizer_conditions)} conditions: {randomizer_conditions}"
                        )
                    elif len(randomizer_conditions) == 1:
                        self._log(
                            LogLevel.INFO, "RANDOMIZER",
                            f"Skipping single-block Randomizer (no real randomization): {randomizer_conditions}"
                        )

                    # Also check description for condition hints
                    description = item.get('Description', '')
                    for inferred in self._extract_conditions_from_description(description):
                        if not self._is_excluded_block_name(inferred):
                            self._add_condition(inferred, conditions)

                elif flow_type == 'BlockRandomizer':
                    # BlockRandomizer with SubSet=1 means between-subjects assignment
                    # Each block in the randomizer is a condition
                    sub_set = item.get('SubSet', None)
                    # Handle both string and int types for SubSet
                    is_between_subjects = (
                        sub_set is None or
                        sub_set == 1 or
                        str(sub_set) == '1'
                    )

                    # Collect blocks in this randomizer
                    # v1.0.0: Blocks inside BlockRandomizer ARE conditions by definition,
                    # only exclude obvious trash/unused blocks (not generic "block X" names)
                    sub_flow = item.get('Flow', [])
                    sub_flow = self._normalize_flow(sub_flow)
                    randomizer_conditions = []

                    for sub_item in sub_flow:
                        if isinstance(sub_item, dict):
                            sub_type = sub_item.get('Type', '')
                            if sub_type in ('Standard', 'Block'):
                                block_id = sub_item.get('ID', '')
                                if block_id:
                                    block_name = blocks_by_id.get(block_id, block_id)
                                    # v1.0.0: Inside BlockRandomizer, only exclude truly trash blocks
                                    # (containing "trash", "unused", "delete", etc.)
                                    # NOT generic patterns like "block X" - those are likely conditions
                                    if not self._is_trash_block_name(block_name):
                                        randomizer_conditions.append(block_name)

                    # Only add as conditions if:
                    # 1. It's between-subjects (SubSet=1)
                    # 2. There are at least 2 blocks (real randomization)
                    if is_between_subjects and len(randomizer_conditions) >= 2:
                        for cond in randomizer_conditions:
                            self._add_condition(cond, conditions)
                        self._log(
                            LogLevel.INFO, "BLOCK_RANDOMIZER",
                            f"Found BlockRandomizer with {len(randomizer_conditions)} conditions: {randomizer_conditions}"
                        )
                    elif len(randomizer_conditions) == 1:
                        self._log(
                            LogLevel.INFO, "BLOCK_RANDOMIZER",
                            f"Skipping single-block BlockRandomizer: {randomizer_conditions}"
                        )
                    elif not is_between_subjects:
                        self._log(
                            LogLevel.INFO, "BLOCK_RANDOMIZER",
                            f"BlockRandomizer is within-subjects (SubSet != 1), not treating as conditions"
                        )

                elif flow_type == 'Branch':
                    description = item.get('Description', '')
                    for inferred in self._extract_conditions_from_description(description):
                        if not self._is_excluded_block_name(inferred):
                            self._add_condition(inferred, conditions)

                # Recurse into nested flows
                if 'Flow' in item:
                    self._find_conditions_in_flow(self._normalize_flow(item['Flow']), conditions, blocks_by_id)

    def _extract_block_id(self, item: Dict[str, Any]) -> str:
        """Extract block ID from a flow item."""
        if item.get('Type') == 'Block':
            return item.get('ID', '') or item.get('BlockID', '')
        if item.get('Type') == 'Group':
            return ''
        if item.get('Type') == 'BlockRandomizer':
            return item.get('BlockID', '') or item.get('ID', '')
        return item.get('ID', '') if item.get('ID') and item.get('Type') in {'Block', 'BlockRandomizer'} else ''

    def _extract_conditions_from_description(self, description: str) -> List[str]:
        """Extract condition labels from flow descriptions."""
        if not description:
            return []

        desc = self._normalize_condition_label(description)
        if not desc:
            return []

        tokens = re.split(r"\b(?:vs\.?|versus|v\.|compared to|compared with)\b", desc, flags=re.IGNORECASE)
        if len(tokens) > 1:
            return [t.strip() for t in tokens if t.strip()]

        if self._looks_like_condition(desc):
            return [desc]

        return []

    def _looks_like_condition(self, label: str) -> bool:
        """Check if a label strongly suggests an experimental condition.

        This is used as a FALLBACK when no randomizers are detected.
        We use strict criteria to avoid false positives.
        """
        if not label:
            return False

        # First check exclusion list
        if self._is_excluded_block_name(label):
            return False

        lowered = label.lower()

        # Strong positive indicators - these terms strongly suggest conditions
        strong_keywords = (
            "condition", "treatment", "control", "experimental",
            "manipulation", "scenario", "stimulus", "stimuli"
        )

        # Weaker indicators - need additional context
        weak_keywords = ("group", "arm", "variant")

        # Check for strong keywords
        if any(keyword in lowered for keyword in strong_keywords):
            return True

        # Check for weaker keywords with additional context
        # e.g., "treatment group" is good, but "age group" is not
        for keyword in weak_keywords:
            if keyword in lowered:
                # Check if it has condition-related context
                condition_context = ("treatment", "control", "experimental", "test", "condition")
                if any(ctx in lowered for ctx in condition_context):
                    return True
                # Check if it follows pattern like "Group A", "Group 1", etc.
                if re.search(rf'{keyword}\s*[a-z0-9]', lowered, re.IGNORECASE):
                    return True

        return False

    def _normalize_condition_label(self, label: str) -> str:
        if not label:
            return ""
        cleaned = re.sub(r"<[^>]+>", "", str(label))
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(
            r"^(condition|treatment|group|arm|variant|scenario|manipulation)\s*[:\-]\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned.strip(" -:")

    def _add_condition(self, label: str, conditions: List[str]) -> None:
        normalized = self._normalize_condition_label(label)
        if not normalized:
            return
        conditions.append(normalized)
        self._log(LogLevel.INFO, "CONDITION", f"Detected potential condition: {normalized}")

    def _build_condition_visibility_map(
        self,
        conditions: List[str],
        blocks: List[BlockInfo],
        questions_map: Dict[str, QuestionInfo],
        flow_data: Any
    ) -> Dict[str, Any]:
        """Build comprehensive condition-to-question visibility mapping.

        v1.0.0: CRITICAL for simulation - determines which questions each condition sees.

        This method analyzes:
        1. BlockRandomizer structure to determine which blocks each condition sees
        2. DisplayLogic on individual questions
        3. BranchLogic in survey flow
        4. Shared blocks (outside randomizers) that all conditions see

        Returns:
            Dict with keys:
            - condition_visibility_map: {condition: {question_id: True/False}}
            - condition_blocks: {condition: [block_ids]}
            - block_questions: {block_id: [question_ids]}
            - shared_questions: [question_ids visible to all]
        """
        # Initialize structures
        condition_visibility_map: Dict[str, Dict[str, bool]] = {c: {} for c in conditions}
        condition_blocks: Dict[str, List[str]] = {c: [] for c in conditions}
        block_questions: Dict[str, List[str]] = {}
        all_question_ids = list(questions_map.keys())

        # v1.0.0: Deduplicate blocks by name (same block can have BL_xxx and numeric IDs)
        # Build unique blocks map by name to avoid duplicates
        unique_blocks_by_name: Dict[str, BlockInfo] = {}
        block_id_to_canonical: Dict[str, str] = {}  # Map any ID format to canonical ID

        for block in blocks:
            block_name = block.block_name or block.block_id
            if block_name not in unique_blocks_by_name:
                unique_blocks_by_name[block_name] = block
                block_id_to_canonical[block.block_id] = block.block_id
            else:
                # This is a duplicate block with different ID format
                # Map this ID to the canonical ID we already have
                canonical_block = unique_blocks_by_name[block_name]
                block_id_to_canonical[block.block_id] = canonical_block.block_id

        # Build block -> questions mapping using canonical IDs only
        for block_name, block in unique_blocks_by_name.items():
            block_questions[block.block_id] = []
            for q in block.questions:
                block_questions[block.block_id].append(q.question_id)

        # Track blocks in randomizers (condition-specific) vs shared
        randomizer_block_ids: Set[str] = set()
        shared_block_ids: Set[str] = set()

        # Analyze flow to find randomizer structure
        if flow_data:
            flow = self._extract_flow_payload(flow_data)
            self._analyze_flow_for_visibility(
                flow, conditions, condition_blocks, randomizer_block_ids,
                {b.block_id: b.block_name for b in unique_blocks_by_name.values()},
                block_id_to_canonical
            )

        # All unique blocks not in randomizers are shared
        all_block_ids = set(b.block_id for b in unique_blocks_by_name.values())
        shared_block_ids = all_block_ids - randomizer_block_ids

        self._log(
            LogLevel.INFO, "VISIBILITY",
            f"Block deduplication: {len(blocks)} total -> {len(unique_blocks_by_name)} unique, "
            f"{len(randomizer_block_ids)} in randomizers, {len(shared_block_ids)} shared"
        )

        # Get shared questions (from non-randomizer blocks)
        shared_questions = []
        for block_id in shared_block_ids:
            shared_questions.extend(block_questions.get(block_id, []))

        # Build visibility map for each condition
        for condition in conditions:
            # Start with shared questions visible to all
            for q_id in shared_questions:
                condition_visibility_map[condition][q_id] = True

            # Add questions from condition-specific blocks
            for block_id in condition_blocks.get(condition, []):
                for q_id in block_questions.get(block_id, []):
                    condition_visibility_map[condition][q_id] = True

            # Questions in OTHER conditions' blocks are NOT visible
            for other_cond, other_blocks in condition_blocks.items():
                if other_cond != condition:
                    for block_id in other_blocks:
                        for q_id in block_questions.get(block_id, []):
                            if q_id not in condition_visibility_map[condition]:
                                condition_visibility_map[condition][q_id] = False

        # Apply question-level DisplayLogic
        for q_id, q_info in questions_map.items():
            # Get display_logic from raw_payload if available
            display_logic = q_info.raw_payload.get('DisplayLogic', {}) if q_info.raw_payload else {}
            if display_logic:
                self._apply_question_display_logic(
                    q_id, display_logic, conditions, condition_visibility_map
                )

        self._log(
            LogLevel.INFO, "VISIBILITY",
            f"Built visibility map: {len(conditions)} conditions, "
            f"{len(shared_questions)} shared questions, "
            f"{len(randomizer_block_ids)} condition-specific blocks"
        )

        return {
            'condition_visibility_map': condition_visibility_map,
            'condition_blocks': condition_blocks,
            'block_questions': block_questions,
            'shared_questions': shared_questions
        }

    def _analyze_flow_for_visibility(
        self,
        flow: List[Dict],
        conditions: List[str],
        condition_blocks: Dict[str, List[str]],
        randomizer_block_ids: Set[str],
        blocks_by_id: Dict[str, str],
        block_id_to_canonical: Dict[str, str],
        parent_condition: str = None
    ):
        """Recursively analyze flow to map conditions to their blocks.

        Args:
            flow: Survey flow items
            conditions: List of condition names
            condition_blocks: Dict to populate with condition -> block mappings
            randomizer_block_ids: Set to populate with blocks inside randomizers
            blocks_by_id: Mapping of block_id -> block_name
            block_id_to_canonical: Mapping of any block_id format -> canonical ID
            parent_condition: If we're inside a condition-specific branch
        """
        for item in flow:
            if not isinstance(item, dict):
                continue

            flow_type = item.get('Type', '')

            if flow_type == 'BlockRandomizer':
                # This is the key structure for between-subjects designs
                # Each child (block or group) in the randomizer represents a condition
                # v1.0.0: SubSet can be int, str, or list - get Flow if it's a count value
                subset = item.get('SubSet', item.get('Flow', []))
                if not isinstance(subset, list):
                    # SubSet is the count (int or str like "1"), get the actual Flow
                    subset = item.get('Flow', [])

                for sub_item in subset if isinstance(subset, list) else []:
                    if not isinstance(sub_item, dict):
                        continue

                    sub_type = sub_item.get('Type', '')

                    if sub_type == 'Group':
                        # Group represents a condition
                        group_name = sub_item.get('Description', '')
                        # Match group name to conditions
                        matched_condition = self._match_to_condition(group_name, conditions)
                        if matched_condition:
                            # Get all blocks inside this group
                            group_flow = sub_item.get('Flow', [])
                            for group_item in self._normalize_flow(group_flow):
                                if group_item.get('Type') in ('Standard', 'Block'):
                                    raw_block_id = group_item.get('ID', '')
                                    if raw_block_id:
                                        # Normalize to canonical ID
                                        block_id = block_id_to_canonical.get(str(raw_block_id), str(raw_block_id))
                                        condition_blocks[matched_condition].append(block_id)
                                        randomizer_block_ids.add(block_id)

                    elif sub_type in ('Standard', 'Block'):
                        # Direct block in randomizer - block name IS the condition
                        raw_block_id = sub_item.get('ID', '')
                        if raw_block_id:
                            # Normalize to canonical ID
                            block_id = block_id_to_canonical.get(str(raw_block_id), str(raw_block_id))
                            block_name = blocks_by_id.get(block_id, block_id)
                            matched_condition = self._match_to_condition(block_name, conditions)
                            if matched_condition:
                                condition_blocks[matched_condition].append(block_id)
                                randomizer_block_ids.add(block_id)

            elif flow_type == 'Randomizer':
                # Similar to BlockRandomizer
                sub_flow = item.get('Flow', [])
                for sub_item in self._normalize_flow(sub_flow):
                    if sub_item.get('Type') in ('Standard', 'Block'):
                        raw_block_id = sub_item.get('ID', '')
                        if raw_block_id:
                            block_id = block_id_to_canonical.get(str(raw_block_id), str(raw_block_id))
                            block_name = blocks_by_id.get(block_id, block_id)
                            matched_condition = self._match_to_condition(block_name, conditions)
                            if matched_condition:
                                condition_blocks[matched_condition].append(block_id)
                                randomizer_block_ids.add(block_id)
                    elif sub_item.get('Type') == 'Group':
                        group_name = sub_item.get('Description', '')
                        matched_condition = self._match_to_condition(group_name, conditions)
                        if matched_condition:
                            group_flow = sub_item.get('Flow', [])
                            for group_item in self._normalize_flow(group_flow):
                                if group_item.get('Type') in ('Standard', 'Block'):
                                    raw_block_id = group_item.get('ID', '')
                                    if raw_block_id:
                                        block_id = block_id_to_canonical.get(str(raw_block_id), str(raw_block_id))
                                        condition_blocks[matched_condition].append(block_id)
                                        randomizer_block_ids.add(block_id)

            # Recurse into nested flows
            if 'Flow' in item and flow_type not in ('BlockRandomizer', 'Randomizer'):
                self._analyze_flow_for_visibility(
                    self._normalize_flow(item['Flow']),
                    conditions, condition_blocks, randomizer_block_ids, blocks_by_id,
                    block_id_to_canonical, parent_condition
                )

    def _match_to_condition(self, label: str, conditions: List[str]) -> Optional[str]:
        """Match a block/group name to a condition.

        Returns the condition that best matches the label, or None if no match.
        v1.0.0: Normalizes all whitespace (including non-breaking spaces) for matching.
        """
        if not label:
            return None

        # Normalize whitespace (including \xa0 non-breaking spaces) to regular spaces
        def normalize_ws(s: str) -> str:
            return re.sub(r'\s+', ' ', s.replace('\xa0', ' ')).lower().strip()

        label_norm = normalize_ws(label)

        # Direct match (exact after normalization)
        for cond in conditions:
            if normalize_ws(cond) == label_norm:
                return cond

        # Partial match - label contains condition or vice versa
        for cond in conditions:
            cond_norm = normalize_ws(cond)
            if cond_norm in label_norm or label_norm in cond_norm:
                return cond

        # Match by extracting key words (ignoring word boundaries for non-ASCII)
        label_parts = set(re.findall(r'\b\w+\b', label_norm))
        best_match = None
        best_overlap = 0
        for cond in conditions:
            cond_parts = set(re.findall(r'\b\w+\b', normalize_ws(cond)))
            overlap = label_parts & cond_parts
            # Prefer matches with more overlap, require at least 2 words or exact single word
            if len(overlap) > best_overlap and (len(overlap) >= 2 or (len(overlap) == 1 and len(cond_parts) == 1)):
                best_match = cond
                best_overlap = len(overlap)

        return best_match

    def _apply_question_display_logic(
        self,
        question_id: str,
        display_logic: Dict[str, Any],
        conditions: List[str],
        visibility_map: Dict[str, Dict[str, bool]]
    ):
        """Apply question-level DisplayLogic to visibility map.

        Handles patterns like:
        - EmbeddedField checks (e.g., Condition = "AI")
        - Question response checks (e.g., Q1 = "Yes")
        """
        # Parse the display logic structure
        for key, value in display_logic.items():
            if not isinstance(value, dict):
                continue

            for sub_key, logic_item in value.items():
                if not isinstance(logic_item, dict):
                    continue

                logic_type = logic_item.get('LogicType', '')
                operator = logic_item.get('Operator', '').lower()
                left_operand = str(logic_item.get('LeftOperand', '')).lower()
                right_operand = str(logic_item.get('RightOperand', '')).lower()

                if logic_type == 'EmbeddedField':
                    # Check if this is a condition-based display logic
                    # e.g., Show if Condition = "AI"
                    if 'condition' in left_operand or 'group' in left_operand or 'treatment' in left_operand:
                        # Find which condition this matches
                        for cond in conditions:
                            cond_lower = cond.lower()
                            cond_parts = set(re.findall(r'\b\w+\b', cond_lower))

                            # Check if right_operand matches this condition
                            if right_operand in cond_lower or cond_lower in right_operand:
                                # This condition should see this question
                                if operator in ['equalto', 'selected', 'is', '=']:
                                    visibility_map[cond][question_id] = True
                                    # Other conditions should NOT see it
                                    for other_cond in conditions:
                                        if other_cond != cond and question_id not in visibility_map[other_cond]:
                                            visibility_map[other_cond][question_id] = False
                            elif any(part in right_operand for part in cond_parts if len(part) > 2):
                                if operator in ['equalto', 'selected', 'is', '=']:
                                    visibility_map[cond][question_id] = True
                                    for other_cond in conditions:
                                        if other_cond != cond and question_id not in visibility_map[other_cond]:
                                            visibility_map[other_cond][question_id] = False

    def _dedupe_conditions(self, conditions: List[str]) -> List[str]:
        """Deduplicate conditions and filter out invalid entries.

        v1.0.0: Enhanced to filter out embedded data placeholders and piped text.
        """
        seen = set()
        unique: List[str] = []

        # Patterns that indicate embedded data/piped text, NOT conditions
        invalid_patterns = [
            r'\$\{',  # ${e://Field/...} or ${rand://...}
            r'\$e://',  # $e://Field/...
            r'rand://int',  # Random number placeholders
            r'^\d+$',  # Just a number
            r'^[A-Za-z]_\d+$',  # Single letter with number like Q_1
        ]

        for cond in conditions:
            key = cond.lower().strip()
            if not key or key in seen:
                continue

            # Skip if matches invalid pattern
            is_invalid = False
            for pattern in invalid_patterns:
                if re.search(pattern, cond, re.IGNORECASE):
                    is_invalid = True
                    self._log(
                        LogLevel.INFO, "CONDITION_FILTER",
                        f"Filtered out invalid condition: {cond}"
                    )
                    break

            if not is_invalid:
                seen.add(key)
                unique.append(cond)

        return unique

    def _detect_scales(self, questions_map: Dict) -> List[Dict[str, Any]]:
        """Detect all potential dependent variables (DVs) from question patterns.

        IMPORTANT: Only detects DVs from QSF structure. Does NOT add default scales.
        Default scales should only be added at the app layer if user hasn't specified DVs.

        Enhanced detection covers:
        - Matrix questions (Likert Scale Matrix)
        - Numbered items (e.g., "Ownership_1", "Ownership_2")
        - Single choice Likert-type questions (grouped and standalone)
        - Slider/visual analog scales
        - Constant sum questions
        - Single-item rating questions (standalone DVs)
        - Numeric input questions

        Returns scales with:
        - name: Display name for the scale
        - variable_name: Variable prefix for export
        - question_text: The question text (for user reference)
        - items: Number of items in the scale (1 for single-item DVs)
        - scale_points: Number of response options (MUST be from QSF, not assumed)
        - type: 'matrix', 'numbered_items', 'likert', 'slider', 'single_item', 'constant_sum'
        """
        scales = []
        scale_patterns = {}
        seen_scale_names = set()  # Track to prevent duplicates

        # Safety check - ensure questions_map is a dict
        if not isinstance(questions_map, dict):
            self._log(LogLevel.WARNING, "SCALES", f"questions_map is {type(questions_map).__name__}, expected dict")
            return scales

        # Track single-choice Likert questions for grouping
        likert_questions = {}
        # Track potential single-item DVs (questions with scale responses but no numbering)
        single_item_dvs = []

        for q_id, q_info in questions_map.items():
            # Skip descriptive text, instructions, etc.
            if q_info.question_type in ['Descriptive Text', 'DB ()', 'Timing']:
                continue

            # v1.0.0: Skip placeholder/template questions with generic text
            q_text_lower = (q_info.question_text or "").strip().lower()
            if q_text_lower in (
                "click to write the question text",
                "click to write the statement",
                "click to write choice",
                "default question block",
                "",
            ):
                continue

            # 1. Matrix questions are scales (multi-item)
            if q_info.is_matrix or q_info.question_type == 'Likert Scale Matrix':
                scale_name = q_info.question_text[:50].strip() or q_id
                # v1.2.0: Use export_tag (DataExportTag) for variable name - this is the EXACT QSF name
                variable_name = q_info.export_tag if q_info.export_tag else q_id

                name_key = variable_name.lower()
                if name_key in seen_scale_names:
                    continue
                seen_scale_names.add(name_key)

                num_items = len(q_info.sub_questions) if q_info.sub_questions else 1
                scale_pts = q_info.scale_points

                # v1.2.0: Extract scale_min/scale_max from scale_anchors keys (RecodeValues)
                scale_min, scale_max = self._extract_scale_range(q_info.scale_anchors, scale_pts)

                scales.append({
                    'name': scale_name,
                    'variable_name': variable_name,
                    'question_id': q_id,
                    'question_text': q_info.question_text[:100],
                    'items': num_items,
                    'scale_points': scale_pts,
                    'type': 'matrix',
                    'detected_from_qsf': True,
                    'item_names': q_info.sub_questions or [],  # Actual item text from QSF
                    'scale_anchors': q_info.scale_anchors or {},  # Scale endpoint labels
                    'scale_min': scale_min,
                    'scale_max': scale_max,
                })
                self._log(LogLevel.INFO, "SCALE", f"Detected matrix scale: {variable_name} ({num_items} items, range {scale_min}-{scale_max})")
                continue

            # 2. Slider/Visual Analog scales
            if q_info.question_type in ['Slider', 'Visual Analog', 'VAS', 'Slider ()']:
                scale_name = q_info.question_text[:50].strip() or q_id
                # v1.2.0: Use export_tag for variable name
                variable_name = q_info.export_tag if q_info.export_tag else q_id
                name_key = variable_name.lower()
                if name_key not in seen_scale_names:
                    seen_scale_names.add(name_key)
                    # Get actual slider range if available
                    slider_pts = q_info.scale_points if q_info.scale_points else 101  # 0-100
                    # Get slider min/max from QuestionInfo
                    slider_min = q_info.slider_min if q_info.slider_min is not None else 0
                    slider_max = q_info.slider_max if q_info.slider_max is not None else 100
                    scales.append({
                        'name': scale_name,
                        'variable_name': variable_name,
                        'question_id': q_id,
                        'question_text': q_info.question_text[:100],
                        'items': 1,
                        'scale_points': slider_pts,
                        'type': 'slider',
                        'detected_from_qsf': True,
                        'item_names': [q_info.question_text[:80]] if q_info.question_text else [],
                        'scale_anchors': q_info.slider_labels or q_info.scale_anchors or {},
                        'scale_min': slider_min,
                        'scale_max': slider_max,
                    })
                    self._log(LogLevel.INFO, "SCALE", f"Detected slider DV: {variable_name}")
                continue

            # 3. Constant Sum questions (allocation tasks)
            if 'Constant Sum' in q_info.question_type or 'CS' in q_info.question_type:
                scale_name = q_info.question_text[:50].strip() or q_id
                # v1.2.0: Use export_tag for variable name
                variable_name = q_info.export_tag if q_info.export_tag else q_id
                name_key = variable_name.lower()
                if name_key not in seen_scale_names:
                    seen_scale_names.add(name_key)
                    num_items = len(q_info.choices) if q_info.choices else 1
                    scales.append({
                        'name': scale_name,
                        'variable_name': variable_name,
                        'question_id': q_id,
                        'question_text': q_info.question_text[:100],
                        'items': num_items,
                        'scale_points': 100,  # Constant sum typically 100
                        'type': 'constant_sum',
                        'detected_from_qsf': True,
                        'item_names': q_info.choices or [],  # Allocation categories
                        'scale_anchors': {},
                        'scale_min': 0,
                        'scale_max': 100,
                    })
                    self._log(LogLevel.INFO, "SCALE", f"Detected constant sum DV: {variable_name}")
                continue

            # 3a. Rank Order questions (v2.4.5: NEW)
            if 'Rank Order' in q_info.question_type or 'RO' in q_info.question_type:
                scale_name = q_info.question_text[:50].strip() or q_id
                # v1.2.0: Use export_tag for variable name
                variable_name = q_info.export_tag if q_info.export_tag else q_id
                name_key = variable_name.lower()
                if name_key not in seen_scale_names:
                    seen_scale_names.add(name_key)
                    num_items = len(q_info.choices) if q_info.choices else 1
                    scales.append({
                        'name': scale_name,
                        'variable_name': variable_name,
                        'question_id': q_id,
                        'question_text': q_info.question_text[:100],
                        'items': num_items,
                        'scale_points': num_items,  # Rank order scale = number of items
                        'type': 'rank_order',
                        'detected_from_qsf': True,
                        'item_names': q_info.choices or [],  # Items to rank
                        'scale_anchors': {'1': 'Most preferred', str(num_items): 'Least preferred'},
                        'scale_min': 1,
                        'scale_max': num_items,
                    })
                    self._log(LogLevel.INFO, "SCALE", f"Detected rank order DV: {variable_name}")
                continue

            # 3b. Pick, Group, and Rank (MaxDiff/Best-Worst) questions (v2.4.5: NEW)
            if 'Pick' in q_info.question_type and 'Rank' in q_info.question_type:
                scale_name = q_info.question_text[:50].strip() or q_id
                # v1.2.0: Use export_tag for variable name
                variable_name = q_info.export_tag if q_info.export_tag else q_id
                name_key = variable_name.lower()
                if name_key not in seen_scale_names:
                    seen_scale_names.add(name_key)
                    num_items = len(q_info.choices) if q_info.choices else 1
                    scales.append({
                        'name': scale_name,
                        'variable_name': variable_name,
                        'question_id': q_id,
                        'question_text': q_info.question_text[:100],
                        'items': num_items,
                        'scale_points': num_items,
                        'type': 'best_worst',
                        'detected_from_qsf': True,
                        'item_names': q_info.choices or [],
                        'scale_anchors': {'1': 'Best', str(num_items): 'Worst'},
                        'scale_min': 1,
                        'scale_max': num_items,
                    })
                    self._log(LogLevel.INFO, "SCALE", f"Detected best-worst (MaxDiff) DV: {variable_name}")
                continue

            # 3c. Paired Comparison / Side by Side questions (v2.4.5: NEW)
            if 'Side by Side' in q_info.question_type or 'SBS' in q_info.question_type:
                scale_name = q_info.question_text[:50].strip() or q_id
                # v1.2.0: Use export_tag for variable name
                variable_name = q_info.export_tag if q_info.export_tag else q_id
                name_key = variable_name.lower()
                if name_key not in seen_scale_names:
                    seen_scale_names.add(name_key)
                    num_items = len(q_info.sub_questions) if q_info.sub_questions else 2
                    scales.append({
                        'name': scale_name,
                        'variable_name': variable_name,
                        'question_id': q_id,
                        'question_text': q_info.question_text[:100],
                        'items': num_items,
                        'scale_points': 2,  # Binary comparison
                        'type': 'paired_comparison',
                        'detected_from_qsf': True,
                        'item_names': q_info.sub_questions or [],
                        'scale_anchors': {'1': 'Option A', '2': 'Option B'},
                        'scale_min': 1,
                        'scale_max': 2,
                    })
                    self._log(LogLevel.INFO, "SCALE", f"Detected paired comparison DV: {variable_name}")
                continue

            # 3d. Hot Spot / Heat Map questions (v2.4.5: NEW)
            if 'Hot Spot' in q_info.question_type or 'Heat Map' in q_info.question_type:
                scale_name = q_info.question_text[:50].strip() or q_id
                # v1.2.0: Use export_tag for variable name
                variable_name = q_info.export_tag if q_info.export_tag else q_id
                name_key = variable_name.lower()
                if name_key not in seen_scale_names:
                    seen_scale_names.add(name_key)
                    scales.append({
                        'name': scale_name,
                        'variable_name': variable_name,
                        'question_id': q_id,
                        'question_text': q_info.question_text[:100],
                        'items': 1,
                        'scale_points': None,  # Coordinates, not fixed scale
                        'type': 'hot_spot',
                        'detected_from_qsf': True,
                        'item_names': [q_info.question_text[:80]] if q_info.question_text else [],
                        'scale_anchors': {},
                        'scale_min': None,
                        'scale_max': None,
                    })
                    self._log(LogLevel.INFO, "SCALE", f"Detected hot spot DV: {variable_name}")
                continue

            # 4. Check for numbered pattern (e.g., WTP_1, WTP_2)
            # v1.2.0: Also check export_tag for better variable naming
            var_name = q_info.export_tag if q_info.export_tag else q_id
            match = re.match(r'^(.+?)[-_]?(\d+)$', var_name)
            if match:
                base_name = match.group(1).rstrip('_-')
                item_num = int(match.group(2))
                if base_name not in scale_patterns:
                    scale_patterns[base_name] = []
                scale_patterns[base_name].append({
                    'item_num': item_num,
                    'scale_points': q_info.scale_points,
                    'question_id': q_id,
                    'export_tag': var_name,  # v1.2.0: Store export_tag
                    'question_text': q_info.question_text,
                    'scale_anchors': q_info.scale_anchors or {}  # v1.2.0: Store scale anchors
                })
                continue

            # 5. Single choice Likert-type questions (MC with scale responses)
            if q_info.question_type in ['Single Choice (Radio)', 'Single Choice', 'MC', 'Multiple Choice']:
                choices = q_info.choices if q_info.choices else []
                if 2 <= len(choices) <= 11:  # Typical Likert range
                    # Check if this looks like a scale question
                    if self._looks_like_scale_choices(choices):
                        # v1.2.0: Use export_tag for grouping
                        var_name = q_info.export_tag if q_info.export_tag else q_id
                        q_prefix = re.sub(r'[-_]?\d+$', '', var_name)
                        if q_prefix and q_prefix != var_name:
                            # Has numbering pattern - group it
                            if q_prefix not in likert_questions:
                                likert_questions[q_prefix] = []
                            likert_questions[q_prefix].append({
                                'question_id': q_id,
                                'export_tag': var_name,  # v1.2.0: Store export_tag
                                'scale_points': len(choices),
                                'question_text': q_info.question_text,
                                'scale_anchors': q_info.scale_anchors or {}  # v1.2.0: Store scale anchors
                            })
                        else:
                            # Standalone single-item DV
                            single_item_dvs.append({
                                'question_id': q_id,
                                'export_tag': var_name,  # v1.2.0: Store export_tag
                                'question_text': q_info.question_text,
                                'scale_points': len(choices),
                                'choices': choices,
                                'scale_anchors': q_info.scale_anchors or {}  # v1.2.0: Store scale anchors
                            })
                continue

            # 6. Numeric input questions (could be DVs like "willingness to pay")
            if 'Text Entry' in q_info.question_type:
                q_text_lower = q_info.question_text.lower()
                # Check if it's likely a numeric DV
                numeric_keywords = ['how much', 'how many', 'amount', 'price', 'cost',
                                    'willing to pay', 'wtp', 'bid', 'offer', 'payment',
                                    'rating', 'score', 'number', 'percentage', '%']
                if any(kw in q_text_lower for kw in numeric_keywords):
                    # v1.2.0: Use export_tag for variable name
                    variable_name = q_info.export_tag if q_info.export_tag else q_id
                    name_key = variable_name.lower()
                    if name_key not in seen_scale_names:
                        seen_scale_names.add(name_key)
                        scale_name = q_info.question_text[:50].strip() or q_id
                        # Get numeric range from validation if available
                        num_min = q_info.number_min if q_info.number_min is not None else 0
                        num_max = q_info.number_max if q_info.number_max is not None else 100
                        scales.append({
                            'name': scale_name,
                            'variable_name': variable_name,
                            'question_id': q_id,
                            'question_text': q_info.question_text[:100],
                            'items': 1,
                            'scale_points': None,  # Numeric input has no fixed points
                            'type': 'numeric_input',
                            'detected_from_qsf': True,
                            'item_names': [q_info.question_text[:80]] if q_info.question_text else [],
                            'scale_anchors': {},
                            'scale_min': num_min,
                            'scale_max': num_max,
                        })
                        self._log(LogLevel.INFO, "SCALE", f"Detected numeric input DV: {variable_name}")

        # Consolidate numbered scales (multi-item scales with _1, _2, etc.)
        for base_name, items in scale_patterns.items():
            if len(items) >= 2:  # At least 2 items = multi-item scale
                name_key = base_name.lower()
                if name_key in seen_scale_names:
                    continue
                seen_scale_names.add(name_key)

                valid_points = [i['scale_points'] for i in items if i['scale_points'] is not None]
                scale_pts = max(set(valid_points), key=valid_points.count) if valid_points else None

                # Get first question text as reference
                first_text = items[0].get('question_text', '')[:100] if items else ''

                # v1.2.0: Get scale_anchors from first item that has them
                combined_anchors = {}
                for item in items:
                    if item.get('scale_anchors'):
                        combined_anchors = item['scale_anchors']
                        break

                # v1.2.0: Extract scale_min/scale_max from scale_anchors
                scale_min, scale_max = self._extract_scale_range(combined_anchors, scale_pts)

                # Collect all item texts for item_names
                item_names_list = [i.get('question_text', f'{base_name}_{i["item_num"]}')[:80] for i in sorted(items, key=lambda x: x['item_num'])]
                scales.append({
                    'name': base_name,
                    'variable_name': base_name,
                    'question_text': first_text,
                    'items': len(items),
                    'scale_points': scale_pts,
                    'type': 'numbered_items',
                    'detected_from_qsf': True,
                    'item_names': item_names_list,
                    'scale_anchors': combined_anchors,
                    'scale_min': scale_min,
                    'scale_max': scale_max,
                })
                self._log(LogLevel.INFO, "SCALE", f"Detected numbered scale: {base_name} ({len(items)} items, range {scale_min}-{scale_max})")
            elif len(items) == 1:
                # Single numbered item - might still be a DV
                item = items[0]
                # v1.2.0: Use export_tag for name_key if available
                name_key = item.get('export_tag', item['question_id']).lower()
                if name_key not in seen_scale_names:
                    single_item_dvs.append({
                        'question_id': item['question_id'],
                        'export_tag': item.get('export_tag', item['question_id']),  # v1.2.0
                        'question_text': item.get('question_text', ''),
                        'scale_points': item.get('scale_points'),
                        'choices': [],
                        'scale_anchors': item.get('scale_anchors', {})  # v1.2.0
                    })

        # Consolidate grouped Likert questions
        for prefix, items in likert_questions.items():
            if len(items) >= 2:
                name_key = prefix.lower()
                if name_key in seen_scale_names:
                    continue
                seen_scale_names.add(name_key)

                valid_points = [i['scale_points'] for i in items if i['scale_points'] is not None]
                scale_pts = max(set(valid_points), key=valid_points.count) if valid_points else None
                first_text = items[0].get('question_text', '')[:100] if items else ''

                # v1.2.0: Get scale_anchors from first item that has them
                combined_anchors = {}
                for item in items:
                    if item.get('scale_anchors'):
                        combined_anchors = item['scale_anchors']
                        break

                # v1.2.0: Extract scale_min/scale_max from scale_anchors
                scale_min, scale_max = self._extract_scale_range(combined_anchors, scale_pts)

                # Collect all item texts for item_names
                item_names_list = [i.get('question_text', prefix)[:80] for i in items]
                scales.append({
                    'name': prefix,
                    'variable_name': prefix,
                    'question_text': first_text,
                    'items': len(items),
                    'scale_points': scale_pts,
                    'type': 'likert',
                    'detected_from_qsf': True,
                    'item_names': item_names_list,
                    'scale_anchors': combined_anchors,
                    'scale_min': scale_min,
                    'scale_max': scale_max,
                })
                self._log(LogLevel.INFO, "SCALE", f"Detected Likert scale: {prefix} ({len(items)} items, range {scale_min}-{scale_max})")
            elif len(items) == 1:
                # Single item in a group - add as single-item DV
                item = items[0]
                single_item_dvs.append({
                    'question_id': item['question_id'],
                    'export_tag': item.get('export_tag', item['question_id']),  # v1.2.0
                    'question_text': item.get('question_text', ''),
                    'scale_points': item.get('scale_points'),
                    'choices': [],
                    'scale_anchors': item.get('scale_anchors', {})  # v1.2.0
                })

        # Add single-item DVs (standalone scale questions)
        for dv in single_item_dvs:
            q_id = dv['question_id']
            # v1.2.0: Use export_tag for variable name and deduplication
            variable_name = dv.get('export_tag', q_id)
            name_key = variable_name.lower()
            if name_key in seen_scale_names:
                continue
            seen_scale_names.add(name_key)

            scale_name = dv['question_text'][:50].strip() or q_id
            scale_pts = dv['scale_points']
            # v1.2.0: Get scale_anchors and extract range
            anchors = dv.get('scale_anchors', {})
            scale_min, scale_max = self._extract_scale_range(anchors, scale_pts)

            scales.append({
                'name': scale_name,
                'variable_name': variable_name,
                'question_id': q_id,
                'question_text': dv['question_text'][:100] if dv['question_text'] else '',
                'items': 1,
                'scale_points': scale_pts,
                'type': 'single_item',
                'detected_from_qsf': True,
                'item_names': [dv['question_text'][:80]] if dv['question_text'] else [],
                'scale_anchors': anchors,
                'scale_min': scale_min,
                'scale_max': scale_max,
            })
            self._log(LogLevel.INFO, "SCALE", f"Detected single-item DV: {variable_name} (range {scale_min}-{scale_max})")

        return scales

    def _detect_open_ended(self, questions_map: Dict) -> List[str]:
        """Detect ALL open-ended text entry questions including MC with text entry.

        v2.4.2: IMPROVED detection to capture:
        1. Standalone TE (Text Entry) questions with all selectors (SL, ML, ESTB)
        2. MC questions with TextEntry choices (e.g., "Other: ____")
        3. Matrix questions with TextEntry columns
        4. FORM fields with multiple text entry inputs
        5. Skip comprehension checks (they have specific expected answers)
        """
        open_ended = []
        # Safety check - ensure questions_map is a dict
        if not isinstance(questions_map, dict):
            return open_ended

        for q_id, q_info in questions_map.items():
            # Skip comprehension checks - they require specific answers
            if q_info.is_comprehension_check:
                self._log(
                    LogLevel.INFO, "OPEN_ENDED_SKIP",
                    f"Skipping comprehension check: {q_id}"
                )
                continue

            # Type 1: Standalone Text Entry questions
            if q_info.question_type == 'Text Entry':
                # Use export_tag if available, otherwise q_id
                var_name = q_info.export_tag if q_info.export_tag else q_id
                if var_name not in open_ended:
                    open_ended.append(var_name)
                    self._log(
                        LogLevel.INFO, "OPEN_ENDED",
                        f"Detected text entry question: {var_name} (selector={q_info.selector})"
                    )

            # Type 2: MC questions with TextEntry choices
            if q_info.text_entry_choices:
                for te_choice in q_info.text_entry_choices:
                    var_name = te_choice.get('variable_name', f"{q_id}_TEXT")
                    if var_name not in open_ended:
                        open_ended.append(var_name)
                        self._log(
                            LogLevel.INFO, "OPEN_ENDED",
                            f"Detected MC text entry choice: {var_name}"
                        )

            # Type 3: FORM fields (multiple text inputs in one question)
            if q_info.form_fields:
                for form_field in q_info.form_fields:
                    var_name = form_field.get('variable_name', f"{q_id}_{form_field.get('field_id', '')}")
                    if var_name not in open_ended:
                        open_ended.append(var_name)
                        self._log(
                            LogLevel.INFO, "OPEN_ENDED",
                            f"Detected FORM field: {var_name} (label: {form_field.get('label', '')[:30]})"
                        )

            # Type 4: Check raw payload for additional text entry patterns
            if q_info.raw_payload:
                payload = q_info.raw_payload

                # Check for Essay Box selector (common for text entry)
                selector = payload.get('Selector', '')
                if selector in ['ESTB', 'ML', 'SL']:  # Essay, Multi-line, Single-line
                    var_name = q_info.export_tag if q_info.export_tag else q_id
                    if var_name not in open_ended:
                        open_ended.append(var_name)
                        self._log(
                            LogLevel.INFO, "OPEN_ENDED",
                            f"Detected text entry by selector ({selector}): {var_name}"
                        )

                # Check Answers for TextEntry (Matrix with text columns)
                answers_data = payload.get('Answers', {})
                if isinstance(answers_data, dict):
                    for ans_id, ans_data in answers_data.items():
                        if isinstance(ans_data, dict) and ans_data.get('TextEntry'):
                            var_name = f"{q_info.export_tag or q_id}_{ans_id}_TEXT"
                            if var_name not in open_ended:
                                open_ended.append(var_name)
                                self._log(
                                    LogLevel.INFO, "OPEN_ENDED",
                                    f"Detected Matrix text entry: {var_name}"
                                )

        self._log(
            LogLevel.INFO, "OPEN_ENDED_TOTAL",
            f"Total open-ended fields detected: {len(open_ended)}"
        )
        return open_ended

    def _detect_attention_checks(self, questions_map: Dict) -> List[str]:
        """Detect attention check questions with comprehensive pattern matching.

        Enhanced detection covers:
        - Instructed response items (IRI)
        - Instructional manipulation checks (IMC)
        - Bogus items / trap questions
        - Directed query items
        - Red herring items
        - Consistency checks (reverse-coded)
        """
        attention_checks = []

        # Expanded keyword patterns for attention checks
        attention_keywords = [
            # Direct instructions
            'attention', 'check', 'please select', 'instructed', 'carefully read',
            'quality', 'verify', 'bot', 'reading carefully',
            # Instructed response items
            'select the option', 'choose the answer', 'click on', 'mark the',
            'select strongly', 'select agree', 'select disagree', 'select neutral',
            # IMC patterns
            'this is an attention', 'demonstrate that you', 'show that you',
            'prove that you are', 'confirm you are reading', 'ensure you are paying',
            # Trap/bogus items
            'trap', 'bogus', 'never happened', 'impossible', 'do not answer',
            'skip this question', 'leave blank', 'do not select',
            # Specific instructions
            'for this question', 'ignore the question above', 'answer option',
            'to show you', 'to demonstrate', 'to prove',
            # Common attention check phrases
            'i have visited', 'i have traveled to', 'i read instructions',
            'i am paying attention', 'i understand the instructions',
            'walkaround test', 'screener',
        ]

        # Question ID patterns that suggest attention checks
        attention_id_patterns = [
            r'att[_-]?check', r'attention', r'ac[_\d]', r'imc', r'iri',
            r'trap', r'bogus', r'quality', r'screen', r'valid',
        ]

        # Safety check - ensure questions_map is a dict
        if not isinstance(questions_map, dict):
            return attention_checks

        for q_id, q_info in questions_map.items():
            text = q_info.question_text.lower() if q_info.question_text else ''
            q_id_lower = q_id.lower()

            # Check text content
            is_attention = any(kw in text for kw in attention_keywords)

            # Check question ID patterns
            if not is_attention:
                is_attention = any(re.search(p, q_id_lower) for p in attention_id_patterns)

            # Check for specific instructed responses in choices
            if not is_attention and q_info.choices:
                # Convert all choices to strings before calling lower() (choices can be int/dict/etc)
                choices_text = ' '.join(str(c).lower() for c in q_info.choices if c)
                if 'please select this' in choices_text or 'choose this option' in choices_text:
                    is_attention = True

            if is_attention:
                attention_checks.append(q_id)
                self._log(
                    LogLevel.INFO, "ATTENTION_CHECK",
                    f"Detected attention check: {q_id}"
                )

        return attention_checks

    def _detect_manipulation_checks(self, questions_map: Dict) -> List[str]:
        """Detect manipulation check questions.

        Manipulation checks verify participants understood/noticed the manipulation.
        Different from attention checks (which verify general attention).
        """
        manipulation_checks = []

        # Patterns that suggest manipulation checks
        manip_keywords = [
            'manipulation check', 'manip check', 'understanding check',
            'did you notice', 'do you recall', 'what was the',
            'in the scenario', 'in the study', 'in the experiment',
            'what did the', 'who was the', 'which condition',
            'the [product/person/article]', 'comprehension',
            'perceived as', 'seemed like', 'appeared to be',
            'treatment you received', 'condition you were in',
            'recall what', 'remember what',
        ]

        # Question ID patterns for manipulation checks
        manip_id_patterns = [
            r'mc[_\d]', r'manip', r'manipulation', r'comprehension',
            r'recall', r'check', r'understand',
        ]

        if not isinstance(questions_map, dict):
            return manipulation_checks

        for q_id, q_info in questions_map.items():
            text = q_info.question_text.lower() if q_info.question_text else ''
            q_id_lower = q_id.lower()

            # Check text content
            is_manip = any(kw in text for kw in manip_keywords)

            # Check question ID patterns
            if not is_manip:
                is_manip = any(re.search(p, q_id_lower) for p in manip_id_patterns)

            # Avoid false positives - don't flag if it's an attention check
            attention_indicators = ['attention', 'please select', 'instructed']
            if is_manip and any(ind in text for ind in attention_indicators):
                continue

            if is_manip:
                manipulation_checks.append(q_id)
                self._log(
                    LogLevel.INFO, "MANIPULATION_CHECK",
                    f"Detected manipulation check: {q_id}"
                )

        return manipulation_checks

    def _extract_open_ended_details(
        self,
        questions_map: Dict,
        blocks: List[BlockInfo]
    ) -> List[Dict[str, Any]]:
        """Extract detailed information about ALL open-ended text entry questions.

        v2.4.1: ENHANCED to capture all text entry types:
        1. Standalone Text Entry questions
        2. MC questions with TextEntry choices
        3. Matrix questions with TextEntry columns
        4. Form fields

        Returns list of dicts with:
        - question_id: Question identifier
        - variable_name: Export variable name (for data column)
        - name: Alias for variable_name (backwards compatibility)
        - question_text: Full question text
        - block_name: Which block it's in
        - context_type: What kind of response expected
        - preceding_questions: Context from questions before this one
        - force_response: Whether response is required
        - source_type: 'text_entry' | 'mc_text_choice' | 'matrix_text' | 'form'
        """
        open_ended_details = []
        seen_variables = set()  # Prevent duplicates

        # Safety check
        if not isinstance(questions_map, dict):
            return open_ended_details

        # Patterns for ID/demographic fields to handle differently
        id_patterns = [
            'mturk', 'worker id', 'prolific', 'participant id',
            'email address', 'phone number', 'zip code', 'zipcode',
            'age', 'year born', 'birth year'
        ]

        for q_id, q_info in questions_map.items():
            text_lower = q_info.question_text.lower() if q_info.question_text else ''

            # TYPE 1: Standalone Text Entry questions
            if q_info.question_type == 'Text Entry':
                var_name = q_info.export_tag if q_info.export_tag else q_id
                var_name = var_name.replace(' ', '_')

                if var_name not in seen_variables:
                    seen_variables.add(var_name)

                    # Determine if it's an ID field (still include but mark differently)
                    is_id_field = any(pat in text_lower for pat in id_patterns)

                    # Determine context type based on question text
                    if is_id_field:
                        context_type = 'identifier'
                    else:
                        context_type = self._classify_open_ended_type(q_info.question_text)

                    # Get surrounding questions for context
                    preceding_questions = self._get_preceding_context(q_id, q_info.block_name, blocks)

                    detail = {
                        'question_id': q_id,
                        'variable_name': var_name,
                        'name': var_name,  # Backwards compatibility
                        'question_text': q_info.question_text,
                        'block_name': q_info.block_name,
                        'context_type': context_type,
                        'preceding_questions': preceding_questions,
                        'force_response': q_info.force_response,
                        'source_type': 'text_entry'
                    }

                    open_ended_details.append(detail)

                    self._log(
                        LogLevel.INFO, "OPEN_ENDED_DETAIL",
                        f"Extracted text entry: {var_name} ({context_type}) - {q_info.question_text[:50]}..."
                    )

            # TYPE 2: MC questions with TextEntry choices
            if q_info.text_entry_choices:
                for te_choice in q_info.text_entry_choices:
                    var_name = te_choice.get('variable_name', f"{q_id}_TEXT")
                    var_name = var_name.replace(' ', '_')

                    if var_name not in seen_variables:
                        seen_variables.add(var_name)

                        # Context is the parent question + choice text
                        choice_text = te_choice.get('choice_text', '')
                        combined_text = f"{q_info.question_text} - {choice_text}"

                        detail = {
                            'question_id': q_id,
                            'variable_name': var_name,
                            'name': var_name,
                            'question_text': combined_text,
                            'block_name': q_info.block_name,
                            'context_type': 'elaboration',  # Usually "Other: please specify"
                            'preceding_questions': [],
                            'force_response': False,  # MC text entry usually optional
                            'source_type': 'mc_text_choice',
                            'parent_question': q_info.question_text,
                            'choice_text': choice_text
                        }

                        open_ended_details.append(detail)

                        self._log(
                            LogLevel.INFO, "OPEN_ENDED_DETAIL",
                            f"Extracted MC text choice: {var_name} - {choice_text[:30]}..."
                        )

            # TYPE 3: Check for text entry via selector (Essay, Multi-line, etc.)
            if q_info.raw_payload:
                payload = q_info.raw_payload
                selector = payload.get('Selector', '')

                # Essay/Form selectors that weren't caught as Text Entry type
                if selector in ['ESTB', 'ML', 'SL', 'FORM'] and q_info.question_type != 'Text Entry':
                    var_name = q_info.export_tag if q_info.export_tag else q_id
                    var_name = var_name.replace(' ', '_')

                    if var_name not in seen_variables:
                        seen_variables.add(var_name)

                        context_type = self._classify_open_ended_type(q_info.question_text)
                        preceding_questions = self._get_preceding_context(q_id, q_info.block_name, blocks)

                        detail = {
                            'question_id': q_id,
                            'variable_name': var_name,
                            'name': var_name,
                            'question_text': q_info.question_text,
                            'block_name': q_info.block_name,
                            'context_type': context_type,
                            'preceding_questions': preceding_questions,
                            'force_response': q_info.force_response,
                            'source_type': 'form'
                        }

                        open_ended_details.append(detail)

                # TYPE 4: Matrix questions with TextEntry answers
                answers_data = payload.get('Answers', {})
                if isinstance(answers_data, dict):
                    for ans_id, ans_data in answers_data.items():
                        if isinstance(ans_data, dict) and ans_data.get('TextEntry'):
                            var_name = f"{q_info.export_tag or q_id}_{ans_id}_TEXT"
                            var_name = var_name.replace(' ', '_')

                            if var_name not in seen_variables:
                                seen_variables.add(var_name)

                                ans_text = ans_data.get('Display', '')
                                combined_text = f"{q_info.question_text} - {ans_text}"

                                detail = {
                                    'question_id': q_id,
                                    'variable_name': var_name,
                                    'name': var_name,
                                    'question_text': combined_text,
                                    'block_name': q_info.block_name,
                                    'context_type': 'elaboration',
                                    'preceding_questions': [],
                                    'force_response': q_info.force_response,
                                    'source_type': 'matrix_text',
                                    'parent_question': q_info.question_text
                                }

                                open_ended_details.append(detail)

            # TYPE 5: FORM fields (v2.4.2) - multiple text inputs in one question
            if q_info.form_fields:
                for form_field in q_info.form_fields:
                    var_name = form_field.get('variable_name', f"{q_id}_{form_field.get('field_id', '')}")
                    var_name = var_name.replace(' ', '_')

                    if var_name not in seen_variables:
                        seen_variables.add(var_name)

                        field_label = form_field.get('label', '')
                        combined_text = f"{q_info.question_text} - {field_label}" if field_label else q_info.question_text

                        # Classify based on field label
                        context_type = self._classify_open_ended_type(field_label) if field_label else 'general'

                        detail = {
                            'question_id': q_id,
                            'variable_name': var_name,
                            'name': var_name,
                            'question_text': combined_text,
                            'block_name': q_info.block_name,
                            'context_type': context_type,
                            'preceding_questions': [],
                            'force_response': q_info.force_response,
                            'source_type': 'form_field',
                            'parent_question': q_info.question_text,
                            'field_label': field_label
                        }

                        open_ended_details.append(detail)

                        self._log(
                            LogLevel.INFO, "OPEN_ENDED_DETAIL",
                            f"Extracted FORM field: {var_name} (label: {field_label[:30]})"
                        )

        self._log(
            LogLevel.INFO, "OPEN_ENDED_DETAILS_TOTAL",
            f"Total open-ended details extracted: {len(open_ended_details)}"
        )

        return open_ended_details

    def _classify_open_ended_type(self, question_text: str) -> str:
        """Classify what type of open-ended response is expected."""
        text_lower = question_text.lower()

        # Explanation/reasoning
        if any(kw in text_lower for kw in ['explain', 'why', 'reason', 'because', 'justify']):
            return 'explanation'

        # Feedback/opinion
        if any(kw in text_lower for kw in ['feedback', 'opinion', 'think', 'feel', 'thoughts']):
            return 'feedback'

        # Description
        if any(kw in text_lower for kw in ['describe', 'description', 'tell us', 'share']):
            return 'description'

        # Reflection
        if any(kw in text_lower for kw in ['reflect', 'experience', 'notice']):
            return 'reflection'

        # Suggestion/improvement
        if any(kw in text_lower for kw in ['suggest', 'improve', 'change', 'better']):
            return 'suggestion'

        # General comment
        if any(kw in text_lower for kw in ['comment', 'anything else', 'additional']):
            return 'comment'

        return 'general'

    def _get_preceding_context(
        self,
        question_id: str,
        block_name: str,
        blocks: List[BlockInfo]
    ) -> List[str]:
        """Get the text of questions preceding this one for context."""
        preceding = []

        for block in blocks:
            if block.block_name == block_name:
                found_target = False
                for q in block.questions:
                    if q.question_id == question_id:
                        found_target = True
                        break
                    # Add preceding question text (limited)
                    if q.question_text:
                        preceding.append(q.question_text[:100])

        # Return last 3 preceding questions
        return preceding[-3:] if len(preceding) > 3 else preceding

    def _extract_study_context(
        self,
        survey_name: str,
        questions_map: Dict,
        blocks: List[BlockInfo]
    ) -> Dict[str, Any]:
        """Extract overall study context from the survey content.

        This analyzes all questions, instructions, and blocks to understand
        what the study is about, enabling contextually appropriate text generation.
        """
        context = {
            'survey_name': survey_name,
            'topics': [],
            'key_concepts': [],
            'study_domain': 'general',
            'instructions_text': '',
            'main_questions': [],
        }

        if not isinstance(questions_map, dict):
            return context

        # Collect all text from questions
        all_text = [survey_name]

        for q_id, q_info in questions_map.items():
            if q_info.question_text:
                all_text.append(q_info.question_text)

        combined_text = ' '.join(all_text).lower()

        # Detect study domain
        domain_keywords = {
            'politics': ['trump', 'biden', 'democrat', 'republican', 'political', 'vote', 'election', 'president'],
            'economics': ['money', 'dollars', 'contribute', 'invest', 'pay', 'price', 'cost', 'economic', 'financial'],
            'behavioral_economics': ['dictator game', 'public goods', 'prisoner', 'ultimatum', 'trust game', 'cooperation'],
            'social_psychology': ['social', 'group', 'other people', 'partner', 'interaction'],
            'consumer': ['product', 'brand', 'purchase', 'buy', 'recommend', 'consumer'],
            'technology': ['ai', 'artificial intelligence', 'robot', 'technology', 'computer'],
            'health': ['health', 'medical', 'doctor', 'illness', 'treatment'],
        }

        detected_domains = []
        for domain, keywords in domain_keywords.items():
            match_count = sum(1 for kw in keywords if kw in combined_text)
            if match_count >= 2:
                detected_domains.append((domain, match_count))

        if detected_domains:
            # Sort by match count and take top domain
            detected_domains.sort(key=lambda x: x[1], reverse=True)
            context['study_domain'] = detected_domains[0][0]
            context['topics'] = [d[0] for d in detected_domains[:3]]

        # Extract key concepts (nouns that appear frequently)
        # Find capitalized words that aren't at sentence start
        import re
        concept_pattern = r'(?<=[a-z]\s)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        concepts = re.findall(concept_pattern, ' '.join(all_text[:10]))
        context['key_concepts'] = list(set(concepts))[:10]

        # Extract instruction text (from descriptive text questions)
        for q_id, q_info in questions_map.items():
            if q_info.question_type == 'Descriptive Text':
                context['instructions_text'] += q_info.question_text + ' '

        # Get main survey questions (not instructions, not demographics)
        main_q_keywords = ['how', 'what', 'please rate', 'indicate', 'to what extent']
        for q_id, q_info in questions_map.items():
            text_lower = q_info.question_text.lower()
            if any(kw in text_lower for kw in main_q_keywords):
                if len(context['main_questions']) < 5:
                    context['main_questions'].append(q_info.question_text[:100])

        self._log(
            LogLevel.INFO, "STUDY_CONTEXT",
            f"Detected study domain: {context['study_domain']}, topics: {context['topics']}"
        )

        return context

    def _extract_forced_response_questions(self, questions_map: Dict) -> List[Dict[str, Any]]:
        """Extract all questions with ForceResponse validation.

        v2.4.3: ENHANCED - These questions MUST be filled in simulation output.
        Now includes all validation details for proper response generation.
        """
        forced_questions = []

        if not isinstance(questions_map, dict):
            return forced_questions

        for q_id, q_info in questions_map.items():
            if q_info.force_response:
                forced_questions.append({
                    'question_id': q_id,
                    'export_tag': q_info.export_tag or q_id,
                    'question_type': q_info.question_type,
                    'question_text': q_info.question_text[:100],
                    'block_name': q_info.block_name,
                    'has_scale': q_info.scale_points is not None,
                    'scale_points': q_info.scale_points,
                    'selector': q_info.selector,
                    'is_text_entry': q_info.question_type in ['Text Entry', 'Text Entry (Essay)', 'Text Entry (Multi-line)', 'Text Entry (Form)'] or q_info.selector in ['ESTB', 'ML', 'SL', 'FORM'],
                    'form_fields': q_info.form_fields if q_info.form_fields else [],
                    # v2.4.3: Enhanced validation details
                    'min_chars': q_info.min_chars,
                    'max_chars': q_info.max_chars,
                    'content_type': q_info.content_type,
                    'number_min': q_info.number_min,
                    'number_max': q_info.number_max,
                    'validation_regex': q_info.validation_regex,
                    # v2.4.3: Slider details
                    'is_slider': q_info.question_type == 'Slider',
                    'slider_min': q_info.slider_min,
                    'slider_max': q_info.slider_max,
                    'slider_snap_to_grid': q_info.slider_snap_to_grid,
                })

        if forced_questions:
            self._log(
                LogLevel.INFO, "FORCED_RESPONSE",
                f"Found {len(forced_questions)} questions with ForceResponse validation"
            )

        return forced_questions

    def _extract_slider_questions(self, questions_map: Dict) -> List[Dict[str, Any]]:
        """Extract all slider questions with their configuration.

        v2.4.3: NEW - Provides detailed slider config for accurate simulation.
        """
        sliders = []

        if not isinstance(questions_map, dict):
            return sliders

        for q_id, q_info in questions_map.items():
            if q_info.question_type == 'Slider':
                sliders.append({
                    'question_id': q_id,
                    'export_tag': q_info.export_tag or q_id,
                    'question_text': q_info.question_text[:100],
                    'block_name': q_info.block_name,
                    'force_response': q_info.force_response,
                    'slider_min': q_info.slider_min,
                    'slider_max': q_info.slider_max,
                    'slider_grid_lines': q_info.slider_grid_lines,
                    'slider_snap_to_grid': q_info.slider_snap_to_grid,
                    'slider_labels': q_info.slider_labels,
                    'scale_points': q_info.scale_points,
                })

        if sliders:
            self._log(
                LogLevel.INFO, "SLIDERS",
                f"Found {len(sliders)} slider questions"
            )

        return sliders

    def _extract_text_entry_questions(self, questions_map: Dict) -> List[Dict[str, Any]]:
        """Extract all text entry questions with their validation requirements.

        v2.4.3: NEW - Provides detailed text entry config for accurate simulation.
        Includes min/max chars, content type validation, regex patterns, etc.
        """
        text_entries = []

        if not isinstance(questions_map, dict):
            return text_entries

        text_entry_types = ['Text Entry', 'Text Entry (Essay)', 'Text Entry (Multi-line)', 'Text Entry (Form)']

        for q_id, q_info in questions_map.items():
            if q_info.question_type in text_entry_types or q_info.selector in ['ESTB', 'ML', 'SL', 'FORM']:
                text_entries.append({
                    'question_id': q_id,
                    'export_tag': q_info.export_tag or q_id,
                    'question_text': q_info.question_text[:150],
                    'block_name': q_info.block_name,
                    'selector': q_info.selector,
                    'force_response': q_info.force_response,
                    'min_chars': q_info.min_chars,
                    'max_chars': q_info.max_chars,
                    'content_type': q_info.content_type,
                    'number_min': q_info.number_min,
                    'number_max': q_info.number_max,
                    'validation_regex': q_info.validation_regex,
                    'form_fields': q_info.form_fields if q_info.form_fields else [],
                    'is_comprehension_check': q_info.is_comprehension_check,
                })

        if text_entries:
            self._log(
                LogLevel.INFO, "TEXT_ENTRIES",
                f"Found {len(text_entries)} text entry questions"
            )

        return text_entries

    def _extract_comprehension_checks(self, questions_map: Dict) -> List[Dict[str, Any]]:
        """Extract comprehension check questions with their expected answers.

        v2.4.2: These questions require specific correct answers for validation.
        Used to ensure participants understand the task correctly.
        """
        comprehension_checks = []

        if not isinstance(questions_map, dict):
            return comprehension_checks

        for q_id, q_info in questions_map.items():
            if q_info.is_comprehension_check:
                comprehension_checks.append({
                    'question_id': q_id,
                    'export_tag': q_info.export_tag or q_id,
                    'question_type': q_info.question_type,
                    'question_text': q_info.question_text[:150],
                    'block_name': q_info.block_name,
                    'expected_answer': q_info.comprehension_expected,
                    'selector': q_info.selector,
                })

        if comprehension_checks:
            self._log(
                LogLevel.INFO, "COMPREHENSION_CHECKS",
                f"Found {len(comprehension_checks)} comprehension check questions"
            )

        return comprehension_checks

    def _detect_embedded_data_conditions(self, flow_data: Optional[Any]) -> List[Dict[str, Any]]:
        """Detect experimental conditions set via EmbeddedData randomization.

        Some surveys (like Hate_Trumps_Love) use EmbeddedData inside BlockRandomizers
        to set condition values like "who loves Trump", "who hates Trump", etc.
        """
        conditions = []

        if not flow_data:
            return conditions

        flow = self._extract_flow_payload(flow_data)
        self._find_embedded_data_conditions(flow, conditions)

        return conditions

    def _find_embedded_data_conditions(
        self,
        flow: List,
        conditions: List[Dict[str, Any]],
        depth: int = 0
    ):
        """Recursively find EmbeddedData conditions inside randomizers."""
        for item in flow:
            if not isinstance(item, dict):
                continue

            flow_type = item.get('Type', '')

            # Look for EmbeddedData inside BlockRandomizer or Randomizer
            if flow_type in {'BlockRandomizer', 'Randomizer'}:
                sub_flow = self._normalize_flow(item.get('Flow', []))

                for sub_item in sub_flow:
                    if isinstance(sub_item, dict):
                        # Check for EmbeddedData in the randomized branch
                        if sub_item.get('Type') == 'EmbeddedData':
                            embedded_fields = sub_item.get('EmbeddedData', [])
                            if isinstance(embedded_fields, list):
                                for field in embedded_fields:
                                    if isinstance(field, dict):
                                        field_name = field.get('Field', '')
                                        field_value = field.get('Value', '')
                                        if field_value:  # Has a set value
                                            conditions.append({
                                                'field': field_name,
                                                'value': field_value,
                                                'source': 'EmbeddedData in Randomizer',
                                                'depth': depth
                                            })
                                            self._log(
                                                LogLevel.INFO, "EMBEDDED_CONDITION",
                                                f"Found embedded condition: {field_name}={field_value}"
                                            )

            # Recurse into nested flows
            if 'Flow' in item:
                self._find_embedded_data_conditions(
                    self._normalize_flow(item['Flow']),
                    conditions,
                    depth + 1
                )

    def _analyze_randomizers(self, flow_data: Optional[Any]) -> Dict[str, Any]:
        """Analyze randomizer structure in detail."""
        if not flow_data:
            return {'has_randomization': False, 'randomizers': []}

        randomizers = []
        flow = self._extract_flow_payload(flow_data)

        self._find_all_randomizers(flow, randomizers, depth=0)

        return {
            'has_randomization': len(randomizers) > 0,
            'num_randomizers': len(randomizers),
            'randomizers': randomizers,
            'is_factorial': len(randomizers) > 1,
            'randomization_level': self._infer_randomization_level(randomizers),
        }

    def _find_all_randomizers(
        self,
        flow: List,
        randomizers: List[Dict],
        depth: int = 0
    ):
        """Recursively find all randomizer elements."""
        for item in flow:
            if not isinstance(item, dict):
                continue

            flow_type = item.get('Type', '')

            if flow_type in {'Randomizer', 'BlockRandomizer'}:
                rand_info = {
                    'flow_id': item.get('FlowID', ''),
                    'description': item.get('Description', ''),
                    'type': flow_type.lower(),
                    'evenly_present': item.get('EvenPresentation', True),
                    'randomize_count': item.get('RandomizeCount'),
                    'depth': depth,
                    'num_branches': len(item.get('Flow', [])),
                }
                randomizers.append(rand_info)

                self._log(
                    LogLevel.INFO, "RANDOMIZER",
                    f"Found randomizer: {rand_info['description']} ({rand_info['num_branches']} branches)"
                )

            # Recurse into nested flows
            if 'Flow' in item:
                self._find_all_randomizers(self._normalize_flow(item['Flow']), randomizers, depth + 1)

    def _infer_randomization_level(self, randomizers: List[Dict]) -> str:
        """Infer the level of randomization from randomizer structure."""
        if not randomizers:
            return "none"

        # Check for group-level randomization markers
        for rand in randomizers:
            desc = rand.get('description', '').lower()
            if 'group' in desc or 'cluster' in desc:
                return "group"

        # Check for within-subject design
        for rand in randomizers:
            count = rand.get('randomize_count')
            if count and str(count).lower() == 'all':
                return "within_subject"

        # Default to between-subjects participant-level
        return "participant"

    def _parse_flow(self, flow_data: Optional[Any]) -> List[str]:
        """Parse survey flow into readable list."""
        if not flow_data:
            return []

        elements = []
        flow = self._extract_flow_payload(flow_data)

        self._parse_flow_recursive(flow, elements, 0)

        return elements

    def _parse_flow_recursive(self, flow: List, elements: List[str], depth: int):
        """Recursively parse flow structure."""
        indent = "  " * depth
        for item in flow:
            if isinstance(item, dict):
                flow_type = item.get('Type', 'Unknown')
                flow_id = item.get('ID', item.get('FlowID', ''))

                if flow_type == 'Block':
                    elements.append(f"{indent}[Block] {flow_id}")
                elif flow_type in {'Randomizer', 'BlockRandomizer'}:
                    elements.append(f"{indent}[Randomizer] {item.get('Description', '')}")
                elif flow_type == 'Group':
                    elements.append(f"{indent}[Group] {item.get('Description', '')}")
                elif flow_type == 'Branch':
                    elements.append(f"{indent}[Branch] {item.get('Description', '')}")
                elif flow_type == 'EmbeddedData':
                    elements.append(f"{indent}[Embedded Data]")
                elif flow_type == 'EndSurvey':
                    elements.append(f"{indent}[End Survey]")
                elif flow_type == 'Quota':
                    elements.append(f"{indent}[Quota]")
                elif flow_type == 'WebService':
                    elements.append(f"{indent}[Web Service]")
                elif flow_type == 'Authenticator':
                    elements.append(f"{indent}[Authenticator]")

                if 'Flow' in item:
                    self._parse_flow_recursive(self._normalize_flow(item['Flow']), elements, depth + 1)

    def _validate_structure(
        self,
        blocks: List[BlockInfo],
        questions_map: Dict,
        conditions: List[str]
    ):
        """Validate survey structure and log issues."""

        # Safety check - ensure questions_map is a dict
        if not isinstance(questions_map, dict):
            self._log(LogLevel.WARNING, "VALIDATION", f"questions_map is {type(questions_map).__name__}, expected dict")
            questions_map = {}  # Use empty dict to continue validation

        # Safety check - ensure blocks is a list
        if not isinstance(blocks, list):
            blocks = []

        # Check for empty blocks (skip trash/unused blocks - they're not relevant)
        for block in blocks:
            if not hasattr(block, 'questions') or not hasattr(block, 'is_randomizer'):
                continue
            # Skip validation for trash/unused blocks - they're intentionally empty
            if hasattr(block, 'block_name') and self._is_excluded_block_name(block.block_name):
                continue
            if len(block.questions) == 0 and not block.is_randomizer:
                self._log(
                    LogLevel.INFO, "VALIDATION",
                    f"Block '{block.block_name}' has no questions"
                )

        # Check for questions without proper scale points
        # Only warn if we TRULY can't determine scale points (very rare with proper QSF parsing)
        for q_id, q_info in questions_map.items():
            if q_info.question_type in ['Likert Scale Matrix', 'Single Choice (Radio)']:
                # Scale points should almost always be detected from QSF
                # Only warn if it's None (not just low) and question type suggests it's a scale
                if q_info.scale_points is None:
                    self._log(
                        LogLevel.INFO, "SCALE_DETECTION",
                        f"Question {q_id}: scale points not explicitly defined, using choices count",
                        {'question_text': q_info.question_text}
                    )

        # Check for potential attention check questions
        attention_keywords = ['attention', 'check', 'please select', 'instructed']
        found_attention = False
        for q_id, q_info in questions_map.items():
            for keyword in attention_keywords:
                if keyword in q_info.question_text.lower():
                    found_attention = True
                    self._log(
                        LogLevel.INFO, "ATTENTION_CHECK",
                        f"Potential attention check found: {q_id}"
                    )

        if not found_attention:
            self._log(
                LogLevel.WARNING, "VALIDATION",
                "No attention check questions detected. Consider adding attention checks."
            )

        # Check condition count
        if len(conditions) == 0:
            self._log(
                LogLevel.ERROR, "VALIDATION",
                "No experimental conditions detected. Please define conditions manually."
            )
        elif len(conditions) > 8:
            self._log(
                LogLevel.WARNING, "VALIDATION",
                f"Large number of conditions detected ({len(conditions)}). "
                "Verify this is correct."
            )

    def generate_log_report(self) -> str:
        """Generate a formatted log report for instructor review."""
        lines = [
            "=" * 70,
            "QSF PARSING LOG REPORT",
            "=" * 70,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        # Summary
        error_count = len([e for e in self.log_entries if e.level in [LogLevel.ERROR, LogLevel.CRITICAL]])
        warning_count = len([e for e in self.log_entries if e.level == LogLevel.WARNING])
        info_count = len([e for e in self.log_entries if e.level == LogLevel.INFO])

        lines.extend([
            "SUMMARY",
            "-" * 70,
            f"  Errors: {error_count}",
            f"  Warnings: {warning_count}",
            f"  Info: {info_count}",
            "",
        ])

        # Errors (if any)
        if error_count > 0:
            lines.extend([
                "ERRORS (Require Attention)",
                "-" * 70,
            ])
            for entry in self.log_entries:
                if entry.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                    lines.append(f"  [{entry.category}] {entry.message}")
            lines.append("")

        # Warnings
        if warning_count > 0:
            lines.extend([
                "WARNINGS (Review Recommended)",
                "-" * 70,
            ])
            for entry in self.log_entries:
                if entry.level == LogLevel.WARNING:
                    lines.append(f"  [{entry.category}] {entry.message}")
            lines.append("")

        # Full log
        lines.extend([
            "FULL LOG",
            "-" * 70,
        ])
        for entry in self.log_entries:
            level_str = entry.level.value.ljust(8)
            lines.append(f"  [{level_str}] [{entry.category}] {entry.message}")

        lines.extend([
            "",
            "=" * 70,
            "END OF LOG REPORT",
            "=" * 70
        ])

        return "\n".join(lines)


class RandomizationPatternDetector:
    """
    Detects 25+ different randomization patterns in QSF files.

    This class provides comprehensive detection of experimental conditions
    across all known Qualtrics randomization methods including:

    Flow-Based Randomization:
    1. BlockRandomizer (SubSet=1) - Standard between-subjects design
    2. BlockRandomizer (SubSet>1) - Within-subjects/mixed designs
    3. Standard Randomizer - General randomizer element
    4. Group Randomizer - Randomization at group level
    5. Nested/Factorial - Multiple randomizers for factorial designs

    Data-Based Randomization:
    6. EmbeddedData Conditions - Conditions set via EmbeddedData fields
    7. RandomInteger/RandomNumber - Numeric randomization
    8. Piped Text - Dynamic text based on randomized values
    9. SetValue Operations - Direct value assignments

    Logic-Based Randomization:
    10. Branch-Based - Using Branch elements with display logic
    11. Skip Logic - Question skip patterns indicating conditions
    12. Display Logic - Show/hide based on conditions
    13. Carry Forward - Response-based routing

    External Randomization:
    14. WebService - External API-based assignment
    15. Reference Survey - Conditions from other surveys
    16. Authenticator - Based on authentication results
    17. Panel/Contact List - External list-based assignment

    Quota-Based:
    18. Quota Groups - Quota-based assignment
    19. Quota Actions - Post-quota branching
    20. Cross-Quota - Multiple quota interactions

    Survey Structure:
    21. EndSurvey Branches - Different endings based on conditions
    22. Question Randomization - Within-block question order
    23. Answer Randomization - Response option randomization
    24. Loop & Merge - Repeated blocks with variations
    25. Conjoint/MaxDiff - Specialized experimental designs

    Version: 2.2.0 - Expanded to 45+ patterns across 6 categories
    """

    # Pattern names for logging and identification (35+ patterns)
    PATTERN_TYPES = {
        # ========== Flow-Based Patterns (8) ==========
        'block_randomizer_between': 'BlockRandomizer (Between-Subjects)',
        'block_randomizer_within': 'BlockRandomizer (Within-Subjects)',
        'randomizer': 'Standard Randomizer',
        'group_randomizer': 'Group Randomizer',
        'nested': 'Nested/Factorial Design',
        'table_randomizer': 'Table of Contents Randomizer',
        'flow_group': 'Flow Group',
        'standard_block': 'Standard Block Flow',

        # ========== Data-Based Patterns (8) ==========
        'embedded_data': 'EmbeddedData Conditions',
        'random_number': 'Random Number Assignment',
        'piped_text': 'Piped Text Conditions',
        'set_value': 'SetValue Operations',
        'math_operation': 'Math Operations',
        'counter': 'Counter/Scoring',
        'response_import': 'Response Import',
        'url_parameter': 'URL Parameter Capture',

        # ========== Logic-Based Patterns (8) ==========
        'branch': 'Branch-Based Conditions',
        'skip_logic': 'Skip Logic Patterns',
        'display_logic': 'Display Logic Conditions',
        'carry_forward': 'Carry Forward Routing',
        'relevance': 'Relevance/Validation',
        'timing': 'Timing-Based Logic',
        'response_validation': 'Response Validation',
        'content_validation': 'Content Validation',

        # ========== External/Integration Patterns (6) ==========
        'webservice': 'WebService Conditions',
        'reference_survey': 'Reference Survey Conditions',
        'authenticator': 'Authenticator-Based',
        'panel_assignment': 'Panel/Contact List Assignment',
        'sso': 'Single Sign-On',
        'api_trigger': 'API Trigger',

        # ========== Quota-Based Patterns (5) ==========
        'quota': 'Quota-Based Assignment',
        'quota_action': 'Quota Action Branching',
        'cross_quota': 'Cross-Quota Interactions',
        'soft_quota': 'Soft Quota',
        'scheduled_quota': 'Scheduled Quota',

        # ========== Survey Structure Patterns (10) ==========
        'end_survey': 'EndSurvey Branches',
        'question_randomization': 'Question Randomization',
        'answer_randomization': 'Answer Randomization',
        'loop_merge': 'Loop & Merge',
        'conjoint': 'Conjoint/MaxDiff Design',
        'evenly_present': 'Even Presentation',
        'matrix_table': 'Matrix/Grid Questions',
        'slider': 'Slider Questions',
        'heat_map': 'Heat Map/Click Map',
        'rank_order': 'Rank Order Questions',
    }

    # Keywords that indicate condition assignment in field names
    CONDITION_FIELD_KEYWORDS = [
        'condition', 'treatment', 'group', 'arm', 'manipulation',
        'scenario', 'stimulus', 'version', 'cond', 'grp', 'experimental',
        'study_arm', 'exp_cond', 'treat', 'variant', 'cell',
    ]

    def __init__(self, logger_callback=None):
        self.logger = logger_callback
        self.detected_patterns: List[Dict[str, Any]] = []
        self.conditions: List[str] = []
        self.embedded_conditions: List[Dict[str, Any]] = []
        self.question_randomization_detected: bool = False
        self.loop_merge_detected: bool = False

    def _log(self, message: str, details: Dict = None):
        """Log detection information."""
        if self.logger:
            self.logger(message, details)

    def detect_all_patterns(
        self,
        flow_data: Any,
        blocks_by_id: Dict[str, str],
        normalize_flow_func,
        extract_payload_func,
        excluded_func,
    ) -> Dict[str, Any]:
        """
        Detect all randomization patterns in a QSF flow.

        Args:
            flow_data: Raw flow data from QSF
            blocks_by_id: Mapping of block ID to block name
            normalize_flow_func: Function to normalize flow lists
            extract_payload_func: Function to extract flow payload
            excluded_func: Function to check if block name is excluded

        Returns:
            Dict with:
            - conditions: List of detected condition names
            - patterns: List of detected pattern types
            - embedded_conditions: List of embedded data conditions
            - is_factorial: Whether design appears factorial
            - design_type: 'between', 'within', or 'mixed'
        """
        self.detected_patterns = []
        self.conditions = []
        self.embedded_conditions = []

        if not flow_data:
            return self._build_result()

        flow = extract_payload_func(flow_data)

        # Run all pattern detectors (25+ patterns)

        # Flow-based randomization (Patterns 1-5)
        self._detect_block_randomizers(flow, blocks_by_id, normalize_flow_func, excluded_func)
        self._detect_standard_randomizers(flow, blocks_by_id, normalize_flow_func, excluded_func)
        self._detect_group_randomizers(flow, blocks_by_id, normalize_flow_func, excluded_func)
        self._detect_nested_randomizers(flow, normalize_flow_func, depth=0)

        # Data-based randomization (Patterns 6-9)
        self._detect_embedded_data_patterns(flow, normalize_flow_func)
        self._detect_random_number_patterns(flow, normalize_flow_func)
        self._detect_piped_text_patterns(flow, normalize_flow_func)
        self._detect_set_value_patterns(flow, normalize_flow_func)

        # Logic-based randomization (Patterns 10-13)
        self._detect_branch_patterns(flow, normalize_flow_func, excluded_func)
        self._detect_skip_logic_patterns(flow, normalize_flow_func)
        self._detect_display_logic_patterns(flow, normalize_flow_func)

        # External randomization (Patterns 14-17)
        self._detect_webservice_patterns(flow, normalize_flow_func)
        self._detect_reference_survey_patterns(flow, normalize_flow_func)
        self._detect_authenticator_patterns(flow, normalize_flow_func)
        self._detect_panel_assignment_patterns(flow, normalize_flow_func)

        # Quota-based (Patterns 18-20)
        self._detect_quota_patterns(flow, normalize_flow_func)
        self._detect_quota_action_patterns(flow, normalize_flow_func)

        # Survey structure (Patterns 21-25)
        self._detect_end_survey_patterns(flow, normalize_flow_func)
        self._detect_loop_merge_patterns(flow, normalize_flow_func)
        self._detect_conjoint_patterns(flow, normalize_flow_func)

        return self._build_result()

    def _build_result(self) -> Dict[str, Any]:
        """Build the final detection result."""
        # Deduplicate conditions
        unique_conditions = []
        seen = set()
        for cond in self.conditions:
            key = cond.lower().strip()
            if key and key not in seen:
                seen.add(key)
                unique_conditions.append(cond)

        # Determine design type
        has_between = any(p['type'] == 'block_randomizer_between' for p in self.detected_patterns)
        has_within = any(p['type'] == 'block_randomizer_within' for p in self.detected_patterns)

        if has_between and has_within:
            design_type = 'mixed'
        elif has_within:
            design_type = 'within'
        else:
            design_type = 'between'

        # Check if factorial (multiple independent randomizers at same level)
        is_factorial = len([p for p in self.detected_patterns if p.get('depth', 0) == 0]) > 1

        return {
            'conditions': unique_conditions,
            'patterns': self.detected_patterns,
            'embedded_conditions': self.embedded_conditions,
            'is_factorial': is_factorial,
            'design_type': design_type,
            'num_patterns': len(self.detected_patterns),
        }

    def _detect_block_randomizers(
        self,
        flow: List,
        blocks_by_id: Dict[str, str],
        normalize_flow_func,
        excluded_func,
        depth: int = 0,
    ):
        """Detect BlockRandomizer patterns (Pattern 1 & 2)."""
        for item in flow:
            if not isinstance(item, dict):
                continue

            if item.get('Type') == 'BlockRandomizer':
                sub_set = item.get('SubSet', 1)
                # Handle string vs int
                try:
                    sub_set_int = int(sub_set) if sub_set is not None else 1
                except (ValueError, TypeError):
                    sub_set_int = 1

                is_between = sub_set_int == 1
                pattern_type = 'block_randomizer_between' if is_between else 'block_randomizer_within'

                # Extract conditions from sub-flow
                sub_flow = normalize_flow_func(item.get('Flow', []))
                randomizer_conditions = []

                for sub_item in sub_flow:
                    if isinstance(sub_item, dict):
                        sub_type = sub_item.get('Type', '')
                        if sub_type in ('Standard', 'Block'):
                            block_id = sub_item.get('ID', '')
                            if block_id:
                                block_name = blocks_by_id.get(block_id, block_id)
                                if not excluded_func(block_name):
                                    randomizer_conditions.append(block_name)

                        # Also check for Group elements
                        elif sub_type == 'Group':
                            group_name = sub_item.get('Description', '')
                            if group_name and not excluded_func(group_name):
                                randomizer_conditions.append(group_name)

                # Only record if there are actual conditions
                if len(randomizer_conditions) >= 2:
                    self.detected_patterns.append({
                        'type': pattern_type,
                        'name': self.PATTERN_TYPES[pattern_type],
                        'conditions': randomizer_conditions,
                        'subset': sub_set_int,
                        'depth': depth,
                        'evenly_present': item.get('EvenPresentation', True),
                    })

                    if is_between:
                        self.conditions.extend(randomizer_conditions)

                    self._log(f"Detected {pattern_type}: {randomizer_conditions}")

            # Recurse
            if 'Flow' in item:
                self._detect_block_randomizers(
                    normalize_flow_func(item['Flow']),
                    blocks_by_id,
                    normalize_flow_func,
                    excluded_func,
                    depth + 1
                )

    def _detect_standard_randomizers(
        self,
        flow: List,
        blocks_by_id: Dict[str, str],
        normalize_flow_func,
        excluded_func,
        depth: int = 0,
    ):
        """Detect standard Randomizer elements (Pattern 3)."""
        for item in flow:
            if not isinstance(item, dict):
                continue

            if item.get('Type') == 'Randomizer':
                sub_flow = normalize_flow_func(item.get('Flow', []))
                randomizer_conditions = []

                for sub_item in sub_flow:
                    if isinstance(sub_item, dict):
                        # Check for blocks
                        if sub_item.get('Type') in ('Standard', 'Block'):
                            block_id = sub_item.get('ID', '')
                            if block_id:
                                block_name = blocks_by_id.get(block_id, block_id)
                                if not excluded_func(block_name):
                                    randomizer_conditions.append(block_name)

                        # Check for groups
                        elif sub_item.get('Type') == 'Group':
                            group_name = sub_item.get('Description', '')
                            if group_name and not excluded_func(group_name):
                                randomizer_conditions.append(group_name)

                if len(randomizer_conditions) >= 2:
                    self.detected_patterns.append({
                        'type': 'randomizer',
                        'name': self.PATTERN_TYPES['randomizer'],
                        'conditions': randomizer_conditions,
                        'depth': depth,
                    })
                    self.conditions.extend(randomizer_conditions)
                    self._log(f"Detected standard randomizer: {randomizer_conditions}")

            # Recurse
            if 'Flow' in item:
                self._detect_standard_randomizers(
                    normalize_flow_func(item['Flow']),
                    blocks_by_id,
                    normalize_flow_func,
                    excluded_func,
                    depth + 1
                )

    def _detect_embedded_data_patterns(self, flow: List, normalize_flow_func, depth: int = 0):
        """Detect EmbeddedData conditions inside randomizers (Pattern 4)."""
        for item in flow:
            if not isinstance(item, dict):
                continue

            flow_type = item.get('Type', '')

            # Look for EmbeddedData inside randomizers
            if flow_type in {'BlockRandomizer', 'Randomizer'}:
                sub_flow = normalize_flow_func(item.get('Flow', []))

                for sub_item in sub_flow:
                    if isinstance(sub_item, dict):
                        if sub_item.get('Type') == 'EmbeddedData':
                            self._extract_embedded_conditions(sub_item, depth)

                        # Also check blocks within randomizer for embedded data
                        if sub_item.get('Type') in ('Standard', 'Block', 'Group'):
                            nested_flow = normalize_flow_func(sub_item.get('Flow', []))
                            for nested in nested_flow:
                                if isinstance(nested, dict) and nested.get('Type') == 'EmbeddedData':
                                    self._extract_embedded_conditions(nested, depth)

            # Also check standalone EmbeddedData with SetValue (might be condition assignment)
            if flow_type == 'EmbeddedData':
                self._extract_embedded_conditions(item, depth)

            # Recurse
            if 'Flow' in item:
                self._detect_embedded_data_patterns(
                    normalize_flow_func(item['Flow']),
                    normalize_flow_func,
                    depth + 1
                )

    def _extract_embedded_conditions(self, item: Dict, depth: int):
        """Extract condition values from EmbeddedData element."""
        embedded_fields = item.get('EmbeddedData', [])
        if isinstance(embedded_fields, list):
            for field in embedded_fields:
                if isinstance(field, dict):
                    field_name = field.get('Field', '')
                    field_value = field.get('Value', '')

                    # Check if this looks like a condition assignment
                    if field_value and self._looks_like_condition_field(field_name, field_value):
                        self.embedded_conditions.append({
                            'field': field_name,
                            'value': field_value,
                            'depth': depth,
                        })

                        # Add value as condition if it looks meaningful
                        if len(field_value) > 1 and not field_value.isdigit():
                            self.conditions.append(field_value)

                        self._log(f"Detected embedded condition: {field_name}={field_value}")

    def _looks_like_condition_field(self, field_name: str, field_value: str) -> bool:
        """Check if an embedded data field looks like a condition assignment."""
        name_lower = field_name.lower()
        condition_keywords = [
            'condition', 'treatment', 'group', 'arm', 'manipulation',
            'scenario', 'stimulus', 'version', 'cond', 'grp', 'experimental'
        ]
        return any(kw in name_lower for kw in condition_keywords) or bool(field_value)

    def _detect_branch_patterns(self, flow: List, normalize_flow_func, excluded_func, depth: int = 0):
        """Detect Branch-based condition assignment (Pattern 5)."""
        for item in flow:
            if not isinstance(item, dict):
                continue

            if item.get('Type') == 'Branch':
                description = item.get('Description', '')

                # Extract potential condition from branch description
                if description and not excluded_func(description):
                    # Check if this looks like a condition branch
                    desc_lower = description.lower()
                    condition_indicators = [
                        'condition', 'treatment', 'control', 'experimental',
                        'group', 'arm', 'scenario', 'if', 'when'
                    ]

                    if any(ind in desc_lower for ind in condition_indicators):
                        self.detected_patterns.append({
                            'type': 'branch',
                            'name': self.PATTERN_TYPES['branch'],
                            'description': description,
                            'depth': depth,
                        })
                        self._log(f"Detected branch pattern: {description}")

                        # Try to extract condition name from description
                        for part in description.split():
                            if len(part) > 2 and part.lower() not in condition_indicators:
                                self.conditions.append(part)
                                break

            # Recurse
            if 'Flow' in item:
                self._detect_branch_patterns(
                    normalize_flow_func(item['Flow']),
                    normalize_flow_func,
                    excluded_func,
                    depth + 1
                )

    def _detect_nested_randomizers(self, flow: List, normalize_flow_func, depth: int = 0):
        """Detect nested/factorial randomizer designs (Pattern 6)."""
        randomizers_at_depth = []

        for item in flow:
            if not isinstance(item, dict):
                continue

            if item.get('Type') in ('Randomizer', 'BlockRandomizer'):
                randomizers_at_depth.append(item)

        # If multiple randomizers at same level, might be factorial
        if len(randomizers_at_depth) > 1:
            self.detected_patterns.append({
                'type': 'nested',
                'name': self.PATTERN_TYPES['nested'],
                'num_factors': len(randomizers_at_depth),
                'depth': depth,
            })
            self._log(f"Detected potential factorial design with {len(randomizers_at_depth)} factors at depth {depth}")

        # Recurse into each item
        for item in flow:
            if isinstance(item, dict) and 'Flow' in item:
                self._detect_nested_randomizers(
                    normalize_flow_func(item['Flow']),
                    normalize_flow_func,
                    depth + 1
                )

    def _detect_quota_patterns(self, flow: List, normalize_flow_func, depth: int = 0):
        """Detect Quota-based condition assignment (Pattern 9)."""
        for item in flow:
            if not isinstance(item, dict):
                continue

            if item.get('Type') == 'Quota':
                quota_name = item.get('Description', '') or item.get('QuotaName', '')
                self.detected_patterns.append({
                    'type': 'quota',
                    'name': self.PATTERN_TYPES['quota'],
                    'quota_name': quota_name,
                    'depth': depth,
                })
                self._log(f"Detected quota pattern: {quota_name}")

            if 'Flow' in item:
                self._detect_quota_patterns(
                    normalize_flow_func(item['Flow']),
                    normalize_flow_func,
                    depth + 1
                )

    def _detect_webservice_patterns(self, flow: List, normalize_flow_func, depth: int = 0):
        """Detect WebService-based condition assignment (Pattern 10)."""
        for item in flow:
            if not isinstance(item, dict):
                continue

            if item.get('Type') == 'WebService':
                service_url = item.get('URL', '') or item.get('WebServiceURL', '')
                self.detected_patterns.append({
                    'type': 'webservice',
                    'name': self.PATTERN_TYPES['webservice'],
                    'url': service_url,
                    'depth': depth,
                })
                self._log(f"Detected webservice pattern")

            if 'Flow' in item:
                self._detect_webservice_patterns(
                    normalize_flow_func(item['Flow']),
                    normalize_flow_func,
                    depth + 1
                )

    def _detect_random_number_patterns(self, flow: List, normalize_flow_func, depth: int = 0):
        """Detect RandomInteger/RandomNumber patterns (Pattern 11)."""
        for item in flow:
            if not isinstance(item, dict):
                continue

            # Check EmbeddedData for random number assignments
            if item.get('Type') == 'EmbeddedData':
                embedded_fields = item.get('EmbeddedData', [])
                if isinstance(embedded_fields, list):
                    for field in embedded_fields:
                        if isinstance(field, dict):
                            field_type = field.get('Type', '')
                            if field_type in ('RandomInteger', 'RandomNumber', 'Random'):
                                field_name = field.get('Field', '')
                                self.detected_patterns.append({
                                    'type': 'random_number',
                                    'name': self.PATTERN_TYPES['random_number'],
                                    'field_name': field_name,
                                    'depth': depth,
                                })
                                self._log(f"Detected random number pattern: {field_name}")

            if 'Flow' in item:
                self._detect_random_number_patterns(
                    normalize_flow_func(item['Flow']),
                    normalize_flow_func,
                    depth + 1
                )

    def _detect_end_survey_patterns(self, flow: List, normalize_flow_func, depth: int = 0):
        """Detect EndSurvey branches that might indicate conditions (Pattern 14)."""
        end_survey_count = 0

        for item in flow:
            if not isinstance(item, dict):
                continue

            if item.get('Type') == 'EndSurvey':
                end_survey_count += 1

            if 'Flow' in item:
                self._detect_end_survey_patterns(
                    normalize_flow_func(item['Flow']),
                    normalize_flow_func,
                    depth + 1
                )

        # Multiple EndSurvey elements might indicate branching conditions
        if end_survey_count > 1:
            self.detected_patterns.append({
                'type': 'end_survey',
                'name': self.PATTERN_TYPES['end_survey'],
                'count': end_survey_count,
                'depth': depth,
            })
            self._log(f"Detected multiple EndSurvey elements: {end_survey_count}")

    # ========== NEW PATTERN DETECTION METHODS (Patterns 15-25) ==========

    def _detect_group_randomizers(
        self,
        flow: List,
        blocks_by_id: Dict[str, str],
        normalize_flow_func,
        excluded_func,
        depth: int = 0,
    ):
        """Detect Group-level randomization (Pattern 4)."""
        for item in flow:
            if not isinstance(item, dict):
                continue

            if item.get('Type') == 'Group':
                group_name = item.get('Description', '') or item.get('GroupName', '')
                sub_flow = normalize_flow_func(item.get('Flow', []))

                # Check if this group contains randomized elements
                has_randomization = any(
                    isinstance(si, dict) and si.get('Type') in ('Randomizer', 'BlockRandomizer')
                    for si in sub_flow
                )

                if has_randomization and group_name:
                    self.detected_patterns.append({
                        'type': 'group_randomizer',
                        'name': self.PATTERN_TYPES.get('group_randomizer', 'Group Randomizer'),
                        'group_name': group_name,
                        'depth': depth,
                    })
                    self._log(f"Detected group randomizer: {group_name}")

            # Recurse
            if 'Flow' in item:
                self._detect_group_randomizers(
                    normalize_flow_func(item['Flow']),
                    blocks_by_id,
                    normalize_flow_func,
                    excluded_func,
                    depth + 1
                )

    def _detect_piped_text_patterns(self, flow: List, normalize_flow_func, depth: int = 0):
        """Detect Piped Text that might indicate conditions (Pattern 8)."""
        for item in flow:
            if not isinstance(item, dict):
                continue

            # Check for piped text in EmbeddedData
            if item.get('Type') == 'EmbeddedData':
                embedded_fields = item.get('EmbeddedData', [])
                if isinstance(embedded_fields, list):
                    for field in embedded_fields:
                        if isinstance(field, dict):
                            value = str(field.get('Value', ''))
                            # Piped text uses ${e://...} or ${q://...} syntax
                            if '${' in value and ('//' in value):
                                field_name = field.get('Field', '')
                                self.detected_patterns.append({
                                    'type': 'piped_text',
                                    'name': self.PATTERN_TYPES.get('piped_text', 'Piped Text Conditions'),
                                    'field_name': field_name,
                                    'value': value[:50],  # Truncate for logging
                                    'depth': depth,
                                })
                                self._log(f"Detected piped text pattern: {field_name}")

            if 'Flow' in item:
                self._detect_piped_text_patterns(
                    normalize_flow_func(item['Flow']),
                    normalize_flow_func,
                    depth + 1
                )

    def _detect_set_value_patterns(self, flow: List, normalize_flow_func, depth: int = 0):
        """Detect SetValue operations for condition assignment (Pattern 9)."""
        for item in flow:
            if not isinstance(item, dict):
                continue

            if item.get('Type') == 'EmbeddedData':
                embedded_fields = item.get('EmbeddedData', [])
                if isinstance(embedded_fields, list):
                    for field in embedded_fields:
                        if isinstance(field, dict):
                            field_type = field.get('Type', '')
                            field_name = field.get('Field', '')

                            # SetValue or explicit value assignment
                            if field_type == 'SetValue' or (
                                field.get('Value') and
                                any(kw in field_name.lower() for kw in self.CONDITION_FIELD_KEYWORDS)
                            ):
                                self.detected_patterns.append({
                                    'type': 'set_value',
                                    'name': self.PATTERN_TYPES.get('set_value', 'SetValue Operations'),
                                    'field_name': field_name,
                                    'depth': depth,
                                })
                                self._log(f"Detected set value pattern: {field_name}")

            if 'Flow' in item:
                self._detect_set_value_patterns(
                    normalize_flow_func(item['Flow']),
                    normalize_flow_func,
                    depth + 1
                )

    def _detect_skip_logic_patterns(self, flow: List, normalize_flow_func, depth: int = 0):
        """Detect Skip Logic patterns that might indicate conditions (Pattern 11)."""
        for item in flow:
            if not isinstance(item, dict):
                continue

            # Check for SkipLogic in flow elements
            skip_logic = item.get('SkipLogic') or item.get('BranchLogic')
            if skip_logic:
                description = item.get('Description', '') or item.get('ID', '')
                self.detected_patterns.append({
                    'type': 'skip_logic',
                    'name': self.PATTERN_TYPES.get('skip_logic', 'Skip Logic Patterns'),
                    'description': description[:50] if description else 'unnamed',
                    'depth': depth,
                })
                self._log(f"Detected skip logic pattern")

            if 'Flow' in item:
                self._detect_skip_logic_patterns(
                    normalize_flow_func(item['Flow']),
                    normalize_flow_func,
                    depth + 1
                )

    def _detect_display_logic_patterns(self, flow: List, normalize_flow_func, depth: int = 0):
        """Detect Display Logic patterns (Pattern 12)."""
        for item in flow:
            if not isinstance(item, dict):
                continue

            # Check for DisplayLogic
            display_logic = item.get('DisplayLogic')
            if display_logic:
                description = item.get('Description', '') or item.get('ID', '')
                self.detected_patterns.append({
                    'type': 'display_logic',
                    'name': self.PATTERN_TYPES.get('display_logic', 'Display Logic Conditions'),
                    'description': description[:50] if description else 'unnamed',
                    'depth': depth,
                })
                self._log(f"Detected display logic pattern")

            if 'Flow' in item:
                self._detect_display_logic_patterns(
                    normalize_flow_func(item['Flow']),
                    normalize_flow_func,
                    depth + 1
                )

    def _detect_reference_survey_patterns(self, flow: List, normalize_flow_func, depth: int = 0):
        """Detect Reference Survey conditions (Pattern 15)."""
        for item in flow:
            if not isinstance(item, dict):
                continue

            if item.get('Type') == 'ReferenceSurvey':
                survey_id = item.get('SurveyID', '') or item.get('ID', '')
                self.detected_patterns.append({
                    'type': 'reference_survey',
                    'name': self.PATTERN_TYPES.get('reference_survey', 'Reference Survey Conditions'),
                    'survey_id': survey_id,
                    'depth': depth,
                })
                self._log(f"Detected reference survey pattern: {survey_id}")

            if 'Flow' in item:
                self._detect_reference_survey_patterns(
                    normalize_flow_func(item['Flow']),
                    normalize_flow_func,
                    depth + 1
                )

    def _detect_authenticator_patterns(self, flow: List, normalize_flow_func, depth: int = 0):
        """Detect Authenticator-based conditions (Pattern 16)."""
        for item in flow:
            if not isinstance(item, dict):
                continue

            if item.get('Type') == 'Authenticator':
                auth_type = item.get('AuthenticatorType', '') or item.get('Type', '')
                self.detected_patterns.append({
                    'type': 'authenticator',
                    'name': self.PATTERN_TYPES.get('authenticator', 'Authenticator-Based'),
                    'auth_type': auth_type,
                    'depth': depth,
                })
                self._log(f"Detected authenticator pattern")

            if 'Flow' in item:
                self._detect_authenticator_patterns(
                    normalize_flow_func(item['Flow']),
                    normalize_flow_func,
                    depth + 1
                )

    def _detect_panel_assignment_patterns(self, flow: List, normalize_flow_func, depth: int = 0):
        """Detect Panel/Contact List assignment patterns (Pattern 17)."""
        for item in flow:
            if not isinstance(item, dict):
                continue

            # Check for panel-related elements
            if item.get('Type') in ('ContactListTrigger', 'PanelData', 'Panel'):
                self.detected_patterns.append({
                    'type': 'panel_assignment',
                    'name': self.PATTERN_TYPES.get('panel_assignment', 'Panel/Contact List Assignment'),
                    'depth': depth,
                })
                self._log(f"Detected panel assignment pattern")

            # Check EmbeddedData for panel fields
            if item.get('Type') == 'EmbeddedData':
                embedded_fields = item.get('EmbeddedData', [])
                if isinstance(embedded_fields, list):
                    for field in embedded_fields:
                        if isinstance(field, dict):
                            field_name = str(field.get('Field', '')).lower()
                            if any(kw in field_name for kw in ['panel', 'contact', 'recipient']):
                                self.detected_patterns.append({
                                    'type': 'panel_assignment',
                                    'name': self.PATTERN_TYPES.get('panel_assignment', 'Panel/Contact List Assignment'),
                                    'field_name': field.get('Field', ''),
                                    'depth': depth,
                                })
                                self._log(f"Detected panel assignment field: {field.get('Field', '')}")

            if 'Flow' in item:
                self._detect_panel_assignment_patterns(
                    normalize_flow_func(item['Flow']),
                    normalize_flow_func,
                    depth + 1
                )

    def _detect_quota_action_patterns(self, flow: List, normalize_flow_func, depth: int = 0):
        """Detect Quota Action branching patterns (Pattern 19)."""
        for item in flow:
            if not isinstance(item, dict):
                continue

            if item.get('Type') in ('QuotaCheck', 'QuotaAction'):
                action_type = item.get('ActionType', '') or item.get('Action', '')
                self.detected_patterns.append({
                    'type': 'quota_action',
                    'name': self.PATTERN_TYPES.get('quota_action', 'Quota Action Branching'),
                    'action_type': action_type,
                    'depth': depth,
                })
                self._log(f"Detected quota action pattern")

            if 'Flow' in item:
                self._detect_quota_action_patterns(
                    normalize_flow_func(item['Flow']),
                    normalize_flow_func,
                    depth + 1
                )

    def _detect_loop_merge_patterns(self, flow: List, normalize_flow_func, depth: int = 0):
        """Detect Loop & Merge patterns (Pattern 24)."""
        for item in flow:
            if not isinstance(item, dict):
                continue

            # Check for Loop & Merge type
            if item.get('Type') in ('LoopAndMerge', 'Loop'):
                loop_name = item.get('Description', '') or item.get('LoopName', '')
                iterations = item.get('LoopCount', 0) or item.get('Iterations', 0)

                self.loop_merge_detected = True
                self.detected_patterns.append({
                    'type': 'loop_merge',
                    'name': self.PATTERN_TYPES.get('loop_merge', 'Loop & Merge'),
                    'loop_name': loop_name,
                    'iterations': iterations,
                    'depth': depth,
                })
                self._log(f"Detected loop & merge pattern: {loop_name}")

            if 'Flow' in item:
                self._detect_loop_merge_patterns(
                    normalize_flow_func(item['Flow']),
                    normalize_flow_func,
                    depth + 1
                )

    def _detect_conjoint_patterns(self, flow: List, normalize_flow_func, depth: int = 0):
        """Detect Conjoint/MaxDiff experimental design patterns (Pattern 25)."""
        for item in flow:
            if not isinstance(item, dict):
                continue

            # Check for Conjoint or MaxDiff elements
            item_type = item.get('Type', '')
            if item_type in ('Conjoint', 'MaxDiff', 'ConjointQuestion', 'MaxDiffQuestion'):
                design_name = item.get('Description', '') or item.get('QuestionText', '')
                self.detected_patterns.append({
                    'type': 'conjoint',
                    'name': self.PATTERN_TYPES.get('conjoint', 'Conjoint/MaxDiff Design'),
                    'design_name': design_name[:50] if design_name else 'unnamed',
                    'design_type': item_type,
                    'depth': depth,
                })
                self._log(f"Detected conjoint/maxdiff pattern: {item_type}")

            # Also check for Randomized choice sets
            if 'RandomizedChoiceSets' in item or 'ChoiceModelTasks' in item:
                self.detected_patterns.append({
                    'type': 'conjoint',
                    'name': self.PATTERN_TYPES.get('conjoint', 'Conjoint/MaxDiff Design'),
                    'design_type': 'RandomizedChoiceSets',
                    'depth': depth,
                })
                self._log(f"Detected randomized choice sets pattern")

            if 'Flow' in item:
                self._detect_conjoint_patterns(
                    normalize_flow_func(item['Flow']),
                    normalize_flow_func,
                    depth + 1
                )


class QSFCorrections:
    """
    Handles user corrections to parsed QSF data.

    When automatic parsing has errors, this class manages:
    - Manual condition definitions
    - Scale corrections
    - Variable mapping overrides
    """

    def __init__(self, preview_result: QSFPreviewResult):
        self.original = preview_result
        self.corrections = {
            'conditions': [],
            'scales': [],
            'variable_mappings': {},
            'ignored_questions': [],
            'notes': []
        }

    def override_conditions(self, conditions: List[str]):
        """Override detected conditions with manual definitions."""
        self.corrections['conditions'] = conditions

    def add_scale(
        self,
        name: str,
        num_items: int,
        scale_points: int,
        reverse_items: List[int] = None
    ):
        """Add or correct a scale definition."""
        self.corrections['scales'].append({
            'name': name,
            'num_items': num_items,
            'scale_points': scale_points,
            'reverse_items': reverse_items or []
        })

    def ignore_question(self, question_id: str, reason: str = ""):
        """Mark a question to be ignored in simulation."""
        self.corrections['ignored_questions'].append({
            'question_id': question_id,
            'reason': reason
        })

    def add_note(self, note: str):
        """Add a note about corrections made."""
        self.corrections['notes'].append({
            'timestamp': datetime.now().isoformat(),
            'note': note
        })

    def get_final_config(self) -> Dict[str, Any]:
        """Get the final configuration after corrections."""
        return {
            'conditions': self.corrections['conditions'] if self.corrections['conditions']
                         else self.original.detected_conditions,
            'scales': self.corrections['scales'] if self.corrections['scales']
                     else self.original.detected_scales,
            'ignored_questions': self.corrections['ignored_questions'],
            'corrections_made': len(self.corrections['notes']) > 0,
            'notes': self.corrections['notes']
        }


# Export
__all__ = [
    '__version__',
    'QSFPreviewParser',
    'QSFPreviewResult',
    'QSFCorrections',
    'QuestionInfo',
    'BlockInfo',
    'LogEntry',
    'LogLevel',
    'RandomizationPatternDetector',
]
