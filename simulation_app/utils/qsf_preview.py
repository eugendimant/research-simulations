"""
Interactive QSF Preview with Error Logging
==========================================
Module for parsing, previewing, and validating Qualtrics Survey Format (QSF) files
with comprehensive error detection and logging for instructor review.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

# Version identifier to help track deployed code
__version__ = "2.1.6"  # Major: context-aware open-ended extraction, embedded data condition detection


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
    EXCLUDED_BLOCK_NAMES = {
        # Generic block names
        'block', 'block 1', 'block 2', 'block 3', 'block 4', 'block 5',
        'default question block', 'default', 'standard',
        # Structural/admin blocks
        'trash', 'trash / unused questions', 'unused',
        'intro', 'introduction', 'welcome',
        'instructions', 'general instructions',
        'consent', 'informed consent',
        'captcha', 'bot check',
        'end', 'end of survey', 'end of games', 'ending', 'debrief',
        'thank you', 'thanks',
        # Demographic/covariate blocks
        'demographics', 'demographic info', 'demographic information',
        'background', 'background info',
        # Quality control blocks
        'attention check', 'attention checks', 'quality check',
        'manipulation check', 'manipulation checks',
        # Feedback blocks
        'feedback', 'comments', 'feedback on the survey',
        'open-ended', 'open ended', 'free response',
        # Game/task specific (but not conditions)
        'game', 'task', 'main task',
        'pairing', 'pairing prompt', 'pair',
        'question', 'questions',
    }

    # Block type keywords that indicate non-condition blocks
    EXCLUDED_BLOCK_TYPES = {'Trash', 'Default'}

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

        # Validate structure
        self._validate_structure(blocks, questions_map, detected_conditions)

        self._log(
            LogLevel.INFO, "PARSE_COMPLETE",
            f"Parsing complete. {len(self.errors)} errors, {len(self.warnings)} warnings"
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
            sub_questions = []

            # For Matrix questions, Choices are the items (rows) and Answers are the scale (columns)
            if question_type == 'Matrix':
                # The "Choices" in Matrix are actually the items/statements
                # The "Answers" are the response scale options
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
            else:
                # For non-matrix questions with sub-questions
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

            # ROBUST SCALE POINT DETECTION
            scale_points = self._detect_scale_points(payload, question_type, selector, choices, sub_questions)

            # Get the data export tag (actual variable name in exported data)
            data_export_tag = payload.get('DataExportTag', q_id)

            questions_map[q_id] = QuestionInfo(
                question_id=q_id,
                question_text=question_text[:200] + ('...' if len(question_text) > 200 else ''),
                question_type=category,
                block_name='',  # Will be filled when mapping
                choices=choices,
                scale_points=scale_points,
                is_matrix=is_matrix,
                sub_questions=sub_questions
            )

            self._log(
                LogLevel.INFO, "QUESTION",
                f"Parsed question {q_id}: {category} (scale_points={scale_points})",
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

    def _categorize_question(self, q_type: str, selector: str) -> str:
        """Categorize question type."""
        if q_type == 'Matrix':
            return 'Likert Scale Matrix'
        elif q_type == 'MC':
            if selector in ['Likert', 'SAVR']:
                return 'Single Choice (Radio)'
            elif selector in ['MAVR', 'MACOL']:
                return 'Multiple Choice'
            else:
                return 'Multiple Choice'
        elif q_type == 'TE':
            return 'Text Entry'
        elif q_type == 'Slider':
            return 'Slider'
        elif q_type == 'DB':
            return 'Descriptive Text'
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

    def _is_excluded_block_name(self, block_name: str) -> bool:
        """Check if a block name is in the exclusion list."""
        if not block_name:
            return True
        normalized = block_name.lower().strip()
        # Check exact match
        if normalized in self.EXCLUDED_BLOCK_NAMES:
            return True
        # Check if it starts with "block" followed by only numbers/spaces
        if re.match(r'^block\s*\d*$', normalized):
            return True
        return False

    def _detect_conditions(
        self,
        flow_data: Optional[Any],
        blocks: List[BlockInfo]
    ) -> List[str]:
        """Detect experimental conditions from randomizers and flow.

        Priority order:
        1. Blocks inside BlockRandomizer with SubSet=1 (between-subjects)
        2. Blocks inside Randomizer elements
        3. Block names containing condition keywords (strict fallback)

        Blocks are EXCLUDED if:
        - Their name is in EXCLUDED_BLOCK_NAMES
        - Their block type is 'Trash' or 'Default'
        - They have generic names like 'Block 1', 'Block 2'
        """
        conditions: List[str] = []
        blocks_by_id = {block.block_id: block.block_name for block in blocks}
        # Also track block types for exclusion
        blocks_type_by_id = {block.block_id: getattr(block, 'block_type', 'Standard')
                            for block in blocks}

        # Parse flow for randomizer/branch elements with block IDs.
        if flow_data:
            flow = self._extract_flow_payload(flow_data)
            self._find_conditions_in_flow(flow, conditions, blocks_by_id)

        # Filter out excluded blocks from conditions
        conditions = [c for c in conditions if not self._is_excluded_block_name(c)]

        # Fallback: use block names that look like conditions (STRICT mode).
        # Only use if no conditions were found from randomizers
        if not conditions:
            self._log(
                LogLevel.INFO, "CONDITIONS",
                "No conditions detected from randomizers. Using fallback detection."
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
                                    if not self._is_excluded_block_name(block_name):
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

    def _dedupe_conditions(self, conditions: List[str]) -> List[str]:
        seen = set()
        unique: List[str] = []
        for cond in conditions:
            key = cond.lower().strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(cond)
        return unique

    def _detect_scales(self, questions_map: Dict) -> List[Dict[str, Any]]:
        """Detect multi-item scales from question patterns.

        IMPORTANT: Only detects scales from QSF structure. Does NOT add default scales.
        Default scales should only be added at the app layer if user hasn't specified DVs.

        Returns scales with:
        - name: Display name for the scale
        - variable_name: Variable prefix for export
        - items: Number of items in the scale
        - scale_points: Number of response options (MUST be from QSF, not assumed)
        - type: 'matrix' or 'numbered_items'
        """
        scales = []
        scale_patterns = {}
        seen_scale_names = set()  # Track to prevent duplicates

        # Safety check - ensure questions_map is a dict
        if not isinstance(questions_map, dict):
            self._log(LogLevel.WARNING, "SCALES", f"questions_map is {type(questions_map).__name__}, expected dict")
            return scales

        # Look for numbered items (e.g., "Ownership_1", "Ownership_2")
        for q_id, q_info in questions_map.items():
            if q_info.is_matrix or q_info.question_type == 'Likert Scale Matrix':
                # Matrix questions are scales - use question_id as variable name
                scale_name = q_info.question_text[:50].strip() or q_id
                variable_name = q_id

                # Skip if we've already seen this scale
                name_key = variable_name.lower()
                if name_key in seen_scale_names:
                    continue
                seen_scale_names.add(name_key)

                num_items = len(q_info.sub_questions) if q_info.sub_questions else 1
                scale_pts = q_info.scale_points  # Keep as-is, may be None

                scales.append({
                    'name': scale_name,
                    'variable_name': variable_name,
                    'question_id': q_id,
                    'items': num_items,
                    'scale_points': scale_pts,
                    'type': 'matrix'
                })

                self._log(
                    LogLevel.INFO, "SCALE",
                    f"Detected matrix scale: {scale_name} ({num_items} items, {scale_pts or 'unknown'}-point)"
                )

            # Check for numbered pattern (e.g., WTP_1, WTP_2)
            match = re.match(r'^(.+?)[-_]?(\d+)$', q_id)
            if match:
                base_name = match.group(1).rstrip('_-')
                item_num = int(match.group(2))
                if base_name not in scale_patterns:
                    scale_patterns[base_name] = []
                scale_patterns[base_name].append({
                    'item_num': item_num,
                    'scale_points': q_info.scale_points,
                    'question_id': q_id
                })

        # Consolidate numbered scales
        for base_name, items in scale_patterns.items():
            if len(items) >= 2:  # At least 2 items = scale
                # Skip if this base_name was already detected as a matrix
                name_key = base_name.lower()
                if name_key in seen_scale_names:
                    continue
                seen_scale_names.add(name_key)

                # Use the most common scale_points value from items
                valid_points = [i['scale_points'] for i in items if i['scale_points'] is not None]
                scale_pts = max(set(valid_points), key=valid_points.count) if valid_points else None

                scales.append({
                    'name': base_name,
                    'variable_name': base_name,
                    'items': len(items),
                    'scale_points': scale_pts,
                    'type': 'numbered_items'
                })
                self._log(
                    LogLevel.INFO, "SCALE",
                    f"Detected numbered scale: {base_name} ({len(items)} items, {scale_pts or 'unknown'}-point)"
                )

        return scales

    def _detect_open_ended(self, questions_map: Dict) -> List[str]:
        """Detect open-ended text entry questions."""
        open_ended = []
        # Safety check - ensure questions_map is a dict
        if not isinstance(questions_map, dict):
            return open_ended
        for q_id, q_info in questions_map.items():
            if q_info.question_type == 'Text Entry':
                open_ended.append(q_id)
                self._log(
                    LogLevel.INFO, "OPEN_ENDED",
                    f"Detected open-ended question: {q_id}"
                )
        return open_ended

    def _detect_attention_checks(self, questions_map: Dict) -> List[str]:
        """Detect attention check questions."""
        attention_checks = []
        attention_keywords = [
            'attention', 'check', 'please select', 'instructed',
            'carefully read', 'quality', 'verify', 'bot'
        ]

        # Safety check - ensure questions_map is a dict
        if not isinstance(questions_map, dict):
            return attention_checks

        for q_id, q_info in questions_map.items():
            text = q_info.question_text.lower()
            if any(kw in text for kw in attention_keywords):
                attention_checks.append(q_id)
                self._log(
                    LogLevel.INFO, "ATTENTION_CHECK",
                    f"Detected attention check: {q_id}"
                )

        return attention_checks

    def _extract_open_ended_details(
        self,
        questions_map: Dict,
        blocks: List[BlockInfo]
    ) -> List[Dict[str, Any]]:
        """Extract detailed information about open-ended text entry questions.

        This provides FULL context for each open-ended question so the text generator
        can create appropriate, study-specific responses.

        Returns list of dicts with:
        - question_id: Question identifier
        - variable_name: Export variable name
        - question_text: Full question text
        - block_name: Which block it's in
        - context_type: What kind of response expected (feedback, explanation, description, etc.)
        - preceding_questions: Context from questions before this one
        """
        open_ended_details = []

        # Safety check
        if not isinstance(questions_map, dict):
            return open_ended_details

        # Build a list of all question texts for context
        all_question_texts = []
        for q_id, q_info in questions_map.items():
            if q_info.question_text:
                all_question_texts.append({
                    'id': q_id,
                    'text': q_info.question_text,
                    'type': q_info.question_type,
                    'block': q_info.block_name
                })

        for q_id, q_info in questions_map.items():
            if q_info.question_type == 'Text Entry':
                # Skip questions that look like ID entry fields
                text_lower = q_info.question_text.lower()
                skip_patterns = [
                    'mturk', 'worker id', 'prolific', 'participant id',
                    'email', 'name', 'address', 'phone', 'zip', 'zipcode',
                    'age', 'year born'
                ]
                if any(pat in text_lower for pat in skip_patterns):
                    continue

                # Determine context type based on question text
                context_type = self._classify_open_ended_type(q_info.question_text)

                # Get surrounding questions for context
                preceding_questions = self._get_preceding_context(q_id, q_info.block_name, blocks)

                detail = {
                    'question_id': q_id,
                    'variable_name': q_id.replace(' ', '_'),
                    'question_text': q_info.question_text,
                    'block_name': q_info.block_name,
                    'context_type': context_type,
                    'preceding_questions': preceding_questions,
                }

                open_ended_details.append(detail)

                self._log(
                    LogLevel.INFO, "OPEN_ENDED_DETAIL",
                    f"Extracted open-ended: {q_id} ({context_type}) - {q_info.question_text[:50]}..."
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

        # Check for empty blocks
        for block in blocks:
            if not hasattr(block, 'questions') or not hasattr(block, 'is_randomizer'):
                continue
            if len(block.questions) == 0 and not block.is_randomizer:
                self._log(
                    LogLevel.WARNING, "VALIDATION",
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
    'LogLevel'
]
