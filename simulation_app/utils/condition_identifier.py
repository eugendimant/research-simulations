"""
Enhanced Condition and Variable Identifier
==========================================
Deep analysis of Qualtrics QSF files to identify:
- Experimental conditions and treatment levels
- Independent variables (IVs) and their manipulation
- Dependent variables (DVs) and scales
- Randomization structure and level

Uses multiple heuristics and cross-references with preregistration documents
to maximize identification accuracy.
"""

from __future__ import annotations

import re
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


class VariableRole(Enum):
    """Role classifications for survey variables."""
    CONDITION = "Condition"
    INDEPENDENT_VARIABLE = "Independent variable"
    PRIMARY_OUTCOME = "Primary outcome"
    SECONDARY_OUTCOME = "Secondary outcome"
    MEDIATOR = "Mediator"
    MODERATOR = "Moderator"
    COVARIATE = "Covariate"
    DEMOGRAPHICS = "Demographics"
    ATTENTION_CHECK = "Attention check"
    MANIPULATION_CHECK = "Manipulation check"
    OPEN_ENDED = "Open-ended"
    FILLER = "Filler"
    OTHER = "Other"


class RandomizationLevel(Enum):
    """Level at which randomization occurs."""
    PARTICIPANT = "Participant-level"
    GROUP = "Group/Cluster-level"
    WITHIN_SUBJECT = "Within-subject"
    MULTIPLE_STAGES = "Multiple stages"
    NOT_RANDOMIZED = "Not randomized"


@dataclass
class IdentifiedVariable:
    """A variable identified from the QSF with its properties."""
    variable_id: str  # The Qualtrics variable name/ID
    display_name: str  # Human-readable name
    role: VariableRole
    confidence: float  # 0-1 confidence score
    source: str  # "QSF", "Preregistration", or "Both"
    question_text: str = ""
    scale_points: Optional[int] = None
    choices: List[str] = field(default_factory=list)
    is_part_of_scale: bool = False
    parent_scale: Optional[str] = None
    reasons: List[str] = field(default_factory=list)  # Why this role was assigned


@dataclass
class IdentifiedCondition:
    """An experimental condition identified from the survey."""
    name: str
    factor: str  # Which factor this belongs to
    source: str  # "QSF Flow", "Block Name", "Preregistration", etc.
    confidence: float
    block_ids: List[str] = field(default_factory=list)
    randomizer_id: Optional[str] = None


@dataclass
class IdentifiedFactor:
    """A factor (independent variable) with its levels."""
    name: str
    levels: List[str]
    is_between_subjects: bool = True
    source: str = "Inferred"
    confidence: float = 0.5


@dataclass
class IdentifiedScale:
    """A multi-item scale identified from the survey."""
    name: str
    variable_name: str  # Qualtrics variable prefix
    items: List[str]  # List of item variable names
    num_items: int
    scale_points: int
    role: VariableRole = VariableRole.OTHER
    confidence: float = 0.5
    reverse_items: List[int] = field(default_factory=list)


@dataclass
class RandomizationInfo:
    """Information about randomization structure."""
    level: RandomizationLevel
    randomizers: List[Dict[str, Any]]
    evenly_distributed: bool = True
    conditions_per_participant: int = 1


@dataclass
class DesignAnalysisResult:
    """Complete result of design analysis."""
    conditions: List[IdentifiedCondition]
    factors: List[IdentifiedFactor]
    scales: List[IdentifiedScale]
    variables: List[IdentifiedVariable]
    randomization: RandomizationInfo
    open_ended_questions: List[str]
    attention_checks: List[str]
    manipulation_checks: List[str]
    warnings: List[str]
    suggestions: List[str]


class EnhancedConditionIdentifier:
    """
    Advanced analyzer for Qualtrics survey design.

    This class performs deep analysis of QSF files to identify:
    - Experimental conditions from randomizer structures
    - Treatment variables and their levels
    - Dependent variables and scales
    - Randomization level and structure

    It also integrates preregistration information to improve accuracy.
    """

    def __init__(self):
        self.warnings: List[str] = []
        self.suggestions: List[str] = []

    def analyze(
        self,
        qsf_data: Dict[str, Any],
        prereg_outcomes: str = "",
        prereg_iv: str = "",
        prereg_text: str = "",
        prereg_pdf_text: str = "",
    ) -> DesignAnalysisResult:
        """
        Perform comprehensive analysis of survey design.

        Args:
            qsf_data: Parsed QSF JSON data
            prereg_outcomes: Primary outcome variables from preregistration
            prereg_iv: Independent variables from preregistration
            prereg_text: Additional preregistration notes
            prereg_pdf_text: Extracted text from preregistration PDF

        Returns:
            DesignAnalysisResult with all identified design elements
        """
        self.warnings = []
        self.suggestions = []

        # Parse QSF structure
        elements = qsf_data.get('SurveyElements', [])
        blocks_map = self._extract_blocks(elements)
        questions_map = self._extract_questions(elements)
        flow_data = self._extract_flow(elements)
        embedded_data = self._extract_embedded_data(elements)

        # Analyze randomization structure (this is the key improvement)
        randomization, raw_conditions = self._analyze_randomization(
            flow_data, blocks_map, questions_map
        )

        # Extract conditions with confidence scores
        conditions = self._identify_conditions(
            raw_conditions, flow_data, blocks_map, prereg_iv, prereg_text, prereg_pdf_text
        )

        # Infer factors from conditions
        factors = self._infer_factors(conditions, prereg_iv, prereg_text)

        # Identify scales from question patterns
        scales = self._identify_scales(questions_map, prereg_outcomes, prereg_text)

        # Classify all variables
        variables = self._classify_variables(
            questions_map, scales, conditions,
            prereg_outcomes, prereg_iv, prereg_text
        )

        # Find open-ended questions
        open_ended = self._find_open_ended(questions_map)

        # Find attention and manipulation checks
        attention_checks = self._find_attention_checks(questions_map)
        manipulation_checks = self._find_manipulation_checks(questions_map, conditions)

        return DesignAnalysisResult(
            conditions=conditions,
            factors=factors,
            scales=scales,
            variables=variables,
            randomization=randomization,
            open_ended_questions=open_ended,
            attention_checks=attention_checks,
            manipulation_checks=manipulation_checks,
            warnings=self.warnings,
            suggestions=self.suggestions,
        )

    def _extract_blocks(self, elements: List[Dict]) -> Dict[str, Dict]:
        """Extract block definitions from QSF elements.

        Handles both QSF formats:
        - Dict format: {"BL_123": {"Description": "Block 1", ...}}
        - List format: [{"ID": "BL_123", "Description": "Block 1", ...}]
        """
        blocks = {}
        for element in elements:
            if element.get('Element') == 'BL':
                payload = element.get('Payload', {})

                # Handle list format (newer QSF exports)
                if isinstance(payload, list):
                    for block_data in payload:
                        if isinstance(block_data, dict):
                            block_id = block_data.get('ID', '')
                            if not block_id:
                                continue
                            blocks[block_id] = {
                                'name': block_data.get('Description', f'Block {block_id}'),
                                'type': block_data.get('Type', 'Standard'),
                                'elements': block_data.get('BlockElements', []),
                                'options': block_data.get('Options', {}),
                            }
                # Handle dict format (older QSF exports)
                elif isinstance(payload, dict):
                    for dict_key, block_data in payload.items():
                        if isinstance(block_data, dict):
                            # Use the ID field if available, otherwise use the dict key
                            # Flow elements reference blocks by ID field
                            block_id = block_data.get('ID', dict_key)
                            blocks[block_id] = {
                                'name': block_data.get('Description', f'Block {block_id}'),
                                'type': block_data.get('Type', 'Standard'),
                                'elements': block_data.get('BlockElements', []),
                                'options': block_data.get('Options', {}),
                            }
                            # Also store with dict key for compatibility
                            if dict_key != block_id:
                                blocks[dict_key] = blocks[block_id]
        return blocks

    def _extract_questions(self, elements: List[Dict]) -> Dict[str, Dict]:
        """Extract question definitions with full metadata."""
        questions = {}
        for element in elements:
            if element.get('Element') == 'SQ':
                payload = element.get('Payload', {})
                q_id = payload.get('QuestionID', element.get('PrimaryAttribute', ''))

                # Get question text and clean HTML
                question_text = payload.get('QuestionText', '')
                question_text_clean = re.sub(r'<[^>]+>', '', question_text).strip()

                # Get question type info
                q_type = payload.get('QuestionType', '')
                selector = payload.get('Selector', '')
                sub_selector = payload.get('SubSelector', '')

                # Extract choices
                choices = []
                choices_data = payload.get('Choices', {})
                if isinstance(choices_data, dict):
                    for choice_id, choice_data in sorted(choices_data.items(), key=lambda x: str(x[0])):
                        if isinstance(choice_data, dict):
                            choices.append(choice_data.get('Display', str(choice_data)))
                        else:
                            choices.append(str(choice_data))

                # Extract sub-questions (for matrix questions)
                sub_questions = []
                answers_data = payload.get('Answers', {})
                if isinstance(answers_data, dict):
                    for ans_id, ans_data in sorted(answers_data.items(), key=lambda x: str(x[0])):
                        if isinstance(ans_data, dict):
                            sub_questions.append({
                                'id': ans_id,
                                'text': ans_data.get('Display', ''),
                            })

                # Also check Subquestions for some matrix types
                subq_data = payload.get('Subquestions', {})
                if isinstance(subq_data, dict) and not sub_questions:
                    for sq_id, sq_data in sorted(subq_data.items(), key=lambda x: str(x[0])):
                        if isinstance(sq_data, dict):
                            sub_questions.append({
                                'id': sq_id,
                                'text': sq_data.get('Display', ''),
                            })

                # Get data export tag (actual variable name)
                data_export_tag = payload.get('DataExportTag', q_id)

                # Determine if it's a matrix/scale
                is_matrix = q_type == 'Matrix' or (q_type == 'MC' and selector in ['Likert', 'Bipolar'])

                questions[q_id] = {
                    'question_id': q_id,
                    'export_tag': data_export_tag,
                    'question_text': question_text_clean,
                    'question_text_raw': question_text,
                    'type': q_type,
                    'selector': selector,
                    'sub_selector': sub_selector,
                    'choices': choices,
                    'scale_points': len(choices) if choices else None,
                    'is_matrix': is_matrix,
                    'sub_questions': sub_questions,
                    'validation': payload.get('Validation', {}),
                    'recoded_values': payload.get('RecodeValues', {}),
                }
        return questions

    def _extract_flow(self, elements: List[Dict]) -> Optional[Dict]:
        """Extract survey flow structure."""
        for element in elements:
            if element.get('Element') == 'FL':
                return element
        return None

    def _extract_embedded_data(self, elements: List[Dict]) -> List[str]:
        """Extract embedded data field names."""
        fields = []
        for element in elements:
            if element.get('Element') == 'ED':
                payload = element.get('Payload', {})
                flow = payload.get('Flow', [])
                for item in flow:
                    if isinstance(item, dict):
                        ed_list = item.get('EmbeddedData', [])
                        if isinstance(ed_list, list):
                            for ed in ed_list:
                                if isinstance(ed, dict):
                                    fields.append(ed.get('Field', ''))
        return [f for f in fields if f]

    def _analyze_randomization(
        self,
        flow_data: Optional[Dict],
        blocks_map: Dict[str, Dict],
        questions_map: Dict[str, Dict],
    ) -> Tuple[RandomizationInfo, List[Dict]]:
        """
        Deep analysis of randomization structure.

        This is the core improvement - we analyze:
        1. Randomizer elements in the flow
        2. What blocks are being randomized
        3. Whether it's evenly distributed
        4. The level of randomization (participant vs group)
        """
        raw_conditions = []
        randomizers = []

        if not flow_data:
            return RandomizationInfo(
                level=RandomizationLevel.NOT_RANDOMIZED,
                randomizers=[],
                evenly_distributed=True,
                conditions_per_participant=1,
            ), []

        payload = flow_data.get('Payload', {})
        flow = payload.get('Flow', [])

        # Recursively find all randomizers
        self._find_randomizers(flow, randomizers, blocks_map, raw_conditions, depth=0)

        # Determine randomization level
        level = RandomizationLevel.NOT_RANDOMIZED
        evenly_distributed = True
        conditions_per_participant = 1

        if randomizers:
            level = RandomizationLevel.PARTICIPANT

            # Check for group randomization markers
            for rand in randomizers:
                if 'group' in rand.get('description', '').lower():
                    level = RandomizationLevel.GROUP

            # Check if within-subject (same participant sees multiple conditions)
            for rand in randomizers:
                if rand.get('randomization_type') == 'present_all':
                    level = RandomizationLevel.WITHIN_SUBJECT
                    conditions_per_participant = rand.get('num_branches', 1)

            # Check for multiple randomization stages
            if len(randomizers) > 1:
                nested = any(r.get('depth', 0) > 0 for r in randomizers)
                if nested:
                    level = RandomizationLevel.MULTIPLE_STAGES

            # Check even distribution
            for rand in randomizers:
                if not rand.get('evenly_present', True):
                    evenly_distributed = False

        return RandomizationInfo(
            level=level,
            randomizers=randomizers,
            evenly_distributed=evenly_distributed,
            conditions_per_participant=conditions_per_participant,
        ), raw_conditions

    def _find_randomizers(
        self,
        flow: List,
        randomizers: List[Dict],
        blocks_map: Dict[str, Dict],
        raw_conditions: List[Dict],
        depth: int = 0,
    ):
        """Recursively find and analyze randomizer elements."""
        for item in flow:
            if not isinstance(item, dict):
                continue

            flow_type = item.get('Type', '')

            if flow_type == 'Randomizer':
                rand_info = self._parse_randomizer(item, blocks_map, depth)
                randomizers.append(rand_info)

                # Extract conditions from this randomizer
                for branch in rand_info.get('branches', []):
                    raw_conditions.append({
                        'name': branch.get('name', ''),
                        'block_id': branch.get('block_id', ''),
                        'source': 'Randomizer Flow',
                        'randomizer_id': rand_info.get('flow_id', ''),
                        'confidence': 0.9,  # High confidence for randomizer-based detection
                    })

                # Recurse into randomizer's flow
                sub_flow = item.get('Flow', [])
                self._find_randomizers(sub_flow, randomizers, blocks_map, raw_conditions, depth + 1)

            elif flow_type == 'BlockRandomizer':
                # BlockRandomizer is another randomization type
                rand_info = self._parse_block_randomizer(item, blocks_map, depth)
                randomizers.append(rand_info)

                # Extract conditions if SubSet=1 (between-subjects assignment)
                # Also check sub_set value directly to handle string "1" case
                is_between_subjects = (
                    rand_info.get('randomization_type') == 'present_one' or
                    rand_info.get('sub_set') == 1 or
                    str(rand_info.get('sub_set', '')) == '1'
                )
                if is_between_subjects:
                    for branch in rand_info.get('branches', []):
                        raw_conditions.append({
                            'name': branch.get('name', ''),
                            'block_id': branch.get('block_id', ''),
                            'source': 'BlockRandomizer',
                            'randomizer_id': rand_info.get('flow_id', ''),
                            'confidence': 0.85,  # High confidence for BlockRandomizer-based detection
                        })

            elif flow_type == 'Branch':
                # Branches can also indicate conditions
                branch_info = self._parse_branch(item, blocks_map)
                if branch_info.get('looks_like_condition'):
                    raw_conditions.append({
                        'name': branch_info.get('description', ''),
                        'source': 'Branch Logic',
                        'confidence': 0.6,  # Lower confidence for branches
                    })

            # Recurse into nested flows
            if 'Flow' in item and flow_type != 'Randomizer':
                self._find_randomizers(item['Flow'], randomizers, blocks_map, raw_conditions, depth)

    def _parse_randomizer(
        self,
        item: Dict,
        blocks_map: Dict[str, Dict],
        depth: int,
    ) -> Dict:
        """Parse a Randomizer element in detail."""
        flow_id = item.get('FlowID', item.get('ID', ''))
        description = item.get('Description', '')

        # Get randomization settings
        randomize_count = item.get('RandomizeCount', None)
        evenly_present = item.get('EvenPresentation', True)

        # Determine randomization type
        rand_type = 'present_one'  # Default: show one of the branches
        if randomize_count is not None:
            if isinstance(randomize_count, int) and randomize_count > 1:
                rand_type = 'present_multiple'
            elif str(randomize_count).lower() == 'all':
                rand_type = 'present_all'

        # Extract branches (what gets randomized)
        branches = []
        sub_flow = item.get('Flow', [])
        for sub_item in sub_flow:
            if isinstance(sub_item, dict):
                sub_type = sub_item.get('Type', '')
                if sub_type == 'Block':
                    block_id = sub_item.get('ID', '')
                    block_name = blocks_map.get(block_id, {}).get('name', block_id)
                    branches.append({
                        'type': 'block',
                        'block_id': block_id,
                        'name': block_name,
                    })
                elif sub_type == 'Group':
                    group_desc = sub_item.get('Description', 'Group')
                    # Groups can contain multiple blocks
                    group_blocks = []
                    group_flow = sub_item.get('Flow', [])
                    for gf_item in group_flow:
                        if isinstance(gf_item, dict) and gf_item.get('Type') == 'Block':
                            group_blocks.append(gf_item.get('ID', ''))
                    branches.append({
                        'type': 'group',
                        'name': group_desc,
                        'blocks': group_blocks,
                    })

        return {
            'flow_id': flow_id,
            'description': description,
            'randomization_type': rand_type,
            'randomize_count': randomize_count,
            'evenly_present': evenly_present,
            'num_branches': len(branches),
            'branches': branches,
            'depth': depth,
        }

    def _parse_block_randomizer(
        self,
        item: Dict,
        blocks_map: Dict[str, Dict],
        depth: int,
    ) -> Dict:
        """Parse a BlockRandomizer element.

        BlockRandomizer with SubSet=1 randomly assigns participants to one of the blocks,
        making each block a condition. This is a common way to implement between-subjects
        designs in Qualtrics.
        """
        flow_id = item.get('FlowID', item.get('ID', ''))
        sub_set = item.get('SubSet', None)
        randomize_all = item.get('RandomizeAll', False)

        # Get the blocks being randomized
        branches = []
        sub_flow = item.get('Flow', [])
        for sub_item in sub_flow:
            if isinstance(sub_item, dict):
                sub_type = sub_item.get('Type', '')
                if sub_type in ('Standard', 'Block'):
                    block_id = sub_item.get('ID', '')
                    block_name = blocks_map.get(block_id, {}).get('name', block_id)
                    branches.append({
                        'type': 'block',
                        'block_id': block_id,
                        'name': block_name,
                    })

        # Determine randomization type based on SubSet
        # SubSet=1 means one block per participant (between-subjects conditions)
        # SubSet=N>1 or "All" means multiple blocks per participant (within-subjects)
        # Note: SubSet can be int or string depending on QSF version
        rand_type = 'present_one'
        if sub_set is not None:
            sub_set_str = str(sub_set).lower()
            if sub_set_str == '1':
                rand_type = 'present_one'  # Between-subjects: each participant sees one
            elif sub_set_str == 'all' or randomize_all:
                rand_type = 'present_all'  # All blocks shown (order randomized)
            elif sub_set_str.isdigit() and int(sub_set_str) > 1:
                rand_type = 'present_multiple'

        return {
            'flow_id': flow_id,
            'type': 'block_randomizer',
            'randomization_type': rand_type,
            'sub_set': sub_set,
            'num_branches': len(branches),
            'branches': branches,
            'depth': depth,
        }

    def _parse_branch(self, item: Dict, blocks_map: Dict[str, Dict]) -> Dict:
        """Parse a Branch element to check if it represents a condition."""
        description = item.get('Description', '')

        # Check if this looks like a condition branch
        condition_keywords = [
            'condition', 'treatment', 'group', 'arm',
            'control', 'experimental', 'manipulation'
        ]
        looks_like_condition = any(
            kw in description.lower() for kw in condition_keywords
        )

        return {
            'description': description,
            'looks_like_condition': looks_like_condition,
        }

    def _identify_conditions(
        self,
        raw_conditions: List[Dict],
        flow_data: Optional[Dict],
        blocks_map: Dict[str, Dict],
        prereg_iv: str,
        prereg_text: str,
        prereg_pdf_text: str,
    ) -> List[IdentifiedCondition]:
        """
        Identify experimental conditions from QSF structure ONLY.

        We ONLY use:
        1. Randomizer structure from QSF flow (highest reliability)
        2. Block names that are part of randomizers
        3. Embedded data fields that indicate conditions

        We do NOT parse preregistration text for conditions as this produces
        garbage fragments. The user can manually add conditions if needed.
        """
        conditions = []
        seen_names = set()

        # Process raw conditions from randomizer analysis (QSF structure only)
        for rc in raw_conditions:
            name = self._normalize_condition_name(rc.get('name', ''))
            if name and name.lower() not in seen_names:
                # Skip generic/default block names
                if name.lower() in ('default question block', 'block', 'standard'):
                    continue
                seen_names.add(name.lower())
                conditions.append(IdentifiedCondition(
                    name=name,
                    factor='Treatment',
                    source='QSF Randomizer',
                    confidence=0.9,  # High confidence for QSF-detected
                    block_ids=[rc.get('block_id')] if rc.get('block_id') else [],
                    randomizer_id=rc.get('randomizer_id'),
                ))

        # Fallback: look at block names that suggest conditions
        # Only if no conditions found from randomizer
        if not conditions:
            for block_id, block_data in blocks_map.items():
                block_name = block_data.get('name', '')
                # Skip generic blocks
                if block_name.lower() in ('default question block', 'block', 'standard', 'trash / unused questions'):
                    continue
                if self._looks_like_condition(block_name):
                    name = self._normalize_condition_name(block_name)
                    if name and name.lower() not in seen_names:
                        seen_names.add(name.lower())
                        conditions.append(IdentifiedCondition(
                            name=name,
                            factor='Treatment',
                            source='QSF Block Name',
                            confidence=0.7,
                            block_ids=[block_id],
                        ))

        if not conditions:
            self.suggestions.append(
                "No experimental conditions auto-detected from QSF. "
                "Please add your conditions manually below."
            )

        return conditions

    def _extract_conditions_from_prereg(
        self,
        prereg_iv: str,
        prereg_text: str,
        prereg_pdf_text: str,
    ) -> List[str]:
        """Extract condition names from preregistration text."""
        conditions = []
        all_text = f"{prereg_iv}\n{prereg_text}\n{prereg_pdf_text}"

        # Pattern 1: "X vs Y" or "X versus Y"
        vs_pattern = r'([^,;.\n]+?)\s+(?:vs\.?|versus|compared to|compared with)\s+([^,;.\n]+)'
        for match in re.finditer(vs_pattern, all_text, re.IGNORECASE):
            conditions.append(match.group(1).strip())
            conditions.append(match.group(2).strip())

        # Pattern 2: Explicit condition mentions
        cond_pattern = r'(?:condition|treatment|group|arm)s?\s*[:=]?\s*([^,;.\n]+(?:[,;]\s*[^,;.\n]+)*)'
        for match in re.finditer(cond_pattern, all_text, re.IGNORECASE):
            parts = re.split(r'[,;/|]', match.group(1))
            for part in parts:
                cleaned = self._normalize_condition_name(part)
                if cleaned and len(cleaned) > 1:
                    conditions.append(cleaned)

        # Pattern 3: Between-subjects design mentions
        design_pattern = r'(\d+)\s*[xX×]\s*(\d+)'
        for match in re.finditer(design_pattern, all_text):
            self.suggestions.append(
                f"Detected factorial design: {match.group(1)} x {match.group(2)}"
            )

        return list(set(conditions))

    def _looks_like_condition(self, text: str) -> bool:
        """Check if text looks like a condition name."""
        if not text:
            return False
        lower = text.lower()
        keywords = [
            'condition', 'treatment', 'control', 'experimental',
            'group', 'arm', 'manipulation', 'scenario', 'stimulus'
        ]
        return any(kw in lower for kw in keywords)

    def _normalize_condition_name(self, name: str) -> str:
        """Normalize a condition name."""
        if not name:
            return ""
        # Remove HTML
        cleaned = re.sub(r'<[^>]+>', '', str(name))
        # Collapse whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        # Remove common prefixes
        cleaned = re.sub(
            r'^(condition|treatment|group|arm|variant|scenario|manipulation)\s*[:\-]\s*',
            '', cleaned, flags=re.IGNORECASE
        )
        return cleaned.strip(' -:')

    def _infer_factors(
        self,
        conditions: List[IdentifiedCondition],
        prereg_iv: str,
        prereg_text: str,
    ) -> List[IdentifiedFactor]:
        """
        Infer factor structure from conditions.

        Handles:
        - Simple designs (single factor)
        - Factorial designs (A x B)
        - Complex nested designs
        """
        if not conditions:
            return [IdentifiedFactor(
                name='Condition',
                levels=['Condition A'],
                is_between_subjects=True,
                source='Default',
                confidence=0.0,
            )]

        condition_names = [c.name for c in conditions]

        # Check for factorial notation in condition names (e.g., "High x Present")
        separators = [' x ', ' X ', ' × ', ' | ', ' + ']
        detected_sep = None

        for sep in separators:
            if any(sep in name for name in condition_names):
                detected_sep = sep
                break

        if detected_sep:
            # Factorial design detected
            split_conditions = [name.split(detected_sep) for name in condition_names]
            max_parts = max(len(parts) for parts in split_conditions)

            # Verify consistent structure
            if all(len(parts) == max_parts for parts in split_conditions):
                factors = []
                for i in range(max_parts):
                    levels = sorted(set(parts[i].strip() for parts in split_conditions))
                    factor_name = self._infer_factor_name(levels, prereg_iv, prereg_text, i)
                    factors.append(IdentifiedFactor(
                        name=factor_name,
                        levels=levels,
                        is_between_subjects=True,
                        source='Factorial Structure',
                        confidence=0.8,
                    ))
                return factors

        # Simple single-factor design
        return [IdentifiedFactor(
            name='Treatment',
            levels=condition_names,
            is_between_subjects=True,
            source='Conditions List',
            confidence=0.7,
        )]

    def _infer_factor_name(
        self,
        levels: List[str],
        prereg_iv: str,
        prereg_text: str,
        index: int,
    ) -> str:
        """Try to infer a meaningful name for a factor from its levels."""
        # Common level patterns and their factor names
        level_hints = {
            frozenset(['high', 'low']): 'Intensity',
            frozenset(['present', 'absent']): 'Presence',
            frozenset(['yes', 'no']): 'Treatment',
            frozenset(['control', 'treatment']): 'Treatment',
            frozenset(['positive', 'negative']): 'Valence',
            frozenset(['ai', 'human']): 'Source',
        }

        level_set = frozenset(l.lower() for l in levels)
        for pattern, name in level_hints.items():
            if level_set == pattern:
                return name

        # Try to find factor name in preregistration
        combined_text = f"{prereg_iv}\n{prereg_text}"
        for level in levels:
            pattern = rf'(\w+)\s*[:(]\s*[^)]*{re.escape(level)}'
            match = re.search(pattern, combined_text, re.IGNORECASE)
            if match:
                potential_name = match.group(1)
                if potential_name.lower() not in ['condition', 'treatment', 'group']:
                    return potential_name.title()

        return f'Factor_{index + 1}'

    def _identify_scales(
        self,
        questions_map: Dict[str, Dict],
        prereg_outcomes: str,
        prereg_text: str,
    ) -> List[IdentifiedScale]:
        """
        Identify multi-item scales from question patterns.

        Uses multiple strategies:
        1. Matrix questions (explicit scales)
        2. Numbered item patterns (Scale_1, Scale_2, etc.)
        3. Matching with preregistration outcomes
        """
        scales = []
        scale_patterns: Dict[str, List[Dict]] = {}

        for q_id, q_info in questions_map.items():
            # Matrix questions are explicit scales
            if q_info.get('is_matrix'):
                sub_qs = q_info.get('sub_questions', [])
                scale_name = q_info.get('export_tag', q_id)

                scales.append(IdentifiedScale(
                    name=self._clean_scale_name(scale_name),
                    variable_name=scale_name,
                    items=[sq.get('id', '') for sq in sub_qs] if sub_qs else [q_id],
                    num_items=len(sub_qs) if sub_qs else 1,
                    scale_points=q_info.get('scale_points', 7) or 7,
                    role=VariableRole.OTHER,
                    confidence=0.85,
                ))
                continue

            # Check for numbered patterns
            export_tag = q_info.get('export_tag', q_id)
            match = re.match(r'^(.+?)[-_]?(\d+)$', export_tag)
            if match:
                base_name = match.group(1)
                item_num = int(match.group(2))

                if base_name not in scale_patterns:
                    scale_patterns[base_name] = []
                scale_patterns[base_name].append({
                    'item_num': item_num,
                    'question_id': q_id,
                    'export_tag': export_tag,
                    'scale_points': q_info.get('scale_points'),
                })

        # Consolidate numbered patterns into scales
        for base_name, items in scale_patterns.items():
            if len(items) >= 2:
                items.sort(key=lambda x: x['item_num'])
                scale_points = items[0].get('scale_points') or 7

                scales.append(IdentifiedScale(
                    name=self._clean_scale_name(base_name),
                    variable_name=base_name,
                    items=[item['export_tag'] for item in items],
                    num_items=len(items),
                    scale_points=scale_points,
                    role=VariableRole.OTHER,
                    confidence=0.7,
                ))

        # Try to match scales with preregistration outcomes
        prereg_outcomes_lower = prereg_outcomes.lower()
        for scale in scales:
            name_lower = scale.name.lower()
            if name_lower in prereg_outcomes_lower or any(
                word in prereg_outcomes_lower
                for word in name_lower.split()
                if len(word) > 3
            ):
                scale.role = VariableRole.PRIMARY_OUTCOME
                scale.confidence = min(1.0, scale.confidence + 0.15)

        return scales

    def _clean_scale_name(self, name: str) -> str:
        """Clean a scale name for display."""
        # Remove common suffixes
        cleaned = re.sub(r'[-_]\d+$', '', name)
        # Convert underscores to spaces
        cleaned = cleaned.replace('_', ' ')
        # Title case
        return cleaned.strip().title()

    def _classify_variables(
        self,
        questions_map: Dict[str, Dict],
        scales: List[IdentifiedScale],
        conditions: List[IdentifiedCondition],
        prereg_outcomes: str,
        prereg_iv: str,
        prereg_text: str,
    ) -> List[IdentifiedVariable]:
        """
        Classify all survey variables by their likely role.

        This creates a comprehensive list for user review.
        """
        variables = []
        scale_items = set()

        # Mark items that are part of scales
        for scale in scales:
            for item in scale.items:
                scale_items.add(item)

        for q_id, q_info in questions_map.items():
            export_tag = q_info.get('export_tag', q_id)
            question_text = q_info.get('question_text', '')

            # Determine role
            role = VariableRole.OTHER
            confidence = 0.5
            reasons = []

            # Check if part of a scale
            is_scale_item = q_id in scale_items or export_tag in scale_items
            parent_scale = None
            if is_scale_item:
                for scale in scales:
                    if q_id in scale.items or export_tag in scale.items:
                        parent_scale = scale.name
                        role = scale.role if scale.role != VariableRole.OTHER else VariableRole.PRIMARY_OUTCOME
                        confidence = scale.confidence
                        reasons.append(f"Part of scale: {scale.name}")
                        break

            # Check for demographic indicators
            demo_keywords = ['age', 'gender', 'sex', 'income', 'education', 'race', 'ethnicity', 'occupation']
            if any(kw in question_text.lower() for kw in demo_keywords):
                role = VariableRole.DEMOGRAPHICS
                confidence = 0.85
                reasons.append("Contains demographic keywords")

            # Check for attention check indicators
            attention_keywords = ['attention', 'check', 'please select', 'instruct', 'carefully read']
            if any(kw in question_text.lower() for kw in attention_keywords):
                role = VariableRole.ATTENTION_CHECK
                confidence = 0.9
                reasons.append("Contains attention check keywords")

            # Check for open-ended (text entry)
            if q_info.get('type') == 'TE':
                role = VariableRole.OPEN_ENDED
                confidence = 0.95
                reasons.append("Text entry question type")

            # Check preregistration for outcome mentions
            if not is_scale_item and export_tag:
                if export_tag.lower() in prereg_outcomes.lower():
                    role = VariableRole.PRIMARY_OUTCOME
                    confidence = 0.85
                    reasons.append("Mentioned in preregistration outcomes")

            variables.append(IdentifiedVariable(
                variable_id=export_tag,
                display_name=self._create_display_name(export_tag, question_text),
                role=role,
                confidence=confidence,
                source='QSF',
                question_text=question_text[:200],
                scale_points=q_info.get('scale_points'),
                choices=q_info.get('choices', []),
                is_part_of_scale=is_scale_item,
                parent_scale=parent_scale,
                reasons=reasons,
            ))

        # Sort by confidence and role importance
        role_order = {
            VariableRole.PRIMARY_OUTCOME: 0,
            VariableRole.CONDITION: 1,
            VariableRole.INDEPENDENT_VARIABLE: 2,
            VariableRole.SECONDARY_OUTCOME: 3,
            VariableRole.MEDIATOR: 4,
            VariableRole.MODERATOR: 5,
            VariableRole.MANIPULATION_CHECK: 6,
            VariableRole.ATTENTION_CHECK: 7,
            VariableRole.COVARIATE: 8,
            VariableRole.DEMOGRAPHICS: 9,
            VariableRole.OPEN_ENDED: 10,
            VariableRole.FILLER: 11,
            VariableRole.OTHER: 12,
        }

        variables.sort(key=lambda v: (role_order.get(v.role, 99), -v.confidence))
        return variables

    def _create_display_name(self, variable_id: str, question_text: str) -> str:
        """Create a human-readable display name."""
        # Use variable ID as base, clean it up
        name = variable_id.replace('_', ' ').replace('-', ' ')
        name = re.sub(r'\d+$', '', name).strip()

        # If very short, add question text snippet
        if len(name) < 5 and question_text:
            name = f"{name}: {question_text[:40]}..."

        return name.title()

    def _find_open_ended(self, questions_map: Dict[str, Dict]) -> List[str]:
        """Find open-ended text entry questions."""
        open_ended = []
        for q_id, q_info in questions_map.items():
            if q_info.get('type') == 'TE':
                export_tag = q_info.get('export_tag', q_id)
                open_ended.append(export_tag)
        return open_ended

    def _find_attention_checks(self, questions_map: Dict[str, Dict]) -> List[str]:
        """Find attention check questions."""
        attention_checks = []
        keywords = ['attention', 'check', 'please select', 'instruct', 'carefully']

        for q_id, q_info in questions_map.items():
            text = q_info.get('question_text', '').lower()
            if any(kw in text for kw in keywords):
                attention_checks.append(q_info.get('export_tag', q_id))

        return attention_checks

    def _find_manipulation_checks(
        self,
        questions_map: Dict[str, Dict],
        conditions: List[IdentifiedCondition],
    ) -> List[str]:
        """Find manipulation check questions."""
        manip_checks = []
        keywords = ['manipulation', 'perceived', 'felt', 'seemed', 'appeared']

        # Also look for questions that reference condition names
        condition_names = [c.name.lower() for c in conditions]

        for q_id, q_info in questions_map.items():
            text = q_info.get('question_text', '').lower()
            if any(kw in text for kw in keywords):
                manip_checks.append(q_info.get('export_tag', q_id))
            elif any(cn in text for cn in condition_names if cn):
                manip_checks.append(q_info.get('export_tag', q_id))

        return manip_checks


def analyze_qsf_design(
    qsf_content: bytes,
    prereg_outcomes: str = "",
    prereg_iv: str = "",
    prereg_text: str = "",
    prereg_pdf_text: str = "",
) -> DesignAnalysisResult:
    """
    Convenience function to analyze QSF design.

    Args:
        qsf_content: Raw QSF file bytes
        prereg_outcomes: Primary outcomes from preregistration
        prereg_iv: Independent variables from preregistration
        prereg_text: Additional preregistration text
        prereg_pdf_text: Extracted text from preregistration PDF

    Returns:
        DesignAnalysisResult with all identified design elements
    """
    try:
        qsf_data = json.loads(qsf_content.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        # Return empty result on parse failure
        return DesignAnalysisResult(
            conditions=[],
            factors=[],
            scales=[],
            variables=[],
            randomization=RandomizationInfo(
                level=RandomizationLevel.NOT_RANDOMIZED,
                randomizers=[],
            ),
            open_ended_questions=[],
            attention_checks=[],
            manipulation_checks=[],
            warnings=[f"Failed to parse QSF: {str(e)}"],
            suggestions=[],
        )

    identifier = EnhancedConditionIdentifier()
    return identifier.analyze(
        qsf_data,
        prereg_outcomes=prereg_outcomes,
        prereg_iv=prereg_iv,
        prereg_text=prereg_text,
        prereg_pdf_text=prereg_pdf_text,
    )


# Export
__all__ = [
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
]
