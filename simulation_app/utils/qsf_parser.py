"""
QSF Parser for Behavioral Experiment Simulation Tool
=============================================================
Parses Qualtrics Survey Format (.qsf) files to extract survey structure,
questions, blocks, and embedded data for simulation.

QSF files are JSON-based exports from Qualtrics survey platform.
"""

import json
import re
from typing import Any, Dict, List


def _normalize_survey_elements(qsf_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize SurveyElements into a list of dicts across QSF variants."""
    elements = qsf_data.get('SurveyElements', [])
    if isinstance(elements, dict):
        elements = list(elements.values())
    if not isinstance(elements, list):
        return []
    return [element for element in elements if isinstance(element, dict)]


def _normalize_flow(flow: Any) -> List[Dict[str, Any]]:
    """Normalize flow structures into a list of dicts across QSF variants."""
    if isinstance(flow, dict):
        if 'Flow' in flow:
            flow = flow.get('Flow', [])
        else:
            flow = list(flow.values())
    if not isinstance(flow, list):
        return []
    return [item for item in flow if isinstance(item, dict)]


def parse_qsf_file(file_content: bytes) -> Dict[str, Any]:
    """
    Parse a QSF file and extract its structure.

    Args:
        file_content: Raw bytes content of the QSF file

    Returns:
        Dictionary containing parsed survey structure
    """
    try:
        # Decode and parse JSON
        content_str = file_content.decode('utf-8')
        qsf_data = json.loads(content_str)

        # Extract key components
        survey_entry = qsf_data.get('SurveyEntry', {})
        survey_elements = _normalize_survey_elements(qsf_data)

        parsed = {
            'survey_name': survey_entry.get('SurveyName', 'Unknown Survey'),
            'survey_id': survey_entry.get('SurveyID', 'Unknown'),
            'status': survey_entry.get('SurveyStatus', 'Unknown'),
            'creation_date': survey_entry.get('SurveyCreationDate', 'Unknown'),
            'blocks': [],
            'questions': [],
            'embedded_data': [],
            'randomizers': [],
            'flow': [],
            'raw_elements': survey_elements
        }

        # Process each survey element
        for element in survey_elements:
            element_type = element.get('Element', '')

            if element_type == 'BL':
                # Block element
                parsed['blocks'].extend(_parse_blocks(element))

            elif element_type == 'SQ':
                # Survey question
                parsed['questions'].append(_parse_question(element))

            elif element_type == 'FL':
                # Flow element
                parsed['flow'] = _parse_flow(element)

            elif element_type == 'RS':
                # Randomizer/Survey Options
                if 'Payload' in element:
                    parsed['randomizers'].append(element.get('Payload', {}))

            elif element_type == 'ED':
                # Embedded data
                parsed['embedded_data'].extend(_parse_embedded_data(element))

        return parsed

    except json.JSONDecodeError as e:
        return {
            'error': f'Invalid JSON in QSF file: {str(e)}',
            'survey_name': 'Parse Error',
            'blocks': [],
            'questions': [],
            'embedded_data': [],
            'randomizers': [],
            'flow': []
        }
    except Exception as e:
        return {
            'error': f'Error parsing QSF file: {str(e)}',
            'survey_name': 'Parse Error',
            'blocks': [],
            'questions': [],
            'embedded_data': [],
            'randomizers': [],
            'flow': []
        }


def _parse_blocks(element: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse block elements from QSF.

    Handles both QSF formats:
    - Dict format: {"BL_123": {"Description": "Block 1", ...}}
    - List format: [{"ID": "BL_123", "Description": "Block 1", ...}]
    """
    blocks = []
    payload = element.get('Payload', {})
    if isinstance(payload, dict) and isinstance(payload.get('Blocks'), (list, dict)):
        payload = payload.get('Blocks', payload)

    # Handle list format (newer QSF exports)
    if isinstance(payload, list):
        for block_data in payload:
            if isinstance(block_data, dict):
                block_id = block_data.get('ID', '')
                if not block_id:
                    continue
                block_info = {
                    'id': block_id,
                    'description': block_data.get('Description', 'Unnamed Block'),
                    'type': block_data.get('Type', 'Standard'),
                    'elements': []
                }

                # Extract block elements (question references)
                block_elements = block_data.get('BlockElements', [])
                for elem in block_elements:
                    if elem.get('Type') == 'Question':
                        block_info['elements'].append(elem.get('QuestionID', ''))

                blocks.append(block_info)
    # Handle dict format (older QSF exports)
    elif isinstance(payload, dict):
        for dict_key, block_data in payload.items():
            if isinstance(block_data, dict):
                # Use ID field if available, otherwise use dict key
                block_id = block_data.get('ID', dict_key)
                block_info = {
                    'id': block_id,
                    'description': block_data.get('Description', 'Unnamed Block'),
                    'type': block_data.get('Type', 'Standard'),
                    'elements': []
                }

                # Extract block elements (question references)
                block_elements = block_data.get('BlockElements', [])
                for elem in block_elements:
                    if elem.get('Type') == 'Question':
                        block_info['elements'].append(elem.get('QuestionID', ''))

                blocks.append(block_info)

    return blocks


def _parse_question(element: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a survey question element."""
    payload = element.get('Payload', {})

    question = {
        'id': payload.get('QuestionID', element.get('PrimaryAttribute', '')),
        'data_export_tag': payload.get('DataExportTag', ''),
        'text': _clean_html(payload.get('QuestionText', '')),
        'type': payload.get('QuestionType', 'Unknown'),
        'selector': payload.get('Selector', ''),
        'sub_selector': payload.get('SubSelector', ''),
        'choices': {},
        'choice_order': [],
        'validation': payload.get('Validation', {}),
        'display_logic': payload.get('DisplayLogic', None),
        'is_mandatory': payload.get('Validation', {}).get('Settings', {}).get('ForceResponse', 'OFF') == 'ON'
    }

    # Parse choices
    choices = payload.get('Choices', {})
    choice_order = payload.get('ChoiceOrder', [])

    for choice_id in choice_order:
        choice_id_str = str(choice_id)
        if choice_id_str in choices:
            choice_data = choices[choice_id_str]
            if isinstance(choice_data, dict):
                question['choices'][choice_id_str] = {
                    'text': _clean_html(choice_data.get('Display', '')),
                    'recode': choice_data.get('RecodeValue', choice_id_str)
                }
            else:
                question['choices'][choice_id_str] = {
                    'text': str(choice_data),
                    'recode': choice_id_str
                }

    question['choice_order'] = [str(c) for c in choice_order]

    # Parse answers (for matrix questions)
    answers = payload.get('Answers', {})
    if answers:
        question['answers'] = {}
        for ans_id, ans_data in answers.items():
            if isinstance(ans_data, dict):
                question['answers'][ans_id] = _clean_html(ans_data.get('Display', ''))
            else:
                question['answers'][ans_id] = str(ans_data)

    return question


def _parse_flow(element: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse survey flow elements."""
    payload = element.get('Payload', {})
    flow = _normalize_flow(payload.get('Flow', payload))

    parsed_flow = []
    for item in flow:
        flow_item = {
            'type': item.get('Type', 'Unknown'),
            'id': item.get('ID', item.get('FlowID', '')),
            'description': item.get('Description', '')
        }

        # Handle randomizers
        if item.get('Type') in {'BlockRandomizer', 'Randomizer'}:
            flow_item['randomize_count'] = item.get('RandomizeCount', 1)
            flow_item['sub_flow'] = []
            for sub in _normalize_flow(item.get('Flow', [])):
                flow_item['sub_flow'].append({
                    'id': sub.get('ID', ''),
                    'type': sub.get('Type', '')
                })

        # Handle branches
        elif item.get('Type') == 'Branch':
            flow_item['branch_logic'] = item.get('BranchLogic', {})

        # Handle embedded data
        elif item.get('Type') == 'EmbeddedData':
            flow_item['embedded_data'] = item.get('EmbeddedData', [])

        elif item.get('Type') == 'Group':
            flow_item['sub_flow'] = [
                {'id': sub.get('ID', ''), 'type': sub.get('Type', '')}
                for sub in _normalize_flow(item.get('Flow', []))
            ]

        parsed_flow.append(flow_item)

    return parsed_flow


def _parse_embedded_data(element: Dict[str, Any]) -> List[Dict[str, str]]:
    """Parse embedded data fields."""
    embedded = []
    payload = element.get('Payload', {})

    flow_payload = payload.get('Flow', payload if isinstance(payload, list) else None)
    if flow_payload is not None:
        for item in _normalize_flow(flow_payload):
            ed_list = item.get('EmbeddedData', [])
            if isinstance(ed_list, list):
                for ed in ed_list:
                    if isinstance(ed, dict):
                        embedded.append({
                            'field': ed.get('Field', ''),
                            'type': ed.get('Type', 'Custom'),
                            'value': ed.get('Value', '')
                        })
    elif isinstance(payload, dict):
        for field, value in payload.items():
            embedded.append({
                'field': field,
                'type': 'Custom',
                'value': str(value) if not isinstance(value, dict) else ''
            })

    return embedded


def _clean_html(text: str) -> str:
    """Remove HTML tags and clean up text."""
    if not text:
        return ''

    # Remove HTML tags
    clean = re.sub(r'<[^>]+>', '', str(text))

    # Decode common HTML entities
    clean = clean.replace('&nbsp;', ' ')
    clean = clean.replace('&amp;', '&')
    clean = clean.replace('&lt;', '<')
    clean = clean.replace('&gt;', '>')
    clean = clean.replace('&quot;', '"')
    clean = clean.replace('&#39;', "'")

    # Clean up whitespace
    clean = ' '.join(clean.split())

    return clean.strip()


def extract_survey_structure(parsed_qsf: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a simplified survey structure for simulation purposes.

    Args:
        parsed_qsf: Output from parse_qsf_file()

    Returns:
        Simplified structure with conditions, variables, and scales
    """
    structure = {
        'survey_name': parsed_qsf.get('survey_name', 'Unknown'),
        'total_blocks': len(parsed_qsf.get('blocks', [])),
        'total_questions': len(parsed_qsf.get('questions', [])),
        'conditions': [],
        'scale_questions': [],
        'single_choice_questions': [],
        'text_questions': [],
        'slider_questions': [],
        'matrix_questions': [],
        'embedded_fields': []
    }

    # Identify conditions from randomizers/blocks
    blocks = parsed_qsf.get('blocks', [])
    flow = parsed_qsf.get('flow', [])

    # Look for block randomizers in flow
    for flow_item in flow:
        if flow_item.get('type') == 'BlockRandomizer':
            sub_flow = flow_item.get('sub_flow', [])
            for sub in sub_flow:
                block_id = sub.get('id', '')
                # Find matching block
                for block in blocks:
                    if block.get('id') == block_id:
                        structure['conditions'].append(block.get('description', block_id))

    # If no randomizer found, check for blocks that look like conditions
    if not structure['conditions']:
        for block in blocks:
            desc = block.get('description', '').lower()
            # Common condition naming patterns
            if any(keyword in desc for keyword in ['condition', 'treatment', 'group', 'arm']):
                structure['conditions'].append(block.get('description', ''))

    # Categorize questions
    questions = parsed_qsf.get('questions', [])

    for q in questions:
        q_type = q.get('type', '')
        selector = q.get('selector', '')

        question_summary = {
            'id': q.get('id', ''),
            'tag': q.get('data_export_tag', ''),
            'text': q.get('text', '')[:100],  # Truncate long text
            'num_choices': len(q.get('choices', {})),
            'type': q_type,
            'selector': selector
        }

        if q_type == 'MC':
            # Multiple choice
            if len(q.get('choices', {})) > 6:
                structure['scale_questions'].append(question_summary)
            else:
                structure['single_choice_questions'].append(question_summary)

        elif q_type == 'Matrix':
            structure['matrix_questions'].append(question_summary)
            # Matrix questions often represent scales
            structure['scale_questions'].append(question_summary)

        elif q_type == 'TE':
            structure['text_questions'].append(question_summary)

        elif q_type == 'Slider':
            structure['slider_questions'].append(question_summary)

        elif q_type in ['DB', 'Descriptive']:
            # Descriptive text blocks - skip
            pass

    # Extract embedded data fields
    for ed in parsed_qsf.get('embedded_data', []):
        if ed.get('field'):
            structure['embedded_fields'].append(ed.get('field'))

    return structure


def generate_qsf_summary(parsed_qsf: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of the QSF file.

    Args:
        parsed_qsf: Output from parse_qsf_file()

    Returns:
        Formatted string summary
    """
    structure = extract_survey_structure(parsed_qsf)

    lines = [
        "=" * 60,
        "QSF FILE SUMMARY",
        "=" * 60,
        "",
        f"Survey Name: {structure['survey_name']}",
        f"Total Blocks: {structure['total_blocks']}",
        f"Total Questions: {structure['total_questions']}",
        "",
    ]

    if structure['conditions']:
        lines.append("DETECTED CONDITIONS:")
        for i, cond in enumerate(structure['conditions'], 1):
            lines.append(f"  {i}. {cond}")
        lines.append("")

    if structure['scale_questions']:
        lines.append(f"SCALE QUESTIONS ({len(structure['scale_questions'])}):")
        for q in structure['scale_questions'][:5]:  # Show first 5
            lines.append(f"  - {q['tag']}: {q['text'][:50]}...")
        if len(structure['scale_questions']) > 5:
            lines.append(f"  ... and {len(structure['scale_questions']) - 5} more")
        lines.append("")

    if structure['single_choice_questions']:
        lines.append(f"SINGLE-CHOICE QUESTIONS ({len(structure['single_choice_questions'])}):")
        for q in structure['single_choice_questions'][:5]:
            lines.append(f"  - {q['tag']}: {q['text'][:50]}...")
        if len(structure['single_choice_questions']) > 5:
            lines.append(f"  ... and {len(structure['single_choice_questions']) - 5} more")
        lines.append("")

    if structure['text_questions']:
        lines.append(f"TEXT ENTRY QUESTIONS ({len(structure['text_questions'])}):")
        for q in structure['text_questions'][:3]:
            lines.append(f"  - {q['tag']}: {q['text'][:50]}...")
        lines.append("")

    if structure['embedded_fields']:
        lines.append(f"EMBEDDED DATA FIELDS ({len(structure['embedded_fields'])}):")
        for field in structure['embedded_fields'][:10]:
            lines.append(f"  - {field}")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)
