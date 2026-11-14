"""QSF check detection: the three defects found while verifying v1.3.1.4 (v1.3.1.5).

v1.3.1.4 made comprehension-check detection conservative and wired up the
manipulation-check detector that had until then been dead code. Verifying it
against the repository's 303 real QSF files turned up three defects in the new
behaviour:

1. The manipulation-check detector fired on Qualtrics elements that collect no
   response — descriptive text, graphic blocks and timing elements. An
   instruction page reading "Please answer the following questions based on the
   scenario you read." matched a text phrase, and an image block named
   ``ManipLuxBrandPen`` matched a name pattern. 46 of 277 detections across the
   corpus were on such elements, and each one reached the student's Design page
   as a manipulation check to confirm, and the engine as a data column.

2. The detector returned one entry per QUESTION rather than per variable. Several
   QIDs routinely share an export tag (the same item shown once per condition),
   so a single check appeared up to ten times: ten rows in the editor and ten
   specs for the engine, which skipped nine of them with a warning each. 74
   duplicate entries across 17 files.

3. The comprehension-check gate read the question's own id, export tag and text
   but not its BLOCK name, which is not yet known where the gate runs. Items in
   blocks named "Comprehension 1" or "ExplanationA_3_checkquestion1_trial2",
   whose own tags are bare ids such as ``C1b`` or ``CQ1_trial2``, were therefore
   not detected at all. That lost the check and, because scale and open-ended
   detection skip comprehension checks, fed the item to DV detection instead: the
   simulated dataset gained a numeric DV or an open-ended answer for a
   comprehension question. 215 items across 46 files.
"""

import json

from utils.condition_identifier import analyze_qsf_design
from utils.qsf_preview import QSFPreviewParser


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sq(qid, tag, text, qtype="MC", selector="SAVR", choices=None, validation=None):
    payload = {
        "QuestionID": qid,
        "DataExportTag": tag,
        "QuestionText": text,
        "QuestionType": qtype,
        "Selector": selector,
    }
    if choices:
        payload["Choices"] = {str(i + 1): {"Display": c} for i, c in enumerate(choices)}
    if validation:
        payload["Validation"] = validation
    return {"Element": "SQ", "PrimaryAttribute": qid, "Payload": payload}


def _equal_to(right_operand):
    """A CustomValidation block stating the correct answer, as Qualtrics writes it."""
    return {"Settings": {"CustomValidation": {"Logic": {"0": {"0": {
        "LogicType": "Question",
        "Operator": "EqualTo",
        "RightOperand": str(right_operand),
    }}}}}}


def _qsf(blocks, questions):
    return json.dumps({
        "SurveyEntry": {"SurveyID": "SV_1", "SurveyName": "Detection"},
        "SurveyElements": [{"Element": "BL", "Payload": blocks}] + questions,
    }).encode()


def _parse(raw):
    return QSFPreviewParser().parse(raw)


# ---------------------------------------------------------------------------
# 1. unanswerable elements are not manipulation checks
# ---------------------------------------------------------------------------

def test_an_instruction_page_is_not_a_manipulation_check():
    """A descriptive block carries text but collects no answer."""
    raw = _qsf(
        [{"ID": "BL_1", "Description": "Scenario", "BlockElements": [
            {"Type": "Question", "QuestionID": "QID1"},
            {"Type": "Question", "QuestionID": "QID2"},
        ]}],
        [
            _sq("QID1", "InstructCCControl",
                "Please answer the following questions based on the scenario you read.",
                qtype="DB", selector="TB"),
            _sq("QID2", "MC_Frustration",
                "What was the main reason for the customer's frustration?",
                choices=["A long wait", "A wrong order"]),
        ],
    )
    detected = _parse(raw).manipulation_checks
    assert "MC_Frustration" in detected
    assert "InstructCCControl" not in detected, detected


def test_a_graphic_block_is_not_a_manipulation_check():
    """The name matched ``manip\\w*``; the element shows an image and takes no answer."""
    raw = _qsf(
        [{"ID": "BL_1", "Description": "Stimulus", "BlockElements": [
            {"Type": "Question", "QuestionID": "QID1"},
        ]}],
        [_sq("QID1", "ManipLuxBrandPen", "This is your last set of questions.",
             qtype="DB", selector="GRB")],
    )
    assert _parse(raw).manipulation_checks == []


def test_a_timing_element_is_not_a_manipulation_check():
    raw = _qsf(
        [{"ID": "BL_1", "Description": "Stimulus", "BlockElements": [
            {"Type": "Question", "QuestionID": "QID1"},
        ]}],
        [_sq("QID1", "Timing_ManipCheck_Emo", "", qtype="Timing", selector="PageTimer")],
    )
    assert _parse(raw).manipulation_checks == []


def test_an_attention_check_named_as_one_is_not_a_manipulation_check():
    """The old guard read only the question text, so the name could still slip past."""
    raw = _qsf(
        [{"ID": "BL_1", "Description": "Quality", "BlockElements": [
            {"Type": "Question", "QuestionID": "QID1"},
        ]}],
        [_sq("QID1", "Attention_check_2", "What was the colour of the car?",
             choices=["Red", "Blue"])],
    )
    assert _parse(raw).manipulation_checks == []


def test_the_condition_identifier_applies_the_same_two_rules():
    """app.py falls back to this detector when the preview finds nothing."""
    raw = _qsf(
        [{"ID": "BL_1", "Description": "Scenario", "BlockElements": [
            {"Type": "Question", "QuestionID": "QID1"},
            {"Type": "Question", "QuestionID": "QID2"},
            {"Type": "Question", "QuestionID": "QID3"},
        ]}],
        [
            _sq("QID1", "Instructions",
                "Please answer the following questions based on the scenario you read.",
                qtype="DB", selector="TB"),
            _sq("QID2", "Attention_check", "What was the main colour shown?",
                choices=["Red", "Blue"]),
            _sq("QID3", "MC_Reason", "What was the main reason for the delay?",
                choices=["Weather", "Staffing"]),
        ],
    )
    found = analyze_qsf_design(raw).manipulation_checks
    assert "MC_Reason" in found
    assert "Instructions" not in found
    assert "Attention_check" not in found


# ---------------------------------------------------------------------------
# 2. one entry per variable
# ---------------------------------------------------------------------------

def test_one_entry_per_variable_when_questions_share_an_export_tag():
    """The same item shown once per condition is one variable, not four."""
    questions = [
        _sq(f"QID{i}", "ManipulationCheck_01",
            "Did you notice anything unusual about the offer?",
            choices=["Yes", "No"])
        for i in range(1, 5)
    ]
    raw = _qsf(
        [{"ID": "BL_1", "Description": "Checks", "BlockElements": [
            {"Type": "Question", "QuestionID": f"QID{i}"} for i in range(1, 5)
        ]}],
        questions,
    )
    detected = _parse(raw).manipulation_checks
    assert detected == ["ManipulationCheck_01"], detected


# ---------------------------------------------------------------------------
# 3. the block name is evidence
# ---------------------------------------------------------------------------

def _block_named_qsf(block_name, tag="C1b"):
    return _qsf(
        [{"ID": "BL_1", "Description": block_name, "BlockElements": [
            {"Type": "Question", "QuestionID": "QID1"},
        ]}],
        [_sq("QID1", tag,
             "If Player A chooses to take $3 and Player B blocks, what does Player C earn?",
             choices=["$0", "$3", "$6"], validation=_equal_to("2"))],
    )


def test_a_check_in_a_block_named_comprehension_is_detected():
    """The tag "C1b" and the wording carry no cue; the block name does."""
    result = _parse(_block_named_qsf("Comprehension 1"))
    tags = [c.get("export_tag") for c in (result.comprehension_checks or [])]
    assert "C1b" in tags, result.comprehension_checks


def test_a_check_in_a_checkquestion_block_is_detected():
    result = _parse(_block_named_qsf("ExplanationA_3_checkquestion1_trial2", tag="CQ1_trial2"))
    tags = [c.get("export_tag") for c in (result.comprehension_checks or [])]
    assert "CQ1_trial2" in tags, result.comprehension_checks


def test_the_block_name_resolves_the_answer_text_not_the_recode_id():
    """Promotion must carry the same answer resolution as the direct route."""
    result = _parse(_block_named_qsf("Comprehension 2"))
    check = (result.comprehension_checks or [])[0]
    assert check.get("expected_answer") == "$3", check


def test_an_ordinary_outcome_in_an_ordinary_block_is_still_not_a_check():
    """The block name alone is not enough: the validation must state an answer."""
    raw = _qsf(
        [{"ID": "BL_1", "Description": "Comprehension 1", "BlockElements": [
            {"Type": "Question", "QuestionID": "QID1"},
        ]}],
        [_sq("QID1", "Attitude", "How much do you agree with the statement?",
             choices=["1", "2", "3", "4", "5"])],
    )
    assert not (_parse(raw).comprehension_checks or [])


def test_a_sum_to_100_validation_is_still_not_a_check():
    """The v1.3.1.4 rule this must not undo: a constraint is not a correct answer."""
    validation = {"Settings": {"CustomValidation": {"Logic": {"0": {"0": {
        "LogicType": "Question", "Operator": "NotEqualTo", "RightOperand": "100",
    }}}}}}
    raw = _qsf(
        [{"ID": "BL_1", "Description": "Comprehension 1", "BlockElements": [
            {"Type": "Question", "QuestionID": "QID1"},
        ]}],
        [_sq("QID1", "Budget", "Split 100 points between the two options.",
             qtype="CS", choices=["Option A", "Option B"], validation=validation)],
    )
    assert not (_parse(raw).comprehension_checks or [])


def test_a_discarded_block_is_not_promoted():
    """A question a researcher moved to the trash is not simulated at all."""
    raw = _qsf(
        [{"ID": "BL_1", "Description": "Trash / Unused comprehension questions",
          "BlockElements": [{"Type": "Question", "QuestionID": "QID1"}]}],
        [_sq("QID1", "C9", "What does Player C earn?",
             choices=["$0", "$3"], validation=_equal_to("1"))],
    )
    assert not (_parse(raw).comprehension_checks or [])


def test_a_promoted_check_is_not_also_simulated_as_an_outcome():
    """The reason the block name must be read BEFORE scale and open-ended detection."""
    raw = _qsf(
        [{"ID": "BL_1", "Description": "LT_checkquestion1", "BlockElements": [
            {"Type": "Question", "QuestionID": "QID1"},
        ]}],
        [_sq("QID1", "CQ1_trial1",
             "How many tokens does the other player receive? (enter a number)",
             qtype="TE", selector="SL", validation=_equal_to("10"))],
    )
    result = _parse(raw)
    assert [c.get("export_tag") for c in (result.comprehension_checks or [])] == ["CQ1_trial1"]
    scale_names = {str(s.get("name", "")) for s in (result.detected_scales or [])}
    assert "CQ1_trial1" not in scale_names, result.detected_scales
    assert "CQ1_trial1" not in (result.open_ended_questions or [])


# ---------------------------------------------------------------------------
# the corpus: the shape of detection over the real files
# ---------------------------------------------------------------------------

def test_no_detection_on_an_empty_or_malformed_questions_map():
    parser = QSFPreviewParser()
    assert parser._detect_manipulation_checks({}) == []
    assert parser._detect_manipulation_checks(None) == []
    parser._promote_block_named_comprehension_checks(None)  # must not raise
    parser._promote_block_named_comprehension_checks({})
