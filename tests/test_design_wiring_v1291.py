"""QSF -> Design page wiring (v1.2.9.1).

Covers four defects, each through the real Streamlit app (``AppTest``, nothing seeded) where the UI is involved:

1. The Design page started from one generic ``Main_DV`` because ``_preview_to_engine_inputs`` had no caller; it now
   starts from the survey's own DVs, keeps their types and ranges, and lets the user edit them.
2. Qualtrics nests the declared range of a number box in ``Validation.Settings.ValidNumber``; the parser read a flat
   ``Settings.Min`` that real exports never contain, so no range ever reached the engine.
3. A different file saved under the SAME name was ignored by the upload step (names only were compared).
4. A second upload kept the first survey's attention checks, mediators, identifiers and generated dataset.

The surveys are tiny synthetic QSFs built below, so the tests do not depend on the collected example files.
"""
import json
import os
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

_APP_DIR = Path(__file__).resolve().parent.parent / "simulation_app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from utils.qsf_preview import QSFPreviewParser  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------------
# Synthetic QSF builder (only the parts of a Qualtrics export the parser reads)
# ---------------------------------------------------------------------------------------------------------------------
def _sq(qid, tag, text, qtype, selector, **extra):
    payload = {"QuestionText": text, "DataExportTag": tag, "QuestionType": qtype, "Selector": selector,
               "QuestionDescription": text, "QuestionID": qid, "Language": [],
               "Validation": {"Settings": {"ForceResponse": "OFF", "Type": "None"}}}
    payload.update(extra)
    return {"SurveyID": "SV_test", "Element": "SQ", "PrimaryAttribute": qid, "SecondaryAttribute": text,
            "TertiaryAttribute": None, "Payload": payload}


def likert_matrix(qid, tag, items, points=7):
    return _sq(qid, tag, "How much do you agree with each statement?", "Matrix", "Likert", SubSelector="SingleAnswer",
               Choices={str(i + 1): {"Display": t} for i, t in enumerate(items)},
               ChoiceOrder=list(range(1, len(items) + 1)),
               Answers={str(i + 1): {"Display": str(i + 1)} for i in range(points)},
               AnswerOrder=list(range(1, points + 1)))


def slider(qid, tag, text, lo, hi):
    return _sq(qid, tag, text, "Slider", "HSLIDER", Choices={"1": {"Display": text}}, ChoiceOrder=["1"],
               Configuration={"CSSliderMin": lo, "CSSliderMax": hi, "GridLines": 10, "SnapToGrid": False})


def number_box(qid, tag, text, number_range=None, content_type="ValidNumber"):
    settings = {"ForceResponse": "OFF", "Type": "ContentType", "ContentType": content_type}
    if number_range is not None:
        settings["ValidNumber"] = dict(number_range)
    return _sq(qid, tag, text, "TE", "SL", Validation={"Settings": settings})


def rank_order(qid, tag, items):
    return _sq(qid, tag, "Rank these options from most to least preferred", "RO", "DND",
               Choices={str(i + 1): {"Display": t} for i, t in enumerate(items)},
               ChoiceOrder=list(range(1, len(items) + 1)))


def essay(qid, tag, text):
    return _sq(qid, tag, text, "TE", "ESTB")


def attention_check(qid, tag):
    return _sq(qid, tag, "Attention check: please select Strongly agree for this item.", "MC", "SAVR",
               Choices={str(i): {"Display": d} for i, d in enumerate(["Strongly disagree", "Disagree", "Agree", "Strongly agree"], 1)},
               ChoiceOrder=[1, 2, 3, 4])


def make_qsf(name, questions, conditions=("Control", "Treatment")):
    """Serialise a one-randomizer survey: a stimulus block per condition, then one block with ``questions``."""
    elements, blocks, stimulus_ids = [], {}, []
    for i, condition in enumerate(conditions):
        qid = f"QID9{i}"
        elements.append(_sq(qid, f"stim{i}", f"You see the {condition} message.", "DB", "TB"))
        block_id = f"BL_{i + 1:03d}"
        blocks[str(i + 1)] = {"Type": "Standard", "Description": condition, "ID": block_id,
                              "BlockElements": [{"Type": "Question", "QuestionID": qid}]}
        stimulus_ids.append(block_id)
    measures_id = f"BL_{len(conditions) + 1:03d}"
    blocks[str(len(conditions) + 1)] = {
        "Type": "Standard", "Description": "Measures", "ID": measures_id,
        "BlockElements": [{"Type": "Question", "QuestionID": q["Payload"]["QuestionID"]} for q in questions]}
    elements.extend(questions)
    flow = [{"Type": "BlockRandomizer", "FlowID": "FL_2", "SubSet": "1", "EvenPresentation": True,
             "Flow": [{"Type": "Block", "ID": b, "FlowID": f"FL_1{i}"} for i, b in enumerate(stimulus_ids)]},
            {"Type": "Standard", "ID": measures_id, "FlowID": "FL_9"}]
    elements.append({"SurveyID": "SV_test", "Element": "BL", "PrimaryAttribute": "Survey Blocks", "SecondaryAttribute": None,
                     "TertiaryAttribute": None, "Payload": blocks})
    elements.append({"SurveyID": "SV_test", "Element": "FL", "PrimaryAttribute": "Survey Flow", "SecondaryAttribute": None,
                     "TertiaryAttribute": None, "Payload": {"Type": "Root", "FlowID": "FL_1", "Flow": flow}})
    survey = {"SurveyEntry": {"SurveyID": "SV_test", "SurveyName": name, "SurveyDescription": "", "SurveyLanguage": "EN"},
              "SurveyElements": elements}
    return json.dumps(survey).encode("utf-8")


def survey_a():
    """Matrix scale, 0-150 slider, year-of-birth box (1900-2003), rank order, an essay and an attention check."""
    return make_qsf("Survey A", [
        likert_matrix("QID1", "Trust", ["I trust it", "It is reliable", "It is honest"]),
        slider("QID2", "Budget", "How much of the budget would you allocate?", 0, 150),
        number_box("QID3", "BirthYear", "In what year were you born?", {"Min": "1900", "Max": "2003", "NumDecimals": "0"}),
        rank_order("QID4", "Priorities", ["Price", "Quality", "Speed"]),
        essay("QID5", "Why", "Please explain your choice"),
        attention_check("QID6", "AttnCheck"),
    ])


A_DVS = ["Trust", "Budget", "Priorities", "BirthYear"]


def survey_b():
    """A different survey: one satisfaction matrix and a willingness-to-pay slider, no attention check."""
    return make_qsf("Survey B", [
        likert_matrix("QID1", "Satisfaction", ["I am satisfied", "I would return"], points=5),
        slider("QID2", "WTP", "How much would you pay?", 0, 100),
        essay("QID3", "Comments", "Any other comments about the product?"),
    ], conditions=("Low price", "High price"))


B_DVS = ["Satisfaction", "WTP"]


def survey_no_dvs():
    """Nothing the parser can call a DV: one essay question."""
    return make_qsf("Survey without measures", [essay("QID1", "Thoughts", "Describe your thoughts about the product")])


def survey_extremes():
    """Values that exceeded the Design page's old number boxes: a 60-item battery, a 0-1000 slider, years from 1900."""
    return make_qsf("Extreme ranges", [
        likert_matrix("QID1", "Battery", [f"Statement {i}" for i in range(1, 61)]),
        slider("QID2", "Points", "How many points would you give?", 0, 1000),
        number_box("QID3", "BirthYear", "In what year were you born?", {"Min": "1900", "Max": "2003"}),
        number_box("QID4", "Income", "What is your yearly income in dollars?", {"Min": "0", "Max": "1000000"}),
    ])


# ---------------------------------------------------------------------------------------------------------------------
# Parser: the declared number range
# ---------------------------------------------------------------------------------------------------------------------
def _parse(raw):
    return QSFPreviewParser().parse(raw)


def _box(preview, tag):
    return next(q for q in preview.open_ended_details if q["variable_name"] == tag)


def test_the_declared_number_range_is_read_from_the_nested_valid_number_settings():
    preview = _parse(make_qsf("Range", [number_box("QID1", "BirthYear", "In what year were you born?",
                                                   {"Min": "1900", "Max": "2003", "NumDecimals": "0"})]))
    assert preview.success
    box = _box(preview, "BirthYear")
    assert (box["content_type"], box["number_min"], box["number_max"]) == ("ValidNumber", 1900.0, 2003.0)
    text_entry = next(q for q in preview.text_entry_questions if q["export_tag"] == "BirthYear")
    assert (text_entry["number_min"], text_entry["number_max"]) == (1900.0, 2003.0)


@pytest.mark.parametrize("declared, expected", [
    ({"Min": "", "Max": "50", "NumDecimals": "0"}, (None, 50.0)),          # one-sided: the blank bound stays open
    ({"Min": "5", "Max": ""}, (5.0, None)),
    ({"Min": "abc", "Max": "10"}, (None, 10.0)),                           # junk keeps the previous (absent) value
    ({"Min": "nan", "Max": "inf"}, (None, None)),                          # never a non-finite bound
    ({"Min": " 18 ", "Max": 99}, (18.0, 99.0)),                            # padded text and real numbers
    ({"NumDecimals": "2"}, (None, None)),                                  # no range declared at all
])
def test_blank_junk_and_one_sided_ranges_never_produce_a_bad_bound(declared, expected):
    preview = _parse(make_qsf("Range", [number_box("QID1", "Amount", "How many tickets do you take?", declared)]))
    box = _box(preview, "Amount")
    assert (box["number_min"], box["number_max"]) == expected


def test_a_range_is_only_read_when_the_box_validates_numbers():
    preview = _parse(make_qsf("Range", [number_box("QID1", "Zip", "What is your ZIP code?",
                                                   {"Min": "1", "Max": "5"}, content_type="ValidZip")]))
    box = _box(preview, "Zip")
    assert box["content_type"] == "ValidZip" and (box["number_min"], box["number_max"]) == (None, None)


def test_numeric_text_box_answers_stay_inside_the_declared_range():
    import numpy as np
    from utils.enhanced_simulation_engine import clean_question_text, draw_numeric_answer, infer_numeric_answer_spec

    preview = _parse(make_qsf("Range", [number_box("QID1", "BirthYear", "In what year were you born?",
                                                   {"Min": "1930", "Max": "2003"}),
                                        number_box("QID2", "Household", "How many people live in your household?",
                                                   {"Min": "1", "Max": "10"})]))
    for tag, lo, hi in (("BirthYear", 1930, 2003), ("Household", 1, 10)):
        box = _box(preview, tag)
        spec = infer_numeric_answer_spec(clean_question_text(box["question_text"]), tag, dict(box))
        draws = [float(draw_numeric_answer(spec, np.random.RandomState(seed))) for seed in range(400)]
        assert lo <= min(draws) and max(draws) <= hi, (tag, min(draws), max(draws))


@pytest.mark.parametrize("text, declared, window", [
    ("If the lottery fails and you lose, how much do you lose? (0-100)", {"number_min": None, "number_max": 2.0}, (0, 2)),
    ("How many people live in your household?", {"number_min": None, "number_max": 10.0}, (0, 10)),
    ("What is your year of birth?", {"number_min": None, "number_max": 2003.0}, (0, 2003)),
    ("What is your age?", {"number_min": 18.0, "number_max": None}, (18, 120)),     # Min-only keeps its old behaviour
])
def test_a_one_sided_declared_range_still_binds(text, declared, window):
    import numpy as np
    from utils.enhanced_simulation_engine import draw_numeric_answer, infer_numeric_answer_spec

    box = dict(declared, content_type="ValidNumber")
    spec = infer_numeric_answer_spec(text, "Box", box)
    draws = [float(draw_numeric_answer(spec, np.random.RandomState(seed))) for seed in range(300)]
    assert window[0] <= min(draws) and max(draws) <= window[1], (text, min(draws), max(draws))
    if declared["number_max"] == 2.0:   # the window is used, not just clipped: all of 0, 1 and 2 occur
        assert {0.0, 1.0, 2.0} <= set(draws)


# ---------------------------------------------------------------------------------------------------------------------
# The real app (AppTest): helpers
# ---------------------------------------------------------------------------------------------------------------------
@pytest.fixture()
def app_test(monkeypatch, tmp_path):
    """A fresh AppTest with its data/ folders in a temp dir (nothing seeded)."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    return AppTest.from_file(str(_APP_DIR / "app.py"), default_timeout=300)


def _button(at, key):
    found = [b for b in at.button if b.key == key]
    assert found, f"no button {key!r}; buttons: {[b.key for b in at.button]}"
    return found[0]


def _no_exception(at):
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


def _start(at, sample_size=20):
    """Landing page -> Setup (title and description typed) -> Study Input, exactly as a student does."""
    at.run()
    _button(at, "landing_cta").click()
    at.run()
    at.text_input(key="study_title").input("Design wiring study")
    at.text_area(key="study_description").input("A pilot study of how product messages change trust and purchase intentions.")
    at.run()
    _button(at, "nav_next_0").click()
    at.run()
    at.session_state["sample_size"] = sample_size
    assert at.session_state["active_page"] == 1
    _no_exception(at)


def _upload(at, name, raw, replace=False):
    if replace:
        at.checkbox(key="change_qsf").check()
        at.run()
    at.file_uploader[0].set_value((name, raw, "application/octet-stream"))
    at.run()
    _no_exception(at)


def _open_design(at):
    _button(at, "nav_next_1").click()
    at.run()
    assert at.session_state["active_page"] == 2
    _no_exception(at)


def _dv_rows(at):
    """(name, type, items, min, max) of every DV row the Design page shows."""
    names = sorted((t for t in at.text_input if str(t.key).startswith("dv_name_v")), key=lambda t: int(str(t.key).rsplit("_", 1)[1]))
    rows = []
    for t in names:
        version, index = str(t.key)[len("dv_name_v"):].split("_")
        rows.append((t.value,
                     at.selectbox(key=f"dv_type_v{version}_{index}").value,
                     at.number_input(key=f"dv_items_v{version}_{index}").value,
                     at.number_input(key=f"dv_min_v{version}_{index}").value,
                     at.number_input(key=f"dv_max_v{version}_{index}").value))
    return rows


def _use_offline_engine(at):
    """Pick the offline template engine (no network, no LLM) the way the Generate page's first card does."""
    at.session_state["generation_method"] = "abe_v2"
    at.session_state["allow_template_fallback_once"] = True


def _confirm(at):
    """Tick the DV (and open-ended) confirmations of the current version, as a student would."""
    for prefix in ("dv_confirm_checkbox_v", "oe_confirm_checkbox_v"):
        boxes = [c for c in at.checkbox if str(c.key).startswith(prefix)]
        if boxes:
            boxes[0].check()
            at.run()
    _no_exception(at)


# ---------------------------------------------------------------------------------------------------------------------
# 1. The Design page starts from the survey's own DVs
# ---------------------------------------------------------------------------------------------------------------------
def test_the_design_page_starts_from_the_surveys_own_dvs_not_a_generic_main_dv(app_test):
    at = app_test
    _start(at)
    _upload(at, "survey_a.qsf", survey_a())
    _open_design(at)
    rows = _dv_rows(at)
    assert [r[0] for r in rows] == A_DVS
    by_name = {r[0]: r for r in rows}
    assert "Main_DV" not in by_name
    assert any("4 DV(s) detected" in m.value for m in at.markdown)
    assert any("Back to top" in m.value for m in at.markdown)       # the page-layout rule: the link ends every page
    # types the old page could not show survive (rank_order used to become "matrix", a numeric box "single_item")
    assert by_name["Priorities"][1] == "rank_order" and by_name["Priorities"][2] == 3
    assert by_name["BirthYear"][1] == "numeric_input"
    assert by_name["Trust"][1] == "matrix" and by_name["Trust"][2] == 3
    # ranges the old number boxes could not hold: year of birth from 1900, a slider to 150
    assert (by_name["BirthYear"][3], by_name["BirthYear"][4]) == (1900, 2003)
    assert (by_name["Budget"][3], by_name["Budget"][4]) == (0, 150)


def test_the_dvs_the_page_hands_to_the_engine_match_what_the_parser_detected(app_test):
    at = app_test
    _start(at)
    _upload(at, "survey_a.qsf", survey_a())
    _open_design(at)
    _confirm(at)
    final = {s["variable_name"]: s for s in at.session_state["inferred_design"]["scales"]}
    assert list(final) == A_DVS
    assert final["Priorities"]["type"] == "rank_order" and final["Priorities"]["num_items"] == 3
    assert final["BirthYear"]["type"] == "numeric_input"
    assert (final["BirthYear"]["scale_min"], final["BirthYear"]["scale_max"]) == (1900, 2003)
    assert (final["Budget"]["scale_min"], final["Budget"]["scale_max"]) == (0, 150)
    assert final["Budget"]["scale_points"] == 151  # the count the parser reported, not "max"
    assert all(s["detected_from_qsf"] for s in (final["Trust"], final["BirthYear"]))
    # the Generate page offers the same list (readiness gate and summary)
    _use_offline_engine(at)
    _button(at, "nav_next_2").click()
    at.run()
    _no_exception(at)
    assert at.session_state["active_page"] == 3
    assert [s["variable_name"] for s in at.session_state["confirmed_scales"]] == A_DVS
    assert any("<strong style=\"color:#374151;\">4</strong> DVs" in m.value for m in at.markdown)
    assert not _button(at, "generate_dataset_btn").disabled
    assert any("Back to top" in m.value for m in at.markdown)


def test_a_survey_without_a_detectable_dv_keeps_the_generic_default(app_test):
    at = app_test
    _start(at)
    _upload(at, "no_dvs.qsf", survey_no_dvs())
    _open_design(at)
    assert [(r[0], r[2]) for r in _dv_rows(at)] == [("Main_DV", 5)]


def test_values_outside_the_old_number_boxes_open_without_an_error(app_test):
    at = app_test
    _start(at)
    _upload(at, "extremes.qsf", survey_extremes())
    _open_design(at)  # a 60-item battery, a 0-1000 slider and a minimum of 1900 used to raise on the first render
    by_name = {r[0]: r for r in _dv_rows(at)}
    assert by_name["Battery"][2] == 60
    assert (by_name["Points"][3], by_name["Points"][4]) == (0, 1000)
    assert (by_name["BirthYear"][3], by_name["BirthYear"][4]) == (1900, 2003)
    # a declared span above 1000 is mapped to the 0-100 grid by the bridge (an absolute scale is arbitrary there)
    assert by_name["Income"][1] == "numeric_input" and (by_name["Income"][3], by_name["Income"][4]) == (0, 100)
    _confirm(at)
    final = {s["variable_name"]: s for s in at.session_state["inferred_design"]["scales"]}
    assert final["Points"]["scale_max"] == 1000 and final["Battery"]["num_items"] == 60


def test_a_seeded_range_can_be_lowered_and_raised_again(app_test):
    at = app_test
    _start(at)
    _upload(at, "survey_a.qsf", survey_a())
    _open_design(at)
    budget = [r for r in _dv_rows(at) if r[0] == "Budget"][0]
    assert budget[4] == 150                                  # arrived as a 0-150 slider
    index = [r[0] for r in _dv_rows(at)].index("Budget")
    at.number_input(key=f"dv_max_v0_{index}").set_value(120)
    at.run()
    _no_exception(at)
    assert at.number_input(key=f"dv_max_v0_{index}").value == 120
    at.number_input(key=f"dv_max_v0_{index}").set_value(150)  # the box must not have shrunk to the edited value
    at.run()
    _no_exception(at)
    assert at.number_input(key=f"dv_max_v0_{index}").value == 150
    assert at.session_state["confirmed_scales"][index]["scale_max"] == 150
    _confirm(at)
    final = [s for s in at.session_state["inferred_design"]["scales"] if s["variable_name"] == "Budget"][0]
    assert (final["scale_min"], final["scale_max"]) == (0, 150) and "_seed_scale_max" not in final


# ---------------------------------------------------------------------------------------------------------------------
# 2. Edits persist and drive generation; a different upload starts over
# ---------------------------------------------------------------------------------------------------------------------
def test_dv_edits_persist_across_reruns_and_the_generated_data_follows_them(app_test):
    at = app_test
    _start(at)
    _upload(at, "survey_a.qsf", survey_a())
    _open_design(at)
    assert [r[0] for r in _dv_rows(at)] == A_DVS

    _button(at, "rm_dv_v0_1").click()                       # remove "Budget"
    at.run()
    _no_exception(at)
    assert [r[0] for r in _dv_rows(at)] == ["Trust", "Priorities", "BirthYear"]
    at.text_input(key="dv_name_v1_0").set_value("Trust_Index")   # rename the first one
    at.run()
    _button(at, "add_dv_btn_v1").click()                    # add a new one
    at.run()
    at.text_input(key="dv_name_v1_3").set_value("Satisfaction")
    at.run()
    expected = ["Trust_Index", "Priorities", "BirthYear", "Satisfaction"]
    assert [r[0] for r in _dv_rows(at)] == expected
    for _ in range(2):                                      # further reruns keep the edited list
        at.run()
        assert [r[0] for r in _dv_rows(at)] == expected
    _confirm(at)
    assert [s["variable_name"] for s in at.session_state["inferred_design"]["scales"]] == expected

    # leaving the page and coming back keeps the edits too
    _use_offline_engine(at)
    _button(at, "nav_next_2").click()
    at.run()
    _button(at, "nav_back_3").click()
    at.run()
    _no_exception(at)
    assert [r[0] for r in _dv_rows(at)] == expected

    # ... and the generated data (offline template engine) uses exactly the edited list
    _button(at, "nav_next_2").click()
    at.run()
    _button(at, "generate_dataset_btn").click()
    at.run()
    _no_exception(at)
    archive = zipfile.ZipFile(BytesIO(at.session_state["last_zip"]))
    metadata = json.loads(archive.read("Metadata.json"))
    assert [s["variable_name"] for s in metadata["scales"]] == expected
    import pandas as pd
    csv_name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
    columns = list(pd.read_csv(BytesIO(archive.read(csv_name))).columns)
    for name in expected:
        assert any(c == name or c.startswith(name + "_") for c in columns), (name, columns)
    assert not any(c.startswith(("Budget", "Main_DV")) for c in columns)


def test_a_new_upload_resets_the_dvs_to_the_new_surveys_detections(app_test):
    at = app_test
    _start(at)
    _upload(at, "survey_a.qsf", survey_a())
    _open_design(at)
    _button(at, "rm_dv_v0_0").click()                       # the user edits survey A's list ...
    at.run()
    assert [r[0] for r in _dv_rows(at)] == A_DVS[1:]
    _button(at, "nav_back_2").click()
    at.run()
    _upload(at, "survey_b.qsf", survey_b(), replace=True)
    _open_design(at)
    assert [r[0] for r in _dv_rows(at)] == B_DVS            # ... and survey B starts from its own DVs, not A's edits
    assert "Main_DV" not in [r[0] for r in _dv_rows(at)]


# ---------------------------------------------------------------------------------------------------------------------
# 3. Same file name, different content
# ---------------------------------------------------------------------------------------------------------------------
@pytest.fixture()
def collector(monkeypatch):
    """Consent box switched on and the GitHub collector replaced by a recorder."""
    import utils.github_qsf_collector as gc

    calls = []
    monkeypatch.setattr(gc, "is_collection_enabled", lambda: True)
    monkeypatch.setattr(gc, "collect_qsf_async", lambda name, content: calls.append((name, bytes(content))))
    return calls


def test_a_different_file_with_the_same_name_is_a_new_upload(app_test, collector):
    at = app_test
    _start(at)
    a, b = survey_a(), survey_b()
    at.checkbox(key="share_survey_consent_0").check()
    at.run()
    _upload(at, "Survey.qsf", a)
    first = at.session_state["qsf_preview"]
    assert first.survey_name == "Survey A" and len(collector) == 1
    assert at.session_state["qsf_file_name"] == "Survey.qsf"

    at.run()                                                # a plain rerun is not a new upload
    assert at.session_state["qsf_preview"] is first and len(collector) == 1

    _upload(at, "Survey.qsf", a, replace=True)              # identical content under the same name: nothing happens
    assert at.session_state["qsf_preview"] is first and len(collector) == 1

    at.checkbox(key="share_survey_consent_1").check()       # a Qualtrics re-export: new content, same name
    at.run()
    _upload(at, "Survey.qsf", b, replace=True)
    assert at.session_state["qsf_preview"].survey_name == "Survey B"
    assert at.session_state["qsf_raw_content"] == b
    assert [name.split("_", 3)[-1] for name, _ in collector] == ["Survey.qsf", "Survey.qsf"]
    assert collector[1][1] == b                             # one collected file per consented upload
    _open_design(at)
    assert [r[0] for r in _dv_rows(at)] == B_DVS


def test_no_consent_means_no_collection_even_for_a_changed_survey(app_test, collector):
    at = app_test
    _start(at)
    _upload(at, "Survey.qsf", survey_a())
    _upload(at, "Survey.qsf", survey_b(), replace=True)
    assert at.session_state["qsf_preview"].survey_name == "Survey B"
    assert collector == []


# ---------------------------------------------------------------------------------------------------------------------
# 4. A second upload does not inherit the first survey's state
# ---------------------------------------------------------------------------------------------------------------------
def test_a_second_upload_drops_the_first_surveys_checks_mediators_and_results(app_test):
    at = app_test
    _start(at)
    _upload(at, "survey_a.qsf", survey_a())
    _open_design(at)
    _confirm(at)
    assert at.session_state["confirmed_attention_checks"] == ["QID6"]
    assert {"[Block] Control", "[Block] Treatment", "QID6"} <= set(map(str, at.session_state["qsf_identifiers"]))
    _button(at, "add_med_btn_v0").click()                    # the user adds a mediator tied to survey A's DVs
    at.run()
    assert len(at.session_state["confirmed_mediators"]) == 1
    assert at.session_state["variable_review_rows"]
    first_rows = {r["Variable"] for r in at.session_state["variable_review_rows"]}
    at.session_state["has_generated"] = True                 # survey A's dataset is on the Generate page
    at.session_state["last_zip"] = b"PK"
    at.session_state["last_metadata"] = {"study_title": "A"}

    _button(at, "nav_back_2").click()
    at.run()
    _upload(at, "survey_b.qsf", survey_b(), replace=True)
    # study-level input the student typed is untouched
    assert at.session_state["_p_study_title"] == "Design wiring study"
    assert at.session_state["_p_study_description"].startswith("A pilot study of how product messages")
    assert at.session_state["sample_size"] == 20
    # everything derived from survey A is gone, before the new design page even renders
    for key in ("confirmed_attention_checks", "confirmed_manipulation_checks", "confirmed_comprehension_checks",
                "confirmed_mediators", "variable_review_rows", "qsf_identifiers", "confirmed_scales", "inferred_design"):
        assert key not in at.session_state, key
    assert not at.session_state["has_generated"] and at.session_state["last_zip"] is None

    _open_design(at)
    assert at.session_state["confirmed_attention_checks"] == []        # survey B has none
    assert at.session_state["confirmed_mediators"] == []
    identifiers = set(map(str, at.session_state["qsf_identifiers"]))
    assert "[Block] Low price" in identifiers                                  # survey B's own blocks ...
    assert not ({"[Block] Control", "[Block] Treatment", "QID6"} & identifiers)  # ... and none of survey A's
    _confirm(at)
    second_rows = {r["Variable"] for r in at.session_state["variable_review_rows"]}
    assert not ({"Trust", "Budget", "Priorities", "BirthYear"} & second_rows) and second_rows != first_rows
    assert [s["variable_name"] for s in at.session_state["inferred_design"]["scales"]] == B_DVS


# ---------------------------------------------------------------------------------------------------------------------
# The conversational builder keeps its own state and path
# ---------------------------------------------------------------------------------------------------------------------
def test_the_builder_path_is_unchanged(app_test):
    at = app_test
    at.session_state["active_page"] = 1
    at.session_state["study_input_mode"] = "describe_study"
    at.session_state["study_title"] = "Builder study"
    at.session_state["study_description"] = "A study built from a description."
    at.session_state["sample_size"] = 40
    at.run()
    examples = [b for b in at.button if str(b.key).startswith("example_btn_")]
    assert examples
    examples[0].click()
    at.run()
    _no_exception(at)
    assert at.session_state["active_page"] == 2 and at.session_state["conversational_builder_complete"]
    built = [s["variable_name"] for s in at.session_state["confirmed_scales"]]
    assert built and "Main_DV" not in built
    assert at.session_state["inferred_design"]["scales"] and not at.session_state.get("qsf_preview")
    _use_offline_engine(at)
    at.session_state["active_page"] = 3
    at.run()
    _no_exception(at)
    assert not _button(at, "generate_dataset_btn").disabled
    assert [s["variable_name"] for s in at.session_state["confirmed_scales"]] == built


# ---------------------------------------------------------------------------------------------------------------------
# QSF-supplied names now reach the Design page, so they must be escaped wherever raw HTML is built
# ---------------------------------------------------------------------------------------------------------------------
def test_a_qsf_variable_name_with_markup_is_escaped_in_the_construct_badges(app_test):
    at = app_test
    _start(at)
    at.session_state["advanced_mode"] = True
    payload = "Trust<img src=x onerror=alert(1)>"
    _upload(at, "markup.qsf", make_qsf("Markup", [likert_matrix("QID1", payload, ["a", "b", "c"]),
                                                  likert_matrix("QID2", "Satisfaction", ["x", "y", "z"])]))
    _open_design(at)
    _confirm(at)
    assert [r[0] for r in _dv_rows(at)] == [payload, "Satisfaction"]
    badges = [m.value for m in at.markdown if "&lt;img src=x" in str(m.value)]
    assert badges, "the construct badges were not rendered"
    assert not [m.value for m in at.markdown if "<img src=x" in str(m.value)]
