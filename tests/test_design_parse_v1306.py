"""Design-page parsing and persistence fixes (v1.3.0.6).

1. Conditions come from the QSF flow (the arms of a between-subjects randomizer, or embedded-data branches);
   survey SECTIONS (consent, demographics, feedback, checkpoints, ...) are never pre-selected as conditions.
2. A text box whose Qualtrics validation says "number" is ONE numeric DV with its declared range: it is no longer
   folded into a numbered 1-7 group and then recovered a second time, so every output column is defined once.
3. Numeric-box recognition lets the survey's validation outrank the wording (an e-mail box that mentions "$50").
4. The advanced "Expected Effect Sizes" widgets survive leaving the Generate page and coming back.

Small synthetic QSFs below; the corpus-level tests use the collected example surveys and skip without them.
Corpus numbers (303 surveys, measured before -> after this change): surveys pre-selecting more than 10 "conditions"
113 -> 5 (all five are genuine multi-randomizer designs with 13-39 arms); surveys whose PRE-SELECTED conditions equal
the randomizer arms 32 -> 139 of the 175 that have a between-subjects randomizer (detected conditions alone: 120 -> 139).
"""
import ast
import contextlib
import io
import json
import re
import sys
from pathlib import Path

import pytest

_APP_DIR = Path(__file__).resolve().parent.parent / "simulation_app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from utils.qsf_preview import QSFPreviewParser  # noqa: E402
from utils.enhanced_simulation_engine import infer_numeric_answer_spec  # noqa: E402

_EXAMPLES = _APP_DIR / "example_files"


# ---------------------------------------------------------------------------------------------------------------------
# Synthetic QSF builder
# ---------------------------------------------------------------------------------------------------------------------
def _sq(qid, tag, text, qtype, selector, **extra):
    payload = {"QuestionText": text, "DataExportTag": tag, "QuestionType": qtype, "Selector": selector,
               "QuestionDescription": text, "QuestionID": qid, "Language": [],
               "Validation": {"Settings": {"ForceResponse": "OFF", "Type": "None"}}}
    payload.update(extra)
    return {"SurveyID": "SV_t", "Element": "SQ", "PrimaryAttribute": qid, "SecondaryAttribute": text,
            "TertiaryAttribute": None, "Payload": payload}


def _text(qid, tag, text="Some text"):
    return _sq(qid, tag, text, "DB", "TB")


def _likert(qid, tag, text="How much do you agree?"):
    return _sq(qid, tag, text, "MC", "SAVR",
               Choices={str(i): {"Display": d} for i, d in enumerate(["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"], 1)},
               ChoiceOrder=[1, 2, 3, 4, 5])


def _number_box(qid, tag, text, lo=None, hi=None, content_type="ValidNumber"):
    settings = {"ForceResponse": "OFF", "Type": "ContentType", "ContentType": content_type}
    if lo is not None:
        settings["ValidNumber"] = {"Min": str(lo), "Max": str(hi), "NumDecimals": "0"}
    return _sq(qid, tag, text, "TE", "SL", Validation={"Settings": settings})


def _qsf(block_names, questions, flow, name="Synthetic"):
    """block_names: {block_id: (description, [question ids])}; flow: list of flow dicts."""
    blocks = {}
    for i, (bid, (desc, qids)) in enumerate(block_names.items(), 1):
        blocks[str(i)] = {"Type": "Standard", "Description": desc, "ID": bid,
                          "BlockElements": [{"Type": "Question", "QuestionID": q} for q in qids]}
    els = list(questions)
    els.append({"SurveyID": "SV_t", "Element": "BL", "PrimaryAttribute": "Survey Blocks", "SecondaryAttribute": None,
                "TertiaryAttribute": None, "Payload": blocks})
    els.append({"SurveyID": "SV_t", "Element": "FL", "PrimaryAttribute": "Survey Flow", "SecondaryAttribute": None,
                "TertiaryAttribute": None, "Payload": {"Type": "Root", "FlowID": "FL_1", "Flow": flow}})
    return json.dumps({"SurveyEntry": {"SurveyID": "SV_t", "SurveyName": name, "SurveyDescription": "",
                                       "SurveyLanguage": "EN"}, "SurveyElements": els}).encode("utf-8")


def _blk(bid):
    return {"Type": "Standard", "ID": bid, "FlowID": f"FL_{bid}"}


def _parse(raw):
    return QSFPreviewParser().parse(raw)


def _sectioned_survey(subset="1", n_arms=3):
    """Consent, Demographics, Feedback and a checkpoint around a randomizer of ``n_arms`` stimulus blocks."""
    qs = [_text(f"QID{i}", f"s{i}") for i in range(1, 5)] + [_likert(f"QID{10 + i}", f"stim{i}") for i in range(n_arms)]
    blocks = {"BL_c": ("Consent", ["QID1"]), "BL_d": ("Demographic Info and Conclusion", ["QID2"]),
              "BL_f": ("Feedback on the Survey", ["QID3"]), "BL_k": ("Checkpoint 1", ["QID4"])}
    arm_names = ["Warm message", "Cold message", "Neutral message", "Humor message"][:n_arms]
    for i, nm in enumerate(arm_names):
        blocks[f"BL_a{i}"] = (nm, [f"QID{10 + i}"])
    flow = [_blk("BL_c"),
            {"Type": "BlockRandomizer", "FlowID": "FL_r", "SubSet": subset, "EvenPresentation": True,
             "Flow": [_blk(f"BL_a{i}") for i in range(n_arms)]},
            _blk("BL_k"), _blk("BL_d"), _blk("BL_f")]
    return _qsf(blocks, qs, flow), arm_names


# ---------------------------------------------------------------------------------------------------------------------
# 1. Conditions
# ---------------------------------------------------------------------------------------------------------------------
def test_randomizer_arms_are_the_conditions_and_sections_are_not():
    raw, arms = _sectioned_survey()
    pv = _parse(raw)
    assert pv.detected_conditions == arms
    assert pv.condition_evidence == "randomizer"
    for sec in ("Consent", "Demographic Info and Conclusion", "Feedback on the Survey", "Checkpoint 1"):
        assert sec in pv.section_blocks
        assert sec not in pv.detected_conditions


def test_order_only_randomizer_is_not_a_condition():
    """SubSet equal to the child count shows every block (a shuffle), so its children are not arms."""
    raw, arms = _sectioned_survey(subset="3", n_arms=3)
    assert _parse(raw).detected_conditions == []


def test_embedded_data_branches_give_the_conditions():
    qs = [_text("QID1", "i"), _likert("QID2", "a"), _likert("QID3", "b")]
    blocks = {"BL_i": ("Intro", ["QID1"]), "BL_a": ("Anchored frame", ["QID2"]), "BL_b": ("Plain frame", ["QID3"])}

    def branch(val, bid):
        return {"Type": "Branch", "FlowID": f"FL_{bid}", "Description": "New Branch",
                "BranchLogic": {"0": {"0": {"LogicType": "EmbeddedField", "LeftOperand": "REC", "Operator": "EqualTo",
                                            "RightOperand": val, "Type": "Expression"}, "Type": "If"}, "Type": "BooleanExpression"},
                "Flow": [_blk(bid)]}
    flow = [_blk("BL_i"), branch("1", "BL_a"), branch("0", "BL_b")]
    pv = _parse(_qsf(blocks, qs, flow))
    assert pv.detected_conditions == ["Anchored frame", "Plain frame"]
    assert pv.condition_evidence == "branch"


def test_branch_on_a_survey_answer_is_not_a_condition():
    qs = [_text("QID1", "i"), _likert("QID2", "a"), _likert("QID3", "b")]
    blocks = {"BL_i": ("Intro", ["QID1"]), "BL_a": ("Part one", ["QID2"]), "BL_b": ("Part two", ["QID3"])}

    def branch(bid):
        return {"Type": "Branch", "FlowID": f"FL_{bid}", "Description": "New Branch",
                "BranchLogic": {"0": {"0": {"LogicType": "Question", "QuestionID": "QID1", "Operator": "Selected", "Type": "Expression"},
                                      "Type": "If"}, "Type": "BooleanExpression"}, "Flow": [_blk(bid)]}
    pv = _parse(_qsf(blocks, qs, [_blk("BL_i"), branch("BL_a"), branch("BL_b")]))
    assert pv.detected_conditions == []


def test_numeric_arm_names_survive():
    qs = [_likert("QID1", "a"), _likert("QID2", "b")]
    blocks = {"BL_a": ("1", ["QID1"]), "BL_b": ("2", ["QID2"])}
    flow = [{"Type": "BlockRandomizer", "FlowID": "FL_r", "SubSet": "1", "EvenPresentation": True,
             "Flow": [_blk("BL_a"), _blk("BL_b")]}]
    assert _parse(_qsf(blocks, qs, flow)).detected_conditions == ["1", "2"]


def test_condition_named_blocks_without_a_randomizer_are_found_by_name():
    """Embedded-data randomizer with digits only, blocks named "<X> Condition n": the names carry the arms."""
    qs = [_likert(f"QID{i}", f"q{i}") for i in range(1, 5)]
    blocks = {f"BL_{i}": (f"Sweater Condition {i}", [f"QID{i}"]) for i in range(1, 5)}
    flow = [{"Type": "BlockRandomizer", "FlowID": "FL_r", "SubSet": "4", "EvenPresentation": True,
             "Flow": [{"Type": "EmbeddedData", "FlowID": f"FL_e{i}", "EmbeddedData": [{"Description": "Condition", "Type": "Custom", "Field": "Condition", "VariableType": "String", "DataVisibility": [], "AnalyzeText": False, "Value": str(i)}]} for i in range(1, 5)]}] + [_blk(b) for b in blocks]
    pv = _parse(_qsf(blocks, qs, flow))
    assert pv.detected_conditions == [f"Sweater Condition {i}" for i in range(1, 5)]
    assert pv.condition_evidence == "name"


def test_unrelated_embedded_data_value_is_not_a_condition():
    qs = [_likert("QID1", "a")]
    blocks = {"BL_a": ("Round 1", ["QID1"])}
    flow = [{"Type": "EmbeddedData", "FlowID": "FL_e", "EmbeddedData": [{"Description": "Excluded", "Type": "Custom", "Field": "Excluded", "VariableType": "String", "DataVisibility": [], "AnalyzeText": False, "Value": "Custom Value"}]},
            _blk("BL_a")]
    assert _parse(_qsf(blocks, qs, flow)).detected_conditions == []


def _app_functions(*names):
    """Pull module-level functions out of app.py without importing the Streamlit script."""
    import typing
    tree = ast.parse((_APP_DIR / "app.py").read_text(encoding="utf-8"))
    ns = {"re": re, "np": __import__("numpy"), "json": json, "Any": typing.Any, "Dict": typing.Dict, "List": typing.List,
          "Optional": typing.Optional, "Tuple": typing.Tuple, "QSFPreviewResult": object, "DesignAnalysisResult": object}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], []), "app.py", "exec"), ns)
    return [ns[n] for n in names]


def test_default_selection_is_the_flow_evidence_not_every_block():
    candidates_fn, default_fn = _app_functions("_get_condition_candidates", "_default_condition_selection")
    raw, arms = _sectioned_survey()
    pv = _parse(raw)
    candidates = candidates_fn(pv, None)
    assert set(arms) <= set(candidates)
    for sec in ("Consent", "Demographic Info and Conclusion", "Feedback on the Survey", "Checkpoint 1"):
        assert sec not in candidates
    assert default_fn(pv, None, candidates) == arms


def test_default_selection_is_empty_without_evidence():
    candidates_fn, default_fn = _app_functions("_get_condition_candidates", "_default_condition_selection")
    qs = [_likert("QID1", "a"), _likert("QID2", "b")]
    blocks = {"BL_a": ("Self Interest Statement", ["QID1"]), "BL_b": ("Charity Statement", ["QID2"])}
    pv = _parse(_qsf(blocks, qs, [_blk("BL_a"), _blk("BL_b")]))
    assert default_fn(pv, None, candidates_fn(pv, None)) == []


def _corpus_files():
    return sorted(_EXAMPLES.glob("*.qsf")) if _EXAMPLES.is_dir() else []


@pytest.fixture(scope="module")
def corpus_parse():
    files = _corpus_files()
    if len(files) < 250:
        pytest.skip("example corpus not available")
    out = {}
    for f in files:
        pv = _parse(f.read_bytes())
        out[f.name] = pv
    return out


def _between_arms(raw):
    """Independent reading of the flow: names of the blocks directly inside SubSet=1 randomizers."""
    d = json.loads(raw.decode("utf-8"))
    blocks, flow = {}, None
    for e in d.get("SurveyElements", []):
        if e.get("Element") == "BL":
            pl = e.get("Payload", {})
            for b in (pl.values() if isinstance(pl, dict) else pl):
                blocks[b.get("ID")] = (b.get("Description") or "").strip()
        elif e.get("Element") == "FL":
            flow = e.get("Payload")
    arms = set()

    def walk(items):
        if isinstance(items, dict):
            items = list(items.values()) if "Type" not in items else [items]
        for it in items or []:
            if not isinstance(it, dict):
                continue
            sub = it.get("Flow", [])
            if isinstance(sub, dict):
                sub = list(sub.values())
            if it.get("Type") in ("Randomizer", "BlockRandomizer") and str(it.get("SubSet", 1)) == "1":
                names = {blocks.get(s.get("ID"), "") for s in sub if isinstance(s, dict) and s.get("Type") in ("Standard", "Block")}
                names.discard("")
                if len(names) >= 2:
                    arms.update(n.replace("\xa0", " ").strip() for n in names)
            walk(sub)
    walk(flow.get("Flow", []) if isinstance(flow, dict) else flow)
    return arms


def test_corpus_few_surveys_preselect_more_than_ten_conditions(corpus_parse):
    over = [n for n, pv in corpus_parse.items() if len(pv.detected_conditions) > 10]
    assert len(over) <= 8, over   # measured 5 (113 before); each is a multi-randomizer design


def test_corpus_detected_conditions_match_randomizer_arms(corpus_parse):
    equal = total = 0
    for f in _corpus_files():
        arms = _between_arms(f.read_bytes())
        if not arms:
            continue
        total += 1
        got = {c.replace("\xa0", " ").strip() for c in corpus_parse[f.name].detected_conditions}
        equal += got == arms
    assert total >= 130
    assert equal >= 108, (equal, total)   # measured 110 of 138 with this strict reading (105 before)


def test_corpus_sound_detections_did_not_regress(corpus_parse):
    """Surveys whose conditions are exactly their 2-6 randomizer arms keep exactly those."""
    sound = [n for n in ("Strategic_silence_experiment.qsf", "Group4_FinalStudy.qsf") if (_EXAMPLES / n).exists()]
    for n in sound:
        got = {c.replace("\xa0", " ").strip() for c in corpus_parse[n].detected_conditions}
        assert got == _between_arms((_EXAMPLES / n).read_bytes()), n


# ---------------------------------------------------------------------------------------------------------------------
# 2. Duplicate DVs
# ---------------------------------------------------------------------------------------------------------------------
def _punish_like():
    qs = [_number_box(f"QID{i}", f"Punish_{a}_{b}", f"How many points for {a} vs {b}?", 0, 5)
          for i, (a, b) in enumerate([(1, 2), (1, 3), (2, 3), (2, 4)], 1)]
    blocks = {"BL_p": ("Punishment", [q["Payload"]["QuestionID"] for q in qs])}
    return _qsf(blocks, qs, [_blk("BL_p")])


def test_validated_number_boxes_are_numeric_dvs_with_their_declared_range():
    pv = _parse(_punish_like())
    by_name = {s["variable_name"]: s for s in pv.detected_scales}
    assert sorted(by_name) == ["Punish_1_2", "Punish_1_3", "Punish_2_3", "Punish_2_4"]
    for s in by_name.values():
        assert s["type"] == "numeric_input" and (s["scale_min"], s["scale_max"]) == (0, 5)


def test_every_output_column_is_defined_once():
    to_inputs = _app_functions("_preview_to_engine_inputs", "_infer_factors_from_conditions", "_infer_factor_name",
                               "_clean_condition_name")[0]
    pv = _parse(_punish_like())
    cols = [s["variable_name"] for s in to_inputs(pv)["scales"]]
    assert len(cols) == len(set(cols)) == 4


def test_colliding_single_item_numeric_is_dropped_in_favour_of_the_multi_item_scale():
    """The same export tag seen by two detectors (matrix column + number box) is generated once."""
    tags = ["Punish_1_1", "Punish_1_2"]
    pv = _parse(_punish_like())
    to_inputs = _app_functions("_preview_to_engine_inputs", "_infer_factors_from_conditions", "_infer_factor_name",
                               "_clean_condition_name")[0]
    pv.detected_scales = [
        {"name": "Punish_1", "variable_name": "Punish_1", "items": 2, "type": "matrix", "scale_points": 7,
         "scale_min": 1, "scale_max": 7, "detected_from_qsf": True},
    ] + pv.detected_scales
    cols = []
    for s in to_inputs(pv)["scales"]:
        k = int(s.get("num_items") or 1)
        cols += [f"{s['variable_name']}_{i}" for i in range(1, k + 1)] if k > 1 else [s["variable_name"]]
    assert all(cols.count(t) <= 1 for t in tags + cols)


def test_corpus_no_duplicate_columns_and_punishers_clean():
    if not (_EXAMPLES / "Punishers_01_Mar07.qsf").exists():
        pytest.skip("example corpus not available")
    to_inputs = _app_functions("_preview_to_engine_inputs", "_infer_factors_from_conditions", "_infer_factor_name",
                               "_clean_condition_name")[0]
    for name in ("Punishers_01_Mar07.qsf", "Punishers_02_Mar13.qsf"):
        scales = to_inputs(_parse((_EXAMPLES / name).read_bytes()))["scales"]
        cols = []
        for s in scales:
            k = int(s.get("num_items") or 1)
            cols += [f"{s['variable_name']}_{i}".lower() for i in range(1, k + 1)] if k > 1 else [s["variable_name"].lower()]
        assert len(cols) == len(set(cols)), name   # measured: 4 duplicated columns in each file before
        # No DV is lost to the dedupe: all ten Punish_a_b number boxes survive (colliding tags get "_<QID>").
        numeric = [s["variable_name"] for s in scales if s.get("type") in ("numeric", "numeric_input")
                   and s["variable_name"].startswith("Punish_")]
        assert len(numeric) == 10, (name, numeric)
        if name.startswith("Punishers_02"):
            assert {"Punish_1", "Punish_2", "Punish_3", "Punish_4"} <= {s["variable_name"] for s in scales}
            assert "Punish_1_2_QID127" in numeric


# ---------------------------------------------------------------------------------------------------------------------
# 3. Numeric boxes: validation first, wording second
# ---------------------------------------------------------------------------------------------------------------------
def test_email_box_that_mentions_money_is_not_numeric():
    q = {"content_type": "ValidEmail", "selector": "SL"}
    assert infer_numeric_answer_spec("Please enter your email if you would like to enter a lottery to win $50.", "Q47", q) is None


@pytest.mark.parametrize("ctype", ["ValidPhone", "ValidDate", "ValidUSPhone", "ValidURL"])
def test_other_validations_are_never_numeric(ctype):
    assert infer_numeric_answer_spec("How much would you pay? Enter the amount.", "Q1", {"content_type": ctype}) is None


@pytest.mark.parametrize("text", [
    "What was your strategy in determining the maximum price that you are willing to pay?",
    "If you are selected to receive the amount of the $10 that you choose to keep, would you provide your email address or phone number?",
    "If you would like to enter your earned tickets into the raffle, please enter your email.",
])
def test_free_text_wording_that_names_money_stays_free_text(text):
    assert infer_numeric_answer_spec(text, "Q1", {"selector": "ML"}) is None


def test_genuine_numeric_boxes_are_unchanged():
    assert infer_numeric_answer_spec("How many tickets will you get? (enter a number)", "Q1", {})["kind"] == "count"
    assert infer_numeric_answer_spec("What is your willingness to pay? It has to be between $0.00 - $10.00.", "Q2", {})["kind"] == "money"
    assert infer_numeric_answer_spec("Anything", "Q3", {"content_type": "ValidNumber", "number_min": 0, "number_max": 5})["kind"] == "count"
    assert infer_numeric_answer_spec("What is your ZIP code?", "Q4", {"content_type": "ValidZip"})["kind"] == "zip"
    assert infer_numeric_answer_spec("How many times a day do you check your phone?", "Q5", {})["kind"] == "count"


def test_corpus_validated_boxes_agree_with_their_validation(corpus_parse):
    bad = []
    for name, pv in corpus_parse.items():
        for te in pv.text_entry_questions:
            ctype = str(te.get("content_type") or "").lower()
            spec = infer_numeric_answer_spec(te.get("question_text", ""), te.get("export_tag", ""), te)
            if ctype in ("validemail", "validphone", "validdate") and spec is not None:
                bad.append((name, te.get("export_tag")))
            if ctype == "validnumber" and spec is None:
                bad.append((name, te.get("export_tag")))
    assert bad == []   # 2 e-mail boxes with "win $50" wording were numeric before


# ---------------------------------------------------------------------------------------------------------------------
# 4. Effect widgets survive a page switch
# ---------------------------------------------------------------------------------------------------------------------
def test_advanced_effect_widgets_survive_leaving_and_returning(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    arms = ["Warm message", "Cold message", "Neutral message"]
    raw, _ = _sectioned_survey(n_arms=3)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        import app as appmod
        pv = _parse(raw)
        inp = appmod._preview_to_engine_inputs(pv)
    usage_counter = _APP_DIR / ".usage_counter.json"
    existed = usage_counter.exists()
    at = AppTest.from_file(str(_APP_DIR / "app.py"), default_timeout=300)
    seed = {"active_page": 3, "study_title": "t", "study_description": "d d d d", "sample_size": 60,
            "study_input_mode": "upload_qsf", "qsf_preview": pv, "qsf_raw_content": raw, "qsf_file_name": "s.qsf",
            "confirmed_scales": inp["scales"], "scales_confirmed": True, "confirmed_conditions": arms,
            "selected_conditions": list(arms), "team_name": "t", "team_members_raw": "a", "advanced_mode": True,
            "inferred_design": {"conditions": arms, "factors": inp["factors"], "scales": inp["scales"],
                                "open_ended_questions": [], "attention_checks": [], "manipulation_checks": [],
                                "randomization_level": "Participant-level", "condition_visibility_map": {}}}
    for k, v in seed.items():
        at.session_state[k] = v

    def run():
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            at.run()

    try:
        run()
        at.checkbox(key="add_effect_checkbox").check()
        run()
        at.slider(key="effect_cohens_d").set_value(0.8)
        run()
        {s.key: s for s in at.selectbox}["effect_level_high"].select(arms[1])
        run()
        {s.key: s for s in at.selectbox}["effect_level_low"].select(arms[0])
        run()
        assert not at.exception, [str(e.value)[:200] for e in at.exception]

        [b for b in at.button if b.key == "nav_back_3"][0].click()
        run()
        assert at.session_state["active_page"] == 2
        assert not any(s.key == "effect_cohens_d" for s in at.slider)   # the widget is really gone on this page

        at.session_state["_pending_nav"] = 3
        run()
        assert not at.exception, [str(e.value)[:200] for e in at.exception]
        assert at.checkbox(key="add_effect_checkbox").value is True
        assert at.slider(key="effect_cohens_d").value == pytest.approx(0.8)
        sel = {s.key: s.value for s in at.selectbox}
        assert sel["effect_level_high"] == arms[1] and sel["effect_level_low"] == arms[0]
    finally:
        if not existed and usage_counter.exists():
            usage_counter.unlink()
