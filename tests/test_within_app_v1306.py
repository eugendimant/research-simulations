"""The design-type selector is real: Design page -> Generate -> ZIP, reports and the instructor email (v1.3.0.6).

Drives the real Streamlit app with AppTest from a temporary working directory (no data/ folder in the repository),
on both paths: the QSF path (design selector on the Design page) and the described-study (builder) path.
"""
import io
import json
import smtplib
import sys
import threading
import zipfile
from pathlib import Path

import pandas as pd
import pytest

_APP_DIR = Path(__file__).resolve().parent.parent / "simulation_app"
_USAGE_FILE = _APP_DIR / ".usage_counter.json"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

WITHIN_OPTION = "Within-subjects (each participant sees all conditions)"
MIXED_OPTION = "Mixed design"


@pytest.fixture()
def apptest_env(monkeypatch, tmp_path):
    existed = _USAGE_FILE.exists()
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    if not existed and _USAGE_FILE.exists():
        _USAGE_FILE.unlink()


class _RecordingSMTP:
    sent = []

    def __init__(self, host, port, timeout=None, context=None):
        pass

    def ehlo(self):
        pass

    def starttls(self, context=None):
        pass

    def login(self, user, password):
        pass

    def send_message(self, msg, from_addr=None, to_addrs=None, **_kw):
        _RecordingSMTP.sent.append(msg)
        return {}

    def quit(self):
        pass

    def close(self):
        pass


def _sibling(name):
    """A sibling test module (its AppTest helpers), whichever way pytest put the tests directory on sys.path."""
    import importlib
    try:
        return importlib.import_module(f"tests.{name}")
    except ImportError:
        return importlib.import_module(name)


def _helpers():
    return _sibling("test_app_hygiene_v1291")


def _zip(at) -> dict:
    archive = zipfile.ZipFile(io.BytesIO(at.session_state["last_zip"]))
    return {name: archive.read(name) for name in archive.namelist()}


def _csv(files, name) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(files[name]))


def _no_exception(at):
    assert not at.exception, [str(e.value) for e in at.exception]


def test_qsf_design_page_within_flow_zip_report_and_email(apptest_env, monkeypatch):
    H = _helpers()
    _RecordingSMTP.sent = []
    monkeypatch.setattr(smtplib, "SMTP", _RecordingSMTP)
    monkeypatch.setattr(smtplib, "SMTP_SSL", _RecordingSMTP)
    at = H._qsf_generate_page(n=60)
    for key, value in {"SMTP_SERVER": "smtp.example.org", "SMTP_PORT": 587, "SMTP_USERNAME": "s@example.org",
                       "SMTP_PASSWORD": "x" * 8, "SMTP_FROM_EMAIL": "s@example.org",
                       "INSTRUCTOR_NOTIFICATION_EMAIL": "owner@example.edu"}.items():
        at.secrets[key] = value
    H._goto(at, 2)
    next(s for s in at.selectbox if s.key == "design_type_select").select(WITHIN_OPTION)
    at.run()
    _no_exception(at)
    assert not [w for w in at.warning if "one condition per participant" in w.value]
    levels = next(t for t in at.text_input if t.key == "rm_within_levels")
    levels.set_value("Pre, Post")
    at.run()
    next(s for s in at.slider if s.key == "rm_corr").set_value(0.6)
    at.run()
    _no_exception(at)
    assert at.session_state["design_type_choice"] == "within"
    assert at.session_state["design_config"]["within_correlation"] == pytest.approx(0.6)

    # the choice survives a visit to another page (the widget key is gone while it is not rendered)
    H._goto(at, 3)
    H._goto(at, 2)
    assert next(s for s in at.selectbox if s.key == "design_type_select").value == WITHIN_OPTION

    H._goto(at, 3)
    files = H._click_generate(at)
    # wide data: one row per participant, one column per measure and condition, the design columns
    wide = _csv(files, "Simulated_Data.csv")
    assert len(wide) == 60
    assert {"Order", "Position_Pre", "Position_Post", "Conditions_Completed"} <= set(wide.columns)
    pre = [c for c in wide.columns if c.startswith("LoyaltyQuestions_Pre_")]
    post = [c for c in wide.columns if c.startswith("LoyaltyQuestions_Post_")]
    assert len(pre) == len(post) >= 3
    # long data: one row per participant and condition, same values
    long = _csv(files, "Simulated_Data_Long.csv")
    assert len(long) == 120 and set(long["Condition"]) == {"Pre", "Post"} and "ResponseId" in long.columns
    one = long[(long["ResponseId"] == wide["ResponseId"].iloc[3]) & (long["Condition"] == "Post")].iloc[0]
    assert one["LoyaltyQuestions_1"] == wide["LoyaltyQuestions_Post_1"].iloc[3]
    metadata = json.loads(files["Metadata.json"])
    d = metadata["design"]
    assert metadata["design_type"] == "within" and metadata["design_review"]["design_type"] == "within"
    assert [c["label"] for c in d["cells"]] == ["Pre", "Post"]
    assert d["within_correlation"] == pytest.approx(0.6) and d["order_effective"] == "random"
    assert d["long_format_file"] == "Simulated_Data_Long.csv"
    # scripts and codebook know the layout and stay honest
    r_script = files["R_Prepare_Data.R"].decode()
    assert "LoyaltyQuestions_Pre_composite" in r_script and "does not run any analysis" in r_script
    assert "REPEATED-MEASURES LAYOUT" in files["Data_Codebook_Handbook.txt"].decode()
    # the student summary explains it
    summary = files["User_Study_Summary.md"].decode()
    assert "Data layout (repeated measures)" in summary and "ttest_rel" in summary
    # the instructor reports carry paired statistics
    for t in threading.enumerate():
        if t.name == "instructor-email":
            t.join(30)
    assert len(_RecordingSMTP.sent) == 2, "the instructor email keeps its two messages"
    body = _RecordingSMTP.sent[0].get_body(preferencelist=("plain",)).get_content()
    assert "Paired comparisons" in body and "d_av" in body
    for banned in ("Independent-samples", "pooled-variance", "One-way ANOVA"):
        assert banned not in body
    attachments = {p.get_filename() for m in _RecordingSMTP.sent for p in m.walk() if p.get_filename()}
    assert any(a and a.endswith(".html") for a in attachments)


def test_qsf_design_page_suggests_but_never_switches_on_a_repeated_design(apptest_env):
    H = _helpers()
    T = _sibling("test_design_wiring_v1291")
    from streamlit.testing.v1 import AppTest
    from utils.qsf_preview import QSFPreviewParser
    import app as appmod

    elements, blocks, flow = [], {}, []
    for i, name in enumerate(("Pre-test", "Post-test")):
        elements.append(T.likert_matrix(f"QID{i + 1}", f"Anx{i + 1}", ["I feel anxious", "I feel tense", "I feel nervous"]))
        bid = f"BL_{i + 1:03d}"
        blocks[str(i + 1)] = {"Type": "Standard", "Description": name, "ID": bid,
                              "BlockElements": [{"Type": "Question", "QuestionID": f"QID{i + 1}"}]}
        flow.append({"Type": "Standard", "ID": bid, "FlowID": f"FL_{i + 10}"})
    elements.append({"SurveyID": "SV_t", "Element": "BL", "PrimaryAttribute": "Survey Blocks", "SecondaryAttribute": None,
                     "TertiaryAttribute": None, "Payload": blocks})
    elements.append({"SurveyID": "SV_t", "Element": "FL", "PrimaryAttribute": "Survey Flow", "SecondaryAttribute": None,
                     "TertiaryAttribute": None, "Payload": {"Type": "Root", "FlowID": "FL_1", "Flow": flow}})
    raw = json.dumps({"SurveyEntry": {"SurveyID": "SV_t", "SurveyName": "PrePost", "SurveyDescription": "",
                                      "SurveyLanguage": "EN"}, "SurveyElements": elements}).encode()
    preview = QSFPreviewParser().parse(raw)
    inputs = appmod._preview_to_engine_inputs(preview)
    at = AppTest.from_file(str(_APP_DIR / "app.py"), default_timeout=300)
    state = {"active_page": 2, "study_title": "Pre post", "study_description": "Anxiety before and after a workshop.",
             "sample_size": 40, "study_input_mode": "upload_qsf", "qsf_preview": preview, "qsf_raw_content": raw,
             "qsf_file_name": "prepost.qsf", "confirmed_scales": inputs["scales"], "scales_confirmed": True,
             "open_ended_confirmed": True, "confirmed_open_ended": [], "team_name": "t", "team_members_raw": "a",
             "inferred_design": {"conditions": [], "factors": [], "scales": inputs["scales"], "open_ended_questions": [],
                                 "attention_checks": [], "manipulation_checks": [], "condition_visibility_map": {}}}
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    _no_exception(at)
    select = next(s for s in at.selectbox if s.key == "design_type_select")
    assert select.value.startswith("Between")                        # never switched on by itself
    assert [i for i in at.info if "same questions" in i.value], [i.value for i in at.info]
    button = next(b for b in at.button if b.key == "use_design_suggestion")
    button.click()
    at.run()
    _no_exception(at)
    assert next(s for s in at.selectbox if s.key == "design_type_select").value == WITHIN_OPTION
    assert next(t for t in at.text_input if t.key == "rm_within_levels").value == "Pre-test, Post-test"


def test_builder_path_mixed_design_with_a_group_by_time_interaction(apptest_env):
    H = _helpers()
    conds = ["Training", "Waitlist"]
    at = H._generate_page(conds, n=120, advanced=True, extra_state={
        "builder_design_type": "mixed", "_user_seed_value": 5, "_auto_effects": False,
        "design_config": {"within_name": "Time", "within_levels": "Pre, Post", "order": "latin_square",
                          "within_correlation": 0.5, "order_effects": False},
        "builder_effect_sizes": [{"variable": "Satisfaction", "factor": "Time", "level_high": "Post", "level_low": "Pre",
                                  "cohens_d": 0.3, "direction": "positive"},
                                 {"variable": "Satisfaction", "factor": "condition", "level_high": "Training",
                                  "level_low": "Waitlist", "cohens_d": 0.8, "direction": "positive", "at": {"Time": "Post"}}]})
    files = H._click_generate(at)
    metadata = json.loads(files["Metadata.json"])
    d = metadata["design"]
    assert metadata["design_type"] == "mixed" and [c["label"] for c in d["cells"]] == ["Pre", "Post"]
    scopes = {(s["kind"], json.dumps(s["scope"])) for s in metadata["effect_sizes_applied"]["specs"]}
    assert ("within", "null") in scopes and ("between", json.dumps({"Time": "Post"})) in scopes
    wide = _csv(files, "Simulated_Data.csv")
    assert set(wide["CONDITION"]) == set(conds)
    long = _csv(files, "Simulated_Data_Long.csv")
    assert len(long) == 240
    comp = metadata["effect_sizes_applied"]["specs"]
    assert all(s["status"] == "applied" for s in comp)


def test_builder_within_requires_two_conditions_before_continuing(apptest_env):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(_APP_DIR / "app.py"), default_timeout=300)
    for key, value in {"active_page": 1, "study_input_mode": "describe_study", "study_title": "T", "study_description": "D study",
                       "builder_conditions_text": "Pre-test\nPost-test", "builder_scales_text": "Trust (1-7 Likert, 5 items)",
                       "cond_input_mode": "Text / Factorial notation"}.items():
        at.session_state[key] = value
    at.run()
    _no_exception(at)
    radio = next(r for r in at.radio if r.key == "builder_design_type_input")
    assert radio.value == "within"
    assert any("Repeated-measures setup" in m.value for m in at.markdown)
    radio.set_value("mixed")
    at.run()
    _no_exception(at)
    assert any(t.key == "rm_within_levels" for t in at.text_input)   # a mixed design asks for the within factor


def test_between_flow_is_unchanged_by_the_feature(apptest_env):
    H = _helpers()
    at = H._qsf_generate_page(n=40)
    files = H._click_generate(at)
    metadata = json.loads(files["Metadata.json"])
    assert "design" not in metadata and metadata["design_review"]["design_type"] == "between"
    assert "Simulated_Data_Long.csv" not in files
    assert "CONDITION" in _csv(files, "Simulated_Data.csv").columns


def test_preview_rows_use_the_repeated_measures_layout(apptest_env):
    H = _helpers()
    at = H._qsf_generate_page(n=40)
    H._goto(at, 2)
    next(s for s in at.selectbox if s.key == "design_type_select").select(WITHIN_OPTION)
    at.run()
    next(t for t in at.text_input if t.key == "rm_within_levels").set_value("Pre, Post")
    at.run()
    H._goto(at, 3)
    next(b for b in at.button if b.key == "preview_button").click()
    at.run()
    _no_exception(at)
    frame = at.session_state["preview_df"]
    assert len(frame) == 5 and {"Order", "Position_Pre", "Position_Post"} <= set(frame.columns)
    assert any(c.startswith("LoyaltyQuestions_Pre_") for c in frame.columns)


def test_fixed_order_warns_on_the_design_page_and_adds_no_drift(apptest_env):
    H = _helpers()
    at = H._qsf_generate_page(n=40)
    H._goto(at, 2)
    next(s for s in at.selectbox if s.key == "design_type_select").select(WITHIN_OPTION)
    at.run()
    assert not [w for w in at.warning if "confounded" in w.value]
    next(s for s in at.selectbox if s.key == "rm_order").select("fixed")
    at.run()
    _no_exception(at)
    assert [w for w in at.warning if "confounded with the conditions themselves" in w.value]
    assert at.session_state["design_config"]["order_effects"] is False
    H._goto(at, 3)
    files = H._click_generate(at)
    d = json.loads(files["Metadata.json"])["design"]
    assert d["order"] == "fixed" and d["order_effect"]["enabled"] is False and d["notes"]


def test_open_ended_question_asked_in_every_condition_gets_a_column_per_condition(apptest_env):
    H = _helpers()
    conds = ["Pre", "Post"]
    oe = [{"variable_name": "Why", "name": "Why", "question_text": "Why did you answer this way?", "source_type": "text",
           "question_context": "reasons for the ratings", "question_purpose": "DV Response", "context_type": "general"}]
    at = H._generate_page(conds, n=40, advanced=False, extra_state={
        "builder_design_type": "within", "confirmed_open_ended": oe,
        "design_config": {"order": "latin_square", "within_correlation": 0.5, "oe_every_condition": True}})
    files = H._click_generate(at)
    wide = _csv(files, "Simulated_Data.csv")
    assert {"Why_Pre", "Why_Post"} <= set(wide.columns)
    d = json.loads(files["Metadata.json"])["design"]
    assert d["open_ended_columns"]["Why"] == {"Pre": "Why_Pre", "Post": "Why_Post"}
    long = _csv(files, "Simulated_Data_Long.csv")
    assert "Why" in long.columns and len(long) == 80
