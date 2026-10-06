"""App-level email tests: the instructor notification, student-email limits and the admin tab.

Offline: smtplib.SMTP is replaced by a recording fake and Streamlit secrets by plain dicts.
"""
import importlib.util
import json
import os
import smtplib
import sys
import threading
from pathlib import Path

import pytest

_APP_DIR = Path(__file__).resolve().parent.parent / "simulation_app"
FAKE_LOGIN_VALUE = "login" + "-value"


def _load_app():
    if str(_APP_DIR) not in sys.path:
        sys.path.insert(0, str(_APP_DIR))
    spec = importlib.util.spec_from_file_location("_app_email_test", str(_APP_DIR / "app.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["_app_email_test"] = module
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass
    return module


class RecordingSMTP:
    sent = []
    plan = []

    def __init__(self, host, port, timeout=None, context=None):
        self.host, self.port = host, port

    def ehlo(self):
        pass

    def starttls(self, context=None):
        pass

    def login(self, user, password):
        pass

    def send_message(self, msg, from_addr=None, to_addrs=None, **_kw):
        if RecordingSMTP.plan:
            outcome = RecordingSMTP.plan.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
        RecordingSMTP.sent.append((msg, list(to_addrs or [])))
        return {}

    def quit(self):
        pass

    def close(self):
        pass


SECRETS = {"SMTP_SERVER": "smtp.example.org", "SMTP_PORT": 587, "SMTP_USERNAME": "sender@example.org",
           "SMTP_PASSWORD": FAKE_LOGIN_VALUE, "SMTP_FROM_EMAIL": "sender@example.org",
           "INSTRUCTOR_NOTIFICATION_EMAIL": "owner@example.edu, second@example.org"}


@pytest.fixture()
def app_env(monkeypatch, tmp_path):
    import streamlit as st

    RecordingSMTP.sent = []
    RecordingSMTP.plan = []
    monkeypatch.chdir(tmp_path)
    app = _load_app()
    monkeypatch.setattr(smtplib, "SMTP", RecordingSMTP)
    monkeypatch.setattr(smtplib, "SMTP_SSL", RecordingSMTP)
    monkeypatch.setattr(st, "secrets", dict(SECRETS))
    monkeypatch.setattr(st, "session_state", {"team_name": "Team 7", "team_members_raw": "Ann\nBen"})
    monkeypatch.setattr(app, "EMAIL_DELIVERY_LOG", tmp_path / "email_delivery_log.jsonl")
    return app, st, tmp_path


def _metadata():
    return {"sample_size": 80, "conditions": ["A", "B"], "open_ended_questions": [], "simulation_mode": "pilot",
            "generation_method_label": "ABE 3.0", "run_id": "R1", "generation_timestamp": "2026-10-06T10:00:00",
            "effect_sizes_observed": [{"variable": "DV_mean", "condition_1": "A", "condition_2": "B", "cohens_d": 0.41}]}


def _wait_for_threads():
    for thread in threading.enumerate():
        if thread.name == "instructor-email":
            thread.join(30)


def _attachment_types(msg):
    return {p.get_filename(): p.get_content_type() for p in msg.walk() if p.get_filename()}


def _last_log_entries(tmp, count=1):
    lines = (tmp / "email_delivery_log.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[-count:]]


def test_instructor_notification_goes_out_in_a_thread_as_summary_plus_attachments(app_env):
    app, _st, tmp = app_env
    files = {"Simulated_Data.csv": b"a,b\n1,2\n"}
    thread = app._notify_instructor(
        title="Coffee study", metadata=_metadata(), files=files, zip_bytes=app._bytes_to_zip(files),
        html_bytes=b"<html>report</html>", md_bytes=b"# Analysis\nt = 2.0", summary_bytes=b"# Summary", usage_summary="USAGE")
    assert isinstance(thread, threading.Thread) and thread.daemon
    thread.join(30)
    (summary_msg, to1), (package_msg, to2) = RecordingSMTP.sent
    assert to1 == to2 == ["owner@example.edu", "second@example.org"]  # several recipients are supported
    # message 1: no attachments, the analysis is in the body, so a filter that holds attachments cannot take it away
    assert _attachment_types(summary_msg) == {}
    text = summary_msg.get_body(preferencelist=("plain",)).get_content()
    assert "t = 2.0" in text and "Team 7" in text and "DV_mean" in text
    # message 2: the attachments with real MIME types, threaded to message 1
    assert _attachment_types(package_msg) == {
        "INSTRUCTOR_Statistical_Report.html": "text/html", "INSTRUCTOR_Detailed_Analysis.md": "text/markdown",
        "simulation_output.zip": "application/zip", "User_Study_Summary.md": "text/markdown"}
    assert package_msg["In-Reply-To"] == summary_msg["Message-ID"] and package_msg["Subject"].endswith("[attachments]")
    first, second = _last_log_entries(tmp, 2)
    assert first["kind"] == "instructor_summary" and second["kind"] == "instructor_package" and first["ok"] and second["ok"]
    assert "owner@example.edu" not in json.dumps([first, second])  # addresses are masked in the log


def test_single_mode_sends_everything_in_one_message(app_env, monkeypatch):
    app, st, tmp = app_env
    monkeypatch.setattr(st, "secrets", {**SECRETS, "INSTRUCTOR_EMAIL_MODE": "single"})
    thread = app._notify_instructor(title="t", metadata=_metadata(), files={}, zip_bytes=b"PK", html_bytes=b"<html/>",
                                    md_bytes=b"# m", summary_bytes=b"s")
    thread.join(30)
    (msg, _to), = RecordingSMTP.sent
    assert len(_attachment_types(msg)) == 4 and "# m" in msg.get_body(preferencelist=("plain",)).get_content()
    assert _last_log_entries(tmp)[0]["kind"] == "instructor"


def test_a_failed_report_is_flagged_in_the_subject_and_body_of_the_instructor_email(app_env):
    app, _st, _tmp = app_env
    thread = app._notify_instructor(
        title="Coffee study", metadata=_metadata(), files={}, zip_bytes=b"PK", html_bytes=b"<html>Report Error</html>",
        md_bytes=b"# Comprehensive Report\nReport generation encountered an error", summary_bytes=b"# Summary",
        report_problem="instructor analysis: KeyError: 'DV_mean' <b>x</b>\r\nBcc: attacker@example.org")
    thread.join(30)
    summary_msg = RecordingSMTP.sent[0][0]
    subject = str(summary_msg["Subject"])
    assert subject.startswith("[REPORT ERROR] [Behavioral Simulation]") and "\n" not in subject and "\r" not in subject
    assert summary_msg["Bcc"] is None  # header injection through the error text is impossible
    text = summary_msg.get_body(preferencelist=("plain",)).get_content()
    assert "could not be built" in text and "KeyError" in text
    html_part = summary_msg.get_body(preferencelist=("html",)).get_content()
    assert "&lt;b&gt;x&lt;/b&gt;" in html_part and "<b>x</b>" not in html_part


def test_instructor_notification_records_a_visible_failure_when_smtp_is_missing(app_env, monkeypatch):
    app, st, tmp = app_env
    monkeypatch.setattr(st, "secrets", {})
    thread = app._notify_instructor(title="t", metadata=_metadata(), files={}, zip_bytes=b"PK", html_bytes=b"h",
                                    md_bytes=b"m", summary_bytes=b"s")
    thread.join(30)
    assert RecordingSMTP.sent == []
    first, second = _last_log_entries(tmp, 2)
    assert first["ok"] is False and first["error_kind"] == "config" and "SMTP_SERVER" in first["error"]
    assert second["error_kind"] == "skipped"  # no point attempting the attachment message


def test_instructor_notification_survives_transient_smtp_errors(app_env, monkeypatch):
    app, _st, tmp = app_env
    import utils.email_delivery as ed

    monkeypatch.setattr(ed.time, "sleep", lambda _s: None)
    RecordingSMTP.plan = [smtplib.SMTPServerDisconnected("dropped"), smtplib.SMTPDataError(451, b"try later")]
    thread = app._notify_instructor(title="t", metadata=_metadata(), files={}, zip_bytes=b"PK", html_bytes=b"h",
                                    md_bytes=b"m", summary_bytes=b"s")
    thread.join(60)
    assert len(RecordingSMTP.sent) == 2  # the summary needed three attempts, the package went through at once
    first, second = _last_log_entries(tmp, 2)
    assert first["ok"] and first["attempts"] == 3 and second["ok"] and second["attempts"] == 1


def test_oversized_zip_is_replaced_by_a_lean_copy_without_source_uploads(app_env):
    app, _st, tmp = app_env
    big_pdf = os.urandom(20 * 1024 * 1024)  # incompressible: a survey PDF the student uploaded
    files = {"Simulated_Data.csv": b"a,b\n1,2\n", "Source_Files/survey.pdf": big_pdf}
    zip_bytes = app._bytes_to_zip(files)
    assert len(zip_bytes) > 15 * 1024 * 1024
    thread = app._notify_instructor(title="t", metadata=_metadata(), files=files, zip_bytes=zip_bytes,
                                    html_bytes=b"<html>report</html>", md_bytes=b"m", summary_bytes=b"s")
    thread.join(60)
    _summary_msg, (msg, _to) = RecordingSMTP.sent[0], RecordingSMTP.sent[1]
    sent_zip = next(p for p in msg.walk() if p.get_filename() == "simulation_output.zip").get_payload(decode=True)
    assert len(sent_zip) < 1024 * 1024
    import io
    import zipfile
    names = zipfile.ZipFile(io.BytesIO(sent_zip)).namelist()
    assert "Simulated_Data.csv" in names and "Source_Files/survey.pdf" not in names
    assert "reduced to keep the message deliverable" in msg.get_body(preferencelist=("plain",)).get_content()
    entry = _last_log_entries(tmp)[0]
    assert entry["ok"] and entry["kind"] == "instructor_package" and entry["omitted"][0]["action"] == "replaced"


def test_student_triggered_emails_are_rate_limited_per_session(app_env, monkeypatch):
    app, st, _tmp = app_env
    monkeypatch.setattr(st, "secrets", {**SECRETS, "USER_EMAIL_MAX_PER_SESSION_PER_HOUR": 2})
    assert app._user_email_allowed()[0] and app._user_email_allowed()[0]
    allowed, reason = app._user_email_allowed()
    assert not allowed and "limit" in reason


def test_student_emails_cannot_exhaust_the_app_wide_limit_for_everyone(app_env, monkeypatch):
    app, st, _tmp = app_env
    monkeypatch.setattr(st, "secrets", {**SECRETS, "USER_EMAIL_MAX_PER_HOUR": 3, "USER_EMAIL_MAX_PER_SESSION_PER_HOUR": 99})
    import utils.email_delivery as ed

    ed._APP_LIMITERS.clear()
    results = []
    for session in range(5):  # five different sessions
        monkeypatch.setattr(st, "session_state", {})
        results.append(app._user_email_allowed()[0])
    assert results == [True, True, True, False, False]
    ed._APP_LIMITERS.clear()


def test_send_email_goes_through_the_logged_path_and_keeps_its_return_contract(app_env):
    app, _st, tmp = app_env
    ok, message = app._send_email("student@example.org", "Subject", "Body", [("results.zip", b"PK")], kind="user_zip")
    assert ok and message == "Email sent successfully!"
    entry = json.loads((tmp / "email_delivery_log.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert entry["kind"] == "user_zip" and entry["ok"]
    RecordingSMTP.plan = [smtplib.SMTPAuthenticationError(535, b"5.7.8 Username and Password not accepted")]
    ok, message = app._send_email("student@example.org", "Subject", "Body")
    assert not ok and "App Password" in message
    ok, message = app._send_email("not-an-address", "Subject", "Body")
    assert not ok and "recipient" in message.lower()


def test_admin_email_tab_shows_status_sends_a_test_and_lists_the_log(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    RecordingSMTP.sent = []
    RecordingSMTP.plan = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(smtplib, "SMTP", RecordingSMTP)
    if str(_APP_DIR) not in sys.path:
        sys.path.insert(0, str(_APP_DIR))
    at = AppTest.from_file(str(_APP_DIR / "app.py"), default_timeout=120)
    for key, value in SECRETS.items():
        at.secrets[key] = value
    at.query_params["admin"] = "1"
    at.session_state["_admin_authenticated"] = True
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    buttons = {b.key: b for b in at.button}
    assert "_admin_email_test_btn" in buttons
    next(s for s in at.selectbox if s.key == "_admin_email_test_choice").select("With a .html report")
    buttons["_admin_email_test_btn"].click()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert any("Accepted by the mail server" in s.value for s in at.success)
    (msg, to_addrs), = RecordingSMTP.sent
    assert to_addrs == ["owner@example.edu", "second@example.org"] and msg["Subject"].startswith("[Behavioral Simulation] Test email")
    assert {p.get_filename(): p.get_content_type() for p in msg.walk() if p.get_filename()} == {"test_report.html": "text/html"}
    log = (tmp_path / "data" / "email_delivery_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(log[-1])["kind"] == "test"
    at.run()  # the table of recent deliveries renders the entry
    assert not at.exception and len(at.dataframe) >= 1
