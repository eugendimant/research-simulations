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


# ---- fixes from the security review ---------------------------------------------------------------
def _clean_limiters():
    import utils.email_delivery as ed

    ed._APP_LIMITERS.clear()


def test_a_pasted_line_separator_in_the_study_title_does_not_cost_the_instructor_the_email(app_env):
    app, _st, tmp = app_env
    _clean_limiters()
    thread = app._notify_instructor(
        title="Coffee\u2028study\u2029two\x0bthree", metadata=_metadata(), files={}, zip_bytes=b"PK", html_bytes=b"<html>r</html>",
        md_bytes=b"# Analysis", summary_bytes=b"# Summary")
    thread.join(30)
    assert len(RecordingSMTP.sent) == 2  # summary and attachments message
    assert all(len(str(msg["Subject"]).splitlines()) == 1 for msg, _to in RecordingSMTP.sent)
    log = [json.loads(line) for line in (tmp / "email_delivery_log.jsonl").read_text(encoding="utf-8").split("\n") if line]
    assert [e["ok"] for e in log] == [True, True]


def test_student_summary_html_escapes_user_text_but_keeps_the_markdown_formatting(app_env):
    app, _st, _tmp = app_env
    from html.parser import HTMLParser

    markdown = ("# Study <script>alert(1)</script>\n\n| a | b |\n|---|---|\n| <img src=x onerror=alert(1)> | 2 |\n\n"
                "- item <b onclick=x()>bold</b>\n\n**strong** and `code` with a < b & c > d\n")
    page = app._markdown_to_html(markdown, title="</title><script>alert(2)</script>")
    live = []

    class Probe(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag in {"script", "img", "iframe", "form"} or any(name.startswith("on") for name, _ in attrs):
                live.append((tag, attrs))

    Probe().feed(page)
    assert live == [], live
    for expected in ("<h1>", "<table>", "<th>", "<li>", "<strong>strong</strong>", "<code>code</code>", "a &lt; b &amp; c &gt; d"):
        assert expected in page, expected
    assert "Content-Security-Policy" in page


def test_a_student_cannot_mail_the_same_person_more_than_twice_a_day_or_exceed_the_daily_budget(app_env, monkeypatch):
    app, st, _tmp = app_env
    _clean_limiters()
    monkeypatch.setattr(st, "secrets", {**SECRETS, "USER_EMAIL_MAX_PER_SESSION_PER_HOUR": 99, "USER_EMAIL_MAX_PER_HOUR": 99,
                                        "USER_EMAIL_MAX_PER_DAY": 4})
    victim = "victim@example.org"
    results = []
    for _ in range(3):
        monkeypatch.setattr(st, "session_state", {})  # a fresh session each time: a new browser tab
        results.append(app._user_email_allowed(victim)[0])
    assert results == [True, True, False]
    for address, expected in (("other1@example.org", True), ("other2@example.org", True), ("other3@example.org", False)):
        monkeypatch.setattr(st, "session_state", {})
        assert app._user_email_allowed(address)[0] is expected, address  # slots 3 and 4 of 4, then the budget is spent
    _clean_limiters()


def test_an_attachment_that_cannot_be_sent_is_reported_instead_of_claiming_success(app_env):
    app, _st, tmp = app_env
    ok, message = app._send_email("student@example.org", "Subject", "Body",
                                  [("simulation_output.zip", os.urandom(11 * 1024 * 1024))], kind="user_zip")
    assert ok is False and "too large" in message and "Download" in message
    entry = json.loads((tmp / "email_delivery_log.jsonl").read_text(encoding="utf-8").split("\n")[-2])
    assert entry["omitted"][0]["action"] == "omitted"


def test_the_emailed_student_zip_leaves_out_the_uploaded_source_files(app_env):
    app, _st, _tmp = app_env
    import io
    import zipfile

    full = app._bytes_to_zip({"Simulated_Data.csv": b"a,b\n1,2\n", "Source_Files/survey.pdf": b"%PDF big", "Source_Files/x/q.qsf": b"{}",
                              "Metadata.json": b"{}"})
    lean = app._zip_without_prefix(full, "Source_Files/")
    assert sorted(zipfile.ZipFile(io.BytesIO(lean)).namelist()) == ["Metadata.json", "Simulated_Data.csv"]
    assert zipfile.ZipFile(io.BytesIO(lean)).read("Simulated_Data.csv") == b"a,b\n1,2\n"
    no_sources = app._bytes_to_zip({"Simulated_Data.csv": b"x"})
    assert app._zip_without_prefix(no_sources, "Source_Files/") == no_sources  # nothing to remove: unchanged bytes
    assert app._zip_without_prefix(b"not a zip", "Source_Files/") == b"not a zip"  # never loses the email over this


def test_a_session_that_loops_cannot_flood_the_instructor_mailbox(app_env, monkeypatch):
    app, st, tmp = app_env
    _clean_limiters()
    monkeypatch.setattr(st, "secrets", {**SECRETS, "INSTRUCTOR_EMAIL_MAX_PER_SESSION_PER_HOUR": 1})

    def notify():
        return app._notify_instructor(title="t", metadata=_metadata(), files={}, zip_bytes=b"PK", html_bytes=b"<html/>",
                                      md_bytes=b"m", summary_bytes=b"s")

    first = notify()
    first.join(30)
    sent_after_first = len(RecordingSMTP.sent)
    assert sent_after_first == 2
    assert notify() is None and len(RecordingSMTP.sent) == sent_after_first  # the second run is not mailed
    last = json.loads((tmp / "email_delivery_log.jsonl").read_text(encoding="utf-8").split("\n")[-2])
    assert last["kind"] == "instructor_skipped" and last["error_kind"] == "rate_limited" and last["ok"] is False
    _clean_limiters()


def test_the_daily_instructor_budget_applies_across_sessions(app_env, monkeypatch):
    app, st, _tmp = app_env
    _clean_limiters()
    monkeypatch.setattr(st, "secrets", {**SECRETS, "INSTRUCTOR_EMAIL_MAX_PER_DAY": 1})
    threads = []
    for _ in range(3):
        monkeypatch.setattr(st, "session_state", {"team_name": "T"})  # three different sessions
        threads.append(app._notify_instructor(title="t", metadata=_metadata(), files={}, zip_bytes=b"PK", html_bytes=b"<html/>",
                                              md_bytes=b"m", summary_bytes=b"s"))
    assert threads[0] is not None and threads[1] is None and threads[2] is None
    threads[0].join(30)
    assert len(RecordingSMTP.sent) == 2
    _clean_limiters()


def test_the_right_access_code_always_works_and_wrong_guesses_are_counted_once_each(app_env, monkeypatch):
    app, st, _tmp = app_env
    _clean_limiters()
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    right = "adm" + "in-code-" + "9x"
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD_SHA256", raising=False)
    monkeypatch.setattr(st, "secrets", {"ADMIN_PASSWORD": right})
    for _ in range(30):  # Streamlit re-evaluates the same wrong text on every rerun: one guess, not thirty
        assert app._access_code_matches("same-wrong-text", "ADMIN_PASSWORD") is False
    assert app._wrong_access_guesses_last_day()["ADMIN_PASSWORD"] == 1 and sleeps == []
    for i in range(40):  # many distinct wrong guesses from many sessions
        monkeypatch.setattr(st, "session_state", {})
        assert app._access_code_matches(f"wrong-{i}", "ADMIN_PASSWORD") is False
    assert app._wrong_access_guesses_last_day()["ADMIN_PASSWORD"] == 41
    assert sleeps and max(sleeps) <= 2.0  # friction for a guessing run, never a refusal
    assert app._access_code_matches(right, "ADMIN_PASSWORD") is True  # the owner is never locked out
    assert app._wrong_access_guesses_last_day()["ANALYTICS_DASHBOARD_PASSWORD"] == 0  # gates are counted separately
    _clean_limiters()


def test_one_students_wrong_analytics_code_does_not_affect_the_owners_admin_login(app_env, monkeypatch):
    app, st, _tmp = app_env
    _clean_limiters()
    monkeypatch.setattr("time.sleep", lambda _s: None)
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("ANALYTICS_DASHBOARD_PASSWORD", raising=False)
    right = "owner-" + "pass-1"
    monkeypatch.setattr(st, "secrets", {"ADMIN_PASSWORD": right, "ANALYTICS_DASHBOARD_PASSWORD": "dash-" + "board-1"})
    for _ in range(200):  # the analytics field keeps its wrong text across 200 interactions
        app._access_code_matches("typo", "ANALYTICS_DASHBOARD_PASSWORD")
    assert app._access_code_matches(right, "ADMIN_PASSWORD") is True
    _clean_limiters()


def test_equivalent_gmail_spellings_share_one_recipient_budget(app_env):
    app, _st, _tmp = app_env
    forms = ["victim.person@gmail.com", "victimperson+a@gmail.com", "Victim.Person@GoogleMail.com", "v.i.c.t.i.m.person@gmail.com"]
    assert len({app._canonical_mailbox(f) for f in forms}) == 1
    assert app._canonical_mailbox("Ann+x@Example.org") == "ann@example.org"  # tags are dropped elsewhere too, dots are kept


def test_an_error_while_preparing_the_instructor_mail_is_a_visible_log_row(app_env):
    app, _st, tmp = app_env
    _clean_limiters()
    bad_metadata = {**_metadata(), "conditions": 5, "open_ended_questions": 7}  # not iterables
    assert app._notify_instructor(title="t", metadata=bad_metadata, files={}, zip_bytes=b"PK", html_bytes=b"h",
                                  md_bytes=b"m", summary_bytes=b"s") is None
    entry = json.loads((tmp / "email_delivery_log.jsonl").read_text(encoding="utf-8").split("\n")[-2])
    assert entry["kind"] == "instructor_error" and entry["ok"] is False and entry["error_class"]


def test_skipped_runs_are_logged_at_most_three_times_an_hour(app_env, monkeypatch):
    app, st, tmp = app_env
    _clean_limiters()
    monkeypatch.setattr(st, "secrets", {**SECRETS, "INSTRUCTOR_EMAIL_MAX_PER_DAY": 1})
    for _ in range(20):
        monkeypatch.setattr(st, "session_state", {})
        thread = app._notify_instructor(title="t", metadata=_metadata(), files={}, zip_bytes=b"PK", html_bytes=b"h",
                                        md_bytes=b"m", summary_bytes=b"s")
        if thread is not None:
            thread.join(30)
    rows = [json.loads(line) for line in (tmp / "email_delivery_log.jsonl").read_text(encoding="utf-8").split("\n") if line]
    assert sum(1 for r in rows if r["kind"] == "instructor_skipped") == 3
    assert sum(1 for r in rows if r["kind"] in ("instructor_summary", "instructor_package")) == 2
    _clean_limiters()


def test_labels_built_from_student_text_cannot_form_links_or_images(app_env):
    app, _st, _tmp = app_env
    label = app._plain_label("[click](http://evil) ![x](http://evil/p.png) **bold** `code` <b>")
    assert not any(ch in label for ch in "[]()`*<>!") and "evil" in label  # the words remain, the syntax is gone

