"""Tests for utils/email_delivery.py: recipients, message building, size fitting, retries, logging.

Everything is offline: SMTP is replaced by small fake classes.
"""
import email
import json
import smtplib
import socket
import ssl
import sys
import threading
from email import policy
from pathlib import Path

import pytest

_APP_DIR = Path(__file__).resolve().parent.parent / "simulation_app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from utils import email_delivery as ed  # noqa: E402

FAKE_LOGIN_VALUE = "s3cret" + "-pass"  # a stand-in credential; the tests check it never reaches logs or messages
CONFIG = ed.SMTPConfig(server="smtp.example.org", port=587, username="sender@example.org",
                       password=FAKE_LOGIN_VALUE, from_email="sender@example.org", from_name="Tool")


# ---------------------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------------------
class FakeServer:
    """Records what the sender does; ``plan`` is a list of exceptions/None consumed per send."""

    instances = []

    def __init__(self, host, port, timeout=None, context=None, plan=None):
        self.host, self.port, self.context = host, port, context
        self.calls = []
        self.sent = []
        self.closed = False
        FakeServer.instances.append(self)

    def ehlo(self):
        self.calls.append("ehlo")

    def starttls(self, context=None):
        self.calls.append("starttls")

    def login(self, user, password):
        self.calls.append("login")
        if FakeServer.login_error is not None:
            raise FakeServer.login_error

    def send_message(self, msg, from_addr=None, to_addrs=None, **_kw):
        outcome = FakeServer.plan.pop(0) if FakeServer.plan else None
        if isinstance(outcome, BaseException):
            raise outcome
        FakeServer.delivered.append((msg, from_addr, list(to_addrs or [])))
        return outcome or {}

    def quit(self):
        self.calls.append("quit")

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_fake():
    FakeServer.instances = []
    FakeServer.plan = []
    FakeServer.delivered = []
    FakeServer.login_error = None
    yield


def _no_sleep(_seconds):
    pass


# ---------------------------------------------------------------------------------------
# Addresses, config, helpers
# ---------------------------------------------------------------------------------------
def test_parse_recipients_accepts_lists_names_and_separators():
    valid, invalid = ed.parse_recipients("a@x.edu, Name <b@y.com>; c@z.org\n d@w.net,a@X.edu, not-an-address, @bad.com")
    assert valid == ["a@x.edu", "b@y.com", "c@z.org", "d@w.net"]
    assert "not-an-address" in invalid or invalid  # junk is reported, not sent to
    assert ed.parse_recipients(None) == ([], [])
    assert ed.parse_recipients("") == ([], [])
    assert ed.parse_recipients(["x@y.com", "z@y.com"])[0] == ["x@y.com", "z@y.com"]
    many = ", ".join(f"u{i}@x.org" for i in range(40))
    assert len(ed.parse_recipients(many)[0]) == 10  # capped


def test_parse_recipients_rejects_header_injection_attempts():
    valid, _ = ed.parse_recipients("a@x.edu\r\nBcc: evil@x.org")
    assert all("\r" not in v and "\n" not in v and " " not in v for v in valid)


def test_mask_address_hides_the_local_part():
    assert ed.mask_address("edimant@sas.upenn.edu") == "e***t@sas.upenn.edu"
    assert ed.mask_address("ab@x.org") == "a***@x.org"
    assert ed.mask_address("garbage") == "***"


def test_load_smtp_config_parses_loosely_typed_secrets():
    secrets = {"SMTP_SERVER": " smtp.gmail.com ", "SMTP_PORT": "465", "SMTP_USERNAME": "me@x.org",
               "SMTP_PASSWORD": "pw", "SMTP_USE_TLS": "false", "EMAIL_MAX_MESSAGE_MB": "30"}
    cfg = ed.load_smtp_config(lambda name, default="": secrets.get(name, default))
    assert cfg.server == "smtp.gmail.com" and cfg.port == 465 and cfg.use_tls is False
    assert cfg.from_email == "me@x.org" and cfg.configured
    assert cfg.max_message_bytes == 30 * 1024 * 1024
    broken = ed.load_smtp_config(lambda name, default="": {"SMTP_PORT": "abc"}.get(name, default))
    assert broken.port == 587 and not broken.configured
    assert "pw" not in json.dumps(cfg.public_summary())  # the display form never carries the password


def test_guess_mime_type_uses_real_types_not_octet_stream():
    assert ed.guess_mime_type("report.html") == ("text", "html")
    assert ed.guess_mime_type("pack.ZIP") == ("application", "zip")
    assert ed.guess_mime_type("notes.md") == ("text", "markdown")
    assert ed.guess_mime_type("run.sps") == ("text", "plain")
    assert ed.guess_mime_type("weird.xyz") == ("application", "octet-stream")


# ---------------------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------------------
def _msg(**kw):
    base = dict(from_email="sender@example.org", from_name="Tool", recipients=["owner@example.edu"], subject="Subject",
                body_text="Body text")
    base.update(kw)
    return ed.build_message(**base)


def test_build_message_has_the_headers_mail_filters_expect():
    msg = _msg(body_html="<p>Hi</p>")
    parsed = email.message_from_bytes(msg.as_bytes(), policy=policy.default)
    assert parsed["Message-ID"].startswith("<") and parsed["Message-ID"].endswith("@example.org>")
    assert parsed["Date"] and parsed["MIME-Version"] == "1.0"
    assert parsed["Auto-Submitted"] == "auto-generated"
    assert parsed["To"] == "owner@example.edu" and "sender@example.org" in parsed["From"]
    kinds = [p.get_content_type() for p in parsed.walk()]
    assert "text/plain" in kinds and "text/html" in kinds


def test_build_message_attachment_types_and_round_trip():
    zip_bytes, html_bytes, md_bytes = b"PK\x03\x04" + b"\x00" * 50, "<html>ü</html>".encode(), b"# title\n"
    msg = _msg(attachments=[ed.Attachment("simulation_output.zip", zip_bytes),
                            ed.Attachment("INSTRUCTOR_Statistical_Report.html", html_bytes),
                            ed.Attachment("INSTRUCTOR_Detailed_Analysis.md", md_bytes)])
    parsed = email.message_from_bytes(msg.as_bytes(), policy=policy.default)
    got = {p.get_filename(): (p.get_content_type(), p.get_payload(decode=True)) for p in parsed.walk() if p.get_filename()}
    assert got["simulation_output.zip"] == ("application/zip", zip_bytes)
    assert got["INSTRUCTOR_Statistical_Report.html"] == ("text/html", html_bytes)
    assert got["INSTRUCTOR_Detailed_Analysis.md"] == ("text/markdown", md_bytes)


def test_build_message_neutralises_header_injection_and_odd_filenames():
    evil = "Study\r\nBcc: attacker@example.com\r\nX-Evil: 1"
    msg = _msg(subject=evil, attachments=[ed.Attachment("../../etc/passwd\r\n.html", b"x")])
    raw = msg.as_bytes().decode("utf-8", "replace")
    head = raw.split("\r\n\r\n", 1)[0]
    assert "\nBcc:" not in head and "\nX-Evil:" not in head
    parsed = email.message_from_bytes(msg.as_bytes(), policy=policy.default)
    assert parsed["Bcc"] is None
    names = [p.get_filename() for p in parsed.walk() if p.get_filename()]
    assert names and all("/" not in n and "\n" not in n and ".." not in n.split(".")[0:1] for n in names)


def test_build_message_handles_unicode_subject_and_sender_name():
    msg = _msg(subject="Étude – α/β ✓", from_name="Outil d'étude")
    parsed = email.message_from_bytes(msg.as_bytes(), policy=policy.default)
    assert parsed["Subject"] == "Étude – α/β ✓"


# ---------------------------------------------------------------------------------------
# Size fitting
# ---------------------------------------------------------------------------------------
MB = 1024 * 1024


def _att(name, mb, **kw):
    return ed.Attachment(name, b"x" * int(mb * MB), **kw)


def test_fit_attachments_leaves_everything_when_it_fits():
    slots = [_att("a.html", 1), _att("b.md", 0.1), _att("c.zip", 2)]
    kept, notes = ed.fit_attachments(slots, 15 * MB)
    assert [a.name for a in kept] == ["a.html", "b.md", "c.zip"] and notes == []


def test_fit_attachments_swaps_the_big_zip_for_its_lean_version_and_keeps_small_files():
    lean = _att("simulation_output.zip", 1)
    slots = [_att("report.html", 1, protected=True), _att("analysis.md", 0.2),
             _att("simulation_output.zip", 30, alternatives=(lean,)), _att("summary.md", 0.05)]
    kept, notes = ed.fit_attachments(slots, 15 * MB)
    names = {a.name: len(a.data) for a in kept}
    assert names["simulation_output.zip"] == len(lean.data)
    assert "analysis.md" in names and "summary.md" in names and "report.html" in names
    assert [n["action"] for n in notes] == ["replaced"]


def test_fit_attachments_drops_when_no_alternative_and_never_drops_protected():
    slots = [_att("report.html", 20, protected=True), _att("zip.zip", 20)]
    kept, notes = ed.fit_attachments(slots, 15 * MB)
    assert [a.name for a in kept] == ["report.html"]  # protected stays even though it alone is too big
    assert notes == [{"name": "zip.zip", "action": "omitted", "bytes": 20 * MB}]


def test_fit_attachments_prefers_dropping_big_low_priority_over_small_ones():
    slots = [_att("report.html", 1, protected=True), _att("small.md", 0.01), _att("huge.bin", 40), _att("tiny.md", 0.01)]
    kept, notes = ed.fit_attachments(slots, 10 * MB)
    assert {a.name for a in kept} == {"report.html", "small.md", "tiny.md"}
    assert notes[0]["name"] == "huge.bin"


# ---------------------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("exc, kind", [
    (smtplib.SMTPAuthenticationError(535, b"5.7.8 Username and Password not accepted"), "auth"),
    (smtplib.SMTPAuthenticationError(454, b"4.7.0 Temporary authentication failure"), "transient"),
    (smtplib.SMTPDataError(552, b"5.3.4 Message size exceeds fixed limit"), "size"),
    (smtplib.SMTPDataError(552, b"5.2.3 Your message exceeded Google's message size limits"), "size"),
    (smtplib.SMTPDataError(554, b"5.3.4 message too large"), "size"),
    (smtplib.SMTPDataError(451, b"4.3.0 Temporary local problem"), "transient"),
    (smtplib.SMTPDataError(550, b"5.4.5 Daily user sending quota exceeded."), "quota"),
    (smtplib.SMTPAuthenticationError(454, b"4.7.0 Too many login attempts, please try again later."), "quota"),
    (smtplib.SMTPSenderRefused(421, b"4.7.0 Try again later, closing connection. rate limit", "a@x.org"), "quota"),
    (smtplib.SMTPDataError(552, b"5.2.2 The email account that you tried to reach is over quota."), "recipient"),
    (smtplib.SMTPDataError(552, b""), "size"),
    (smtplib.SMTPSenderRefused(550, b"5.7.1 not allowed", "a@x.org"), "permanent"),
    (smtplib.SMTPRecipientsRefused({"a@x.org": (550, b"5.1.1 no such user")}), "recipient"),
    (smtplib.SMTPServerDisconnected("Connection unexpectedly closed"), "transient"),
    (smtplib.SMTPConnectError(421, b"4.3.2 service not available"), "transient"),
    (smtplib.SMTPNotSupportedError("STARTTLS extension not supported by server."), "config"),
    (smtplib.SMTPException("SMTP AUTH extension not supported by server."), "config"),
    (socket.timeout("timed out"), "transient"),
    (ConnectionRefusedError("refused"), "transient"),
    (ssl.SSLCertVerificationError("bad cert"), "config"),
    (ValueError("boom"), "permanent"),
])
def test_classify_error(exc, kind):
    assert ed.classify_error(exc)[0] == kind


# ---------------------------------------------------------------------------------------
# Sending with retries
# ---------------------------------------------------------------------------------------
def _send(config=CONFIG, recipients=("owner@example.edu",), **kw):
    msg = _msg(recipients=list(recipients))
    return ed.send_with_retries(config, msg, list(recipients), smtp_factory=FakeServer, sleep=_no_sleep, **kw)


def test_send_succeeds_and_uses_starttls_on_587_ssl_on_465():
    result = _send()
    assert result.ok and result.attempts == 1 and result.accepted == ["owner@example.edu"]
    server = FakeServer.instances[0]
    assert server.calls == ["ehlo", "starttls", "ehlo", "login", "quit"]
    FakeServer.instances.clear()
    ssl_cfg = ed.SMTPConfig(server="s", port=465, username="u", password=FAKE_LOGIN_VALUE, from_email="u@x.org")
    result = ed.send_with_retries(ssl_cfg, _msg(), ["owner@example.edu"], smtp_factory=FakeServer, sleep=_no_sleep)
    assert result.ok and "starttls" not in FakeServer.instances[0].calls
    no_tls = ed.SMTPConfig(server="s", port=25, username="u", password=FAKE_LOGIN_VALUE, from_email="u@x.org", use_tls=False)
    FakeServer.instances.clear()
    ed.send_with_retries(no_tls, _msg(), ["owner@example.edu"], smtp_factory=FakeServer, sleep=_no_sleep)
    assert "starttls" not in FakeServer.instances[0].calls


def test_transient_failures_are_retried_with_backoff_then_succeed():
    FakeServer.plan = [smtplib.SMTPServerDisconnected("gone"), smtplib.SMTPDataError(451, b"try later"), None]
    waits = []
    msg = _msg()
    result = ed.send_with_retries(CONFIG, msg, ["owner@example.edu"], smtp_factory=FakeServer, sleep=waits.append)
    assert result.ok and result.attempts == 3
    assert waits == [2.0, 6.0]
    assert len(FakeServer.delivered) == 1


def test_permanent_failures_are_not_retried():
    FakeServer.plan = [smtplib.SMTPDataError(552, b"5.3.4 Message size exceeds fixed limit")]
    result = _send()
    assert not result.ok and result.attempts == 1 and result.error_kind == "size" and result.smtp_code == 552
    FakeServer.plan = []
    FakeServer.login_error = smtplib.SMTPAuthenticationError(535, b"5.7.8 Username and Password not accepted")
    result = _send()
    assert not result.ok and result.error_kind == "auth" and "App Password" in result.message
    assert all(inst.closed for inst in FakeServer.instances)  # half-open connections are closed


def test_gives_up_after_max_attempts_on_persistent_transient_errors():
    FakeServer.plan = [smtplib.SMTPServerDisconnected("gone")] * 5
    result = _send()
    assert not result.ok and result.attempts == 3 and result.error_kind == "transient"


def test_partial_recipient_refusal_still_counts_as_delivered():
    FakeServer.plan = [{"bad@x.org": (550, b"5.1.1 no such user")}]
    result = _send(recipients=("good@x.org", "bad@x.org"))
    assert result.ok and result.accepted == ["good@x.org"] and "bad@x.org" in result.refused


def test_all_recipients_refused_is_a_permanent_recipient_error():
    FakeServer.plan = [smtplib.SMTPRecipientsRefused({"a@x.org": (550, b"5.1.1 no such user")})]
    result = _send()
    assert not result.ok and result.error_kind == "recipient" and result.attempts == 1


def test_error_detail_never_contains_the_password():
    FakeServer.login_error = smtplib.SMTPAuthenticationError(535, ("bad login for " + FAKE_LOGIN_VALUE).encode())
    result = _send()
    assert FAKE_LOGIN_VALUE not in result.error_detail


# ---------------------------------------------------------------------------------------
# deliver(): the single entry point
# ---------------------------------------------------------------------------------------
def test_deliver_reports_missing_configuration_and_recipients_without_raising(tmp_path):
    log = tmp_path / "log.jsonl"
    res = ed.deliver(ed.SMTPConfig(), ["a@x.org"], "s", "b", log_path=log)
    assert not res.ok and res.error_kind == "config" and "not configured" in res.message
    res = ed.deliver(CONFIG, [], "s", "b", log_path=log, smtp_factory=FakeServer)
    assert not res.ok and res.error_kind == "recipient"
    entries = ed.read_delivery_log(log)
    assert len(entries) == 2 and entries[0]["error_kind"] == "recipient"


def test_deliver_logs_a_masked_entry_without_secrets(tmp_path):
    log = tmp_path / "log.jsonl"
    res = ed.deliver(CONFIG, ["owner@example.edu"], "Subject", "Body", kind="instructor", log_path=log,
                     attachments=[ed.Attachment("a.zip", b"1234")], smtp_factory=FakeServer, sleep=_no_sleep)
    assert res.ok
    raw = log.read_text(encoding="utf-8")
    assert "owner@example.edu" not in raw and "o***r@example.edu" in raw and FAKE_LOGIN_VALUE not in raw
    entry = ed.read_delivery_log(log)[0]
    assert entry["ok"] and entry["kind"] == "instructor" and entry["attachments"] == [{"name": "a.zip", "bytes": 4}]
    assert entry["message_id"].startswith("<") and entry["message_bytes"] > 0


def test_deliver_swaps_oversized_zip_for_the_lean_one_and_says_so(tmp_path):
    lean = _att("simulation_output.zip", 0.5)
    big = _att("simulation_output.zip", 40, alternatives=(lean,))
    res = ed.deliver(CONFIG, ["owner@example.edu"], "S", "Body", body_html="<p>Body</p>", log_path=tmp_path / "l.jsonl",
                     attachments=[_att("report.html", 1, protected=True), big], smtp_factory=FakeServer, sleep=_no_sleep)
    assert res.ok and res.omitted and res.omitted[0]["action"] == "replaced"
    sent = FakeServer.delivered[0][0]
    text = sent.get_body(preferencelist=("plain",)).get_content()
    html = sent.get_body(preferencelist=("html",)).get_content()
    assert text.startswith("NOTE: some attachments were reduced") and "NOTE: some attachments" in html
    sizes = {p.get_filename(): len(p.get_payload(decode=True)) for p in sent.walk() if p.get_filename()}
    assert sizes["simulation_output.zip"] == len(lean.data)


def test_deliver_halves_the_budget_once_when_the_server_reports_size_too_large(tmp_path):
    FakeServer.plan = [smtplib.SMTPDataError(552, b"5.3.4 Message size exceeds fixed limit"), None]
    small_alt = _att("data.zip", 0.2)
    res = ed.deliver(CONFIG, ["owner@example.edu"], "S", "Body", log_path=tmp_path / "l.jsonl",
                     attachments=[_att("report.html", 0.5, protected=True), _att("data.zip", 6, alternatives=(small_alt,))],
                     smtp_factory=FakeServer, sleep=_no_sleep)
    assert res.ok and len(FakeServer.delivered) == 1
    assert any(n["name"] == "data.zip" for n in res.omitted)  # the retry used a reduced attachment set


def test_deliver_never_raises_even_on_unexpected_errors(tmp_path):
    class Exploding:
        def __init__(self, *a, **k):
            raise RuntimeError("kaboom with " + FAKE_LOGIN_VALUE + " inside")

    res = ed.deliver(CONFIG, ["owner@example.edu"], "S", "B", log_path=tmp_path / "l.jsonl", smtp_factory=Exploding,
                     sleep=_no_sleep, max_attempts=1)
    assert not res.ok
    assert FAKE_LOGIN_VALUE not in json.dumps(ed.read_delivery_log(tmp_path / "l.jsonl"))


# ---------------------------------------------------------------------------------------
# Log handling and background execution
# ---------------------------------------------------------------------------------------
def test_log_is_trimmed_and_read_newest_first(tmp_path):
    log = tmp_path / "log.jsonl"
    res = ed.DeliveryResult(ok=True, accepted=["a@x.org"], attempts=1)
    long_subject = "x" * 150
    for i in range(1200):
        ed.record_delivery(log, res, kind=f"k{i}", subject=long_subject, recipients=["a@x.org"])
    assert log.stat().st_size < 500_000
    newest = ed.read_delivery_log(log, limit=3)
    assert [e["kind"] for e in newest] == ["k1199", "k1198", "k1197"]
    assert ed.read_delivery_log(tmp_path / "missing.jsonl") == []


def test_record_delivery_survives_an_unwritable_path():
    entry = ed.record_delivery(Path("/proc/definitely/not/writable.jsonl"), ed.DeliveryResult(ok=False),
                               kind="x", subject="s", recipients=[])
    assert entry["ok"] is False


def test_run_in_background_runs_independently_and_swallows_errors():
    done = threading.Event()
    thread = ed.run_in_background(lambda: done.set(), name="t1")
    assert done.wait(5) and thread.daemon
    ed.run_in_background(lambda: 1 / 0, name="t2").join(5)  # must not raise into the caller


# ---------------------------------------------------------------------------------------
# Instructor notification content
# ---------------------------------------------------------------------------------------
def _compose(**kw):
    base = dict(title="Coffee study", team_name="Team 4", team_members="Ann\nBen", generation_label="ABE 3.0", mode="pilot",
                metadata={"sample_size": 120, "conditions": ["A", "B"], "open_ended_questions": [1, 2, 3],
                          "generation_timestamp": "2026-10-06T10:00:00", "run_id": "RUN123",
                          "effect_sizes_observed": [
                              {"variable": "Trust_1", "condition_1": "A", "condition_2": "B", "cohens_d": 0.1},
                              {"variable": "Trust_mean", "condition_1": "A", "condition_2": "B", "cohens_d": -0.62},
                              {"variable": "Liking_mean", "condition_1": "A", "condition_2": "B", "cohens_d": 0.31}],
                          "exclusion_summary": {"flagged_speed": 3, "total_excluded": 5}},
                usage_summary="USAGE: 12 simulations", analysis_markdown="# Analysis\nt(118) = 2.1, p = .04",
                attachment_names=["INSTRUCTOR_Statistical_Report.html", "simulation_output.zip"], zip_listing=["Simulated_Data.csv"])
    base.update(kw)
    return ed.compose_instructor_notification(**base)


def test_instructor_notification_has_headline_numbers_and_the_full_analysis():
    subject, text, html = _compose()
    assert subject == "[Behavioral Simulation] Output (pilot) [ABE 3.0] - Coffee study"
    assert "N=120" in text and "RUN123" in text and "total_excluded=5" in text
    assert text.index("Trust_mean") < text.index("Liking_mean")  # scale means, largest |d| first
    assert "Trust_1" not in text  # single items are left out when scale means exist
    assert "t(118) = 2.1, p = .04" in text and "t(118) = 2.1, p = .04" in html
    assert "INSTRUCTOR_Statistical_Report.html" in text and "USAGE: 12 simulations" in text


def test_instructor_notification_escapes_user_text_and_cleans_the_subject():
    subject, text, html = _compose(title="<script>alert(1)</script>\r\nBcc: x@y.z", analysis_markdown="a < b & c > d")
    assert "\n" not in subject and "\r" not in subject
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "a &lt; b &amp; c &gt; d" in html and "a < b & c > d" in text


def test_instructor_notification_truncates_very_long_analyses_and_survives_empty_metadata():
    _, text, _ = _compose(analysis_markdown="x" * 5000, **{"max_inline_chars": 1000})
    assert "truncated in the email body" in text and text.count("x") < 1200
    subject, text, html = ed.compose_instructor_notification(
        title="t", team_name="", team_members="", generation_label="m", mode="", metadata={}, usage_summary="",
        analysis_markdown="", attachment_names=[])
    assert subject.startswith("[Behavioral Simulation] Output (pilot)") and "<html>" in html


# ---------------------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------------------
def test_rate_limiter_is_a_sliding_window_per_key():
    now = [0.0]
    limiter = ed.RateLimiter(max_events=2, window_s=60, clock=lambda: now[0])
    assert limiter.allow("a") and limiter.allow("a") and not limiter.allow("a")
    assert limiter.allow("b")  # other keys are independent
    now[0] = 61.0
    assert limiter.allow("a")  # the window slid


def test_quota_errors_are_not_retried(tmp_path):
    FakeServer.plan = [smtplib.SMTPDataError(550, b"5.4.5 Daily user sending quota exceeded.")]
    result = _send()
    assert not result.ok and result.attempts == 1 and result.error_kind == "quota" and "sending limit" in result.message


def test_shared_limiter_is_one_instance_per_name_and_follows_config_changes():
    ed._APP_LIMITERS.clear()
    a = ed.shared_limiter("x", 3)
    assert ed.shared_limiter("x", 3) is a
    b = ed.shared_limiter("x", 5)
    assert b is not a and b.max_events == 5
    ed._APP_LIMITERS.clear()
