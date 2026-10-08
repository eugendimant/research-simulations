"""Regression tests for four findings of the v1.3.0.4 audit (one section per finding).

1. ``app._secret`` read ``st.secrets`` only, although docs/DEPLOYMENT_SECRETS.md promises
   "environment variables or Streamlit secrets": SMTP settings, the instructor address and the
   email limits set as environment variables were silently ignored.
2. The root README still named the Cerebras and Mistral keys after both providers left the chain.
3. ``utils.llm_response_generator`` had no stand-ins for two of the nine helpers it takes from
   ``utils.text_cleanup`` (``NameError`` at the first post-processed answer) and none at all when
   that module cannot be imported.
4. ``EnhancedSimulationEngine._is_control_arm`` matched control words as substrings, so "Unusual
   outcome", "Standardized message" and "Uncontrolled spending" counted as reference arms.

Offline and fast: the app module is executed once from a temporary working directory, SMTP is a
recording fake, and the import-failure tests run in subprocesses so they cannot leave anything
behind in ``sys.modules``.
"""
from __future__ import annotations

import importlib.util
import json
import re
import smtplib
import subprocess
import sys
import threading
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_APP_DIR = _REPO_ROOT / "simulation_app"
_UTILS_DIR = _APP_DIR / "utils"
FAKE_LOGIN_VALUE = "login" + "-value"  # concatenated: not a literal password assignment for the linters


# =====================================================================================================
# 1. _secret reads the environment first, then st.secrets (docs/DEPLOYMENT_SECRETS.md)
# =====================================================================================================
_SECRET_NAMES = (
    "SMTP_SERVER", "SMTP_PORT", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_FROM_EMAIL", "SMTP_FROM_NAME",
    "SMTP_USE_TLS", "EMAIL_MAX_MESSAGE_MB", "INSTRUCTOR_NOTIFICATION_EMAIL", "INSTRUCTOR_EMAIL_MODE",
    "USER_EMAIL_MAX_PER_SESSION_PER_HOUR", "USER_EMAIL_MAX_PER_HOUR", "USER_EMAIL_MAX_PER_DAY",
    "INSTRUCTOR_EMAIL_MAX_PER_SESSION_PER_HOUR", "INSTRUCTOR_EMAIL_MAX_PER_DAY", "AUDIT_PROBE_SECRET",
)


@pytest.fixture(scope="module")
def app_module(tmp_path_factory):
    """app.py executed once, the way tests/test_email_app_integration_v1291.py loads it.

    The working directory is temporary while it runs, so no ``data/`` folder lands in the repository.
    """
    if str(_APP_DIR) not in sys.path:
        sys.path.insert(0, str(_APP_DIR))
    name = "_app_audit_fixes_v1304"
    spec = importlib.util.spec_from_file_location(name, str(_APP_DIR / "app.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(tmp_path_factory.mktemp("audit_app_cwd"))
        try:
            spec.loader.exec_module(module)
        except SystemExit:  # app.py reaches the Streamlit runtime at the end of the import
            pass
    yield module
    sys.modules.pop(name, None)


@pytest.fixture()
def secrets_env(app_module, monkeypatch, tmp_path):
    """The app module with none of the deployment secrets in the environment and an empty ``st.secrets``."""
    import streamlit as st
    import utils.email_delivery as email_delivery

    for name in _SECRET_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(st, "secrets", {})
    monkeypatch.setattr(st, "session_state", {"team_name": "Team 7", "team_members_raw": "Ann\nBen"})
    monkeypatch.setattr(app_module, "EMAIL_DELIVERY_LOG", tmp_path / "email_delivery_log.jsonl")
    email_delivery._APP_LIMITERS.clear()
    yield app_module, st
    email_delivery._APP_LIMITERS.clear()


class RecordingSMTP:
    """Replacement for ``smtplib.SMTP``: records the connection, whether STARTTLS ran and every message."""

    instances: List["RecordingSMTP"] = []

    def __init__(self, host, port, timeout=None, context=None):
        self.host, self.port, self.tls_started, self.sent = host, port, False, []
        RecordingSMTP.instances.append(self)

    def ehlo(self):
        pass

    def starttls(self, context=None):
        self.tls_started = True

    def login(self, user, password):
        self.login_user = user

    def send_message(self, msg, from_addr=None, to_addrs=None, **_kw):
        self.sent.append((msg, list(to_addrs or [])))
        return {}

    def quit(self):
        pass

    def close(self):
        pass


@pytest.fixture()
def smtp_recorder(monkeypatch):
    RecordingSMTP.instances = []
    monkeypatch.setattr(smtplib, "SMTP", RecordingSMTP)
    monkeypatch.setattr(smtplib, "SMTP_SSL", RecordingSMTP)
    return RecordingSMTP


def _set_smtp_environment(monkeypatch, **extra: str) -> None:
    for name, value in {"SMTP_SERVER": "smtp.example.org", "SMTP_USERNAME": "sender@example.org",
                        "SMTP_PASSWORD": FAKE_LOGIN_VALUE, **extra}.items():
        monkeypatch.setenv(name, value)


def test_smtp_settings_given_only_as_environment_variables_are_used(secrets_env, monkeypatch):
    app, _st = secrets_env
    assert app._email_delivery.load_smtp_config(app._secret).configured is False  # nothing anywhere yet
    _set_smtp_environment(monkeypatch, SMTP_PORT="2525", SMTP_USE_TLS="false", SMTP_FROM_EMAIL="from@example.org")
    config = app._email_config()
    assert config.configured is True
    assert (config.server, config.port, config.username) == ("smtp.example.org", 2525, "sender@example.org")
    assert config.password == FAKE_LOGIN_VALUE and config.from_email == "from@example.org"
    assert config.use_tls is False  # the text "false" from the environment is not "truthy"


def test_st_secrets_still_work_when_the_environment_is_silent(secrets_env, monkeypatch):
    app, st = secrets_env
    monkeypatch.setattr(st, "secrets", {"SMTP_SERVER": "smtp.example.net", "SMTP_PORT": 465, "SMTP_USERNAME": "u@example.net",
                                        "SMTP_PASSWORD": FAKE_LOGIN_VALUE, "SMTP_USE_TLS": False})
    config = app._email_config()
    assert config.configured is True
    assert (config.server, config.port, config.use_tls) == ("smtp.example.net", 465, False)  # types are kept


@pytest.mark.parametrize("environment, secrets, expected", [
    ("from-env", "from-secrets", "from-env"),        # the environment wins
    ("", "from-secrets", "from-secrets"),            # an empty variable is "not set"
    ("   ", "from-secrets", "from-secrets"),         # so is a blank one
    ("  padded  ", "from-secrets", "padded"),        # a used value is stripped
    (None, "from-secrets", "from-secrets"),
    (None, None, "the-default"),
])
def test_secret_precedence_is_environment_then_streamlit_secrets_then_default(
        secrets_env, monkeypatch, environment, secrets, expected):
    app, st = secrets_env
    if environment is not None:
        monkeypatch.setenv("AUDIT_PROBE_SECRET", environment)
    monkeypatch.setattr(st, "secrets", {} if secrets is None else {"AUDIT_PROBE_SECRET": secrets})
    assert app._secret("AUDIT_PROBE_SECRET", "the-default") == expected


def test_secret_never_raises_and_keeps_the_type_of_the_default(secrets_env, monkeypatch):
    app, st = secrets_env

    class Unreadable:
        def get(self, name, default=None):
            raise FileNotFoundError("no secrets.toml")

    monkeypatch.setattr(st, "secrets", Unreadable())  # what Streamlit does on a local run without a secrets file
    assert app._secret("AUDIT_PROBE_SECRET", 587) == 587
    assert app._secret("AUDIT_PROBE_SECRET") == ""
    monkeypatch.setenv("AUDIT_PROBE_SECRET", "7")
    assert app._secret("AUDIT_PROBE_SECRET", 587) == "7"  # an environment value is text; callers coerce


def test_instructor_address_and_email_limits_can_be_set_as_environment_variables(secrets_env, monkeypatch):
    app, _st = secrets_env
    assert app._instructor_recipients()[0] == [app._DEFAULT_INSTRUCTOR_EMAIL]
    monkeypatch.setenv("INSTRUCTOR_NOTIFICATION_EMAIL", "owner@example.edu, second@example.org")
    assert app._instructor_recipients() == (["owner@example.edu", "second@example.org"], [])
    # student mail: one message per session and hour
    monkeypatch.setenv("USER_EMAIL_MAX_PER_SESSION_PER_HOUR", "1")
    assert app._user_email_allowed()[0] is True
    allowed, reason = app._user_email_allowed()
    assert allowed is False and "limit" in reason
    # instructor mail: one run per session and hour, and a mistyped limit falls back to the default
    monkeypatch.setenv("INSTRUCTOR_EMAIL_MAX_PER_SESSION_PER_HOUR", "1")
    assert app._instructor_email_blocked() == ""
    assert "INSTRUCTOR_EMAIL_MAX_PER_SESSION_PER_HOUR" in app._instructor_email_blocked()
    monkeypatch.setenv("USER_EMAIL_MAX_PER_HOUR", "plenty")
    assert app._user_email_allowed()[0] is True  # no exception; the default limits (5 per session and hour) apply


def test_the_instructor_mail_is_sent_when_smtp_is_configured_only_through_the_environment(
        secrets_env, monkeypatch, smtp_recorder):
    app, _st = secrets_env
    _set_smtp_environment(monkeypatch, INSTRUCTOR_NOTIFICATION_EMAIL="owner@example.edu, second@example.org")
    metadata = {"sample_size": 80, "conditions": ["A", "B"], "open_ended_questions": [], "simulation_mode": "pilot",
                "generation_method_label": "ABE 3.0", "run_id": "R1", "generation_timestamp": "2026-10-06T10:00:00"}
    thread = app._notify_instructor(title="Coffee study", metadata=metadata, files={}, zip_bytes=b"PK",
                                    html_bytes=b"<html>report</html>", md_bytes=b"# Analysis", summary_bytes=b"# Summary")
    assert isinstance(thread, threading.Thread)
    thread.join(30)
    messages = [item for smtp in smtp_recorder.instances for item in smtp.sent]
    assert len(messages) == 2, "summary message and attachment message"
    assert all(to == ["owner@example.edu", "second@example.org"] for _msg, to in messages)
    assert all(smtp.host == "smtp.example.org" and smtp.tls_started for smtp in smtp_recorder.instances)


def test_the_legacy_sender_reads_the_environment_and_understands_false(secrets_env, monkeypatch, smtp_recorder):
    app, _st = secrets_env
    _set_smtp_environment(monkeypatch, SMTP_USE_TLS="false", SMTP_PORT="not-a-number")
    ok, _message = app._send_email_with_smtp_legacy("student@example.org", "Subject", "Body")
    assert ok is True
    (smtp,) = smtp_recorder.instances
    assert smtp.host == "smtp.example.org" and smtp.port == 587  # a mistyped port falls back, it does not raise
    assert smtp.tls_started is False and len(smtp.sent) == 1  # "false" switches STARTTLS off
    monkeypatch.delenv("SMTP_USE_TLS")
    smtp_recorder.instances = []
    app._send_email_with_smtp_legacy("student@example.org", "Subject", "Body")
    assert smtp_recorder.instances[0].tls_started is True  # the default is on


@pytest.mark.parametrize("raw", ["false", "FALSE", " No ", "off", "0", "true", "TRUE", "1", "yes", "banana", ""])
@pytest.mark.parametrize("default", [True, False])
def test_secret_bool_gives_the_same_answers_as_the_email_modules_rule(secrets_env, monkeypatch, raw, default):
    app, st = secrets_env
    expected = app._email_delivery._as_bool(raw, default)
    monkeypatch.setenv("AUDIT_PROBE_SECRET", raw)
    assert app._secret_bool("AUDIT_PROBE_SECRET", default) is expected
    monkeypatch.delenv("AUDIT_PROBE_SECRET")
    monkeypatch.setattr(st, "secrets", {"AUDIT_PROBE_SECRET": raw})  # a quoted TOML value is text too
    assert app._secret_bool("AUDIT_PROBE_SECRET", default) is expected


def test_secret_bool_keeps_real_booleans_from_st_secrets(secrets_env, monkeypatch):
    app, st = secrets_env
    for value in (True, False):
        monkeypatch.setattr(st, "secrets", {"AUDIT_PROBE_SECRET": value})
        assert app._secret_bool("AUDIT_PROBE_SECRET", not value) is value


# =====================================================================================================
# 2. The README names only secrets the code reads
# =====================================================================================================
def _known_secret_names() -> set:
    from utils.llm_response_generator import BUILTIN_PROVIDER_SECRETS

    return {name for names in BUILTIN_PROVIDER_SECRETS.values() for name in names}


@pytest.mark.parametrize("relative_path", ["README.md", "docs/DEPLOYMENT_SECRETS.md", "secrets.toml.example"])
def test_provider_secret_names_in_the_docs_are_names_the_code_reads(relative_path):
    """Cerebras and Mistral left the chain in v1.3.0.0; a README that still tells people to create
    their keys sends them to providers whose keys the app never reads."""
    text = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")
    documented = set(re.findall(r"\b[A-Z][A-Z0-9]*_API_KEY\b", text))
    assert documented, f"{relative_path} names no provider key any more; update this test with the README"
    unknown = documented - _known_secret_names()
    assert not unknown, (f"{relative_path} names secrets the code does not read: {sorted(unknown)}; "
                         f"the code reads {sorted(_known_secret_names())}")


# =====================================================================================================
# 3. llm_response_generator takes every helper from text_cleanup in the top-level import layout
# =====================================================================================================
_TEXT_CLEANUP_NAMES = ("apply_contractions", "drop_one_optional_word", "finalize_generated_text", "has_opener",
                       "insert_filler", "lower_first", "swap_one_word", "split_sentences", "is_probably_non_english")


def test_the_top_level_import_layout_gets_every_helper_from_text_cleanup():
    """``utils/`` on sys.path (scripts, some test layouts): the second import used to omit two names, which
    raised NameError at the first post-processed answer. A fresh interpreter keeps sys.modules clean."""
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import llm_response_generator as m; "
        "print([n for n in sys.argv[2].split(',') if not callable(getattr(m, n, None))])"
    )
    process = subprocess.run(
        [sys.executable, "-I", "-c", code, str(_UTILS_DIR), ",".join(_TEXT_CLEANUP_NAMES)],
        capture_output=True, text=True, timeout=240, check=False,
    )
    assert process.returncode == 0, process.stderr[-1500:]
    assert process.stdout.strip().splitlines()[-1] == "[]"


# =====================================================================================================
# 4. _is_control_arm matches whole words, not substrings
# =====================================================================================================
_CONTROL_LABELS = [
    "Control", "CONTROL", "control group", "Control Condition", "Control (no message)", "Norm message (Control)",
    "Healthy controls", "Control B", "Control1", "ControlGroup", "control_block", "cognitive_dissonance_control",
    "Baseline", "Behavior Baseline", "placebo", "Placebo group", "Waitlist control", "wait-list", "wait list",
    "Wait_List", "WaitList", "No treatment", "NoTreatment", "No intervention", "neutral", "Neutral AI framing",
    "Comparison group", "Treatment as usual", "Standard care", "Untreated", "None", "None AI information",
]
_TREATMENT_LABELS = [
    "Unusual outcome", "Standardized message", "Uncontrolled spending", "Usually late", "Controlling manager",
    "Neutrality violation", "Nonexistent product", "Uncontrollable stressor", "Substandard care", "UnusualOutcome",
    "StandardizedMessage", "Treatment A", "High price", "Loss frame", "", None,
]


def _engine(conditions: List[str]):
    from utils.enhanced_simulation_engine import EnhancedSimulationEngine

    return EnhancedSimulationEngine(
        study_title="Audit probe", study_description="", sample_size=20, conditions=list(conditions), factors=[],
        scales=[{"name": "attitude", "variable_name": "attitude", "num_items": 4, "scale_points": 7, "_validated": True,
                 "scale_min": 1, "scale_max": 7, "question_text": "How favourable is your attitude?",
                 "dv_description": "attitude"}],
        additional_vars=[], demographics={}, seed=5)


@pytest.fixture(scope="module")
def plain_engine():
    """``_is_control_arm`` reads no per-engine state, so one engine serves every label."""
    return _engine(["A", "B"])


@pytest.mark.parametrize("label", _CONTROL_LABELS)
def test_genuine_reference_arm_labels_are_still_control_arms(plain_engine, label):
    assert plain_engine._is_control_arm(label) is True


@pytest.mark.parametrize("label", _TREATMENT_LABELS)
def test_labels_that_only_contain_a_control_word_are_not_control_arms(plain_engine, label):
    assert plain_engine._is_control_arm(label) is False


@pytest.mark.parametrize("label, is_control", [(label, True) for label in
                                              ("Control", "control group", "Baseline", "placebo", "Healthy controls",
                                               "ControlGroup", "cognitive_dissonance_control")]
                         + [(label, False) for label in
                            ("Unusual outcome", "Standardized message", "Uncontrolled spending", "Usually late")])
def test_the_literature_fallback_skips_the_reference_arm_and_only_the_reference_arm(monkeypatch, label, is_control):
    """Call-site check: the reference arm is the zero point; any other unmatched arm takes the published effect."""
    from utils import enhanced_simulation_engine as engine_module

    lookups: List[str] = []

    def fake_lookup(condition, variable, study_context, rng=None):
        lookups.append(condition)
        return types.SimpleNamespace(effect_d=0.5, key="probe", source="probe", published_d=0.5, status="verified",
                                     as_dict=lambda: {"key": "probe"})

    monkeypatch.setattr(engine_module._literature_effects, "lookup", fake_lookup)
    engine = _engine([label, "Zeta variant"])
    # the keyword rules are not under test: pin them to "nothing matched" so only the arm decision differs
    monkeypatch.setattr(engine, "_get_automatic_condition_effect", lambda *args, **kwargs: 0.0)
    value = engine._inferred_effect_value(label, "attitude")
    if is_control:
        assert lookups == [] and value == 0.0
    else:
        assert lookups == [label] and value > 0.02


@pytest.mark.parametrize("conditions, expected", [
    # a genuine reference arm is the zero point and the other arm moves by the published effect (2 * 0.109 * d)
    (["Control", "Self-affirmation"], {"Control": 0.0, "Self-affirmation": 0.109}),
    (["ControlGroup", "Self-affirmation"], {"ControlGroup": 0.0, "Self-affirmation": 0.109}),
    (["Self-affirmation", "cognitive_dissonance_control"], {"Self-affirmation": 0.109, "cognitive_dissonance_control": 0.0}),
    # no reference arm and nothing that orders the arms: neither is promoted to "control" because of a substring
    (["Unusual outcome", "Typical outcome"], {"Unusual outcome": 0.0, "Typical outcome": 0.0}),
    (["Standardized message", "Personalized message"], {"Standardized message": 0.0, "Personalized message": 0.0}),
])
def test_the_meta_anchored_effect_uses_the_same_whole_word_reference_arm(monkeypatch, conditions, expected):
    engine = _engine(conditions)
    monkeypatch.setattr(engine, "_get_automatic_condition_effect", lambda c, v, _raw=False: 0.0)
    got = {c: engine._meta_anchored_effect(c, "attitude", 0.5) for c in conditions}
    assert got == pytest.approx(expected, abs=1e-9)
