"""Runtime tests for app.py helpers that import-safety and py_compile cannot catch."""
import ast
import hashlib
import logging
import os
import types
from pathlib import Path

import pytest

_APP = Path(__file__).resolve().parent.parent / "simulation_app" / "app.py"


def _load(names, extra_ns):
    """Compile only the named top-level functions from app.py (the module body renders pages)."""
    tree = ast.parse(_APP.read_text(encoding="utf-8"))
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in body} == set(names)
    ns = dict(extra_ns)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(_APP), "exec"), ns)
    return ns


class _Secrets(dict):
    pass


def _st(secrets=None, state=None):
    return types.SimpleNamespace(secrets=_Secrets(secrets or {}), session_state=state or {})


@pytest.fixture
def clean_env(monkeypatch):
    for k in ("TEST_CODE", "TEST_CODE_SHA256"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def _matcher(st):
    return _load(["_access_code_matches"], {"os": os, "hashlib": hashlib, "st": st})["_access_code_matches"]


def test_unset_secret_fails_closed(clean_env):
    m = _matcher(_st())
    assert m("anything", "TEST_CODE") is False
    assert m("", "TEST_CODE") is False


def test_plaintext_secret(clean_env):
    clean_env.setenv("TEST_CODE", "s3cret")
    m = _matcher(_st())
    assert m("s3cret", "TEST_CODE") is True
    assert m("wrong", "TEST_CODE") is False
    assert m("", "TEST_CODE") is False


def test_sha256_secret_and_st_secrets(clean_env):
    digest = hashlib.sha256(b"hunter2").hexdigest().upper()
    m = _matcher(_st({"TEST_CODE_SHA256": digest}))
    assert m("hunter2", "TEST_CODE") is True
    assert m("hunter3", "TEST_CODE") is False


def test_returns_bool_not_none(clean_env):
    assert _matcher(_st())("x", "TEST_CODE") is False


def _collector(state, calls, raises=False):
    def collect(fn, content):
        if raises:
            raise RuntimeError("boom")
        calls.append((fn, content))
    st = _st(state=state)
    ns = _load(["_collect_qsf_if_consented"],
               {"st": st, "collect_qsf_async": collect, "_app_logging": logging})
    return ns["_collect_qsf_if_consented"], st


def test_collect_requires_consent_and_never_raises():
    calls = []
    fn, _ = _collector({}, calls)
    fn("a.qsf", b"x")
    assert calls == []
    fn, _ = _collector({"share_survey_consent_0": True}, calls)
    fn("a.qsf", b"x")
    assert calls == [("a.qsf", b"x")]
    fn, _ = _collector({"share_survey_consent_0": True}, [], raises=True)
    fn("a.qsf", b"x")  # must not raise


def test_consent_is_consumed_by_one_file():
    """One tick authorizes one upload: a later upload (or generated design) needs a new tick."""
    calls = []
    fn, st = _collector({"share_survey_consent_0": True}, calls)
    fn("first.qsf", b"1")
    fn("second.qsf", b"2")
    assert calls == [("first.qsf", b"1")]
    assert st.session_state["_share_consent_nonce"] == 1
    st.session_state["share_survey_consent_1"] = True  # user ticks the fresh checkbox
    fn("third.qsf", b"3")
    assert [c[0] for c in calls] == ["first.qsf", "third.qsf"]


def test_builder_never_collects():
    src = _APP.read_text(encoding="utf-8")
    start = src.index("def _finalize_builder_design")
    end = src.index("\ndef ", start + 10)
    assert "_collect_qsf_if_consented" not in src[start:end]
