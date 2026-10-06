"""Built-in LLM provider configuration from deployment secrets.

Regression guard for the v1.2.9.0 → v1.2.9.1 defect: the committed free-tier
keys were (correctly) removed in v1.2.9.0, but with no keys in the environment
the app still told users "Free AI providers are currently not responding" and
to try again in a few hours. The real state was "no key is configured", which
no amount of waiting fixes.

These tests pin down three things:
  1. With no keys set, the provider chain is empty and both status calls report
     reason == "not_configured" together with the documented secret names.
  2. With a key set in the environment, the matching provider slot is built —
     i.e. a configured deployment actually picks its keys up, including when
     the key becomes readable after this module was first imported.
  3. The documented secret names stay in sync with the provider slots.
"""

import os
import re
import sys
import types

import pytest

from utils import llm_response_generator as lrg
from utils.llm_response_generator import (
    BUILTIN_PROVIDER_SECRETS,
    LLMResponseGenerator,
    builtin_provider_key_status,
    missing_builtin_provider_secrets,
)

# Every env var that can feed a built-in provider slot, plus the user-key vars
# the constructor also consults.
_ALL_KEY_VARS = sorted(
    {name for names in BUILTIN_PROVIDER_SECRETS.values() for name in names}
    | {"LLM_API_KEY", "LLM_PROVIDER_HINT"}
)


@pytest.fixture
def no_keys(monkeypatch):
    """Simulate a deployment with no provider secrets configured at all."""
    for var in _ALL_KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    # The module-level constants are resolved at import time; a developer
    # machine with keys exported would otherwise leak them into the test.
    for const in ("_DEFAULT_GROQ_KEY", "_DEFAULT_CEREBRAS_KEY",
                  "_DEFAULT_GOOGLE_AI_KEY", "_DEFAULT_OPENROUTER_KEY",
                  "_DEFAULT_MISTRAL_KEY", "_DEFAULT_SAMBANOVA_KEY",
                  "_DEFAULT_API_KEY"):
        monkeypatch.setattr(lrg, const, "", raising=False)
    return None


def test_no_keys_yields_empty_provider_chain(no_keys):
    gen = LLMResponseGenerator()
    assert gen._providers == []
    assert gen.is_llm_available is False


def test_no_keys_reports_not_configured_not_unreachable(no_keys):
    """The UI branches on `reason`; it must not say 'not responding' here."""
    gen = LLMResponseGenerator()

    status = gen.check_connectivity(timeout=1)
    assert status["available"] is False
    assert status["reason"] == "not_configured"
    assert "GROQ_API_KEY" in status["missing_secrets"]

    health = gen.health_check(timeout=1)
    assert health["ok"] is False
    assert health["reason"] == "not_configured"
    assert "GROQ_API_KEY" in health["missing_secrets"]


def test_missing_secrets_lists_every_slot_when_nothing_is_set(no_keys):
    assert builtin_provider_key_status() == {
        slot: False for slot in BUILTIN_PROVIDER_SECRETS
    }
    assert sorted(missing_builtin_provider_secrets()) == sorted(
        names[0] for names in BUILTIN_PROVIDER_SECRETS.values()
    )


@pytest.mark.parametrize(
    ("slot", "env_var"),
    [(slot, names[0]) for slot, names in BUILTIN_PROVIDER_SECRETS.items()],
)
def test_each_documented_secret_builds_its_provider(no_keys, monkeypatch, slot, env_var):
    """A key set in the environment must reach the provider chain.

    Keys are resolved per instance, so one set *after* import still counts —
    that is what makes `st.secrets` on Streamlit Cloud work.
    """
    monkeypatch.setenv(env_var, "test-key-%s" % slot)

    assert builtin_provider_key_status()[slot] is True
    assert env_var not in missing_builtin_provider_secrets()

    gen = LLMResponseGenerator()
    assert gen._providers, "a configured key must produce at least one provider"
    assert any(p.name.startswith(slot) for p in gen._providers), (
        "%s did not build a provider for slot %s; chain was %s"
        % (env_var, slot, [p.name for p in gen._providers])
    )

    # With a key present the status is "unreachable" at worst, never
    # "not_configured" — we must not tell an admin to set a key they already
    # set. Stub the transport so the assertion needs no network.
    monkeypatch.setattr(lrg, "_call_llm_api", lambda *a, **k: None)
    assert gen.health_check(timeout=1)["reason"] == "unreachable"
    assert gen.check_connectivity(timeout=1)["reason"] == "unreachable"


def test_secret_names_are_documented(no_keys):
    """Every secret name must appear in the deployment doc."""
    doc = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs", "DEPLOYMENT_SECRETS.md",
    )
    with open(doc, encoding="utf-8") as fh:
        text = fh.read()
    for names in BUILTIN_PROVIDER_SECRETS.values():
        for name in names:
            assert name in text, "%s is undocumented in DEPLOYMENT_SECRETS.md" % name


def test_no_key_material_in_source():
    """The generator must never carry keys or an obfuscation scheme again."""
    path = os.path.abspath(lrg.__file__).replace(".pyc", ".py")
    with open(path, encoding="utf-8") as fh:
        source = fh.read()
    assert "_XK" not in source, "XOR key-obfuscation reintroduced"
    # The provider-detection helper legitimately mentions bare key prefixes, so
    # match only a prefix followed by enough key characters to be a real key.
    for prefix in ("gsk_", "csk-", "sk-or-", "AIzaSy", "snova-"):
        hit = re.search(re.escape(prefix) + r"[A-Za-z0-9_\-]{20,}", source)
        assert hit is None, "literal %s key found in source: %s" % (prefix, hit)


# The failover order users rely on. v1.2.9.0 removed the committed keys but did
# NOT change this order; pinning it here means a future key/provider edit cannot
# silently reorder or drop a provider.
_EXPECTED_ORDER = [
    "google_ai_3_lite",
    "google_ai_flash",
    "google_ai_lite",
    "groq_builtin",
    "groq_qwen_builtin",
    "cerebras_builtin",
    "sambanova_builtin",
    "mistral_builtin",
    "openrouter_builtin",
]


def test_provider_failover_order_is_stable(no_keys, monkeypatch):
    """With every key set, the chain must be tried in the documented order."""
    for names in BUILTIN_PROVIDER_SECRETS.values():
        monkeypatch.setenv(names[0], "test-key-%s" % names[0].lower())

    gen = LLMResponseGenerator()
    builtin = [p.name for p in gen._providers if p.name.endswith("_builtin")
               or p.name.startswith("google_ai_")]
    assert builtin == _EXPECTED_ORDER, (
        "built-in provider order drifted: %s" % builtin
    )


def test_every_configured_provider_is_tried_in_order(no_keys, monkeypatch):
    """check_connectivity must actually probe the chain, not bail out early."""
    for names in BUILTIN_PROVIDER_SECRETS.values():
        monkeypatch.setenv(names[0], "test-key-%s" % names[0].lower())

    tried = []

    def _record(api_url, api_key, model, *a, **k):
        tried.append(model)
        return None  # every provider "fails" so the loop runs to the end

    monkeypatch.setattr(lrg, "_call_llm_api", _record)

    gen = LLMResponseGenerator()
    status = gen.check_connectivity(timeout=1)

    assert status["available"] is False
    assert status["reason"] == "unreachable"
    expected = [p.model for p in gen._providers]
    assert tried == expected, (
        "providers were not tried in chain order: tried %s, chain %s"
        % (tried, expected)
    )


def test_first_working_provider_wins(no_keys, monkeypatch):
    """The chain stops at the first provider that answers."""
    for names in BUILTIN_PROVIDER_SECRETS.values():
        monkeypatch.setenv(names[0], "test-key-%s" % names[0].lower())

    calls = []

    def _first_ok(api_url, api_key, model, *a, **k):
        calls.append(model)
        return "OK"

    monkeypatch.setattr(lrg, "_call_llm_api", _first_ok)

    gen = LLMResponseGenerator()
    status = gen.check_connectivity(timeout=1)

    assert status["available"] is True
    assert status["provider"] == _EXPECTED_ORDER[0]
    assert len(calls) == 1, "chain kept probing after a provider answered"


def test_provider_chain_log_never_contains_key_bytes(no_keys, monkeypatch, caplog):
    """The diagnostic log line must not carry even a key prefix."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_supersecretvalue123456")

    with caplog.at_level("INFO", logger=lrg.logger.name):
        LLMResponseGenerator()

    text = caplog.text
    assert "gsk_super" not in text
    assert "supersecret" not in text
    assert "(key set)" in text


# ---------------------------------------------------------------------------
# Deployment secrets that live ONLY in st.secrets (Streamlit Community Cloud)
# ---------------------------------------------------------------------------

class _FakeSecrets(dict):
    """Stand-in for st.secrets, which is a Mapping with .get()."""


def _install_fake_streamlit_secrets(monkeypatch, **secrets):
    """Make `import streamlit` yield a module whose .secrets holds `secrets`.

    This is the Streamlit Community Cloud path: the key is readable through
    st.secrets and is NOT in os.environ.
    """
    fake = types.ModuleType("streamlit")
    fake.secrets = _FakeSecrets(secrets)
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    return fake


@pytest.mark.parametrize("slot", sorted(BUILTIN_PROVIDER_SECRETS))
def test_secrets_only_key_builds_a_provider_for_every_slot(
        no_keys, monkeypatch, slot):
    """A key set only in st.secrets must build that slot's provider.

    Guards the v1.2.9.1 defect where the per-slot override blocks read
    os.environ directly: a deployment that set its keys the documented
    Streamlit way had them silently ignored for those slots.
    """
    secret_name = BUILTIN_PROVIDER_SECRETS[slot][0]
    _install_fake_streamlit_secrets(monkeypatch, **{secret_name: "secrets-%s" % slot})

    gen = LLMResponseGenerator()
    keys = {p.api_key for p in gen._providers}
    assert "secrets-%s" % slot in keys, (
        "a key readable only through st.secrets was dropped for slot %r" % slot
    )


@pytest.mark.parametrize("slot", sorted(BUILTIN_PROVIDER_SECRETS))
def test_env_and_secrets_are_interchangeable(no_keys, monkeypatch, slot):
    """The two documented configuration paths must behave identically.

    Guards the v1.2.9.1 defect: the per-slot override blocks read os.environ
    directly, so a deployment that set its keys the documented Streamlit way
    had them silently ignored for those slots.
    """
    secret_name = BUILTIN_PROVIDER_SECRETS[slot][0]

    monkeypatch.setenv(secret_name, "key-%s" % slot)
    from_env = sorted(p.name for p in LLMResponseGenerator()._providers)
    monkeypatch.delenv(secret_name)

    _install_fake_streamlit_secrets(monkeypatch, **{secret_name: "key-%s" % slot})
    from_secrets = sorted(p.name for p in LLMResponseGenerator()._providers)

    assert from_env, "no provider was built for slot %r from the environment" % slot
    assert from_env == from_secrets, (
        "slot %r builds a different chain from st.secrets than from env" % slot
    )


def test_identical_key_from_both_sources_is_not_duplicated(no_keys, monkeypatch):
    """Same key in env and st.secrets must not be added to the chain twice."""
    monkeypatch.setenv("MISTRAL_API_KEY", "same-key")
    _install_fake_streamlit_secrets(monkeypatch, MISTRAL_API_KEY="same-key")

    gen = LLMResponseGenerator()
    assert [p.api_key for p in gen._providers].count("same-key") == 1


# ---------------------------------------------------------------------------
# verify_providers(): per-provider OK / failed / not configured
# ---------------------------------------------------------------------------

def test_verify_reports_every_slot_when_nothing_is_configured(no_keys):
    """With no keys the sweep still names all six slots and their secrets."""
    results = LLMResponseGenerator().verify_providers(timeout=1)

    assert [r["slot"] for r in results] == list(BUILTIN_PROVIDER_SECRETS)
    assert all(r["status"] == "not_configured" for r in results)
    for r in results:
        assert r["secret"] == BUILTIN_PROVIDER_SECRETS[r["slot"]][0]
        assert r["secret"] in r["detail"]


def test_verify_makes_no_network_call_for_unconfigured_slots(no_keys, monkeypatch):
    calls = []
    monkeypatch.setattr(lrg, "_call_llm_api",
                        lambda *a, **k: calls.append(1) or "OK")

    LLMResponseGenerator().verify_providers(timeout=1)
    assert calls == [], "an unconfigured slot must not be dialled"


def test_verify_reports_ok_for_a_working_key(no_keys, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_working")
    monkeypatch.setattr(lrg, "_call_llm_api", lambda *a, **k: "OK")

    results = {r["slot"]: r for r in
               LLMResponseGenerator().verify_providers(timeout=1)}

    assert results["groq"]["status"] == "ok"
    assert results["groq"]["latency_ms"] is not None
    assert results["mistral"]["status"] == "not_configured"


def test_verify_reports_failed_for_a_rejected_key(no_keys, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_rejected")
    monkeypatch.setattr(lrg, "_call_llm_api", lambda *a, **k: None)

    results = {r["slot"]: r for r in
               LLMResponseGenerator().verify_providers(timeout=1)}

    assert results["groq"]["status"] == "failed"
    assert results["groq"]["detail"], "a failure must carry a reason"
    assert results["groq"]["latency_ms"] is None


def test_verify_never_leaks_the_key_in_a_failure_reason(no_keys, monkeypatch):
    """Some provider error bodies echo the credential back — scrub it."""
    # Not a real credential — a fake shaped like one, so the scrubbing is
    # exercised against the format a provider would echo back.
    secret = "gsk_" + "supersecretvalue123456"  # noqa: S105
    monkeypatch.setenv("GROQ_API_KEY", secret)

    def _echoes_the_key(*a, **k):
        raise RuntimeError("401 invalid api key: %s" % secret)

    monkeypatch.setattr(lrg, "_call_llm_api", _echoes_the_key)

    results = LLMResponseGenerator().verify_providers(timeout=1)
    blob = repr(results)
    assert secret not in blob, "verify_providers leaked a key in its output"
    assert "supersecret" not in blob
    assert "***" in blob


def test_verify_leaves_providers_usable_afterwards(no_keys, monkeypatch):
    """A diagnostic sweep must not disable the chain for the real run."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_working")
    monkeypatch.setattr(lrg, "_call_llm_api", lambda *a, **k: None)

    gen = LLMResponseGenerator()
    gen.verify_providers(timeout=1)
    assert all(p.available for p in gen._providers)
