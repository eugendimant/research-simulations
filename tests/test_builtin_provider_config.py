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
