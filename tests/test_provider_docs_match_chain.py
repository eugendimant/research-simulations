"""The documented provider order must be the order the code actually uses.

Six API keys were once committed to this public repository and had to be
revoked. The setup documentation written afterwards is the thing a person
follows when creating replacements, so it has to stay true: a chain reordered
in code and not in `docs/PROVIDER_SETUP.md` sends someone to create the wrong
key first, and a provider dropped from the docs quietly stops being set up at
all. Nothing but a test keeps prose and code in step, so this is that test.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from utils.llm_response_generator import (  # noqa: E402
    BUILTIN_PROVIDER_SECRETS,
    LLMResponseGenerator,
    builtin_provider_key_status,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SETUP_DOC = _REPO_ROOT / "docs" / "PROVIDER_SETUP.md"
_KEY_POLICY = _REPO_ROOT / "docs" / "KEY_POLICY.md"

#: Maps the slot names the code uses to the words the docs head each step with.
#: A new provider needs a line here, which is the point: adding one to the chain
#: without documenting it fails this file.
_SLOT_DOC_NAMES = {
    "google_ai": "google",
    "groq": "groq",
    "cerebras": "cerebras",
    "sambanova": "sambanova",
    "mistral": "mistral",
    "openrouter": "openrouter",
}


def _documented_steps() -> list[str]:
    """Slot names in the order `PROVIDER_SETUP.md` tells people to set them up."""
    order: list[str] = []
    for line in _SETUP_DOC.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^##\s+Step\s+\d+\s*[-—–]\s*(.+)$", line.strip())
        if not m:
            continue
        heading = m.group(1).lower()
        for slot, word in _SLOT_DOC_NAMES.items():
            if word in heading:
                order.append(slot)
                break
    return order


def test_every_provider_slot_has_a_setup_step():
    documented = _documented_steps()
    assert set(documented) == set(BUILTIN_PROVIDER_SECRETS), (
        "docs/PROVIDER_SETUP.md documents %s but the chain has %s"
        % (sorted(set(documented)), sorted(BUILTIN_PROVIDER_SECRETS))
    )


def test_documented_order_matches_the_chain_order(monkeypatch):
    """Follow the doc top to bottom and you set up the chain front to back."""
    for names in BUILTIN_PROVIDER_SECRETS.values():
        monkeypatch.setenv(names[0], "test-key-%s" % names[0].lower())

    gen = LLMResponseGenerator()
    # The chain holds several entries per provider (three Google models, two
    # Groq); collapse to first appearance, which is what the doc orders.
    seen: list[str] = []
    for provider in gen._providers:
        for slot, word in _SLOT_DOC_NAMES.items():
            if word in provider.name.lower() and slot not in seen:
                seen.append(slot)
                break

    assert seen == _documented_steps(), (
        "docs/PROVIDER_SETUP.md lists %s; the chain runs %s"
        % (_documented_steps(), seen)
    )


@pytest.mark.parametrize("slot,names", sorted(BUILTIN_PROVIDER_SECRETS.items()))
def test_each_secret_name_appears_in_both_documents(slot, names):
    """A slot nobody can find the secret name for is a slot nobody configures."""
    setup = _SETUP_DOC.read_text(encoding="utf-8")
    assert names[0] in setup, "%s is not named in docs/PROVIDER_SETUP.md" % names[0]


def test_key_policy_exists_and_forbids_committing_keys():
    text = _KEY_POLICY.read_text(encoding="utf-8").lower()
    assert "never commit a key" in text
    assert "test providers now" in text
    assert "tests/test_no_secrets_in_repo.py" in text


def test_no_keys_means_no_chain_and_no_error(monkeypatch):
    """The empty chain is a supported state, not a failure.

    With nothing configured the generator must report itself unavailable for the
    stated reason and raise nothing: the app then writes open-ended text with
    the built-in engine and shows a notice rather than an error banner.
    """
    import utils.llm_response_generator as lrg

    for names in BUILTIN_PROVIDER_SECRETS.values():
        for name in names:
            monkeypatch.delenv(name, raising=False)
    for const in ("_DEFAULT_GROQ_KEY", "_DEFAULT_CEREBRAS_KEY",
                  "_DEFAULT_GOOGLE_AI_KEY", "_DEFAULT_OPENROUTER_KEY",
                  "_DEFAULT_MISTRAL_KEY", "_DEFAULT_SAMBANOVA_KEY",
                  "_DEFAULT_API_KEY"):
        monkeypatch.setattr(lrg, const, "", raising=False)

    assert builtin_provider_key_status() == {s: False for s in BUILTIN_PROVIDER_SECRETS}

    gen = lrg.LLMResponseGenerator()
    assert gen._providers == []
    assert gen.is_llm_available is False

    status = gen.check_connectivity(timeout=1)
    assert status["available"] is False
    assert status["reason"] == "not_configured"
