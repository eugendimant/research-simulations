"""The repository must never carry provider key material.

Eugen's requirement (2026-10-06): keys live only in the deployment environment
(Streamlit secrets or env vars), stored so they cannot leak, and the repository
must not be able to drift back into holding them. Three commits in a row had
tried to paste a key block into a tracked file, so this is a guard with teeth
rather than a convention.

What this checks, over every tracked text file:
  1. No string that looks like a live provider key for any of the six
     providers the app supports.
  2. No file named like the removed bundled-key module, under any directory.
  3. No high-entropy XOR-style obfuscated key block (the shape the old
     committed keys used), which a plain prefix scan would miss.
"""

import os
import re
import subprocess

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Live-key prefixes for the six supported providers. Each is followed by a run
# of key characters; the length floor keeps documentation examples ("gsk_...",
# "sk-or-v1-<your-key>") from tripping the scan.
_KEY_PATTERNS = [
    ("Groq", r"gsk_[A-Za-z0-9]{40,}"),
    ("Google AI", r"AIzaSy[A-Za-z0-9_\-]{30,}"),
    # v1.3.0.0: Google AI Studio's newer key shape. Without this the scanner
    # would have waved a live key through purely because it was issued after
    # the "AIza" era.
    ("Google AI (new)", r"AQ\.[A-Za-z0-9_\-]{30,}"),
    ("OpenRouter", r"sk-or-v1-[a-f0-9]{48,}"),
    ("OpenAI-style", r"sk-[A-Za-z0-9]{44,}"),
    ("Cerebras", r"csk-[a-z0-9]{40,}"),
]

# The bundled-key module removed in v1.2.9.3, plus the obvious variations.
_FORBIDDEN_BASENAMES = {
    "builtin_free_keys.py",
    "built_in_free_keys.py",
    "free_keys.py",
    "default_keys.py",
    "api_keys.py",
    "secrets.py",
}

# Files that legitimately discuss key formats (docs, this test, the scanner).
_SCAN_EXEMPT = {
    "tests/test_no_secrets_in_repo.py",
    "tests/test_builtin_provider_config.py",
}

_TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".yml", ".yaml", ".json",
                  ".cfg", ".ini", ".sh", ".example"}


def _tracked_files():
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout
    return [f for f in out.split("\0") if f]


@pytest.fixture(scope="module")
def tracked():
    return _tracked_files()


def test_no_live_provider_key_in_any_tracked_file(tracked):
    """A key-shaped string in a tracked file fails the build."""
    hits = []
    for rel in tracked:
        if rel in _SCAN_EXEMPT:
            continue
        if os.path.splitext(rel)[1] not in _TEXT_SUFFIXES:
            continue
        path = os.path.join(REPO_ROOT, rel)
        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
        except OSError:
            continue
        for provider, pattern in _KEY_PATTERNS:
            if re.search(pattern, content):
                # Report the file and provider only — never the matched text.
                hits.append("%s (looks like a %s key)" % (rel, provider))
    assert not hits, (
        "Provider key material must never be committed. Offending files:\n  "
        + "\n  ".join(sorted(set(hits)))
        + "\nKeys belong in Streamlit secrets or environment variables only; "
          "see docs/PROVIDER_SETUP.md."
    )


def test_no_bundled_key_module_exists(tracked):
    """The in-repository key store must stay removed."""
    offenders = [rel for rel in tracked
                 if os.path.basename(rel) in _FORBIDDEN_BASENAMES]
    assert not offenders, (
        "These files are an in-repository key store, which v1.2.9.3 removed "
        "on purpose: %s. A key committed here is readable by anyone who can "
        "read the repository, and stays readable in history after deletion. "
        "Configure keys as deployment secrets instead "
        "(docs/PROVIDER_SETUP.md)." % ", ".join(sorted(offenders))
    )


def test_no_xor_obfuscated_key_block(tracked):
    """Obfuscation is not storage — the old committed block must not return.

    The pre-v1.2.9.0 keys were XOR-encoded byte lists, which no prefix scan
    would catch. This looks for that shape: a long list of small integers
    feeding a `b ^ <name>` decode.
    """
    offenders = []
    for rel in tracked:
        if rel in _SCAN_EXEMPT or not rel.endswith(".py"):
            continue
        path = os.path.join(REPO_ROOT, rel)
        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
        except OSError:
            continue
        if re.search(r"bytes\(\s*\w+\s*\^\s*\w+\s+for\s+\w+\s+in\s+", content):
            offenders.append(rel)
    assert not offenders, (
        "XOR-decoded byte blocks look like obfuscated credentials and are not "
        "a storage mechanism: %s. Use deployment secrets."
        % ", ".join(sorted(offenders))
    )


def test_secrets_example_carries_no_real_values():
    """secrets.toml.example must ship placeholders, not keys."""
    path = os.path.join(REPO_ROOT, "secrets.toml.example")
    assert os.path.exists(path), "secrets.toml.example is the paste template"
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    for provider, pattern in _KEY_PATTERNS:
        assert not re.search(pattern, content), (
            "secrets.toml.example contains something shaped like a real %s key"
            % provider
        )


def test_gitignore_blocks_the_real_secrets_file():
    """A local secrets.toml must never be committable by accident."""
    with open(os.path.join(REPO_ROOT, ".gitignore"), encoding="utf-8") as fh:
        ignored = fh.read()
    assert "secrets.toml" in ignored, (
        ".gitignore must cover .streamlit/secrets.toml so a local key file "
        "cannot be committed by accident"
    )
