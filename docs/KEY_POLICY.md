# Key policy

This repository is public. Everything in it, including everything ever
committed and later deleted, is readable by anyone forever. Six provider API
keys were once committed here; they had to be treated as compromised and
revoked. This page exists so that cannot happen a second time.

## The rules

1. **A key lives in exactly two places: the deployment and a password manager.**
   On Streamlit Community Cloud that means **Settings → Secrets**; locally it
   means an environment variable or `.streamlit/secrets.toml`, which
   `.gitignore` excludes. Nowhere else.

2. **Never commit a key, in any form.** Not in a module, not in a comment, not
   in a test fixture, not XOR-encoded, base64'd, split across lines, or
   "obfuscated". `tests/test_no_secrets_in_repo.py` fails the build on a
   key-shaped string in any tracked file, on a bundled key module, and on an
   XOR-style key block. Do not work around it.

3. **Never put a key in chat, an issue, a PR description or a commit message.**
   Those are as public as the code, and GitHub's secret-scanning partners
   notify providers on detection — Google and OpenRouter auto-revoke.
   A key that reaches any of these places is burned: rotate it, do not reuse it.

4. **Never remove a provider from the chain without a replacement.** The chain
   in `simulation_app/utils/llm_response_generator.py` is the app's redundancy.
   Dropping an entry narrows it silently; the next rate limit then has nothing
   to fall through to. Removing one means adding one.

   The one exception is a provider whose free tier no longer exists, because a
   slot nobody can obtain a key for is not redundancy — it is a row that always
   reads "not configured". Cerebras (payment card now required) and Mistral AI
   (no longer issues free API keys) were removed on this ground in v1.3.0.0. If
   either reopens a free tier, add it back rather than leaving the chain short.

5. **After any change to the LLM chain, run "Test providers now".** It is on the
   admin page (`?admin=1` → **LLM** tab) and sends one minimal request per
   configured provider. A change that compiles and passes tests can still have
   broken every live call, because tests stub the network. Run it before
   merging.

6. **No key is ever required for the app to work.** With none configured, the
   built-in engine writes the open-ended text, every run completes, numeric
   data is identical, and the user sees an explanatory notice rather than an
   error. Any change that turns a missing key into a failure is a bug —
   `tests/test_free_path_never_errors.py` and
   `test_no_keys_reports_not_configured_not_unreachable` hold that line.

## What is checked automatically

| Check | Test |
|---|---|
| No key-shaped string in any tracked file | `tests/test_no_secrets_in_repo.py` |
| No bundled key module, no XOR key block | `tests/test_no_secrets_in_repo.py` |
| `.gitignore` excludes the real secrets file | `tests/test_no_secrets_in_repo.py` |
| `secrets.toml.example` holds placeholders only | `tests/test_no_secrets_in_repo.py` |
| Every documented secret name builds its provider | `tests/test_builtin_provider_config.py` |
| Env vars and `st.secrets` are interchangeable | `tests/test_builtin_provider_config.py` |
| The chain is tried in order, first success wins | `tests/test_builtin_provider_config.py` |
| No key material in logs or failure reasons | `tests/test_builtin_provider_config.py` |
| No keys → "not configured", never "not responding" | `tests/test_builtin_provider_config.py` |
| No keys → a complete run, no error | `tests/test_free_path_never_errors.py` |
| The documented order matches the real chain | `tests/test_provider_docs_match_chain.py` |

CI runs all of them on every push.

## If a key leaks anyway

1. Revoke it at the provider first. Do not start with the git history — the
   key is live until it is revoked, and rewriting history does not revoke it.
2. Create a replacement and put it in **Settings → Secrets**.
3. Run **Test providers now** to confirm the replacement answers.
4. Only then decide what to do about the history. For a public repository,
   assume the old value was captured and treat revocation as the fix.

Setting up keys for the first time: [`PROVIDER_SETUP.md`](PROVIDER_SETUP.md).
