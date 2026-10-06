# Deployment secrets

The repository never contains API keys or passwords. Everything below is read at
runtime from environment variables or, on Streamlit Community Cloud, from
`st.secrets`. A secret that is not set simply disables the feature it powers —
the app still loads and still generates data.

## Built-in AI (free LLM providers)

The **Built-in AI** generation method calls free-tier LLM providers to write
open-ended text. Each provider slot takes one key. Set **at least one**; the
generator tries them in the order below and falls through to the next when one
rate-limits or fails.

| Order | Provider | Secret name | Where to get a free key |
|-------|----------|-------------|-------------------------|
| 1 | Google AI Studio (Gemini) | `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) | https://aistudio.google.com/apikey |
| 2 | Groq | `GROQ_API_KEY` | https://console.groq.com/keys |
| 3 | Cerebras | `CEREBRAS_API_KEY` | https://cloud.cerebras.ai |
| 4 | SambaNova Cloud | `SAMBANOVA_API_KEY` | https://cloud.sambanova.ai/apis |
| 5 | Mistral AI | `MISTRAL_API_KEY` | https://console.mistral.ai/api-keys |
| 6 | OpenRouter | `OPENROUTER_API_KEY` | https://openrouter.ai/keys |

### These secrets are optional

The app is fully usable with none of them set. Built-in AI tries every
configured provider in order and takes whatever it cannot get from them from
the built-in (non-LLM) text engine instead, so a run always completes: no error
banner, no disabled button, a complete dataset with an open-ended response for
every participant. After generation the data-source breakdown reports exactly
what came from where.

What the keys change is only whether open-ended text is **AI-written** or
**engine-written**. Numeric data is never affected by them.

Users can also paste their own key in the app ("AI (your API key)") without any
deployment secret being set. Unlike the deployment secrets, that path reports an
error when the key does not work — someone who supplies their own key needs to
know it is failing rather than silently receiving engine-written text.

## Bundling keys in the repository (optional)

A deployment secret only covers *this* deployment. Someone who clones or forks
the repository does not inherit it, so the tool will not have AI text for them
until they supply their own key. To make a clone work with no setup, keys can
instead live in a module the chain picks up automatically:

`simulation_app/utils/builtin_free_keys.py`

It must define these six names, each a non-empty key string:

| Name | Provider |
|------|----------|
| `_DEFAULT_GOOGLE_AI_KEY` | Google AI Studio (Gemini) |
| `_DEFAULT_GROQ_KEY` | Groq |
| `_DEFAULT_CEREBRAS_KEY` | Cerebras |
| `_DEFAULT_SAMBANOVA_KEY` | SambaNova Cloud |
| `_DEFAULT_MISTRAL_KEY` | Mistral AI |
| `_DEFAULT_OPENROUTER_KEY` | OpenRouter |

Any subset works; slots it leaves out fall back to the deployment secrets above.
How each value is produced does not matter — a literal, a decode, anything that
ends in a string. Keys found here are tried **first**, in the normal provider
order; deployment secrets are then appended behind them, so neither is dropped.

The file is absent from this repository by default, and nothing requires it. It
is imported inside a `try`/`except`, so a missing or malformed file degrades to
the deployment secrets rather than breaking the app.

### Removing bundled keys when you rotate

Delete the one file:

```bash
git rm simulation_app/utils/builtin_free_keys.py
git commit -m "Remove bundled free-tier keys"
```

That is the whole rotation step — no other file needs touching, no version bump
is required for it, and the app keeps working afterwards (open-ended text falls
back to the built-in engine, or to whatever deployment secrets are set).

### What bundling costs you

A key committed to a **public** repository is readable by anyone, including in
history after it is removed. Two concrete consequences: strangers can spend your
free-tier quota, which surfaces as the app being rate-limited for your own
users; and GitHub's secret-scanning partner programme notifies several of these
providers on detection, and some revoke automatically, which switches AI text
off without warning. Keep the repository private, or accept that bundled keys
are disposable and rotate them when they stop working.

## Password-gated pages

| Secret name | Gates |
|-------------|-------|
| `ADMIN_PASSWORD` or `ADMIN_PASSWORD_SHA256` | the `?admin=1` diagnostics page |
| `ANALYTICS_DASHBOARD_PASSWORD` | the analytics dashboard |

With neither set, the corresponding page stays locked. This is the intended
default: no secret, no access.

## Setting them on Streamlit Community Cloud

1. Open the app at https://share.streamlit.io and pick it.
2. **⋮ → Settings → Secrets**.
3. Paste the keys in TOML form, one per line:

   ```toml
   GOOGLE_API_KEY = "..."
   GROQ_API_KEY = "..."
   CEREBRAS_API_KEY = "..."
   SAMBANOVA_API_KEY = "..."
   MISTRAL_API_KEY = "..."
   OPENROUTER_API_KEY = "..."
   ADMIN_PASSWORD = "..."
   ANALYTICS_DASHBOARD_PASSWORD = "..."
   ```

4. **Save**. Streamlit restarts the app; the keys are picked up on the next run.

## Setting them locally

Either export them in the shell:

```bash
export GOOGLE_API_KEY="..."
streamlit run simulation_app/app.py
```

or write `.streamlit/secrets.toml` (git-ignored) with the same TOML as above.

## Rotating a key

Keys previously committed to this repository remain readable in its git
history. Revoke them at the provider and issue new ones before setting the
secrets above — a key that was ever committed should never be reused.

## Verifying

`simulation_app/utils/llm_response_generator.py` exposes
`builtin_provider_key_status()` (which slots have a key) and
`missing_builtin_provider_secrets()` (which documented names are still unset).
The admin diagnostics page shows the resolved provider chain, and the app log
line `LLM provider chain: …` lists it at startup.
