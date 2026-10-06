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

With **none** of these set, Built-in AI reports "not configured on this
deployment" (not "not responding") and the UI recommends the Adaptive
Behavioral Engine 3.0, which runs entirely offline. Numeric data is never
affected by these keys.

Users can always paste their own key in the app ("AI (your API key)") without
any deployment secret being set.

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
