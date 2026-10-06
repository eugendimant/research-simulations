"""Bundled free-tier API keys for Built-in AI.

_XK = 0x5A

# Groq (primary) ΓÇö 14,400 requests/day free
_EB_GROQ = [61, 41, 49, 5, 51, 28, 28, 106, 10, 106, 60, 61, 54, 107, 48, 59,
            2, 13, 46, 22, 35, 35, 8, 98, 13, 29, 62, 35, 56, 105, 28, 3,
            20, 22, 105, 49, 42, 44, 62, 49, 10, 34, 29, 2, 23, 12, 40, 46,
            108, 49, 52, 28, 43, 8, 15, 48]
_DEFAULT_GROQ_KEY = bytes(b ^ _XK for b in _EB_GROQ).decode()

# Cerebras ΓÇö 1M tokens/day free, Llama 3.3 70B
_EB_CEREBRAS = [57, 41, 49, 119, 111, 63, 98, 104, 50, 49, 104, 50, 52, 46, 45, 42,
                50, 62, 108, 62, 49, 60, 40, 50, 40, 111, 46, 60, 40, 48, 52, 104,
                98, 52, 108, 104, 48, 62, 42, 108, 62, 42, 108, 50, 42, 34, 104, 35,
                45, 40, 105, 46]
_DEFAULT_CEREBRAS_KEY = bytes(b ^ _XK for b in _EB_CEREBRAS).decode()

# Google AI Studio ΓÇö truly free, no credit card
# Gemini 2.5 Flash: 15 RPM, 1M TPM (high-quality volume)
# Gemini 2.5 Flash Lite: 30 RPM, 250K TPM (cost-efficient)
_EB_GOOGLE_AI = [27, 19, 32, 59, 9, 35, 25, 20, 35, 17, 42, 107, 109, 106, 47, 61,
                 13, 57, 43, 12, 57, 34, 47, 18, 44, 11, 110, 119, 61, 16, 60, 59,
                 41, 111, 98, 15, 11, 51, 15]
_DEFAULT_GOOGLE_AI_KEY = bytes(b ^ _XK for b in _EB_GOOGLE_AI).decode()

# OpenRouter ΓÇö free models (Mistral Small 3.1 24B ΓÇö more generous rate limits than Llama free)
_EB_OPENROUTER = [41, 49, 119, 53, 40, 119, 44, 107, 119, 105, 110, 63, 105, 62, 107,
                  111, 104, 98, 59, 56, 62, 104, 63, 59, 57, 56, 109, 105, 110, 63,
                  105, 60, 59, 109, 56, 106, 57, 107, 63, 107, 63, 109, 104, 60, 62,
                  63, 57, 62, 60, 62, 111, 104, 63, 57, 57, 111, 63, 59, 110, 63,
                  105, 104, 109, 111, 105, 60, 111, 99, 107, 98, 110, 62, 107]
_DEFAULT_OPENROUTER_KEY = bytes(b ^ _XK for b in _EB_OPENROUTER).decode()

# v1.2.1.2: Mistral AI ΓÇö 1B tokens/month free, 2 RPM, no credit card
_EB_MISTRAL = [46, 8, 111, 109, 109, 42, 57, 55, 40, 107, 106, 110, 105, 18, 29, 2,
               44, 21, 106, 34, 60, 3, 109, 15, 62, 25, 44, 25, 55, 29, 22, 17]
_DEFAULT_MISTRAL_KEY = bytes(b ^ _XK for b in _EB_MISTRAL).decode()

# v1.2.1.2: SambaNova Cloud ΓÇö persistent free tier, Llama 3.3 70B, 20 RPM
_EB_SAMBANOVA = [109, 105, 57, 99, 107, 56, 109, 107, 119, 110, 63, 98, 107, 119, 110,
                 106, 60, 109, 119, 56, 109, 57, 111, 119, 107, 107, 104, 59, 105, 106,
                 106, 62, 62, 56, 60, 59]
_DEFAULT_SAMBANOVA_KEY = bytes(b ^ _XK for b in _EB_SAMBANOVA).decode()

# Legacy alias
_DEFAULT_API_KEY = _DEFAULT_GROQ_KEY

    git rm simulation_app/utils/builtin_free_keys.py

That is the entire rotation step. No other file needs touching, no version bump
is required, and the app keeps working without it: open-ended text falls back to
the deployment secrets (env / st.secrets), and then to the built-in engine.

The six names below are the same ones the key block used before v1.2.9.0, so the
old block pastes in here unchanged. Any subset works; empty slots fall through to
the deployment secrets. See docs/DEPLOYMENT_SECRETS.md.
"""

_DEFAULT_GOOGLE_AI_KEY = ""
_DEFAULT_GROQ_KEY = ""
_DEFAULT_CEREBRAS_KEY = ""
_DEFAULT_SAMBANOVA_KEY = ""
_DEFAULT_MISTRAL_KEY = ""
_DEFAULT_OPENROUTER_KEY = ""
