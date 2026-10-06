"""Bundled free-tier API keys for Built-in AI.

ROTATION: this whole file is deleted in one commit when the keys are rotated.

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
