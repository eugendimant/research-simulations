"""Bundled free-tier keys. Delete this file to rotate."""

_DEFAULT_GROQ_KEY = bytes(b ^ _XK for b in _EB_GROQ).decode()
_DEFAULT_CEREBRAS_KEY = bytes(b ^ _XK for b in _EB_CEREBRAS).decode()
_DEFAULT_GOOGLE_AI_KEY = bytes(b ^ _XK for b in _EB_GOOGLE_AI).decode()
_DEFAULT_OPENROUTER_KEY = bytes(b ^ _XK for b in _EB_OPENROUTER).decode()
_DEFAULT_MISTRAL_KEY = bytes(b ^ _XK for b in _EB_MISTRAL).decode()
_DEFAULT_SAMBANOVA_KEY = bytes(b ^ _XK for b in _EB_SAMBANOVA).decode()
_DEFAULT_API_KEY = _DEFAULT_GROQ_KEY

