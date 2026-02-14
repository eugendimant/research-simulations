"""LocalLLMBackend — inference via locally-running open-source models.

Supports Ollama, llama.cpp server, vLLM, or HuggingFace Transformers.
No per-token billing — the user runs the model on their own hardware.
Falls back to OfflineBackend if no local model is available.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

import numpy as np

from ..backend import AgentBackend, GameState, Action, ActionResult
from ...persona import Persona
from ...config import get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local provider detection (passive, no downloads)
# ---------------------------------------------------------------------------

def _detect_ollama(host: str, timeout_s: float = 0.2) -> bool:
    """Check if Ollama is running at the given host."""
    if not host:
        return False
    try:
        import urllib.request
        url = f"{host.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, method="GET")
        urllib.request.urlopen(req, timeout=timeout_s)
        return True
    except Exception:
        return False


def _detect_llama_cpp(url: str, timeout_s: float = 0.2) -> bool:
    """Check if llama.cpp server is running."""
    if not url:
        return False
    try:
        import urllib.request
        req = urllib.request.Request(f"{url.rstrip('/')}/health", method="GET")
        urllib.request.urlopen(req, timeout=timeout_s)
        return True
    except Exception:
        return False


def _detect_vllm(url: str, timeout_s: float = 0.2) -> bool:
    """Check if vLLM server is running."""
    if not url:
        return False
    try:
        import urllib.request
        req = urllib.request.Request(f"{url.rstrip('/')}/v1/models", method="GET")
        urllib.request.urlopen(req, timeout=timeout_s)
        return True
    except Exception:
        return False


def detect_local_provider() -> Optional[str]:
    """Return the name of the first available local provider, or None."""
    cfg = get_config()
    timeout = cfg.local_detect_timeout_ms / 1000.0
    if _detect_ollama(cfg.ollama_host, timeout):
        return "ollama"
    if _detect_llama_cpp(cfg.llama_cpp_server_url, timeout):
        return "llama_cpp"
    if _detect_vllm(cfg.vllm_url, timeout):
        return "vllm"
    if cfg.use_transformers_local:
        try:
            import transformers  # noqa: F401
            return "transformers"
        except ImportError:
            pass
    return None


# ---------------------------------------------------------------------------
# Backend implementation
# ---------------------------------------------------------------------------

class LocalLLMBackend(AgentBackend):
    """Local LLM inference backend — no per-token API billing.

    If no local model is detected, all methods gracefully fall back
    to the OfflineBackend.
    """

    def __init__(self, fallback: Optional[AgentBackend] = None) -> None:
        self._fallback = fallback
        self._provider: Optional[str] = None
        self._available: Optional[bool] = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._provider = detect_local_provider()
            self._available = self._provider is not None
        return self._available

    def _call_local(self, messages: list[dict], temperature: float = 0.7) -> str:
        """Send a chat-completion request to the detected local provider."""
        cfg = get_config()
        if self._provider == "ollama":
            return self._call_ollama(cfg.ollama_host, cfg.local_llm_model, messages, temperature)
        if self._provider == "llama_cpp":
            return self._call_openai_compat(cfg.llama_cpp_server_url, messages, temperature)
        if self._provider == "vllm":
            return self._call_openai_compat(cfg.vllm_url, messages, temperature)
        return ""

    @staticmethod
    def _call_ollama(host: str, model: str, messages: list, temperature: float) -> str:
        import urllib.request
        body = json.dumps({
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }).encode()
        req = urllib.request.Request(
            f"{host.rstrip('/')}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return data.get("message", {}).get("content", "")

    @staticmethod
    def _call_openai_compat(base_url: str, messages: list, temperature: float) -> str:
        import urllib.request
        body = json.dumps({
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 256,
        }).encode()
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""

    # ------------------------------------------------------------------
    def sample_action(
        self,
        state: GameState,
        persona: Persona,
        rng: np.random.Generator,
    ) -> ActionResult:
        if not self.available:
            if self._fallback:
                return self._fallback.sample_action(state, persona, rng)
            raise RuntimeError("No local LLM available and no fallback configured")

        dist = self.action_distribution(state, persona)
        names = list(dist.keys())
        probs = np.array([dist[n] for n in names], dtype=float)
        probs = probs / probs.sum()
        idx = int(rng.choice(len(names), p=probs))
        chosen_name = names[idx]
        chosen = Action(name=chosen_name, value=chosen_name)
        for a in state.available_actions:
            if a.name == chosen_name:
                chosen = a
                break
        return ActionResult(chosen=chosen, distribution=dict(zip(names, probs.tolist())),
                            trace={"backend": "local_llm", "provider": self._provider})

    def action_distribution(
        self,
        state: GameState,
        persona: Persona,
    ) -> Dict[str, float]:
        if not self.available:
            if self._fallback:
                return self._fallback.action_distribution(state, persona)
            n = max(len(state.available_actions), 1)
            return {a.name: 1.0 / n for a in state.available_actions}

        # Build prompt for action selection
        action_names = [a.name for a in state.available_actions]
        prompt = self._build_action_prompt(state, persona, action_names)

        try:
            raw = self._call_local([
                {"role": "system", "content": "You are simulating a participant in a behavioral experiment. Respond with ONLY a JSON object."},
                {"role": "user", "content": prompt},
            ], temperature=0.3)
            return self._parse_distribution(raw, action_names)
        except Exception as e:
            logger.warning(f"Local LLM action failed: {e}")
            if self._fallback:
                return self._fallback.action_distribution(state, persona)
            n = len(action_names) or 1
            return {nm: 1.0 / n for nm in action_names}

    @staticmethod
    def _build_action_prompt(state: GameState, persona: Persona, action_names: list) -> str:
        persona_desc = ", ".join(f"{k}={v:.2f}" for k, v in sorted(persona.params.items()))
        return (
            f"Game: {state.game_name}\n"
            f"Parameters: {json.dumps(state.game_params)}\n"
            f"Your persona traits: {persona_desc}\n"
            f"Available actions: {action_names}\n"
            f"Round: {state.round_number}\n\n"
            f"Return a JSON object mapping each action name to a probability (0-1). "
            f"Probabilities must sum to 1.\n"
            f'Example: {{"action_a": 0.7, "action_b": 0.3}}'
        )

    @staticmethod
    def _parse_distribution(raw: str, action_names: list) -> Dict[str, float]:
        try:
            # Try to extract JSON from the response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end])
                dist = {}
                for name in action_names:
                    dist[name] = max(0.0, float(data.get(name, 0.0)))
                total = sum(dist.values())
                if total > 0:
                    return {k: v / total for k, v in dist.items()}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        # Uniform fallback
        n = len(action_names) or 1
        return {nm: 1.0 / n for nm in action_names}

    # ------------------------------------------------------------------
    def generate_open_ended(
        self,
        prompt_spec: Dict[str, Any],
        persona: Persona,
        rng: np.random.Generator,
    ) -> str:
        if not self.available:
            if self._fallback:
                return self._fallback.generate_open_ended(prompt_spec, persona, rng)
            return ""
        try:
            question = prompt_spec.get("question_text", "")
            context = prompt_spec.get("context", "")
            condition = prompt_spec.get("condition", "")
            persona_desc = ", ".join(f"{k}={v:.2f}" for k, v in sorted(persona.params.items()))
            prompt = (
                f"You are a research participant with these traits: {persona_desc}\n"
                f"Condition: {condition}\n"
                f"Context: {context}\n\n"
                f"Question: {question}\n\n"
                f"Write a brief, natural response (1-3 sentences) as this participant would."
            )
            return self._call_local([
                {"role": "system", "content": "You are simulating a human research participant. Write naturally."},
                {"role": "user", "content": prompt},
            ], temperature=0.8)
        except Exception as e:
            logger.warning(f"Local LLM text generation failed: {e}")
            return ""

    def supports_text(self) -> bool:
        return self.available

    def metadata(self) -> Dict[str, Any]:
        return {
            "backend": "local_llm",
            "available": self.available,
            "provider": self._provider,
            "network_required": False,
            "billing": False,
        }
