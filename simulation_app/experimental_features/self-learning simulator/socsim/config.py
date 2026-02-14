"""SocSim configuration — single source of truth for backend selection and cost policy.

Default behaviour: **offline-only**.  No outbound HTTP requests unless
explicitly opted in by the user.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BackendMode(str, Enum):
    """Which inference back-end to use."""
    OFFLINE = "offline"
    LOCAL_LLM = "local_llm"
    REMOTE_LLM = "remote_llm"
    AUTO = "auto"  # offline → local_llm (if detected) → never remote


class CostPolicy(str, Enum):
    """How aggressive the system is allowed to be with paid resources."""
    OFFLINE_ONLY = "offline_only"
    LOCAL_ONLY = "local_only"
    OPT_IN_REMOTE = "opt_in_remote"


@dataclass
class SocSimConfig:
    """Runtime configuration for SocSim."""

    backend_mode: BackendMode = BackendMode.AUTO
    cost_policy: CostPolicy = CostPolicy.OFFLINE_ONLY
    allow_network_calls: bool = False

    # Local-LLM settings (used only when backend_mode is LOCAL_LLM or AUTO)
    ollama_host: str = ""
    llama_cpp_server_url: str = ""
    vllm_url: str = ""
    use_transformers_local: bool = False
    local_llm_model: str = "llama3.2:3b"

    # Remote-LLM settings (used only when backend_mode is REMOTE_LLM
    # AND cost_policy is OPT_IN_REMOTE)
    remote_llm_enabled: bool = False
    remote_api_key: str = ""

    # Detection timeouts (ms)
    local_detect_timeout_ms: int = 200

    @classmethod
    def from_env(cls) -> "SocSimConfig":
        """Build config from environment variables with safe defaults."""
        mode_str = os.environ.get("SOCSIM_BACKEND_MODE", "auto").lower()
        try:
            mode = BackendMode(mode_str)
        except ValueError:
            mode = BackendMode.AUTO

        policy_str = os.environ.get("SOCSIM_COST_POLICY", "offline_only").lower()
        try:
            policy = CostPolicy(policy_str)
        except ValueError:
            policy = CostPolicy.OFFLINE_ONLY

        allow_net = os.environ.get("SOCSIM_ALLOW_NETWORK", "false").lower() in ("1", "true", "yes")
        remote_enabled = os.environ.get("SOCSIM_REMOTE_LLM_ENABLED", "false").lower() in ("1", "true", "yes")

        return cls(
            backend_mode=mode,
            cost_policy=policy,
            allow_network_calls=allow_net,
            ollama_host=os.environ.get("OLLAMA_HOST", ""),
            llama_cpp_server_url=os.environ.get("LLAMA_CPP_SERVER_URL", ""),
            vllm_url=os.environ.get("VLLM_URL", ""),
            use_transformers_local=os.environ.get("SOCSIM_USE_TRANSFORMERS", "false").lower() in ("1", "true"),
            local_llm_model=os.environ.get("SOCSIM_LOCAL_MODEL", "llama3.2:3b"),
            remote_llm_enabled=remote_enabled,
            remote_api_key=os.environ.get("SOCSIM_REMOTE_API_KEY", ""),
        )

    def resolved_backend_mode(self) -> BackendMode:
        """Resolve AUTO to an actual mode (offline or local_llm)."""
        if self.backend_mode != BackendMode.AUTO:
            return self.backend_mode
        # AUTO: try local LLM detection (passive), else offline
        if self.ollama_host or self.llama_cpp_server_url or self.vllm_url or self.use_transformers_local:
            return BackendMode.LOCAL_LLM
        return BackendMode.OFFLINE

    def to_manifest_dict(self) -> dict:
        """Produce a dict suitable for inclusion in run manifests."""
        return {
            "backend_mode": self.backend_mode.value,
            "resolved_backend": self.resolved_backend_mode().value,
            "cost_policy": self.cost_policy.value,
            "allow_network_calls": self.allow_network_calls,
            "remote_llm_enabled": self.remote_llm_enabled,
            "local_llm_model": self.local_llm_model,
        }


# Singleton default config
_default_config: Optional[SocSimConfig] = None


def get_config() -> SocSimConfig:
    """Return the current global config (lazily initialised from env)."""
    global _default_config
    if _default_config is None:
        _default_config = SocSimConfig.from_env()
    return _default_config


def set_config(cfg: SocSimConfig) -> None:
    """Override the global config (mainly for tests)."""
    global _default_config
    _default_config = cfg
