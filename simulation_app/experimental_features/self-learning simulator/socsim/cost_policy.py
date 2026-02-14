"""Cost policy helpers — enforces the "no-surprise-costs" rule.

Every simulation run should record its cost policy in the manifest so
auditors can verify that no paid calls were made unintentionally.
"""
from __future__ import annotations

from .config import CostPolicy, get_config


def effective_policy() -> CostPolicy:
    """Return the active cost policy."""
    return get_config().cost_policy


def is_remote_allowed() -> bool:
    """True only when the user has explicitly opted in to remote LLM."""
    cfg = get_config()
    return (
        cfg.cost_policy == CostPolicy.OPT_IN_REMOTE
        and cfg.remote_llm_enabled
        and cfg.allow_network_calls
    )


def cost_policy_manifest() -> dict:
    """Dict suitable for embedding in a run manifest."""
    cfg = get_config()
    return {
        "cost_policy": cfg.cost_policy.value,
        "remote_allowed": is_remote_allowed(),
        "backend_mode": cfg.resolved_backend_mode().value,
    }
