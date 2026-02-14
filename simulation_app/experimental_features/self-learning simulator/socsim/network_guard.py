"""Network guard — blocks remote calls by default.

Raises ``NetworkBlockedError`` when code attempts an outbound HTTP request
and ``allow_network_calls`` is False in the active config.
"""
from __future__ import annotations

import contextlib
import logging
from typing import Generator

from .config import get_config

logger = logging.getLogger(__name__)


class NetworkBlockedError(RuntimeError):
    """Raised when a remote call is attempted while network is disabled."""


def assert_network_allowed(action: str = "remote LLM call") -> None:
    """Raise if the current config forbids network access.

    Parameters
    ----------
    action:
        Human-readable description of what was attempted (for the error msg).
    """
    cfg = get_config()
    if not cfg.allow_network_calls:
        msg = (
            f"Network call blocked: '{action}'. "
            f"SocSim is running in cost_policy={cfg.cost_policy.value} / "
            f"allow_network_calls=False.  Set SOCSIM_ALLOW_NETWORK=true "
            f"and SOCSIM_REMOTE_LLM_ENABLED=true to enable remote calls."
        )
        logger.warning(msg)
        raise NetworkBlockedError(msg)


@contextlib.contextmanager
def network_guard(action: str = "remote call") -> Generator[None, None, None]:
    """Context manager that asserts network is allowed before entering."""
    assert_network_allowed(action)
    yield
