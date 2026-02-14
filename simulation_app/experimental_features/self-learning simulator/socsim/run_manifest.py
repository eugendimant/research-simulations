"""Run manifest — complete record of every simulation run.

Every run produces a manifest containing:
  - Backend info (type, version, provider)
  - Model IDs and strategy weights
  - Seeds (master + per-component)
  - Cost policy (offline_only, local_only, opt_in_remote)
  - Environment info
  - Timestamp and duration
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .config import get_config
from .cost_policy import cost_policy_manifest


@dataclass
class RunManifest:
    """Complete manifest for one simulation run."""
    run_id: str = ""
    timestamp: str = ""
    duration_s: float = 0.0

    # Configuration
    backend_info: Dict[str, Any] = field(default_factory=dict)
    strategy_weights: Dict[str, float] = field(default_factory=dict)
    cost_policy: Dict[str, Any] = field(default_factory=dict)

    # Seeds
    master_seed: int = 0
    component_seeds: Dict[str, int] = field(default_factory=dict)

    # Spec info
    game_name: str = ""
    n_samples: int = 0
    spec_hash: str = ""

    # Environment
    environment: Dict[str, str] = field(default_factory=dict)

    # Results summary
    results_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "duration_s": self.duration_s,
            "backend_info": self.backend_info,
            "strategy_weights": self.strategy_weights,
            "cost_policy": self.cost_policy,
            "master_seed": self.master_seed,
            "component_seeds": self.component_seeds,
            "game_name": self.game_name,
            "n_samples": self.n_samples,
            "spec_hash": self.spec_hash,
            "environment": self.environment,
            "results_summary": self.results_summary,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RunManifest":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def create_manifest(
    seed: int,
    game_name: str,
    n_samples: int,
    backend_info: Dict[str, Any] | None = None,
    strategy_weights: Dict[str, float] | None = None,
) -> RunManifest:
    """Create a new run manifest with current config and environment."""
    cfg = get_config()
    run_id = hashlib.sha256(f"{time.time()}_{seed}_{game_name}".encode()).hexdigest()[:12]

    return RunManifest(
        run_id=run_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        backend_info=backend_info or cfg.to_manifest_dict(),
        strategy_weights=strategy_weights or {},
        cost_policy=cost_policy_manifest(),
        master_seed=seed,
        game_name=game_name,
        n_samples=n_samples,
        environment={
            "platform": platform.system(),
            "python": platform.python_version(),
            "socsim_version": _get_version(),
        },
    )


def _get_version() -> str:
    try:
        from . import __version__
        return __version__
    except ImportError:
        return "unknown"
