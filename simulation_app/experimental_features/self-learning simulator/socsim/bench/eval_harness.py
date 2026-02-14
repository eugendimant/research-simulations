"""Evaluation harness — run backends across a pre-committed manifest.

Collects:
  - Action distributions per game spec
  - Invalid response rate
  - Response diversity metrics
  - Calibration metrics (if human data available)
  - Backend metadata

A single CLI command runs end-to-end and writes a report.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .precommit import manifest_to_specs
from ..agents.backend import AgentBackend, GameState, Action
from ..persona import Persona, PersonaGenerator
from ..utils import parse_priors
from ..games.families.money_request_family import FamilySpec


@dataclass
class EvalResult:
    """Results from evaluating one backend on one game spec."""
    spec_hash: str
    game_name: str
    action_distribution: Dict[str, float]
    mean_action: float
    std_action: float
    invalid_rate: float
    diversity: float  # Shannon entropy of action distribution
    n_samples: int
    elapsed_ms: float
    trace: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HarnessReport:
    """Aggregate evaluation report across all specs."""
    backend_name: str
    n_specs: int
    n_total_samples: int
    mean_invalid_rate: float
    mean_diversity: float
    mean_elapsed_ms: float
    results: List[EvalResult]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "n_specs": self.n_specs,
            "n_total_samples": self.n_total_samples,
            "mean_invalid_rate": self.mean_invalid_rate,
            "mean_diversity": self.mean_diversity,
            "mean_elapsed_ms": self.mean_elapsed_ms,
            "results": [
                {
                    "spec_hash": r.spec_hash,
                    "game_name": r.game_name,
                    "action_distribution": r.action_distribution,
                    "mean_action": r.mean_action,
                    "std_action": r.std_action,
                    "invalid_rate": r.invalid_rate,
                    "diversity": r.diversity,
                    "n_samples": r.n_samples,
                    "elapsed_ms": r.elapsed_ms,
                }
                for r in self.results
            ],
            "metadata": self.metadata,
        }


def _shannon_entropy(dist: Dict[str, float]) -> float:
    """Compute Shannon entropy of a probability distribution."""
    h = 0.0
    for p in dist.values():
        if p > 1e-12:
            h -= p * np.log2(p)
    return float(h)


def _make_game_state(spec: FamilySpec) -> GameState:
    """Create a GameState from a FamilySpec."""
    params = spec.params
    min_req = int(params.get("min_request", 11))
    max_req = int(params.get("max_request", 20))
    actions = [Action(name=str(r), value=r) for r in range(min_req, max_req + 1)]
    return GameState(
        game_name=spec.game_name,
        game_params=params,
        available_actions=actions,
    )


def evaluate_backend(
    backend: AgentBackend,
    manifest: Dict[str, Any],
    n_samples_per_spec: int = 100,
    seed: int = 42,
    priors: Dict[str, Any] | None = None,
    latent_classes: Dict[str, Any] | None = None,
) -> HarnessReport:
    """Run full evaluation of a backend across a manifest.

    Parameters
    ----------
    backend : AgentBackend
        The backend to evaluate.
    manifest : dict
        Pre-committed manifest from ``generate_precommit_manifest``.
    n_samples_per_spec : int
        Number of agent simulations per game spec.
    seed : int
        Master RNG seed.
    priors : dict, optional
        Parameter priors. Uses defaults if not provided.
    latent_classes : dict, optional
        Latent class definitions. Uses defaults if not provided.
    """
    rng = np.random.default_rng(seed)
    specs = manifest_to_specs(manifest)

    # Simple priors for evaluation
    if priors is None:
        priors = {
            "fairness_alpha": {"mean": 0.8, "sd": 0.4, "min": 0.0, "max": 3.0},
            "fairness_beta": {"mean": 0.2, "sd": 0.3, "min": 0.0, "max": 2.0},
            "prosociality": {"mean": 0.0, "sd": 1.0, "min": -3.0, "max": 3.0},
            "noise_lambda": {"mean": 3.0, "sd": 2.0, "min": 0.1, "max": 20.0},
            "strategic_depth": {"mean": 1.5, "sd": 1.0, "min": 0.0, "max": 4.0},
            "risk_aversion": {"mean": 0.5, "sd": 0.3, "min": 0.0, "max": 2.0},
            "ingroup_bias": {"mean": 0.0, "sd": 0.3, "min": -1.0, "max": 1.0},
            "norm_weight": {"mean": 0.3, "sd": 0.2, "min": 0.0, "max": 2.0},
            "reciprocity": {"mean": 0.5, "sd": 0.3, "min": 0.0, "max": 2.0},
            "reputation_concern": {"mean": 0.5, "sd": 0.3, "min": 0.0, "max": 2.0},
            "rule_salience": {"mean": 0.3, "sd": 0.2, "min": 0.0, "max": 1.0},
            "learning_rate": {"mean": 0.3, "sd": 0.2, "min": 0.01, "max": 1.0},
            "exploration": {"mean": 0.1, "sd": 0.1, "min": 0.0, "max": 0.5},
            "memory": {"mean": 0.9, "sd": 0.1, "min": 0.0, "max": 1.0},
            "baseline_prosocial": {"mean": 0.0, "sd": 0.5, "min": -2.0, "max": 2.0},
        }
    parsed_priors = parse_priors(priors)

    if latent_classes is None:
        latent_classes = {
            "selfish": {"weight": 0.25, "shifts": {"fairness_alpha": -0.3, "prosociality": -0.5}},
            "fairness_minded": {"weight": 0.35, "shifts": {"fairness_alpha": 0.4}},
            "reciprocator": {"weight": 0.20, "shifts": {"reciprocity": 0.3}},
            "strategic": {"weight": 0.10, "shifts": {"strategic_depth": 1.0, "noise_lambda": 2.0}},
            "conformist": {"weight": 0.10, "shifts": {"norm_weight": 0.5, "rule_salience": 0.3}},
        }

    personas = PersonaGenerator(priors=parsed_priors, latent_classes=latent_classes)

    results: List[EvalResult] = []
    total_samples = 0

    for spec in specs:
        state = _make_game_state(spec)
        action_counts: Dict[str, int] = {a.name: 0 for a in state.available_actions}
        action_values: List[float] = []
        invalid_count = 0

        t0 = time.monotonic()
        for _ in range(n_samples_per_spec):
            persona = personas.sample(rng, persona_id="eval", mean_shifts={}, extra_sd={})
            try:
                result = backend.sample_action(state, persona, rng)
                name = result.chosen.name
                if name in action_counts:
                    action_counts[name] += 1
                    val = result.chosen.value
                    if isinstance(val, (int, float)):
                        action_values.append(float(val))
                else:
                    invalid_count += 1
            except Exception:
                invalid_count += 1
        elapsed = (time.monotonic() - t0) * 1000

        # Compute distribution
        total = sum(action_counts.values())
        dist = {k: v / max(total, 1) for k, v in action_counts.items()}

        mean_act = float(np.mean(action_values)) if action_values else 0.0
        std_act = float(np.std(action_values)) if action_values else 0.0

        results.append(EvalResult(
            spec_hash=spec.spec_hash,
            game_name=spec.game_name,
            action_distribution=dist,
            mean_action=mean_act,
            std_action=std_act,
            invalid_rate=invalid_count / max(n_samples_per_spec, 1),
            diversity=_shannon_entropy(dist),
            n_samples=n_samples_per_spec,
            elapsed_ms=elapsed,
        ))
        total_samples += n_samples_per_spec

    # Aggregate
    mean_invalid = float(np.mean([r.invalid_rate for r in results])) if results else 0.0
    mean_div = float(np.mean([r.diversity for r in results])) if results else 0.0
    mean_elapsed = float(np.mean([r.elapsed_ms for r in results])) if results else 0.0

    return HarnessReport(
        backend_name=backend.metadata().get("backend", "unknown"),
        n_specs=len(specs),
        n_total_samples=total_samples,
        mean_invalid_rate=mean_invalid,
        mean_diversity=mean_div,
        mean_elapsed_ms=mean_elapsed,
        results=results,
        metadata=backend.metadata(),
    )
