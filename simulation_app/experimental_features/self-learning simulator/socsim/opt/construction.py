"""Construction optimization — optimize persona trait parameters.

Finds optimal numeric parameters for:
  - Prior distribution means and SDs
  - Strategy-specific numeric parameters
  - Latent class mixture weights

Uses bounded search (all parameters have declared bounds).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import numpy as np


@dataclass
class ParamBounds:
    """Bounds for a single parameter."""
    name: str
    lo: float
    hi: float
    default: float

    def clip(self, value: float) -> float:
        return float(np.clip(value, self.lo, self.hi))


@dataclass
class ConstructionResult:
    """Result of construction optimization."""
    best_params: Dict[str, float]
    best_score: float
    history: List[float]
    n_evaluations: int


# Default parameter bounds (matching priors.json structure)
DEFAULT_BOUNDS: List[ParamBounds] = [
    ParamBounds("fairness_alpha_mean", 0.0, 3.0, 0.8),
    ParamBounds("fairness_alpha_sd", 0.05, 1.5, 0.4),
    ParamBounds("fairness_beta_mean", 0.0, 2.0, 0.2),
    ParamBounds("fairness_beta_sd", 0.05, 1.0, 0.3),
    ParamBounds("prosociality_mean", -3.0, 3.0, 0.0),
    ParamBounds("prosociality_sd", 0.1, 2.0, 1.0),
    ParamBounds("noise_lambda_mean", 0.1, 20.0, 3.0),
    ParamBounds("noise_lambda_sd", 0.1, 5.0, 2.0),
    ParamBounds("strategic_depth_mean", 0.0, 4.0, 1.5),
    ParamBounds("strategic_depth_sd", 0.1, 2.0, 1.0),
    ParamBounds("risk_aversion_mean", 0.0, 2.0, 0.5),
    ParamBounds("risk_aversion_sd", 0.05, 1.0, 0.3),
    ParamBounds("norm_weight_mean", 0.0, 2.0, 0.3),
    ParamBounds("norm_weight_sd", 0.05, 1.0, 0.2),
    ParamBounds("reciprocity_mean", 0.0, 2.0, 0.5),
    ParamBounds("reciprocity_sd", 0.05, 1.0, 0.3),
]


class ConstructionOptimizer:
    """Optimize persona trait parameters to match human data.

    Uses bounded random search with shrinking step size.
    """

    def __init__(
        self,
        bounds: List[ParamBounds] | None = None,
        objective_fn: Callable[[Dict[str, float]], float] | None = None,
        max_evaluations: int = 300,
        seed: int = 42,
    ) -> None:
        self.bounds = bounds or DEFAULT_BOUNDS
        self.objective = objective_fn
        self.max_eval = max_evaluations
        self.rng = np.random.default_rng(seed)

    def optimize(self) -> ConstructionResult:
        """Run bounded optimization."""
        if self.objective is None:
            raise ValueError("objective_fn must be set")

        dim = len(self.bounds)
        # Initialise at defaults
        current = np.array([b.default for b in self.bounds])
        best = current.copy()
        best_params = self._to_dict(best)
        best_score = self.objective(best_params)
        history = [best_score]
        n_eval = 1

        step_sizes = np.array([(b.hi - b.lo) * 0.1 for b in self.bounds])

        while n_eval < self.max_eval:
            # Perturb
            direction = self.rng.standard_normal(dim)
            candidate = current + step_sizes * direction

            # Clip to bounds
            for i, b in enumerate(self.bounds):
                candidate[i] = b.clip(candidate[i])

            params = self._to_dict(candidate)
            score = self.objective(params)
            n_eval += 1
            history.append(score)

            if score < best_score:
                best = candidate.copy()
                best_score = score
                best_params = params
                current = candidate
            else:
                step_sizes *= 0.998  # slow shrink

            # Random restart
            if n_eval % 60 == 0:
                current = np.array([
                    self.rng.uniform(b.lo, b.hi) for b in self.bounds
                ])
                step_sizes = np.array([(b.hi - b.lo) * 0.05 for b in self.bounds])

        return ConstructionResult(
            best_params=best_params,
            best_score=best_score,
            history=history,
            n_evaluations=n_eval,
        )

    def _to_dict(self, values: np.ndarray) -> Dict[str, float]:
        return {b.name: float(values[i]) for i, b in enumerate(self.bounds)}
