"""Selection optimization — optimize mixture weights for strategy modules.

Uses CMA-ES or simple coordinate descent (offline-friendly, no LLM needed).
Finds the best mixture of strategies to match observed human behaviour.

References:
  Hansen & Ostermeier (2001). Completely Derandomized Self-Adaptation
  in Evolution Strategies. Evolutionary Computation 9(2):159-195.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SelectionResult:
    """Result of selection optimization."""
    best_weights: Dict[str, float]
    best_score: float
    history: List[float]
    n_evaluations: int


def _softmax_weights(raw: np.ndarray, names: List[str]) -> Dict[str, float]:
    """Convert raw parameters to valid probability weights via softmax."""
    exp = np.exp(raw - np.max(raw))
    probs = exp / exp.sum()
    return dict(zip(names, probs.tolist()))


class SelectionOptimizer:
    """Optimize mixture weights over strategy modules.

    Uses coordinate descent with random restarts (simple, robust,
    no external dependencies).  Can be replaced with CMA-ES if
    ``cma`` package is available.
    """

    def __init__(
        self,
        strategy_names: List[str],
        objective_fn: Callable[[Dict[str, float]], float],
        max_evaluations: int = 500,
        seed: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        strategy_names : list
            Names of strategies to weight.
        objective_fn : callable
            Takes weights dict → returns scalar score (lower is better).
        max_evaluations : int
            Budget of objective function evaluations.
        seed : int
            RNG seed.
        """
        self.names = strategy_names
        self.objective = objective_fn
        self.max_eval = max_evaluations
        self.rng = np.random.default_rng(seed)
        self.dim = len(strategy_names)

    def optimize(self) -> SelectionResult:
        """Run optimization and return the best weights found."""
        best_raw = np.zeros(self.dim)
        best_score = self.objective(_softmax_weights(best_raw, self.names))
        history = [best_score]
        n_eval = 1

        # Try CMA-ES if available
        try:
            return self._optimize_cma(best_raw, best_score, history, n_eval)
        except ImportError:
            pass

        # Fallback: coordinate descent with perturbation
        return self._optimize_coordinate_descent(best_raw, best_score, history, n_eval)

    def _optimize_coordinate_descent(
        self,
        init_raw: np.ndarray,
        init_score: float,
        history: List[float],
        n_eval: int,
    ) -> SelectionResult:
        """Simple coordinate descent with random perturbations."""
        current = init_raw.copy()
        best = current.copy()
        best_score = init_score
        step_size = 1.0

        while n_eval < self.max_eval:
            # Random perturbation
            direction = self.rng.standard_normal(self.dim)
            candidate = current + step_size * direction

            weights = _softmax_weights(candidate, self.names)
            score = self.objective(weights)
            n_eval += 1
            history.append(score)

            if score < best_score:
                best = candidate.copy()
                best_score = score
                current = candidate
            else:
                # Shrink step size slowly
                step_size *= 0.995

            # Random restart occasionally
            if n_eval % 50 == 0:
                current = self.rng.standard_normal(self.dim)
                step_size = 0.5

        return SelectionResult(
            best_weights=_softmax_weights(best, self.names),
            best_score=best_score,
            history=history,
            n_evaluations=n_eval,
        )

    def _optimize_cma(
        self,
        init_raw: np.ndarray,
        init_score: float,
        history: List[float],
        n_eval: int,
    ) -> SelectionResult:
        """CMA-ES optimization (requires ``cma`` package)."""
        import cma  # type: ignore

        def _obj(x: np.ndarray) -> float:
            weights = _softmax_weights(np.array(x), self.names)
            return self.objective(weights)

        opts = cma.CMAOptions()
        opts["maxfevals"] = self.max_eval - n_eval
        opts["seed"] = int(self.rng.integers(0, 2**31))
        opts["verbose"] = -9  # suppress output

        es = cma.CMAEvolutionStrategy(init_raw.tolist(), 0.5, opts)
        while not es.stop():
            solutions = es.ask()
            es.tell(solutions, [_obj(s) for s in solutions])

        best_raw = np.array(es.result.xbest)
        best_weights = _softmax_weights(best_raw, self.names)
        best_score = es.result.fbest

        return SelectionResult(
            best_weights=best_weights,
            best_score=best_score,
            history=history + [best_score],
            n_evaluations=self.max_eval,
        )
