"""Objective functions for optimization.

Metrics used by the evaluation harness:
  - Distributional distance to human data
  - Coverage and diversity
  - Rule sensitivity
  - Invalid rate penalty
  - Calibration error

All objectives are deterministic for offline runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


def distributional_distance(
    predicted: Dict[str, float],
    observed: Dict[str, float],
    method: str = "kl",
) -> float:
    """Compute distance between two action distributions.

    Parameters
    ----------
    predicted : dict
        Predicted action distribution {action: probability}.
    observed : dict
        Observed (human) action distribution.
    method : str
        Distance metric: "kl" (KL divergence), "js" (Jensen-Shannon),
        "hellinger", "l2" (Euclidean).

    Returns
    -------
    float
        Distance value (lower is better for all metrics).
    """
    all_keys = sorted(set(predicted) | set(observed))
    p = np.array([predicted.get(k, 0.0) for k in all_keys], dtype=float)
    q = np.array([observed.get(k, 0.0) for k in all_keys], dtype=float)

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p /= p.sum()
    q /= q.sum()

    if method == "kl":
        return float(np.sum(q * np.log(q / p)))
    elif method == "js":
        m = 0.5 * (p + q)
        return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))
    elif method == "hellinger":
        return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))
    elif method == "l2":
        return float(np.sqrt(np.sum((p - q) ** 2)))
    else:
        raise ValueError(f"Unknown distance method: {method}")


def diversity_score(distribution: Dict[str, float]) -> float:
    """Shannon entropy of an action distribution (higher = more diverse)."""
    probs = np.array(list(distribution.values()), dtype=float)
    probs = probs[probs > 1e-12]
    if len(probs) == 0:
        return 0.0
    probs /= probs.sum()
    return float(-np.sum(probs * np.log2(probs)))


def calibration_error(
    predicted_means: Dict[str, float],
    observed_means: Dict[str, float],
) -> float:
    """Mean absolute error between predicted and observed moment means."""
    common_keys = set(predicted_means) & set(observed_means)
    if not common_keys:
        return float("inf")
    errors = [abs(predicted_means[k] - observed_means[k]) for k in common_keys]
    return float(np.mean(errors))


def invalid_rate_penalty(invalid_rate: float, penalty_weight: float = 10.0) -> float:
    """Penalty for invalid responses."""
    return penalty_weight * invalid_rate


def effect_size_error(
    predicted_means: Dict[str, float],
    predicted_sds: Dict[str, float],
    target_d: float,
    condition_a: str,
    condition_b: str,
) -> float:
    """Compute error between simulated and target Cohen's d effect size.

    Parameters
    ----------
    predicted_means : dict
        {condition_name: mean_value} from simulation.
    predicted_sds : dict
        {condition_name: sd_value} from simulation.
    target_d : float
        Published meta-analytic effect size (Cohen's d).
    condition_a, condition_b : str
        The two conditions to compare.

    Returns
    -------
    float
        Absolute error: |simulated_d - target_d|.
    """
    if condition_a not in predicted_means or condition_b not in predicted_means:
        return float("inf")

    m_a = predicted_means[condition_a]
    m_b = predicted_means[condition_b]
    sd_a = predicted_sds.get(condition_a, 1.0)
    sd_b = predicted_sds.get(condition_b, 1.0)

    # Pooled SD
    pooled_sd = np.sqrt(0.5 * (sd_a ** 2 + sd_b ** 2))
    if pooled_sd < 1e-9:
        return float("inf")

    simulated_d = (m_a - m_b) / pooled_sd
    return abs(simulated_d - target_d)


@dataclass
class ObjectiveFunction:
    """Composite objective combining multiple metrics.

    Parameters
    ----------
    weights : dict
        Weights for each metric: distributional, diversity, calibration, invalid.
    human_data : dict, optional
        Observed human distributions for calibration.
    """
    weights: Dict[str, float]
    human_data: Optional[Dict[str, Dict[str, float]]] = None

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = {
                "distributional": 1.0,
                "diversity": 0.5,
                "calibration": 1.0,
                "invalid": 10.0,
            }

    def score(
        self,
        results: List[Dict[str, Any]],
    ) -> float:
        """Compute composite score from evaluation results.

        Lower is better.

        Parameters
        ----------
        results : list of dict
            Each dict must have: action_distribution, invalid_rate, diversity,
            and optionally mean_action.
        """
        total = 0.0

        for r in results:
            dist = r.get("action_distribution", {})

            # Diversity (we want high diversity, so negate)
            div = diversity_score(dist)
            total -= self.weights.get("diversity", 0.5) * div

            # Invalid rate penalty
            inv = r.get("invalid_rate", 0.0)
            total += self.weights.get("invalid", 10.0) * inv

            # Distributional distance to human data
            if self.human_data:
                spec_hash = r.get("spec_hash", "")
                if spec_hash in self.human_data:
                    d = distributional_distance(dist, self.human_data[spec_hash])
                    total += self.weights.get("distributional", 1.0) * d

        return total / max(len(results), 1)
