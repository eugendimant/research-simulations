"""Mixture orchestration — produce heterogeneous populations from strategy weights.

Given a set of mixture weights over strategy modules, generate a
population of agents whose behaviour is consistent with those weights.
This implements the "strategic sample" concept from the paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..persona import Persona, PersonaGenerator
from ..agents.backend import GameState, Action
from .strategy_router import StrategyRouter, STRATEGY_LIBRARY


@dataclass
class MixtureSpec:
    """Specification for a strategic mixture."""
    weights: Dict[str, float]  # strategy_name → weight
    n_agents: int = 100

    def validate(self) -> bool:
        total = sum(max(0, v) for v in self.weights.values())
        return total > 0 and all(k in STRATEGY_LIBRARY for k in self.weights)


@dataclass
class PopulationSample:
    """A sampled population with assigned strategies."""
    agents: List[Persona]
    strategy_assignments: Dict[str, str]  # persona_id → strategy_name
    effective_weights: Dict[str, float]   # empirical proportions


def sample_population(
    spec: MixtureSpec,
    persona_gen: PersonaGenerator,
    rng: np.random.Generator,
    mean_shifts: Dict[str, float] | None = None,
    extra_sd: Dict[str, float] | None = None,
) -> PopulationSample:
    """Sample a population of agents with strategy assignments.

    Each agent is assigned ONE strategy from the mixture, proportional
    to the specified weights.  This creates behavioural heterogeneity
    that mirrors real experimental populations.
    """
    mean_shifts = mean_shifts or {}
    extra_sd = extra_sd or {}

    # Normalise weights
    names = list(spec.weights.keys())
    ws = np.array([max(0, spec.weights[n]) for n in names], dtype=float)
    total = ws.sum()
    if total <= 0:
        ws = np.ones(len(names)) / len(names)
    else:
        ws /= total

    # Assign strategies
    assignments_idx = rng.choice(len(names), size=spec.n_agents, p=ws)
    strategy_map: Dict[str, str] = {}
    agents: List[Persona] = []
    counts: Dict[str, int] = {n: 0 for n in names}

    for i, idx in enumerate(assignments_idx):
        strat_name = names[idx]
        counts[strat_name] += 1

        persona = persona_gen.sample(
            rng=rng,
            persona_id=f"pop_{i}",
            mean_shifts=mean_shifts,
            extra_sd=extra_sd,
            intervention=None,
        )
        agents.append(persona)
        strategy_map[persona.id] = strat_name

    # Effective weights
    eff = {n: counts[n] / spec.n_agents for n in names}

    return PopulationSample(
        agents=agents,
        strategy_assignments=strategy_map,
        effective_weights=eff,
    )
