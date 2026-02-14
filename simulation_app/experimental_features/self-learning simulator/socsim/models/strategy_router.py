"""Strategy Router — maps personas to strategy mixtures.

Each persona has a ``strategy_prior`` distribution over the 12 strategy
modules.  The router:
  1. Reads the persona's strategy_prior
  2. Adjusts weights based on game context (e.g., repeated games boost RL)
  3. Validates persona-strategy alignment (high prosociality → fairness strategies)
  4. Samples or mixes across strategies to produce final action probabilities

This is the core "selection" mechanism from the paper:
  fit mixture weights over a finite strategy library.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from ..persona import Persona
from ..agents.backend import GameState
from .strategies.base import Strategy, STRATEGY_VALIDATED_GAMES

# Import all strategies
from .strategies import (
    LevelKStrategy,
    CognitiveHierarchyStrategy,
    QREStrategy,
    InequityAversionStrategy,
    ReciprocityStrategy,
    ReputationSensitiveStrategy,
    NormSensitiveStrategy,
    RuleFollowingStrategy,
    NoisyBestResponseStrategy,
    RLBanditStrategy,
    ImitationStrategy,
    IntergroupBiasStrategy,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default strategy library (12 strategies)
# ---------------------------------------------------------------------------

STRATEGY_LIBRARY: Dict[str, Strategy] = {
    "level_k": LevelKStrategy(),
    "cognitive_hierarchy": CognitiveHierarchyStrategy(),
    "qre": QREStrategy(),
    "inequity_aversion": InequityAversionStrategy(),
    "reciprocity": ReciprocityStrategy(),
    "reputation_sensitive": ReputationSensitiveStrategy(),
    "norm_sensitive": NormSensitiveStrategy(),
    "rule_following": RuleFollowingStrategy(),
    "noisy_best_response": NoisyBestResponseStrategy(),
    "rl_bandit": RLBanditStrategy(),
    "imitation": ImitationStrategy(),
    "intergroup_bias": IntergroupBiasStrategy(),
}

# ---------------------------------------------------------------------------
# Default priors by latent class
# ---------------------------------------------------------------------------

DEFAULT_STRATEGY_PRIORS: Dict[str, Dict[str, float]] = {
    "selfish": {
        "noisy_best_response": 0.30,
        "level_k": 0.15,
        "qre": 0.15,
        "cognitive_hierarchy": 0.10,
        "inequity_aversion": 0.05,
        "reciprocity": 0.04,
        "reputation_sensitive": 0.04,
        "norm_sensitive": 0.03,
        "rule_following": 0.04,
        "rl_bandit": 0.03,
        "imitation": 0.04,
        "intergroup_bias": 0.03,
    },
    "fairness_minded": {
        "inequity_aversion": 0.28,
        "norm_sensitive": 0.18,
        "noisy_best_response": 0.12,
        "reciprocity": 0.10,
        "reputation_sensitive": 0.10,
        "qre": 0.05,
        "cognitive_hierarchy": 0.04,
        "level_k": 0.03,
        "rule_following": 0.02,
        "rl_bandit": 0.01,
        "imitation": 0.04,
        "intergroup_bias": 0.03,
    },
    "reciprocator": {
        "reciprocity": 0.30,
        "norm_sensitive": 0.14,
        "inequity_aversion": 0.14,
        "noisy_best_response": 0.08,
        "reputation_sensitive": 0.10,
        "qre": 0.05,
        "rl_bandit": 0.05,
        "cognitive_hierarchy": 0.03,
        "level_k": 0.01,
        "rule_following": 0.01,
        "imitation": 0.05,
        "intergroup_bias": 0.04,
    },
    "strategic": {
        "cognitive_hierarchy": 0.23,
        "level_k": 0.18,
        "qre": 0.18,
        "noisy_best_response": 0.10,
        "rl_bandit": 0.10,
        "reciprocity": 0.05,
        "inequity_aversion": 0.03,
        "reputation_sensitive": 0.03,
        "norm_sensitive": 0.02,
        "rule_following": 0.02,
        "imitation": 0.03,
        "intergroup_bias": 0.03,
    },
    "conformist": {
        "norm_sensitive": 0.25,
        "rule_following": 0.17,
        "reputation_sensitive": 0.12,
        "imitation": 0.15,  # conformists are strong imitators
        "noisy_best_response": 0.08,
        "inequity_aversion": 0.08,
        "reciprocity": 0.05,
        "qre": 0.03,
        "cognitive_hierarchy": 0.02,
        "level_k": 0.02,
        "rl_bandit": 0.01,
        "intergroup_bias": 0.02,
    },
    # New latent class: partisan/tribal agents
    "partisan": {
        "intergroup_bias": 0.35,
        "norm_sensitive": 0.15,
        "imitation": 0.12,
        "noisy_best_response": 0.10,
        "inequity_aversion": 0.08,
        "reciprocity": 0.05,
        "reputation_sensitive": 0.05,
        "rule_following": 0.04,
        "qre": 0.03,
        "cognitive_hierarchy": 0.01,
        "level_k": 0.01,
        "rl_bandit": 0.01,
    },
    "default": {
        "noisy_best_response": 0.20,
        "inequity_aversion": 0.13,
        "reciprocity": 0.10,
        "norm_sensitive": 0.10,
        "qre": 0.09,
        "cognitive_hierarchy": 0.07,
        "level_k": 0.06,
        "reputation_sensitive": 0.06,
        "rule_following": 0.05,
        "rl_bandit": 0.04,
        "imitation": 0.05,
        "intergroup_bias": 0.05,
    },
}

# ---------------------------------------------------------------------------
# Game-context adjustments
# ---------------------------------------------------------------------------

GAME_STRATEGY_BOOSTS: Dict[str, Dict[str, float]] = {
    # Repeated games: boost RL, reciprocity, and imitation
    "repeated_pd": {"rl_bandit": 2.0, "reciprocity": 1.5, "imitation": 1.5},
    "repeated_trust": {"rl_bandit": 2.0, "reciprocity": 1.5, "reputation_sensitive": 1.5, "imitation": 1.3},
    "repeated_public_goods": {"rl_bandit": 2.0, "norm_sensitive": 1.3, "imitation": 1.5},
    # Strategic games: boost level-k and CH
    "beauty_contest": {"level_k": 3.0, "cognitive_hierarchy": 2.5, "qre": 1.5},
    "coordination_min_effort": {"level_k": 1.5, "cognitive_hierarchy": 1.5, "imitation": 1.3},
    # Social/dictator: boost fairness, norms, and intergroup when relevant
    "dictator": {"inequity_aversion": 1.5, "norm_sensitive": 1.3, "reputation_sensitive": 1.3, "intergroup_bias": 1.3},
    "ultimatum": {"inequity_aversion": 1.5, "reciprocity": 1.3},
    "trust": {"reciprocity": 1.5, "reputation_sensitive": 1.3, "intergroup_bias": 1.3},
    "public_goods": {"norm_sensitive": 1.5, "reciprocity": 1.3, "imitation": 1.3},
    # Dishonesty: boost rule-following and norms
    "die_roll": {"rule_following": 1.5, "norm_sensitive": 1.3},
    "bribery_game": {"norm_sensitive": 1.5, "rule_following": 1.3},
    # Intergroup games: strong intergroup bias activation
    "intergroup_dictator": {"intergroup_bias": 2.5, "inequity_aversion": 1.3},
    "intergroup_trust": {"intergroup_bias": 2.5, "reciprocity": 1.3},
    "intergroup_pd": {"intergroup_bias": 2.0, "reciprocity": 1.3},
    # Gift exchange: reciprocity + reputation
    "gift_exchange": {"reciprocity": 2.0, "reputation_sensitive": 1.5, "norm_sensitive": 1.3},
    # Money request: strategic thinking
    "money_request_11_20": {"level_k": 2.0, "cognitive_hierarchy": 1.8, "qre": 1.5},
    # Stag hunt: coordination + trust
    "stag_hunt": {"norm_sensitive": 1.5, "imitation": 1.5, "level_k": 1.3},
    # Common pool: norms + self-interest tension
    "common_pool_resource": {"norm_sensitive": 1.5, "noisy_best_response": 1.3, "rule_following": 1.3},
    # Tullock contest: strategic competition
    "tullock_contest": {"noisy_best_response": 1.5, "level_k": 1.3, "qre": 1.3},
}


class StrategyRouter:
    """Maps persona → strategy mixture → action distribution.

    The router computes a weighted mixture over strategy modules,
    where weights come from the persona's latent class prior adjusted
    by game-specific context boosts.
    """

    def __init__(
        self,
        strategies: Dict[str, Strategy] | None = None,
        priors: Dict[str, Dict[str, float]] | None = None,
        game_boosts: Dict[str, Dict[str, float]] | None = None,
    ) -> None:
        self.strategies = strategies or STRATEGY_LIBRARY
        self.priors = priors or DEFAULT_STRATEGY_PRIORS
        self.game_boosts = game_boosts or GAME_STRATEGY_BOOSTS

    def get_weights(self, persona: Persona, game_name: str) -> Dict[str, float]:
        """Compute strategy weights for a persona in a given game."""
        # Start with class-specific prior
        cls = persona.latent_class
        base = dict(self.priors.get(cls, self.priors.get("default", {})))

        # Apply game-specific boosts
        boosts = self.game_boosts.get(game_name, {})
        for strat, mult in boosts.items():
            if strat in base:
                base[strat] *= mult

        # Normalise
        total = sum(max(0, v) for v in base.values())
        if total <= 0:
            n = len(base)
            return {k: 1.0 / n for k in base}
        return {k: max(0, v) / total for k, v in base.items()}

    def distribution(self, state: GameState, persona: Persona) -> Dict[str, float]:
        """Compute the final action distribution as a mixture over strategies."""
        weights = self.get_weights(persona, state.game_name)

        # Collect per-strategy distributions
        mixed: Dict[str, float] = {}
        for strat_name, w in weights.items():
            if w < 1e-6:
                continue
            strat = self.strategies.get(strat_name)
            if strat is None:
                continue
            try:
                dist = strat.action_probabilities(state, persona)
                for action_name, p in dist.items():
                    mixed[action_name] = mixed.get(action_name, 0.0) + w * p
            except Exception:
                continue

        # Normalise
        total = sum(max(0, v) for v in mixed.values())
        if total <= 0:
            n = max(len(state.available_actions), 1)
            return {a.name: 1.0 / n for a in state.available_actions}
        return {k: max(0, v) / total for k, v in mixed.items()}

    def _apply_persona_alignment(
        self, weights: Dict[str, float], persona: Persona,
    ) -> Dict[str, float]:
        """Adjust strategy weights based on persona trait alignment.

        High prosociality → boost fairness strategies.
        High strategic depth → boost reasoning strategies.
        High conformity → boost imitation/norms.
        High ingroup bias → boost intergroup strategy.
        """
        prosoc = float(persona.params.get("prosociality", 0.0))
        depth = float(persona.params.get("strategic_depth", 0.0))
        ig_bias = float(persona.params.get("ingroup_bias", 0.0))
        conformity = float(persona.params.get("conformity", 0.0))

        adjusted = dict(weights)

        # Prosocial agents should use fairness-oriented strategies more
        if prosoc > 0.6:
            for s in ("inequity_aversion", "reciprocity", "norm_sensitive"):
                if s in adjusted:
                    adjusted[s] *= 1.3
        elif prosoc < -0.3:
            for s in ("noisy_best_response", "level_k", "qre"):
                if s in adjusted:
                    adjusted[s] *= 1.2

        # Strategic agents should use reasoning strategies
        if depth > 2.0:
            for s in ("cognitive_hierarchy", "level_k", "qre"):
                if s in adjusted:
                    adjusted[s] *= 1.3

        # Conformist agents should imitate more
        if conformity > 0.5:
            if "imitation" in adjusted:
                adjusted["imitation"] *= 1.5
            if "norm_sensitive" in adjusted:
                adjusted["norm_sensitive"] *= 1.2

        # Biased agents should use intergroup strategy
        if ig_bias > 0.3:
            if "intergroup_bias" in adjusted:
                adjusted["intergroup_bias"] *= 1.5

        # Renormalize
        total = sum(max(0, v) for v in adjusted.values())
        if total > 0:
            adjusted = {k: max(0, v) / total for k, v in adjusted.items()}
        return adjusted

    def sample(
        self,
        state: GameState,
        persona: Persona,
        rng: np.random.Generator,
    ) -> tuple[str, Dict[str, float]]:
        """Sample a single strategy (not a mixture) from the weights.

        Returns (strategy_name, action_distribution_from_that_strategy).
        """
        weights = self.get_weights(persona, state.game_name)
        weights = self._apply_persona_alignment(weights, persona)
        names = list(weights.keys())
        ws = np.array([weights[n] for n in names], dtype=float)
        ws /= ws.sum()
        chosen = str(rng.choice(names, p=ws))
        strat = self.strategies.get(chosen)
        if strat is None:
            strat = self.strategies.get("noisy_best_response", NoisyBestResponseStrategy())
        dist = strat.action_probabilities(state, persona, rng)
        return chosen, dist
