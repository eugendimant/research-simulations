"""Offline strategy modules for SocSim.

Each strategy takes a game state + persona parameters and returns
action probabilities.  All strategies produce valid probability
distributions (non-negative, sum to 1, respect action constraints).

Strategies (12 total):
  - Level-K reasoning (Stahl & Wilson 1994)
  - Cognitive Hierarchy (Camerer et al. 2004)
  - QRE with convergence + game-calibrated λ (McKelvey & Palfrey 1995)
  - Inequity Aversion (Fehr & Schmidt 1999)
  - Reciprocity (Rabin 1993)
  - Reputation Sensitive (Benabou & Tirole 2006)
  - Norm Sensitive (Krupka & Weber 2013)
  - Rule Following (Kimbrough & Vostroknutov 2016)
  - Noisy Best Response (baseline logit)
  - RL Bandit / EWA with surprise learning (Camerer & Ho 1999)
  - Imitation / Social Learning (Bandura 1977, Cialdini 2005)
  - Intergroup Bias (Iyengar & Westwood 2015, Balliet et al. 2014)
"""
from .level_k import LevelKStrategy
from .cognitive_hierarchy import CognitiveHierarchyStrategy
from .qre import QREStrategy
from .inequity_aversion import InequityAversionStrategy
from .reciprocity import ReciprocityStrategy
from .reputation_sensitive import ReputationSensitiveStrategy
from .norm_sensitive import NormSensitiveStrategy
from .rule_following import RuleFollowingStrategy
from .noisy_best_response import NoisyBestResponseStrategy
from .rl_bandit import RLBanditStrategy
from .imitation import ImitationStrategy
from .intergroup_bias import IntergroupBiasStrategy

__all__ = [
    "LevelKStrategy",
    "CognitiveHierarchyStrategy",
    "QREStrategy",
    "InequityAversionStrategy",
    "ReciprocityStrategy",
    "ReputationSensitiveStrategy",
    "NormSensitiveStrategy",
    "RuleFollowingStrategy",
    "NoisyBestResponseStrategy",
    "RLBanditStrategy",
    "ImitationStrategy",
    "IntergroupBiasStrategy",
]
