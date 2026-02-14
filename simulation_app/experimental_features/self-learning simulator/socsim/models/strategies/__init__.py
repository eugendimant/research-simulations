"""Offline strategy modules for SocSim.

Each strategy takes a game state + persona parameters and returns
action probabilities.  All strategies produce valid probability
distributions (non-negative, sum to 1, respect action constraints).
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
]
