from __future__ import annotations
from typing import Dict, Type

from .base import Game

from .dictator import DictatorGame
from .pd import PrisonersDilemma
from .ultimatum import UltimatumGame
from .trust import TrustGame
from .public_goods import PublicGoodsGame
from .public_goods_punishment import PublicGoodsWithPunishment
from .repeated_public_goods import RepeatedPublicGoods
from .repeated_pd import RepeatedPD
from .repeated_trust import RepeatedTrust

from .sender_receiver import SenderReceiverGame
from .die_roll import DieRollTask
from .gift_exchange import GiftExchangeGame

from .risk_holt_laury import HoltLauryRisk
from .time_mpl import TimeMPL
from .bdm import BDMTask
from .discrete_choice import DiscreteChoiceTask

from .survey_likert import SurveyLikert

from .coordination_min_effort import CoordinationMinEffort
from .stag_hunt import StagHunt
from .beauty_contest import BeautyContest
from .common_pool_resource import CommonPoolResource
from .tullock_contest import TullockContest
from .bribery_game import BriberyGame
from .money_request_11_20 import MoneyRequest1120


_REGISTRY: Dict[str, Type[Game]] = {
    # classic economic games
    "dictator": DictatorGame,
    "ultimatum": UltimatumGame,
    "trust": TrustGame,
    "public_goods": PublicGoodsGame,
    "public_goods_punishment": PublicGoodsWithPunishment,
    "pd": PrisonersDilemma,

    # repeated
    "repeated_public_goods": RepeatedPublicGoods,
    "repeated_pd": RepeatedPD,
    "repeated_trust": RepeatedTrust,

    # information / dishonesty
    "sender_receiver": SenderReceiverGame,
    "die_roll": DieRollTask,

    # labor / reciprocity
    "gift_exchange": GiftExchangeGame,

    # elicitation tasks
    "risk_holt_laury": HoltLauryRisk,
    "holt_laury": HoltLauryRisk,
    "time_mpl": TimeMPL,
    "mpl_time": TimeMPL,
    "bdm": BDMTask,
    "discrete_choice": DiscreteChoiceTask,

    # survey
    "survey_likert": SurveyLikert,

    # less common / broader set
    "coordination_min_effort": CoordinationMinEffort,
    "stag_hunt": StagHunt,
    "beauty_contest": BeautyContest,
    "common_pool_resource": CommonPoolResource,
    "tullock_contest": TullockContest,
    "bribery_game": BriberyGame,

    # parameterised game families
    "money_request_11_20": MoneyRequest1120,
}

# Public alias for backwards compatibility
GAME_REGISTRY = _REGISTRY


def make_game(name: str) -> Game:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown game: {name}. Available: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[name]()
