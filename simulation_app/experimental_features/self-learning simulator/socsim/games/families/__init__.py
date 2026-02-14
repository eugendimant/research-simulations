"""Game family generators for parameterised evaluation."""
from .money_request_family import MoneyRequestFamily, FamilySpec
from .dictator_family import DictatorFamily
from .public_goods_family import PublicGoodsFamily
from .dedup import deduplicate_specs

__all__ = [
    "MoneyRequestFamily",
    "DictatorFamily",
    "PublicGoodsFamily",
    "FamilySpec",
    "deduplicate_specs",
]
