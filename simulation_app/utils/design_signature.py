"""
Design signature — the second lookup key for the empirical registry.
====================================================================

`construct_matcher` answers "what is this variable about?". This module answers
"what shape is this measurement?", which is the other half of deciding whether a
published or measured benchmark may be applied to it.

The distinction matters because almost every misuse of a real number in a
simulator has the same shape: a value that is true under the conditions it was
measured under, applied where those conditions do not hold. Real examples this
guards against:

* straight-lining runs ~3.3% in a same-keyed 9-item block and ~0.3% in a
  mixed-keyed one — a 10x error if the keying is ignored;
* a 5-point attitude scale is flat (excess kurtosis about -0.5) while a 9-point
  polarised political scale is bimodal (+0.8) with 3-5% on the midpoint;
* answering identically across a whole 50-item instrument (~0.1%) is far rarer
  than within any one block of it;
* public-goods contributions decay across rounds, so a one-shot pooled mean does
  not describe a repeated game.

Everything here is derived from the design and the generated frame. Nothing is
asked of the user except `population`, which has no default beyond "unknown" —
and "unknown" never satisfies a stated condition, so an unanswered question makes
entries decline rather than fire on an assumption.

This module imports nothing from the engine and has no numpy/pandas requirement,
so it is cheap to construct per block and safe to unit-test on its own.
"""
from __future__ import annotations

__version__ = "1.0.0"

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Optional, Sequence

#: Populations an entry can be scoped to. "online_volunteer" is the openpsychometrics
#: style self-selected web sample; "online_panel" is a paid panel (Prolific/MTurk);
#: they are not interchangeable, and attention-check failure rates differ sharply
#: between either of those and a commercial opt-in panel.
POPULATIONS = ("online_panel", "online_volunteer", "student", "community",
               "commercial_panel", "unknown")

KEYING = ("mixed", "same", "unknown")
DESIGNS = ("between", "within", "mixed", "unknown")
REPETITION = ("one_shot", "repeated", "unknown")


@dataclass(frozen=True)
class DesignSignature:
    """What a benchmark needs to know about a measurement before speaking about it.

    Any field left as its "unknown" value fails every stated condition on that
    dimension. Declining is the intended behaviour: the caller keeps whatever it
    was already doing, which is explicit and inspectable, rather than taking a
    number that may not apply.
    """
    scale_points: Optional[int] = None
    items_per_block: Optional[int] = None
    n_blocks: Optional[int] = None
    keying: str = "unknown"
    design: str = "unknown"
    population: str = "unknown"
    repetition: str = "unknown"
    dv_kind: str = "unknown"          # likert_composite | single_rating | allocation |
                                      # binary | count | free_numeric | unknown
    n_conditions: Optional[int] = None
    item_position: Optional[int] = None   # 1-based index within the whole instrument
    instrument_length: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def key(self) -> str:
        """Stable, hashable form for memoizing registry lookups."""
        return "|".join(f"{k}={v}" for k, v in sorted(self.as_dict().items()))

    def with_position(self, item_position: int,
                      instrument_length: Optional[int] = None) -> "DesignSignature":
        d = self.as_dict()
        d["item_position"] = int(item_position)
        if instrument_length is not None:
            d["instrument_length"] = int(instrument_length)
        return DesignSignature(**d)

    @property
    def position_fraction(self) -> Optional[float]:
        """How far through the instrument this item sits, in [0, 1].

        Carelessness in real data has an onset partway through an instrument
        rather than being uniform across it, so fatigue belongs on this axis
        rather than on elapsed time.
        """
        if not self.item_position or not self.instrument_length:
            return None
        return max(0.0, min(1.0, float(self.item_position) / float(self.instrument_length)))


def _norm(value: Optional[str], allowed: Sequence[str]) -> str:
    v = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    return v if v in allowed else allowed[-1]       # the "unknown" sentinel is last


def infer_keying(reverse_flags: Optional[Iterable[Any]]) -> str:
    """mixed when the block contains at least one reverse-worded item, else same.

    `reverse_flags` is whatever the design carries per item: booleans, 0/1, or the
    strings the builder uses. None (no information) stays "unknown", which makes
    keying-scoped benchmarks decline rather than guess.
    """
    if reverse_flags is None:
        return "unknown"
    flags = list(reverse_flags)
    if not flags:
        return "unknown"
    def truthy(f: Any) -> bool:
        if isinstance(f, str):
            return f.strip().lower() in ("1", "true", "yes", "r", "reverse", "reversed")
        return bool(f)
    return "mixed" if any(truthy(f) for f in flags) else "same"


def infer_repetition(rounds: Optional[int]) -> str:
    if rounds is None:
        return "unknown"
    try:
        return "repeated" if int(rounds) > 1 else "one_shot"
    except Exception:
        return "unknown"


def for_block(
    *,
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None,
    scale_points: Optional[int] = None,
    n_items: Optional[int] = None,
    n_blocks: Optional[int] = None,
    reverse_flags: Optional[Iterable[Any]] = None,
    keying: Optional[str] = None,
    design_type: Optional[str] = None,
    n_conditions: Optional[int] = None,
    population: Optional[str] = None,
    rounds: Optional[int] = None,
    dv_kind: Optional[str] = None,
    item_position: Optional[int] = None,
    instrument_length: Optional[int] = None,
) -> DesignSignature:
    """Build a signature for one multi-item block or single DV.

    `scale_points` is taken directly when given, else derived from the integer
    endpoints. A non-integer or inverted range yields None, which makes every
    scale-scoped entry decline.
    """
    pts = scale_points
    if pts is None and scale_min is not None and scale_max is not None:
        try:
            lo, hi = float(scale_min), float(scale_max)
            if lo.is_integer() and hi.is_integer() and hi > lo:
                pts = int(hi - lo) + 1
        except Exception:
            pts = None
    if pts is not None and not (2 <= int(pts) <= 101):
        pts = None

    _dt = str(design_type or "").strip().lower()
    if _dt == "factorial":              # factorial is a between-subjects layout
        _dt = "between"
    _design = _norm(_dt, DESIGNS)

    return DesignSignature(
        scale_points=int(pts) if pts else None,
        items_per_block=int(n_items) if n_items else None,
        n_blocks=int(n_blocks) if n_blocks else None,
        keying=_norm(keying, KEYING) if keying else infer_keying(reverse_flags),
        design=_design,
        population=_norm(population, POPULATIONS),
        repetition=_norm(None, REPETITION) if rounds is None else infer_repetition(rounds),
        dv_kind=str(dv_kind or "unknown"),
        n_conditions=int(n_conditions) if n_conditions else None,
        item_position=int(item_position) if item_position else None,
        instrument_length=int(instrument_length) if instrument_length else None,
    )
