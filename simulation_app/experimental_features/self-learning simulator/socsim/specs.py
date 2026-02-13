from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class PopulationSpec:
    label: str = "adult_online"
    country: Optional[str] = None
    sampling_frame: Optional[str] = None

@dataclass
class TopicSpec:
    name: str
    tags: List[str] = field(default_factory=list)
    notes: str = ""

@dataclass
class GameSpec:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExperimentSpec:
    game: GameSpec
    topic: TopicSpec
    context: Dict[str, Any] = field(default_factory=dict)
    population: PopulationSpec = field(default_factory=PopulationSpec)

    def to_feature_dict(self) -> Dict[str, Any]:
        f: Dict[str, Any] = {}
        f["game"] = self.game.name
        f["topic"] = self.topic.name
        for t in self.topic.tags:
            f[f"topic_tag::{t}"] = 1
        for k, v in (self.context or {}).items():
            if isinstance(v, bool):
                f[f"ctx::{k}"] = int(v)
            elif isinstance(v, (int, float, str)):
                f[f"ctx::{k}"] = v
        if self.population.country:
            f["pop::country"] = self.population.country
        if self.population.sampling_frame:
            f["pop::sampling_frame"] = self.population.sampling_frame
        return f
