from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class Node:
    name: str
    parents: List[str]

class SCM:
    """Minimal SCM container.

    - Stores a graph skeleton
    - Records do-interventions
    - Applies interventions as deterministic overrides
    """

    def __init__(self, nodes: List[Node]) -> None:
        self.nodes = {n.name: n for n in nodes}
        self.interventions: Dict[str, Any] = {}

    def do(self, var: str, value: Any) -> None:
        self.interventions[str(var)] = value

    def apply(self, features: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(features)
        for k, v in self.interventions.items():
            out[k] = v
        return out

def default_scm() -> SCM:
    nodes = [
        Node("stakes_level", []),
        Node("anonymity", []),
        Node("repeated", []),
        Node("identity_salience", []),
        Node("trust_belief", ["repeated"]),
        Node("social_image", ["anonymity"]),
    ]
    return SCM(nodes)
