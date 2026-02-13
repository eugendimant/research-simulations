from __future__ import annotations
from typing import Any, Dict

def default_spec(game_name: str = "ultimatum") -> Dict[str, Any]:
    return {
        "game": {"name": game_name, "params": {"endowment": 10.0}},
        "topic": {"name": "baseline", "tags": [], "notes": ""},
        "population": {"label": "adult_online"},
        "context": {"stakes_level": "low"}
    }
