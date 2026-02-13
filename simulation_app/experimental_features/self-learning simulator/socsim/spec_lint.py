from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

REQUIRED_TOPLEVEL = ["game", "topic", "population", "context"]

@dataclass
class LintResult:
    ok: bool
    errors: List[str]
    warnings: List[str]

def lint_spec(spec: Dict[str, Any]) -> LintResult:
    errors: List[str] = []
    warnings: List[str] = []
    for k in REQUIRED_TOPLEVEL:
        if k not in spec:
            errors.append(f"missing:{k}")
    game = spec.get("game") or {}
    if isinstance(game, dict) and "name" not in game:
        errors.append("missing:game.name")
    ctx = spec.get("context") or {}
    if isinstance(ctx, dict) and "stakes_level" not in ctx:
        warnings.append("defaulted:context.stakes_level=low")
    return LintResult(ok=(len(errors)==0), errors=errors, warnings=warnings)
