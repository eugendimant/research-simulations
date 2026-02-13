from __future__ import annotations
from typing import Any, Dict
from .specs import ExperimentSpec
from . import __version__

def to_model_card_md(spec: ExperimentSpec, summary: Dict[str, Any]) -> str:
    # Deterministic text, no claims of accuracy.
    lines = []
    lines.append(f"# SOCSIM Model Card (v{__version__})")
    lines.append("")
    lines.append("## What this run is")
    lines.append(f"- game: {spec.game.name}")
    lines.append(f"- topic: {spec.topic.name}")
    if spec.topic.tags:
        lines.append(f"- topic tags: {', '.join(spec.topic.tags)}")
    lines.append("")
    lines.append("## Evidence + transport")
    cov = summary.get("coverage", {})
    lines.append(f"- coverage_score: {cov.get('coverage_score')}")
    lines.append(f"- matched_evidence: {len(summary.get('matched_evidence', []) or [])}")
    cr = summary.get("conflict_report", {})
    lines.append(f"- conflicts: {cr.get('n_conflicts')}")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append(f"- run_id: {summary.get('run_id')}")
    lines.append(f"- started_at_utc: {summary.get('started_at_utc')}")
    lines.append("")
    lines.append("## Safety constraints")
    lines.append("- Web expansion is metadata-only. No effect sizes are extracted from web text.")
    lines.append("- Causal shifts require structured evidence units that pass schema validation.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("- This simulator is not a guarantee of human behavior. It is a structured, auditable generator based on explicit priors and explicit evidence units.")
    return "\n".join(lines) + "\n"
