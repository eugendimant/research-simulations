from __future__ import annotations
from .specs import ExperimentSpec, GameSpec, TopicSpec, PopulationSpec

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List
import json
import csv

import numpy as np
from jsonschema import validate as js_validate

from .utils import parse_priors
from .persona import PersonaGenerator
from .context import ContextEngine
from .evidence import EvidenceStore
from .games.registry import make_game
from .hash_utils import stable_json_sha256
from .coverage import coverage as coverage_fn
from .logging_utils import new_run, log_event
from .scm import default_scm
from .survey import SurveySimulator, GRMItem

@dataclass
class SimulationResult:
    rows: List[Dict[str, Any]]
    summary: Dict[str, Any]

def load_experiment_spec(path: Path, schema_path: Optional[Path] = None) -> ExperimentSpec:
    data = json.loads(path.read_text(encoding="utf-8"))
    if schema_path is not None:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        js_validate(instance=data, schema=schema)

    game = GameSpec(**data["game"])
    topic = TopicSpec(**data["topic"])
    pop = PopulationSpec(**data.get("population", {}))
    ctx = data.get("context", {})
    return ExperimentSpec(game=game, topic=topic, population=pop, context=ctx)

def _numeric_mean(rows: List[Dict[str, Any]], key: str) -> float | None:
    xs = []
    for r in rows:
        v = r.get(key)
        if isinstance(v, (int, float)):
            xs.append(float(v))
        elif isinstance(v, str):
            try:
                xs.append(float(v))
            except Exception:
                pass
    if not xs:
        return None
    return float(sum(xs) / len(xs))

def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def simulate(
    spec: ExperimentSpec,
    n: int,
    seed: int,
    priors_path: Path,
    latent_classes_path: Path,
    evidence_store_path: Path,
    schema_path: Path | None,
    return_traces: bool = False,
    corpus_path: Path | None = None,
    safe_mode: bool = False,
    min_coverage: float = 0.0,
    abort_on_conflict: bool = False,
    log_path: str | None = None,
) -> SimulationResult:
    rng = np.random.default_rng(int(seed))
    runctx = new_run()
    log_event('run_start', {'run_id': runctx.run_id, 'seed': int(seed), 'n': int(n)}, path=log_path)

    _ = corpus_path  # unused (reserved for future corpus-driven context)

    priors_json = json.loads(priors_path.read_text(encoding='utf-8'))
    priors = parse_priors(priors_json)
    latent_classes = json.loads(latent_classes_path.read_text(encoding='utf-8'))
    personas = PersonaGenerator(priors=priors, latent_classes=latent_classes)

    store = EvidenceStore.load(evidence_store_path)
    ctx_engine = ContextEngine(store=store)

    game = make_game(spec.game.name)
    features = spec.to_feature_dict()
    # Apply optional do-interventions to context features (SCM override)
    scm = default_scm()
    if isinstance(spec.context.get('interventions', None), dict):
        for k, v in spec.context['interventions'].items():
            scm.do(k, v)
    features = scm.apply(features)
    ctx = ctx_engine.build(features)

    rows: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}

    # context helpers
    is_ingroup = int(bool(spec.context.get("ingroup_partner", 0)))

    # group simulation for public goods (optional)
    if spec.game.name == "public_goods" and bool(spec.context.get("simulate_group", False)):
        group_size = int(spec.game.params.get("group_size", 4))
        for i in range(n):
            group = [
                personas.sample(rng, persona_id=f"G{i}_0", mean_shifts=ctx.mean_shifts, extra_sd=ctx.extra_sd, intervention=None)
            ] + [
                personas.sample(rng, persona_id=f"G{i}_{j}", mean_shifts=ctx.mean_shifts, extra_sd=ctx.extra_sd, intervention=None)
                for j in range(1, group_size)
            ]
            contribs, pays, gtrace = game.simulate_group(rng, group, spec.game.params)
            for j, person in enumerate(group):
                counts[person.latent_class] = counts.get(person.latent_class, 0) + 1
                row: Dict[str, Any] = {
                    "group_id": f"G{i}",
                    "member_idx": j,
                    "persona_id": person.id,
                    "latent_class": person.latent_class,
                    "game": spec.game.name,
                    "topic": spec.topic.name,
                    "act::contribute": float(contribs[j]),
                    "pay::A": float(pays[j]),
                }
                row.update({f"param::{k}": float(v) for k, v in person.params.items()})
                if return_traces:
                    row["trace_json"] = json.dumps({
                        "matched_evidence": ctx.matched_evidence_ids,
                        "mean_shifts": ctx.mean_shifts,
                        "extra_sd": ctx.extra_sd,
                        "conflict_report": ctx.conflict_report,
                        "group_trace": gtrace,
                    }, ensure_ascii=False)
                rows.append(row)
    else:
        # default: per-person simulation (optionally with opponent)
        needs_opponent = spec.game.name in {"pd", "stag_hunt", "ultimatum", "trust", "sender_receiver", "gift_exchange", "repeated_pd"}
        for i in range(n):
            a = personas.sample(rng, persona_id=f"A_{i}", mean_shifts=ctx.mean_shifts, extra_sd=ctx.extra_sd, intervention=None)
            b = None
            if needs_opponent:
                b = personas.sample(rng, persona_id=f"B_{i}", mean_shifts=ctx.mean_shifts, extra_sd=ctx.extra_sd, intervention=None)

            counts[a.latent_class] = counts.get(a.latent_class, 0) + 1
            if b is not None:
                counts[b.latent_class] = counts.get(b.latent_class, 0) + 1

            outcome = game.simulate_one(rng, a, b, dict(spec.game.params))

            # Optional survey battery (actions-only)
            sv = {}
            survey_cfg = spec.context.get("survey", None)
            if isinstance(survey_cfg, dict) and isinstance(survey_cfg.get("items", None), list):
                theta_param = str(survey_cfg.get("theta_param", "prosociality"))
                theta = float(a.params.get(theta_param, 0.0))
                items = [GRMItem(name=str(i["name"]), a=float(i["a"]), thresholds=[float(x) for x in i["thresholds"]]) for i in survey_cfg["items"]]
                sv = SurveySimulator(items).simulate(theta=theta, rng=rng)

            row: Dict[str, Any] = {
                "persona_id": a.id,
                "latent_class": a.latent_class,
                "game": spec.game.name,
                "topic": spec.topic.name,
                "context::ingroup_partner": is_ingroup,
            }
            row.update({f"param::{k}": float(v) for k, v in a.params.items()})
            row.update({f"act::{k}": v for k, v in outcome.actions.items()})
            row.update({f"pay::{k}": float(v) for k, v in outcome.payoffs.items()})
            for k, v in sv.items():
                row[f"sv::{k}"] = int(v)

            if return_traces:
                row["trace_json"] = json.dumps({
                    "matched_evidence": ctx.matched_evidence_ids,
                    "mean_shifts": ctx.mean_shifts,
                    "extra_sd": ctx.extra_sd,
                        "conflict_report": ctx.conflict_report,
                    "task_trace": outcome.trace
                }, ensure_ascii=False)

            rows.append(row)

    # Aggregate means
    action_means: Dict[str, float] = {}
    payoff_means: Dict[str, float] = {}
    for k in sorted({k for r in rows for k in r.keys()}):
        if k.startswith("act::"):
            m = _numeric_mean(rows, k)
            if m is not None:
                action_means[k] = m
        if k.startswith("pay::"):
            m = _numeric_mean(rows, k)
            if m is not None:
                payoff_means[k] = m

    summary: Dict[str, Any] = {
        "n_rows": len(rows),
        "game": spec.game.name,
        "topic": spec.topic.name,
        "matched_evidence_ids": ctx.matched_evidence_ids,
        "mean_shifts": ctx.mean_shifts,
        "extra_sd": ctx.extra_sd,
                        "conflict_report": ctx.conflict_report,
        "action_means": action_means,
        "payoff_means": payoff_means,
        "latent_class_counts": counts,
        "coverage_report": coverage_fn(store, features).__dict__,
    }

    log_event('run_end', {'run_id': runctx.run_id, 'rows': len(rows)}, path=log_path)
    return SimulationResult(rows=rows, summary=summary)