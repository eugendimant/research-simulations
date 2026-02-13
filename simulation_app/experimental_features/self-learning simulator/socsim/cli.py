from __future__ import annotations
import argparse
import json
from pathlib import Path

from .simulator import load_experiment_spec, simulate, write_csv
from .eval import compare_moments, write_report
from .corpus.store import CorpusStore
from .web.http import WebClient
from .web.harvest import HarvestConfig, harvest_bibliography, bibitem_to_atomic_unit

from .bench.fetch import fetch_all
from .bench.targets import build_all_targets
from .bench.run import run_benchmarks
from .bench.report import write_benchmark_report
from .bench.calibrate import calibrate_global_means_via_ridge
from .bench.scout import scout_openalex, hits_to_registry_candidates
from .bench.bootstrap import bootstrap_candidates

def cmd_simulate(args: argparse.Namespace) -> None:
    spec = load_experiment_spec(Path(args.spec), schema_path=Path(args.exp_schema) if args.exp_schema else None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = simulate(
        spec=spec,
        n=int(args.n),
        seed=int(args.seed),
        priors_path=Path(args.priors),
        latent_classes_path=Path(args.latent_classes),
        evidence_store_path=Path(args.evidence_store),
        schema_path=Path(args.ev_schema) if args.ev_schema else None,
        return_traces=bool(args.traces),
        corpus_path=Path(args.corpus) if args.corpus else None,
    )
    write_csv(res.rows, out / "sim.csv")
    (out / "summary.json").write_text(json.dumps(res.summary, indent=2), encoding="utf-8")
    print(f"Wrote {out/'sim.csv'} and {out/'summary.json'}")

def cmd_eval_moments(args: argparse.Namespace) -> None:
    cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    errors = compare_moments(Path(args.pred), Path(args.obs), cols)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    write_report(errors, outp)
    print(f"Wrote moment report to {outp}")

def cmd_corpus_expand_web(args: argparse.Namespace) -> None:
    corpus_path = Path(args.corpus)
    corpus = CorpusStore.load(corpus_path)
    cache_path = Path(args.cache) if args.cache else None
    ua = args.user_agent or f"socsim/0.12.0 (metadata-only; contact: {args.mailto or 'unspecified'})"
    client = WebClient(user_agent=ua, timeout_s=int(args.timeout), min_interval_s=float(args.min_interval), cache_path=cache_path)
    cfg = HarvestConfig(
        sources=tuple([s.strip() for s in args.sources.split(',') if s.strip()]),
        per_source_rows=int(args.rows),
        mailto=args.mailto,
    )
    total = 0
    for q in args.queries:
        items = harvest_bibliography(client, q, cfg)
        for it in items:
            unit = bibitem_to_atomic_unit(it, query=q, fetched=None)
            corpus.add_dict(unit, schema_path=Path(args.schema))
            total += 1
    corpus.save(corpus_path)
    print(f"Added/updated {total} metadata-only atomic units into {corpus_path}")

def cmd_bench_fetch(args: argparse.Namespace) -> None:
    res = fetch_all(
        registry_path=Path(args.registry),
        out_dir=Path(args.out),
        user_agent=args.user_agent,
        timeout_s=int(args.timeout),
        min_interval_s=float(args.min_interval),
    )
    print(f"Fetched {len(res)} benchmarks into {args.out}")

def cmd_bench_targets(args: argparse.Namespace) -> None:
    paths = build_all_targets(
        registry_path=Path(args.registry),
        bench_root=Path(args.bench_root),
        out_root=Path(args.out),
    )
    print(f"Wrote {len(paths)} targets to {args.out}")

def cmd_bench_run(args: argparse.Namespace) -> None:
    results = run_benchmarks(
        registry_path=Path(args.registry),
        targets_dir=Path(args.targets),
        specs_dir=Path(args.specs),
        out_dir=Path(args.out),
        priors_path=Path(args.priors),
        latent_classes_path=Path(args.latent_classes),
        evidence_store_path=Path(args.evidence_store),
        n=int(args.n),
        seed=int(args.seed),
        corpus_path=Path(args.corpus) if args.corpus else None,
    )
    write_benchmark_report(Path(args.out), Path(args.out) / "BENCHMARK_REPORT.md")
    print(f"Ran {len(results)} benchmarks and wrote report to {Path(args.out)/'BENCHMARK_REPORT.md'}")

def cmd_bench_calibrate(args: argparse.Namespace) -> None:
    calibrate_global_means_via_ridge(
        registry_path=Path(args.registry),
        targets_dir=Path(args.targets),
        specs_dir=Path(args.specs),
        out_dir=Path(args.out),
        priors_path=Path(args.priors),
        latent_classes_path=Path(args.latent_classes),
        evidence_store_path=Path(args.evidence_store),
        n=int(args.n),
        seed=int(args.seed),
        alpha=float(args.alpha),
    )
    print(f"Wrote calibrated priors to {Path(args.out)/'priors_calibrated.json'}")

def cmd_bench_scout(args: argparse.Namespace) -> None:
    hits = scout_openalex(args.query, per_page=int(args.per_page), mailto=args.mailto, user_agent=args.user_agent)
    cands = hits_to_registry_candidates(hits, default_game=args.game)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({"query":args.query,"candidates":cands}, indent=2), encoding="utf-8")
    print(f"Wrote {len(cands)} candidate benchmark entries to {outp}")

def cmd_bench_bootstrap(args: argparse.Namespace) -> None:
    payload = bootstrap_candidates(Path(args.out), per_query=int(args.per_query), mailto=args.mailto, user_agent=args.user_agent)
    print(f"Wrote {len(payload.get('candidates', []))} candidates to {args.out}")

def cmd_bench_scoreboard(args: argparse.Namespace) -> None:
    from pathlib import Path
    from .bench.scoreboard import build_scoreboard
    build_scoreboard(Path(args.run_dir), Path(args.out_json), Path(args.out_md))
    print(f"Wrote {args.out_json} and {args.out_md}")

def cmd_bench_cv(args: argparse.Namespace) -> None:
    from pathlib import Path
    from .bench.cv import leave_one_out_cv
    leave_one_out_cv(
        registry_path=Path(args.registry),
        targets_dir=Path(args.targets),
        specs_dir=Path(args.specs),
        out_dir=Path(args.out),
        priors_path=Path(args.priors),
        latent_classes_path=Path(args.latent_classes),
        evidence_store_path=Path(args.evidence_store),
        n=int(args.n),
        seed=int(args.seed),
        corpus_path=Path(args.corpus) if args.corpus else None,
    )
    print(f"Wrote {Path(args.out) / 'cv_summary.json'}")

def cmd_spec_lint(args: argparse.Namespace) -> None:
    import json as _json
    from pathlib import Path
    from .spec_lint import lint_spec
    spec = _json.loads(Path(args.spec).read_text(encoding="utf-8"))
    res = lint_spec(spec)
    for w in res.warnings:
        print("WARN", w)
    if not res.ok:
        for e in res.errors:
            print("ERR", e)
        raise SystemExit(2)
    print("ok")

def cmd_spec_template(args: argparse.Namespace) -> None:
    import json as _json
    from .spec_template import default_spec
    print(_json.dumps(default_spec(game_name=args.game), indent=2))

def cmd_insight_add(args: argparse.Namespace) -> None:
    import json as _json
    from pathlib import Path
    from .insights.store import InsightStore
    store = InsightStore.load(Path(args.store))
    item = _json.loads(Path(args.item).read_text(encoding="utf-8"))
    store.upsert(item, schema_path=Path(args.schema) if args.schema else None)
    store.save(Path(args.store))
    print("ok")

def cmd_insight_search(args: argparse.Namespace) -> None:
    from pathlib import Path
    import json as _json
    from .insights.store import InsightStore
    store = InsightStore.load(Path(args.store))
    hits = store.search(game=args.game, topic=args.topic)
    print(_json.dumps({"hits": hits}, indent=2))

def cmd_bench_make_targets(args: argparse.Namespace) -> None:
    from pathlib import Path
    import json as _json
    from .bench.pack_runner import run_benchmark_pack

    benches_root = Path(args.bench_root)
    out_root = Path(args.out_targets)
    cache_dir = Path(args.cache)
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for bench_dir in sorted([p for p in benches_root.iterdir() if p.is_dir()]):
        out_dir = out_root / bench_dir.name
        res = run_benchmark_pack(bench_dir, out_dir, cache_dir=cache_dir, timeout_s=int(args.timeout_s))
        results.append({"benchmark_id": res.benchmark_id, "targets_path": str(res.targets_path), "provenance_path": str(res.provenance_path)})

    print(_json.dumps({"results": results}, indent=2))

def cmd_bench_pack(args: argparse.Namespace) -> None:
    from pathlib import Path
    from .bench.pack import create_bench_pack
    create_bench_pack(Path(args.bench_dir), Path(args.out))
    print(f"Wrote {args.out}")

def cmd_list_games(args: argparse.Namespace) -> None:
    from .games.registry import GAME_REGISTRY
    for k in sorted(GAME_REGISTRY.keys()):
        print(k)

def cmd_version(args: argparse.Namespace) -> None:
    import socsim
    print(socsim.__version__)

def cmd_doctor(args: argparse.Namespace) -> None:
    import importlib
    mods = ["numpy", "pandas", "requests", "jsonschema"]
    ok = True
    for m in mods:
        try:
            importlib.import_module(m)
            print("OK", m)
        except Exception as e:
            ok = False
            print("MISSING", m, str(e))
    if not ok:
        raise SystemExit(2)

def cmd_bench_run_pack(args: argparse.Namespace) -> None:
    from pathlib import Path
    from .bench.pack_runner import run_benchmark_pack
    res = run_benchmark_pack(
        bench_dir=Path(args.bench_dir),
        out_dir=Path(args.out_dir),
        cache_dir=Path(args.cache_dir),
        timeout_s=int(args.timeout),
    )
    print(f"Wrote {res.targets_path} and {res.provenance_path}")

def main() -> None:
    p = argparse.ArgumentParser(prog="socsim")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("simulate", help="Run a simulation and write outputs")
    ps.add_argument("--spec", required=True)
    ps.add_argument("--n", type=int, default=200)
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--out", required=True)
    ps.add_argument("--priors", default="socsim/data/priors.json")
    ps.add_argument("--latent-classes", dest="latent_classes", default="socsim/data/latent_classes.json")
    ps.add_argument("--evidence-store", dest="evidence_store", default="socsim/data/evidence_store.json")
    ps.add_argument("--traces", action="store_true")
    ps.add_argument("--exp-schema", default="socsim/schema/experiment_schema.json")
    ps.add_argument("--ev-schema", default="socsim/schema/evidence_unit_schema.json")
    ps.add_argument("--corpus", default="socsim/data/corpus.json")
    ps.set_defaults(func=cmd_simulate)

    pe = sub.add_parser("eval_moments", help="Compare predicted vs observed CSV moments and write JSON report")
    pe.add_argument("--pred", required=True)
    pe.add_argument("--obs", required=True)
    pe.add_argument("--cols", required=True, help="Comma-separated numeric columns to compare")
    pe.add_argument("--out", required=True)
    pe.set_defaults(func=cmd_eval_moments)

    pw = sub.add_parser("corpus_expand_web", help="Web-expand bibliography into metadata-only atomic units")
    pw.add_argument("queries", nargs="+")
    pw.add_argument("--sources", default="crossref,openalex,semanticscholar")
    pw.add_argument("--rows", type=int, default=20)
    pw.add_argument("--mailto", default=None)
    pw.add_argument("--user-agent", dest="user_agent", default=None)
    pw.add_argument("--timeout", type=int, default=30)
    pw.add_argument("--min-interval", dest="min_interval", type=float, default=1.0)
    pw.add_argument("--cache", default="socsim/data/web_cache.sqlite")
    pw.add_argument("--corpus", default="socsim/data/corpus.json")
    pw.add_argument("--schema", default="socsim/schema/atomic_unit_schema.json")
    pw.set_defaults(func=cmd_corpus_expand_web)

    bf = sub.add_parser("bench_fetch", help="Download benchmark datasets with checksums and manifests")
    bf.add_argument("--registry", default="socsim/data/benchmark_registry.json")
    bf.add_argument("--out", default="benchmarks")
    bf.add_argument("--user-agent", dest="user_agent", default="socsim/0.12.0 (bench fetch)")
    bf.add_argument("--timeout", type=int, default=30)
    bf.add_argument("--min-interval", dest="min_interval", type=float, default=1.0)
    bf.set_defaults(func=cmd_bench_fetch)

    bt = sub.add_parser("bench_targets", help="Extract observed moment targets from downloaded benchmark data")
    bt.add_argument("--registry", default="socsim/data/benchmark_registry.json")
    bt.add_argument("--bench-root", dest="bench_root", default="benchmarks")
    bt.add_argument("--out", default="bench_targets")
    bt.set_defaults(func=cmd_bench_targets)

    br = sub.add_parser("bench_run", help="Run simulations for each benchmark and compare to targets")
    br.add_argument("--registry", default="socsim/data/benchmark_registry.json")
    br.add_argument("--targets", default="bench_targets")
    br.add_argument("--specs", default="socsim/data/bench_specs")
    br.add_argument("--out", default="bench_runs")
    br.add_argument("--priors", default="socsim/data/priors.json")
    br.add_argument("--latent-classes", dest="latent_classes", default="socsim/data/latent_classes.json")
    br.add_argument("--evidence-store", dest="evidence_store", default="socsim/data/evidence_store.json")
    br.add_argument("--n", type=int, default=4000)
    br.add_argument("--seed", type=int, default=0)
    br.add_argument("--corpus", default="socsim/data/corpus.json")
    br.set_defaults(func=cmd_bench_run)

    bc = sub.add_parser("bench_calibrate", help="Calibrate global priors against benchmark moment targets")
    bc.add_argument("--registry", default="socsim/data/benchmark_registry.json")
    bc.add_argument("--targets", default="bench_targets")
    bc.add_argument("--specs", default="socsim/data/bench_specs")
    bc.add_argument("--out", default="bench_calibration")
    bc.add_argument("--priors", default="socsim/data/priors.json")
    bc.add_argument("--latent-classes", dest="latent_classes", default="socsim/data/latent_classes.json")
    bc.add_argument("--evidence-store", dest="evidence_store", default="socsim/data/evidence_store.json")
    bc.add_argument("--n", type=int, default=4000)
    bc.add_argument("--seed", type=int, default=0)
    bc.add_argument("--alpha", type=float, default=1.0)
    bc.set_defaults(func=cmd_bench_calibrate)

    bs = sub.add_parser("bench_scout", help="Find candidate benchmark datasets via OpenAlex search")
    bs.add_argument("--query", required=True)
    bs.add_argument("--game", default="dictator")
    bs.add_argument("--per-page", dest="per_page", type=int, default=25)
    bs.add_argument("--mailto", default=None)
    bs.add_argument("--user-agent", dest="user_agent", default="socsim/0.12.0 (bench scout)")
    bs.add_argument("--out", default="bench_candidates.json")
    bs.set_defaults(func=cmd_bench_scout)

bb = sub.add_parser("bench_bootstrap", help="Generate ~100 candidate benchmark entries via OpenAlex (one command)")
bb.add_argument("--per-page", dest="per_page", type=int, default=10)
bb.add_argument("--mailto", default=None)
bb.add_argument("--user-agent", dest="user_agent", default="socsim/0.12.0 (bench bootstrap)")
bb.add_argument("--out", default="socsim/data/benchmark_registry_candidates_100.json")
bb.set_defaults(func=cmd_bench_bootstrap)

bsb = sub.add_parser("bench_scoreboard", help="Build a scoreboard from bench_run outputs")
bsb.add_argument("--run-dir", default="bench_runs")
bsb.add_argument("--out-json", default="bench_runs/scoreboard.json")
bsb.add_argument("--out-md", default="bench_runs/SCOREBOARD.md")
bsb.set_defaults(func=cmd_bench_scoreboard)

bcv = sub.add_parser("bench_cv", help="Leave-one-out CV over benchmarks")
bcv.add_argument("--registry", default="socsim/data/benchmark_registry.json")
bcv.add_argument("--targets", default="bench_targets")
bcv.add_argument("--specs", default="socsim/data/bench_specs")
bcv.add_argument("--out", default="bench_cv")
bcv.add_argument("--priors", default="socsim/data/priors.json")
bcv.add_argument("--latent-classes", dest="latent_classes", default="socsim/data/latent_classes.json")
bcv.add_argument("--evidence-store", dest="evidence_store", default="socsim/data/evidence_store.json")
bcv.add_argument("--n", type=int, default=2000)
bcv.add_argument("--seed", type=int, default=0)
bcv.add_argument("--corpus", default="socsim/data/corpus.json")
bcv.set_defaults(func=cmd_bench_cv)

sl = sub.add_parser("spec_lint", help="Lint an experiment spec for required fields and common issues")
sl.add_argument("--spec", required=True)
sl.set_defaults(func=cmd_spec_lint)

st = sub.add_parser("spec_template", help="Print a skeleton ExperimentSpec+ JSON")
st.add_argument("--game", default="ultimatum")
st.set_defaults(func=cmd_spec_template)

ia = sub.add_parser("insight_add", help="Upsert an insight unit into an insights store")
ia.add_argument("--store", default="socsim/data/insights.json")
ia.add_argument("--item", required=True)
ia.add_argument("--schema", default="socsim/schema/insight_unit_schema.json")
ia.set_defaults(func=cmd_insight_add)

isr = sub.add_parser("insight_search", help="Search insight units by scope")
isr.add_argument("--store", default="socsim/data/insights.json")
isr.add_argument("--game", default=None)
isr.add_argument("--topic", default=None)
isr.set_defaults(func=cmd_insight_search)

bp = sub.add_parser("bench_pack", help="Create a metadata pack for a benchmark directory")
bp.add_argument("--bench-dir", required=True)
bp.add_argument("--out", required=True)
bp.set_defaults(func=cmd_bench_pack)

lg = sub.add_parser("list_games", help="List available games")
lg.set_defaults(func=cmd_list_games)

v = sub.add_parser("version", help="Print socsim version")
v.set_defaults(func=cmd_version)

d = sub.add_parser("doctor", help="Check runtime dependencies")
d.set_defaults(func=cmd_doctor)

brp = sub.add_parser("bench_run_pack", help="Download a benchmark pack dataset, verify sha, and compute targets")
brp.add_argument("--bench-dir", required=True)
brp.add_argument("--out-dir", required=True)
brp.add_argument("--cache-dir", default=".cache/bench")
brp.add_argument("--timeout", type=int, default=60)
brp.set_defaults(func=cmd_bench_run_pack)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
