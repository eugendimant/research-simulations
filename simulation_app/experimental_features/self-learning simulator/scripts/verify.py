from __future__ import annotations
import sys, json
from pathlib import Path

def main() -> int:
    sys.path.insert(0, ".")
    import socsim
    from socsim.simulator import load_experiment_spec, simulate
    from socsim.games.registry import GAME_REGISTRY

    for k in ["dictator", "ultimatum", "trust"]:
        assert k in GAME_REGISTRY, f"missing game {k}"

    spec = {
        "game": {"name": "ultimatum", "params": {"endowment": 10.0}},
        "topic": {"name": "baseline", "tags": [], "notes": ""},
        "population": {"label": "adult_online"},
        "context": {"stakes_level": "low"}
    }
    p = Path("scripts/_verify_spec.json")
    p.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    s = load_experiment_spec(p)
    res = simulate(
        spec=s, n=10, seed=0,
        priors_path=Path("socsim/data/priors.json"),
        latent_classes_path=Path("socsim/data/latent_classes.json"),
        evidence_store_path=Path("socsim/data/evidence_store.json"),
        schema_path=Path("socsim/schema/evidence_unit_schema.json"),
        return_traces=False,
        corpus_path=Path("socsim/data/corpus.json"),
    )
    assert len(res.rows) == 10
    assert isinstance(res.summary, dict)
    print("verify_ok", socsim.__version__)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
