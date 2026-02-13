import json
from pathlib import Path
from socsim.simulator import load_experiment_spec, simulate

def test_smoke_simulate(tmp_path):
    spec = load_experiment_spec(Path("examples/spec_trust_identity.json"))
    res = simulate(
        spec=spec,
        n=30,
        seed=1,
        priors_path=Path("socsim/data/priors.json"),
        latent_classes_path=Path("socsim/data/latent_classes.json"),
        evidence_store_path=Path("socsim/data/evidence_store.json"),
        schema_path=Path("socsim/schema/evidence_unit_schema.json"),
        return_traces=True,
        corpus_path=Path("socsim/data/corpus.json"),
    )
    assert len(res.rows) == 30
    assert "coverage_report" in res.summary
