from pathlib import Path
import pandas as pd
from socsim.simulator import load_experiment_spec, simulate

def test_simulate_with_survey(tmp_path):
    spec = load_experiment_spec(Path("examples/spec_trust_identity.json"))
    spec.context["survey"] = {
        "theta_param": "prosociality",
        "items": [{"name":"q1","a":1.0,"thresholds":[-0.5,0.5]}]
    }
    res = simulate(
        spec=spec,
        n=30,
        seed=1,
        priors_path=Path("socsim/data/priors.json"),
        latent_classes_path=Path("socsim/data/latent_classes.json"),
        evidence_store_path=Path("socsim/data/evidence_store.json"),
        schema_path=Path("socsim/schema/evidence_unit_schema.json"),
        return_traces=False,
        corpus_path=Path("socsim/data/corpus.json"),
    )
    df = pd.DataFrame(res.rows)
    assert "sv::q1" in df.columns

def test_benchmark_suite_exists():
    assert Path("socsim/data/benchmark_suite.json").exists()
