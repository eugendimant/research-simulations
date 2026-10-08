# Replication Package — Behavioral Experiment Simulation Tool (v1.3.0.4)

This archive contains everything needed to run, inspect, and evaluate the
behavioral-experiment simulation tool in its entirety.

## What this tool does
Given a Qualtrics survey export (`.qsf`) or a study description, it generates
realistic synthetic participant data: numeric scale/DV responses, open-ended
text, demographics, attention/manipulation/comprehension checks, timing
paradata, and exclusion flags — grounded in published behavioral-science norms.

## Contents
- `simulation_app/` — the application
  - `app.py` — Streamlit entry point + the QSF→engine bridge (`_preview_to_engine_inputs`)
  - `utils/enhanced_simulation_engine.py` — the ABE 3.0 simulation engine (DV generation, effects, personas)
  - `utils/qsf_preview.py` — QSF parsing & DV/condition detection
  - `utils/response_library.py`, `persona_library.py`, `llm_response_generator.py` — open-ended text generation
  - `utils/scientific_knowledge_base.py` — meta-analytic effects, game calibrations, construct norms
  - `utils/instructor_report.py`, `schema_validator.py`, `svg_charts.py` — analysis/reporting/validation
  - `example_files/` — real Qualtrics `.qsf` files used for end-to-end testing (the corpus grows as surveys are collected; count with `ls simulation_app/example_files/*.qsf | wc -l`)
  - `requirements.txt` — Python dependencies
  - `skills/` — the development protocol the project follows
- `tests/` — pytest suites + standalone validation harnesses
- `docs/` — CHANGELOG, COVERAGE_ROADMAP (audit + remaining roadmap), methods summary
- `CLAUDE.md`, `AGENTS.md` — contributor/agent guidelines

## Setup
```bash
python3 -m venv venv && source venv/bin/activate     # optional
pip install -r simulation_app/requirements.txt        # streamlit, pandas, numpy, reportlab, pypdf, tabulate
pip install pytest                                    # for the test suite
```

## Run the app
```bash
streamlit run simulation_app/app.py
```

## Reproduce the validation
```bash
# Unit / regression suite (offline; the slow tests are deselected, run them with -m slow)
python3 -m pytest tests -m "not slow" -q

# Parse every example QSF — expect 0 crashes
python3 tests/qsf_robustness.py

# End-to-end simulate a diverse QSF sample — expect "ALL CHECKS PASSED"
python3 tests/qsf_e2e_sim.py 25

# Crash/scoping fuzz over 2,592 context×condition×variable combos
python3 tests/effect_fuzz.py

# Deep output-data inspection of the 10 most-recent student QSFs
python3 tests/student_qsf_inspect.py 30
```
(The standalone harnesses are scripts, not pytest files; run them directly.)

## Headless simulation in code (no Streamlit UI, no network)
```python
import sys; sys.path.insert(0, "simulation_app")
from utils.qsf_preview import QSFPreviewParser
import app          # needs streamlit installed; no server is started
from utils.enhanced_simulation_engine import EnhancedSimulationEngine

preview = QSFPreviewParser().parse(open("simulation_app/example_files/Coffee_Shop_Loyalty_Programs.qsf", "rb").read())
inp = app._preview_to_engine_inputs(preview)
eng = EnhancedSimulationEngine(
    study_title="Demo", study_description="demo", sample_size=100,
    conditions=inp["conditions"], factors=inp["factors"], scales=inp["scales"],
    additional_vars=[], demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12},
    open_ended_questions=inp.get("open_ended_questions"), seed=7)
eng.llm_generator.disable_permanently("offline")   # skip free-LLM open-ended text
df, meta = eng.generate()
print(df.shape); print(df.head())
```

## Notes for evaluation
- **Offline by default for tests:** call `eng.llm_generator.disable_permanently(...)`
  to force the offline (non-LLM) open-ended generator; otherwise the tool tries
  free LLM providers for open-ended text and falls back to templates.
- **Version sync:** the app checks `REQUIRED_UTILS_VERSION == utils.__version__`
  at startup; all 11 version locations are kept in sync (currently `1.3.0.4`). Ten
  of them are enforced in CI by `scripts/check_version_sync.py`; `BUILD_ID` is the
  eleventh and is checked by eye. The canonical list is in `CLAUDE.md`.
- **Known remaining roadmap** (not bugs) is documented in
  `docs/COVERAGE_ROADMAP.md` — e.g. within-subjects repeated-measures done as a
  proper long format, dyadic/per-trial output modes, population-source profiles.
