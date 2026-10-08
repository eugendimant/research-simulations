# Behavioral Experiment Simulation Tool

Turn a Qualtrics survey into a synthetic dataset. Use it to test survey logic and analysis code before you collect data, and to teach data cleaning and analysis on a dataset whose design you know.

You upload a `.qsf` file (or describe your study). The tool detects the conditions, scales and open-ended questions, simulates participants with realistic response styles, and returns a ZIP file with the dataset, a codebook, run metadata and data-preparation scripts for R, Python, Julia, SPSS and Stata.

> **The data are synthetic.** They are not evidence about real people, and reporting them as collected data is research misconduct. Read [what the tool does and does not do](docs/guide/limitations.md) before relying on a simulated dataset for anything beyond testing.

## What you get

| In the ZIP | Purpose |
|---|---|
| `Simulated_Data.csv` | One row per simulated participant, laid out like a Qualtrics export: survey-metadata columns, condition, demographics, scale items, open-ended text |
| `Simulation_Diagnostics.csv` | The simulator's own columns, keyed by `ResponseId`: scale means, quality flags and the exclusion recommendation, response-time summaries, the run seed, the text source |
| `Simulated_Data_Qualtrics_Raw.csv` | The same responses with Qualtrics' three header rows (column names, question text, import IDs) |
| `Data_Codebook_Handbook.txt` | Every column, scale and coding rule in plain language |
| `Metadata.json` | Seed, version, design, the effects built in and the effects observed, how the open-ended text was produced |
| `R_Prepare_Data.R`, `Python_Prepare_Data.py`, `Julia_Prepare_Data.jl`, `SPSS_Prepare_Data.sps`, `Stata_Prepare_Data.do` | Load the CSV, code the conditions, reverse-score items, build scale composites, apply the recommended exclusions |
| `Schema_Validation.json` | Automatic checks (conditions present, values inside each scale's range, missing data) |
| `User_Study_Summary.md` / `.html` | Readable summary of the design and the data, with example t-test/ANOVA code |

## What it simulates

- **Numeric responses** for Likert, matrix, slider, numeric-input, constant-sum, rank-order and similar scales. Participants have response styles drawn from survey-methodology research (engaged, satisficing, extreme, acquiescent and careless responding, reverse-item failure, social desirability, fatigue, straight-lining). Values always stay inside the scale's range.
- **Condition effects.** Specify a Cohen's d for an outcome and a contrast and the data show roughly that d on the scale mean. Where you specify nothing, small differences are inferred from the condition names; you can switch that off for a true null. See [How effects work](docs/guide/how-effects-work.md).
- **Open-ended text** written to fit each participant's ratings and the question topic. In the built-in AI mode, free language-model providers write the first 100 responses and a template engine writes the rest; the template engine alone works offline and goes up to 10,000 participants. Numeric text boxes (age, counts, amounts) get numbers.
- **Quality flags and exclusions** (speed, attention checks, straight-lining) so you can practice or pre-register exclusion rules.
- **Between-subjects designs**, including factorial designs. Within-subject and clustered designs are not modeled yet; the design step says so when you pick one.

## Run it locally

```bash
pip install -r simulation_app/requirements.txt
streamlit run simulation_app/app.py     # http://localhost:8501
```

Python 3.11 or newer (CI runs 3.11 and 3.12). Optional packages are listed in `simulation_app/requirements-optional.txt`.

The free built-in AI mode needs at least one provider key in the environment or in Streamlit secrets (`GOOGLE_API_KEY` or `GEMINI_API_KEY`, `GROQ_API_KEY`, `SAMBANOVA_API_KEY`, `OPENROUTER_API_KEY`; see [docs/DEPLOYMENT_SECRETS.md](docs/DEPLOYMENT_SECRETS.md) and [docs/PROVIDER_SETUP.md](docs/PROVIDER_SETUP.md)); you can also paste your own key into the app. With no key, open-ended text comes from the template engine.

## Documentation

- [Limitations and responsible use](docs/guide/limitations.md)
- [How effects work](docs/guide/how-effects-work.md)
- [Developer and feature reference](simulation_app/README.md)
- [Methods summary](docs/methods_summary.md), [coverage roadmap](docs/COVERAGE_ROADMAP.md) and [changelog](docs/CHANGELOG.md)

## For contributors

```bash
python3 -m pytest tests -m "not slow" -q      # fast suite, no network
python3 -m pytest tests/test_app_import_safety.py -q   # the app loads the way Streamlit loads it
python3 scripts/check_version_sync.py         # every version location agrees
ruff check simulation_app tests scripts --exclude simulation_app/experimental_features
```

Rules the code base relies on (version synchronization, import resilience, page layout, the LLM anti-hang design) are in `CLAUDE.md`.

## Privacy

You upload a survey file, never participant data. Sharing the file with the tool's researchers is optional and off by default; the checkbox is on the upload step and applies to one file. Generated data and generated designs are never shared.

## License and citation

[PolyForm Noncommercial License 1.0.0](LICENSE): noncommercial use is free, including research, teaching and study at educational institutions, public research organizations and charities. For commercial use, contact Eugen Dimant. `CITATION.cff` carries the citation (GitHub's "Cite this repository" button reads it); please name the software version, which the app shows and `Metadata.json` records.
