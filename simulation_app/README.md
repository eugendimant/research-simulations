# Behavioral Experiment Simulation Tool

**Version 1.2.8.7** — a Streamlit app that turns a Qualtrics survey export into a realistic synthetic pilot dataset.

## What it does

Upload a `.qsf` (or describe your study) and get a complete data package: numeric DV responses, open-ended text, demographics, attention/manipulation/comprehension checks, timing paradata, exclusion flags, and ready-to-run analysis scripts in R, Python, Julia, SPSS and Stata.

Conditions, factors and scales are detected from the QSF automatically; you confirm or edit them before generating.

### Why simulated pilot data

- Test an analysis pipeline before collecting real data
- Practice data cleaning against realistic quality problems
- Verify survey logic and variable coding
- Check pre-registration consistency before launch

## Generation methods

All three use the same behavioral engine for numeric data. They differ only in where open-ended text comes from.

| Method | Open-ended text | Sample size |
|---|---|---|
| **Adaptive Behavioral Engine 3.0** (default) | Offline template engine — no API calls | up to 10,000 |
| **Built-in AI** | Free LLM providers, built-in keys, no setup | **LLM text for the first 100 participants; ABE 3.0 for the rest** |
| **Your API Key** | Your own provider key | up to 10,000 |

The 100-participant cap on Built-in AI exists to keep shared free-tier keys from being exhausted. The app warns before generating and tells you the split afterwards. Use your own key for larger runs with AI text throughout.

Built-in provider chain, tried in order until one responds: Google Gemini 3.1 Flash Lite → Gemini 2.5 Flash → Gemini 2.5 Flash Lite → Groq GPT-OSS 120B → Groq Qwen3.6 27B → Cerebras GPT-OSS 120B → SambaNova Llama 3.3 70B → Mistral Small → OpenRouter Mistral Small 3.1.

## The behavioral engine (v1.2.8.7)

**Numeric responses.** Each participant is one person with a persistent identity: census-weighted demographics (age, education, income, party ID, ideology, state), eight response-style traits, and a latent attitude vector. Condition effects are applied as deterministic mean shifts; individual variance is applied separately.

**Condition effects.** 43 effect-detection domains, each with published effect sizes. Relational conditions are parsed before simple valence, so "matched with an outgroup member" produces discrimination rather than generic negativity (Iyengar & Westwood 2015). Economic games use published baselines, not 50% (dictator ≈ 28%, Engel 2011; trust ≈ 50%, Berg et al. 1995). Total semantic effect is capped at ±0.50.

**Strategic games.** For beauty contest, money-request/11-20, stag hunt, minimum-effort coordination and Tullock contest, players reason recursively about other players via Level-k (Stahl & Wilson 1994; Nagel 1995) and Cognitive Hierarchy (Camerer, Ho & Chong 2004). Each persona's `strategic_depth` sets its recursion depth.

**Response styles** (78 personas: 6 response-style, 72 domain-specific across 24 categories). Weights:

| Persona | Weight | Basis |
|---|---|---|
| Engaged Responder | 0.35 | Krosnick (1991) optimizers |
| Satisficer | 0.22 | Krosnick (1991) |
| Socially Desirable Responder | 0.12 | Paulhus (2002) |
| Extreme Responder | 0.10 | Greenleaf (1992) |
| Acquiescent Responder | 0.08 | Billiet & McClendon (2000) |
| Careless Responder | 0.05 | Meade & Craig (2012) |

**Open-ended text.** Compositional assembly (opener + core + elaboration + coda) over 116 domain template sets, 8 structural archetypes, and domain vocabulary banks, with per-participant stylometric fingerprinting (vocabulary richness, sentence length, punctuation rate, filler words, capitalization) held constant across all of a participant's answers. Text is coherent with that participant's numeric responses: straight-liners write short, positive raters don't write negative text.

**Realism layers.** Survey fatigue drift, reverse-item failure that is trait-like within session (Woods 2006), domain-sensitive social desirability (Nederhof 1985), typing-error and reading-speed rates calibrated to education, ex-Gaussian response times, MCAR/MAR missingness, inter-item α targeting, cross-DV correlation, and a post-generation audit that checks completion-time plausibility, open-ended uniqueness, straight-lining prevalence and rating–text coherence, then repairs failures.

**Difficulty levels** (easy / medium / hard / expert) scale noise, straight-lining, careless responding and text effort together, so you can practice cleaning at a chosen difficulty.

## What the QSF parser detects

- Conditions and factors, including embedded-data randomization, with 644 block-name exclusions and 77 regex patterns filtering trash/admin/structural blocks
- DVs by type: matrix, single-item, numbered items, Likert, slider, numeric input, constant sum, rank order, and more. Constant-sum items are renormalized to sum exactly to the total; rank-order DVs are valid 1..k permutations
- Well-known instruments (BFI-10, PANAS, SWLS, PSS, RSE) and reverse-coded items
- DisplayLogic and SkipLogic, as a question dependency graph
- Pre-registration documents in OSF, AEA Registry and AsPredicted formats, checked against the current design (only shown when one is uploaded)

Everything detected is editable before generation, and a 5-row live preview shows the exact column structure first.

## Output package

| File | Contents |
|---|---|
| `Simulated_Data.csv` | The dataset |
| `Data_Codebook_Handbook.txt` | Variable and coding descriptions |
| `R_Prepare_Data.R` | R loading/prep script |
| `Python_Prepare_Data.py` | pandas |
| `Julia_Prepare_Data.jl` | DataFrames.jl |
| `SPSS_Prepare_Data.sps` | SPSS syntax |
| `Stata_Prepare_Data.do` | Stata do-file |
| `Metadata.json` | Every simulation parameter |
| `Schema_Validation.json` | Data quality checks |
| `User_Study_Summary.md` / `.html` | Summary report |
| `Source_Files/` | Your uploaded QSF and PDF |

## Quick start

```bash
pip install -r simulation_app/requirements.txt
streamlit run simulation_app/app.py     # http://localhost:8501
```

Optional dependencies (plotly, scipy, matplotlib, pdfplumber, openpyxl, jsonschema) are in `requirements-optional.txt`. All are lazy imports with fallbacks — the core app runs without them.

**Streamlit Community Cloud:** fork, then point a new app at `simulation_app/app.py`.

**Headless, no Streamlit UI:** see `REPLICATION_README.md`.

## Usage

1. **Setup** — team, study title and description, target N
2. **Upload** — the QSF (required); the survey PDF (optional) improves domain detection by supplying question wording
3. **Design** — confirm detected conditions, factors and DVs; build a factorial crossing if needed; set condition allocation
4. **Generate** — pick a method, generate, download the ZIP

Export both files from Qualtrics under Survey → Tools → Import/Export (Export Survey for the QSF, Print Survey for the PDF).

### Factorial designs

Pick row and column factors and the app crosses them. A 2×3 — {Dictator game, PGG} × {Matched with Hater, Matched with Lover, Matched with Unknown} — produces 6 properly crossed conditions. 2- and 3-factor designs are supported.

## Research domains

189 domains across 23 categories, detected from 273 keyword patterns. The categories, as named in `DOMAIN_CATEGORIES`: behavioral economics, social psychology, political science, consumer & marketing, organizational behavior, technology & AI, AI alignment & ethics, ethics & moral psychology, clinical psychology, personality psychology, health psychology, health disparities, education, environmental, financial psychology, decision science, trust & credibility, gaming & entertainment, social media research, innovation & creativity, risk & safety, future of work, digital society.

Calibration knowledge base: 187 meta-analytic effect entries, 68 economic-game calibrations, 201 construct norms, 12 cultural adjustments.

## Research foundations

- **Argyle et al. (2023)** "Out of One, Many", *Political Analysis* — [10.1017/pan.2023.2](https://doi.org/10.1017/pan.2023.2)
- **Horton (2023)** "Homo Silicus", *NBER WP* — [10.3386/w31122](https://doi.org/10.3386/w31122)
- **Aher, Arriaga & Kalai (2023)** *ICML* — [paper](https://proceedings.mlr.press/v202/aher23a.html)
- **Binz & Schulz (2023)** *PNAS* — [10.1073/pnas.2218523120](https://doi.org/10.1073/pnas.2218523120)
- **Park et al. (2023)** "Generative Agents", *ACM UIST* — [10.1145/3586183.3606763](https://doi.org/10.1145/3586183.3606763)
- **Dillion et al. (2023)** *Trends in Cognitive Sciences* — [10.1016/j.tics.2023.04.008](https://doi.org/10.1016/j.tics.2023.04.008)
- **Westwood (2025)** "Existential threat of LLMs to survey research", *PNAS* — [10.1073/pnas.2518075122](https://doi.org/10.1073/pnas.2518075122)

Full methodology and citations: `docs/methods_summary.md` and `docs/papers/methods_summary.pdf`. Known limitations and the remaining roadmap: `docs/COVERAGE_ROADMAP.md`.

## Layout

```
research-simulations/
├── simulation_app/
│   ├── app.py                      # Streamlit entry point + QSF→engine bridge
│   ├── requirements.txt
│   ├── utils/                      # 31 modules, including:
│   │   ├── enhanced_simulation_engine.py   # the simulation pipeline
│   │   ├── adaptive_behavioral_engine_v2.py
│   │   ├── qsf_preview.py                  # QSF parsing, DV/condition detection
│   │   ├── scientific_knowledge_base.py    # meta-analytic effects, game calibrations
│   │   ├── persona_library.py              # 78 personas
│   │   ├── response_library.py             # offline open-ended generation
│   │   ├── llm_response_generator.py       # LLM open-ended generation
│   │   ├── hbs_*.py                        # participant state, stylometry, validation
│   │   ├── socsim_adapter.py               # bridge to the ABE 3.0 / socsim engine
│   │   └── schema_validator.py, instructor_report.py, group_management.py, …
│   ├── experimental_features/
│   │   └── self-learning simulator/        # socsim: ABE 3.0 engine, Level-k / CH strategies
│   ├── example_files/              # real Qualtrics QSFs used for end-to-end testing
│   └── skills/                     # the development protocol this project follows
├── tests/                          # pytest suites + standalone validation harnesses
├── docs/
│   ├── methods_summary.md          # methodology
│   ├── COVERAGE_ROADMAP.md         # coverage audit + remaining gaps
│   ├── CHANGELOG.md
│   └── papers/, internal/
├── CLAUDE.md, AGENTS.md            # contributor and agent guidelines
└── REPLICATION_README.md           # how to reproduce the validation
```

## Configuration

Email delivery is optional. Set these Streamlit secrets to enable it: `SMTP_SERVER`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM_EMAIL`, `INSTRUCTOR_NOTIFICATION_EMAIL`.

## Credits and license

Created by Dr. Eugen Dimant. For academic and educational use.

```
Dimant, E. (2026). Behavioral Experiment Simulation Tool (Version 1.2.8.7) [Computer software].
https://github.com/eugendimant/research-simulations
```
