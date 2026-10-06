# Behavioral Experiment Simulation Tool

**Version 1.2.9.2** — a Streamlit app that turns a Qualtrics survey export into a realistic synthetic pilot dataset.

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
| **Adaptive Behavioral Engine 3.0** (listed first; no method is pre-selected) | ABE 3.0's own narrative engine, offline — no API calls | up to 10,000 |
| **Built-in AI** | Free LLM providers, keys supplied by the deployment, no setup | **LLM text for the first 100 participants; the compositional template engine for the rest** |
| **Your API Key** | Your own provider key | up to 10,000 |

The 100-participant cap on Built-in AI exists to keep shared free-tier keys from being exhausted. The app warns before generating and tells you the split afterwards. Use your own key for larger runs with AI text throughout.

Note which engine covers the remainder: picking Built-in AI or Your API Key sets `_use_abe_v2 = False` (`app.py:12836`, `:13022`), so participants past the cap get text from the compositional template engine (`ComprehensiveResponseGenerator`), not from ABE 3.0. ABE 3.0 generates the text only when you select its own tile, or if you re-select it in the recovery prompt after a mid-run fallback.

Built-in provider chain, tried in order until one responds: Google Gemini 3.1 Flash Lite → Gemini 2.5 Flash → Gemini 2.5 Flash Lite → Groq GPT-OSS 120B → Groq Qwen3.6 27B → Cerebras GPT-OSS 120B → SambaNova Llama 3.3 70B → Mistral Small → OpenRouter Mistral Small 3.1.

## The behavioral engine (v1.2.9.2)

**Numeric responses.** Each participant is one person with a persistent identity: eight response-style traits and a latent attitude vector, which together drive their answers. Condition effects are applied as deterministic mean shifts; individual variance is applied separately.

Alongside these, census-weighted demographics are written to `Simulation_Diagnostics.csv` as seven descriptive `ABE3_*` columns (education, income, party ID, ideology, state, region, response style). These are drawn from census margins but are **descriptive only as far as the numeric DVs go** — they do not shift any DV. Two of them, education and response style, do feed the open-ended text's stylometric fingerprint. The `Age` column is a separate normal draw from the mean and SD you set, not a census weighting. `docs/COVERAGE_ROADMAP.md` tracks wiring them into generation.

**Condition effects.** Forty-three effect-detection domains, each with keyword-to-effect mappings written into the engine as literals. Where the study text names a paradigm that `META_ANALYTIC_DB` has an estimate for — anchoring, default effects, scarcity and so on — the coarse domain multiplier is replaced by one derived from that published estimate (`enhanced_simulation_engine.py:3127`), so an uncalibrated design of that kind gets a literature magnitude rather than a generic domain guess. Paradigms the table does not cover still fall back to the literals. Relational conditions are parsed before simple valence, so "matched with an outgroup member" produces discrimination rather than generic negativity (Iyengar & Westwood 2015). Economic games start from published baselines rather than a generic 50% (dictator 0.28, Engel 2011; trust 0.50, Berg et al. 1995; Johnson & Mislin 2011). The total automatically-detected effect is capped at ±0.50 before the Cohen's-*d* conversion, which bounds the shift that reaches generation at roughly ±0.12 in normalized units.

**On effect magnitudes.** A configured Cohen's `d` is now recovered, which it was not before v1.2.8.8 — the gap used to run several times the target. Recovery is defined on the scale composite, and two independent measurement runs (N=400, three seeds each) bracketed it at 0.15–0.24 for a configured 0.2, 0.45–0.55 for 0.5 and 0.74–0.86 for 0.8, so expect the composite within roughly a quarter of what you asked for and individual items a little below it. At small `d` the run-to-run spread is wide enough that one dataset can land near zero or above target; `tests/test_effect_size_recovery.py` covers scale widths (5-, 7-, 11-point and 0-100), item counts, and that a null effect stays null. Effects the engine infers from condition wording when you configure no `d` are literature-sized and directional, not fitted to a target you chose, so configure `d` when magnitude matters. Every run writes its achieved effects to `Metadata.json` under `effect_sizes_observed`; check there rather than assuming the configured number. Known deviations are tracked in `docs/COVERAGE_ROADMAP.md`.

**Strategic games.** Where a strategic game is detected, players reason recursively about other players via Level-k (Stahl & Wilson 1994; Nagel 1995) and Cognitive Hierarchy (Camerer, Ho & Chong 2004), with each persona's `strategic_depth` setting its recursion depth. Of the games the engine implements, **beauty contest and stag hunt** are reachable from a QSF today, alongside dictator, trust, ultimatum, public goods, prisoner's dilemma, die-roll, gift exchange, Holt-Laury, bribery and common-pool games. The engine's registry holds 24 games in all; the other twelve — money-request/11-20, minimum-effort coordination, Tullock contest, sender-receiver, public goods with punishment, the three repeated games (PD, trust, public goods), time MPL, BDM, discrete choice and survey-Likert — have no QSF detection path yet.

**Response styles** (78 personas: 6 response-style, 72 domain-specific across 24 categories). Weights:

| Persona | Weight | Basis |
|---|---|---|
| Engaged Responder | 0.35 | Krosnick (1991) optimizers |
| Satisficer | 0.22 | Krosnick (1991) |
| Socially Desirable Responder | 0.12 | Paulhus (2002) |
| Extreme Responder | 0.10 | Greenleaf (1992) |
| Acquiescent Responder | 0.08 | Billiet & McClendon (2000) |
| Careless Responder | 0.05 | Meade & Craig (2012) |

**Open-ended text.** Compositional assembly (opener + core + elaboration + coda) over all 116 domain template sets, 8 structural archetypes, and domain vocabulary banks, with per-participant stylometric fingerprinting (vocabulary richness, filler and hedge words, contractions, capitalization, typos) held constant across all of a participant's answers. Text is coherent with that participant's numeric responses: straight-liners write short, positive raters don't write negative text.

**Realism layers.** Survey fatigue drift, reverse-item failure that is trait-like within session (Woods 2006), domain-sensitive social desirability (Nederhof 1985), typing-error rates calibrated to education, ex-Gaussian response times, inter-item α targeting and cross-DV correlation.

A post-generation audit then checks completion-time plausibility, open-ended uniqueness, straight-lining prevalence, open-ended length distribution and rating–text coherence. The first four are repaired automatically; coherence failures are reported for review rather than corrected, since rewriting text to match a rating risks introducing artifacts.

Item-level missingness and dropout are available under advanced settings and are **off by default** (DVs are forced-response). The advanced panel sets the rates; the mechanism itself is not exposed in the UI and defaults to `realistic` — trait- and position-dependent (MAR-like), so inattentive participants and later items go missing more often. A plain `mcar` mechanism exists in the engine for callers that ask for it.

**Difficulty levels** (easy / medium / hard / expert) scale noise, straight-lining, careless responding and text effort together, so you can practice cleaning at a chosen difficulty.

## What the QSF parser detects

- Conditions and factors, including embedded-data randomization, with 644 block-name exclusions and 77 regex patterns filtering trash/admin/structural blocks
- DVs by type: matrix, single-item, numbered items, slider, numeric input, constant sum, rank order, and more
Generation then respects those types: constant-sum items are renormalized to sum exactly to the total (largest-remainder), and rank-order DVs are valid 1..k permutations rather than independent integers.

**Not detected from the QSF, despite tables existing for it.** Validated instruments (`WELL_KNOWN_SCALES`, 10 entries) and reverse-coded items each have a detector in `qsf_preview.py` with no callers, so no parsed scale comes back carrying an instrument name or a `reverse_items` list — verified across 427 scales in 40 corpus QSFs. Reverse-keyed items still work when you mark them yourself on the Design page or describe them to the builder, whose own `KNOWN_SCALES` table (84 entries, including BFI-10, GAD-7 and PHQ-9) does recognize instruments. Branching logic is the third: a question gets a `has_display_logic` / `has_skip_logic` boolean, but nothing in the repo reads either one, the two logic-parse maps come back empty, and the dependency-graph builder is uncalled — so DisplayLogic and SkipLogic do not reach generation at all. All three are tracked in `docs/COVERAGE_ROADMAP.md`.

Separately from the parser, the app reads pre-registration documents in OSF, AEA Registry and AsPredicted formats and checks them against the current design (shown only when one is uploaded).

Everything detected is editable before generation, and a "Generate Preview (5 rows)" button gives a rough 5-row sample of the DV and open-ended columns beforehand. It is an approximation, not the real layout: it covers the first five scales only, omits the Qualtrics metadata block the delivered CSV carries, and omits the run-metadata, timing, quality-flag, scale-composite and `ABE3_*` columns, which ship separately in `Simulation_Diagnostics.csv`.

## Output package

| File | Contents |
|---|---|
| `Simulated_Data.csv` | The dataset, laid out like a Qualtrics download: 17 survey-metadata columns (`StartDate`, `ResponseId`, `Duration (in seconds)`, …) then condition, demographics, attention checks and the raw scale items |
| `Simulation_Diagnostics.csv` | Everything a real Qualtrics export would not have, keyed by `ResponseId`: participant and run IDs, the seed, scale composites (`<Scale>_mean`), timing, the quality flags and `Exclude_Recommended`, and the seven `ABE3_*` columns |
| `Simulated_Data_Qualtrics_Raw.csv` | The same data with Qualtrics' three-row header (names / question text / ImportId) |
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

Optional dependencies (scipy, matplotlib, pdfplumber, PyMuPDF, requests, openpyxl, jsonschema) are in `requirements-optional.txt`. All are lazy imports with fallbacks — the core app runs without them. `plotly` is listed there too but is also a core requirement, so the analytics dashboard's charts work on a default install.

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

**277 research domains** are keyword-detectable, via 3,473 keyword patterns. 189 of them are grouped into the 23 categories below; the remaining 88 are detectable but ungrouped. All 116 `DOMAIN_TEMPLATES` keys carry a selectable open-ended template set, 68 of them inside the 23 categories — 110 are `StudyDomain` values and the other 6 are reached through `_DOMAIN_TEMPLATE_ALIASES` (`response_library.py:4662`, resolved in the lookup at `:8639`). The categories, as named in `DOMAIN_CATEGORIES`: behavioral economics, social psychology, political science, consumer & marketing, organizational behavior, technology & AI, AI alignment & ethics, ethics & moral psychology, clinical psychology, personality psychology, health psychology, health disparities, education, environmental, financial psychology, decision science, trust & credibility, gaming & entertainment, social media research, innovation & creativity, risk & safety, future of work, digital society.

Calibration knowledge base: 187 meta-analytic effect entries, 68 economic-game calibrations, 201 construct norms, 12 cultural adjustments. Of these the game calibrations, construct norms and response-time norms are queried during generation; the meta-analytic effect entries and the cultural adjustments are tables that nothing calls yet (tracked in `docs/COVERAGE_ROADMAP.md`).

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
│   ├── utils/                      # 27 modules, including:
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
Dimant, E. (2026). Behavioral Experiment Simulation Tool (Version 1.2.9.2) [Computer software].
https://github.com/eugendimant/research-simulations
```
