# Behavioral Experiment Simulation Tool

**Version 1.3.0.4** — a Streamlit app that turns a Qualtrics survey export into a realistic synthetic pilot dataset.

## What it does

Upload a `.qsf` (or describe your study) and get a complete data package: numeric DV responses, open-ended text, demographics, attention/manipulation/comprehension checks, timing paradata, exclusion flags, and data-preparation scripts in R, Python, Julia, SPSS and Stata (load the CSV, code the conditions, reverse-score items, build scale composites, apply the recommended exclusions; the statistical analysis is yours to run).

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
| **Adaptive Behavioral Engine 3.0** (listed first; no method is pre-selected) | Offline template engine — no API calls | up to 10,000 |
| **Built-in AI** | Free LLM providers, built-in keys, no setup | **LLM text for the first 100 participants; ABE 3.0 for the rest** |
| **Your API Key** | Your own provider key | up to 10,000 |

The 100-participant cap on Built-in AI exists to keep shared free-tier keys from being exhausted. The app warns before generating and tells you the split afterwards. Use your own key for larger runs with AI text throughout.

Built-in provider chain, tried in order until one responds: Google Gemini 3.1 Flash Lite → Gemini 2.5 Flash → Gemini 2.5 Flash Lite → Groq GPT-OSS 120B → Groq Qwen3.6 27B → SambaNova Llama 3.3 70B → OpenRouter Mistral Small 3.1.

## The behavioral engine (v1.3.0.4)

**Numeric responses.** Each participant is one person with a persistent identity: eight response-style traits and a latent attitude vector, which together drive their answers. Condition effects are applied as deterministic mean shifts; individual variance is applied separately.

Alongside these, census-weighted demographics are exported as seven descriptive `ABE3_*` columns (education, income, party ID, ideology, state, region, response style). These are drawn from census margins but are **descriptive only as far as the numeric DVs go** — they do not shift any DV. Two of them, education and response style, do feed the open-ended text's stylometric fingerprint. The `Age` column is a separate normal draw from the mean and SD you set, not a census weighting. `docs/COVERAGE_ROADMAP.md` tracks wiring them into generation.

**Condition effects.** Forty-three effect-detection domains, each with keyword-to-effect mappings grounded in the literature but written into the engine as literals — the meta-analytic effect table is not consulted at runtime. Relational conditions are parsed before simple valence, so "matched with an outgroup member" produces discrimination rather than generic negativity (Iyengar & Westwood 2015). Economic games start from published baselines rather than a generic 50% (dictator 0.28, Engel 2011; trust 0.50, Berg et al. 1995; Johnson & Mislin 2011). The total automatically-detected effect is capped at ±0.50 before the Cohen's-*d* conversion, which bounds the shift that reaches generation at roughly ±0.12 in normalized units.

**On effect magnitudes.** A configured Cohen's `d` is now recovered, which it was not before v1.2.8.8 — the gap used to run several times the target. Recovery is defined on the scale composite. Averaged over 6 to 8 independent runs of 1,200 participants per cell, the composite lands within about 8% of the request across scale widths (2-point, 3-point, 5-point, 7-point, 11-point and 0-100) and item counts from 1 to 20, alone or next to other scales (0.97 of the request with two scales and 1.05 with eight in 8-run checks; before v1.2.9.1 a second scale cut it to 0.2-0.5); v1.2.9.1 corrected a 15-25% overshoot at 8 to 20 items and a 25% shortfall on two-point scales (full table in `docs/guide/how-effects-work.md`). A single run varies by roughly ±0.06 in *d* from sampling alone at that size and by about ±0.1 at N=400, so at small `d` one dataset can land near zero or above target; `tests/test_effect_size_recovery.py` and `tests/test_effect_fidelity_v1291.py` cover scale widths, item counts, and that a null effect stays null. Where you configure no `d`, the engine infers literature-sized, directional effects from the condition wording; they are not fitted to a target, and an Advanced Settings checkbox switches them off for a true null. When you do configure a `d` for a variable, nothing inferred from the condition names is added on top of it, and outcomes that the economic-game model rewrites keep the configured `d` too (restored after the model runs). Every run writes what was built in and what was achieved to `Metadata.json` (`effect_sizes_applied`, `effect_sizes_observed`); check there rather than assuming the configured number. Known deviations are tracked in `docs/COVERAGE_ROADMAP.md`.

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

**Open-ended text.** Compositional assembly (opener + core + elaboration + coda) over all 116 domain template sets, 8 structural archetypes, and domain vocabulary banks, with per-participant stylometric fingerprinting (vocabulary richness, filler and hedge words, contractions, capitalization, typos) held constant across all of a participant's answers. Text is coherent with that participant's numeric responses: straight-liners write short, positive raters don't write negative text. Text boxes that expect a number (Qualtrics numeric validation, or wording such as age, how many, amount) and ID boxes (MTurk, Prolific, participant ID) are answered with numbers or IDs, and a text box that repeats a numeric question already in the data is skipped. All post-processing of generated text (stylometry, validation, variation) edits only at grammatical positions through `utils/text_cleanup.py`.

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

Everything detected is editable before generation, and a "Generate Preview (5 rows)" button gives a rough 5-row sample of the DV and open-ended columns beforehand. It is an approximation, not the real layout: it covers the first five scales only and omits the run metadata, timing, quality-flag and `ABE3_*` columns the full export carries.

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

Optional dependencies (plotly, scipy, matplotlib, pdfplumber, PyMuPDF, requests, openpyxl, jsonschema) are in `requirements-optional.txt`. All are lazy imports with fallbacks — the core app runs without them.

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

**273 research domains** are keyword-detectable, via 3,452 keyword patterns. 189 of them are grouped into the 23 categories below; the remaining 84 are detectable but ungrouped. 104 of them carry a reachable open-ended template set, 68 of which fall inside the 23 categories (`DOMAIN_TEMPLATES` holds 116 keys, but 10 are not `StudyDomain` values and can never be selected — `docs/COVERAGE_ROADMAP.md` item 12d lists them). The categories, as named in `DOMAIN_CATEGORIES`: behavioral economics, social psychology, political science, consumer & marketing, organizational behavior, technology & AI, AI alignment & ethics, ethics & moral psychology, clinical psychology, personality psychology, health psychology, health disparities, education, environmental, financial psychology, decision science, trust & credibility, gaming & entertainment, social media research, innovation & creativity, risk & safety, future of work, digital society.

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
│   ├── utils/                      # 31 modules, including:
│   │   ├── enhanced_simulation_engine.py   # the simulation pipeline
│   │   ├── adaptive_behavioral_engine_v2.py
│   │   ├── qsf_preview.py                  # QSF parsing, DV/condition detection
│   │   ├── scientific_knowledge_base.py    # meta-analytic effects, game calibrations
│   │   ├── persona_library.py              # 78 personas
│   │   ├── response_library.py             # offline open-ended generation
│   │   ├── text_cleanup.py                 # grammar-safe helpers for all generated-text post-processing
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
│   ├── guide/                      # user-facing pages: how effects work, limitations
│   ├── methods_summary.md          # methodology
│   ├── COVERAGE_ROADMAP.md         # coverage audit + remaining gaps
│   ├── CHANGELOG.md
│   └── papers/, internal/
├── README.md, LICENSE, CITATION.cff
├── CLAUDE.md, AGENTS.md            # contributor and agent guidelines
└── REPLICATION_README.md           # how to reproduce the validation
```

## Configuration

Email delivery is optional. Set these Streamlit secrets to enable it: `SMTP_SERVER`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM_EMAIL`, `INSTRUCTOR_NOTIFICATION_EMAIL` (several addresses may be listed, separated by commas, for example a second inbox outside the Outlook filters).

Every run sends the instructor notification (the statistical report, the detailed analysis, the student ZIP) from a background thread, so closing the browser tab no longer cancels it. It is two messages: a summary without attachments (the headline numbers and the full analysis are in its body) and, in the same thread, a second message with the attachments, so a mail filter that holds attachments cannot delay the analysis (`INSTRUCTOR_EMAIL_MODE=single` sends one message instead). Transient SMTP errors are retried (5 attempts over several minutes); when the attachments would exceed `EMAIL_MAX_MESSAGE_MB` (default 15, counted after base64 encoding) the ZIP shrinks to a copy without source uploads, or is left out, with a note in the message. To protect the mailbox's daily quota, instructor mails are limited to 12 per session per hour (`INSTRUCTOR_EMAIL_MAX_PER_SESSION_PER_HOUR`) and 200 runs a day (`INSTRUCTOR_EMAIL_MAX_PER_DAY`, two messages each; raise it for an institutional account, 0 disables a limit); a skipped run is logged and its analyses stay in the admin dashboard. Emails that students trigger from the download page are limited per session (`USER_EMAIL_MAX_PER_SESSION_PER_HOUR`, default 5), per recipient (2 a day) and for the whole app (`USER_EMAIL_MAX_PER_HOUR`, default 60; `USER_EMAIL_MAX_PER_DAY`, default 100). Every attempt (including "SMTP not configured") is written to `data/email_delivery_log.jsonl`, printed as one masked `EMAIL-DELIVERY` line to the app log, and shown in the admin dashboard (`?admin=1`, tab Email Delivery). That tab also warns about settings that make a receiving system hold or drop accepted mail, has a test-email button (choose the content: body only, a `.md`, `.html` or `.zip` file, or a 3 MB / 10 MB attachment, to find out what the mail system holds back), and keeps each run's instructor analyses for download or re-sending. The admin page and the analytics dashboard are protected by `ADMIN_PASSWORD` / `ANALYTICS_DASHBOARD_PASSWORD` (or the `*_SHA256` secrets); use long random values: wrong guesses are counted and shown on the admin page but never lock the owner out.

## Credits and license

Created by Dr. Eugen Dimant. Licensed under the [PolyForm Noncommercial License 1.0.0](../LICENSE): noncommercial use, including research and teaching, is free; commercial use needs the author's permission.

```
Dimant, E. (2026). Behavioral Experiment Simulation Tool (Version 1.3.0.4) [Computer software].
https://github.com/eugendimant/research-simulations
```
