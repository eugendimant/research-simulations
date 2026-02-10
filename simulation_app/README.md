# Behavioral Experiment Simulation Tool

**Version 1.0.1.7** | A Streamlit application for generating realistic synthetic behavioral experiment data using theory-grounded persona-driven simulation.

## What This Tool Does

**Generate realistic pilot datasets from your Qualtrics survey** — Upload your QSF file and receive a complete data package with:
- Simulated participant responses that reflect actual human survey behavior
- Automatic detection of conditions, factors, and scales
- Open-ended text responses that align with numeric ratings
- Attention check failures and exclusion flags
- Analysis scripts ready for immediate use in your preferred statistical software

### Why Use Simulated Pilot Data?

- **Test your analysis pipeline** before collecting real data
- **Practice data cleaning** with realistic quality issues
- **Verify survey logic** and variable coding
- **Develop analysis scripts** on properly structured data
- **Check pre-registration consistency** before data collection

## Features (v1.0.1.7)

### NEW: AI-Powered Open-Ended Responses with 3-Provider Failover (v1.4.10)
- **Zero-config AI**: open-ended responses are automatically AI-generated using built-in API keys — no setup needed
- **3-provider failover chain**: Groq (Llama 3.3 70B) → Cerebras (Llama 3.3 70B) → OpenRouter (Llama 3.3 70B) — if one provider rate-limits, the next is tried automatically
- **Bring your own key**: optionally enter a personal API key from Groq, Cerebras, or OpenRouter for unlimited capacity (auto-detected from key prefix)
- **Large batch architecture**: 20 persona-guided responses per API call for maximum efficiency
- **Smart pool scaling**: pool size auto-calculated from sample_size (works for 50–5,000+ participants)
- **Draw-with-replacement + 7-layer deep variation**: persona-driven transformations ensure 90%+ uniqueness even at 2,000 participants from a pool of 30 base responses
- **Graceful fallback**: silently falls back to 225-domain template system if all LLM providers are exhausted
- **Generate tab status**: clear indicator shows whether AI responses are active, which provider is in use, and an option to enter your own free key if built-in capacity is reached

### NEW: Tab Navigation & UI Fixes (v1.4.7)
- **Tab jumping fix**: widgets no longer reset view to Setup tab on changes
- **Scroll-to-top**: each tab opens at the top instead of middle
- **Collapsed open-ended section**: open-ended questions in Design tab now use an expander

### NEW: Enhanced Scale/Matrix Detection
- **Semantic scale type detection** (satisfaction, trust, intention, risk, etc.)
- **Well-known scale recognition** (Big Five, PANAS, SWLS, PSS, RSE, etc.)
- **Reverse-coded item detection** with automatic flagging
- **Scale quality scoring** with warnings and recommendations
- **10+ scale types supported**: Matrix, Likert, slider, numeric, constant sum, rank order, best-worst, paired comparison, and more

### NEW: Live Data Preview (5 Rows)
- **Preview before generation**: See 5 rows of sample data before full simulation
- **Format verification**: Preview shows exact column structure and data types
- **Difficulty-aware preview**: Preview reflects selected difficulty level

### NEW: Conditional/Skip Logic Awareness
- **Full DisplayLogic parsing** from QSF structure
- **SkipLogic destination tracking** for conditional questions
- **Question dependency graph** showing which questions depend on others
- **Conditional branching detection** for complex survey flows

### NEW: Difficulty Levels for Data Quality
- **4 difficulty levels**: Easy, Medium, Hard, Expert
- **Impacts numeric data**: Noise levels, straight-lining, careless responding
- **Impacts open-text responses**: Response length, effort, coherence, typos
- **Training-focused**: Practice data cleaning at your skill level

### NEW: Mediation Variable Support
- **Automatic mediator detection** based on position and keywords
- **Mediator hints** suggesting likely mechanism variables
- **Path coefficient simulation** for mediation models
- **Support for moderation** and moderated mediation

### NEW: Pre-registration Consistency Checker
- **OSF format parsing** with section extraction
- **AEA Registry format parsing** for RCT pre-registrations
- **AsPredicted format parsing** with all 7 standard sections
- **Pre-reg number extraction** from uploaded documents
- **Consistency warnings** comparing pre-reg to current design
- **Only appears when pre-registration uploaded** - no clutter otherwise

### Comprehensive DV Detection
- **Automatic DV identification** from QSF survey structure
- **10+ DV types supported**: Matrix scales, Likert scales, sliders, single-item DVs, numeric inputs, constant sum, rank order, best-worst, paired comparison, hot spot
- **Question text display** for easy verification of detected DVs
- **Easy add/remove** with one-click removal buttons
- **Type badges** showing DV category (Matrix, Slider, Single Item, etc.)

### Automatic Survey Parsing
- Extracts conditions, factors, and scales from Qualtrics QSF
- **Enhanced scale detection**: Matrix, numbered items, Likert-type, slider, and numeric input
- **225+ research domains** for context-aware response generation
- **40 question type handlers** for open-ended responses
- **200+ trash/unused block exclusion patterns** for clean condition detection

### Enhanced Condition Detection
- **Comprehensive filtering** of non-condition blocks (trash, admin, structural)
- **Pattern-based exclusion** prevents false positives from unused blocks
- **Embedded data extraction** for randomization-based conditions
- **Smart deduplication** preserves condition order

### State Persistence
- **Cross-step state saving** ensures selections persist when navigating
- **Automatic state restoration** when returning to previous steps
- **Session-based persistence** for complete workflow continuity

### Visual Factorial Design Table
For factorial experiments (2×2, 2×3, 3×3, etc.):
- **Enhanced table interface** with design type selector and examples
- **Visual crossing display** showing all condition combinations
- **Numbered cell display** in design table for easy reference
- **Expandable condition list** showing all crossed combinations
- Supports 2-factor and 3-factor designs with clear visual feedback

### Theory-Grounded Personas
Response styles based on survey methodology literature:
- **Engaged Responder** (30%) - High attention, full scale use
- **Satisficer** (20%) - Midpoint tendency, minimal text
- **Extreme Responder** (8%) - Endpoint use, emphatic responses
- **Acquiescent Responder** (7%) - Agreement bias
- **Careless Responder** (5%) - Low attention, fails checks

### Behavioral Realism
- Attention check patterns with realistic pass/fail rates
- Satisficing and extreme responding patterns
- Individual trait-based variation
- Open-ended responses matching numeric sentiment

### Complete Output Package
| File | Description |
|------|-------------|
| `Simulated.csv` | Generated dataset |
| `Metadata.json` | All simulation parameters |
| `Schema_Validation.json` | Data quality checks |
| `R_Prepare_Data.R` | Ready-to-use R script |
| `Column_Explainer.txt` | Variable descriptions |
| `Instructor_Report.md` | Documentation for verification |

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Open browser to http://localhost:8501
```

### Streamlit Community Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" and select this repository
4. Set main file path to `simulation_app/app.py`
5. Deploy

## Usage

### Step 1: Quick Setup
- Enter team name and members
- Provide study title and description
- Set target sample size (N)

### Step 2: Upload QSF
- Upload your Qualtrics QSF file (required)
- Optionally upload survey PDF for better domain detection
- Fill in pre-registration style checklist

### Step 3: Design Setup
- **Conditions**: Select from auto-detected conditions or add manually
- **Factorial Design**: Use the visual table to create crossed conditions
- **Scales**: Review and verify detected dependent variables
- **Sample Size**: Set target N and condition allocation

### Step 4: Generate
- Click Generate to create your simulation package
- Download ZIP with all outputs
- Optional: Send via email

## Factorial Design Table

For experiments with multiple factors (e.g., 2×3 design):

1. **Factor 1 (Rows)**: Select conditions like "Dictator game", "PGG"
2. **Factor 2 (Columns)**: Select conditions like "Match with Hater", "Match with Lover", "Match with Unknown"
3. **View the table**: See the visual crossing of all combinations
4. **Result**: 6 conditions automatically generated and properly crossed

| Factor 1 | Match with Hater | Match with Lover | Match with Unknown |
|----------|------------------|------------------|-------------------|
| **Dictator game** | ✓ | ✓ | ✓ |
| **PGG** | ✓ | ✓ | ✓ |

## Uploading Qualtrics Files

### QSF File (Required)
**Export from Qualtrics**: Survey → Tools → Import/Export → Export Survey

### Survey PDF (Optional but Recommended)
**Export from Qualtrics**: Survey → Tools → Import/Export → Print Survey → Save as PDF

The PDF improves domain detection and persona selection by providing question wording context.

## Research Domains (175+)

The tool supports 175+ research domains across 24 categories:

- **Behavioral Economics**: Dictator game, ultimatum, trust, public goods, risk, time preferences
- **Social Psychology**: Intergroup relations, identity, norms, conformity, prosocial behavior
- **Political Science**: Polarization, partisanship, voting, media effects, policy attitudes
- **Consumer/Marketing**: Brand attitudes, advertising, pricing, purchase decisions
- **Organizational Behavior**: Leadership, teamwork, motivation, job satisfaction
- **Technology/AI**: AI attitudes, automation, human-robot interaction, privacy
- **Health Psychology**: Risk perception, health behaviors, medical decisions
- **Education**: Learning, motivation, feedback, assessment
- And many more...

## Research Foundations

This tool implements simulation approaches from recent LLM research:

- **Argyle et al. (2023)** - "Out of One, Many" *Political Analysis* — [DOI: 10.1017/pan.2023.2](https://doi.org/10.1017/pan.2023.2)
- **Horton (2023)** - "Homo Silicus" *NBER Working Paper* — [DOI: 10.3386/w31122](https://doi.org/10.3386/w31122)
- **Aher, Arriaga & Kalai (2023)** - *ICML* — [Paper](https://proceedings.mlr.press/v202/aher23a.html)
- **Binz & Schulz (2023)** - *PNAS* — [DOI: 10.1073/pnas.2218523120](https://doi.org/10.1073/pnas.2218523120)
- **Park et al. (2023)** - "Generative Agents" *ACM UIST* — [DOI: 10.1145/3586183.3606763](https://doi.org/10.1145/3586183.3606763)
- **Dillion et al. (2023)** - *Trends in Cognitive Sciences* — [DOI: 10.1016/j.tics.2023.04.008](https://doi.org/10.1016/j.tics.2023.04.008)
- **Westwood (2025)** - "Existential threat of LLMs to survey research" *PNAS* — [DOI: 10.1073/pnas.2518075122](https://doi.org/10.1073/pnas.2518075122)

See `docs/methods_summary.md` and `docs/papers/methods_summary.pdf` for complete methodology documentation with full citations.

## Directory Structure

```
research-simulations/
├── simulation_app/
│   ├── app.py                      # Main Streamlit application
│   ├── requirements.txt            # Python dependencies
│   ├── README.md                   # This file
│   ├── example_files/              # QSF training data & examples
│   └── utils/
│       ├── __init__.py
│       ├── enhanced_simulation_engine.py  # Core simulation logic
│       ├── persona_library.py      # Behavioral personas
│       ├── qsf_preview.py          # QSF parsing & scale detection
│       ├── response_library.py     # 225+ domains, 40 question types
│       ├── survey_builder.py       # Conversational study builder
│       ├── text_generator.py       # Open-ended response generation
│       ├── condition_identifier.py # Condition & factor detection
│       ├── schema_validator.py     # Data validation
│       ├── instructor_report.py    # Report generation
│       └── group_management.py     # Team/API management
├── tests/                          # All test files
│   ├── conftest.py                 # Shared fixtures & path setup
│   └── test_e2e.py                 # Main E2E test suite
├── docs/                           # Documentation
│   ├── papers/                     # Research papers & methods PDF
│   ├── methods_summary.md          # Detailed methodology
│   └── technical_methods.md        # Technical documentation
└── CLAUDE.md                       # Development guidelines
```

## Configuration

### Email Delivery (Optional)

Set these Streamlit secrets for email functionality:
- `SMTP_SERVER` (e.g., "smtp.gmail.com")
- `SMTP_PORT` (e.g., 587)
- `SMTP_USERNAME` (your email address)
- `SMTP_PASSWORD` (app password)
- `SMTP_FROM_EMAIL` (sender email)
- `INSTRUCTOR_NOTIFICATION_EMAIL` (where to send reports)

## Credits

- **Created by**: Dr. Eugen Dimant

## License

For academic and educational use.

## Citation

```
Dimant, E. (2025). Behavioral Experiment Simulation Tool (Version 2.2).
https://github.com/edimant/research-simulations
```
