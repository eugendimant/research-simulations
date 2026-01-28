# Behavioral Experiment Simulation Tool

A Streamlit application for generating realistic synthetic behavioral experiment data using theory-grounded persona-driven simulation.

## What This Tool Does

**Generate realistic pilot datasets from your Qualtrics survey** — Upload your QSF file and receive a complete data package with:
- Simulated participant responses that reflect actual human survey behavior
- Automatic detection of conditions, factors, and scales
- Open-ended text responses that align with numeric ratings
- Attention check failures and exclusion flags
- R script ready for analysis

### Why Use Simulated Pilot Data?

- **Test your analysis pipeline** before collecting real data
- **Practice data cleaning** with realistic quality issues
- **Verify survey logic** and variable coding
- **Develop R/Python scripts** on properly structured data

## Features

- **Automatic Survey Parsing**: Extracts conditions, factors, and scales from Qualtrics QSF
- **Theory-Grounded Personas**: Response styles based on survey methodology literature (Krosnick 1991, Greenleaf 1992)
- **Behavioral Realism**: Attention check patterns, satisficing, extreme responding
- **Variable Text Responses**: Open-ended answers that match numeric response sentiment
- **Standardized Defaults**: Consistent parameters for comparable datasets across teams
- **Complete Output Package**: CSV, R script, metadata, schema validation, instructor report

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

### Step 3: Review
- Verify auto-detected conditions, factors, and scales
- Edit in Advanced mode if needed

### Step 4: Generate
- Click Generate to create your simulation package
- Download ZIP with all outputs
- Optional: Send via email

## Uploading Qualtrics Files

### QSF File (Required)
**Export from Qualtrics**: Survey → Tools → Import/Export → Export Survey

### Survey PDF (Optional but Recommended)
**Export from Qualtrics**: Survey → Tools → Import/Export → Print Survey → Save as PDF

The PDF improves domain detection and persona selection by providing question wording context.

## Output Package Contents

| File | Description |
|------|-------------|
| `Simulated.csv` | Generated dataset |
| `Metadata.json` | All simulation parameters |
| `Schema_Validation.json` | Data quality checks |
| `R_Prepare_Data.R` | Ready-to-use R script |
| `Column_Explainer.txt` | Variable descriptions |
| `Instructor_Report.md` | Documentation for verification |

## Research Foundations

This tool implements simulation approaches from recent LLM research:

- **Argyle et al. (2023)** - "Out of One, Many" *Political Analysis* — [DOI: 10.1017/pan.2023.2](https://doi.org/10.1017/pan.2023.2)
- **Horton (2023)** - "Homo Silicus" *NBER Working Paper* — [DOI: 10.3386/w31122](https://doi.org/10.3386/w31122)
- **Aher, Arriaga & Kalai (2023)** - *ICML* — [Paper](https://proceedings.mlr.press/v202/aher23a.html)
- **Binz & Schulz (2023)** - *PNAS* — [DOI: 10.1073/pnas.2218523120](https://doi.org/10.1073/pnas.2218523120)
- **Park et al. (2023)** - "Generative Agents" *ACM UIST* — [DOI: 10.1145/3586183.3606763](https://doi.org/10.1145/3586183.3606763)
- **Dillion et al. (2023)** - *Trends in Cognitive Sciences* — [DOI: 10.1016/j.tics.2023.04.008](https://doi.org/10.1016/j.tics.2023.04.008)

See `docs/methods_summary.md` for complete methodology documentation with full citations.

## Directory Structure

```
simulation_app/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── docs/
│   └── methods_summary.md      # Detailed methodology
└── utils/
    ├── __init__.py
    ├── enhanced_simulation_engine.py  # Core simulation logic
    ├── persona_library.py      # Behavioral personas
    ├── qsf_preview.py          # QSF parsing
    ├── qsf_parser.py           # QSF utilities
    ├── schema_validator.py     # Data validation
    ├── pdf_generator.py        # Audit log generation
    ├── group_management.py     # Team/API management
    └── instructor_report.py    # Report generation
```

## Configuration

### Email Delivery (Optional)

Set these Streamlit secrets for email functionality:
- `SENDGRID_API_KEY`
- `SENDGRID_FROM_EMAIL`
- `SENDGRID_FROM_NAME` (optional)
- `INSTRUCTOR_NOTIFICATION_EMAIL` (optional)

## Credits

- **Created by**: Dr. Eugen Dimant
- **Institution**: University of Pennsylvania

## License

For academic and educational use.

## Citation

```
Dimant, E. (2025). Behavioral Experiment Simulation Tool (Version 2.0).
https://github.com/edimant/research-simulations
```
