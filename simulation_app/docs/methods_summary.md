# Behavioral Experiment Simulation Tool — Methods Summary

**Version 2.0** | Created by Dr. [Eugen Dimant](https://github.com/edimant)

---

## Overview

The Behavioral Experiment Simulation Tool generates realistic synthetic pilot data for behavioral science experiments. It automates the creation of datasets that mirror real survey response patterns, enabling researchers and students to:

- Test analysis pipelines before data collection
- Practice data cleaning and exclusion procedures
- Verify survey logic and variable coding
- Develop analysis scripts on realistic data structures

---

## What This Tool Does

### 1. Automatic Survey Parsing
Upload your Qualtrics QSF file and the tool automatically:
- Detects experimental conditions from survey flow
- Identifies Likert scales and their properties
- Extracts open-ended question specifications
- Infers factorial design structure

### 2. Behaviorally Realistic Response Generation
The simulator generates responses that reflect actual human survey behavior:
- **Response styles**: Engaged, satisficing, extreme responding, acquiescence
- **Attention patterns**: Realistic attention check pass/fail rates
- **Individual differences**: Trait-based variation across participants
- **Sentiment alignment**: Open-ended responses consistent with numeric ratings

### 3. Complete Output Package
Each simulation produces:
- `Simulated.csv` — Generated dataset
- `Metadata.json` — Simulation parameters and settings
- `Schema_Validation.json` — Data quality checks
- `R_Prepare_Data.R` — Analysis-ready R script
- `Column_Explainer.txt` — Variable descriptions
- `Instructor_Report.md` — Documentation for verification

---

## Persona-Based Simulation Methodology

### Theory-Grounded Response Styles

The simulator assigns behavioral personas based on established survey methodology literature:

| Persona | Weight | Basis |
|---------|--------|-------|
| Engaged Responder | 30% | Careful, thoughtful responding |
| Satisficer | 20% | Krosnick (1991) satisficing theory |
| Extreme Responder | 8% | Greenleaf (1992) extreme response styles |
| Acquiescent Responder | 7% | Yes-saying bias patterns |
| Careless Responder | 5% | Inattentive response patterns |

### Domain-Specific Personas

The tool also includes specialized personas for research domains:
- **Consumer Behavior**: Brand loyalist, deal seeker, impulse buyer, conscious consumer
- **AI/Technology**: Tech enthusiast, tech skeptic, AI pragmatist, privacy-concerned
- **Behavioral Economics**: Loss-averse, present-biased, rational deliberator
- **Organizational Behavior**: High performer, disengaged employee
- **Social Psychology**: Prosocial, individualist, conformist

Personas are automatically selected based on keywords detected in your study description.

### Trait-Based Individual Variation

Each simulated participant receives unique trait values drawn from persona-specific distributions:
- Attention level
- Response consistency
- Scale use breadth (endpoint vs. midpoint tendency)
- Acquiescence tendency
- Social desirability concern

This creates realistic individual differences without requiring manual configuration.

---

## Open-Ended Response Generation

Open-ended text responses are generated to:
- Match the sentiment of numeric responses (positive, negative, neutral)
- Reflect persona-specific verbosity and engagement
- Include natural language variation (hedging, fillers, varying sentence structure)
- Align with the study topic and experimental conditions

The system uses extensive template libraries with natural language variation to produce responses that resemble authentic participant feedback.

---

## Recommended Uploads

### Required: Qualtrics QSF File
Export from Qualtrics: **Survey → Tools → Import/Export → Export Survey**

The QSF file provides:
- Question structure and types
- Survey flow and randomization
- Response option formats
- Condition/block assignments

### Optional: Survey PDF Export
Export from Qualtrics: **Survey → Tools → Import/Export → Print Survey → Save as PDF**

The PDF provides:
- Improved question wording detection
- Better domain inference for persona selection
- Visual formatting context

---

## Research Foundations

### Core Methodological Papers

**LLM Simulation Methodology:**
- Argyle, L. P., et al. (2023). Out of one, many: Using language models to simulate human samples. *Political Analysis*, 31(3), 337-351. — Demonstrates LLMs can replicate human response distributions across demographic subgroups.

- Horton, J. J. (2023). Large language models as simulated economic agents: What can we learn from Homo Silicus? *NBER Working Paper*. — Shows LLMs exhibit human-like economic behaviors in experimental settings.

**Validation Research:**
- Manning, B. L., & Horton, J. J. (2025). Behavioral experiments with LLM-simulated participants. [arXiv:2301.07543](https://arxiv.org/abs/2301.07543)

- Santurkar, S., et al. (2023). Whose opinions do language models reflect? [arXiv:2303.17548](https://arxiv.org/abs/2303.17548)

### Response Style Literature

- **Krosnick, J. A. (1991)**. Response strategies for coping with the cognitive demands of attitude measures in surveys. *Applied Cognitive Psychology*, 5(3), 213-236.

- **Greenleaf, E. A. (1992)**. Measuring extreme response style. *Public Opinion Quarterly*, 56(3), 328-351.

- **Paulhus, D. L. (1991)**. Measurement and control of response bias. In *Measures of Personality and Social Psychological Attitudes* (pp. 17-59).

---

## Standardization Features

### Why Standardization Matters

When multiple teams simulate pilot data, standardized defaults ensure:
- Comparable datasets across teams
- Consistent exclusion criteria
- Reproducible simulation parameters
- Fair assessment of analysis approaches

### Default Settings (Simple Mode)

| Parameter | Default Value |
|-----------|--------------|
| Gender distribution | 50% male |
| Mean age | 35 years |
| Age SD | 12 years |
| Attention check pass rate | 95% |
| Random responder rate | 5% |
| Min completion time | 60 seconds |
| Max completion time | 1800 seconds |
| Straight-line threshold | 10 consecutive identical responses |

Advanced mode allows customization of all parameters.

---

## Exclusion Criteria Simulation

The tool simulates realistic data quality issues and flags participants for potential exclusion:

| Flag | Description |
|------|-------------|
| `Flag_Speed` | Completion time outside acceptable range |
| `Flag_Attention` | Failed attention check(s) |
| `Flag_StraightLine` | Excessive straight-line responding |
| `Exclude_Recommended` | Recommended for exclusion (any flag) |

These flags allow students to practice applying pre-registered exclusion criteria.

---

## Reproducibility

- Each simulation run receives a unique `RUN_ID` and `SIMULATION_SEED`
- Seeds are derived from study metadata plus timestamp for uniqueness
- Providing the same seed reproduces identical output
- Metadata JSON captures all parameters for replication

---

## Technical Requirements

- **Python 3.9+**
- **Dependencies**: streamlit, pandas, numpy, reportlab, sendgrid, pypdf

---

## Citation

If you use this tool in your research or teaching, please cite:

> Dimant, E. (2025). Behavioral Experiment Simulation Tool (Version 2.0). https://github.com/edimant/research-simulations

---

## Contact

For questions, issues, or feature requests:
- GitHub: [github.com/edimant](https://github.com/edimant)
- Email: edimant@sas.upenn.edu
