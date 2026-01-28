# Behavioral Experiment Simulation Tool — Methods Summary

Created by Dr. [Eugen Dimant](https://github.com/edimant).

## What this tool does
- **Standardized simulation**: Generates consistent, comparable pilot datasets from Qualtrics QSF exports by
  inferring conditions, factors, and scales automatically.
- **Behavioral realism**: Simulates response styles (e.g., engaged, satisficing, extreme responding), attention
  checks, and realistic open-ended responses that align with numeric response patterns.
- **Reproducible outputs**: Produces CSV data, metadata, schema validation, and an instructor report for
  transparent review.

## How persona modeling works (automated)
The simulator assigns response-style personas internally based on established survey response behavior and
uses those styles to:
- Modulate scale usage (e.g., endpoint use, midpoint use).
- Inject attention and careless-response patterns.
- Generate open-text responses that reflect numeric response sentiment.

**No persona input is required from students**; the model handles this automatically to keep the workflow
simple and consistent across teams.

## Research foundations & reading list
This tool builds on methodological guidance for:
- survey response styles and attention checks,
- standardized experimental simulation,
- and AI-assisted qualitative response generation.

### Research highlights (direct links)
- **LLM-based simulation personas** informed the persona library that powers heterogeneous response styles in
  the simulator and anchors open-ended response variation.
  - Manning & Horton (2025): https://arxiv.org/abs/2301.07543
  - Horton (2023): https://psyarxiv.com/9kqcg
- **Optimization and calibration of simulated responses** motivated the use of standardized defaults and
  consistent schema validation.
  - Santurkar et al. (2023): https://arxiv.org/abs/2303.17548
  - Method notes: https://benjaminmanning.io/les/optimize.pdf

## Recommended uploads
- **Qualtrics QSF** (required): primary source for detecting conditions, factors, and scales.
- **Exported Qualtrics survey PDF** (optional): improves detection of relevant sections and domain inference.

## Output package
Each run produces a ZIP containing:
- `Simulated.csv`
- `Metadata.json`
- `Schema_Validation.json`
- `R_Prepare_Data.R`
- `Column_Explainer.txt`
- `Instructor_Report.md`
