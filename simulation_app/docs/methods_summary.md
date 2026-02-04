# Behavioral Experiment Simulation Tool

**Version 1.0.0** | Proprietary Software

**Developer**: Dr. Eugen Dimant, University of Pennsylvania

---

## Overview

The Behavioral Experiment Simulation Tool is a proprietary software system that generates high-fidelity synthetic behavioral data for experimental research. The tool produces datasets that accurately replicate the statistical properties, individual differences, and response patterns observed in human subjects research.

This software addresses a critical gap in research methodology: the need to validate analysis pipelines, test pre-registrations, and train students on realistic data before committing resources to actual data collection.

---

## Core Capabilities

### Synthetic Data Generation

The tool generates synthetic survey responses that exhibit the same distributional properties as real behavioral data:

- Realistic means and standard deviations for Likert-type scales
- Appropriate inter-item correlations within multi-item measures
- Naturalistic variation in attention and engagement across participants
- Authentic patterns of careless responding and satisficing behavior

### Survey Structure Fidelity

When a Qualtrics survey file (QSF) is uploaded, the simulation engine:

- Parses the complete survey structure including all question types
- Identifies experimental conditions from randomization elements
- Detects dependent variables, manipulation checks, and attention checks
- Respects display logic and skip patterns in the original survey

The output dataset maps precisely onto the programmed experiment—every variable, every condition, every branching path is reflected accurately in the synthetic data.

### Persona-Based Response Modeling

Rather than generating responses from a single distribution, the tool models participant heterogeneity through empirically-grounded response personas:

| Persona Type | Prevalence | Theoretical Basis |
|--------------|------------|-------------------|
| Engaged Responder | 30-35% | Krosnick's (1991) "optimizing" response mode |
| Satisficer | 20-25% | Krosnick's (1991) satisficing theory |
| Extreme Responder | 8-12% | Greenleaf's (1992) extreme response style |
| Acquiescent Responder | 6-8% | Billiet & McClendon (2000) |
| Careless Responder | 3-8% | Meade & Craig (2012) |

Each simulated participant is assigned a persona, and their responses across all measures reflect that persona's characteristic patterns.

---

## Applications

### Pre-Data Collection Validation

Researchers can test their complete analysis pipeline on synthetic data structured identically to their planned study. This identifies coding errors, variable miscalculations, and logical flaws before they compromise actual research.

### Pre-Registration Verification

The tool compares uploaded pre-registration documents against the survey structure, flagging discrepancies between registered measures and implemented variables.

### Methods Training

Instructors can generate synthetic datasets with known properties for coursework. Difficulty levels control data quality characteristics:

| Level | Attention Pass Rate | Careless Responders | Data Quality |
|-------|---------------------|---------------------|--------------|
| Easy | 98% | Minimal | Clean |
| Medium | 92% | Some | Moderate noise |
| Hard | 85% | Realistic | MTurk-typical |
| Expert | 75% | High | Extensive cleaning required |

Students learn to identify and handle data quality issues on datasets where the ground truth is known.

### Power Analysis Validation

By generating synthetic data with specified effect sizes, researchers can verify that their planned sample sizes provide adequate power for the anticipated effects.

---

## Technical Foundation

### Effect Size Calibration

Experimental effects are calibrated to published meta-analytic benchmarks:

- Small effects: Cohen's d = 0.20
- Medium effects: Cohen's d = 0.50 (consistent with Richard et al., 2003 meta-analytic mean)
- Large effects: Cohen's d = 0.80

Effect direction is determined by semantic analysis of condition names, ensuring that effects follow the logical structure of the experimental design rather than arbitrary ordering.

### Open-Ended Response Generation

Text responses for open-ended questions are generated using domain-specific templates spanning 175+ research areas. Responses reflect:

- Question content and type
- Participant's overall response sentiment
- Persona-appropriate verbosity and formality
- Condition-specific contextual elements

### Reproducibility

All stochastic elements use seeded random number generation. Identical parameters produce identical outputs across sessions and platforms.

---

## Output Files

| File | Description |
|------|-------------|
| `Simulated_Data.csv` | Complete synthetic dataset |
| `Study_Summary.md` | Documentation of study parameters |
| `Metadata.json` | Full simulation configuration |
| `Analysis_Script.R` | R code for data preparation |
| `Analysis_Script.py` | Python code for data preparation |
| `Codebook.md` | Variable descriptions |

---

## Proprietary Notice

This software is the proprietary intellectual property of Dr. Eugen Dimant. Unauthorized reproduction, distribution, or derivative works are prohibited.

---

## Contact

Dr. Eugen Dimant
University of Pennsylvania
edimant@sas.upenn.edu
[eugendimant.github.io](https://eugendimant.github.io/)

---

*© 2026 Dr. Eugen Dimant. All rights reserved.*
