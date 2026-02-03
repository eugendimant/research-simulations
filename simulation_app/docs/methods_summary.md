# Behavioral Experiment Simulation Tool

**Version 1.0.0** | Dr. Eugen Dimant | [eugendimant.github.io](https://eugendimant.github.io/)

---

## What This Actually Is

This tool creates fake survey data that looks real. Not random numbers—actual data that behaves the way human responses behave, with all the messiness and individual variation you'd see in a real sample.

Why would you want fake data? Because analyzing real data is high-stakes. You collect once, you analyze once, and if you screwed something up—wrong coding, bad exclusion logic, broken analysis script—you might not find out until a reviewer points it out. That's a disaster.

With simulated data, you can run your entire analysis pipeline a hundred times before you touch real data. Find the bugs when they're cheap to fix.

---

## Who Uses This

**Researchers** upload their Qualtrics survey, configure their experimental conditions, and get realistic pilot data. They test their analysis scripts, check their variable coding, verify their pre-registration actually matches their measures. All before spending a dime on data collection.

**Students** get practice datasets for coursework. The data has known properties—I know exactly what effects are in there—so I can check whether students' analyses found them correctly. They learn on data that teaches something, not data that just frustrates them.

**Methods instructors** can generate datasets at different difficulty levels. Easy mode: clean data, obvious effects. Hard mode: realistic attention check failures, careless responders, noisy effects. Students learn to handle real-world data quality problems.

---

## How the Simulation Works

### Personas, Not Random Numbers

Real survey respondents are different from each other. Some read carefully and think hard about every question. Some rush through and satisfice. Some always agree with things. Some always use extreme responses.

The simulation models this. Each simulated participant gets assigned a persona—engaged responder, satisficer, extreme responder, acquiescent responder, careless responder. Their responses then follow that persona's patterns across the whole survey.

The persona weights come from published research:
- About 30% of respondents are genuinely engaged (Krosnick's "optimizers")
- About 20-25% are satisficers (good enough responding)
- About 8-12% show extreme response style (Greenleaf's research)
- About 5-8% are careless responders (Meade & Craig's estimates)

This means your simulated data has the same heterogeneity as real data. The standard deviations are realistic. The attention check fail rates are realistic. The straight-lining patterns are realistic.

### Effect Sizes That Make Sense

When you configure an experimental effect, the simulation uses effect size benchmarks from meta-analyses. A "medium" effect is Cohen's d = 0.5, which matches what most experimental studies actually find (Richard et al., 2003 meta-analysis: average social psychology effect is d = 0.43).

The simulation also understands your conditions semantically. If your conditions are "AI Recommendation" vs "Human Recommendation," it knows AI conditions typically show algorithm aversion effects. If your conditions are "High Trust" vs "Low Trust," it knows the direction. You don't have to manually specify which condition should score higher—it figures it out from the names.

### Open-Ended Responses

The simulator generates text for open-ended questions. Not random words—actual sentences that reflect the question being asked, the participant's overall sentiment (based on their scale responses), and their persona's verbosity.

A satisficer writes "It was fine." An engaged responder writes three sentences explaining their reasoning. A careless responder writes "idk" or something off-topic.

The text varies by question content. Different questions get different responses, even for the same participant. This matters when you're testing how your text coding scheme handles variation.

---

## What You Actually Get

Upload your QSF file, configure your study parameters, run the simulation. You get:

| File | What's In It |
|------|--------------|
| CSV data | Complete simulated dataset, properly formatted |
| Metadata | JSON with all simulation parameters |
| R script | Analysis-ready code for your specific variables |
| Python script | Same, but Python |
| Codebook | Variable descriptions for documentation |

The data looks like Qualtrics export format because it's meant to drop into your existing workflow.

---

## Survey Logic Awareness

The tool parses your survey's display logic and skip logic. If certain questions only appear for certain conditions, simulated participants in other conditions get blank responses for those questions—just like real data.

This matters when your survey has branching. You don't want fake data where everyone answered everything regardless of condition assignment. The simulation respects your survey structure.

---

## Difficulty Levels

For teaching purposes, you can set difficulty:

| Level | What It Means |
|-------|---------------|
| Easy | 98% attention pass rate, minimal noise, clear effects |
| Medium | 92% attention pass rate, some careless responders |
| Hard | 85% attention pass rate, realistic MTurk-quality data |
| Expert | 75% attention pass rate, extensive cleaning required |

Students can progress from clean data to messy data as they develop skills.

---

## Pre-Registration Checking

Upload a pre-registration PDF (OSF, AEA, AsPredicted formats). The tool compares your pre-registered IVs and DVs against what's actually in your survey. Catches discrepancies before they become problems.

---

## Technical Details

The simulation engine uses:
- MD5-based stable hashing for reproducibility
- Condition-semantic parsing for effect direction
- Multi-factor persona trait sampling
- Domain detection (175+ research domains) for appropriate response calibration
- Template-based text generation with persona modulation

Full technical documentation available in `technical_methods.md`.

---

## Limitations

This is simulated data. It's useful for testing, training, and verification. It's not a substitute for actual human subjects research, and it shouldn't be represented as real data.

The personas are based on published research but are still approximations. Real human behavior is more complex. The simulation captures major patterns, not every nuance.

---

## Contact

Dr. Eugen Dimant
University of Pennsylvania
edimant@sas.upenn.edu
[eugendimant.github.io](https://eugendimant.github.io/)

---

*© 2026 Dr. Eugen Dimant. Proprietary software.*
