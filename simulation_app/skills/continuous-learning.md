# Continuous Learning and Calibration System

The simulator improves with every run. This file specifies how simulation outputs are archived, compared against benchmarks, and used to calibrate future simulations.

## Core Principle

Every simulation run is both a deliverable AND a training sample. The system accumulates evidence about what works and what fails, then uses this evidence to make future simulations better.

## Architecture Overview

```
SIMULATION RUN
     │
     ├──→ DELIVER output to user
     │
     └──→ AUTO-ARCHIVE to GitHub
          │
          ├── Raw output (CSV, metadata)
          ├── Quality metrics (computed automatically)
          ├── Prompt snapshots (exact prompts used)
          └── Configuration (personas, parameters, condition specs)
                    │
                    ▼
          CALIBRATION DATABASE (grows over time)
                    │
                    ├── Compare against real human data (when available)
                    ├── Track quality metrics across runs
                    ├── Identify prompt patterns that produce best output
                    └── Update template parameters and covariance matrices
                              │
                              ▼
                    IMPROVED DEFAULTS for next simulation
```

## Auto-Archive Protocol

After every simulation run that produces output, automatically archive:

### What to Store

```
simulation_archive/
├── runs/
│   └── {timestamp}_{experiment_name}/
│       ├── config.json              # Full simulation configuration
│       │   ├── n_participants
│       │   ├── persona_library (types, mixture priors, trait distributions)
│       │   ├── condition_specs
│       │   ├── target_population
│       │   └── survey_structure (item list, skip logic, scales)
│       │
│       ├── prompts/                 # Exact prompts used (snapshots)
│       │   ├── subject_profile_template.txt
│       │   ├── item_prompts/        # One file per item type
│       │   └── prompt_version.txt   # Hash or version identifier
│       │
│       ├── output/
│       │   ├── simulated_data.csv   # The actual output
│       │   └── paradata.csv         # Timing, metadata (if generated)
│       │
│       ├── quality/
│       │   ├── metrics.json         # Automated quality metrics
│       │   ├── consistency_audit.json  # Intra-subject audit results
│       │   ├── distribution_stats.json # Means, SDs, skewness per item
│       │   └── flags.json           # Items/subjects that failed checks
│       │
│       └── generation_log.json      # Per-item: method used (LLM/template/hybrid),
│                                    # attempts needed, fallbacks triggered
│
├── training_data/                   # Real human data for calibration
│   └── {dataset_name}/
│       ├── data.csv
│       ├── metadata.json            # Source, N, population, measures
│       └── benchmarks.json          # Computed benchmarks from real data
│
├── calibration/
│   ├── covariance_matrices/         # Estimated construct correlations
│   │   └── {experiment_type}.json
│   ├── distribution_targets/        # Target distributions from real data
│   │   └── {item_type}.json
│   └── prompt_performance.json      # Which prompt versions produce best quality
│
└── response_library/                # Template response library (see template-system.md)
    └── ...
```

### When to Archive

- **Always**: After any simulation run that produces ≥50 participant rows
- **Quality metrics**: Computed automatically during the validation step — no extra work needed
- **Prompt snapshots**: Capture the exact prompt text used, including any per-run customizations
- **Generation log**: Track which generation method (LLM vs template vs hybrid) was used for each item, how many attempts were needed, and whether fallbacks were triggered

### Git Workflow for Archives

```bash
# After simulation completes and passes quality checks:
git add simulation_archive/runs/{latest_run}/
git commit -m "archive: {experiment_name} N={n} quality_score={score}"
git push origin main
```

Archives should be committed to the repository automatically. They are lightweight (CSVs + JSON) and invaluable for calibration.

## Calibration Against Real Human Data

When real human data is available for the same or similar experiment, use it as a benchmark following the Manning & Horton train/test methodology.

### Train/Test Protocol

1. **Split real data**: If you have one dataset, split 70/30 train/test. If you have multiple datasets from similar experiments, use one for training and another for testing.

2. **Calibrate on training data**:
   - Compare simulated response distributions (means, SDs, skewness) to training data distributions
   - Compute item-level KS statistics (continuous items) or chi-square statistics (categorical items)
   - Identify items where simulated output deviates most from human data
   - Adjust: persona parameters, noise levels, acquiescence bias, covariance matrix entries
   - Re-simulate and compare again (iterate until KS p-values > 0.05 on training set)

3. **Validate on testing data**:
   - Run the calibrated simulation against the held-out test data
   - If test-set quality metrics are comparable to training-set metrics → calibration is generalizing
   - If test-set metrics are substantially worse → overfitting to training data; simplify the calibration

4. **Store calibrated parameters** in `calibration/` for reuse in future simulations of similar experiments.

### What to Compare

| Metric | Computation | Target |
|--------|-------------|--------|
| Item means | Mean per item, simulated vs real | Within 0.5 SD of real mean |
| Item SDs | SD per item | Ratio between 0.7 and 1.3 of real SD |
| Scale reliability | Cronbach's alpha per scale | Within 0.1 of real alpha |
| Construct correlations | Inter-scale correlations | Within 0.15 of real correlations |
| Effect sizes (between conditions) | Cohen's d per condition contrast | Within CI of real effect size |
| Open-text length distribution | Mean, median, SD of word counts | Within 20% of real |
| Optional field completion rate | Proportion non-missing | Within 10pp of real rate |
| Attention check failure rate | Proportion failing | Within 3pp of real rate |
| Response time distribution (if applicable) | Median, IQR | Within 20% of real |

### When No Real Data Exists

If no real human data is available for calibration:
- Search the literature for published experiments with similar designs (this is where the Step 0c Research Scan is critical)
- Use published norms for standard scales (e.g., Big Five means and SDs are well-documented across many populations)
- Use general benchmarks: Likert SD typically 1.2–1.8 on 7-point scales, Cronbach's alpha typically 0.7–0.9 for validated scales, attention check failure rates of 3–8%
- Flag the simulation as "uncalibrated" in the archive and prioritize collecting real data for future calibration

## Quality Tracking Across Runs

Maintain a running quality log that tracks how the simulator performs over time.

### Quality Score Computation

Compute a composite quality score after each run:

```python
def compute_quality_score(simulated_data, config, real_data=None):
    scores = {}
    
    # 1. Distribution realism (0-1)
    scores['distribution'] = assess_distribution_realism(simulated_data, real_data)
    
    # 2. Intra-subject consistency (0-1)
    scores['consistency'] = assess_intra_subject_consistency(simulated_data, config)
    
    # 3. Open-text quality (0-1)
    scores['open_text'] = assess_open_text_quality(simulated_data, config)
    
    # 4. Human-likeness (0-1) — AI detection heuristics inverted
    scores['human_likeness'] = assess_human_likeness(simulated_data)
    
    # 5. Schema compliance (0-1)
    scores['schema'] = assess_schema_compliance(simulated_data, config)
    
    # Weighted composite
    weights = {
        'distribution': 0.25,
        'consistency': 0.30,  # highest weight — this is the hardest to get right
        'open_text': 0.20,
        'human_likeness': 0.15,
        'schema': 0.10
    }
    composite = sum(scores[k] * weights[k] for k in scores)
    
    return {'composite': composite, 'components': scores}
```

### Regression Detection

After each run, compare quality scores against the running average:
- If composite score drops > 10% from the 5-run moving average → **ALERT**: quality regression detected
- Identify which component(s) regressed
- Trace back to what changed (prompt modification, parameter change, new template)
- Revert the change if it caused the regression

### Performance Tracking Dashboard Data

Store time-series data for each component score across runs. This enables:
- Visualization of quality trends over time
- Identification of which changes improved vs degraded quality
- Evidence-based decisions about prompt engineering and template evolution

## Prompt Evolution

The prompts used to generate responses are themselves improvable artifacts.

### Prompt Versioning

- Every distinct prompt configuration gets a version identifier
- When modifying prompts, create a new version (never overwrite)
- Track which prompt version was used in each run
- Compare quality scores across prompt versions to identify improvements

### Prompt Improvement Triggers

Improve prompts when:
1. A specific item type consistently fails quality checks across multiple runs
2. Real human data becomes available and reveals a systematic bias in simulated output
3. A new detection method is published that the current output would fail
4. The template fallback is being triggered too frequently (>20% of items for a given type)

### Prompt Improvement Protocol

1. Identify the weakest quality component from the most recent runs
2. Search for recent literature or best practices relevant to that weakness
3. Draft an improved prompt version
4. Run a small comparison (N=50): old prompt vs new prompt on the same config
5. Compare quality scores — adopt the new version only if it improves the target metric WITHOUT degrading other metrics
6. Archive the comparison results for future reference

## Feedback Integration

### User Feedback Loop

When the user identifies quality issues in simulation output:
1. Log the specific issue (what was wrong, which item/subject)
2. Classify the issue (distribution | consistency | open-text | realism | schema)
3. Trace to root cause (prompt | template | parameter | architecture)
4. Implement fix
5. Add a regression test that would catch this issue in future runs
6. Update quality checks to detect this class of issue automatically

### Automated Feedback

The quality metrics system generates automated feedback:
- Per-item quality flags → inform which items need better prompts or templates
- Per-subject quality flags → inform which persona configurations produce poor output
- Aggregate trend data → inform whether recent changes are improvements or regressions

## Bootstrapping (First Runs)

When the simulator has no accumulated data:

1. **Use published norms as initial calibration** — Big Five scales, standard attitude measures, etc. have well-documented distributions.
2. **Start with conservative persona parameters** — moderate trait values, moderate noise, moderate correlations. Undershoot rather than overshoot on variance.
3. **Run initial simulations at small N** (N=50) and manually audit before scaling up.
4. **Actively seek real data for calibration** — use Step 0c Research Scan to find published datasets.
5. **After 5-10 runs**, the accumulated archive provides enough data to begin self-calibration.

## Degradation Prevention

Over time, accumulated fixes and optimizations can interact in unexpected ways. Prevent degradation by:

1. **Never remove quality checks** — only add new ones. Checks are monotonically increasing.
2. **Run the full quality battery on every run** — not just the checks relevant to the latest change.
3. **Maintain a "golden set"** — a small set of simulation configs (3-5) that are re-run periodically as regression tests. If quality drops on these, something broke.
4. **Version everything** — prompts, templates, parameters, covariance matrices. Any change can be reverted.
