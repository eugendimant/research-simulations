# Detailed Protocol Reference

Extended guidance for the simulator agent protocol. Read this file when handling COMPLEX tasks, data quality improvements, or unfamiliar scenarios.

## Complexity Assessment: Worked Examples

### TRIVIAL
- Update a UI label in Streamlit → 1 loop
- Fix a typo in a validation message → 1 loop
- Change a default parameter value → 1 loop

### MODERATE
- Add a new column to simulation output CSV → 3 loops
- Refactor a validation function to handle new edge case → 3 loops
- Update persona parameter ranges with new bounds → 3 loops
- Fix a branching logic bug affecting one condition → 3 loops

### COMPLEX
- Implement a new simulation mode (e.g., dyadic interactions) → 5 loops
- Restructure the data pipeline for streaming output → 5 loops
- Add a new experiment type with custom branching → 5 loops
- Integrate a new statistical validation method → 5 loops

### RESEARCH_NEEDED
- "Use a new theory-grounded persona framework I haven't seen before" → STOP, ask
- "Integrate with an API you haven't used" → STOP, ask
- "Implement a novel elicitation technique" → STOP, ask

## Data Quality Validation: Deep Checks

When changes affect simulation output, apply these domain-specific checks:

### Statistical Distribution Checks
- **Variance**: Simulated responses should show variance comparable to human pilot data when available. If not available, variance should be non-trivial (no constant columns, no binary-only responses on Likert scales).
- **Central tendency**: Means should not cluster at scale midpoints unless theoretically justified.
- **Correlations**: Within-persona response patterns should show internal consistency (e.g., a "prosocial" persona should show correlated prosocial responses across items).
- **Between-condition differences**: If the experiment has conditions, simulated data should show plausible effect sizes, not zero effects or implausibly large ones.

### Persona Consistency Checks
- Each persona type should produce responses within its theoretically expected range.
- No persona should produce responses that violate its defining construct (e.g., a "risk-averse" persona choosing the riskiest option consistently).
- Persona assignment should be balanced as specified in mixture priors.

### Survey Structure Checks
- Skip logic: Items not shown must be NA, never filled.
- Branching: Condition-specific items must appear only in the relevant condition.
- Response scales: All responses must fall within the defined scale range.
- Required items: No missing values where the survey requires a response.
- Open-ended responses: Must show lexical diversity; flag if >30% share phrasing patterns.

## Git Conflict Resolution Protocol

When conflicts arise during rebase:

1. **Identify the conflict type:**
   - Config file (package.json, requirements.txt) → Keep remote version unless the task explicitly added a dependency
   - Code file → Read BOTH versions, understand both intents
   - Data file (CSV schemas, JSON configs) → Merge field-by-field

2. **Resolution steps:**
   ```
   a) Open conflicting file
   b) Remove ALL conflict markers (<<<<<<, ======, >>>>>>)
   c) Merge logically — preserve both intents where possible
   d) Re-run tests on the merged code
   e) If tests fail, root-cause the merge logic
   f) Continue rebase only when tests pass
   ```

3. **Never:**
   - Blindly accept "ours" or "theirs"
   - Leave conflict markers in committed code
   - Skip testing after resolution

## Commit Message Convention

Format: `<type>(<scope>): <description>`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code restructuring (no behavior change)
- `test`: Adding or modifying tests
- `docs`: Documentation only
- `perf`: Performance improvement
- `chore`: Maintenance (dependency updates, config)

Scope: Module or component name (e.g., `simulation`, `validation`, `streamlit`, `personas`, `csv-export`)

Examples:
```
feat(simulation): add dyadic interaction mode
fix(validation): handle empty CSV in schema check
refactor(personas): extract trait calculation into helper
test(csv-export): add edge case for missing headers
```

## Error Handling Standards

All new code in the simulator must follow these error handling patterns:

### Input Validation
```python
# Validate at function boundaries, fail fast
def simulate_experiment(config: dict) -> pd.DataFrame:
    if not config.get("n_participants"):
        raise ValueError("n_participants is required and must be > 0")
    if config["n_participants"] < 1:
        raise ValueError(f"n_participants must be positive, got {config['n_participants']}")
```

### Actionable Error Messages
```python
# Bad: silent failure or cryptic message
return None  # or raise Exception("error")

# Good: explain what went wrong and what to do
raise ValueError(
    f"Persona '{persona_name}' not found in library. "
    f"Available personas: {', '.join(library.keys())}. "
    f"Check spelling or add the persona to the library first."
)
```

### Logging for Debugging
```python
import logging
logger = logging.getLogger(__name__)

# Log at appropriate levels
logger.debug(f"Processing participant {pid} with persona {persona}")
logger.info(f"Simulation complete: {n_rows} rows generated")
logger.warning(f"Condition imbalance detected: {distribution}")
logger.error(f"CSV schema mismatch: expected {expected}, got {actual}")
```

## Performance Guidelines

For the simulator, performance matters when:
- N_TOTAL > 500 participants
- Batch operations on CSV data
- Streamlit UI responsiveness

Rules of thumb:
- Profile before optimizing (never optimize on intuition)
- Vectorize pandas operations instead of row-by-row loops
- Cache expensive computations in Streamlit with `@st.cache_data`
- If a single operation takes > 5s for N=500, investigate

## Stopping Criteria for Recursive Improvement

Stop iterating when ALL of these are true:
1. All tests pass (unit + integration if applicable)
2. All edge cases from Step 0c are handled
3. Code is readable without inline comments explaining "what" (only "why")
4. No obvious performance bottlenecks for typical use cases
5. The improvement you would make next is cosmetic or speculative

If you have done 3 improvement cycles and still have critical issues, escalate to the user with a clear description of what remains.
