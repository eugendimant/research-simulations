---
name: simulator-agent-protocol
description: Enforces a structured development protocol for the behavioral science simulator software. ALWAYS use this skill whenever the user asks to build, fix, extend, refactor, debug, deploy, or otherwise modify code for the behavioral science simulator, synthetic data pipeline, Streamlit app, or any related tooling. Also trigger when the user mentions simulation quality, data validation, agent personas, experiment design code, GitHub workflows related to the simulator, or anything about making synthetic data more realistic or human-like. This skill ensures every code change goes through complexity assessment, adaptive iteration, and mandatory testing before completion. Even for seemingly small tasks — use this skill.
---

# Simulator Agent Protocol

Structured development protocol for the behavioral science simulator. Every code task follows: **Analyze → Research → Plan → Implement → Validate → Deliver**. No exceptions.

## When This Skill Activates

This skill governs ALL code work on the simulator and related tooling. If the task touches code, config, data pipelines, or deployment for the behavioral science simulator — follow this protocol.

## Step 0: Pre-Execution Analysis (MANDATORY — DO FIRST)

Before writing any code, complete all five substeps. Do not skip any.

### 0a. State the Problem

Write one precise sentence describing what needs to change and why. If you cannot do this, you do not understand the task yet — ask clarifying questions.

### 0b. Complexity Assessment and User Verification

Classify the task. This determines how many iteration loops are required.

| Level | Criteria | Iteration Loops | Example |
|-------|----------|-----------------|---------|
| **TRIVIAL** | Single-file, <20 lines changed, no new dependencies | 1 (implement + verify) | Fix a typo, update a constant |
| **MODERATE** | Multi-file or logic change, clear requirements | 3 (implement → edge cases → full validation) | Add a new validation check, refactor a function |
| **COMPLEX** | Cross-module, new features, architectural decisions | 5 (implement → edge cases → integration → performance → final validation) | New simulation mode, data pipeline restructure |
| **RESEARCH_NEEDED** | Ambiguous requirements, unknown APIs, novel algorithms | **STOP. Ask clarifying questions. Do NOT guess.** | New statistical test, unfamiliar library integration |

**Classification procedure:**
1. State your proposed classification and justify it in one sentence.
2. **Present the classification to the user for confirmation.** Ask: "I'm classifying this as [LEVEL] ([N] iteration loops). Does that match your sense of the scope, or should I adjust?" The user can override up or down.
3. If the task straddles two levels, default to the higher one unless the user explicitly says otherwise.
4. Proceed only after user confirms or overrides.

### 0c. Research Scan

Use web search to find relevant information that could improve the implementation. This step scales with complexity:

- **TRIVIAL**: Skip unless the task involves an unfamiliar API or library.
- **MODERATE**: Search for best practices, known pitfalls, or existing solutions related to the specific problem.
- **COMPLEX**: Conduct a thorough research sweep — search for published research, datasets, benchmarks, open-source implementations, and state-of-the-art approaches. For simulation quality work specifically, search for recent papers on LLM-based survey simulation, synthetic data validation, and human-likeness benchmarks.

What to search for (examples):
- Published behavioral science experiments with similar designs (for calibration data)
- Statistical distributions from real human survey responses (for variance targets)
- Recent literature on AI detection in surveys and how to avoid detection artifacts
- Open datasets that could serve as training/calibration benchmarks
- Domain-specific norms and effect sizes relevant to the experiment being simulated

Summarize findings briefly and incorporate them into the plan.

### 0d. Identify Edge Cases

List 3–5 things that could break. For simulator-specific work, always consider:
- Empty/malformed CSV inputs
- Persona parameter edge values (0, 1, negative, missing)
- Condition assignment imbalances
- Survey branching logic failures
- Intra-subject response contradictions (see Intra-Subject Consistency below)
- Concurrent user scenarios (Streamlit)

### 0e. Define Success Criteria

State testable, measurable criteria. Examples: "All existing tests pass," "Output CSV has correct header order," "Response time < 2s for N=500 simulations," "Within-subject scale correlations > 0.3 for theoretically related constructs."

## Step 1: Plan

Before touching code:

1. **Read relevant code files completely** — do not skim. Map dependencies and data flow.
2. **Identify existing patterns** — match the project's conventions (naming, structure, error handling).
3. **Design the change** in 3–7 bullets: what changes, what doesn't, how to verify.
4. **Consider 2+ alternative approaches** — state why the chosen approach wins.
5. **Incorporate research findings** — if Step 0c surfaced relevant patterns, benchmarks, or techniques, integrate them into the design.

## Response Generation Architecture

When the task involves generating or modifying simulated responses, follow the tiered pipeline. Read `references/response-generation-pipeline.md` for full details.

**The core principle: no response exists in isolation.** Every item response is conditioned on the subject's full profile and all prior responses. The pipeline is:

1. **Subject Profile Generation** — demographics (coherent bundle), persona assignment, latent disposition vector (multivariate normal draw from construct covariance matrix), condition assignment. This is the seed for ALL subsequent responses.

2. **Sequential Item Processing** — items processed in survey order. Each item prompt includes: full subject profile + ALL prior responses + disposition anchor for the relevant construct + item-type-specific instructions. The running context grows with each item.

3. **Tiered Generation Strategy** — not every item needs a full LLM call:
   - **LLM generation** (default): open-text, complex conditionals, behavioral decision tasks, novel items
   - **Template generation** (fast fallback): standard Likert items, demographics, constrained-choice items where the latent disposition directly determines the response with calibrated noise
   - **Hybrid** (template skeleton + LLM polish): structured open-text with predictable format
   - See `references/template-system.md` for the template architecture, variation functions, and response library management

4. **Post-Generation Consistency Audit** — after all items, cross-check the full response profile: scale reliability, reverse-code verification, open-text/quantitative alignment, demographic coherence, skip logic, behavioral consistency.

**Fallback trigger:** if LLM generation fails validation 3 times for the same item → fall back to template → log the failure for template improvement.

**Template improvement:** templates are not static. They encode persona parameters, cross-correlations, and variation. They improve continuously via quality-gated library additions, A/B comparisons against LLM output, and distribution calibration against real human data. See `references/template-system.md`.

## Step 2: Implement with Verification Loops

Execute the number of iteration loops determined in Step 0b. Each loop follows this cycle:

```
FOR EACH iteration loop:
  1. Make the minimal change
  2. Run affected tests / linting / type checks
  3. IF failure:
     → Read the actual error (do not skim)
     → Identify root cause (do not guess)
     → Fix precisely
     → Re-test
     → Repeat until green
  4. IF pass:
     → Review your own code critically (see quality gates below)
     → Proceed to next loop or finalize
```

### Quality Gates (check every loop)

Every change must pass ALL of these before proceeding:

- [ ] Handles the edge cases identified in Step 0d
- [ ] Error messages are actionable (not silent failures)
- [ ] No new warnings or linting errors
- [ ] No duplicated logic introduced
- [ ] Backwards compatible (or breaking change documented)
- [ ] Consistent with project style and conventions
- [ ] Intra-subject consistency checks pass (if simulation output is affected)
- [ ] Response generation pipeline maintains subject context continuity (no isolated item generation)
- [ ] Template fallbacks triggered < 20% of items (if higher, investigate prompt quality)
- [ ] Quality score ≥ previous run's score (no regressions)

### Iteration Loop Details by Complexity

**TRIVIAL (1 loop):**
- Implement → run tests → verify → done.

**MODERATE (3 loops):**
- Loop 1: Implement core change → run tests → fix any failures.
- Loop 2: Add edge case handling → review intra-subject consistency implications → run expanded tests.
- Loop 3: Run full test suite → verify all success criteria → confirm no regressions.

**COMPLEX (5 loops):**
- Loop 1: Implement core change → run tests → fix failures.
- Loop 2: Add edge case handling → run expanded tests → review quality gates.
- Loop 3: Integration testing → verify cross-module interactions → check intra-subject consistency end-to-end.
- Loop 4: Performance and realism audit → run human-likeness checks (see below) → optimize bottlenecks.
- Loop 5: Full validation battery → verify ALL success criteria → document tradeoffs → final review.

**Critical rule:** If at any loop, a test fails or a quality gate is not met, do NOT proceed. Fix the root cause first. The loop counter does not advance until all gates pass.

## Intra-Subject Consistency (CRITICAL FOR SIMULATION QUALITY)

Simulated survey responses must never be generated in isolation. Every response from a simulated participant must be coherent with that participant's full response profile. This is the single most important quality dimension — inconsistent within-subject responses are the primary signal that flags synthetic data as fake.

### Mandatory Cross-Correlation Rules

**1. Scale-to-scale consistency:**
- Items measuring the same construct must correlate positively within subjects (e.g., all agreeableness items, all risk aversion items).
- Reverse-coded items must actually reverse. If a participant scores 6/7 on "I enjoy helping others," they should score low on "I prefer to work alone" (if these measure the same construct).
- Theoretically related but distinct constructs should show moderate correlations (e.g., conscientiousness and self-control).
- Theoretically unrelated constructs should NOT show suspiciously high correlations.

**2. Open-text responses must match quantitative responses:**
- If a participant gives a 2/7 on a satisfaction scale and is then asked "Why did you give that rating?", the open text MUST reflect dissatisfaction — not generic positive language.
- If a participant selects "strongly disagree" on a policy item, their written explanation must articulate reasons for disagreement, not hedging or neutral language.
- The specificity and emotional tone of open text should match the extremity of the quantitative response. A 1/7 or 7/7 should produce more emotionally charged text than a 4/7.

**3. Demographic-to-response consistency:**
- A participant who selected "married" as a life event must not later select "single" in demographics.
- Reported education level should be plausible given reported age (e.g., no 18-year-old with a PhD).
- Occupation, income, and education should form plausible combinations.
- Geographic location should align with cultural norms where relevant.

**4. Behavioral consistency across decision tasks:**
- A participant who reveals risk aversion in one task (choosing a sure thing over a gamble) should show risk aversion in related tasks, modulated by domain but not randomly.
- Time preferences should be internally consistent (someone who discounts the future heavily in one question should not suddenly become patient in another).
- Prosocial preferences should be consistent across donation, cooperation, and fairness tasks.

**5. Attention and engagement signals:**
- Not every optional open-text field should be filled. Real humans skip optional fields, especially late in a survey. Simulated subjects who fill every field are a detection signal.
- Response patterns should show some fatigue effects: slightly less variance and faster (shorter open-text) responses toward the end of long surveys.
- A small fraction of subjects should fail attention checks — zero failures across all subjects is a red flag.

**6. Conditional logic coherence:**
- If skip logic routes a participant away from a section, those fields must be NA — never filled.
- Condition-specific items must only appear in the assigned condition.
- Follow-up questions must be consistent with the triggering response (e.g., "You said you exercise regularly — how many times per week?" should not appear for someone who said they never exercise).

### Implementation Requirement

Whenever code changes could affect how individual participant responses are generated, verify intra-subject consistency by:
1. Generating a small sample (N=20–50)
2. Computing within-subject correlations for related constructs
3. Spot-checking 5 individual response profiles end-to-end for logical coherence
4. Checking open-text responses against their quantitative anchors
5. Verifying demographic fields do not contradict each other

## Human-Likeness Realism Checks

Simulated data must be indistinguishable from real human survey data. The following are known detection signals for AI-generated survey responses — the simulator must actively avoid all of them.

### Known AI Detection Signals (Avoid These)

**Response distribution artifacts:**
- LLMs tend to produce lower variance than real humans. Ensure response distributions have realistic spread, including use of scale extremes. Real humans use 1s and 7s; LLMs tend to cluster around 3–5.
- Avoid uniform distributions across scale points — real human data is typically skewed or bimodal depending on the item.
- Condition means should show plausible effect sizes. Neither zero effects nor implausibly large ones.

**Linguistic tells in open-ended responses:**
- Avoid AI-characteristic phrasing: "resilience and determination," "navigating challenges," "fostering a sense of," "I believe that," "it is important to note." These flag synthetic responses immediately.
- Real human open-ended responses are messy: grammatical errors, incomplete sentences, colloquialisms, hedging ("idk," "kinda," "lol"), and varying length. Some are one word. Some ramble.
- Open-ended responses should vary dramatically in length and effort across participants. Some people write paragraphs; many write 2–5 words for optional fields.
- Do not fill optional comment fields for every participant. Real completion rates on optional open-text fields are typically 30–60%, not 100%.

**Temporal and clustering artifacts:**
- Submission times should not cluster in tight bursts. Real respondents arrive with natural spacing — some bunched when a recruitment email goes out, then a long tail.
- Survey completion duration should follow a right-skewed distribution (most finish in a typical range, a long tail of slow respondents, very few extremely fast ones).

**Response pattern artifacts:**
- Real humans show some acquiescence bias (slight tendency to agree). AI agents often show too much or none.
- Real humans occasionally give contradictory responses — not frequently, but never is suspicious.
- Straightlining (choosing the same response for all items in a scale) happens in real data at low rates (~2–5%). Zero straightlining is a red flag.
- "No" avoidance: real respondents answer "no" to screening and eligibility questions at natural rates. AI agents tend to answer "yes" to everything to avoid disqualification.

**Demographic pattern artifacts:**
- The demographic profile of simulated respondents should match the target population specification. A sudden shift in demographics mid-dataset (e.g., a block of male respondents in a previously female-majority sample) is a clustering signal.
- Demographic distributions should reflect realistic recruitment patterns, not perfect quotas. Real data has slight imbalances.

### Paradata Simulation (When Applicable)

If the simulator generates paradata (timing, metadata), ensure:
- Response times per item correlate with item complexity (open-text items take longer than single-click Likert items).
- Total survey duration is right-skewed with realistic mean for the survey length.
- No copy-paste timing signatures (instantaneous text appearance for long responses).

## Step 3: Validate

After all iteration loops complete, run the full validation battery:

```bash
# Run in this order — stop if any step fails
1. Run all unit tests
2. Run linter
3. Run type checker (if applicable)
4. Run build (if applicable)
5. Manual smoke test of primary use case
6. Simulation quality checks (if output is affected — see below)
```

If ANY step fails: do not push, do not deliver. Fix the root cause and re-run from the failed step.

### Simulation Output Validation

For changes that affect simulation output quality, also verify:
- Output variance matches expected human distributions (not compressed toward center)
- Scale extremes (1s and 7s) appear at realistic rates
- Open-text responses show lexical diversity and match quantitative anchors
- Persona parameter consistency is maintained within subjects
- Condition assignments are balanced as specified
- CSV schema matches locked header specification
- No demographic contradictions within any individual subject
- Optional fields have realistic non-completion rates
- At least some attention check failures exist in the sample
- Composite quality score ≥ previous run (check against archive)
- If real calibration data exists, item-level KS tests pass (p > 0.05)
- Generation log shows < 20% template fallback rate on LLM-targeted items
- Response library updated with quality-gated new entries

## Continuous Learning and Calibration

The simulator improves with every run. Read `references/continuous-learning.md` for the full system.

**Auto-archive protocol:** after every simulation run producing ≥50 rows, automatically archive to GitHub: raw output CSV, quality metrics (computed during validation), exact prompt snapshots, simulation configuration, and per-item generation log (LLM vs template, attempts needed, fallbacks triggered).

**Calibration against real data:** when real human data is available (training set), calibrate simulation parameters until item-level KS test p-values > 0.05. Then validate against held-out test data. Store calibrated parameters for reuse. When no real data exists, use published norms and literature-derived benchmarks (search for these in Step 0c).

**Quality tracking:** compute a composite quality score after each run (distribution realism 25%, intra-subject consistency 30%, open-text quality 20%, human-likeness 15%, schema compliance 10%). If score drops > 10% from the 5-run moving average → quality regression alert. Trace to root cause and revert if needed.

**Prompt evolution:** version all prompts. When improving prompts, run A/B comparison (N=50, old vs new) — adopt only if target metric improves WITHOUT degrading others. Track which prompt versions produce the best quality scores.

**Golden set regression tests:** maintain 3–5 simulation configs that are re-run periodically. If quality drops on these, something broke.

## Step 4: Git Workflow

Follow this exactly. Zero tolerance for merge conflicts reaching the user.

```
1. git fetch origin && git pull origin main
2. git checkout -b feature/<descriptive-name>
3. Commit atomically with precise messages:
   ✓ "fix: resolve off-by-one in pagination logic"
   ✗ "fix bug" or "update code"
4. Before push: git fetch origin main && git rebase origin/main
5. IF conflicts: resolve ALL yourself, re-test, then continue rebase
6. Push ONLY if all tests pass
```

**Files to never touch** (unless the task explicitly requires it): README.md, CHANGELOG.md, LICENSE, .gitignore, MEMORY.md.

## Step 5: Deliver

Provide a clear summary:

1. **What changed** — precise description of code changes, files modified and why
2. **Why** — root cause or motivation; alternative approaches considered and rejected
3. **How to verify** — copy-pastable commands
4. **Edge cases handled** — list them
5. **Intra-subject consistency** — how within-subject coherence was verified
6. **Generation method** — which items use LLM vs template vs hybrid, and any fallback triggers
7. **Quality score** — composite and component scores; comparison to previous run
8. **Risks/tradeoffs** — any breaking changes, performance implications, follow-up needed

## Failure Recovery

**If tests fail:**
1. Read the actual error message completely
2. Reproduce with minimal input
3. Add a test that captures the failure
4. Fix the root cause (not the symptom)
5. Verify the fix doesn't break other tests

**If stuck:**
1. State what you tried
2. State what failed
3. State what you don't understand
4. Ask for guidance — do NOT thrash

**If the task is ambiguous:**
1. State 2–3 interpretations
2. State implications of each
3. Ask the user to choose
4. Do NOT guess and implement the wrong thing

## Recursive Improvement (COMPLEX tasks only)

After initial implementation passes all gates, run up to 3 improvement cycles:

```
FOR EACH improvement cycle (max 3):
  1. Review your own code critically
  2. Identify ONE specific weakness: performance | edge case | realism | consistency
  3. Propose a specific improvement (not "make better")
  4. Implement and verify (measurable metric)
  5. IF improvement < 10% gain: STOP iterating
  6. IF critical flaw found: FIX and continue
```

STOP when: all tests pass, all edge cases handled, intra-subject consistency verified, human-likeness checks pass, code is clear and maintainable, further changes would be premature optimization.

## Critical Rules (Never Violate)

1. **Think before code** — if you cannot explain why in 2 sentences, you do not understand it yet
2. **Verify every change** — untested code is broken code
3. **Confirm complexity with user** — always present classification for user approval before proceeding
4. **No conflicts ever** — you resolve them, not the user
5. **Minimal diffs** — touch only what is necessary
6. **No guessing** — if unclear, ask. If research is needed, search the web first.
7. **No half-measures** — fix it completely or do not touch it
8. **Test edge cases** — the happy path is 20% of the work
9. **Fail loudly** — silent failures are unacceptable
10. **Intra-subject coherence** — no simulated response exists in isolation; every answer must be consistent with the full participant profile
11. **Human-likeness** — if a detection algorithm could flag it as AI-generated, it is not good enough
12. **Context continuity** — every item prompt must include the subject's full profile and all prior responses; never generate items without running context
13. **Calibrate or justify** — use real human data when available; when not, cite the published norm or benchmark you are targeting
14. **No regressions** — every run must score ≥ the previous run; if it doesn't, diagnose and fix before delivering
15. **Archive everything** — every simulation run is a training sample; auto-commit to GitHub

## References

For detailed protocol on specific scenarios, see:
- `references/response-generation-pipeline.md` — subject profile generation, sequential item processing, item-type-specific prompting, context window management
- `references/template-system.md` — template architecture, cross-correlation encoding, response library management, variation functions, template improvement loop
- `references/continuous-learning.md` — auto-archive protocol, train/test calibration against real data, quality tracking, prompt evolution, regression prevention
- `references/detailed-protocol.md` — complexity worked examples, git conflict resolution, error handling standards
- `references/human-likeness-checklist.md` — comprehensive checklist for simulation realism audits
