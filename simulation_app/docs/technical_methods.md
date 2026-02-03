# Technical Methods

**Behavioral Experiment Simulation Tool v1.0.0**

---

## Response Generation Model

### Persona System

The simulation generates individual differences by assigning each simulated participant a "persona" that determines their response patterns throughout the survey.

**Persona Distribution** (based on published estimates):

| Persona | Weight | Source |
|---------|--------|--------|
| Engaged Responder | 30-35% | Krosnick (1991) "optimizers" |
| Satisficer | 20-25% | Krosnick (1991) satisficing theory |
| Extreme Responder | 8-12% | Greenleaf (1992) ERS prevalence |
| Acquiescent | 6-8% | Billiet & McClendon (2000) |
| Careless | 3-8% | Meade & Craig (2012) online samples |
| Socially Desirable | 10-15% | Paulhus (1991) |

Each persona specifies probability distributions for traits like response tendency, extremity, acquiescence, and attention. Individual participants sample from these distributions, so there's variation within personas too.

### Likert Scale Algorithm

Scale responses go through nine steps:

1. **Base tendency**: Start from persona's preferred scale region
2. **Domain calibration**: Adjust for construct norms (satisfaction runs high, anxiety runs low)
3. **Condition effect**: Apply experimental effect based on semantic parsing of condition name
4. **Reverse coding**: Handle reverse-coded items, with acquiescence artifacts
5. **Within-person variance**: Add noise based on persona consistency trait
6. **Extreme responding**: Probabilistically shift to endpoints based on ERS trait
7. **Acquiescence bias**: Inflate responses based on acquiescence trait
8. **Social desirability**: Further adjustment for self-presentation
9. **Boundary enforcement**: Round and clamp to valid scale range

The result: realistic means (4.0-5.2 on 7-point scales), realistic SDs (1.2-1.8), realistic distributions.

### Effect Size Implementation

When you configure an effect, the simulation applies it based on the semantic content of condition names—not their order. The algorithm parses condition names for valence keywords:

- Positive keywords: "high", "good", "positive", "friend", "reward"
- Negative keywords: "low", "bad", "negative", "enemy", "punishment"
- Domain-specific: "AI" triggers algorithm aversion adjustment

For factorial designs (e.g., "AI × Hedonic"), each factor's contribution is parsed and combined.

Effect magnitudes follow Cohen (1988):
- Small: d = 0.20
- Medium: d = 0.50
- Large: d = 0.80

The meta-analytic average for social psychology is d = 0.43 (Richard et al., 2003), so "medium" effects are realistic for most studies.

---

## Survey Flow Logic

### Question Visibility

The simulation respects survey branching. If a question only appears for certain conditions, participants in other conditions get blank responses.

**Detection methods**:
1. Explicit condition restrictions in question metadata
2. DisplayLogic parsing from QSF
3. Block name analysis (e.g., "AI_Block" → only AI conditions see it)
4. Question text hints (e.g., "For those who saw the AI recommendation...")
5. Factor-level matching for factorial designs

For factorial designs, the handler parses crossed conditions (e.g., "AI × Hedonic") and handles partial visibility correctly—if a question is for "AI" participants, both "AI × Hedonic" and "AI × Utilitarian" participants see it.

### DisplayLogic Parsing

The system parses Qualtrics DisplayLogic structures to extract:
- Logic type (BooleanExpression, etc.)
- Conditions that must be met
- Question dependencies (which questions must be answered first)

This information determines which simulated participants would actually see each question.

---

## Open-Ended Response Generation

### Method

Text responses use template-based generation with:
1. **Domain detection**: 175+ research domains identified by keyword matching
2. **Question type classification**: 30+ types (explanation, evaluation, feedback, etc.)
3. **Sentiment alignment**: Response valence matches participant's overall scale responses
4. **Persona modulation**: Verbosity, formality, and engagement adjust output
5. **Question-specific personalization**: Keywords from question text shape the response

### Persona Effects on Text

| Persona | Text Characteristics |
|---------|----------------------|
| Engaged | Multi-sentence, detailed, on-topic |
| Satisficer | Short, minimal effort, generic |
| Careless | Very short, typos, sometimes off-topic |
| Extreme | Strong language, emphatic |
| Acquiescent | Agreeable tone, positive framing |

### Uniqueness

Each question gets different responses because:
- Question text is parsed for keywords that modify the template
- Question type determines response structure
- Participant seed combines with question hash for reproducible variation
- Domain detection may differ per question

---

## Exclusion Criteria Simulation

The simulation generates realistic exclusion flags:

| Flag | Basis |
|------|-------|
| Completion time | Too fast or too slow based on persona attention |
| Attention check failures | Careless and satisficer personas fail at higher rates |
| Straight-lining | Low-consistency personas produce more straight-line patterns |

Flags can be combined into an overall "exclude recommended" flag based on configurable thresholds.

---

## Reproducibility

The simulation uses seeded random number generation throughout:
- Main seed: Timestamp + study hash if not specified
- Participant seeds: Main seed + participant index
- Column seeds: Main seed + column hash

Same seed + same parameters = identical output across runs.

Hashes use MD5 (not Python's `hash()`) for cross-platform stability.

---

## Validation

### Effect Recovery

The simulation includes diagnostic checks that compare configured effect sizes to observed effects in generated data. Strong order effects (correlation > 0.7 between condition position and means) trigger warnings—effects should come from semantic content, not position.

### Distributional Checks

Generated data should match:
- Likert means: 4.0-5.2 on 7-point scales
- Likert SDs: 1.2-1.8
- Attention pass rates: 85-95% depending on difficulty
- Careless responder rate: 3-10%

---

## References

Billiet, J. B., & McClendon, M. J. (2000). Modeling acquiescence in measurement models for two balanced sets of items. *Structural Equation Modeling, 7*(4), 608-628.

Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Erlbaum.

Greenleaf, E. A. (1992). Measuring extreme response style. *Public Opinion Quarterly, 56*(3), 328-351.

Krosnick, J. A. (1991). Response strategies for coping with the cognitive demands of attitude measures in surveys. *Applied Cognitive Psychology, 5*(3), 213-236.

Meade, A. W., & Craig, S. B. (2012). Identifying careless responses in survey data. *Psychological Methods, 17*(3), 437-455.

Paulhus, D. L. (1991). Measurement and control of response bias. *Measures of Personality and Social Psychological Attitudes*, 17-59.

Richard, F. D., Bond Jr, C. F., & Stokes-Zoota, J. J. (2003). One hundred years of social psychology quantitatively described. *Review of General Psychology, 7*(4), 331-363.

---

*© 2026 Dr. Eugen Dimant. Proprietary.*
