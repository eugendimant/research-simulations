# Behavioral Experiment Simulation Tool — Methods Summary

**Version 2.4** | Created by Dr. [Eugen Dimant](https://eugendimant.github.io/)

---

## Overview

The Behavioral Experiment Simulation Tool generates realistic synthetic pilot data for behavioral science experiments. It automates the creation of datasets that mirror real survey response patterns, enabling researchers and students to:

- Test analysis pipelines before data collection
- Practice data cleaning and exclusion procedures
- Verify survey logic and variable coding
- Develop analysis scripts on realistic data structures
- Learn to interpret statistical outputs on data with known properties

This tool is designed for **educational and pilot testing purposes**; it does not replace actual human subjects research but provides a rigorous training ground for developing analytical skills.

---

## Key Features (v2.4)

### Automatic Survey Parsing
Upload your Qualtrics QSF file and the tool automatically:
- Detects experimental conditions from survey flow and randomizers
- Identifies Likert scales with their specific point ranges (5-point, 7-point, 10-point, etc.)
- Extracts open-ended question specifications
- Infers factorial design structure (2×2, 2×3, etc.)
- Detects attention checks and manipulation checks

### Enhanced Scale Detection
The tool now detects four types of scales from QSF files:
- **Matrix scales**: Likert-type matrix questions with multiple items
- **Numbered items**: Questions with _1, _2, _3 suffixes (e.g., WTP_1, WTP_2)
- **Single-choice Likert**: Individual MC questions grouped by variable prefix
- **Slider scales**: Visual analog and slider-type questions (0-100 range)

### Visual Factorial Design Table
For experiments with multiple independent factors:
- **Intuitive table interface** to assign detected conditions to factors
- **Visual crossing display** showing all condition combinations
- **Automatic condition generation** (e.g., "Dictator game + Match with Hater")
- Supports 2-factor and 3-factor designs
- State persistence when navigating between steps

Example for a 2×3 design:
| Game Type | Match with Hater | Match with Lover | Match with Unknown |
|-----------|------------------|------------------|-------------------|
| **Dictator game** | ✓ | ✓ | ✓ |
| **PGG** | ✓ | ✓ | ✓ |

This generates 6 properly crossed conditions automatically.

### 175+ Research Domains
Context-aware response generation across 24 categories:

**Behavioral Economics (12 domains)**
- Dictator game, ultimatum, trust game, public goods, risk preferences
- Time preferences, loss aversion, fairness, reciprocity, cooperation

**Social Psychology (15 domains)**
- Intergroup relations, social identity, norms, conformity, prosocial behavior
- Prejudice, stereotypes, attitude change, persuasion, self-concept

**Political Science (10 domains)**
- Polarization, partisanship, voting behavior, media effects, policy attitudes
- Political trust, civic engagement, democracy, ideology

**Consumer/Marketing (10 domains)**
- Brand attitudes, advertising effectiveness, pricing psychology
- Purchase intentions, customer satisfaction, product evaluation

**Organizational Behavior (8 domains)**
- Leadership, teamwork, motivation, job satisfaction
- Organizational commitment, work engagement, performance

**Technology & AI (8 domains)**
- AI attitudes, automation anxiety, human-robot interaction
- Privacy concerns, technology adoption, digital trust

**Additional Categories**
- Health Psychology, Education, Environmental, Ethics & Morality
- Cognitive Psychology, Communication, Development, and more

### 30 Question Type Handlers
Comprehensive open-ended response generation:

**Explanatory Questions**
- Explanation, justification, reasoning, causation, motivation

**Descriptive Questions**
- Description, narration, elaboration, detail

**Evaluative Questions**
- Evaluation, assessment, comparison, critique, rating explanation

**Reflective Questions**
- Reflection, introspection, self-assessment

**Opinion/Attitude Questions**
- Opinion, attitude, preference, belief

**Forward-Looking Questions**
- Prediction, intention, expectation, recommendation

**Feedback Questions**
- Feedback, suggestion, complaint, compliment

### Attention & Manipulation Check Management
Users can now:
- Review automatically detected attention checks
- Review detected manipulation checks
- Manually add or edit check questions
- Track which checks will be simulated

### Behaviorally Realistic Response Generation
The simulator generates responses that reflect actual human survey behavior:
- **Response styles**: Engaged, satisficing, extreme responding, acquiescence
- **Attention patterns**: Realistic attention check pass/fail rates
- **Individual differences**: Trait-based variation across participants
- **Sentiment alignment**: Open-ended responses consistent with numeric ratings
- **Careless responding**: Realistic patterns of inattention for exclusion practice

### Complete Output Package
Each simulation produces:

| File | Description |
|------|-------------|
| `Simulated_Data.csv` | Generated dataset with all variables |
| `Study_Summary.md` | Summary report in Markdown format |
| `Study_Summary.html` | Same summary in browser-viewable HTML |
| `Metadata.json` | Simulation parameters and settings |
| `Schema_Validation.json` | Data quality checks and validation |
| `R_Prepare_Data.R` | Analysis-ready R script |
| `Data_Codebook_Handbook.txt` | Complete variable descriptions |

**Instructor-Only Reports** (sent via email, not in student download):
- Comprehensive statistical analysis with visualizations
- T-tests, ANOVA, regression results
- Effect size calculations
- Hypothesis evaluation against pre-registration

---

## Persona-Based Simulation Methodology

### Theoretical Foundation

The persona-based approach is grounded in decades of survey methodology research demonstrating that participants systematically differ in how they respond to surveys. Rather than treating all respondents as homogeneous, this tool simulates the heterogeneous response patterns observed in real data.

### Core Response Style Personas

The following personas represent well-documented response styles from the survey methodology literature:

| Persona | Weight | Theoretical Basis | Key Characteristics |
|---------|--------|-------------------|---------------------|
| **Engaged Responder** | 30% | Ideal respondent model | High attention (μ=0.95), full scale use, consistent responses |
| **Satisficer** | 20% | Krosnick (1991) satisficing theory | Midpoint tendency, faster completion, minimal text |
| **Extreme Responder** | 8% | Greenleaf (1992) ERS literature | Endpoint use (1s and 7s), emphatic text |
| **Acquiescent Responder** | 7% | Paulhus (1991) yes-saying bias | Agreement tendency regardless of content |
| **Careless Responder** | 5% | Meade & Craig (2012) | Low attention, fails checks, inconsistent |

### Trait-Based Individual Differences

Each persona defines probability distributions for behavioral traits. When a participant is assigned a persona, their individual trait values are sampled from these distributions, creating realistic within-persona variation:

**Core Traits:**
- **Attention Level**: Probability of reading carefully (affects attention check performance)
- **Response Consistency**: Stability across similar items (affects reliability)
- **Scale Use Breadth**: Tendency toward endpoints vs. midpoints
- **Acquiescence**: Agreement bias independent of item content
- **Social Desirability**: Tendency to present favorably
- **Reading Speed**: Affects completion time simulation

### Domain-Specific Personas

The tool includes specialized personas activated by keywords in your study description:

**Consumer Behavior & Marketing:**
- Brand Loyalist (high attachment, low price sensitivity)
- Deal Seeker (high price sensitivity, low brand loyalty)
- Impulse Buyer (low deliberation, high novelty seeking)
- Conscious Consumer (values-driven purchasing)
- Early Adopter (high novelty seeking, tech enthusiasm)

**AI & Technology Research:**
- Tech Enthusiast (high AI trust, optimistic about automation)
- Tech Skeptic (privacy concerned, cautious about AI)
- AI Pragmatist (balanced, use-case dependent attitudes)
- Privacy-Concerned User (data sensitivity, security focus)

**Behavioral Economics:**
- Loss-Averse Agent (prospect theory patterns)
- Present-Biased Agent (hyperbolic discounting)
- Rational Deliberator (utility-maximizing patterns)
- Status Quo Defender (endowment effect, inertia)

**Organizational Behavior:**
- High Performer (engaged, achievement-oriented)
- Disengaged Employee (cynical, low commitment)
- Team Player (prosocial, collaborative)

**Social Psychology:**
- Prosocial Individual (high empathy, cooperation)
- Individualist (self-focused, competitive)
- Conformist (social influence susceptibility)

---

## Open-Ended Response Generation

### Methodology

Open-ended text responses are generated using a multi-layer approach:

1. **Sentiment Alignment**: Text sentiment matches numeric response patterns
2. **Persona Voice**: Language style reflects the assigned persona
3. **Topic Relevance**: Responses address the study topic appropriately
4. **Natural Variation**: Hedging, fillers, and varied sentence structures
5. **Domain Awareness**: Responses reflect the 175+ supported research domains

### Response Variation for Large Samples

To ensure uniqueness across thousands of data points, the system applies:
- **Synonym substitution** for word-level variation
- **Filler phrase insertion/removal** based on persona verbosity
- **Punctuation style variation** (periods, ellipses, exclamation marks)
- **Realistic typos** for low-attention respondents
- **Trailing phrase additions** for conversational variation

### Response Characteristics by Persona

| Persona | Verbosity | Detail | Coherence | Example Pattern |
|---------|-----------|--------|-----------|-----------------|
| Engaged | Moderate | Specific | High | Thoughtful, balanced responses |
| Satisficer | Minimal | Vague | Low | "It was fine." "OK I guess." |
| Extreme | Moderate | Emphatic | Moderate | Strong opinions, superlatives |
| Acquiescent | Moderate | Agreeable | Moderate | Positive framing, agreement |
| Careless | Minimal | Irrelevant | Very Low | Off-topic, nonsensical |

---

## Research Foundations

### Literature-Grounded Simulation (v2.4)

The simulation engine now incorporates **100+ manipulation effects** from **75+ published sources** across 16 research domains. Effect sizes are calibrated to meta-analytic findings where available.

#### Manipulation Effect Categories (75+ Sources)

| Domain | Key Sources | Example Effects |
|--------|-------------|-----------------|
| **AI/Technology** | Dietvorst et al. (2015); Epley et al. (2007) | Algorithm aversion d≈-0.3, Anthropomorphism d≈+0.3 |
| **Consumer/Marketing** | Babin et al. (1994); Barton meta (2022) | Scarcity d≈+0.30, Social proof r≈0.15 |
| **Social Psychology** | Tajfel (1971); Balliet meta (2014); Milgram (1963) | In-group bias d≈0.3-0.5, Authority +46pp compliance |
| **Behavioral Economics** | Tversky & Kahneman (1981); Johnson & Goldstein (2003) | Loss aversion λ≈2.0, Defaults +60-80pp |
| **Game Theory** | Fehr & Gächter (2000); Sally meta (1995) | PGG punishment +40%, Cooperation ≈47% |
| **Health/Risk** | Bandura (1977); Witte & Allen meta (2000) | Self-efficacy +0.22, Fear appeals d≈0.3-0.5 |
| **Organizational** | Colquitt meta (2001); Judge & Piccolo meta (2004) | Justice ρ≈.40-.50, Leadership ρ≈.44 |
| **Political/Moral** | Graham & Haidt (2009); Iyengar & Westwood (2015) | Polarization d>0.5, Moral foundations |
| **Cognitive/Decision** | Iyengar & Lepper (2000); Staw (1976); Trope & Liberman (2010) | Choice overload d≈0.77, Sunk cost d≈0.37 |
| **Communication** | Hovland & Weiss (1951); McGuire (1961); Banas & Rains meta (2010) | Source credibility +0.20, Inoculation d≈0.29 |
| **Learning/Memory** | Roediger & Karpicke (2006); Rowland meta (2014) | Testing effect d≈0.50, Spacing effect robust |
| **Social Identity** | Pettigrew & Tropp meta (2006); Gaertner (1993) | Contact r≈-0.21, Common identity +0.22 |
| **Motivation** | Gollwitzer & Sheeran meta (2006); Deci et al. meta (1999) | Implementation intentions d≈0.65, Crowding out d≈-0.40 |
| **Environmental** | Anderson et al. (2000); Berman et al. (2008) | Heat +aggression, Nature +mood |
| **Embodiment** | Coles et al. many-labs (2019); Credé & Phillips (2017) | Facial feedback r≈0.03 (contested) |
| **Temporal** | Dror et al. (1999); Sievertsen et al. (2016) | Time pressure -0.15, Circadian effects |

### Core LLM Simulation Research

The use of language models and algorithmic approaches for simulating human survey responses is supported by a growing body of rigorous research:

**Foundational Validation Studies:**

1. **Argyle, L. P., Busby, E. C., Fulda, N., Gubler, J. R., Rytting, C., & Wingate, D. (2023)**. Out of one, many: Using language models to simulate human samples. *Political Analysis*, 31(3), 337-351.
   - **DOI:** [10.1017/pan.2023.2](https://doi.org/10.1017/pan.2023.2)
   - **Key Finding:** LLMs achieve "algorithmic fidelity"—replicating human response distributions.

2. **Horton, J. J. (2023)**. Large language models as simulated economic agents: What can we learn from Homo Silicus? *NBER Working Paper* No. 31122.
   - **Key Finding:** LLM agents exhibit canonical economic behaviors including status quo bias and anchoring.

3. **Aher, G. V., Arriaga, R. I., & Kalai, A. T. (2023)**. Using large language models to simulate multiple humans. *ICML 2023*.
   - **Key Finding:** "Turing Experiments" successfully replicate classic psychology studies.

4. **Binz, M. & Schulz, E. (2023)**. Using cognitive psychology to understand GPT-3. *PNAS*, 120(6).
   - **Key Finding:** LLMs exhibit human-like patterns in decision-making and cognitive biases.

5. **Westwood, S. J. (2025)**. The potential existential threat of LLMs to survey research. *PNAS*, 122(47).
   - **Key Finding:** LLM responses can evade 99.8% of attention checks; underscores simulation standards.

### Survey Methodology Foundations

6. **Krosnick, J. A. (1991)**. Response strategies for coping with cognitive demands. *Applied Cognitive Psychology*.
   - **Key Concept:** Satisficing theory—"good enough" rather than optimal responses.

7. **Meade, A. W., & Craig, S. B. (2012)**. Identifying careless responses. *Psychological Methods*.
   - **Key Concept:** Methods for detecting careless/inattentive responding.

### Behavioral Science Sources (Selected from 75+)

**Decision Making:** Tversky & Kahneman (1981); Thaler (1985); Johnson & Goldstein (2003); Iyengar & Lepper (2000); Scheibehenne meta (2010); Staw (1976); Sleesman meta (2012)

**Social Influence:** Cialdini (2001); Milgram (1963); Asch (1951); Brehm (1966); Rains meta (2013); Hatfield et al. (1993); Bond & Smith meta (1996)

**Motivation:** Deci, Koestner & Ryan meta (1999); Gollwitzer & Sheeran meta (2006); Dweck (2006); Sisk meta (2018); Higgins (1997)

**Intergroup Relations:** Tajfel (1971); Allport (1954); Pettigrew & Tropp meta (2006); Gaertner (1993); Balliet meta (2014)

**Learning/Memory:** Roediger & Karpicke (2006); Cepeda meta (2006); Rowland meta (2014); Bjork (1994)

**Communication:** Hovland & Weiss (1951); McGuire (1961); Allen & Preiss (1997); Banas & Rains meta (2010)

**Embodiment/Context:** Anderson et al. (2000); Berman (2008); Schnall (2008); Strack (1988); Coles many-labs (2019)

---

## Standardization Features

### Why Standardization Matters

When multiple teams simulate pilot data for coursework or research training, standardized defaults ensure:
- **Comparable datasets** across teams and projects
- **Consistent exclusion criteria** for fair assessment
- **Reproducible simulation parameters** for verification
- **Fair evaluation** of different analysis approaches

### Default Settings (Simple Mode)

| Parameter | Default Value | Rationale |
|-----------|--------------|-----------|
| Gender distribution | 50% male | Balanced representation |
| Mean age | 35 years | Adult population midpoint |
| Age SD | 12 years | Realistic adult variance |
| Attention check pass rate | 95% | Typical online sample quality |
| Random responder rate | 5% | Expected careless rate |
| Min completion time | 60 seconds | Impossibly fast threshold |
| Max completion time | 1800 seconds | 30 minute maximum |
| Straight-line threshold | 10 items | Consecutive identical responses |

Advanced mode allows full customization of all parameters, including effect size specification for power analysis practice.

---

## Exclusion Criteria Simulation

The tool simulates realistic data quality issues to provide practice with exclusion decisions:

| Flag | Description | Detection Method |
|------|-------------|------------------|
| `Flag_Speed` | Completion time outside acceptable range | Time < min or > max |
| `Flag_Attention` | Failed attention check(s) | Incorrect attention responses |
| `Flag_StraightLine` | Excessive straight-line responding | ≥ threshold consecutive identical |
| `Flag_Careless` | Overall careless responding pattern | Multiple indicators combined |
| `Exclude_Recommended` | Recommended for exclusion | Any flag triggered |

These flags enable students to practice:
- Applying pre-registered exclusion criteria
- Making defensible exclusion decisions
- Understanding the impact of exclusions on results

---

## Effect Size Simulation (Advanced Mode)

For power analysis training, Advanced Mode allows specification of expected effects:

- **Select dependent variable**: Choose which scale shows the effect
- **Select factor**: Which experimental factor creates the effect
- **Set magnitude**: Cohen's d from 0.0 (null) to 1.5 (very large)
- **Set direction**: Which condition scores higher/lower

This allows students to:
- Understand how effect sizes manifest in real data
- Practice power analysis interpretation
- See how sample size affects statistical significance

---

## Instructor Analysis Reports

The tool generates comprehensive instructor-only reports (not included in student downloads) containing:

### Statistical Analyses
- **Descriptive statistics**: Means, SDs, confidence intervals by condition
- **Primary tests**: Independent samples t-tests, one-way ANOVA
- **Effect sizes**: Cohen's d with confidence intervals
- **Factorial ANOVA**: For multi-factor designs
- **Non-parametric alternatives**: Mann-Whitney U, Kruskal-Wallis

### Visualizations
- Distribution plots by condition
- Means comparison charts with error bars
- Effect size forest plots

### Pre-registration Alignment
- Extracted hypotheses from uploaded pre-registration
- Results mapped to registered predictions
- Badges indicating PRE-REGISTERED vs. ADDITIONAL analyses

---

## Reproducibility

- **Unique Run ID**: Each simulation receives a timestamped identifier
- **Simulation Seed**: Deterministic seeding from study metadata
- **Reproducible Output**: Same seed = identical results
- **Complete Metadata**: All parameters captured in JSON for replication

---

## Technical Requirements

- **Python 3.9+**
- **Core Dependencies**: streamlit, pandas, numpy
- **PDF Generation**: reportlab
- **Email Delivery**: sendgrid (optional)
- **PDF Parsing**: pypdf (optional, for pre-registration upload)

---

## Ethical Considerations

This tool is designed for **educational and pilot testing purposes only**. It should not be used to:
- Fabricate research data for publication
- Misrepresent simulated data as human subjects data
- Bypass IRB requirements for actual research

The tool includes clear labeling in all outputs indicating data is simulated.

---

## Citation

If you use this tool in your research or teaching, please cite:

> Dimant, E. (2026). Behavioral Experiment Simulation Tool (Version 2.4). University of Pennsylvania.

---

## Contact & Support

For questions, issues, or feature requests:
- **Website:** [https://eugendimant.github.io/](https://eugendimant.github.io/)
- **Email:** edimant@sas.upenn.edu
