# Behavioral Experiment Simulation Tool

**Proprietary Software by Dr. Eugen Dimant**

*Generate publication-ready synthetic data for behavioral science experiments*

---

## Overview

The Behavioral Experiment Simulation Tool is a sophisticated platform that generates high-fidelity synthetic datasets for behavioral science research. Whether you're piloting a new experiment, preparing analysis code, or teaching research methods, this tool produces data that mirrors what you would collect from actual human participants.

Unlike simple random number generators, this system applies decades of survey methodology research to create responses that exhibit realistic psychological properties—including response styles, attention patterns, and treatment effects calibrated to your specifications.

---

## Who Benefits from This Tool?

### Academic Researchers

- **Pre-registration preparation**: Generate synthetic data to develop and test your analysis scripts before collecting real data
- **Power analysis**: Validate sample size calculations with data that matches your expected effect sizes
- **Grant applications**: Include preliminary analyses in proposals using realistic simulated data
- **Pilot testing**: Test your experimental design logic and identify issues before investing in data collection

### Graduate Students and Postdocs

- **Methods training**: Learn data analysis techniques with realistic datasets that have known properties
- **Thesis preparation**: Develop analysis pipelines while awaiting IRB approval
- **Replication planning**: Generate data matching published effect sizes to plan replication studies

### Course Instructors

- **Teaching materials**: Create datasets with specific statistical properties for classroom exercises
- **Exam development**: Generate realistic data for assessment questions
- **Student projects**: Provide synthetic data for students who cannot access real participant data

### Industry Researchers

- **A/B test planning**: Simulate experiment outcomes before deploying to production
- **UX research**: Generate survey response data for prototyping analysis dashboards
- **Stakeholder presentations**: Demonstrate analysis approaches with representative data

---

## How It Works

The tool offers **two input pathways** to accommodate different stages of the research process:

### Pathway A: Upload Your Qualtrics Survey (.qsf)

If you already have a Qualtrics survey built, simply upload the .qsf file. The system automatically extracts:

- Experimental conditions from your BlockRandomizer or branch logic
- Dependent variables (Likert scales, sliders, matrices)
- Open-ended questions and their visibility rules
- Survey flow logic determining which questions each participant sees

### Pathway B: Describe Your Experiment (Conversational Builder)

If you haven't built your survey yet — or prefer a faster setup — you can **describe your experiment in plain language**. The conversational builder guides you through:

1. **Study title and description**: What is your study about?
2. **Conditions**: Describe your experimental design in natural language. The system automatically detects:
   - Simple designs: `"Treatment vs Control"`
   - Factorial designs: `"3 (Annotation: AI vs Human vs None) × 2 (Product: Hedonic vs Utilitarian)"`
   - Numbered lists: `"1. Low dose, 2. Medium dose, 3. High dose"`
   - Any N×M crossed design with automatic condition generation
3. **Scales and DVs**: Describe your measures in paragraph or list format. The parser recognizes:
   - Standard scale specifications: `"Trust scale (4 items, 1-7)"`
   - Detailed academic format: `"Perceived Quality (PQ): 3 items (7-point Likert; 1=low, 7=high)"`
   - Known validated instruments: `"BFI-10"`, `"PANAS"`, `"GAD-7"`, `"PHQ-9"` and others. Three tables recognize these: `KNOWN_SCALES` (`utils/survey_builder.py`, 84 entries) supplies expected structure on this builder path — it is where `GAD-7` (7 items, 0-3), `PHQ-9` (9 items, 0-3) and a `BFI-10`-specific entry (10 items, 1-5) live; `WELL_KNOWN_SCALES` (`utils/qsf_preview.py`, 10 entries) is present for the QSF path but its detector `_detect_well_known_scale()` has no callers, so the QSF path recognizes no validated instruments today; and the construct map supplies published norms for calibration. A name may hit one, two or all three
   - Numeric measures: `"Willingness to Pay (WTP): 1 item (open-ended numeric)"`
   - Binary measures: `"Manipulation check (Yes/No)"`
4. **Open-ended questions**: Simply list your qualitative questions
5. **Research domain**: Pick one of the 16 broad domains from the dropdown — the only domain widget in the builder is a selectbox (`app.py:5541`), and the domain auto-detected from your description is prepended to the list when it is not one of the 16; the 273-domain detector then refines it from your question wording
6. **Sample size and effect sizes**: Configure your simulation parameters

The builder outputs the same structured design specification used by the QSF pathway, ensuring identical simulation quality regardless of input method.

### Design Review

Regardless of input method, you review and adjust the detected design:

- Verify conditions, scales, and open-ended questions
- Edit names, scale ranges, and item counts inline
- **Scale type auto-correction**: Single-item DVs are automatically identified (not mislabeled as "Likert Scale"); multi-item scales are properly categorized by type (matrix, numbered items, single item). Scale min/max values are propagated from QSF detection.
- Add or remove measures as needed
- **Custom demographic variables**: Add demographic questions beyond the defaults (Age, Gender). Quick-add templates include Political Orientation, Education Level, Ethnicity, Household Income, Employment Status, Religion, and Party Identification. Each demographic is fully customizable:
  - **Categorical**: Edit options and their probability weights
- **Ordinal**: Ordered categories with weights (e.g. Political Orientation, "Very Liberal" through "Very Conservative")
  - **Ordinal**: Set ordered categories with center-weighted distribution
  - **Numeric**: Configure mean, standard deviation, and min/max bounds (e.g., household income)
  - **Distribution preview**: See the expected distribution before generating
- Customize persona weights for domain-specific response patterns
- Set expected effect sizes (Cohen's d) with visual condition selectors

### Generate and Download

The system produces a publication-ready CSV file containing:

- Participant IDs and condition assignments
- Likert scale responses with realistic distributions
- Unique open-ended text responses
- Demographics (including custom variables) and metadata
- Quality metrics and validation flags
- A study summary report (`User_Study_Summary.md` / `.html`) with persona breakdowns, trait profiles by condition, and a configured-vs-observed effect size table

---

## The Science Behind the Simulation

### Persona-Based Response Generation

Rather than generating random responses, the system assigns each simulated participant a "persona" based on survey methodology research. **These personas have been trained on hundreds of scientific insights from decades of research across the social and behavioral sciences**, including:

- **Behavioral Economics**: Trust, cooperation, altruism, fairness, reciprocity, risk preferences, loss aversion, framing effects, sunk cost, anchoring
- **Social Psychology**: Social identity, group dynamics, conformity, prosocial behavior, intergroup relations, attitudes, persuasion, social comparison
- **Cognitive Psychology**: Decision-making heuristics, cognitive biases, memory, attention, reasoning processes, construal level
- **Organizational Behavior**: Leadership, motivation, job satisfaction, team dynamics, workplace attitudes, power dynamics
- **Political Psychology**: Polarization, partisanship, civic engagement, media effects, political trust, sacred values
- **Consumer Behavior**: Brand perception, purchase intent, product evaluation, advertising effectiveness, choice architecture
- **Moral Psychology**: Ethical judgment, values, moral emotions, fairness perceptions, moral cleansing, sacred value tradeoffs
- **Health Psychology**: Medical decisions, wellbeing, health behaviors, treatment preferences, gratitude interventions
- **Narrative & Communication**: Narrative transportation, story persuasion, source credibility, elaboration likelihood, inoculation theory
- **Digital & Technology**: Attention economy, phone distraction, social media comparison, digital wellbeing, algorithm aversion
- **Positive Psychology**: Gratitude interventions, savoring, best possible self, acts of kindness, growth mindset

This rich scientific foundation enables each persona to generate responses that align with documented human response patterns across these diverse research domains.

**The personas reflect actual patterns observed in human respondents.** The six response-style personas below carry the following sampling weights (relative weights, normalized at assignment, not a partition of 100%); 72 further domain-specific personas across 24 categories are layered on top when the study domain matches:

**Engaged Responders (35%)**: High attention, thoughtful responses, full scale use. Based on Krosnick's (1991) "optimizers" who invest cognitive effort in providing accurate answers. These participants draw on genuine reflection about the topic at hand.

**Satisficers (22%)**: Lower effort responses, tendency toward agreement, restricted scale range. Krosnick's research documented this common response strategy where participants provide acceptable rather than optimal answers.

**Socially Desirable Responders (12%)**: Inflate socially favorable answers and suppress unfavorable ones. Paulhus (2002) distinguishes impression management from self-deception; the engine applies the adjustment proportionally to item sensitivity.

**Extreme Responders (10%)**: Consistent use of scale endpoints. Greenleaf's (1992) work identified this stable response style that varies across individuals.

**Acquiescent Responders (8%)**: Strong agreement bias regardless of item content. Billiet & McClendon's (2000) studies documented this tendency to agree with statements.

**Careless Responders (5%)**: Low attention, random patterns. Meade & Craig's (2012) research characterized these inattentive participants common in online samples.

### Effect Size Calibration

Treatment effects are calibrated using Cohen's d, the standard measure in behavioral science:

```
d = (Treatment Mean - Control Mean) / Pooled Standard Deviation
```

When you specify d = 0.5, the system shifts response distributions between
conditions in the configured direction. The configured *d* is a **target, not an
achieved value** — see the effect-size note under Validation below. This works
through:

1. **Semantic parsing** of condition names to determine effect direction
2. **Graduated adjustments** applied at the individual response level

Achieved effects are written to `Metadata.json` under `effect_sizes_observed`
for you to check. No automatic target check runs during generation:
`_validate_effect_sizes()` exists in the engine, with a 0.15 tolerance, but has
no callers.

### Scale Reliability Modeling

Multi-item scales exhibit realistic internal consistency (Cronbach's alpha) through a factor model approach:

```
Response = lambda * Common_Factor + sqrt(1 - lambda^2) * Unique_Error
```

Where lambda (factor loading) is derived from the target reliability. Items measuring the same construct share common variance while retaining item-specific variation, producing alpha values from about 0.75 up to the mid-0.90s. Correlation injection is one-sided — it raises alpha toward the target (default 0.75) and never lowers it (`enhanced_simulation_engine.py:11877`, whose comment notes items "often exceed the target") — so a fair share of scales land above 0.90.

### Response Style Modeling

The system models several well-documented response styles:

**Acquiescence Bias**: Tendency to agree with statements, producing higher means on positively-worded items. Modeled as a per-participant offset.

**Extreme Response Style**: Tendency to use scale endpoints. Higher extremity = more responses at 1 or 7 (on 7-point scales).

**Social Desirability**: Inflation of socially favorable responses. Applied proportionally based on item content, with **domain-sensitive intensity**: highly sensitive topics (prejudice, dishonesty) receive 1.5× the social desirability adjustment, while factual/behavioral reports receive only 0.5×. Based on Nederhof (1985) and Paulhus (2002).

**Midpoint Avoidance** *(table present, not yet wired)*: Cultural variation in willingness to use the neutral midpoint. East Asian samples typically show lower midpoint avoidance than Western samples. The `CULTURAL_RESPONSE_STYLES` table encodes this, but `_apply_cultural_response_style()` is not called during generation and the `midpoint_preference` trait is not read by the engine — tracked as Tier C work in `docs/COVERAGE_ROADMAP.md`.

### Persona-Demographic Coupling

Custom demographic variables are not assigned randomly—they are **coupled to each participant's behavioral persona** to create realistic correlations between demographic characteristics and response patterns, as observed in real survey data.

The coupling algorithm uses a **swap-sort** approach that preserves exact marginal distributions while creating realistic associations:

1. **Affinity lookup**: A mapping defines which persona types tend toward which demographic values (e.g., "Extreme Responder" personas are more likely to hold strong political orientations; "Engaged" personas correlate with higher education levels)
2. **Trait-based soft coupling**: Beyond explicit persona-demographic mappings, participant traits like extremity, attention, and consistency create additional correlations (e.g., high-extremity participants skew toward extreme political positions)
3. **Distribution-preserving swaps**: Rather than generating new values, the algorithm only *swaps* existing demographic assignments between participants when a swap improves the persona-demographic fit—ensuring the overall distribution exactly matches the user's specified weights

This means a simulated dataset where 35% of participants are "Engaged Responders" will show a natural correlation between engagement and education level, mirroring real-world patterns without distorting the researcher's target demographic distributions.

### Behavioral Coherence (Rating–Text Consistency)

A key advancement is the **behavioral coherence pipeline** that ensures each simulated participant's numeric ratings and open-text responses tell a coherent story. The same person who rates trust at 6-7/7 writes positively about trust; a participant who straight-lines 4s across all items writes brief, disengaged text.

This is enforced through:
1. **Behavioral profiling**: Each participant's numeric pattern (mean, variability, straight-lining) is computed before text generation
2. **Profile-guided text**: The behavioral profile flows to all text generators, constraining tone, length, and engagement
3. **Post-generation validation**: Text responses are checked against numeric patterns and corrected if mismatched
4. **Cross-item consistency tracking**: Participants who fail reverse-coded items are more likely to fail subsequent ones, matching Woods (2006) findings that reversal failure is trait-like within session

### Reverse-Coded Item Modeling

Reverse-coded items receive sophisticated handling that goes beyond simple scale inversion:
- **Engagement-dependent accuracy**: Engaged respondents correctly reverse ~95% of the time; careless respondents only ~30-50%
- **Acquiescence interaction**: Even respondents who correctly reverse show partial acquiescence pull (~0.5 point, Weijters et al. 2010)
- **Cross-item failure consistency**: A participant who fails one reverse item is more likely to fail the next (trait-like within session)

### Response Validation Layer *(implemented, not yet called during generation)*

`_validate_participant_responses()` checks generated responses against expected
patterns per persona type, but `generate()` does not invoke it, so none of these
checks currently run:
- **Longstring detection**: flags unrealistic straight-lining for engaged personas
- **IRV checks**: response variability against persona engagement level
- **Endpoint utilization**: whether extreme-response personas really use endpoints

The validation that *does* run post-generation is `HBSValidator` — completion-time
plausibility, open-ended uniqueness and length, straight-lining prevalence and
rating–text coherence. Wiring this layer in is tracked in `docs/COVERAGE_ROADMAP.md`.

### Survey Flow Logic

The system respects your experimental design by tracking which questions each participant would actually see:

- **Block-level conditions**: Participants only receive responses for their assigned condition's blocks
- **Display logic**: Questions with condition-specific visibility rules are handled appropriately
- **Factorial designs**: Crossed conditions (e.g., AI x Hedonic, AI x Utilitarian) are properly parsed

---

## Open-Ended Response Generation

Open-ended text responses are generated by a three-level cascade: each level is tried in turn and the next is reached only if the one above it yields nothing usable.

### Tier 1: AI-Powered Generation (Primary)

When available, responses are generated by a large language model (LLM) that receives the full experimental context — study description, condition assignment, and participant persona — and produces natural, question-specific text that mirrors real survey responses.

**Zero-configuration AI**: The tool ships with built-in API keys for six free LLM providers, so AI-powered responses work out of the box with no setup required. Nine provider entries are tried in order (some providers contribute more than one model line, so the retirement of a single model degrades the chain instead of breaking it):

| Order | Provider | Model |
|-------|----------|-------|
| 1 | **Google AI** | `gemini-3.1-flash-lite` |
| 2 | **Google AI** | `gemini-2.5-flash` |
| 3 | **Google AI** | `gemini-2.5-flash-lite` |
| 4 | **Groq** | `openai/gpt-oss-120b` |
| 5 | **Groq** | `qwen/qwen3.6-27b` |
| 6 | **Cerebras** | `gpt-oss-120b` |
| 7 | **SambaNova** | `Meta-Llama-3.3-70B-Instruct` |
| 8 | **Mistral AI** | `mistral-small-latest` |
| 9 | **OpenRouter** | `mistralai/mistral-small-3.1-24b-instruct:free` |

If one provider reaches its rate limit or errors, the system automatically tries the next.

A key you supply is **appended after** the built-ins, not put ahead of them — the tool deliberately spends its own free capacity first and reaches your key only once the built-ins are exhausted (`llm_response_generator.py:2508`). Keys supplied through environment variables land at provider-specific positions in the chain. Model assignments and ordering change as free tiers are retired; `_builtin_providers` in `utils/llm_response_generator.py` is authoritative.

**Sample-size cap**: built-in free-tier keys are shared across all users, so the **Built-in AI** method generates LLM open-ended text for the first `MAX_FREE_LLM_N` = 100 participants only and falls back to the compositional template engine for the remainder. The app warns before generating and reports the resulting split. Supplying your own key removes the cap.

**Key features:**

1. **Batch generation**: 20 persona-guided responses are generated per API call, each tailored to a different participant profile (varying in verbosity, formality, engagement level, and sentiment)
2. **Draw-with-replacement pooling**: A pool of LLM-generated base responses is pre-built for each question × condition × sentiment bucket; individual participants draw from this pool with deep persona-driven variation applied, ensuring no two responses are identical even when they share a common base
3. **8-layer deep variation**: Each drawn response passes through word-level micro-variation, sentence restructuring, verbosity control, formality adjustment, engagement modulation, typo injection, synonym substitution and punctuation variation (Layers 0-7, several with sub-layers) — producing unique output for every participant
4. **Smart pool scaling**: Pool size automatically adapts to sample size. Per sentiment bucket the target is `sqrt(participants_per_bucket) * 2.4 + 8`, clamped to [18, 60], where `participants_per_bucket = sample_size / (n_conditions × n_sentiments)` — balancing API efficiency against response diversity
5. **9-entry failover chain**: see the provider table above; a user-supplied key is appended after all built-ins, so it is used only once the built-in free capacity is spent

### Tier 2: Adaptive Behavioral Engine 3.0 (selected explicitly)

No generation method is pre-selected; ABE 3.0 runs when you pick its tile. It is *not* the within-run LLM fallback: choosing Built-in AI or Your API Key forces `_use_abe_v2 = False` (`app.py:12836`, `:13022`), so when a run exceeds the 100-participant cap, exhausts the open-ended budget, or gets an empty LLM response, the text comes from the compositional template engine (`ComprehensiveResponseGenerator`, `enhanced_simulation_engine.py:3427`). ABE 3.0 takes over mid-run only if you re-select it in the recovery prompt. ABE 3.0 itself is a narrative-enhanced behavioral engine that integrates census-weighted demographics, stylometric voice fingerprinting, and 5 individual-level consistency improvements into the domain template engine. Building on the compositional architecture introduced in v1.2.3.1, ABE 3.0 adds dedicated narrative intent builders (Brotherton 2013, Pennebaker 1997, Green & Brock 2000) and produces highly varied, topic-grounded responses:

1. **Intent-driven composition**: Each response is assembled from opener + intent-matched core + domain-enriched elaboration + coda. Question intent is classified into 16 categories (opinion, explanation, description, emotional reaction, evaluation, prediction, causal explanation, decision explanation, creative belief, personal disclosure, creative narrative, personal story, hypothetical, recommendation, comparison, recall) and templates are selected accordingly
2. **40 domain vocabulary branches** in `_get_domain_vocabulary()` (`response_library.py:9956`): specialized terminology for clinical/mental health, sports, legal, food, developmental, personality, cognitive, neuroscience, financial and cross-cultural work among them, matched on any of several hundred domain cues, so responses use field-appropriate language
3. **Rich question-text mining**: 32 action verb patterns, 23 object/target pattern groups, and 19 key phrase patterns extract the actual topic from the question text for template insertion
4. **Domain-gated condition modifiers**: Condition-specific personalizations (e.g., "As someone who leans progressive") are only applied when the domain matches — political modifiers only fire for political studies, health modifiers only for health studies
5. **Behavioral coherence**: Templates are post-processed to match the participant's numeric response pattern — straight-liners get truncated text, extreme raters get intensified language, high social desirability personas get qualifying hedges
6. **Careless responses are intent-keyed, not generic.** `_make_careless()` (`response_library.py:14235`) holds 21 templates across 7 intent banks, with 6 more in `response_library`'s own `careless_templates` and 6 in `persona_library`'s fallback — 33 across 9 banks in total. The point is the keying rather than the volume: even a low-effort response references the actual topic ("trump is ok i guess") rather than generic off-topic text ("fine"), and the topic is substituted in at generation time
7. **Context-awareness**: Responses reference the experimental manipulation when appropriate
8. **Condition-specificity**: Only participants who would see a question receive a response

### LLM Exhaustion Recovery

When free AI providers reach their rate limits during generation, the system provides a **transparent, user-controlled recovery flow** rather than silently switching methods:

1. **Immediate notification**: The user sees a clear message showing how many participants were completed with AI and how many remain
2. **Two simultaneous recovery options** presented as equal choices:
   - **Use Your Own API Key**: Enter a personal (free) key from Groq, Google AI, Cerebras, etc. for dedicated capacity
   - **Adaptive Behavioral Engine 3.0**: Complete remaining participants with the full ABE 3.0 engine (census-weighted demographics, stylometric fingerprinting, 5 consistency layers) — runs entirely offline, no API needed
3. **Data preservation**: Responses already generated by AI are preserved; only remaining participants use the selected fallback method
4. **Source transparency**: The final dataset includes metadata showing which responses were AI-generated vs. ABE 3.0-generated

This ensures the researcher always knows—and controls—the data source for their simulated dataset.

### Self-Healing Error Pipeline

The system includes an **automatic error logging and self-healing pipeline** that captures generation failures and facilitates rapid fixes:

1. **Automatic error capture**: Every generation error is logged with full context—traceback, session state snapshot, generation method, timing, and phase
2. **Error fingerprinting**: Errors are deduplicated using a SHA-256 fingerprint of the error type + last traceback frame, so recurring issues are counted rather than duplicated
3. **Structured reporting**: On software updates (version changes), the system automatically generates a structured `PENDING_FIXES.md` report listing all unresolved errors with their full context
4. **Privacy-preserving**: Sensitive keys (API keys, passwords, binary content) are automatically redacted from error logs
5. **Fix tracking**: Each error carries a status (pending → acknowledged → fixed) with the version that resolved it

This pipeline ensures that user-encountered errors are never lost and are systematically addressed in subsequent updates.

### Example Responses

For a question "What did you think about the AI recommendations?":

**Treatment participant** (saw AI):
> "I found the AI-generated recommendations to be helpful. The system seemed to capture my preferences well. I appreciated how it addressed my needs effectively."

**Control participant** (no AI):
> *(Empty - this participant didn't see this question)*

### Tier 3: Last-resort template generator

If both levels above return nothing — an exhausted LLM chain and a template miss
for an unusual question type — `TextResponseGenerator`
(`utils/persona_library.py`) emits a short, topic-grounded response from the
participant's persona and the extracted question topic. It is wrapped in its own
try/except so a failure here leaves a plausible answer rather than a blank cell.
This level is reached rarely; when it is, the response is still about the
question's topic, never generic filler.

---

## Research Domain Coverage

The response generation system has been trained on **hundreds of scientific insights** drawn from decades of research across the social and behavioral sciences. This extensive knowledge base enables the tool to generate contextually appropriate responses for virtually any research topic you might study.

### Major Research Fields

**273 research domains** are keyword-detectable, via 3,452 keyword patterns.
**189** of them are grouped into the 23 categories below (191 memberships — two
domains, `social_media` and `algorithmic_fairness`, are cross-listed); the other
84 are detectable but ungrouped. **104** of the detectable domains carry a reachable
open-ended template set, 68 of them inside these categories. (`DOMAIN_TEMPLATES`
holds 116 keys, but template lookup goes through `domain.value`
(`response_library.py:8622`), so the 10 keys that are not `StudyDomain` values —
`artificial_intelligence`, `climate_change`, `ethical_dilemma`, `forgiveness`,
`gratitude_experience`, `gratitude_intervention`, `moral_cleansing`,
`narrative_transportation`, `nostalgia`, `sleep_quality` — can never be
selected. Two more, `general` and `survey_feedback`, are reachable as fallbacks
but are not keyword-detectable domains.)

The table is generated from `DOMAIN_CATEGORIES` in `utils/response_library.py`,
which is authoritative. "Example domains" lists the first few members of each
category verbatim, not a paraphrase.

| Category | Domains | Example domains |
|----------|---------|-----------------|
| **Organizational Behavior** | 16 | organizational, workplace, leadership, teamwork, motivation, job satisfaction |
| **Social Psychology** | 15 | social psychology, intergroup, identity, norms, conformity, prosocial |
| **Environmental** | 14 | environmental, sustainability, climate attitudes, pro environmental, green consumption, conservation |
| **Behavioral Economics** | 12 | behavioral economics, dictator game, public goods, trust game, ultimatum game, prisoners dilemma |
| **Consumer & Marketing** | 10 | consumer, brand, advertising, product evaluation, purchase intent, brand loyalty |
| **Health Psychology** | 10 | health, medical decision, wellbeing, health behavior, mental health, vaccination |
| **Political Science** | 10 | political, polarization, partisanship, voting, media, policy attitudes |
| **Technology & AI** | 10 | technology, ai attitudes, privacy, automation, algorithm aversion, technology adoption |
| **Ethics & Moral Psychology** | 9 | ethics, moral judgment, moral dilemma, ethical leadership, corporate ethics, research ethics |
| **Decision Science** | 8 | decision science, choice architecture, nudge, default effects, information overload, regret |
| **Education** | 8 | education, learning, academic motivation, teaching effectiveness, online learning, educational technology |
| **AI Alignment & Ethics** | 7 | ai alignment, ai ethics, ai safety, machine values, ai governance, ai transparency |
| **Social Media Research** | 7 | social media, social media use, online identity, digital communication, influencer marketing, online communities |
| **Clinical Psychology** | 6 | clinical, anxiety, depression, coping, therapy attitudes, stress |
| **Financial Psychology** | 6 | financial psychology, financial literacy, investment behavior, debt attitudes, retirement planning, financial stress |
| **Gaming & Entertainment** | 6 | gaming psychology, esports, gambling, entertainment media, streaming behavior, virtual reality |
| **Health Disparities** | 6 | health disparities, healthcare access, health equity, social determinants, health literacy, medical mistrust |
| **Personality Psychology** | 6 | personality, big five, narcissism, dark triad, trait assessment, self concept |
| **Digital Society** | 5 | digital divide, online polarization, algorithmic fairness, data privacy, digital literacy |
| **Future of Work** | 5 | automation anxiety, gig economy, skills obsolescence, universal basic income, human machine collaboration |
| **Innovation & Creativity** | 5 | innovation, creativity, entrepreneurship, idea generation, creative process |
| **Risk & Safety** | 5 | risk perception, safety attitudes, hazard perception, disaster preparedness, risk communication |
| **Trust & Credibility** | 5 | institutional trust, expert credibility, source credibility, science trust, media trust |

### Topic-Specific Response Generation

For each research domain, the system maintains specialized knowledge about:

1. **Key constructs and terminology** used in that field
2. **Typical response patterns** observed in empirical studies
3. **Common participant concerns and attitudes** documented in the literature
4. **Domain-specific language and phrasing** that real participants use

This means when you study cooperation in public goods games, the system generates responses that reference concepts like "contribution," "free-riding," "collective benefit," and "reciprocity"—just as real participants would. Similarly, for AI trust research, responses naturally mention "algorithmic recommendations," "automation," "reliability," and "transparency."

### Scientific Foundation

The domain knowledge is built on insights from:

- **Classic experiments**: Kahneman & Tversky's prospect theory work, Milgram's obedience studies, Asch's conformity experiments
- **Meta-analyses**: Aggregated findings from hundreds of studies in each domain
- **Replication projects**: Many Labs, Psychological Science Accelerator, and other large-scale replication efforts
- **Contemporary research**: Recent publications in top journals (JPSP, Psychological Science, JEP:G, Management Science, etc.)

---

## Data Quality Features

### Attention Check Simulation

Configurable proportion of participants fail attention checks, matching real-world rates. Failed attention checks are flagged for potential exclusion.

### Careless Response Detection

The generated dataset flags (and can recommend excluding) simulated careless
responses. These columns ship: `Max_Straight_Line`, `Flag_StraightLine`,
`Flag_Speed`, `Flag_Attention`, `Exclude_Recommended`.

- **Straight-lining**: same response repeated across items
- **Response time anomalies**: unrealistically fast completion

Alternating patterns (1-7-1-7) **are** detected, in the live exclusion path
(`enhanced_simulation_engine.py:11161-11167`), and folded into the shipped
`Max_Straight_Line` column by taking the worse of the two streaks (`:11169`),
which in turn drives `Flag_StraightLine`. Midpoint overuse is implemented only
in `_detect_careless_patterns()` (`:1512`), which has no callers, so it never
reaches an output column.

### Validation Metrics

Generated datasets include quality metrics:

- Achieved effect sizes — Cohen's *d*, both group means and both Ns (no confidence intervals). 95% CIs appear in the emailed instructor report, as per-condition means, and in the password-gated Analytics Dashboard, which also plots Cohen's *d* with 95% CIs (`app.py:2753`, `:2958`) — though that dashboard returns early unless `plotly` is importable, and `plotly` is not in `requirements.txt`
- Condition balance verification
- Missing data rates
- Response distribution statistics

---

## Technical Specifications

### Input Requirements

- **Option A**: Qualtrics Survey Format (.qsf) file — for researchers with existing surveys
- **Option B**: Plain-language experiment description — for researchers at the design stage
- Internet connection for web interface

### Output Format

- **CSV file** compatible with R, SPSS, Stata, Python
- **Study summary** (Markdown + HTML) with persona breakdowns, trait profiles by condition and configured-vs-observed effect sizes. The fuller statistical report — inferential tests, Condition × Gender chi-squared, charts — is generated separately and emailed to the instructor; it is not part of the download
- **Metadata** JSON with simulation parameters
- **Analysis scripts** auto-generated for R, Python, Julia, SPSS, and Stata

### Supported Question Types

| Type | Example | Input Methods |
|------|---------|---------------|
| Likert Scales | 7-point agreement scales | QSF, Builder |
| Matrix Tables | Multi-item scales with shared options | QSF, Builder |
| Sliders | Visual analog scales (0-100) | QSF, Builder |
| Multiple Choice | Single selection questions | QSF |
| Text Entry | Open-ended responses | QSF, Builder |
| Numeric Input | Willingness to pay, quantities | QSF, Builder |
| Binary | Yes/No, True/False | QSF, Builder |
| Constant Sum | Budget allocation across items | QSF |
| Rank Order | Preference rankings | QSF |

**Detected but not generated.** The parser recognizes hot-spot/heatmap questions,
but no data is produced for them — `_generate_heatmap_response` exists in the
engine and is never called. Best-worst and paired-comparison DV types likewise
have parser paths with no generation behind them; none of these occur in the
example QSF corpus (see `docs/COVERAGE_ROADMAP.md`). Single-choice items are not
a separate DV type — they are grouped into `likert`/`single_item` and generate
normally.

**Not supported.** Semantic differential (bipolar adjective) scales are neither
detected nor generated; scale-type detection expansion is on the roadmap in
`CLAUDE.md`.

### Supported Experimental Designs

| Design | Example | Detection |
|--------|---------|-----------|
| Between-subjects | Treatment vs Control | Automatic |
| Factorial (2×2) | AI × Product Type | Automatic (NxM) |
| Factorial (3×2) | Annotation × Product | Automatic |
| Factorial (N×M) | Any crossed design | Automatic |
| Multi-level | Low / Medium / High | Automatic |
| Mixed designs | Between + within factors | QSF path |
| Complex branching | Skip logic, visibility rules | QSF path |

The conversational builder automatically detects factorial designs from natural language input and generates all crossed conditions. For example, entering `"3 (Source: AI vs Human vs None) × 2 (Product: Hedonic vs Utilitarian)"` produces 6 conditions with proper × notation.

---

## Frequently Asked Questions

### Is this generating "fake data"?

No—this is **synthetic data generation**, a legitimate research methodology. The tool produces data with known statistical properties for specific purposes:

- Testing analysis code before real data collection
- Teaching data analysis with realistic datasets
- Power analysis and sample size planning
- Developing preprocessing pipelines

Synthetic data should never be misrepresented as real participant data in publications.

### How realistic are the responses?

The responses exhibit statistical properties matching published research on human survey behavior:

- Mean responses around 4.0-5.2 on 7-point scales before domain calibration (documented positive response bias); realized DV means span roughly 3.5-5.5 once construct norms apply, with clinical DVs centering lower and satisfaction DVs higher
- Standard deviations of 1.2-1.8 (typical for Likert data)
- Cronbach's alphas from about 0.75 up to the mid-0.90s for multi-item scales (raised toward the target, never lowered)
- Effect sizes: a configured Cohen's *d* is recovered to within roughly -8% to +12% across scale widths and item counts, and a null effect stays null (`tests/test_effect_size_recovery.py`). Effects inferred from condition wording when no *d* is configured are literature-sized and directional rather than fitted to a target. Verify the achieved effect in `Metadata.json` (`effect_sizes_observed`) before relying on the magnitude.

### Can I use this for any survey?

The tool is optimized for behavioral science experiments with:

- Clear experimental conditions (treatment/control)
- Likert or similar scale DVs
- Defined effect size expectations

It may not be suitable for purely exploratory surveys or complex longitudinal designs.

### Do I need a Qualtrics survey file?

No. You can describe your experiment in plain language using the **Conversational Builder**. The system parses your natural language description to extract conditions, scales, and open-ended questions. This is especially useful for:

- Early-stage study design before building the actual survey
- Quick pilot data generation
- Teaching contexts where students describe hypothetical experiments
- Power analysis before IRB submission

### What factorial designs are supported?

The system supports any N×M crossed factorial design:

- **2×2**: `"AI (present, absent) × Trust (high, low)"`
- **3×2**: `"3 (Source: AI vs Human vs None) × 2 (Product: Hedonic vs Utilitarian)"`
- **2×2×2**: Three-factor designs with automatic crossing
- **Custom**: Any combination using "vs", commas, or numbered lists

The system automatically generates all crossed conditions and displays them in a design table.

### How are effect directions determined?

The system parses condition names to determine which should produce higher/lower responses:

- **Positive indicators**: "high", "treatment", "reward", "positive"
- **Negative indicators**: "low", "control", "loss", "negative"

For complex designs, effect direction can be specified manually in the design review.

### What does the instructor report contain?

The comprehensive HTML report includes:

- **Study overview**: Design, conditions, factors, scales, sample size
- **Research context**: Domain, input method, participant characteristics, persona domains
- **Statistical analysis**: Per-DV descriptive statistics, ANOVA/t-tests, effect sizes, visualizations
- **Persona analysis**: Distribution table, per-condition persona counts, personality trait profiles
- **Effect size verification**: Configured vs. observed effects with Cohen's d interpretation
- **Data quality**: Exclusion breakdown (speed, attention, straight-lining), validation corrections
- **Categorical analysis**: Condition × Gender cross-tabulation with chi-squared test
- **Executive summary**: automatically generated synthesis of key findings — rule-based, computed from the scale statistics; no LLM is involved
- **Scientific references**: Full citations for the methodological foundations

---

## Getting Started

### Option A: With a Qualtrics Survey

1. **Export your Qualtrics survey** as a .qsf file (Survey > Tools > Import/Export > Export Survey)
2. **Access the tool** at the provided URL
3. **Upload your .qsf file** and review the detected structure
4. **Configure parameters** for your specific needs
5. **Generate and download** your synthetic dataset

### Option B: With the Conversational Builder

1. **Access the tool** and select "Describe my study" in the Study Input tab
2. **Enter your study title** and a brief description
3. **Describe your conditions** (the system detects simple, factorial, and multi-level designs)
4. **Describe your scales/DVs** (supports standard abbreviations, paragraph format, and detailed specs)
5. **Add open-ended questions** if applicable
6. **Review and adjust** the auto-detected design in the Design tab
7. **Generate and download** your synthetic dataset

---

## Scientific References

The simulation algorithms are grounded in established survey methodology research:

### Core Survey Methodology
1. **Cohen, J. (1988)**. Statistical power analysis for the behavioral sciences. *Effect size conventions and calculations.*
2. **Krosnick, J. A. (1991)**. Response strategies for coping with the cognitive demands of attitude measures in surveys. *Applied Cognitive Psychology, 5*, 213-236.
3. **Greenleaf, E. A. (1992)**. Measuring extreme response style. *Public Opinion Quarterly, 56*, 328-351.
4. **Billiet, J. B., & McClendon, M. J. (2000)**. Modeling acquiescence in measurement models for two balanced sets of items. *Structural Equation Modeling, 7*, 608-628.
5. **Meade, A. W., & Craig, S. B. (2012)**. Identifying careless responses in survey data. *Psychological Methods, 17*, 437-455.
6. **Paulhus, D. L. (2002)**. Socially desirable responding: The evolution of a construct. In H. I. Braun, D. N. Jackson & D. E. Wiley (Eds.), *The role of constructs in psychological and educational measurement* (pp. 49-69). Erlbaum.
7. **Nederhof, A. J. (1985)**. Methods of coping with social desirability bias. *European Journal of Social Psychology, 15*, 263-280.
8. **Woods, C. M. (2006)**. Careless responding to reverse-worded items: Implications for confirmatory factor analysis. *Journal of Psychopathology and Behavioral Assessment, 28*(3), 186-191.
9. **Weijters, B., et al. (2010)**. The effect of rating scale format on response styles. *International Journal of Research in Marketing, 27*, 236-247.

### Behavioral Economics & Game Theory
10. **Engel, C. (2011)**. Dictator games: A meta study. *Experimental Economics, 14*, 583-610.
11. **Dimant, E. (2024)**. Partisan intergroup discrimination in economic games.
12. **Iyengar, S., & Westwood, S. J. (2015)**. Fear and loathing across party lines. *American Journal of Political Science, 59*, 690-707.
13. **Kahneman, D., & Tversky, A. (1979)**. Prospect theory. *Econometrica, 47*, 263-291.

### Paradigms & Phenomena
14. **Green, M. C., & Brock, T. C. (2000)**. The role of transportation in the persuasiveness of public narratives. *JPSP, 79*, 701-721.
15. **Festinger, L. (1954)**. A theory of social comparison processes. *Human Relations, 7*, 117-140.
16. **Emmons, R. A., & McCullough, M. E. (2003)**. Counting blessings versus burdens. *JPSP, 84*, 377-389.
17. **Zhong, C. B., & Liljenquist, K. (2006)**. Washing away your sins: Threatened morality and physical cleansing. *Science, 313*, 1451-1452.
18. **Ward, A. F., et al. (2017)**. Brain drain: The mere presence of one's own smartphone reduces available cognitive capacity. *JACR, 2*, 140-154.
19. **Tetlock, P. E., et al. (2000)**. The psychology of the unthinkable: Taboo trade-offs, forbidden base rates, and heretical counterfactuals. *JPSP, 78*, 853-870.
20. **Podsakoff, P. M., et al. (2003)**. Common method biases in behavioral research. *JAP, 88*, 879-903.

---

## Citation

If you use this tool in your research or teaching, please acknowledge:

> Dimant, E. (2026). Behavioral Experiment Simulation Tool (Version 1.2.8.9) [Computer software].

---

## Changelog Highlights

### Version 1.2.8.9 (and earlier 1.2.8.x)
- **Custom demographic variables**: Full flexibility to add and customize demographic questions (Political Orientation, Education, Ethnicity, Income, Employment, Religion, Party ID) with editable options, weights, and distributions
- **Persona-demographic coupling**: Swap-sort algorithm creates realistic correlations between persona types and demographic values while preserving exact marginal distributions
- **Scale type auto-correction**: Single-item DVs properly identified; scale type/min/max propagated from QSF detection
- **LLM exhaustion recovery**: Transparent recovery UI when free AI providers are exhausted—user always controls the data source
- **Self-healing error pipeline**: Automatic error logging with full context capture, fingerprinting, deduplication, and structured fix reporting on software updates
- **Stale-phase recovery**: Unprotected code paths in the generation pipeline now wrapped in comprehensive error handling

### Version 1.0.4.9
- **5 new research paradigm domains**: Narrative transportation, social comparison, gratitude interventions, moral cleansing/sacred values, digital attention economy
- **6 new domain-specific personas**: Narrative Thinker, Social Comparer, Grateful Optimist, Moral Absolutist, Digital Native, Financial Deliberator
- **Enhanced social desirability**: Domain-sensitive construct detection for moral identity, gratitude, digital habits, social comparison
- **Cross-item reverse-failure tracking**: Participants who fail one reverse item are more likely to fail subsequent ones (Woods 2006)
- **Response validation layer**: Post-generation checks for longstring, IRV, and endpoint utilization anomalies
- **Expanded domain templates**: 5 new domain template sets with both explanation and evaluation question types
- **Behavioral coherence pipeline**: Rating–text consistency ensures numeric patterns match open-text tone
- **Front-facing methods documentation**: Comprehensive update with new paradigms and scientific references

### Version 1.0.4.8
- **AI-powered open-ended responses**: LLM-generated text with draw-with-replacement pooling and 7-layer deep persona variation
- **Multi-provider failover**: Groq, Cerebras, and OpenRouter with automatic key detection
- **Smart pool scaling**: Automatically adapts response pool size to study sample size
- **Template fallback**: Seamless degradation to template engine when AI is unavailable

### Earlier: Conversational Builder
- **Conversational Builder**: Describe experiments in plain language — no QSF file required
- **Automatic factorial detection**: Parses N×M designs from natural language (e.g., "3 × 2, between-subjects")
- **Comprehensive instructor report**: Persona distribution tables, personality trait profiles by condition, effect size verification, exclusion breakdowns
- **Scale auto-detection**: Recognizes detailed academic scale formats, validated instruments (Big Five, PANAS and others), numeric inputs, binary measures
- **Custom persona weights**: Adjust response style distributions for domain-specific realism
- **Domain-specific personas**: the detected research domain influences which persona archetypes are activated

### Earlier: persona and factorial expansion
- Enhanced persona system with 50+ behavioral archetypes across 15 research domains
- Factorial design tables with visual cell numbering
- Effect size specification with Cohen's d calibration
- Auto-generated analysis scripts for R, Python, Julia, SPSS, and Stata

### Initial release
- Initial release with QSF upload, persona-based response generation, and basic instructor reports

---

## Support and Contact

For questions, feature requests, or collaboration inquiries, please contact through official university channels.

---

*Version 1.2.8.9 | Proprietary Software | All Rights Reserved*

*Developed by Dr. Eugen Dimant*
