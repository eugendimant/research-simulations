# Template Fallback and Hybrid Generation System

When LLM prompting fails quality checks or is too costly for certain item types, the system falls back to structured templates. Templates are not static — they encode persona parameters, cross-correlations, and variation, and they improve continuously based on accumulated simulation data.

## When to Use Templates vs LLM Prompting

The system uses a **tiered generation strategy**. Not every item needs a full LLM call.

### Tier 1: Template Generation (Fast, Deterministic)
Use templates when:
- The item is a simple demographic field with constrained options
- The item is a standard Likert scale where the persona's latent disposition can directly determine the response with calibrated noise
- The survey has 100+ items and LLM-per-item cost is prohibitive
- Prior LLM attempts for this item type consistently fail quality checks
- The item type has a well-calibrated template with validated output distributions

### Tier 2: LLM Generation (Rich, Context-Sensitive)
Use LLM prompting when:
- The item is open-ended text (always use LLM for open-text, unless fallback is triggered)
- The item involves complex conditional reasoning (e.g., "Given that you chose X, why?")
- The item is a behavioral decision task with strategic considerations
- The item requires cultural or contextual sensitivity that templates cannot encode
- The item is novel (no template exists yet)

### Tier 3: Hybrid (Template Skeleton + LLM Polish)
Use hybrid when:
- The item is open-text but follows a predictable structure (e.g., "Describe your job")
- Template provides the structure and content direction; LLM adds natural language variation
- Quality check failures on pure LLM output can be constrained by template guardrails

### Fallback Trigger
If an LLM-generated response fails validation 3 times for the same item:
1. Log the failure pattern (what went wrong, which quality check failed)
2. Fall back to template generation for that item
3. Flag the item type for template improvement in the learning log

## Template Architecture

Templates are NOT simple string substitution. They are parameterized generation functions that produce varied, persona-consistent output.

### Quantitative Item Template

```python
def generate_likert_response(
    item_id: str,
    construct: str,
    latent_disposition: float,      # 0-1, from persona trait vector
    reverse_coded: bool,
    scale_min: int,
    scale_max: int,
    prior_responses: dict,          # {item_id: response} for correlation
    fatigue_position: float,        # 0-1, how far into survey
    persona_noise_sd: float = 0.15  # per-persona variation parameter
) -> int:
    """
    Generate a Likert response anchored on latent disposition,
    with correlated noise and human-realistic patterns.
    """
    # 1. Map disposition to scale
    scale_range = scale_max - scale_min
    base = latent_disposition * scale_range + scale_min
    
    # 2. Apply reverse coding
    if reverse_coded:
        base = (scale_max + scale_min) - base
    
    # 3. Add persona-specific noise (calibrated, not uniform)
    noise = np.random.normal(0, persona_noise_sd * scale_range)
    
    # 4. Apply cross-correlation with prior items on same construct
    related_prior = [v for k, v in prior_responses.items() 
                     if item_construct_map[k] == construct]
    if related_prior:
        # Pull toward prior mean on same construct (ensures correlation)
        prior_mean = np.mean(related_prior)
        correlation_weight = 0.3  # tunable
        base = base * (1 - correlation_weight) + prior_mean * correlation_weight
    
    # 5. Apply fatigue effects (slight regression toward center late in survey)
    if fatigue_position > 0.7:
        center = (scale_max + scale_min) / 2
        fatigue_pull = (fatigue_position - 0.7) * 0.3
        base = base * (1 - fatigue_pull) + center * fatigue_pull
    
    # 6. Apply acquiescence bias (slight positive skew)
    acquiescence = 0.1  # tunable
    if not reverse_coded:
        base += acquiescence
    
    # 7. Rare events: straightlining, extreme responses
    if np.random.random() < 0.03:  # 3% straightline probability
        return prior_responses.get(last_same_block_item, round(base + noise))
    
    # 8. Round, clip, return
    response = int(np.clip(round(base + noise), scale_min, scale_max))
    return response
```

### Open-Text Template (Skeleton + Variation)

When LLM generation fails or is unavailable, templates provide structured open-text:

```python
def generate_open_text_template(
    item_id: str,
    prompt_text: str,
    quantitative_anchor: int | None,  # preceding Likert score, if any
    scale_max: int | None,
    persona: dict,
    demographics: dict,
    response_library: dict,           # accumulated good responses by item type
    optional: bool = False
) -> str:
    """
    Generate open-text response using template + variation.
    """
    # 1. Optional field skip probability
    if optional:
        skip_prob = 0.4 + (0.2 * fatigue_position)  # more likely to skip late
        if np.random.random() < skip_prob:
            return None  # SKIP
    
    # 2. Determine response valence from quantitative anchor
    if quantitative_anchor is not None:
        normalized = quantitative_anchor / scale_max
        if normalized < 0.35:
            valence = "negative"
        elif normalized > 0.65:
            valence = "positive"
        else:
            valence = "mixed"
    else:
        valence = "neutral"
    
    # 3. Select from response library (persona-matched, valence-matched)
    candidates = response_library.get(item_id, {}).get(valence, [])
    persona_matched = [c for c in candidates 
                       if c['education_level'] == demographics['education_bucket']]
    
    if persona_matched:
        # Select and vary
        template = np.random.choice(persona_matched)['text']
        response = apply_variation(template, demographics, persona)
    else:
        # Fallback to generic valence-matched response
        response = generate_generic_response(valence, demographics)
    
    # 4. Apply length variation (right-skewed)
    target_words = int(np.random.lognormal(2.5, 0.8))  # median ~12 words
    response = truncate_or_extend(response, target_words)
    
    # 5. Apply education-appropriate language
    response = adjust_language_level(response, demographics['education'])
    
    # 6. Inject human imperfections
    response = add_human_noise(response, demographics)
    
    return response
```

### Demographic Template

```python
def generate_demographic_profile(
    target_population: dict,
    persona_type: str
) -> dict:
    """
    Generate a coherent demographic bundle.
    All fields are generated jointly, not independently.
    """
    # 1. Sample age from target distribution
    age = sample_from_distribution(target_population['age_distribution'])
    
    # 2. Sample gender (may correlate with other fields)
    gender = sample_categorical(target_population['gender_distribution'])
    
    # 3. Education constrained by age
    max_education = education_ceiling(age)
    education = sample_categorical(
        target_population['education_distribution'],
        constraint=lambda e: education_years(e) <= max_education
    )
    
    # 4. Occupation constrained by education
    occupation = sample_occupation(education, target_population)
    
    # 5. Income constrained by occupation and age
    income = sample_income(occupation, age, target_population)
    
    # 6. Marital status constrained by age
    marital = sample_marital_status(age, target_population)
    
    # 7. Coherence check
    profile = {
        'age': age, 'gender': gender, 'education': education,
        'occupation': occupation, 'income': income, 'marital_status': marital
    }
    assert is_coherent(profile), f"Incoherent profile generated: {profile}"
    
    return profile
```

## Cross-Correlation Encoding in Templates

Templates maintain cross-item correlations through the **latent disposition vector**:

### How It Works

1. At subject creation, generate a **latent vector** of K disposition values (one per measured construct), drawn from a multivariate normal distribution with a covariance matrix reflecting known inter-construct correlations.

2. Each construct's disposition value anchors all items measuring that construct. This automatically produces:
   - Positive within-construct correlations (same anchor)
   - Realistic between-construct correlations (covariance matrix)
   - Persona-appropriate response tendencies (persona determines disposition distribution parameters)

3. The covariance matrix should be:
   - Calibrated from published scale validation data when available
   - Estimated from accumulated simulation data when published norms are unavailable
   - Updated as the response library grows (see Continuous Learning)

### Example Covariance Structure

```python
# For a survey measuring risk aversion (RA), trust (TR), 
# prosociality (PS), and conscientiousness (CO):

construct_correlations = {
    ('RA', 'TR'):  -0.15,  # slightly negative (risk-averse people slightly less trusting)
    ('RA', 'PS'):   0.05,  # near-zero (largely independent)
    ('RA', 'CO'):   0.25,  # moderate positive (risk-averse people tend to be conscientious)
    ('TR', 'PS'):   0.35,  # moderate positive (trusting people tend to be prosocial)
    ('TR', 'CO'):   0.10,  # weak positive
    ('PS', 'CO'):   0.20,  # weak-to-moderate positive
}

# Use these to build the covariance matrix for the multivariate normal draw.
# Source: calibrate from published literature or accumulated training data.
```

## Response Library Management

The response library is a growing collection of validated open-text responses indexed by item type, valence, persona, and demographics. It serves as the template source for open-text generation.

### Library Structure

```
response_library/
├── item_types/
│   ├── likert_explanation/       # "Why did you give that rating?"
│   │   ├── positive.json         # responses for high-anchor ratings
│   │   ├── negative.json         # responses for low-anchor ratings
│   │   └── mixed.json            # responses for mid-range ratings
│   ├── free_description/         # "Describe your experience with..."
│   ├── opinion_elaboration/      # "What do you think about...?"
│   └── behavioral_justification/ # "Why did you choose...?"
├── personas/
│   ├── prosocial.json            # persona-specific response patterns
│   ├── individualist.json
│   └── mixed.json
└── demographics/
    ├── high_education.json       # education-appropriate phrasing
    ├── mid_education.json
    └── low_education.json
```

### Adding to the Library

After each simulation run:
1. Extract open-text responses that passed all quality checks
2. Tag them with item type, valence, persona, demographics
3. Add to the library (deduplicate, cap at ~200 per category)
4. Periodically prune low-quality entries (identified by consistency audit failures)

### Variation Functions

Templates are never used verbatim. Variation functions transform them:

- **Synonym substitution**: Replace adjectives and adverbs with register-appropriate alternatives
- **Sentence restructuring**: Reorder clauses, change active/passive voice
- **Length variation**: Truncate or extend based on sampled target length
- **Formality adjustment**: Match education level (formal vocabulary vs. colloquial)
- **Imperfection injection**: Randomly add typos (1-2% of characters), remove punctuation, add filler words ("like," "um," "idk"), lowercase everything

## Template Improvement Loop

Templates are not static. They improve through:

1. **Quality-gated addition**: Only responses that pass the full consistency audit enter the library.
2. **A/B comparison**: Periodically compare template-generated vs LLM-generated responses on the same items. If templates match or exceed LLM quality on a specific item type, expand template coverage.
3. **Failure analysis**: When a template-generated response fails a quality check, diagnose why and update the template function (e.g., if cross-correlation is too weak, increase the correlation_weight parameter).
4. **Distribution calibration**: When real human data is available, compare template output distributions to human distributions. Adjust noise parameters, skewness, and central tendency.
