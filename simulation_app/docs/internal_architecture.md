# Internal Architecture

**CONFIDENTIAL** — Implementation details for v1.0.0

---

## File Structure

```
simulation_app/
├── app.py                      # Streamlit UI, ~3000 lines
├── utils/
│   ├── __init__.py             # Version: must match app.py REQUIRED_UTILS_VERSION
│   ├── enhanced_simulation_engine.py   # Core generation engine
│   ├── qsf_preview.py          # QSF parsing, block detection, scale detection
│   ├── qsf_parser.py           # Low-level QSF structure parsing
│   ├── persona_library.py      # Persona definitions, text generation fallback
│   ├── response_library.py     # 175+ domain templates for open-ended responses
│   ├── text_generator.py       # Basic text generation utilities
│   ├── condition_identifier.py # Extracts conditions from QSF
│   └── instructor_report.py    # Statistical summaries
```

---

## Version Sync

These must all match when you release:
1. `app.py`: `REQUIRED_UTILS_VERSION` (line ~46), `APP_VERSION` (line ~100)
2. `utils/__init__.py`: `__version__`
3. `utils/qsf_preview.py`: `__version__`
4. `utils/response_library.py`: `__version__`
5. `utils/enhanced_simulation_engine.py`: `__version__`

The app checks on startup. Mismatch triggers a warning.

---

## EnhancedSimulationEngine

**Location**: `enhanced_simulation_engine.py`

### Constructor

Takes:
- `study_title`, `study_description`, `sample_size`
- `conditions`: list of condition names
- `factors`: list of {name, levels} dicts
- `scales`: list of {name, num_items, scale_points, reverse_items}
- `open_ended_questions`: list of question specs
- `effect_sizes`: list of EffectSizeSpec
- `exclusion_criteria`: ExclusionCriteria dataclass
- `seed`: optional, uses timestamp+hash if not provided

### Key Methods

`generate() -> (DataFrame, Dict)`: Main entry point. Returns data + metadata.

`_generate_scale_response(min, max, traits, is_reverse, condition, variable_name, seed)`: Single item.

`_generate_open_response(question_spec, persona, traits, condition, seed, response_mean)`: Text response.

`_compute_semantic_condition_effect(condition, variable_name, default_d)`: Parses condition name for effect direction.

### Semantic Parsing

The key insight: effects should come from condition *meaning*, not condition *position*.

```python
# BAD: First condition gets higher scores
# GOOD: "High Trust" gets higher scores than "Low Trust" because of semantics

positive_keywords = ['high', 'good', 'positive', 'friend', 'reward', ...]
negative_keywords = ['low', 'bad', 'negative', 'enemy', 'punishment', ...]
```

The algorithm:
1. Parse condition name for keywords
2. Compute net valence (positive keywords - negative keywords)
3. Add domain-specific adjustments (e.g., AI → algorithm aversion)
4. Scale by configured effect size

---

## SurveyFlowHandler

**Location**: `enhanced_simulation_engine.py` (lines 311-545)

Determines which questions each participant sees based on their condition.

### Detection Methods

1. **Explicit restrictions**: Question spec includes `condition` or `visible_conditions`
2. **DisplayLogic**: Parse Qualtrics logic structures
3. **Block name matching**: "AI_Block" → only AI conditions
4. **Question text hints**: "For those who saw the AI..." → restrict visibility
5. **Factor-level matching**: For "AI × Hedonic", questions for "AI" are visible to both "AI × Hedonic" and "AI × Utilitarian"

### Usage

```python
self.survey_flow_handler = SurveyFlowHandler(conditions, open_ended_questions)

# In generate loop:
if not self.survey_flow_handler.is_question_visible(question_name, participant_condition):
    responses.append("")  # NA
    continue
# else generate response
```

---

## QSF Parsing

**Location**: `qsf_preview.py`

### Block Exclusion

200+ patterns for blocks that shouldn't be treated as conditions:
- Prefixes: `trash_`, `unused_`, `old_`, `test_`, `copy_`
- Suffixes: `_old`, `_copy`, `_backup`, `_v1`
- Names: `consent`, `demographics`, `debrief`, `instructions`

Both exact matches (`EXCLUDED_BLOCK_NAMES` set) and regex patterns (`EXCLUDED_BLOCK_PATTERNS` list).

### Scale Detection

Six types:
1. Matrix scales (multi-item Likert)
2. Numbered items (Scale_1, Scale_2, ...)
3. Likert scales (grouped single-choice)
4. Sliders (visual analog)
5. Single-item DVs
6. Numeric inputs (WTP, etc.)

Detection uses question type, selector, and name patterns.

### DisplayLogic Parsing

```python
def _parse_display_logic(self, display_logic: Dict) -> Dict:
    # Returns:
    # {
    #     'type': 'BooleanExpression',
    #     'conditions': [...],
    #     'depends_on': ['QID123', ...]
    # }
```

---

## Response Library

**Location**: `response_library.py`

### Domain Templates

175+ domains organized by category:
- Behavioral Economics (dictator, ultimatum, trust, public goods, risk, time)
- Social Psychology (intergroup, identity, norms, conformity)
- Consumer (product feedback, brand perception)
- etc.

Each domain has templates for each sentiment level.

### Question Type Detection

30+ types detected by regex patterns:
- Explanatory (why, explain, reason)
- Evaluative (rate, assess, compare)
- Reflective (remember, recall, experience)
- etc.

### Generation Flow

```python
def generate(question_text, sentiment, persona_verbosity, persona_formality, persona_engagement, condition):
    q_type = detect_question_type(question_text)
    domain = detect_study_domain(self.study_context, question_text)
    response = self._get_template_response(domain, q_type, sentiment)
    response = self._personalize_for_question(response, question_text, keywords, condition)
    # Apply persona modifiers...
    return response
```

---

## Seeding Strategy

Everything uses deterministic seeds for reproducibility:

```python
self.seed = base_seed  # Or timestamp + hash if None
participant_seed = (self.seed + i * 100) % (2**31)
column_seed = (self.seed + i * 100 + _stable_int_hash(column_name)) % (2**31)
```

`_stable_int_hash` uses MD5, not Python's `hash()`, for cross-platform reproducibility.

---

## Common Gotchas

### Type Safety in QSF Parsing

QSF files can have unexpected structures. Always check types:

```python
# BAD
message = custom_val.get('Message', '')
if 'email' in message.lower():  # Fails if message is dict

# GOOD
message = custom_val.get('Message', '')
if isinstance(message, str) and 'email' in message.lower():
```

### Version Mismatch

If versions don't match, Streamlit's module caching can cause weird behavior. The app warns but doesn't crash. Fix by updating all version constants.

### Effect Order

If effects seem to follow condition order rather than meaning, check:
1. Semantic parsing is detecting keywords correctly
2. Effect specification has correct level_high/level_low
3. Condition names have meaningful content

---

## Testing

Key things to verify:
1. **Effect recovery**: Observed effects approximate configured values
2. **Survey flow**: NA values appear for non-visible questions
3. **Text uniqueness**: Different questions get different responses
4. **Reproducibility**: Same seed → same output

---

*Internal use only. Do not distribute.*
