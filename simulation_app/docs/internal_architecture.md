# Internal Architecture Documentation

**Behavioral Experiment Simulation Tool v1.0.0**

**CONFIDENTIAL** | Proprietary Implementation Details

---

## System Overview

This document contains proprietary implementation details for the Behavioral Experiment Simulation Tool. It is intended for internal development and maintenance purposes only.

---

## 1. Module Architecture

```
simulation_app/
├── app.py                              # Streamlit application (~3000 lines)
├── utils/
│   ├── __init__.py                     # Package initialization, version control
│   ├── enhanced_simulation_engine.py   # Core data generation engine
│   ├── qsf_preview.py                  # QSF parsing, scale detection
│   ├── qsf_parser.py                   # Low-level QSF structure parsing
│   ├── persona_library.py              # Persona definitions, trait distributions
│   ├── response_library.py             # Domain templates for open-ended responses
│   ├── text_generator.py               # Text generation utilities
│   ├── condition_identifier.py         # Experimental condition extraction
│   └── instructor_report.py            # Statistical report generation
└── docs/
    ├── methods_summary.md              # Public documentation
    ├── technical_methods.md            # Scientific methods documentation
    └── internal_architecture.md        # This document
```

---

## 2. Version Management

**Critical**: These version identifiers must be synchronized on every release:

| Location | Variable |
|----------|----------|
| `app.py` | `REQUIRED_UTILS_VERSION` (line ~46) |
| `app.py` | `APP_VERSION` (line ~100) |
| `app.py` | `BUILD_ID` (line ~47) |
| `utils/__init__.py` | `__version__` |
| `utils/qsf_preview.py` | `__version__` |
| `utils/response_library.py` | `__version__` |
| `utils/enhanced_simulation_engine.py` | `__version__` |

The application performs a version check at startup. Mismatches trigger a user-visible warning.

---

## 3. EnhancedSimulationEngine

**Location**: `enhanced_simulation_engine.py`

### 3.1 Constructor Parameters

```python
EnhancedSimulationEngine(
    study_title: str,
    study_description: str,
    sample_size: int,
    conditions: List[str],
    factors: List[Dict[str, Any]],
    scales: List[Dict[str, Any]],
    additional_vars: List[Dict[str, Any]],
    demographics: Dict[str, Any],
    attention_rate: float = 0.95,
    random_responder_rate: float = 0.05,
    effect_sizes: Optional[List[EffectSizeSpec]] = None,
    exclusion_criteria: Optional[ExclusionCriteria] = None,
    open_ended_questions: Optional[List[Dict[str, Any]]] = None,
    study_context: Optional[Dict[str, Any]] = None,
    condition_allocation: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
    mode: str = "pilot"
)
```

### 3.2 Key Methods

| Method | Purpose |
|--------|---------|
| `generate()` | Main entry point; returns (DataFrame, metadata_dict) |
| `_generate_scale_response()` | Produces single Likert item value |
| `_generate_open_response()` | Produces open-ended text response |
| `_compute_semantic_condition_effect()` | Parses condition semantics for effect direction |
| `_generate_condition_assignment()` | Allocates participants to conditions |

### 3.3 Semantic Effect Parsing

The system determines effect direction from condition name semantics:

```python
positive_keywords = ['high', 'good', 'positive', 'friend', 'reward', ...]
negative_keywords = ['low', 'bad', 'negative', 'enemy', 'punishment', ...]

# Domain-specific adjustments
if 'ai' in condition.lower():
    effect -= 0.08  # Algorithm aversion
```

This ensures effects follow logical design structure rather than arbitrary condition ordering.

---

## 4. SurveyFlowHandler

**Location**: `enhanced_simulation_engine.py` (lines 311-545)

### 4.1 Purpose

Determines which questions each participant sees based on their experimental condition, ensuring synthetic data respects the survey's programmed logic.

### 4.2 Detection Methods

1. **Explicit restrictions**: `condition` or `visible_conditions` in question spec
2. **DisplayLogic parsing**: Extract Qualtrics logic structures
3. **Block name analysis**: Condition keywords in block names
4. **Question text analysis**: Conditional phrases in question text
5. **Factor-level matching**: Partial matches for factorial designs

### 4.3 Implementation

```python
class SurveyFlowHandler:
    def __init__(self, conditions, open_ended_questions):
        self.visibility_map = self._build_visibility_map()

    def is_question_visible(self, question_name, condition) -> bool:
        # Returns True if participant in this condition would see the question

    def get_visible_questions(self, condition) -> List[Dict]:
        # Returns all questions visible for this condition
```

---

## 5. QSF Parsing

**Location**: `qsf_preview.py`

### 5.1 Block Exclusion Patterns

The system excludes 200+ patterns from condition detection:

**Prefixes**: `trash_`, `unused_`, `old_`, `test_`, `copy_`, `delete_`, `archive_`

**Suffixes**: `_old`, `_copy`, `_backup`, `_unused`, `_v1`, `_v2`, `_DONOTUSE`

**Names**: `consent`, `demographics`, `debrief`, `instructions`, `attention_check`, `manipulation_check`

### 5.2 Scale Detection

Six detection types:
1. Matrix scales (multi-item Likert)
2. Numbered items (Scale_1, Scale_2, ...)
3. Likert scales (grouped single-choice)
4. Sliders (visual analog)
5. Single-item DVs
6. Numeric inputs

### 5.3 DisplayLogic Parsing

```python
def _parse_display_logic(self, display_logic: Dict) -> Dict:
    return {
        'type': display_logic.get('Type', 'Unknown'),
        'conditions': [...],
        'depends_on': [question_ids...]
    }
```

---

## 6. Response Library

**Location**: `response_library.py`

### 6.1 Domain Coverage

175+ domains organized by research area:
- Behavioral Economics (12 domains)
- Social Psychology (15 domains)
- Consumer Behavior (10 domains)
- Technology/AI (10 domains)
- And 26 additional categories

### 6.2 Generation Pipeline

```python
def generate(question_text, sentiment, persona_traits, condition):
    q_type = detect_question_type(question_text)
    domain = detect_study_domain(study_context, question_text)
    response = _get_template_response(domain, q_type, sentiment)
    response = _personalize_for_question(response, question_text, keywords, condition)
    response = apply_persona_modulation(response, persona_traits)
    return response
```

---

## 7. Seeding Strategy

All stochastic elements use deterministic seeding:

```python
self.seed = base_seed  # Or timestamp + MD5(study_title) if None
participant_seed = (self.seed + i * 100) % (2**31)
column_seed = (self.seed + i * 100 + _stable_int_hash(column_name)) % (2**31)
```

`_stable_int_hash` uses MD5 for cross-platform reproducibility.

---

## 8. Common Implementation Issues

### 8.1 Type Safety in QSF Parsing

QSF files may contain unexpected data structures. Always validate types:

```python
# Incorrect:
message = custom_val.get('Message', '')
if 'email' in message.lower():  # Fails if message is dict

# Correct:
message = custom_val.get('Message', '')
if isinstance(message, str) and 'email' in message.lower():
```

### 8.2 Effect Order Artifacts

If effects correlate with condition position rather than semantics:
1. Verify semantic parser detects keywords correctly
2. Check EffectSizeSpec has correct level_high/level_low
3. Ensure condition names contain meaningful content

### 8.3 Version Mismatch

Streamlit caches modules aggressively. Version mismatches cause unpredictable behavior. Always update BUILD_ID when deploying changes.

---

## 9. Quality Assurance

### 9.1 Validation Checks

| Check | Target |
|-------|--------|
| Effect recovery | Observed d within 0.1 of configured d |
| Mean Likert response | 4.0-5.2 on 7-point scales |
| Within-condition SD | 1.2-1.8 |
| Attention pass rate | 85-95% (depending on difficulty) |
| Survey flow | NA values for non-visible questions |

### 9.2 Reproducibility Testing

```python
engine1 = EnhancedSimulationEngine(..., seed=12345)
engine2 = EnhancedSimulationEngine(..., seed=12345)
df1, _ = engine1.generate()
df2, _ = engine2.generate()
assert df1.equals(df2)
```

---

*CONFIDENTIAL — Internal use only. Do not distribute.*

*© 2026 Dr. Eugen Dimant. All rights reserved.*
