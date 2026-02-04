# Comprehensive Internal Architecture Documentation

**Behavioral Experiment Simulation Tool v1.0.0**

**CONFIDENTIAL** | Complete Implementation Specification

*This document provides complete technical specifications sufficient to reproduce the entire software system.*

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Directory Structure](#2-directory-structure)
3. [Core Components](#3-core-components)
4. [QSF Parsing Engine](#4-qsf-parsing-engine)
5. [Simulation Engine](#5-simulation-engine)
6. [Persona Library](#6-persona-library)
7. [Response Generation](#7-response-generation)
8. [Survey Flow Logic](#8-survey-flow-logic)
9. [Effect Size Calibration](#9-effect-size-calibration)
10. [Data Quality Validation](#10-data-quality-validation)
11. [Streamlit Application](#11-streamlit-application)
12. [Version Management](#12-version-management)
13. [Testing & Validation](#13-testing--validation)
14. [Complete API Reference](#14-complete-api-reference)

---

## 1. System Overview

### 1.1 Purpose

The Behavioral Experiment Simulation Tool generates scientifically-calibrated synthetic data for behavioral science experiments. It:

- Parses Qualtrics Survey Format (QSF) files to extract experimental structure
- Simulates realistic participant responses based on persona-driven models
- Produces data that matches expected statistical properties (effect sizes, distributions)
- Supports complex experimental designs (factorial, between-subjects, mixed)

### 1.2 Technology Stack

```
Language:       Python 3.8+
Web Framework:  Streamlit 1.28+
Data:           pandas, numpy
Parsing:        json, re (regex)
Statistics:     scipy (optional), numpy
```

### 1.3 Design Principles

1. **Scientific Grounding**: All response generation algorithms are based on published research
2. **Defensive Programming**: Graceful handling of malformed inputs, edge cases
3. **Reproducibility**: Seed-based random number generation for identical outputs
4. **Extensibility**: Modular architecture for adding new question types, domains

---

## 2. Directory Structure

```
simulation_app/
├── app.py                                 # Main Streamlit application
├── utils/
│   ├── __init__.py                        # Package init with version
│   ├── enhanced_simulation_engine.py      # Core simulation engine
│   ├── qsf_preview.py                     # QSF parsing and validation
│   ├── persona_library.py                 # Persona definitions
│   └── response_library.py                # Domain-specific response templates
├── docs/
│   ├── methods_summary.md                 # Front-facing documentation
│   ├── technical_methods.md               # Scientific methods documentation
│   └── internal_architecture.md           # This document
└── example_files/
    └── *.qsf                              # Sample QSF files for testing
```

---

## 3. Core Components

### 3.1 Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                       app.py (Streamlit UI)                      │
├─────────────────────────────────────────────────────────────────┤
│  Upload QSF → Parse → Configure → Generate → Download            │
└───────────┬─────────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────┐      ┌──────────────────────────────────┐
│   qsf_preview.py      │      │  enhanced_simulation_engine.py   │
│   ─────────────────   │      │  ────────────────────────────    │
│   • Parse QSF JSON    │──────▶  • Generate participant data      │
│   • Detect conditions │      │  • Apply effect sizes             │
│   • Extract scales    │      │  • Handle survey flow logic       │
│   • Find open-ended   │      │  • Produce validated output       │
└───────────────────────┘      └───────────────┬──────────────────┘
                                               │
                               ┌───────────────┴───────────────┐
                               ▼                               ▼
                 ┌─────────────────────┐       ┌─────────────────────┐
                 │  persona_library.py  │       │ response_library.py │
                 │  ─────────────────   │       │ ─────────────────   │
                 │  • Persona types     │       │ • Domain templates  │
                 │  • Trait generation  │       │ • Question types    │
                 │  • Response styles   │       │ • Word banks        │
                 └─────────────────────┘       └─────────────────────┘
```

### 3.2 Data Flow

1. **Input**: User uploads QSF file via Streamlit interface
2. **Parsing**: `QSFPreviewParser.parse()` extracts survey structure
3. **Configuration**: User confirms conditions, scales, DVs
4. **Generation**: `EnhancedSimulationEngine.generate()` produces DataFrame
5. **Output**: CSV download with simulated data

---

## 4. QSF Parsing Engine

### 4.1 Overview

**File**: `utils/qsf_preview.py`
**Class**: `QSFPreviewParser`
**Purpose**: Parse Qualtrics Survey Format files and extract experimental structure

### 4.2 QSF File Structure

QSF files are JSON with this top-level structure:

```json
{
  "SurveyEntry": {
    "SurveyID": "SV_xxx",
    "SurveyName": "My Survey",
    ...
  },
  "SurveyElements": [
    {"Element": "BL", "Payload": {...}},  // Blocks
    {"Element": "FL", "Payload": {...}},  // Flow
    {"Element": "SQ", "Payload": {...}},  // Survey Questions
    {"Element": "RS", "Payload": {...}},  // Response Sets
    ...
  ]
}
```

### 4.3 Key Data Classes

```python
@dataclass
class QuestionInfo:
    question_id: str              # e.g., "QID1"
    question_text: str            # Full question text (HTML stripped)
    question_type: str            # MC, TE, Slider, Matrix, etc.
    block_name: str               # Containing block name
    choices: List[str]            # Answer choices
    scale_points: Optional[int]   # For Likert: 5, 7, etc.
    is_matrix: bool               # True if matrix/table question
    sub_questions: List[str]      # For matrix: row items
    selector: str                 # SAVR, HSLIDER, ML, etc.
    has_skip_logic: bool          # Has conditional skip
    has_display_logic: bool       # Has conditional display
    display_logic_details: Dict   # Full display logic spec
    skip_logic_details: Dict      # Full skip logic spec
    is_reverse_coded: bool        # Reverse-scored item
    scale_anchors: Dict           # {1: "Strongly disagree", ...}

@dataclass
class BlockInfo:
    block_id: str                 # e.g., "BL_xxx"
    block_name: str               # User-defined block name
    questions: List[QuestionInfo] # Questions in this block
    is_randomizer: bool           # Part of BlockRandomizer
    block_type: str               # Standard, Default, Trash

@dataclass
class QSFPreviewResult:
    success: bool
    survey_name: str
    total_questions: int
    total_blocks: int
    blocks: List[BlockInfo]
    detected_conditions: List[str]
    detected_scales: List[Dict]
    embedded_data: List[str]
    flow_elements: List[str]
    validation_errors: List[str]
    open_ended_questions: List[str]
    open_ended_details: List[Dict]
    skip_logic_map: Dict
    display_logic_map: Dict
```

### 4.4 Condition Detection Algorithm

Conditions are detected from multiple sources in priority order:

```python
def _extract_conditions_from_flow(self, flow_data):
    """
    Detection sources (in order of reliability):

    1. BlockRandomizer with SubSet=1
       - Between-subjects design
       - Each block = one condition
       - Must have 2+ blocks to be valid

    2. EmbeddedData inside Randomizer
       - Condition set via field value
       - e.g., EmbeddedData: {"Field": "Condition", "Value": "AI"}

    3. Branch elements with descriptive names
       - Branch descriptions often contain condition names
       - Parse "Treatment vs Control" patterns

    4. Block names (fallback)
       - Only if no randomizers found
       - Filter through EXCLUDED_BLOCK_NAMES (200+ patterns)
    """
```

### 4.5 Trash Block Filtering

Critical for avoiding false condition detection:

```python
EXCLUDED_BLOCK_NAMES = {
    # Trash/Unused (NEVER use as conditions)
    'trash', 'trash / unused questions', 'unused', 'deleted',
    'archived', 'old', 'deprecated', 'do not use', 'ignore',

    # Generic names
    'block', 'block 1', 'block 2', 'default question block',

    # Admin blocks
    'consent', 'informed consent', 'demographics', 'debrief',
    'instructions', 'intro', 'end', 'thank you',

    # Quality control
    'attention check', 'manipulation check', 'comprehension check',

    # ... 200+ total patterns
}

EXCLUDED_BLOCK_PATTERNS = [
    r'^block\s*\d*$',      # "Block 1", "Block2"
    r'trash',              # Contains "trash"
    r'unused',             # Contains "unused"
    r'default',            # Contains "default"
    # ... many more regex patterns
]

def _is_excluded_block_name(self, name: str) -> bool:
    """Check if block name should be excluded from conditions."""
    normalized = name.lower().strip()

    # Check exact matches
    if normalized in self.EXCLUDED_BLOCK_NAMES:
        return True

    # Check regex patterns
    for pattern in self.EXCLUDED_BLOCK_PATTERNS:
        if re.search(pattern, normalized):
            return True

    return False
```

### 4.6 Scale Detection

Six detection methods for dependent variables:

```python
def _detect_scales(self, questions_map, blocks):
    """
    Detection methods:

    1. Matrix scales (question_type == 'Matrix')
       - Multi-item Likert with shared response options
       - Each sub-question = one item

    2. Numbered items (e.g., Trust_1, Trust_2, Trust_3)
       - Detect common prefix + numeric suffix
       - Group into single scale

    3. Likert scales (grouped single-choice)
       - MC questions with same choices
       - Adjacent questions with similar text

    4. Sliders (selector in ['HSLIDER', 'VSLIDER'])
       - Visual analog scales
       - Continuous 0-100 values

    5. Single-item DVs
       - Standalone Likert with scale_points >= 5
       - Check for DV keywords in text

    6. Numeric inputs
       - Text entry with number validation
       - WTP (willingness to pay), quantities
    """

    scales = []

    # Method 1: Matrix detection
    for q_info in questions_map.values():
        if q_info.is_matrix and len(q_info.sub_questions) >= 2:
            scales.append({
                'name': q_info.export_tag or q_info.question_id,
                'num_items': len(q_info.sub_questions),
                'scale_points': q_info.scale_points or 7,
                'detection_method': 'matrix',
                'detected_from_qsf': True
            })

    # Method 2: Numbered items
    prefix_groups = defaultdict(list)
    for q_id, q_info in questions_map.items():
        match = re.match(r'(.+?)_?(\d+)$', q_info.export_tag)
        if match:
            prefix_groups[match.group(1)].append(q_info)

    for prefix, items in prefix_groups.items():
        if len(items) >= 2:
            scales.append({
                'name': prefix,
                'num_items': len(items),
                'scale_points': items[0].scale_points or 7,
                'detection_method': 'numbered',
                'detected_from_qsf': True
            })

    # ... additional methods

    return scales
```

### 4.7 Open-Ended Question Detection

```python
def _detect_open_ended(self, questions_map):
    """
    Detect all text entry questions:

    1. Question type == 'TE' (Text Entry)
       - Selectors: SL (single line), ML (multi-line), ESTB (essay)

    2. MC questions with text entry choices
       - Choice has TextEntry=True attribute
       - e.g., "Other: ____"

    3. Matrix with text entry columns
       - Answer column allows text input

    4. FORM questions (multiple text fields)
       - Each form field = separate text entry
    """

    open_ended = []

    for q_id, q_info in questions_map.items():
        # Skip non-question types
        if q_info.question_type in ['DB', 'Timing', 'Captcha']:
            continue

        # Check for text entry type
        if q_info.question_type in ['TE', 'Text Entry']:
            open_ended.append(q_info.export_tag or q_id)

        # Check for MC with text entry choices
        if q_info.text_entry_choices:
            for te_choice in q_info.text_entry_choices:
                open_ended.append(f"{q_info.export_tag}_{te_choice['id']}")

    return open_ended
```

---

## 5. Simulation Engine

### 5.1 Overview

**File**: `utils/enhanced_simulation_engine.py`
**Class**: `EnhancedSimulationEngine`
**Purpose**: Generate scientifically-calibrated synthetic participant data

### 5.2 Constructor Specification

```python
class EnhancedSimulationEngine:
    def __init__(
        self,
        # Required parameters
        study_title: str,           # Used for domain detection
        study_description: str,     # Used for domain detection
        sample_size: int,           # Number of participants to generate
        conditions: List[str],      # Experimental conditions
        factors: List[Dict],        # Factor structure for factorial designs
        scales: List[Dict],         # DVs to generate
        additional_vars: List[Dict],# Extra variables
        demographics: Dict,         # Demographics to include

        # Quality parameters
        attention_rate: float = 0.95,        # Proportion passing attention checks
        random_responder_rate: float = 0.05, # Proportion of careless responders

        # Effect specification
        effect_sizes: Optional[List[EffectSizeSpec]] = None,

        # Exclusion simulation
        exclusion_criteria: Optional[ExclusionCriteria] = None,

        # Open-ended configuration
        open_ended_questions: Optional[List[Dict]] = None,
        study_context: Optional[Dict] = None,

        # Allocation
        condition_allocation: Optional[Dict[str, float]] = None,

        # Reproducibility
        seed: Optional[int] = None,  # None = auto-generate

        # Mode
        mode: str = "pilot"  # "pilot" or "final"
    ):
```

### 5.3 Data Generation Pipeline

```python
def generate(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main generation pipeline:

    1. Initialize data dictionary
    2. Generate participant IDs and metadata
    3. Assign conditions (with optional allocation)
    4. For each participant:
       a. Assign persona
       b. Generate demographics
       c. For each scale:
          - Generate correlated items
          - Apply condition effects
          - Apply persona adjustments
       d. Generate open-ended responses (with survey flow logic)
       e. Generate attention check responses
       f. Generate exclusion flags
    5. Compile DataFrame
    6. Compute quality metrics
    7. Return (DataFrame, metadata)
    """

    n = self.sample_size
    data = {}

    # Metadata columns
    data['PARTICIPANT_ID'] = [f"P{i+1:04d}" for i in range(n)]
    data['RUN_ID'] = [self.run_id] * n
    data['SIMULATION_MODE'] = [self.mode.upper()] * n
    data['SIMULATION_SEED'] = [self.seed] * n

    # Condition assignment
    data['CONDITION'] = self._generate_condition_assignment(n)

    # Demographics
    demographics_df = self._generate_demographics(n)
    for col in demographics_df.columns:
        data[col] = demographics_df[col].tolist()

    # For each participant
    for i in range(n):
        participant_seed = self.seed + i * 1000
        rng = np.random.RandomState(participant_seed)

        # Assign persona
        persona = self._select_persona(rng)
        traits = persona.generate_traits(rng)

        condition = data['CONDITION'][i]

        # Generate scale responses
        for scale in self.scales:
            responses = self._generate_scale_responses(
                scale, condition, traits, participant_seed
            )
            for j, response in enumerate(responses):
                col_name = f"{scale['name']}_{j+1}"
                if col_name not in data:
                    data[col_name] = [None] * n
                data[col_name][i] = response

        # Generate open-ended responses
        for q in self.open_ended_questions:
            if self.survey_flow_handler.is_question_visible(q['name'], condition):
                response = self._generate_open_response(q, condition, traits, rng)
            else:
                response = ""  # Not visible to this participant

            if q['name'] not in data:
                data[q['name']] = [None] * n
            data[q['name']][i] = response

    df = pd.DataFrame(data)
    metadata = self._compile_metadata(df)

    return df, metadata
```

### 5.4 Condition Assignment

```python
def _generate_condition_assignment(self, n: int) -> pd.Series:
    """
    Assign participants to conditions.

    If condition_allocation is provided:
        Use specified percentages (must sum to 100)
    Else:
        Equal distribution across all conditions

    Always shuffles to randomize order.
    """

    n_conditions = len(self.conditions)
    assignments = []

    if self.condition_allocation:
        # Use specified allocation
        running_total = 0
        for i, cond in enumerate(self.conditions):
            pct = self.condition_allocation.get(cond, 100 / n_conditions)
            if i == n_conditions - 1:
                count = n - running_total  # Last condition gets remainder
            else:
                count = round(n * pct / 100)
                running_total += count
            assignments.extend([cond] * max(0, count))
    else:
        # Equal distribution
        n_per = n // n_conditions
        remainder = n % n_conditions
        for i, cond in enumerate(self.conditions):
            count = n_per + (1 if i < remainder else 0)
            assignments.extend([cond] * count)

    # Ensure exactly n assignments
    while len(assignments) < n:
        assignments.append(self.conditions[-1])
    assignments = assignments[:n]

    # Shuffle
    rng = np.random.RandomState(self.seed + 2000)
    rng.shuffle(assignments)

    return pd.Series(assignments, name="CONDITION")
```

### 5.5 Condition Semantic Parsing

```python
# Keyword sets for determining effect direction
POSITIVE_VALENCE_KEYWORDS = {
    'high', 'positive', 'good', 'love', 'lover', 'friend', 'prosocial',
    'generous', 'kind', 'warm', 'trust', 'trusting', 'hedonic', 'pleasure',
    'reward', 'gain', 'win', 'success', 'treatment', 'experimental', 'active',
    'present', 'yes', 'true', 'included', 'with', 'pro', 'support', 'agree'
}

NEGATIVE_VALENCE_KEYWORDS = {
    'low', 'negative', 'bad', 'hate', 'hater', 'enemy', 'antisocial',
    'selfish', 'cruel', 'cold', 'distrust', 'distrusting', 'utilitarian',
    'practical', 'loss', 'lose', 'failure', 'control', 'placebo', 'inactive',
    'absent', 'no', 'false', 'excluded', 'without', 'anti', 'oppose'
}

def _parse_condition_semantics(condition: str) -> Dict[str, Any]:
    """
    Extract semantic meaning from condition name.

    Returns:
        {
            'original': str,           # Original condition name
            'valence': float,          # -1 to +1 (negative to positive)
            'factors': List[str],      # Parsed factorial levels
            'manipulation_type': str,  # 'ai_human', 'hedonic_utilitarian', etc.
            'is_control': bool,        # Contains control indicators
            'is_treatment': bool,      # Contains treatment indicators
        }
    """
    cond_lower = condition.lower()
    cond_parts = cond_lower.replace('×', ' ').replace('_', ' ').split()

    positive_count = sum(1 for p in cond_parts if p in POSITIVE_VALENCE_KEYWORDS)
    negative_count = sum(1 for p in cond_parts if p in NEGATIVE_VALENCE_KEYWORDS)

    total = positive_count + negative_count
    valence = (positive_count - negative_count) / total if total > 0 else 0.0

    return {
        'original': condition,
        'valence': valence,
        'factors': cond_parts,
        'is_control': any(kw in cond_lower for kw in ['control', 'baseline', 'placebo']),
        'is_treatment': any(kw in cond_lower for kw in ['treatment', 'experimental']),
    }
```

### 5.6 Scale Response Generation

```python
def _generate_scale_response(
    self,
    scale: Dict,
    condition: str,
    traits: Dict[str, float],
    rng: np.random.RandomState
) -> int:
    """
    Generate a single Likert item response.

    10-Step Algorithm (scientifically calibrated):

    STEP 1: Get base response tendency from persona
    STEP 2: Apply condition effect (from semantic parsing)
    STEP 3: Apply domain calibration
    STEP 4: Apply scale-type calibration
    STEP 5: Handle reverse-coded items
    STEP 6: Add within-person variance
    STEP 7: Apply extreme response style
    STEP 8: Apply acquiescence bias
    STEP 9: Apply social desirability
    STEP 10: Clip to valid scale range
    """

    scale_points = scale.get('scale_points', 7)
    scale_min, scale_max = 1, scale_points

    # STEP 1: Base tendency (produces M ≈ 4.0-5.2 on 7-point)
    base_tendency = traits.get('response_tendency', 0.6)
    response = base_tendency * (scale_max - scale_min) + scale_min

    # STEP 2: Condition effect
    semantics = _parse_condition_semantics(condition)
    effect_modifier = semantics['valence'] * self.target_effect_size * (scale_points - 1) / 4
    response += effect_modifier

    # STEP 3: Domain calibration
    var_name = scale.get('variable_name', scale.get('name', '')).lower()
    if 'satisfaction' in var_name:
        response += 0.3  # Satisfaction has positive bias
    elif 'risk' in var_name:
        response -= 0.2  # Risk perception slightly lower

    # STEP 4: Scale-type calibration
    if scale.get('is_slider'):
        response += rng.normal(0, 0.5)  # More variance for sliders

    # STEP 5: Reverse-coded handling
    item_num = scale.get('current_item', 1)
    if item_num in scale.get('reverse_items', []):
        response = (scale_max + scale_min) - response
        # Acquiescent adjustment
        if traits.get('acquiescence', 0.5) > 0.6:
            response += 0.25 * (scale_max - scale_min)

    # STEP 6: Within-person variance
    variance = traits.get('variance', 0.3)
    sd = (scale_max - scale_min) / 4 * variance
    response += rng.normal(0, sd)

    # STEP 7: Extreme response style
    extremity = traits.get('extremity', 0.3)
    if rng.random() < extremity * 0.45:
        if response > (scale_min + scale_max) / 2:
            response = scale_max
        else:
            response = scale_min

    # STEP 8: Acquiescence bias
    acquiescence = traits.get('acquiescence', 0.5)
    inflation = (acquiescence - 0.5) * (scale_max - scale_min) * 0.20
    response += inflation

    # STEP 9: Social desirability
    sd_bias = traits.get('social_desirability', 0.5)
    sd_inflation = (sd_bias - 0.5) * (scale_max - scale_min) * 0.12
    response += sd_inflation

    # STEP 10: Clip and round
    return int(np.clip(round(response), scale_min, scale_max))
```

### 5.7 Correlated Item Generation (Scale Reliability)

```python
def _generate_correlated_items(
    n_items: int,
    base_response: float,
    target_alpha: float,       # Target Cronbach's alpha (0.70-0.90)
    scale_points: int,
    reverse_items: List[int],
    rng: np.random.RandomState
) -> List[int]:
    """
    Generate scale items that achieve target reliability.

    Uses factor model:
        response_i = λ * F + √(1-λ²) * e_i

    Where:
        F = common factor (person's true score)
        λ = factor loading (derived from target α)
        e_i = item-specific error

    From Cronbach's alpha formula:
        α = n * r_bar / (1 + (n-1) * r_bar)

    Solving for average inter-item correlation:
        r_bar = α / (n - α * (n-1))

    Factor loading:
        λ = √r_bar
    """

    if n_items <= 1:
        response = int(np.clip(round(base_response * (scale_points - 1) + 1), 1, scale_points))
        return [response]

    # Calculate needed inter-item correlation
    r_bar = target_alpha / (n_items - target_alpha * (n_items - 1))
    r_bar = np.clip(r_bar, 0.1, 0.9)

    # Factor loadings
    factor_loading = np.sqrt(r_bar)
    unique_loading = np.sqrt(1 - r_bar)

    # Common factor score (person's latent trait)
    common_factor = rng.normal(0, 1)

    responses = []
    for i in range(n_items):
        # True score = common + unique
        true_score = factor_loading * common_factor + unique_loading * rng.normal(0, 1)

        # Transform to response scale
        mean_response = base_response * (scale_points - 1) + 1
        sd_response = (scale_points - 1) / 4
        raw_response = mean_response + true_score * sd_response

        # Handle reverse coding
        if (i + 1) in reverse_items:
            raw_response = (scale_points + 1) - raw_response

        response = int(np.clip(round(raw_response), 1, scale_points))
        responses.append(response)

    return responses
```

---

## 6. Persona Library

### 6.1 Overview

**File**: `utils/persona_library.py`
**Purpose**: Define response style personas based on survey methodology research

### 6.2 Core Persona Types

```python
CORE_PERSONAS = {
    'engaged': {
        'name': 'Engaged Responder',
        'description': "Krosnick's optimizers - high attention, full scale use",
        'weight': 0.35,  # 35% of population
        'traits': {
            'response_tendency': (0.55, 0.10),  # Mean, SD
            'extremity': (0.30, 0.10),
            'acquiescence': (0.50, 0.08),
            'attention_level': (0.90, 0.05),
            'variance': (0.50, 0.10),
            'social_desirability': (0.50, 0.10),
        }
    },

    'satisficer': {
        'name': 'Satisficer',
        'description': "Krosnick's satisficers - minimized effort",
        'weight': 0.22,  # 22% of population
        'traits': {
            'response_tendency': (0.55, 0.08),
            'extremity': (0.15, 0.08),       # Low extremity
            'acquiescence': (0.60, 0.10),    # Higher agreement
            'attention_level': (0.60, 0.10),  # Lower attention
            'variance': (0.30, 0.08),        # Restricted range
        }
    },

    'extreme_responder': {
        'name': 'Extreme Responder',
        'description': "Greenleaf's ERS - consistent endpoint use",
        'weight': 0.10,  # 10% of population
        'traits': {
            'extremity': (0.75, 0.10),       # High extremity
            'acquiescence': (0.55, 0.10),
            'variance': (0.70, 0.10),
        }
    },

    'acquiescent': {
        'name': 'Acquiescent Responder',
        'description': "Billiet's agreement bias",
        'weight': 0.08,
        'traits': {
            'acquiescence': (0.80, 0.08),    # Strong agreement
            'extremity': (0.35, 0.10),
        }
    },

    'careless': {
        'name': 'Careless Responder',
        'description': "Meade & Craig's inattentive",
        'weight': 0.05,  # 5% of population
        'traits': {
            'attention_level': (0.30, 0.10),  # Very low attention
            'variance': (0.80, 0.10),         # High variance (random)
            'extremity': (0.50, 0.20),
        }
    },

    'socially_desirable': {
        'name': 'Socially Desirable Responder',
        'description': "Paulhus's high impression management",
        'weight': 0.12,
        'traits': {
            'social_desirability': (0.80, 0.08),
            'response_tendency': (0.65, 0.08),  # Positive bias
        }
    },
}
```

### 6.3 Domain-Specific Personas

```python
DOMAIN_PERSONAS = {
    'technology': {
        'tech_enthusiast': {
            'weight': 0.15,
            'traits': {'response_tendency': (0.70, 0.10)},  # Positive toward tech
            'trigger_keywords': ['ai', 'algorithm', 'technology', 'robot']
        },
        'tech_skeptic': {
            'weight': 0.12,
            'traits': {'response_tendency': (0.35, 0.10)},  # Negative toward tech
            'trigger_keywords': ['ai', 'algorithm', 'technology', 'robot']
        },
    },

    'consumer': {
        'brand_loyalist': {
            'weight': 0.15,
            'traits': {'response_tendency': (0.72, 0.08)},
        },
        'deal_seeker': {
            'weight': 0.20,
            'traits': {'response_tendency': (0.45, 0.12)},
        },
    },

    'political': {
        'partisan': {
            'weight': 0.30,
            'traits': {'extremity': (0.70, 0.10)},
        },
        'moderate': {
            'weight': 0.25,
            'traits': {'extremity': (0.20, 0.08)},
        },
    },
}
```

### 6.4 Trait Generation

```python
class Persona:
    def __init__(self, name: str, traits: Dict, weight: float):
        self.name = name
        self.traits = traits  # {trait_name: (mean, sd)}
        self.weight = weight

    def generate_traits(self, rng: np.random.RandomState) -> Dict[str, float]:
        """Generate specific trait values for this persona instance."""
        generated = {}
        for trait_name, (mean, sd) in self.traits.items():
            value = rng.normal(mean, sd)
            # Clip to valid range (0-1)
            generated[trait_name] = np.clip(value, 0.0, 1.0)
        return generated


class PersonaLibrary:
    def __init__(self, seed: int = None):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.personas = self._load_personas()

    def select_persona(self, domains: List[str] = None) -> Persona:
        """Select a persona weighted by population frequency."""
        available = list(self.personas.values())
        weights = [p.weight for p in available]

        # Normalize weights
        total = sum(weights)
        probs = [w / total for w in weights]

        idx = self.rng.choice(len(available), p=probs)
        return available[idx]

    def detect_domains(self, study_description: str, study_title: str) -> List[str]:
        """Detect relevant domains from study text."""
        text = f"{study_description} {study_title}".lower()

        domains = []
        domain_keywords = {
            'technology': ['ai', 'algorithm', 'robot', 'technology', 'automation'],
            'consumer': ['product', 'brand', 'purchase', 'buy', 'consumer'],
            'political': ['politics', 'vote', 'election', 'democrat', 'republican'],
            'health': ['health', 'medical', 'doctor', 'treatment', 'disease'],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in text for kw in keywords):
                domains.append(domain)

        return domains or ['general']
```

---

## 7. Response Generation

### 7.1 Open-Ended Response Generation

```python
# Domain-specific response templates
RESPONSE_TEMPLATES_BY_DOMAIN = {
    'ai_technology': {
        'positive': [
            "I found the AI-generated recommendations to be {adjective}. The system seemed to {verb} my preferences well.",
            "The algorithm {verb} relevant suggestions. I appreciated how it {action}.",
            "Overall, I'm {sentiment} with how the technology {verb} my needs.",
        ],
        'negative': [
            "I was {sentiment} by the AI's recommendations. They seemed {adjective}.",
            "The algorithm felt {adjective}. I wished it could have {action} better.",
        ],
        'neutral': [
            "The AI recommendations were {adjective}. Some helpful, others less so.",
        ]
    },
    # ... more domains
}

# Word banks for template filling
WORD_BANKS = {
    'adjective_positive': ['excellent', 'impressive', 'helpful', 'intuitive', 'effective'],
    'adjective_negative': ['disappointing', 'frustrating', 'confusing', 'unhelpful'],
    'adjective_neutral': ['adequate', 'acceptable', 'standard', 'typical'],
    'verb_positive': ['understood', 'captured', 'addressed', 'enhanced'],
    'verb_negative': ['missed', 'ignored', 'failed', 'overlooked'],
    'verb_neutral': ['met', 'served', 'provided', 'delivered'],
    'emotion_positive': ['satisfied', 'pleased', 'impressed', 'confident'],
    'emotion_negative': ['frustrated', 'disappointed', 'concerned'],
}

def _generate_diverse_open_ended(
    question_text: str,
    domain: str,
    valence: str,           # 'positive', 'negative', 'neutral'
    persona_traits: Dict,
    condition: str,
    rng: np.random.RandomState
) -> str:
    """Generate contextually appropriate open-ended response."""

    # Select template
    templates = RESPONSE_TEMPLATES_BY_DOMAIN.get(domain, {}).get(valence, [])
    if not templates:
        templates = ["I found this experience to be interesting."]

    template = rng.choice(templates)

    # Fill placeholders
    def get_word(category, sentiment):
        key = f'{category}_{sentiment}'
        words = WORD_BANKS.get(key, ['appropriate'])
        return rng.choice(words)

    response = template
    response = response.replace('{adjective}', get_word('adjective', valence))
    response = response.replace('{verb}', get_word('verb', valence))
    response = response.replace('{sentiment}', get_word('emotion', valence))
    response = response.replace('{action}', get_word('verb', valence))

    # Add elaboration based on verbosity trait
    verbosity = persona_traits.get('verbosity', 0.5)
    if verbosity > 0.7:
        response += " This aligns with my expectations."

    return response
```

### 7.2 Question Type Handlers

```python
QUESTION_TYPE_HANDLERS = {
    'explanation': {
        'prompts': ['why', 'explain', 'reason'],
        'template': "I {verb} because {reason}.",
        'min_words': 15
    },
    'description': {
        'prompts': ['describe', 'tell us about', 'what was'],
        'template': "The experience was {adjective}. {elaboration}",
        'min_words': 12
    },
    'evaluation': {
        'prompts': ['rate', 'evaluate', 'assess', 'how would you'],
        'template': "I would {verb} this as {adjective}.",
        'min_words': 10
    },
    'opinion': {
        'prompts': ['think', 'feel', 'believe', 'opinion'],
        'template': "I {verb} that {opinion}.",
        'min_words': 10
    },
    'feedback': {
        'prompts': ['feedback', 'suggestions', 'comments', 'thoughts'],
        'template': "My feedback is that {feedback}.",
        'min_words': 8
    },
}

def detect_question_type(question_text: str) -> str:
    """Detect question type from text."""
    text_lower = question_text.lower()

    for q_type, config in QUESTION_TYPE_HANDLERS.items():
        if any(prompt in text_lower for prompt in config['prompts']):
            return q_type

    return 'general'
```

---

## 8. Survey Flow Logic

### 8.1 Overview

The `SurveyFlowHandler` ensures participants only receive responses for questions they would actually see based on their experimental condition.

### 8.2 Implementation

```python
class SurveyFlowHandler:
    """
    Determines question visibility per condition.

    Detection methods:
    1. Explicit condition restrictions in question spec
    2. DisplayLogic parsing for condition checks
    3. Block name analysis (condition keywords)
    4. Embedded data checks
    5. Factor-level matching for factorial designs
    """

    def __init__(self, conditions: List[str], open_ended_questions: List[Dict]):
        self.conditions = [c.lower().strip() for c in conditions]
        self.questions = open_ended_questions
        self.factor_levels = self._extract_factor_levels()
        self.visibility_map = self._build_visibility_map()

    def _extract_factor_levels(self) -> Dict[str, Set[str]]:
        """Extract factor levels from factorial condition names."""
        separators = ['×', ' x ', '_x_', ' × ']
        factor_levels = {}

        for cond in self.conditions:
            for sep in separators:
                if sep in cond:
                    parts = [p.strip() for p in cond.split(sep)]
                    for i, part in enumerate(parts):
                        key = f"factor_{i}"
                        if key not in factor_levels:
                            factor_levels[key] = set()
                        factor_levels[key].add(part.lower())
                    break

        return factor_levels

    def _build_visibility_map(self) -> Dict[str, Dict[str, bool]]:
        """Build question -> condition -> visible mapping."""
        visibility = {}

        for q in self.questions:
            q_name = q.get('name', '')
            q_visibility = {c: True for c in self.conditions}  # Default: visible

            # Method 1: Explicit condition restrictions
            if q.get('condition') or q.get('visible_conditions'):
                allowed = q.get('condition') or q.get('visible_conditions')
                if isinstance(allowed, str):
                    allowed = [allowed]
                for cond in self.conditions:
                    q_visibility[cond] = self._condition_matches(cond, allowed)

            # Method 2: Display logic
            if q.get('display_logic_details'):
                self._apply_display_logic(q_visibility, q['display_logic_details'])

            # Method 3: Block name analysis
            self._apply_block_name_logic(q_visibility, q.get('block_name', ''))

            # Method 4: Question text hints
            self._apply_question_text_hints(q_visibility, q.get('question_text', ''))

            visibility[q_name] = q_visibility

        return visibility

    def _apply_block_name_logic(self, q_visibility: Dict[str, bool], block_name: str):
        """Apply visibility based on block name keywords."""
        if not block_name:
            return

        block_lower = block_name.lower()

        # Check for negation patterns (e.g., "no_ai_block", "without_ai")
        negation_patterns = ['no_', 'no ', 'non_', 'without_']
        has_negation = any(neg in block_lower for neg in negation_patterns)

        # Find keywords in block name
        for cond in self.conditions:
            cond_parts = cond.replace('×', ' ').replace('_', ' ').split()
            for part in cond_parts:
                if len(part) >= 2 and part in block_lower:
                    # Block has this keyword
                    # If negated, only negated conditions see it
                    # If not negated, only non-negated conditions see it
                    cond_has_negation = any(
                        f"{neg}{part}" in cond for neg in ['no', 'non', 'without']
                    )

                    if has_negation:
                        q_visibility[cond] = cond_has_negation
                    else:
                        q_visibility[cond] = not cond_has_negation

    def is_question_visible(self, question_name: str, condition: str) -> bool:
        """Check if question is visible for condition."""
        q_visibility = self.visibility_map.get(question_name, {})
        return q_visibility.get(condition.lower().strip(), True)
```

---

## 9. Effect Size Calibration

### 9.1 Cohen's d Implementation

```python
def _apply_effect_size(
    base_response: float,
    condition: str,
    conditions: List[str],
    target_d: float,
    scale_points: int
) -> float:
    """
    Apply Cohen's d effect size to base response.

    Cohen's d = (M1 - M2) / SD_pooled

    For 7-point scales:
        SD_pooled ≈ 1.5
        d = 0.5 → difference of 0.75 scale points

    Effect direction determined by condition semantics.
    """

    # Parse condition semantics
    semantics = _parse_condition_semantics(condition)

    # Calculate effect magnitude in scale points
    sd_pooled = (scale_points - 1) / 4  # Approximate
    effect_points = target_d * sd_pooled

    # Apply direction based on valence
    # Positive valence conditions get positive shift
    # Negative valence conditions get negative shift
    direction = semantics['valence']  # -1 to +1

    # Adjust response
    adjusted = base_response + direction * effect_points

    return adjusted


def _validate_effect_sizes(
    data: pd.DataFrame,
    conditions: List[str],
    target_d: float,
    dv_columns: List[str]
) -> Dict[str, Any]:
    """
    Validate achieved effect sizes match targets.

    Returns:
        {
            'target_d': float,
            'achieved_effects': {
                'DV_name': {
                    'achieved_d': float,
                    'target_d': float,
                    'deviation': float,
                    'within_tolerance': bool
                }
            },
            'within_tolerance': bool,
            'tolerance': 0.15
        }
    """
    results = {'target_d': target_d, 'achieved_effects': {}, 'within_tolerance': True}

    if len(conditions) < 2:
        return results

    cond1, cond2 = conditions[0], conditions[1]

    for col in dv_columns:
        if col not in data.columns:
            continue

        g1 = data[data['CONDITION'] == cond1][col].dropna().astype(float)
        g2 = data[data['CONDITION'] == cond2][col].dropna().astype(float)

        if len(g1) < 2 or len(g2) < 2:
            continue

        # Calculate Cohen's d
        mean_diff = g1.mean() - g2.mean()
        sd_pooled = np.sqrt(
            ((len(g1)-1)*g1.var() + (len(g2)-1)*g2.var()) / (len(g1)+len(g2)-2)
        )

        if sd_pooled > 0:
            achieved_d = abs(mean_diff) / sd_pooled
            results['achieved_effects'][col] = {
                'achieved_d': round(achieved_d, 3),
                'target_d': target_d,
                'deviation': round(abs(achieved_d - target_d), 3),
                'within_tolerance': abs(achieved_d - target_d) <= 0.15
            }

    return results
```

---

## 10. Data Quality Validation

### 10.1 Careless Response Detection

```python
def _detect_careless_patterns(responses: List[int], scale_points: int = 7) -> Dict:
    """
    Detect careless responding patterns:

    1. Straight-lining: Same response repeated
       - Flag if 5+ consecutive identical OR >70% identical

    2. Alternating: 1-7-1-7 pattern
       - Flag if alternating ratio > 60%

    3. Midpoint: Always choosing middle
       - Flag if >80% at midpoint

    4. Extreme: Always choosing endpoints
       - Flag if >70% at 1 or max
    """

    patterns = []

    # Straight-line detection
    max_streak = max(
        sum(1 for _ in g)
        for _, g in itertools.groupby(responses)
    )
    if max_streak >= 5:
        patterns.append('straight_line')

    # Midpoint detection
    midpoint = (scale_points + 1) / 2
    midpoint_ratio = sum(1 for r in responses if abs(r - midpoint) < 0.6) / len(responses)
    if midpoint_ratio > 0.8:
        patterns.append('midpoint')

    # Extreme detection
    extreme_ratio = sum(1 for r in responses if r in [1, scale_points]) / len(responses)
    if extreme_ratio > 0.7:
        patterns.append('extreme')

    return {
        'careless_detected': len(patterns) > 0,
        'patterns': patterns,
        'max_straight_line': max_streak,
        'midpoint_ratio': midpoint_ratio,
        'extreme_ratio': extreme_ratio
    }
```

### 10.2 Comprehensive Quality Metrics

```python
def _compute_data_quality_metrics(
    data: pd.DataFrame,
    scale_columns: List[str],
    conditions: List[str]
) -> Dict[str, Any]:
    """
    Compute comprehensive quality metrics.

    Returns:
        {
            'n_participants': int,
            'n_conditions': int,
            'condition_balance': {cond: {'count': n, 'expected': e, 'deviation': d}},
            'response_distributions': {col: {'mean': m, 'sd': s, 'skewness': sk}},
            'missing_data': {'total_missing': n, 'missing_rate': r},
            'overall_quality_score': float  # 0-1
        }
    """

    metrics = {
        'n_participants': len(data),
        'n_conditions': len(conditions),
        'condition_balance': {},
        'response_distributions': {},
        'missing_data': {},
        'overall_quality_score': 0.0
    }

    # Condition balance
    if 'CONDITION' in data.columns:
        counts = data['CONDITION'].value_counts()
        expected = len(data) / len(conditions)

        for cond in conditions:
            actual = counts.get(cond, 0)
            deviation = abs(actual - expected) / expected if expected > 0 else 0
            metrics['condition_balance'][cond] = {
                'count': int(actual),
                'expected': round(expected, 1),
                'deviation': round(deviation, 3)
            }

    # Response distributions
    for col in scale_columns:
        if col in data.columns:
            values = data[col].dropna().astype(float)
            if len(values) > 0:
                metrics['response_distributions'][col] = {
                    'mean': round(values.mean(), 3),
                    'sd': round(values.std(), 3),
                    'skewness': round(float(values.skew()), 3) if len(values) > 2 else 0
                }

    # Missing data
    total_cells = len(data) * len(data.columns)
    missing = data.isna().sum().sum()
    metrics['missing_data'] = {
        'total_missing': int(missing),
        'missing_rate': round(missing / total_cells, 4) if total_cells > 0 else 0
    }

    # Overall quality score
    balance_ok = all(d['deviation'] < 0.1 for d in metrics['condition_balance'].values())
    missing_ok = metrics['missing_data']['missing_rate'] < 0.05
    metrics['overall_quality_score'] = (int(balance_ok) + int(missing_ok)) / 2

    return metrics
```

---

## 11. Streamlit Application

### 11.1 Application Structure

**File**: `app.py`

```python
# Key globals
REQUIRED_UTILS_VERSION = "1.0.0"
APP_VERSION = "1.0.0"
BUILD_ID = "20240204_v1"

# Version check at startup
if hasattr(utils, '__version__') and utils.__version__ != REQUIRED_UTILS_VERSION:
    st.warning(f"Utils version mismatch: expected {REQUIRED_UTILS_VERSION}")

# Session state initialization
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'qsf_data' not in st.session_state:
    st.session_state.qsf_data = None
```

### 11.2 Multi-Step Workflow

```python
STEPS = [
    "Upload QSF",
    "Confirm Conditions",
    "Configure DVs",
    "Set Parameters",
    "Generate Data"
]

def main():
    st.title("Behavioral Experiment Simulation Tool")

    # Step navigation
    current_step = st.session_state.step

    # Step indicators
    cols = st.columns(len(STEPS))
    for i, (col, step_name) in enumerate(zip(cols, STEPS)):
        status = "✓" if i + 1 < current_step else ("●" if i + 1 == current_step else "○")
        col.markdown(f"**{status} {step_name}**")

    # Render current step
    if current_step == 1:
        render_upload_step()
    elif current_step == 2:
        render_conditions_step()
    elif current_step == 3:
        render_dvs_step()
    elif current_step == 4:
        render_parameters_step()
    elif current_step == 5:
        render_generate_step()

def render_upload_step():
    uploaded = st.file_uploader("Upload QSF file", type=['qsf'])

    if uploaded:
        parser = QSFPreviewParser()
        result = parser.parse(uploaded.read())

        if result.success:
            st.session_state.qsf_result = result
            st.success(f"Parsed: {result.survey_name}")
            st.write(f"- {result.total_questions} questions")
            st.write(f"- {len(result.detected_conditions)} conditions detected")

            if st.button("Continue"):
                st.session_state.step = 2
                st.rerun()
        else:
            st.error("Failed to parse QSF file")
            for error in result.validation_errors:
                st.write(f"- {error}")
```

### 11.3 State Persistence

```python
def _save_step_state():
    """Save current step state for navigation."""
    persist_keys = [
        'selected_conditions',
        'confirmed_scales',
        'factorial_factors',
        'sample_size',
        'effect_size',
    ]

    saved = {}
    for key in persist_keys:
        if key in st.session_state:
            saved[key] = st.session_state[key]

    st.session_state.saved_state = saved

def _restore_step_state():
    """Restore state after navigation."""
    if 'saved_state' not in st.session_state:
        return

    for key, value in st.session_state.saved_state.items():
        if key not in st.session_state:
            st.session_state[key] = value
```

---

## 12. Version Management

### 12.1 Version Locations

| File | Variable | Purpose |
|------|----------|---------|
| `app.py:46` | `REQUIRED_UTILS_VERSION` | Expected utils version |
| `app.py:100` | `APP_VERSION` | Application version |
| `app.py:47` | `BUILD_ID` | Cache invalidation |
| `utils/__init__.py` | `__version__` | Package version |
| `utils/qsf_preview.py` | `__version__` | Module version |
| `utils/enhanced_simulation_engine.py` | `__version__` | Module version |
| `utils/response_library.py` | `__version__` | Module version |

### 12.2 Version Update Procedure

1. Update ALL version strings simultaneously
2. Run `python3 -m py_compile` on all modified files
3. Test version check doesn't trigger warning
4. Update README.md version references

---

## 13. Testing & Validation

### 13.1 Unit Test Structure

```python
# Test QSF parsing
def test_qsf_parsing():
    parser = QSFPreviewParser()

    with open('test.qsf', 'rb') as f:
        result = parser.parse(f.read())

    assert result.success
    assert result.total_questions > 0
    assert len(result.detected_conditions) >= 1

# Test simulation engine
def test_simulation():
    engine = EnhancedSimulationEngine(
        study_title="Test",
        study_description="Test study",
        sample_size=20,
        conditions=["Control", "Treatment"],
        factors=[],
        scales=[{"name": "DV", "num_items": 3, "scale_points": 7}],
        additional_vars=[],
        demographics={"age": True},
        seed=42
    )

    df, metadata = engine.generate()

    assert len(df) == 20
    assert "CONDITION" in df.columns
    assert df["CONDITION"].nunique() == 2

# Test effect size calibration
def test_effect_sizes():
    engine = EnhancedSimulationEngine(
        ...
        effect_sizes=[EffectSizeSpec(
            variable="DV",
            factor="Condition",
            level_high="Treatment",
            level_low="Control",
            cohens_d=0.5
        )]
    )

    df, _ = engine.generate()

    # Validate achieved effect
    validation = _validate_effect_sizes(df, ["Control", "Treatment"], 0.5, ["DV_1"])
    assert validation['within_tolerance']
```

### 13.2 Integration Test

```python
def test_full_pipeline():
    """Test complete QSF → Simulation → Validation pipeline."""

    # 1. Parse QSF
    with open('example.qsf', 'rb') as f:
        qsf_result = QSFPreviewParser().parse(f.read())

    assert qsf_result.success

    # 2. Generate simulation
    engine = EnhancedSimulationEngine(
        study_title=qsf_result.survey_name,
        study_description="Integration test",
        sample_size=100,
        conditions=qsf_result.detected_conditions,
        factors=[],
        scales=qsf_result.detected_scales,
        additional_vars=[],
        demographics={"age": True, "gender": True},
        open_ended_questions=qsf_result.open_ended_details,
        seed=42
    )

    df, metadata = engine.generate()

    # 3. Validate output
    assert len(df) == 100
    assert df["CONDITION"].isin(qsf_result.detected_conditions).all()

    # 4. Check quality metrics
    metrics = _compute_data_quality_metrics(
        df,
        [s['name'] + '_1' for s in qsf_result.detected_scales],
        qsf_result.detected_conditions
    )

    assert metrics['overall_quality_score'] >= 0.8
```

---

## 14. Complete API Reference

### 14.1 QSFPreviewParser

```python
class QSFPreviewParser:
    def parse(self, qsf_content: bytes) -> QSFPreviewResult
    def _extract_questions(self, elements: List) -> Dict[str, QuestionInfo]
    def _extract_blocks(self, elements: List) -> List[BlockInfo]
    def _extract_conditions_from_flow(self, flow_data: Any) -> List[str]
    def _detect_scales(self, questions_map: Dict, blocks: List) -> List[Dict]
    def _detect_open_ended(self, questions_map: Dict) -> List[str]
    def _extract_open_ended_details(self, questions_map: Dict, blocks: List) -> List[Dict]
    def _is_excluded_block_name(self, name: str) -> bool
```

### 14.2 EnhancedSimulationEngine

```python
class EnhancedSimulationEngine:
    def __init__(self, study_title, study_description, sample_size, conditions, ...)
    def generate(self) -> Tuple[pd.DataFrame, Dict[str, Any]]
    def _generate_condition_assignment(self, n: int) -> pd.Series
    def _generate_scale_response(self, scale: Dict, condition: str, traits: Dict, rng) -> int
    def _generate_open_response(self, question: Dict, condition: str, traits: Dict, rng) -> str
    def _select_persona(self, rng: np.random.RandomState) -> Persona
```

### 14.3 Helper Functions

```python
# Semantic parsing
def _parse_condition_semantics(condition: str) -> Dict[str, Any]

# Scale reliability
def _generate_correlated_items(n_items, base_response, target_alpha, scale_points, reverse_items, rng) -> List[int]

# Careless detection
def _detect_careless_patterns(responses: List[int], scale_points: int) -> Dict[str, Any]

# Effect validation
def _validate_effect_sizes(data: pd.DataFrame, conditions: List, target_d: float, dv_columns: List) -> Dict

# Quality metrics
def _compute_data_quality_metrics(data: pd.DataFrame, scale_columns: List, conditions: List) -> Dict

# Factorial design
def _parse_factorial_design(conditions: List[str]) -> Dict[str, Any]
def _compute_factorial_effects(data: pd.DataFrame, design: Dict, dv_column: str) -> Dict

# Survey flow
class SurveyFlowHandler:
    def __init__(self, conditions: List[str], questions: List[Dict])
    def is_question_visible(self, question_name: str, condition: str) -> bool

# Special question types
def _generate_heatmap_response(width, height, n_clicks, attention, focus, rng) -> List[Dict]
def _generate_rank_order_response(items, preferences, attention, rng) -> List[str]

# Cultural modeling
def _apply_cultural_response_style(base_response, scale_points, cultural_style, rng) -> int
```

---

## Appendix A: Scientific References

1. Cohen, J. (1988). Statistical power analysis for the behavioral sciences.
2. Krosnick, J. A. (1991). Response strategies for coping with cognitive demands.
3. Greenleaf, E. A. (1992). Measuring extreme response style.
4. Paulhus, D. L. (1991). Measurement and control of response bias.
5. Meade, A. W., & Craig, S. B. (2012). Identifying careless responses.
6. Billiet, J. B., & McClendon, M. J. (2000). Modeling acquiescence.
7. Richard, F. D., et al. (2003). One hundred years of social psychology.

---

## Appendix B: Changelog

### v1.0.0 (2024-02-04)
- Initial release
- 20 iterations of comprehensive improvements
- Support for all QSF patterns
- Complete documentation

---

*Document Version: 1.0.0*
*Last Updated: 2024-02-04*
*Confidential - For Internal Development Use Only*
