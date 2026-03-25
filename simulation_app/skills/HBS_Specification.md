Here's the plain-language summary of the 5 most impactful HBS aspects for the tile, followed by the full MD for Claude Code.

---

**What makes HBS worth its own tile (the 5 things that matter most for copy):**

1. **DANEEL-grade persona coherence** — Every simulated participant is demographically profiled (age, education, party ID, ZIP, ideology) and stays internally consistent across every question, every page, every response — exactly like DANEEL's cross-page memory system.
2. **Programmatic tool-use (DANEEL+)** — A Python tool layer handles the tasks LLMs systematically fail at: image geometry for visual illusions, character-counting, arithmetic, real-time clock. This is what makes DANEEL+ pass 81/81 bot traps.
3. **Calibrated human errors** — Wrong answers are drawn from empirical error distributions by education level, not random noise. A HS grad fails the bat-and-ball problem at the documented 82% rate. This is what makes populations statistically match real data.
4. **Stylometric voice fingerprinting** — Each participant gets a writing DNA (sentence length, vocabulary richness, typo signature, filler words) that locks all their open-ended responses to sound like the same real person.
5. **Self-validating output** — After generation, HBS scores the dataset against the full DANEEL benchmark battery (timing, TIPI variance, IAT d-scores, attention rates, OE coherence) and auto-corrects any metric outside the human-plausible range before delivery.

---

```markdown
# HBS Tile Replacement — Claude Code Instructions

## Context

The generation method chooser in `simulation_app/app.py` currently shows 4 method cards.
**Option 3** is the one being replaced:

**CURRENT (to remove):**
- Label: `Built-in Template Engine`
- Tag: `(no API needed)`  
- Description: `225+ domain templates, 58+ personas, no API needed`
- Tooltip/subtext: Template-based fallback system for open-ended responses

**NEW (to implement):**
- Label: `Human Behavior Simulator (HBS)`
- Tag: `(most realistic)`
- Description and tooltip content: see §2 below

The rest of the UI — card layout, styling, selection behavior, session_state keys for the other 3 methods — must remain completely unchanged.

---

## 1. Locate the Existing Tile

Search `app.py` for the string `"Built-in Template Engine"` or `"no API needed"` to find the exact card definition block. The 4-card chooser was introduced in the v1.1.0.x series and uses a custom card UI with an icon, title, tag badge, and description string. Match that exact structure.

Do NOT use a different layout, different widget type, or different number of cards. Replace the content inside the existing card 3 slot only.

---

## 2. New Tile Content — Exact Copy

### Card (compact, in-chooser display)

```
Icon:        ??   (brain emoji, or use existing icon system — match the style of other 3 cards)
Title:       Human Behavior Simulator (HBS)
Badge tag:   most realistic
Description: DANEEL-grade persona coherence, calibrated human error rates, and
             stylometric voice fingerprinting — no API needed for core simulation.
```

### Expandable Tooltip / Info Section (matches the existing expandable info pattern for other cards)

```
Title: Human Behavior Simulator (HBS)

What it does:
HBS combines the existing simulation engines with the full architecture of DANEEL and
DANEEL+ — the autonomous survey bots from the 2025 PNAS and 2026 Nature papers on
AI survey fraud — to generate synthetic behavioral data that passes every known
human-detection benchmark.

Key capabilities:

• Demographic persona coherence
  Each participant carries a full U.S.-census-weighted profile: age, education,
  income, party ID, ideology, state, ZIP. Every response — numeric and text — stays
  consistent with that profile across all questions, exactly like DANEEL's cross-page
  memory system.

• Programmatic tool-use (DANEEL+ architecture)
  A Python tool layer handles the tasks LLMs systematically fail at: pixel-area
  comparison for visual illusions, character-level counting, arithmetic, and
  real-time clock queries. This is the mechanism that gives DANEEL+ a perfect 81/81
  score on bot-trap batteries.

• Calibrated human error distributions
  Wrong answers are drawn from empirically documented error rates by education level,
  not random noise. The bat-and-ball problem fails at the correct 82% rate for
  high-school graduates. Chess game count fails at 100% — because knowing the exact
  answer (120 zeros) is itself a bot signal.

• Stylometric voice fingerprinting
  Each participant receives a writing fingerprint at construction: vocabulary richness,
  sentence length distribution, punctuation habits, filler-word rate, typo signature.
  Every open-ended response is post-processed to match that fingerprint — so all
  responses from the same participant sound like the same real person.

• Adversarial self-validation
  After generation, HBS scores the full dataset against the DANEEL benchmark battery:
  survey completion timing (target: 80% of participants in 10–18 min), TIPI
  personality variance (SD ? 0.8 per participant), IAT d-score validity, attention
  check pass rates (82–96%), and open-ended coherence. Any metric outside the
  human-plausible range is auto-corrected before the file is returned to you.

• True survey flow simulation
  Each participant's prior answers determine which questions they would see via
  skip logic and display logic. The output CSV has the same structural missing-data
  pattern as a real Qualtrics export — not uniform missingness, but
  condition-dependent missingness that mirrors the survey's actual logic tree.

Best for:
  Pre-registration validation, adversarial pilot testing, bot-detection research,
  power analysis requiring population-level distributional accuracy.

No API required for core demographic and behavioral simulation.
AI providers are used for open-ended responses and follow the same
multi-provider failover chain as the Built-in AI method.
```

---

## 3. Session State Key

The existing method chooser stores the selected method as a string in `st.session_state`.
Find the current key name (likely something like `generation_method` or `selected_method`).

The value string for the new tile must be:

```python
"hbs"   # or match whatever snake_case convention the other 3 values use
```

Ensure the generation pipeline in `enhanced_simulation_engine.py` routes
`generation_method == "hbs"` to the HBS engine (see HBS_SPEC.md for full routing logic).
If HBS is not yet implemented, route to the existing Adaptive Behavioral Engine as a
temporary stub — clearly comment this with `# TODO: route to HBS once implemented`.

---

## 4. Method Chooser Order (do not reorder other methods)

```
Card 1 (unchanged): Built-in AI — Recommended
Card 2 (unchanged): Your Own API Key
Card 3 (REPLACE):   Human Behavior Simulator (HBS) — most realistic   ? THIS ONE
Card 4 (unchanged): Adaptive Behavioral Engine (Beta)
```

---

## 5. Version Requirements

After making this change, update ALL 9 version locations per CLAUDE.md.
Increment the last version digit by 1.
Update BUILD_ID with today's date and description: `"YYYYMMDD-vXXXXX-hbs-tile-replace"`

Commit message format:
```
vX.X.X.X: Replace Template Engine tile with Human Behavior Simulator (HBS)
- Swap card 3 content: title, badge, description, tooltip
- Route generation_method="hbs" to HBS engine (stub to Adaptive if not yet built)
- Version sync: all 9 locations updated
```

---

## 6. What NOT to Change

- Do not alter the landing page feature tiles (Test Before You Collect, Realistic Open-Ended Responses, Ready-to-Run Analysis Code, Built for Research & Teaching).
- Do not rename or remove the Template Engine from the fallback cascade inside `enhanced_simulation_engine.py` — it is still Tier 3 in the response generation pipeline.
- Do not change the card layout, CSS, or icon system for the other 3 method cards.
- Do not add a 5th card. This is a direct replacement, not an addition.
- Follow AGENTS.md: submit a conflict-free, mergeable PR. Do not touch CHANGELOG.md, README.md, or MEMORY.md unless explicitly required by the version sync protocol.
```


---

```markdown
# Human Behavior Simulator (HBS) — Complete Claude Code Specification v2.0
# Incorporating DANEEL + DANEEL+ Architecture + 5 Critical Enhancements

**Target repo:** `eugendimant/research-simulations`
**Target path:** `simulation_app/` (extend existing Streamlit app)
**Versioning:** Mandatory 9-location version sync per `CLAUDE.md` — every commit
**Workflow:** `AGENTS.md` — all PRs must be conflict-free and mergeable before push
**Complexity level:** COMPLEX — 5 iteration loops mandatory per `CLAUDE.md` protocol

---

## PART 0 — GOVERNING DESIGN PHILOSOPHY

HBS is built on three identical pillars to DANEEL:

> **Behavioral realism** — Responses fall within the normal distribution of human variation. Parameters drawn from Gaussian and weighted distributions, never uniform random. Perfection is a detection signal.

> **Cognitive realism** — Each participant inhabits a demographic persona with consistent views, education-matched writing, realistic knowledge limits, and cross-item coherence. The AI does not answer questions; it *is* the respondent.

> **Structural realism** — The output CSV is structurally identical to what Qualtrics exports from real participants: correct missing data patterns, correct timing metadata, correct question ordering per each participant's flow path.

**Anti-pattern above all others:** Never generate "good" responses. Generate *human* responses. A simulated dataset where every respondent passes every attention check, answers every logic question correctly, and writes fluent OE text is not realistic — it is detectably synthetic. Imperfection must be deliberate, calibrated, and demographically grounded.

---

## PART 1 — COMPLETE FILE STRUCTURE

```
research-simulations/
??? simulation_app/
    ??? app.py                              # Streamlit entry — add HBS as 5th method
    ??? utils/
    ?   ??? __init__.py                     # Version string — 9-location sync
    ?   ??? enhanced_simulation_engine.py   # Orchestrator — extend only, never rewrite
    ?   ??? persona_library.py              # Extend: add DANEEL demographic fields
    ?   ??? response_library.py             # Keep: template fallback (Tier 3)
    ?   ??? llm_response_generator.py       # Extend: add HBS prompt template + refusal recovery
    ?   ??? qsf_preview.py                  # Extend: add full flow graph extraction
    ?   ?
    ?   ??? hbs_participant_state.py        # NEW: ParticipantState dataclass (central data model)
    ?   ??? hbs_response_engine.py          # NEW: Unified 3-tier response routing
    ?   ??? hbs_tool_dispatcher.py          # NEW: DANEEL+ programmatic tool-use layer [IMP 1]
    ?   ??? hbs_error_calibrator.py         # NEW: Distributional human-error tables [IMP 2]
    ?   ??? hbs_stylometric_engine.py       # NEW: Per-participant voice fingerprint [IMP 3]
    ?   ??? hbs_validator.py                # NEW: Adversarial self-validation pipeline [IMP 4]
    ?   ??? hbs_flow_simulator.py           # NEW: Survey flow + skip logic engine [IMP 5]
    ?   ??? hbs_behavioral_realism.py       # NEW: Timing, typing, reading metadata
    ?   ??? hbs_knowledge_limiter.py        # NEW: Education-gated knowledge caps
    ?   ??? hbs_coherence_enforcer.py       # NEW: Cross-item consistency + ECLAIR rules
    ?   ??? hbs_question_classifier.py      # NEW: Full DANEEL question type taxonomy
    ?   ??? hbs_trap_handler.py             # NEW: Attention check + bot-trap logic
    ??? data/
    ?   ??? hbs_zip_table.json              # NEW: Valid ZIPs per state (?20 per state)
    ?   ??? hbs_sports_lookup.json          # NEW: Actual sports results 2024–2026
    ?   ??? hbs_weather_stubs.json          # NEW: State-level weather baseline data
    ?   ??? hbs_error_calibration.json      # NEW: Question×demographic?error tables [IMP 2]
    ?   ??? hbs_qwerty_adjacency.json       # NEW: QWERTY key adjacency map for typo model
    ??? skills/
        ??? HBS_SPEC.md                     # This file
```

---

## PART 2 — CENTRAL DATA MODEL (`hbs_participant_state.py`)

This is the single source of truth for every simulated participant. Every module reads from and writes to this dataclass. No module should maintain its own participant state.

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ParticipantState:
    # ?? Identity ??????????????????????????????????????????????????????????????
    participant_id: str                  # UUID4
    persona_id: str                      # response style label (see §3.2)
    response_style: str                  # Krosnick taxonomy

    # ?? DANEEL Demographic Profile ????????????????????????????????????????????
    age: int                             # specific year, e.g. 34
    birth_year: int                      # derived; 1924–2006; IMMUTABLE once set
    education_level: str                 # "<HS" | "HS" | "Some college" | "BA" | "Graduate"
    income_bracket: str                  # "<30k" | "30-60k" | "60-100k" | "100k+"
    gender: str
    race_ethnicity: str
    party_id: str                        # 7-point: "Strong Dem" … "Strong Rep"
    ideology: float                      # -3.0 (very liberal) to +3.0 (very conservative)
    state: str                           # 2-letter code
    zip_code: str                        # valid USPS zip in state; IMMUTABLE once set
    region: str                          # "Northeast"|"South"|"Midwest"|"West"
    media_market: str                    # nearest DMA market name
    religious_affiliation: Optional[str] # None | "None" | "Protestant" | "Catholic" | etc.
    news_consumption: str                # "TV" | "Online" | "Radio" | "None"

    # ?? DANEEL+ Programmatic Tool Flags ??????????????????????????????????????
    tool_access_enabled: bool = True     # can invoke HBSTool dispatcher
    tool_call_log: list = field(default_factory=list)  # audit of tool invocations

    # ?? Behavioral Parameters (drawn ONCE at construction, immutable) ?????????
    typing_speed_wpm: float              # N(55, 15), clip [20, 110]
    reading_speed_wpm: float             # N(238, 60), clip [100, 450]
    mouse_accuracy: float                # U[0.6, 1.0]
    attention_level: float               # drives check pass rate; drawn from Beta(5,2) or Beta(2,5)
    inter_key_interval_ms: float         # derived from typing_speed; consistent throughout
    target_completion_seconds: float     # N(840, 120), clip [480, 1500]

    # ?? Cognitive Style ???????????????????????????????????????????????????????
    knowledge_cap: dict = field(default_factory=dict)   # topic ? can_answer_correctly bool
    translation_capable: bool = False
    uses_web_search: bool = False        # always False for survey context

    # ?? Stylometric Fingerprint (built from first OE response) ???????????????
    # See §6 (Improvement 3)
    vocab_richness: Optional[float] = None    # type-token ratio; set after first OE
    mean_sentence_len: Optional[float] = None # words per sentence
    punct_rate: Optional[float] = None        # punctuation marks per 100 words
    filler_word_rate: Optional[float] = None  # "um", "like", "you know" per 100 words
    liwc_tone: Optional[str] = None           # "positive" | "negative" | "analytical" | "tentative"
    capitalization_style: str = "normal"      # "normal" | "all_lower" | "inconsistent"
    typo_signature: list = field(default_factory=list)  # recurring QWERTY errors

    # ?? Survey State (grows as survey progresses) ?????????????????????????????
    prior_responses: dict = field(default_factory=dict)   # {question_id: answer}
    questions_seen: list = field(default_factory=list)     # ordered list of q_ids seen
    voice_memory: list = field(default_factory=list)       # prior OE text excerpts
    established_political_lean: Optional[str] = None       # set on first political question
    established_birth_year_digit: Optional[int] = None     # last digit of birth year
    dollar_bill_serial: Optional[str] = None               # ECLAIR consistency anchor
    prolific_id: Optional[str] = None                      # generated 16-digit string; immutable
    survey_url: Optional[str] = None                       # generated Qualtrics URL; immutable

    # ?? Behavioral Flags ?????????????????????????????????????????????????????
    is_straight_liner: bool = False      # straight-lines scale items
    is_satisficer: bool = False
    is_careless: bool = False
    failed_attention_checks: int = 0
    survey_start_time: float = 0.0       # unix timestamp

    # ?? Validation Scores (populated by HBSValidator post-generation) ?????????
    tipi_sd: Optional[float] = None      # SD across 10 TIPI items; must be ? 0.8
    iat_d_score: Optional[float] = None
    completion_time_seconds: Optional[float] = None
    oe_mean_word_count: Optional[float] = None
    validation_passed: bool = False
    validation_flags: list = field(default_factory=list)
```

**Construction rule:** `HBSParticipantFactory.create(condition, domain, sample_idx)` draws all behavioral parameters from their distributions at creation time and sets them immutably. Parameters are NEVER re-drawn mid-survey.

---

## PART 3 — PROFILE ENGINE (extends `persona_library.py`)

### 3.1 Demographic Sampling

`_sample_demographics(n: int) -> list[dict]` uses census-weighted U.S.-adult distributions:

| Dimension | Distribution |
|---|---|
| Age brackets | 18–29: 22%, 30–44: 25%, 45–59: 25%, 60+: 28% |
| Education | <HS: 10%, HS: 27%, Some college: 29%, BA: 22%, Graduate: 12% |
| Party ID (7-pt) | Strong Dem 16%, Weak Dem 10%, Lean Dem 10%, Ind 13%, Lean Rep 9%, Weak Rep 11%, Strong Rep 16%, Other 5% |
| Income | <30k: 20%, 30–60k: 25%, 60–100k: 28%, 100–150k: 16%, 150k+: 11% |
| State | Weighted by U.S. adult population |
| ZIP | Drawn from `hbs_zip_table.json` — must be valid USPS ZIP within sampled state |
| Birth year | `current_year - age`; must be 1924–2006; stored as `established_birth_year` |

**Persona-demographic coupling (existing swap-sort algorithm):** Preserve this. Engaged Responders correlate with higher education; Extreme Responders with stronger political identity; Careless Responders with younger age and lower income. The swap-sort ensures marginal distributions match targets exactly.

### 3.2 Response Style Distribution

| Style | Weight | Key Behavioral Rules |
|---|---|---|
| engaged_responder | 0.30 | Full scale use, correct reverse-codes 95%, passes all attention checks |
| satisficer | 0.20 | Midpoint tendency, survey 35% faster, OE ? 12 words |
| extreme_responder | 0.08 | Items cluster at endpoints, emphatic OE |
| acquiescent_responder | 0.07 | +0.5pt inflation on all items (Weijters 2010) |
| careless_responder | 0.05 | 40% attention check fail rate, IRV detectable |
| social_desirability | 0.10 | Inflates O, C, A; deflates N; hedges OE |
| moderate_responder | 0.10 | Never uses endpoints; 2–6 on 7-point scale |
| deliberate_responder | 0.10 | Slow completion, high variance, longest OE |

### 3.3 Load from Real Data (DANEEL feature — add to HBS)

`HBSProfileLoader.from_csv(path: str, n: int) -> list[ParticipantState]`:

- Flexible column-name matching: handles CES, ANES, and custom schemas
- Supported fields: age, education, income, race, gender, party_id, ideology, state, zip, urbanicity, voter_reg, church_attendance, marital_status, employment, presidential_vote_history
- Draws n rows with replacement if n > file length
- Maps education strings to HBS canonical levels
- This mode is used when researchers want to simulate SPECIFIC demographic segments

---

## PART 4 — IMPROVEMENT 1: PROGRAMMATIC TOOL-USE LAYER (`hbs_tool_dispatcher.py`)

**Source:** DANEEL+ architecture — the primary reason DANEEL+ passes 81/81 vs DANEEL's 49/59.

**Core insight:** LLMs fail systematically and predictably at: pixel-level image geometry, character-position counting, arithmetic, and real-time clock queries. DANEEL+ lets the LLM *invoke* Python functions for these tasks. HBS must do the same.

### 4.1 The HBSTool Interface

```python
class HBSTool:
    """
    General-purpose programmatic tools the LLM can call on-demand.
    These are NOT question-specific. The LLM decides when to invoke them.
    Functions execute locally in Python; results returned to LLM as structured JSON.
    Architecture mirrors DANEEL+ tool-use routines exactly.
    """

    @staticmethod
    def analyze_image(image_bytes: bytes, query: str) -> dict:
        """
        Image analysis for optical illusions and counting tasks.
        Calculates area, length, radius, and relative size of objects.
        Resolves: Ebbinghaus, Mueller-Lyer, Ponzo, checker-shadow illusions.
        Resolves: finger counting, object counting tasks.
        
        Returns:
            {
                "objects_found": list[str],
                "relative_sizes": dict,     # object ? relative size ranking
                "measurements": dict,       # object ? pixel area/length
                "comparison_result": str,   # e.g. "left circle is smaller"
                "confidence": float
            }
        """
        # Implementation: use PIL to extract image regions, compute pixel counts
        # For Ebbinghaus: isolate central circles, compare pixel areas
        # For finger counting: edge detection + blob counting
        pass

    @staticmethod
    def analyze_text(text: str, query: str) -> dict:
        """
        Character-level and word-level text analysis.
        Handles: letter counting, nth-word extraction, positional tasks.
        Resolves: "how many b's in blueberry", "what is the 4th word of this sentence"
        
        Returns:
            {
                "letter_counts": dict,       # char ? count
                "word_count": int,
                "nth_word": str,             # if requested
                "target_count": int,         # direct answer to counting query
                "answer": str               # human-readable result
            }
        """
        # Implementation: pure Python string operations — no LLM
        # "how many b's in blueberry" ? count('b') = 2
        # "4th word of X sentence" ? text.split()[3]
        pass

    @staticmethod
    def compute_arithmetic(expression: str) -> dict:
        """
        Safe arithmetic evaluation for ECLAIR calculation questions.
        e.g. "(5.5 + 2.2 + 1.1) × 2.5^1.50" ? 34.79
        Uses Python's decimal module for precision; no eval() on untrusted input.
        
        Returns:
            {"result": float, "formatted": str, "steps": list[str]}
        """
        pass

    @staticmethod
    def get_current_datetime(timezone: str) -> dict:
        """
        Real-time clock for ECLAIR temporal questions.
        Returns current time in participant's timezone.
        
        Returns:
            {"year": int, "month": str, "day_of_week": str, "time_12h": str,
             "date_formatted": str}
        """
        pass

    @staticmethod
    def lookup_factual(query_type: str, key: str) -> dict:
        """
        Lookup table for ECLAIR context questions.
        query_type: "sports_result" | "weather" | "current_president" | "recent_movie"
        
        Returns: {"answer": str, "source": str, "confidence": float}
        """
        # Reads from hbs_sports_lookup.json, hbs_weather_stubs.json
        pass
```

### 4.2 LLM Tool-Call Integration

The LLM prompt is extended with a `[TOOLS AVAILABLE]` block:

```
[TOOLS AVAILABLE]
If this question requires counting characters/words, analyzing an image, doing arithmetic,
or knowing the current time — invoke a tool FIRST, then use its output to construct your answer.

To invoke a tool, output ONLY:
{"tool_call": "analyze_text", "args": {"text": "blueberry", "query": "count letter b"}}

The tool result will be returned to you. Then give your final answer as:
{"answer": "...", "reasoning": "..."}

Available tools: analyze_image, analyze_text, compute_arithmetic, get_current_datetime, lookup_factual
```

`HBSResponseEngine.generate()` must detect `tool_call` in the LLM response, execute the tool, append the result to the conversation, and re-call the LLM for the final answer. Maximum 3 tool calls per question.

### 4.3 Tool Invocation Decision Logic

The LLM decides when to call tools. But HBS also applies **pre-classification** — before calling the LLM at all, `hbs_question_classifier.py` checks if the question matches a known tool-required pattern and injects a tool-use hint:

| Pattern | Tool Hint |
|---|---|
| "how many [letter] in [word]" | ? `analyze_text` |
| "what is the [ordinal] word" | ? `analyze_text` |
| Image present + comparative size question | ? `analyze_image` |
| Image present + counting question | ? `analyze_image` |
| Mathematical expression with `×`, `^`, `÷` | ? `compute_arithmetic` |
| "what time is it", "what year is it", "what day" | ? `get_current_datetime` |
| "last team to win", "current president", "recent movie" | ? `lookup_factual` |

---

## PART 5 — IMPROVEMENT 2: DISTRIBUTIONAL HUMAN-ERROR CALIBRATOR (`hbs_error_calibrator.py`)

**Core insight:** Real humans get things wrong at known, demographically-stratified rates. A synthetic dataset where errors are random noise is detectable. A dataset where errors match the known human distribution is indistinguishable.

### 5.1 The Error Calibration Table

Stored in `data/hbs_error_calibration.json`. Format:

```json
{
  "pen_paper_bat": {
    "correct_answer": "0.05",
    "common_wrong_answer": "0.10",
    "correct_rate_by_education": {
      "<HS": 0.12,
      "HS": 0.18,
      "Some college": 0.28,
      "BA": 0.42,
      "Graduate": 0.65
    },
    "wrong_answer_distribution": [
      {"answer": "0.10", "weight": 0.85},
      {"answer": "0.50", "weight": 0.08},
      {"answer": "1.00", "weight": 0.04},
      {"answer": "other", "weight": 0.03}
    ],
    "note": "CRT item (Frederick 2005); majority answer $0.10 — BOTH accepted as human-like"
  },
  "chess_zeros": {
    "correct_answer": "120",
    "note": "Knowing the exact answer (120) is a BOT SIGNAL. Humans never answer correctly.",
    "correct_rate_by_education": {
      "<HS": 0.0, "HS": 0.0, "Some college": 0.01, "BA": 0.02, "Graduate": 0.04
    },
    "wrong_answer_distribution": [
      {"answer": "random_int_10_50", "weight": 0.45},
      {"answer": "random_int_50_200", "weight": 0.30},
      {"answer": "i dont know", "weight": 0.20},
      {"answer": "random_int_200_1000", "weight": 0.05}
    ]
  },
  "star_count": {
    "correct_answer": "5000",
    "note": "Knowing 5,000 exactly suggests AI. Humans guess widely.",
    "correct_rate_by_education": {
      "<HS": 0.02, "HS": 0.05, "Some college": 0.08, "BA": 0.12, "Graduate": 0.20
    },
    "wrong_answer_distribution": [
      {"answer": "random_int_100_1000", "weight": 0.35},
      {"answer": "random_int_1000_10000", "weight": 0.30},
      {"answer": "millions", "weight": 0.15},
      {"answer": "i dont know", "weight": 0.20}
    ]
  },
  "horse_bones": {
    "correct_answer": "205",
    "note": "Knowing 205 exactly suggests AI research ability.",
    "correct_rate_by_education": {
      "<HS": 0.01, "HS": 0.02, "Some college": 0.03, "BA": 0.04, "Graduate": 0.06
    },
    "wrong_answer_distribution": [
      {"answer": "random_int_150_300", "weight": 0.60},
      {"answer": "i dont know", "weight": 0.30},
      {"answer": "random_int_300_500", "weight": 0.10}
    ]
  },
  "tipi_reverse_code": {
    "correct_rate_by_persona": {
      "engaged_responder": 0.95,
      "satisficer": 0.70,
      "extreme_responder": 0.65,
      "acquiescent_responder": 0.55,
      "careless_responder": 0.35,
      "social_desirability": 0.88,
      "moderate_responder": 0.80,
      "deliberate_responder": 0.92
    },
    "failure_mode": "ignores_direction"
  },
  "logical_reasoning_syllogism": {
    "correct_rate_by_education": {
      "<HS": 0.35, "HS": 0.55, "Some college": 0.65, "BA": 0.75, "Graduate": 0.85
    }
  },
  "recycling_newspaper_color": {
    "correct_answer": "Yellow",
    "note": "Yellow recycling bin for paper is US convention but not universal.",
    "correct_rate_global": 0.72,
    "wrong_answer_distribution": [
      {"answer": "Blue", "weight": 0.20},
      {"answer": "Green", "weight": 0.08}
    ]
  }
}
```

### 5.2 HBSErrorCalibrator Interface

```python
class HBSErrorCalibrator:

    def should_answer_correctly(
        self,
        question_id: str,
        participant: ParticipantState,
        calibration_table: dict
    ) -> bool:
        """
        Returns True if this participant should get the correct answer.
        Uses education_level + persona_id as lookup keys.
        Probabilistic draw — same participant may answer some correctly, some not.
        """

    def get_wrong_answer(
        self,
        question_id: str,
        correct_answer: str,
        participant: ParticipantState
    ) -> str:
        """
        Returns a calibrated wrong answer drawn from the human error distribution.
        NEVER returns a random wrong answer — always draws from the empirical
        distribution of what humans actually say when they get this question wrong.
        """

    def calibrate_open_ended_knowledge(
        self,
        question_type: str,
        participant: ParticipantState
    ) -> str:
        """
        For factual OE questions (horse bones, historical temperatures):
        Returns a plausible-sounding but typically wrong answer.
        Education-matched precision: Graduate gives closer wrong answer than <HS.
        """
```

### 5.3 Integration Point

`HBSResponseEngine.generate()` calls `HBSErrorCalibrator.should_answer_correctly()` BEFORE the LLM call. If False, the engine skips LLM and calls `get_wrong_answer()` directly. This prevents the LLM from "over-performing" on knowledge questions.

---

## PART 6 — IMPROVEMENT 3: STYLOMETRIC FINGERPRINT ENGINE (`hbs_stylometric_engine.py`)

**Core insight (from DANEEL's cognitive realism pillar + AGENTS.md "free-form text issue"):** The hardest human signal to fake is *sounding like the same person across all OE responses*. A stylometric fingerprint captures an individual's writing DNA and enforces it across every text response. This is what makes participant voices internally consistent.

### 6.1 The Stylometric Fingerprint

```python
@dataclass
class StylometricFingerprint:
    """
    Per-participant writing identity. Built from first OE response.
    Constrains all subsequent OE responses.
    Based on Pennebaker & King (1999) LIWC; Denscombe (2008) response lengths;
    Zipf's law for vocabulary naturalness.
    """
    # Lexical features
    vocab_richness: float           # type-token ratio; 0.4–0.9 typical human range
    mean_word_length: float         # chars per word; education-correlated
    lexical_density: float          # content words / total words

    # Syntactic features
    mean_sentence_length: float     # words per sentence; 8–22 typical
    sentence_length_variance: float # low for satisficers; high for deliberate
    clause_complexity: str          # "simple" | "compound" | "complex"

    # Stylistic markers
    capitalization: str             # "standard" | "all_lower" | "inconsistent" | "all_caps"
    punctuation_rate: float         # marks per 100 words; education-correlated
    exclamation_rate: float         # per 100 words; persona-correlated
    contraction_rate: float         # it's, don't, etc.; informality marker
    filler_word_rate: float         # "um", "like", "you know", "kinda"
    hedge_word_rate: float          # "maybe", "I think", "probably"

    # LIWC-style tone
    liwc_posemo: float              # positive emotion word rate
    liwc_negemo: float              # negative emotion word rate
    liwc_analytic: float            # analytical thinking markers
    liwc_authentic: float           # authenticity markers ("I", "me", "my")

    # Error signature
    typo_keys: list[str]            # recurring QWERTY pairs this participant confuses
    apostrophe_omission: bool       # "dont" instead of "don't" — persistent pattern
    comma_splice_tendency: bool     # runs sentences together with commas
```

### 6.2 Fingerprint Construction

`HBSStylometricEngine.bootstrap(first_oe_text: str, participant: ParticipantState) -> StylometricFingerprint`:

The first OE response is generated with only the participant's persona and demographics as constraints (no fingerprint yet). After generation, compute the fingerprint from the output text itself, then store it in `participant`. All subsequent OE responses are constrained by it.

**Demographic priors** (used before fingerprint exists, and as regularization):

| Education | mean_sentence_len | punct_rate | vocab_richness | filler_rate |
|---|---|---|---|---|
| <HS | 8–12 words | 1–3/100 | 0.40–0.55 | high (0.05+) |
| HS | 10–15 words | 2–5/100 | 0.50–0.65 | medium |
| Some college | 12–16 words | 4–7/100 | 0.60–0.72 | low-medium |
| BA | 14–18 words | 5–9/100 | 0.68–0.80 | low |
| Graduate | 16–22 words | 7–12/100 | 0.75–0.90 | very low |

### 6.3 Fingerprint Enforcement in OE Generation

`HBSStylometricEngine.apply(draft_text: str, fingerprint: StylometricFingerprint) -> str`:

Post-processes any generated OE response to match the fingerprint:

1. **Sentence length normalization:** Split/merge sentences to match `mean_sentence_length ± variance`
2. **Capitalization enforcement:** Convert to match `capitalization` style
3. **Typo injection:** Apply `typo_keys` QWERTY errors at `persona_typo_rate`; always use delayed correction (type wrong, backspace, retype)
4. **Filler insertion:** Inject filler words at `filler_word_rate` at sentence boundaries
5. **Hedge insertion:** For hedge-prone profiles, add "I think" or "maybe" to opinion statements
6. **Contraction normalization:** Either always contract or never contract per `contraction_rate`
7. **Apostrophe enforcement:** If `apostrophe_omission=True`, remove apostrophes from contractions ("dont", "cant", "its")
8. **Vocabulary diversity:** Use Zipf-law sampling for word substitution — more common words for lower vocab_richness

**Critical rule:** `apply()` is called on EVERY OE response, including those from LLM Tier 1. The stylometric fingerprint overrides the LLM's own stylistic choices to ensure the participant always sounds the same.

### 6.4 Voice Memory Integration

After each OE response passes fingerprint enforcement, append a 15-word excerpt to `participant.voice_memory`. When generating subsequent OE responses, inject the last 3 excerpts into the LLM prompt as "How this person wrote before:" — this gives the LLM the tone signal AND the fingerprint engine the enforcement backstop.

---

## PART 7 — IMPROVEMENT 4: ADVERSARIAL SELF-VALIDATION PIPELINE (`hbs_validator.py`)

**Source:** DANEEL was built by running it against its own detection battery iteratively. HBS must embed this process. After generation, before returning data to the user, HBS validates the dataset against every DANEEL benchmark axis and auto-corrects failures.

### 7.1 Validation Axes

```python
class HBSValidator:

    DANEEL_BENCHMARKS = {
        # Behavioral timing (DANEEL behavioral benchmark Q5)
        "completion_time": {
            "human_min_seconds": 480,    # 8 minutes
            "human_max_seconds": 1500,   # 25 minutes
            "target_pct_in_range": 0.80, # 80% of participants in 10-18 min range
        },
        # TIPI personality (DANEEL behavioral benchmark Q4)
        "tipi_variance": {
            "min_sd_per_participant": 0.8,  # straight-lining if SD < 0.5
            "min_pct_passing": 0.90,        # 90%+ of participants must have SD ? 0.8
        },
        # IAT d-score (DANEEL behavioral benchmark Q1)
        "iat_d_score": {
            "valid_range": (-2.0, 2.0),
            "human_typical": (-0.5, 1.5),
            "min_pct_valid": 1.0,           # 100% must have valid d-score
        },
        # Attention check rates (realistic, not perfect)
        "attention_checks": {
            "overall_pass_rate_range": (0.82, 0.96),  # real online sample range
            "careless_persona_pass_rate_max": 0.70,
        },
        # OE response characteristics
        "oe_responses": {
            "min_word_count": 3,
            "max_word_count_for_satisficer": 15,
            "min_uniqueness_pct": 0.90,     # 90%+ of OE responses must differ
            "max_pct_off_topic": 0.02,      # <2% can be off-topic
        },
        # Context consistency (DANEEL context awareness benchmarks)
        "context_consistency": {
            "zip_state_match_rate": 1.0,    # 100% — always enforced
            "birth_year_consistency": 1.0,  # 100% — always enforced
            "political_consistency": 0.95,  # 95%+ — small drift allowed
        },
        # Open-ended length distribution (Denscombe 2008)
        "oe_length_distribution": {
            "mean_words_range": (8, 35),
            "sd_words_range": (3, 20),
        },
        # Rating-text coherence (Krosnick 1999; Podsakoff 2003)
        "rating_text_coherence": {
            "min_pct_coherent": 0.95,
        },
        # Straight-lining rate (Meade & Craig 2012)
        "straightlining_rate": {
            "expected_range": (0.03, 0.08),  # 3-8% of real online respondents
        },
    }
```

### 7.2 Validation + Auto-Correction Logic

`HBSValidator.validate_and_correct(dataset: pd.DataFrame, participants: list[ParticipantState]) -> tuple[pd.DataFrame, dict]`:

For each benchmark:

**Completion time out of range:**
- Detect participants outside [480, 1500] seconds
- Auto-correct: resample `target_completion_seconds` within range; re-scale all question timing metadata proportionally

**TIPI straight-lining (SD < 0.5):**
- Detect affected rows
- Auto-correct: apply `_add_tipi_variance()` — randomly perturb 2–3 items by ±1 scale point in the direction consistent with the participant's persona trait scores

**IAT invalid d-score:**
- Detect d-scores outside (-2.0, 2.0)
- Auto-correct: clamp and re-derive RT distributions to produce the target d-score

**Attention check pass rate outside [0.82, 0.96]:**
- If too high: randomly flip 1–2 passing checks to fails on careless/satisficer participants
- If too low: randomly flip fails to passes on engaged/deliberate participants

**OE uniqueness < 90%:**
- Detect duplicate/near-duplicate responses (cosine similarity > 0.85 on TF-IDF vectors)
- Auto-correct: re-run stylometric `apply()` with higher variance parameters until uniqueness ? 90%

**ZIP/state mismatch:**
- Always fixed at generation time; flag as critical error if detected in validation

**Rating-text incoherence:**
- Detect: positive numeric rating (?5/7) + negative sentiment OE text, or vice versa
- Auto-correct: re-run `apply()` with explicit tone constraint from numeric pattern
