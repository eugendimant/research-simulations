# Response Generation Pipeline

This is the core architecture for producing high-quality simulated survey responses. Read this file whenever building or modifying the response generation system.

The fundamental principle: **every response is conditioned on everything that came before it for that subject.** Responses are never generated in isolation. The simulator maintains a running subject context that grows with each item and feeds forward into subsequent items.

## Pipeline Overview

```
1. SUBJECT PROFILE GENERATION
   → Demographics, persona, trait parameters, condition assignment
   
2. SEQUENTIAL ITEM PROCESSING (item by item, in survey order)
   → For each item: build prompt with full subject context → generate → validate → append to context
   
3. POST-GENERATION CONSISTENCY AUDIT
   → Cross-check all responses within subject → flag/repair inconsistencies
   
4. QUALITY GATE
   → Pass → emit subject row. Fail → repair or regenerate.
```

## Stage 1: Subject Profile Generation

Before generating any survey responses, construct the complete subject profile. This profile is the seed for ALL subsequent responses.

### Profile Components

**Demographics** (generated as a coherent bundle, never independently):
```
Generate all of the following as a single coherent profile:
- Age, gender, education, income, occupation, marital status, location
- These must form a plausible combination (e.g., age constrains education,
  education constrains occupation, occupation constrains income)
- Draw from target population distributions with realistic noise
```

**Persona assignment:**
- Assign persona type from the mixture prior (e.g., 40% prosocial, 30% individualist, 30% mixed)
- Generate persona-specific trait parameters: values on each construct dimension
- Trait parameters are continuous, not categorical — a "prosocial" persona has a distribution of prosociality, not a fixed value

**Condition assignment:**
- Assign experimental condition according to the randomization scheme
- Record the condition-specific treatment description — this gets passed into the prompt context

**Latent disposition vector:**
- Generate a vector of latent dispositions that will drive response tendencies across the survey. This is the key mechanism for cross-item correlation.
- Example: if the survey measures risk aversion, time preference, and trust, generate correlated draws for these constructs based on the persona type and known inter-construct correlations from the literature.
- These latent values anchor responses. They are NOT the responses themselves — they set the central tendency around which individual item responses vary.

### Profile Prompt Pattern

```
You are simulating a single survey respondent with the following profile:

DEMOGRAPHICS:
[age], [gender], [education], [occupation], [income], [marital status], [location]

PERSONA TYPE: [type] — [brief behavioral description]

TRAIT PARAMETERS:
- [Construct A]: [value] (scale: [min]-[max])
- [Construct B]: [value]
- [Construct C]: [value]
[Include known correlations: "This person's high [A] is consistent with moderate [B]"]

EXPERIMENTAL CONDITION: [condition label]
TREATMENT: [description of what this participant experiences/sees]

You will now answer survey items one at a time as this person.
Maintain consistency with this profile throughout.
Do NOT break character. Do NOT explain your reasoning.
Answer exactly as this person would.
```

## Stage 2: Sequential Item Processing

Process items in survey presentation order. For EACH item:

### 2a. Build the Item Prompt

The prompt for item N includes:
1. The full subject profile (from Stage 1)
2. ALL prior responses (items 1 through N-1) — this is the running context
3. The current item text, response options, and any conditional context
4. Item-type-specific generation instructions (see below)

**Critical: the running context grows with each item.** By the time the subject reaches item 30, the prompt includes their profile plus all 29 prior responses. This is what produces intra-subject coherence.

### 2b. Item-Type-Specific Prompting

**Likert/Rating Scales:**
```
Item: "[item text]"
Scale: 1 ([low anchor]) to 7 ([high anchor])

Given this person's profile and their previous responses, what would they answer?
Respond with ONLY a single integer from 1 to 7.

[Include disposition anchor]: "This person's latent [construct] is [X]/10,
suggesting they tend toward [description]. But individual items vary —
add realistic noise (±1-2 points from the disposition-implied value)."
```

**Open-Text Following a Quantitative Item:**
```
The participant just answered [X] out of 7 on "[item text]".
Now they are asked: "[open-text prompt, e.g., 'Why did you give that rating?']"

Write their response AS this person. Key constraints:
- The explanation MUST be consistent with a rating of [X]/7
- [If X ≤ 2]: Express clear dissatisfaction/disagreement. Use specific reasons.
- [If X = 3-5]: Express mixed or moderate views. Some hedging is natural.
- [If X ≥ 6]: Express clear satisfaction/agreement. Be specific about what works.
- Match the person's education level in vocabulary and sentence complexity
- Length: [draw from realistic distribution — most responses are 5-30 words for this item type]
- It is OK to use informal language, abbreviations, typos, or be terse
- Do NOT use phrases like "I believe that," "it is important to note," "resilience and determination"
```

**Standalone Open-Text (No Quantitative Anchor):**
```
The participant is asked: "[open-text prompt]"

Write their response AS this person, given their profile and all prior responses.
- Draw on their demographic background and persona for perspective
- Reference specific experiences plausible for someone with their profile
- Length: [sample from right-skewed distribution, most are 10-50 words]
- [If this is an optional field]: There is a [40-70]% chance this person skips this entirely.
  If they skip, respond with exactly: SKIP
```

**Binary/Multiple Choice:**
```
Item: "[item text]"
Options: [A, B, C, ...]

Given this person's profile, condition, and previous responses, which option would they choose?
Respond with ONLY the option letter/text.

[If relevant]: "Note — this person previously said [relevant prior response].
Their answer here should be consistent with that."
```

**Behavioral/Decision Tasks (Dictator Game, Risk Lottery, etc.):**
```
TASK: [Full task description including stakes, options, and context]

This person's relevant latent dispositions:
- [Construct]: [value] → suggests they would [behavioral tendency]

Given their profile and prior behavior in this survey, what do they choose?
Respond with ONLY their choice.

[If there were prior decision tasks]: "In the previous task, this person chose
[prior choice], which revealed [trait inference]. Their choice here should
be broadly consistent with that preference, with domain-appropriate variation."
```

**Attention Check Items:**
```
[For ~95% of subjects]: Answer this correctly as instructed.
[For ~5% of subjects]: This person is slightly distracted/fatigued.
  They may misread the instruction. Answer INCORRECTLY — either by choosing
  a plausible-but-wrong option or by not reading carefully.
```

### 2c. Validate Each Response

After generating each item response, run immediate checks:
- Is the response within the valid range? (e.g., 1-7 for Likert)
- Does it contradict any prior response? (flag if so — may be acceptable noise, or may need repair)
- For open-text: does the tone/direction match the quantitative anchor?
- For demographics: does it contradict the profile?

If validation fails: regenerate that item (up to 3 attempts), then flag for manual review if still failing.

### 2d. Append to Running Context

After validation, add the response to the running context:
```
RESPONSE HISTORY (updated):
...
Item 14 ("I feel safe in my neighborhood"): 5/7
Item 15 ("Why did you give that rating?"): "its pretty quiet here, not much crime i know of"
Item 16 (Condition: Control — "Choose allocation"): Chose $6 for self, $4 for other
...
```

## Stage 3: Post-Generation Consistency Audit

After all items are generated for a subject, run the full consistency audit:

1. **Scale reliability**: Compute Cronbach's alpha for each multi-item scale. Flag if < 0.5.
2. **Reverse-code check**: Verify reverse-coded items are actually reversed relative to forward-coded items.
3. **Open-text alignment**: For each open-text item that follows a quantitative item, verify directional consistency (positive text for high scores, negative for low).
4. **Demographic coherence**: Cross-check all demographic fields for contradictions.
5. **Skip logic**: Verify all skipped items are NA.
6. **Behavioral consistency**: Check that choices in decision tasks are directionally consistent with stated preferences.

**Repair protocol**: If inconsistencies are found:
- For minor issues (off-by-one on a correlated scale): accept as realistic noise.
- For moderate issues (open-text slightly misaligned with score): regenerate the open-text item with stronger conditioning on the quantitative anchor.
- For severe issues (demographic contradiction, skip logic violation): regenerate the problematic items. If 3+ severe issues, regenerate the entire subject.

## Context Window Management

For long surveys (50+ items), the running context may approach LLM context limits. Strategies:

1. **Summarize early sections**: After completing a survey block, summarize responses into a compact format rather than carrying full verbatim text.
2. **Prioritize recent and relevant context**: Always include the full subject profile and the most recent 10-15 items verbatim. Summarize earlier items.
3. **Preserve key anchors**: Always carry forward verbatim: quantitative responses that later open-text items will reference, decision task choices, and demographic responses.

## Prompt Quality Principles

Regardless of item type, all prompts should:

- **Be specific about what format to return** — "Respond with ONLY a single integer" reduces parsing failures.
- **Never ask the LLM to explain its reasoning** — this breaks the simulation frame and wastes tokens.
- **Include the disposition anchor** — the latent trait value that sets the response tendency for this item.
- **Reference relevant prior responses explicitly** — do not rely on the LLM noticing them in a long context.
- **Specify the education/sophistication level** — an MBA holder writes differently than a high school dropout. The prompt must make this explicit.
- **Inject human imperfection instructions** — explicitly tell the LLM it is OK to use informal language, make minor errors, be terse, or skip optional items. LLMs default to overly polished output; they need permission to be messy.
