# OE Response Human-Likeness Improvement Plan — v1.0.9.8 → v1.1.0.2

## Problem Statement

Current OE responses, while topically grounded and behaviorally coherent, still exhibit detectable patterns that distinguish them from real human survey responses. Key gaps:

1. **Structural uniformity**: LLM-generated responses follow predictable sentence structures (Subject-Verb-Object) even after deep variation. Real humans have far more chaotic writing patterns.
2. **Missing disfluencies**: Real survey responses contain false starts, self-corrections, mid-thought pivots, parenthetical asides, and incomplete sentences at much higher rates than currently generated.
3. **Typo/error model too shallow**: Current typo system only swaps ~10 common words. Real typos include letter transpositions, omissions, autocorrect artifacts, and phone-typing patterns.
4. **Response opener monotony**: Despite 16 casual starters, responses still cluster around "I think/feel" patterns after deep variation.
5. **Missing demographic voice authenticity**: Age/education/culture cues exist in prompts but aren't enforced in post-processing or fallback generators.
6. **Template responses lack structural variety**: ComprehensiveResponseGenerator and TextResponseGenerator produce grammatically correct, well-structured text that's too clean for survey data.
7. **Length distribution unrealistic**: Real survey response lengths follow a heavy right-skewed distribution (many 1-5 word responses, few long ones). Current generation is more normally distributed.
8. **Missing real-world response artifacts**: Real survey data contains emoji use, URL fragments, copy-paste artifacts, text-speak abbreviations, and platform-specific typing patterns (phone vs desktop).
9. **Cross-participant vocabulary overlap**: Deep variation swaps individual words but doesn't change the underlying vocabulary register, making responses from "different people" sound like the same person rephrasing.
10. **Banned phrase list reactive not proactive**: We catch AI-sounding phrases after generation but don't prevent the underlying patterns (balanced argumentation, topic sentences, etc.).

## 5-Iteration Plan

### Iteration 1: Deep Disfluency Engine & Realistic Error Model
**Files**: `llm_response_generator.py`
**Version**: 1.0.9.8

**1a. Expand typo/error system** (Layer 5 in `_apply_deep_variation`)
- Replace shallow `_TYPO_MAP` (10 words) with a comprehensive error model:
  - **Adjacent-key typos**: Based on QWERTY layout proximity (e.g., 'think' → 'thibk', 'about' → 'avout')
  - **Letter transposition**: Swap adjacent letters (e.g., 'from' → 'form', 'with' → 'wiht')
  - **Letter omission**: Drop a letter (e.g., 'because' → 'becuse', 'should' → 'shoud')
  - **Letter doubling**: Repeat a letter (e.g., 'really' → 'rreally', 'good' → 'goood')
  - **Autocorrect artifacts**: Common autocorrect errors (e.g., 'ducking', 'well' for 'we'll')
  - **Phone typing patterns**: Missing spaces between words, missing apostrophes
- Error rate calibration: ~3-5% of words in low-formality responses, ~1% in mid, ~0% in high
- Apply DIFFERENT error types per participant (one person does transpositions, another does omissions)

**1b. Natural disfluency injection** (New Layer 3b)
- **False starts**: "I thought— well actually I think the..." (5-10% of responses)
- **Self-corrections**: "it was pretty bad, well not bad exactly, more like disappointing" (8%)
- **Mid-thought pivots**: Insert tangential clause mid-sentence (5%)
- **Trailing off**: Remove ending of last sentence, replace with "..." or just end abruptly (8%)
- **Parenthetical asides**: Insert "(like I said before)" or "(not sure if that makes sense)" (6%)
- **Repeated words**: "I I think" or "the the problem was" (3%)
- Each disfluency type assigned per-participant based on seed, so the same person consistently uses the same disfluency patterns

**1c. Expand banned phrases** and add structural pattern detection
- Add 25+ new banned phrases caught in real LLM outputs
- Add structural pattern detection: if response has "On one hand... on the other" structure, rewrite to one-sided
- Detect and break "First, X. Second, Y. Third, Z" enumeration patterns
- Detect topic-sentence + supporting-evidence essay structure and flatten it

### Iteration 2: Demographic Voice Differentiation & Vocabulary Registers
**Files**: `llm_response_generator.py`, `response_library.py`
**Version**: 1.0.9.9

**2a. Vocabulary register system** (New system in deep variation)
- Define 5 vocabulary registers based on education/age:
  - **Register 1 (Minimal)**: Very basic words, short sentences, text-speak heavy. Age 18-22, low education.
  - **Register 2 (Casual)**: Colloquial but competent. Age 22-35, some college.
  - **Register 3 (Standard)**: Average adult vocabulary. Age 25-55, college educated.
  - **Register 4 (Articulate)**: Rich vocabulary, complex sentences. Age 30-65, graduate education.
  - **Register 5 (Expert)**: Domain-specific jargon, precise language. Any age, domain expert.
- Each participant assigned a register based on persona traits
- Register affects: word choice, sentence complexity, use of slang, punctuation habits

**2b. Age-specific writing patterns** (Enhanced Layer 3)
- **Gen Z (18-25)**: "lowkey", "ngl", "fr", "tbh", no periods at end, extensive emoji-adjacent language, run-on sentences connected with "and", "like" as filler, abbreviations (rn, imo, idk, w/, bc)
- **Millennial (26-40)**: Mix of casual/proper, occasional text-speak, parenthetical humor, pop culture references, moderate sentence length
- **Gen X (41-55)**: More complete sentences, fewer abbreviations, occasional ellipsis use ("well..."), more measured tone
- **Boomer+ (56+)**: Complete sentences, proper punctuation, longer responses, life experience references, more formal word choice, occasional technology unfamiliarity markers
- Apply age patterns AFTER main generation, as a voice-casting layer

**2c. Condition-aware response starter banks** (Replace `_CASUAL_STARTERS`)
- Instead of generic starters, build condition-type-aware starters:
  - **After reading an article**: "so that article...", "after reading that..."
  - **After a game/decision**: "honestly the decision was...", "when I had to decide..."
  - **After viewing a product**: "that product...", "looking at it..."
  - **After political stimulus**: "look I know people disagree but...", "politically speaking..."
  - **After health info**: "health wise...", "when it comes to my health..."
- 50+ starters organized by study domain, injected based on detected domain

### Iteration 3: Response Length Realism & Structural Diversity
**Files**: `llm_response_generator.py`, `response_library.py`, `persona_library.py`
**Version**: 1.1.0.0

**3a. Realistic length distribution** (Enhanced `_apply_deep_variation` Layer 2)
- Implement right-skewed length distribution matching real survey data:
  - 15% ultra-short: 1-4 words ("its fine", "liked it", "no opinion")
  - 20% short: 5-15 words (one incomplete sentence)
  - 30% medium: 15-40 words (1-2 sentences)
  - 20% moderate: 40-80 words (2-3 sentences)
  - 10% long: 80-150 words (paragraph)
  - 5% very long: 150+ words (detailed paragraph with tangents)
- Length drawn from distribution, modulated by engagement and verbosity traits
- Ultra-short responses get special handling: no deep variation (ruins them), just direct topic-word responses

**3b. Structural diversity templates for ComprehensiveResponseGenerator**
- Add 10 structural patterns that break the Subject-Verb-Object monotony:
  1. **Fragment opener**: "The price though. Like way too much for what you get."
  2. **Question-then-answer**: "Did I like it? Yeah actually I did."
  3. **List without formatting**: "pros: good price, looked nice. cons: felt cheap, slow shipping"
  4. **Emotional exclamation**: "God that was frustrating. Nothing worked the way..."
  5. **Comparative**: "Way better than I expected honestly"
  6. **Concessive**: "Ok sure the idea was good BUT the execution..."
  7. **Stream of consciousness**: Run-on with commas instead of periods
  8. **Abrupt opinion**: "Nope. Just no. Not for me."
  9. **Story micro-narrative**: "So I read it and immediately thought of my friend who..."
  10. **Trailing qualifier**: "It was good I guess, could have been better maybe, idk"
- Each participant gets a primary structural tendency based on their seed

**3c. Ultra-short response generator** (New specialized handler)
- For engagement < 0.25 or straight-liners, bypass normal generation entirely
- Generate from ultra-short topic-relevant bank:
  - Pattern: "{topic_word} {brief_opinion}" → "trump ok", "price high", "liked it"
  - Pattern: "{short_reaction}" → "meh", "yeah no", "nah", "sure ig"
  - Pattern: "{topic_word} idk" → "politics idk", "ai stuff idk"
- Maximum 8 words, lowercase, minimal punctuation

### Iteration 4: Anti-Detection Hardening & Cross-Participant Uniqueness
**Files**: `llm_response_generator.py`, `enhanced_simulation_engine.py`
**Version**: 1.1.0.1

**4a. Response fingerprint deduplication** (Enhanced `_ensure_unique_start`)
- Current system only checks first 6 words for duplication
- Expand to check:
  - **Semantic similarity**: Detect responses that express the same idea with different words
  - **Structural similarity**: Detect responses with same sentence count + similar lengths
  - **Shared rare phrases**: Track all 3+ word phrases across participants, flag repetition
- When duplication detected, apply transformative rewrite (not just prepend an opener):
  - Sentence reordering
  - Active↔passive voice swap
  - Merge two short sentences / split one long sentence
  - Replace key opinion word with synonym

**4b. Vocabulary diversity enforcement**
- Track vocabulary across all generated responses in a batch
- If a word appears in >30% of responses, apply synonym rotation for subsequent responses
- Maintain per-participant "vocabulary fingerprint": set of characteristic words that participant uses across all their OE responses
- Inject participant-specific verbal tics: "honestly" person, "like" person, "basically" person, "I mean" person

**4c. Real-world artifact injection** (New Layer 8 in deep variation)
- 2-3% of responses get one of:
  - **Emoji use**: Simple emoji (common in younger demographics) — 😂, 👍, 🤷, 💯
  - **Emphasis via caps**: "the price was WAY too high" or "I absolutely LOVED it"
  - **Text formatting artifacts**: Extra spaces, no-space-after-period, random capitalization
  - **Copy-paste hint**: Rare — response starts mid-sentence as if copied from elsewhere
  - **Time pressure indicator**: "running out of time but basically I think..."
- Artifact type correlates with demographic profile (young = emoji, old = ellipsis overuse)

### Iteration 5: LLM Prompt Optimization & Template Cascade Enhancement
**Files**: `llm_response_generator.py`, `response_library.py`, `persona_library.py`
**Version**: 1.1.0.2

**5a. Prompt engineering for maximal human-likeness**
- Add **exemplar responses** to the system prompt: 3-5 REAL survey responses (anonymized) per question type showing authentic human writing patterns
- Add **anti-pattern examples** showing what NOT to generate (with explanations)
- Add **calibration rule**: "If you removed the participant ID, a human coder should NOT be able to tell these were AI-generated. The #1 detection signal is: all responses sound like they were written by the same educated 30-year-old. Make them sound like they came from people aged 18-75 with different education levels."
- Add **structural diversity mandate**: "No two responses in this batch should have the same sentence structure. If Participant 1 writes 'I think X because Y', Participant 2 must NOT follow the same 'I think [opinion] because [reason]' pattern."

**5b. Enhanced template cascade in ComprehensiveResponseGenerator**
- Add **interview transcript templates**: Based on real qualitative research coding, these templates mimic how people actually respond in semi-structured interviews
- Add **text message style templates**: For low-formality, high-engagement responses
- Add **complaint letter style**: For very negative sentiment responses
- Add **diary entry style**: For reflective/introspective questions
- Each style has 10+ templates with slots for topic, sentiment, and condition

**5c. Cross-response coherence in TextResponseGenerator**
- When the same participant answers multiple OE questions:
  - Maintain consistent vocabulary register
  - Maintain consistent disfluency patterns (same person keeps making same types of errors)
  - Maintain consistent length tendency (verbose person stays verbose)
  - Reference previous answers: "like I said before..." or "same as my other answer..."
  - Slight engagement decay: Later OE answers tend to be shorter (survey fatigue)

## Version Sync Checklist (ALL 9 locations per commit)

Each iteration bumps version in:
1. `app.py` REQUIRED_UTILS_VERSION
2. `app.py` APP_VERSION
3. `app.py` BUILD_ID
4. `utils/__init__.py` __version__
5. `utils/__init__.py` docstring
6. `utils/qsf_preview.py` __version__
7. `utils/response_library.py` __version__
8. `README.md` header
9. `README.md` features section

## Testing Strategy

After each iteration:
1. `python3 -m py_compile` on all modified files
2. `grep -r "OLD_VERSION"` to verify no version mismatches
3. Inspect generated samples to verify human-likeness improvements
