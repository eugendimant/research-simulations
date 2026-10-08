"""Small, dependency-free text helpers shared by the open-ended text generators.

Everything here is a pure function (no randomness of its own, no I/O). Callers pass in
their own seeded ``random.Random`` where a choice is needed, so seeded datasets stay
reproducible.

The helpers exist because the post-processing layers that make generated answers look
"human" (hedges, fillers, synonym swaps, typos) used to edit text at random word positions.
That produced grammar damage such as "moral felt calls surprisingly human clearly" or
"I wanna to". Each helper here only edits at positions where the edit is grammatical.
"""

import random
import re
from typing import List, Sequence

__version__ = "1.3.0.4"

# ---------------------------------------------------------------------------
# Sentence endings
# ---------------------------------------------------------------------------

# Words that cannot end a sentence, so a response stopping on one was cut off mid-phrase
# ("... let down by a"). Words that can legitimately end a casual sentence ("though", "while",
# "her", "about", "that") are deliberately NOT here.
_ALWAYS_DANGLING = frozenset({
    "a", "an", "the", "my", "your", "their", "our", "its",
    "and", "but", "or", "because", "which", "whether", "although", "than",
})
_DANGLING_PRONOUN = frozenset({"i"})   # "... that's why I." is a cut-off clause
# Prepositions and auxiliaries that are usually cut-off fragments, but can end a real
# sentence ("the person I voted for."). Only trimmed when the text has no closing punctuation.
_DANGLING_WITHOUT_PUNCT = frozenset({
    "of", "to", "for", "in", "on", "at", "by", "with", "from", "into", "as", "about",
    "that", "is", "are", "was", "were", "be", "been", "have", "has", "had", "will",
    "would", "can", "could", "should", "not", "so", "very", "really", "when", "if",
    "who", "whom", "whose", "what", "where", "how", "why", "while", "since", "unless",
    "until", "both", "either", "neither", "each", "every", "some", "any", "no",
})

_TRAILING_PUNCT_CHARS = " \t\r\n\u00a0.!?,;:…\"')]"
_TERMINAL_RE = re.compile(r"[.!?…][\"')\]]*\s*$")


def _core(token: str) -> str:
    return re.sub(r"^[^\w']+|[^\w']+$", "", token)


def trim_dangling_tail(text: str) -> str:
    """Drop trailing words that leave a sentence unfinished and close it with a period.

    Articles, possessives and conjunctions are always trimmed. Prepositions and auxiliaries
    are trimmed only when the text has no closing punctuation (a sign it was cut off). At most
    three words are removed and the text is never reduced below two words. A bare capital "A"
    is kept because it is usually a condition label ("Group A"). Text that already ends
    cleanly is returned unchanged.
    """
    if not text or not isinstance(text, str):
        return text
    has_terminal = bool(_TERMINAL_RE.search(text))
    body = text.rstrip().rstrip(_TRAILING_PUNCT_CHARS)
    words = body.split()
    removed = 0
    trimmed_always = False
    while removed < 3 and len(words) > 2:
        raw = _core(words[-1])
        last = raw.lower()
        if raw == "A":
            break
        # v-review: _DANGLING_WITHOUT_PUNCT is NOT applied: casual answers legitimately end without a
        # period on "that", "so", "not", "about", "to"... ("I really don't like that"), and trimming
        # them changed the meaning ("... good either not really" -> "... good.").
        if last in _ALWAYS_DANGLING:
            words.pop()
            removed += 1
            trimmed_always = True
        elif trimmed_always and last in _DANGLING_WITHOUT_PUNCT:   # "... good for the" -> "... good"
            words.pop()
            removed += 1
        elif not has_terminal and raw == "I" and len(words) > 4:    # "... that's why I"
            words.pop()
            removed += 1
        else:
            break
    if not removed:
        return text
    trimmed = " ".join(words).rstrip(_TRAILING_PUNCT_CHARS)
    return trimmed + "."


_ABBREVIATIONS = frozenset({
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.", "e.g.", "i.e.", "u.s.", "u.k.",
    "no.", "fig.", "inc.", "ltd.", "approx.", "dept.", "est.", "a.m.", "p.m.", "al.",
})


def split_sentences(text: str) -> List[str]:
    """Split after . ! ? followed by whitespace, but not after an abbreviation ("Dr.", "e.g.", "U.S.")."""
    if not text:
        return [text]
    parts = re.split(r"(?<=[.!?])\s+", text)
    out: List[str] = []
    for part in parts:
        if out:
            last = out[-1].split()[-1].lower() if out[-1].split() else ""
            if last in _ABBREVIATIONS and part and not (len(out[-1].split()) == 1 and False):
                out[-1] = out[-1] + " " + part
                continue
        out.append(part)
    return out


def is_probably_non_english(text: str) -> bool:
    """True for Spanish/German/French/... (few English function words) and for non-Latin scripts."""
    if looks_non_english(text):
        return True
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return False
    non_latin = sum(1 for c in letters if ord(c) > 0x024F)
    return non_latin / len(letters) > 0.3


def truncate_to_sentences(text: str, max_words: int) -> str:
    """Shorten ``text`` to about ``max_words`` words without cutting a phrase in half.

    Keeps whole sentences while they fit (when they fill at least two thirds of the budget);
    otherwise cuts at the last sentence end or comma past the halfway point, then at the budget
    itself. In every case a function word left hanging at the end is removed.
    """
    if not text:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    kept, count = [], 0
    for sent in split_sentences(text.strip()):
        n = len(sent.split())
        if count + n > max_words:
            break
        kept.append(sent)
        count += n
    if kept and count >= max(5, (max_words * 2) // 3):
        return " ".join(kept)
    # Whole sentences would leave too little (for example a one-line opener and nothing else),
    # so cut inside the budget: at the last sentence end or comma past the halfway point.
    head = words[:max_words]
    for i in range(len(head) - 1, max(2, len(head) // 2) - 1, -1):
        if head[i].endswith((".", "!", "?", ",", ";", ":")):
            head = head[:i + 1]
            break
    cut = " ".join(head).rstrip(",;: ")
    cut = trim_dangling_tail(cut)
    return cut if _TERMINAL_RE.search(cut) else cut + "."


# ---------------------------------------------------------------------------
# Capitalisation and articles
# ---------------------------------------------------------------------------

_FIRST_WORD_RE = re.compile(r"^(\W*)([A-Za-z][A-Za-z'’]*)(.*)$", re.DOTALL)


_LOWERABLE_STARTERS = frozenset("""
the this that these those it its there here they we he she you my our your their his her some many most
people everyone everybody nobody nothing something anything everything all both each every no not
in on at for with from by as if when while because although though however but and so yet then also
yes yeah well honestly basically actually frankly overall generally personally maybe perhaps probably
definitely certainly clearly obviously sure fine good great bad hard easy interesting important
i'm i've i'd i'll it's that's what how why who which one two three first second finally lastly
after before during since until about around over under through between against without within
""".split())


def lower_first(text: str) -> str:
    """Lower-case the first letter so a prefix can be put in front of the text.

    Keeps "I" / "I'm", acronyms ("AI", "NATO") and words that are capitalised again later in
    the text at a non-sentence-start position (a proper noun such as "Trump").
    """
    if not text:
        return text
    m = _FIRST_WORD_RE.match(text)
    if not m:
        return text
    lead, first, rest = m.groups()
    base = first.replace("’", "'")
    if base == "I" or base.startswith("I'"):
        return text
    if len(first) >= 2 and first.isupper():
        return text
    if first.lower() not in _LOWERABLE_STARTERS:
        return text   # proper noun, brand ("YouTube", "Trump", "Americans") or unknown word: keep
    for occ in re.finditer(r"\b" + re.escape(first) + r"\b", rest):
        before = rest[:occ.start()].rstrip()
        if before and before[-1] not in ".!?":
            return text  # capitalised mid-sentence elsewhere => proper noun
    return lead + first[0].lower() + first[1:] + rest


_AN_PREFIXES = ("hour", "honest", "honor", "honour", "heir")
_A_PREFIXES = ("uni", "use", "usu", "uti", "uto", "una", "eu", "one", "once", "ubiq", "ura", "ute", "uku")
_LETTER_NAMES_WITH_VOWEL_SOUND = set("AEFHILMNORSX")


def indefinite_article(next_word: str) -> str:
    """Return "a" or "an" for the word that follows (approximate English rules)."""
    raw = re.sub(r"^[^A-Za-z0-9]+", "", next_word or "")
    if not raw:
        return "a"
    if raw[0].isdigit():
        digits = re.match(r"\d+", raw).group(0)
        return "an" if digits[0] == "8" or digits in ("11", "18") else "a"
    low = raw.lower()
    letters = re.match(r"[A-Za-z]+", raw).group(0)
    if 2 <= len(letters) <= 4 and letters.isupper():
        if len(letters) == 4 and any(c in "AEIOU" for c in letters):
            return "an" if letters[0] in "AEIOU" else "a"      # NATO, NASA, MAGA, FEMA are said as words
        return "an" if letters[0] in _LETTER_NAMES_WITH_VOWEL_SOUND else "a"
    if low.startswith(_AN_PREFIXES):
        return "an"
    if low.startswith(_A_PREFIXES):
        return "a"
    return "an" if low[0] in "aeiou" else "a"


_ARTICLE_RE = re.compile(r"(?<![\w'’])(an?|An?)(\s+)([A-Za-z0-9][\w'’-]*)")
_NOT_A_NOUN_AFTER_ARTICLE = frozenset({"and", "or", "but", "vs", "versus", "to", "through", "nor", "is", "was", "are", "were",
                                       "over", "in", "on", "at", "as", "if", "with", "than", "for", "from", "by", "it", "i",
                                       "we", "you", "they"})


def fix_indefinite_articles(text: str) -> str:
    """Make "a"/"an" agree with the word after it ("an cool idea" -> "a cool idea").

    A capital "A" is only treated as an article at the start of a sentence, so labels such
    as "Group A and B" are left alone.
    """
    if not text:
        return text

    def _repl(m: "re.Match[str]") -> str:
        art, gap, nxt = m.groups()
        if nxt.lower() in _NOT_A_NOUN_AFTER_ARTICLE:
            return m.group(0)
        if art[0].isupper():
            before = text[:m.start()].rstrip()
            if before and before[-1] not in ".!?":
                return m.group(0)
        want = indefinite_article(nxt)
        if art.lower() == want:
            return m.group(0)
        _letters = re.match(r"[A-Za-z]+", nxt)
        if (art.lower() == "an" and want == "a" and nxt == nxt.lower() and _letters
                and len(_letters.group(0)) <= 4 and nxt[0].lower() in "aefhilmnorsx"):
            return m.group(0)   # "an nfl game", "an hr thing", "an x-ray": probably a spelled-out initialism
        new = want.capitalize() if art[0].isupper() else want
        return new + gap + nxt

    return _ARTICLE_RE.sub(_repl, text)


# ---------------------------------------------------------------------------
# Clause-boundary editing
# ---------------------------------------------------------------------------

_CLAUSE_STARTERS = frozenset({
    "and", "but", "so", "because", "though", "although", "which", "while", "since",
    "then", "or", "yet", "if", "when",
})


_NOT_BEFORE_CLAUSE_STARTER = frozenset({
    "in", "to", "of", "for", "with", "at", "by", "on", "from", "into", "as", "than", "so", "even",
    "just", "not", "only", "ever", "a", "an", "the", "all", "some", "many", "most", "one", "such",
    "what", "how", "i", "we", "they", "you",
})


def clause_boundary_indices(words: Sequence[str]) -> List[int]:
    """Indices where a filler such as "you know," can go without splitting a phrase:
    right after a word that already ends in a comma, or just before a conjunction that follows a
    complete word group ("in which", "for a while", "so far", "even if" are never split).
    Sentence starts (previous word ends in . ! ?) are excluded: callers handle them themselves."""
    out: List[int] = []
    for i in range(2, len(words) - 1):
        prev = words[i - 1]
        if prev.endswith((".", "!", "?", "\u2026")) or words[i][:1].isupper() and words[i].lower().strip(",.") in _CLAUSE_STARTERS:
            continue
        if prev.endswith((",", ";", ":")):
            out.append(i)
        elif (words[i].lower().strip(",.") in _CLAUSE_STARTERS
              and re.sub(r"[^\w']", "", prev).lower() not in _NOT_BEFORE_CLAUSE_STARTER):
            out.append(i)
    return out


def insert_filler(words: List[str], filler: str, rng: random.Random) -> bool:
    """Insert ``filler`` ("you know", "I mean", ...) at a clause boundary, in place. Returns
    whether it did; with no boundary available the text is left untouched, not damaged."""
    spots = clause_boundary_indices(words)
    if not spots:
        return False
    idx = rng.choice(spots)
    core = filler.strip(", ")
    if not words[idx - 1].endswith((",", ";", ":")):
        words[idx - 1] = words[idx - 1] + ","
    words.insert(idx, core + ",")
    return True


_SUBJECT_PRONOUNS = frozenset({"i", "we", "they", "you", "i'm", "i've", "i'd", "i'll", "we're",
                               "they're", "you're", "we've", "they've", "you've"})
_AUX_AFTER_IT = frozenset({"is", "was", "feels", "felt", "seems", "seemed", "looks", "sounds",
                           "does", "did", "will", "would", "has", "had", "can", "could",
                           "makes", "made"})


def insert_adverb_after_subject(words: List[str], adverb: str) -> bool:
    """Put an adverb ("honestly", "kinda", "just") right after a leading subject pronoun,
    in place ("I think ..." -> "I honestly think ..."). Returns whether it did; text that does
    not start with a subject pronoun is left alone."""
    if len(words) < 3:
        return False
    first = _core(words[0]).lower().replace("\u2019", "'")
    nxt = _core(words[1]).lower()
    ok = first in _SUBJECT_PRONOUNS or (first in {"it", "this", "that"} and nxt in _AUX_AFTER_IT)
    if not ok or nxt == adverb.lower():
        return False
    words.insert(1, adverb)
    return True


_ENGLISH_FUNCTION_WORDS = frozenset({
    "the", "of", "to", "and", "a", "in", "is", "you", "that", "it", "for", "on", "are", "with",
    "as", "this", "be", "or", "by", "what", "how", "your", "please", "about", "do", "did", "was",
    "were", "have", "has", "why", "which", "when", "where", "who", "i", "my", "me", "we", "our",
    "they", "their", "not", "at", "from", "an", "can", "would", "if", "so", "than", "then",
    "there", "these", "those", "any", "some", "more", "us", "he", "she", "his", "her",
})


def looks_non_english(text: str) -> bool:
    """Heuristic: True for text of six or more words that is mostly not English (very few
    English function words, or many non-ASCII words). Short phrases are never flagged."""
    tokens = re.findall(r"[^\W\d_]+", (text or "").lower())
    if len(tokens) < 6:
        return False
    hits = sum(1 for t in tokens if t in _ENGLISH_FUNCTION_WORDS)
    non_ascii = sum(1 for t in tokens if not t.isascii())
    return hits / len(tokens) < 0.15 or non_ascii / len(tokens) > 0.25


# ---------------------------------------------------------------------------
# Opener detection (avoid stacking two openers: "Honestly I mean ...")
# ---------------------------------------------------------------------------

_OPENER_RE = re.compile(
    r"^\W*(?:(?:honestly|frankly|basically|overall|generally|actually|well|so|ok|okay|like|"
    r"look|hmm+|yeah|idk|tbh|ngl|lol|fr|lowkey|i mean|i think|i feel|i guess|i believe|i'd say|"
    r"i dunno|i suppose|i reckon|in my|for me|personally|to be fair|in general|as i see it|"
    r"my take|from my|on balance|um|uh|eh|meh|sure|fine|whatever|maybe|probably|perhaps|"
    r"absolutely|definitely|certainly|no way|strongly|this is exactly|couldn't agree)\b|\d+%(?=\s|$))",
    re.IGNORECASE,
)


def has_opener(text: str) -> bool:
    """True when the text already begins with a discourse marker or hedge."""
    return bool(text) and bool(_OPENER_RE.match(text))


# ---------------------------------------------------------------------------
# Context-checked word swaps
# ---------------------------------------------------------------------------

def _split_token(token: str):
    """Split a token into (leading punctuation, core word, trailing punctuation) in linear time."""
    body = token.lstrip("\"'([")
    lead = token[:len(token) - len(body)]
    core = body.rstrip(".,!?;:\"')]")
    return lead, core, body[len(core):]


def _match_case(original: str, replacement: str) -> str:
    if original[:1].isupper() and (len(original) == 1 or not original.isupper()):
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _is_negated(prev: str) -> bool:
    return prev in {"not", "never", "no", "hardly", "barely", "nt"} or prev.endswith("n't")


_BE_FORMS = frozenset({"is", "was", "are", "were", "be", "been", "it's", "that's"})
_CLAUSE_START = frozenset({
    "it", "it's", "its", "this", "that", "the", "there", "they", "we", "people", "most",
    "many", "everyone", "everybody", "my", "our", "if", "a", "an", "these", "those", "some",
    "he", "she", "you", "i", "what", "how",
})
_POSITIVE_NOUNS = frozenset({
    "experience", "view", "impression", "feeling", "feelings", "effect", "effects", "outcome",
    "outcomes", "attitude", "response", "reaction", "thing", "things", "way", "light", "sense",
    "opinion", "impact", "change", "influence", "result", "results", "one", "side", "message",
})
_GOOD_PREV = _BE_FORMS | {"feel", "feels", "felt", "seems", "seem", "seemed", "looks", "sounds",
                          "pretty", "really", "very", "so", "quite", "fairly", "a", "an",
                          "something", "too", "as"}
_NOT_GOOD_NEXT = frozenset({"news", "morning", "evening", "night", "luck", "afternoon", "day",
                            "bye", "grief", "faith", "will", "of", "samaritan"})


def _g_any(prev: str, nxt: str) -> bool:
    return True


def _g_good(prev: str, nxt: str) -> bool:
    return prev in _GOOD_PREV and nxt not in _NOT_GOOD_NEXT


def _g_positive(prev: str, nxt: str) -> bool:
    return nxt in _POSITIVE_NOUNS


def _g_concerned(prev: str, nxt: str) -> bool:
    return nxt in {"about", "that", "because", "by", "for"}


def _g_very(prev: str, nxt: str) -> bool:
    return not _is_negated(prev) and prev not in {"the", "this", "that", "a", "an", "at", "is", "was", "are"} and nxt not in {
        "first", "last", "same", "best", "least", "end", "beginning", "much", "many", "few", "own", ""}


def _g_really(prev: str, nxt: str) -> bool:
    return not _is_negated(prev) and nxt != ""


def _g_i_think(prev: str, nxt: str) -> bool:
    return prev == "i" and nxt in _CLAUSE_START and nxt not in {"a", "an"}   # "I think a lot about it"


def _g_i_understand(prev: str, nxt: str) -> bool:
    return prev == "i" and nxt != ""


def _g_because(prev: str, nxt: str) -> bool:
    # "is because", "just because", "not because" are fixed phrases that "since"/"cuz" would break
    return nxt not in {"of", ""} and prev not in {"is", "was", "are", "were", "be", "it's", "that's", "just", "not",
                                                      "only", "simply", "mainly", "partly", "mostly", "also"}


def _g_important(prev: str, nxt: str) -> bool:
    return not _is_negated(prev) and prev not in {"very", "so", "more", "most", "less", "too", "quite", "really"} and nxt not in {"to"}


def _g_opinion(prev: str, nxt: str) -> bool:
    return prev in {"my", "our", "your", "their", "his", "her", "an", "a", "the"} and nxt not in {"poll", "polls"}


# core word -> (casual choices, formal choices, guard(prev, next) -> bool). A swap only
# happens when the guard allows it, so a word is never replaced in a sense it cannot take.
_WORD_SWAPS = {
    "good": (["nice", "solid", "decent", "fine"], ["solid", "decent"], _g_good),
    "important": (["key", "huge", "a big deal"], ["key", "significant", "essential"], _g_important),
    "difficult": (["hard", "tough", "tricky"], ["hard", "challenging"], _g_any),
    "interesting": (["cool", "neat"], ["intriguing", "notable"], _g_any),
    "concerned": (["worried", "uneasy"], ["worried", "uneasy"], _g_concerned),
    "positive": (["good", "encouraging"], ["favorable", "encouraging"], _g_positive),
    "really": (["truly", "genuinely", "honestly"], ["truly", "genuinely"], _g_really),
    "very": (["pretty", "super", "quite"], ["quite", "rather"], _g_very),
    "think": (["feel", "guess", "reckon"], ["believe", "feel"], _g_i_think),
    "understand": (["get", "see"], ["see", "recognize"], _g_i_understand),
    "because": (["cuz", "cause"], ["since"], _g_because),
    "although": (["though", "even though"], ["though", "even though"], _g_any),
    "definitely": (["totally", "absolutely"], ["certainly", "absolutely"], _g_really),
    "opinion": (["view"], ["view", "perspective"], _g_opinion),
}


def swap_one_word(words: List[str], rng: random.Random, formal: bool) -> bool:
    """Replace ONE word with a grammar-safe synonym, in place. Returns whether it swapped.

    The article before the replaced word is corrected ("an interesting" -> "a cool").
    """
    cands = []
    for i, tok in enumerate(words):
        core = _split_token(tok)[1].lower()
        entry = _WORD_SWAPS.get(core)
        if not entry:
            continue
        prev = _split_token(words[i - 1])[1].lower() if i > 0 else ""
        nxt = _split_token(words[i + 1])[1].lower() if i + 1 < len(words) else ""
        if entry[2](prev, nxt):
            cands.append(i)
    if not cands:
        return False
    i = rng.choice(cands)
    lead, core, trail = _split_token(words[i])
    casual_choices, formal_choices, _ = _WORD_SWAPS[core.lower()]
    pool = list(formal_choices if formal else casual_choices)
    prev = _split_token(words[i - 1])[1].lower() if i > 0 else ""
    if prev not in _BE_FORMS:  # "a big deal" only fits right after a form of "be"
        pool = [c for c in pool if c != "a big deal"] or pool
    new = _match_case(core, rng.choice(pool))
    words[i] = lead + new + trail
    if i > 0 and words[i - 1].lower() in ("a", "an"):
        words[i - 1] = _match_case(words[i - 1], indefinite_article(new))
    return True


_DROPPABLE = frozenset({"really", "very", "just", "actually", "basically", "literally",
                        "quite", "pretty", "simply", "truly"})


def drop_one_optional_word(words: List[str], rng: random.Random) -> bool:
    """Remove ONE optional intensifier ("really", "just", ...), in place. Only bare words
    (no attached punctuation) are removed, never the first or last word, and never after a
    negation or article where the word carries meaning."""
    cands = []
    for i in range(1, len(words) - 1):
        if words[i].lower() not in _DROPPABLE:
            continue
        prev = _split_token(words[i - 1])[1].lower()
        if _is_negated(prev) or prev in {"a", "an", "the", "more", "less", "very", "so"}:
            continue
        if words[i - 1].endswith((",", ";", ":")):
            continue
        if words[i].lower() == "pretty" and _split_token(words[i + 1])[1].lower() == "much":
            continue
        cands.append(i)
    if not cands:
        return False
    words.pop(rng.choice(cands))
    return True


# ---------------------------------------------------------------------------
# Contractions
# ---------------------------------------------------------------------------

# (long form, short form, regex lookahead that must hold to use the short form for the long
# one, regex lookahead that must hold to expand the short form).
_NEXT_WORD = r"(?=\s+\w)"        # a copula cannot be contracted at the end of a clause
_NOT_PERFECT = r"(?!\s+(?:been|got|gotten)\b)"   # "it's been" is "it has", not "it is"
_CONTRACTIONS = [
    ("do not", "don't", "", ""), ("does not", "doesn't", "", ""), ("did not", "didn't", "", ""),
    ("is not", "isn't", "", ""), ("was not", "wasn't", "", ""), ("are not", "aren't", "", ""),
    ("have not", "haven't", "", ""), ("has not", "hasn't", "", ""),
    ("would not", "wouldn't", "", ""), ("could not", "couldn't", "", ""),
    ("should not", "shouldn't", "", ""), ("will not", "won't", "", ""), ("cannot", "can't", "", ""),
    ("I am", "I'm", _NEXT_WORD, ""), ("it is", "it's", _NEXT_WORD, _NOT_PERFECT),
    ("that is", "that's", _NEXT_WORD, _NOT_PERFECT), ("I will", "I'll", _NEXT_WORD, ""),
    ("they are", "they're", _NEXT_WORD, ""), ("we are", "we're", _NEXT_WORD, ""),
]


_CONTRACTION_PATTERNS = {}
for _long, _short, _cg, _eg in _CONTRACTIONS:
    _CONTRACTION_PATTERNS[(_long, False)] = re.compile(r"(?<![\w'])" + re.escape(_long) + r"(?![\w'])" + _cg, re.IGNORECASE)
    _CONTRACTION_PATTERNS[(_short, True)] = re.compile(r"(?<![\w'])" + re.escape(_short) + r"(?![\w'])" + _eg, re.IGNORECASE)


def apply_contractions(text: str, rng: random.Random, expand: bool, prob: float = 0.6) -> str:
    """Contract ("do not" -> "don't") or expand the reverse way, each pair with probability
    ``prob``. Whole-word matching; an expansion that would be wrong ("it's been") is skipped."""
    for long_form, short_form, contract_guard, expand_guard in _CONTRACTIONS:
        if expand:
            src, dst, guard = short_form, long_form, expand_guard
        else:
            src, dst, guard = long_form, short_form, contract_guard
        pat = _CONTRACTION_PATTERNS[(src, expand)]
        if not pat.search(text) or rng.random() >= prob:
            continue

        def _sub(m: "re.Match[str]", dst=dst) -> str:
            g = m.group(0)
            if g[:1].islower() and dst[:1].isupper():
                return dst[:1].lower() + dst[1:]
            if g[:1].isupper() and dst[:1].islower():
                return dst[:1].upper() + dst[1:]
            return dst

        text = pat.sub(_sub, text, count=1)
    return text


# ---------------------------------------------------------------------------
# Filler clean-up and final tidy
# ---------------------------------------------------------------------------

_FILLER_WORDS = (r"honestly|literally|basically|actually|genuinely|you know|i mean|kind of|"
                 r"sort of|totally|frankly|obviously|seriously|truly")
_FUNCTION_BEFORE_FILLER = (r"when it|as far as i|it|the|a|an|of|to|about|and|or|but|with|in|on|"
                           r"at|for|my|your|their|our|i|we|they|you|is|are|was")
_MISPLACED_FILLER_RE = re.compile(
    r"\b(" + _FUNCTION_BEFORE_FILLER + r")\s+(?:" + _FILLER_WORDS + r")\s*,\s*", re.IGNORECASE)
_REPEATED_FILLER_RE = re.compile(r"\b(" + _FILLER_WORDS + r"|like|well|right)\s*,\s*\1\s*,", re.IGNORECASE)


def remove_misplaced_fillers(text: str) -> str:
    """Delete a filler that sits directly after a function word with a comma after it
    ("when it honestly, comes" -> "when it comes"), and collapse a repeated one
    ("you know, you know," -> "you know,"). Fillers at clause boundaries are untouched."""
    if not text:
        return text
    t = _MISPLACED_FILLER_RE.sub(lambda m: m.group(1) + " ", text)
    return _REPEATED_FILLER_RE.sub(lambda m: m.group(1) + ",", t)


def tidy_spacing(text: str) -> str:
    """Collapse repeated spaces and remove stray spaces before punctuation."""
    if not text:
        return text
    t = re.sub(r"(?<=\S)\s+(?=[,;:!?])", "", text)
    t = re.sub(r"(?<=\S)\s+\.(?!\.)", ".", t)
    t = re.sub(r",(?:\s*,)+", ",", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


_ENGLISH_ONLY_WORDS = frozenset({
    "the", "and", "is", "of", "that", "was", "with", "this", "are", "not", "my", "it", "to", "for",
    "be", "have", "i", "you", "they", "think", "what", "when", "how", "about", "we", "but", "if",
    "he", "she", "there", "would", "could", "will", "had", "been", "were", "your", "their", "them",
    "which", "who", "just", "very", "more", "also", "than", "then", "these", "those", "because",
    "really", "feel", "know", "honestly", "actually", "basically",
})


def looks_english(text: str) -> bool:
    """True when a text contains at least three distinct words that are English-only function
    words. English fix-ups ("a"/"an", cut-off endings) must not touch other languages: in
    Spanish or Italian "a" is a preposition ("voy a España")."""
    tokens = set(re.findall(r"[^\W\d_]+", (text or "").lower()))
    return len(tokens & _ENGLISH_ONLY_WORDS) >= 3


def finalize_generated_text(text: str) -> str:
    """Last deterministic pass over one generated answer: spacing, a/an agreement, misplaced
    fillers and a cut-off tail. Safe to call more than once. Only spacing is repaired when the
    text does not look English."""
    if not text:
        return text
    if not looks_english(text):
        return tidy_spacing(text)
    return trim_dangling_tail(fix_indefinite_articles(tidy_spacing(remove_misplaced_fillers(text))))
