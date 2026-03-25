"""
HBS Tool Dispatcher — Programmatic tool-use layer for the Human Behavior Simulator.

Inspired by the DANEEL+ architecture, this module provides Python functions that
handle tasks LLMs systematically fail at: character counting, arithmetic, image
geometry analysis, and temporal queries.

All tools use pure Python stdlib — no external dependencies required.
"""

import ast
import logging
import operator
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["HBSTool", "HBSToolDispatcher"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UTC offsets for common US timezone names (no pytz dependency)
# ---------------------------------------------------------------------------
_TIMEZONE_OFFSETS: Dict[str, int] = {
    "UTC": 0,
    "GMT": 0,
    "US/Eastern": -5,
    "US/Central": -6,
    "US/Mountain": -7,
    "US/Pacific": -8,
    "US/Alaska": -9,
    "US/Hawaii": -10,
    "EST": -5,
    "CST": -6,
    "MST": -7,
    "PST": -8,
    "ET": -5,
    "CT": -6,
    "MT": -7,
    "PT": -8,
}

# ---------------------------------------------------------------------------
# Factual lookup tables
# ---------------------------------------------------------------------------
_US_STATE_CAPITALS: Dict[str, str] = {
    "california": "Sacramento",
    "texas": "Austin",
    "florida": "Tallahassee",
    "new york": "Albany",
    "pennsylvania": "Harrisburg",
    "illinois": "Springfield",
    "ohio": "Columbus",
    "georgia": "Atlanta",
    "north carolina": "Raleigh",
    "michigan": "Lansing",
    "new jersey": "Trenton",
    "virginia": "Richmond",
    "washington": "Olympia",
    "arizona": "Phoenix",
    "massachusetts": "Boston",
    "tennessee": "Nashville",
    "indiana": "Indianapolis",
    "maryland": "Annapolis",
    "minnesota": "Saint Paul",
    "missouri": "Jefferson City",
    "wisconsin": "Madison",
    "colorado": "Denver",
    "alabama": "Montgomery",
    "south carolina": "Columbia",
    "louisiana": "Baton Rouge",
    "kentucky": "Frankfort",
    "oregon": "Salem",
    "oklahoma": "Oklahoma City",
    "connecticut": "Hartford",
    "iowa": "Des Moines",
    "utah": "Salt Lake City",
    "nevada": "Carson City",
    "arkansas": "Little Rock",
    "mississippi": "Jackson",
    "kansas": "Topeka",
    "new mexico": "Santa Fe",
    "nebraska": "Lincoln",
    "hawaii": "Honolulu",
    "idaho": "Boise",
    "west virginia": "Charleston",
    "maine": "Augusta",
    "new hampshire": "Concord",
    "montana": "Helena",
    "rhode island": "Providence",
    "delaware": "Dover",
    "south dakota": "Pierre",
    "north dakota": "Bismarck",
    "alaska": "Juneau",
    "vermont": "Montpelier",
    "wyoming": "Cheyenne",
}

_DAYS_IN_MONTH: Dict[str, int] = {
    "january": 31, "february": 28, "march": 31, "april": 30,
    "may": 31, "june": 30, "july": 31, "august": 31,
    "september": 30, "october": 31, "november": 30, "december": 31,
}

_CONTINENTS: List[str] = [
    "Africa", "Antarctica", "Asia", "Australia/Oceania",
    "Europe", "North America", "South America",
]

_OCEANS: List[str] = [
    "Pacific Ocean", "Atlantic Ocean", "Indian Ocean",
    "Southern Ocean", "Arctic Ocean",
]

# ---------------------------------------------------------------------------
# Safe arithmetic AST walker
# ---------------------------------------------------------------------------
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


def _safe_eval_ast(node: ast.AST, steps: List[str]) -> float:
    """Recursively evaluate a whitelisted AST node tree.

    Only numeric literals and the operators listed in ``_ALLOWED_OPERATORS``
    are permitted.  Raises ``ValueError`` for anything else.
    """
    if isinstance(node, ast.Expression):
        return _safe_eval_ast(node.body, steps)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    # Python < 3.8 compat (ast.Num deprecated but may appear)
    if isinstance(node, ast.Num):  # type: ignore[attr-defined]
        return float(node.n)  # type: ignore[attr-defined]

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _safe_eval_ast(node.operand, steps)
        result = _ALLOWED_OPERATORS[op_type](operand)
        steps.append(f"{op_type.__name__}({operand}) = {result}")
        return result

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval_ast(node.left, steps)
        right = _safe_eval_ast(node.right, steps)

        # Guard against excessively large exponents
        if op_type is ast.Pow and abs(right) > 1000:
            raise ValueError(f"Exponent too large: {right}")

        result = _ALLOWED_OPERATORS[op_type](left, right)

        op_symbols = {
            ast.Add: "+", ast.Sub: "-", ast.Mult: "*",
            ast.Div: "/", ast.Pow: "**", ast.FloorDiv: "//",
            ast.Mod: "%",
        }
        symbol = op_symbols.get(op_type, op_type.__name__)
        steps.append(f"{left} {symbol} {right} = {result}")
        return result

    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


# ---------------------------------------------------------------------------
# Keyword patterns for tool-hint detection
# ---------------------------------------------------------------------------
_LETTER_COUNT_RE = re.compile(
    r"how\s+many\s+['\"]?([a-zA-Z])['\"]?(?:'?s)?\s+(?:are\s+)?(?:in|does)\b",
    re.IGNORECASE,
)
_WORD_COUNT_RE = re.compile(
    r"how\s+many\s+words?\b",
    re.IGNORECASE,
)
_NTH_WORD_RE = re.compile(
    r"(?:what\s+is\s+the\s+)?(\d+)(?:st|nd|rd|th)\s+word\b",
    re.IGNORECASE,
)
_ARITHMETIC_RE = re.compile(
    r"(?:what\s+is\s+|calculate\s+|compute\s+|solve\s+)?"
    r"[\d]+\s*[+\-*/^x\u00d7\u00f7]\s*[\d]",
    re.IGNORECASE,
)
_DATE_TIME_RE = re.compile(
    r"\b(?:what\s+(?:day|date|time|year)|current\s+(?:date|time|day|year)"
    r"|today(?:'s)?\s+date|day\s+of\s+(?:the\s+)?week)\b",
    re.IGNORECASE,
)
_PRESIDENT_RE = re.compile(
    r"\b(?:who\s+is\s+(?:the\s+)?(?:current\s+)?president|current\s+president)\b",
    re.IGNORECASE,
)
_CAPITAL_RE = re.compile(
    r"\b(?:capital\s+of|what\s+is\s+the\s+capital)\b",
    re.IGNORECASE,
)


# ===================================================================
# HBSTool — static methods for each tool capability
# ===================================================================
class HBSTool:
    """Collection of deterministic tool functions.

    Every method is static, side-effect free, and uses only stdlib.
    """

    # ---------------------------------------------------------------
    # 1. Text analysis
    # ---------------------------------------------------------------
    @staticmethod
    def analyze_text(text: str, query: str) -> Dict[str, Any]:
        """Character-level and word-level text analysis.

        Handles:
        - Letter counting (``how many b's in blueberry``)
        - Nth-word extraction (``3rd word``)
        - Word counting (``how many words``)

        Parameters
        ----------
        text : str
            The text to analyze.
        query : str
            Natural-language question about *text*.

        Returns
        -------
        dict
            ``letter_counts``, ``word_count``, ``target_count``, ``answer``
        """
        if not text:
            return {
                "letter_counts": {},
                "word_count": 0,
                "target_count": 0,
                "answer": "Empty text provided.",
            }

        text_lower = text.lower()
        query_lower = query.lower() if query else ""

        # Full letter frequency
        letter_counts: Dict[str, int] = {}
        for ch in text_lower:
            if ch.isalpha():
                letter_counts[ch] = letter_counts.get(ch, 0) + 1

        words = text.split()
        word_count = len(words)

        # --- Determine what the query asks for ---
        target_count = 0
        answer = ""

        # Case 1: count a specific letter
        letter_match = _LETTER_COUNT_RE.search(query_lower)
        if letter_match:
            target_letter = letter_match.group(1).lower()
            target_count = letter_counts.get(target_letter, 0)
            answer = (
                f"The letter '{target_letter}' appears {target_count} time(s) "
                f"in \"{text}\"."
            )
            return {
                "letter_counts": letter_counts,
                "word_count": word_count,
                "target_count": target_count,
                "answer": answer,
            }

        # Also handle simpler patterns like "count the letter r in ..."
        simple_letter = re.search(
            r"(?:count|number\s+of)\s+(?:the\s+)?(?:letter\s+)?['\"]?([a-zA-Z])['\"]?(?:'?s)?\s+in\b",
            query_lower,
        )
        if simple_letter:
            target_letter = simple_letter.group(1).lower()
            target_count = letter_counts.get(target_letter, 0)
            answer = (
                f"The letter '{target_letter}' appears {target_count} time(s) "
                f"in \"{text}\"."
            )
            return {
                "letter_counts": letter_counts,
                "word_count": word_count,
                "target_count": target_count,
                "answer": answer,
            }

        # Case 2: nth word
        nth_match = _NTH_WORD_RE.search(query_lower)
        if nth_match:
            n = int(nth_match.group(1))
            if 1 <= n <= word_count:
                target_word = words[n - 1]
                answer = f"The {n}{'st' if n == 1 else 'nd' if n == 2 else 'rd' if n == 3 else 'th'} word is \"{target_word}\"."
            else:
                answer = f"There is no {n}th word — the text has only {word_count} word(s)."
            return {
                "letter_counts": letter_counts,
                "word_count": word_count,
                "target_count": 0,
                "answer": answer,
            }

        # Case 3: word count
        if _WORD_COUNT_RE.search(query_lower):
            answer = f"The text contains {word_count} word(s)."
            return {
                "letter_counts": letter_counts,
                "word_count": word_count,
                "target_count": word_count,
                "answer": answer,
            }

        # Default: return full letter frequency
        answer = (
            f"Text has {word_count} word(s). "
            f"Letter frequencies: {dict(sorted(letter_counts.items()))}."
        )
        return {
            "letter_counts": letter_counts,
            "word_count": word_count,
            "target_count": 0,
            "answer": answer,
        }

    # ---------------------------------------------------------------
    # 2. Arithmetic
    # ---------------------------------------------------------------
    @staticmethod
    def compute_arithmetic(expression: str) -> Dict[str, Any]:
        """Safe arithmetic evaluation using AST whitelisting.

        Handles ``+``, ``-``, ``*``, ``/``, ``**`` (power), parentheses,
        and common Unicode operator substitutions.

        Parameters
        ----------
        expression : str
            Mathematical expression (e.g. ``"2 + 3 * 4"``).

        Returns
        -------
        dict
            ``result``, ``formatted``, ``steps``
        """
        if not expression or not expression.strip():
            return {
                "result": None,
                "formatted": "Error: empty expression",
                "steps": [],
            }

        # Normalise the expression
        expr = expression.strip()
        # Replace Unicode operators
        expr = expr.replace("\u00d7", "*")  # ×
        expr = expr.replace("\u00f7", "/")  # ÷
        expr = expr.replace("^", "**")
        # Replace 'x' used as multiplication when between digits
        expr = re.sub(r"(\d)\s*x\s*(\d)", r"\1*\2", expr)
        # Strip trailing '=' if someone writes "2+3="
        expr = expr.rstrip("= ")

        steps: List[str] = []

        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            return {
                "result": None,
                "formatted": f"Syntax error: {exc}",
                "steps": [],
            }

        try:
            result = _safe_eval_ast(tree, steps)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
            return {
                "result": None,
                "formatted": f"Evaluation error: {exc}",
                "steps": steps,
            }

        # Use Decimal for clean formatting when possible
        try:
            d = Decimal(str(result)).quantize(
                Decimal("0.0000000001"), rounding=ROUND_HALF_UP
            ).normalize()
            formatted = str(d)
        except (InvalidOperation, ValueError):
            formatted = str(result)

        # If the result is effectively an integer, show without decimals
        if isinstance(result, float) and result == int(result) and abs(result) < 1e15:
            formatted = str(int(result))

        return {
            "result": result,
            "formatted": formatted,
            "steps": steps,
        }

    # ---------------------------------------------------------------
    # 3. Date / time
    # ---------------------------------------------------------------
    @staticmethod
    def get_current_datetime(tz: str = "US/Eastern") -> Dict[str, Any]:
        """Return current date/time information.

        Parameters
        ----------
        tz : str
            Timezone name (e.g. ``"US/Eastern"``, ``"UTC"``).

        Returns
        -------
        dict
            ``year``, ``month``, ``day_of_week``, ``time_12h``,
            ``date_formatted``, ``timezone``
        """
        offset_hours = _TIMEZONE_OFFSETS.get(tz)
        if offset_hours is None:
            # Try case-insensitive lookup
            tz_lower = tz.lower().strip()
            for known, hours in _TIMEZONE_OFFSETS.items():
                if known.lower() == tz_lower:
                    offset_hours = hours
                    break
            if offset_hours is None:
                # Fall back to UTC
                offset_hours = 0
                tz = "UTC (fallback)"

        utc_now = datetime.now(timezone.utc)
        local_now = utc_now + timedelta(hours=offset_hours)

        month_name = local_now.strftime("%B")
        day_of_week = local_now.strftime("%A")
        time_12h = local_now.strftime("%I:%M %p").lstrip("0")
        date_formatted = local_now.strftime("%B %d, %Y")

        return {
            "year": local_now.year,
            "month": month_name,
            "day": local_now.day,
            "day_of_week": day_of_week,
            "time_12h": time_12h,
            "date_formatted": date_formatted,
            "timezone": tz,
        }

    # ---------------------------------------------------------------
    # 4. Factual lookup
    # ---------------------------------------------------------------
    @staticmethod
    def lookup_factual(query_type: str, key: str = "") -> Dict[str, Any]:
        """Lookup table for common factual questions.

        Parameters
        ----------
        query_type : str
            One of ``"current_president"``, ``"days_in_month"``,
            ``"state_capital"``, ``"basic_geography"``.
        key : str
            Qualifier (e.g. month name, state name, ``"continents"``).

        Returns
        -------
        dict
            ``answer``, ``source``, ``confidence``
        """
        qt = query_type.lower().strip()
        k = key.lower().strip()

        if qt == "current_president":
            return {
                "answer": "Donald Trump",
                "source": "Inaugurated January 20, 2025 (47th President)",
                "confidence": 1.0,
            }

        if qt == "days_in_month":
            days = _DAYS_IN_MONTH.get(k)
            if days is not None:
                note = " (28 in common years, 29 in leap years)" if k == "february" else ""
                return {
                    "answer": str(days) + note,
                    "source": "Gregorian calendar",
                    "confidence": 1.0,
                }
            return {
                "answer": "unknown",
                "source": f"Month '{key}' not recognized",
                "confidence": 0.0,
            }

        if qt == "state_capital":
            capital = _US_STATE_CAPITALS.get(k)
            if capital is not None:
                return {
                    "answer": capital,
                    "source": "US state capitals reference",
                    "confidence": 1.0,
                }
            return {
                "answer": "unknown",
                "source": f"State '{key}' not recognized",
                "confidence": 0.0,
            }

        if qt == "basic_geography":
            if k in ("continents", "continent"):
                return {
                    "answer": ", ".join(_CONTINENTS),
                    "source": "Standard geographic classification (7 continents)",
                    "confidence": 1.0,
                }
            if k in ("oceans", "ocean"):
                return {
                    "answer": ", ".join(_OCEANS),
                    "source": "Standard geographic classification (5 oceans)",
                    "confidence": 1.0,
                }
            return {
                "answer": "unknown",
                "source": f"Geography key '{key}' not recognized. Try 'continents' or 'oceans'.",
                "confidence": 0.0,
            }

        return {
            "answer": "unknown",
            "source": f"Query type '{query_type}' not recognized. "
                      f"Available: current_president, days_in_month, state_capital, basic_geography.",
            "confidence": 0.0,
        }

    # ---------------------------------------------------------------
    # 5. Object counting from description
    # ---------------------------------------------------------------
    @staticmethod
    def count_objects_description(description: str) -> Dict[str, Any]:
        """Parse a textual description and count mentioned objects.

        This does NOT do image processing — it extracts numbers and object
        nouns from a natural-language description of a scene.

        Parameters
        ----------
        description : str
            Natural-language description (e.g. ``"3 red apples and 2 oranges"``).

        Returns
        -------
        dict
            ``count``, ``objects``, ``confidence``
        """
        if not description or not description.strip():
            return {"count": 0, "objects": "", "confidence": 0.0}

        desc = description.strip()

        # Pattern: "<number> <object(s)>"
        pattern = re.compile(
            r"\b(\d+)\s+([a-zA-Z][a-zA-Z\s]{0,30}?)(?:\s+and\s+|\s*,\s*|\s*;\s*|$|\.|!)",
            re.IGNORECASE,
        )

        total_count = 0
        object_parts: List[str] = []

        for m in pattern.finditer(desc):
            num = int(m.group(1))
            obj = m.group(2).strip().rstrip(",;.")
            total_count += num
            object_parts.append(f"{num} {obj}")

        # If no pattern matched, look for standalone numbers
        if not object_parts:
            numbers = re.findall(r"\b(\d+)\b", desc)
            if numbers:
                total_count = sum(int(n) for n in numbers)
                return {
                    "count": total_count,
                    "objects": f"numbers found: {', '.join(numbers)}",
                    "confidence": 0.4,
                }
            return {
                "count": 0,
                "objects": "No countable objects detected in description.",
                "confidence": 0.1,
            }

        # Also handle written-out number words
        word_numbers = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12,
        }
        word_pattern = re.compile(
            r"\b(" + "|".join(word_numbers.keys()) + r")\s+([a-zA-Z][a-zA-Z\s]{0,30}?)(?:\s+and\s+|\s*,\s*|$|\.|!)",
            re.IGNORECASE,
        )
        for m in word_pattern.finditer(desc):
            num = word_numbers[m.group(1).lower()]
            obj = m.group(2).strip().rstrip(",;.")
            total_count += num
            object_parts.append(f"{num} {obj}")

        objects_summary = "; ".join(object_parts) if object_parts else ""
        confidence = min(0.95, 0.6 + 0.1 * len(object_parts))

        return {
            "count": total_count,
            "objects": objects_summary,
            "confidence": confidence,
        }


# ===================================================================
# HBSToolDispatcher — routes tool calls and provides tool hints
# ===================================================================
class HBSToolDispatcher:
    """Routes named tool calls to :class:`HBSTool` methods and provides
    question-level tool hints for the simulation pipeline.
    """

    _TOOL_MAP: Dict[str, str] = {
        "analyze_text": "analyze_text",
        "compute_arithmetic": "compute_arithmetic",
        "get_current_datetime": "get_current_datetime",
        "lookup_factual": "lookup_factual",
        "count_objects_description": "count_objects_description",
    }

    @classmethod
    def dispatch(cls, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        """Route a tool call to the appropriate :class:`HBSTool` method.

        Parameters
        ----------
        tool_name : str
            One of the registered tool names.
        **kwargs
            Arguments forwarded to the tool function.

        Returns
        -------
        dict
            Tool result, or an error dict if the tool is unknown or fails.
        """
        if tool_name not in cls._TOOL_MAP:
            return {
                "error": f"Unknown tool: '{tool_name}'",
                "available": list(cls._TOOL_MAP.keys()),
            }

        method_name = cls._TOOL_MAP[tool_name]
        method = getattr(HBSTool, method_name)

        try:
            return method(**kwargs)
        except TypeError as exc:
            logger.error("Tool '%s' argument error: %s", tool_name, exc)
            return {
                "error": f"Invalid arguments for '{tool_name}': {exc}",
                "available_params": list(method.__code__.co_varnames[
                    :method.__code__.co_argcount
                ]),
            }
        except Exception as exc:
            logger.error("Tool '%s' execution error: %s", tool_name, exc, exc_info=True)
            return {"error": f"Execution error in '{tool_name}': {exc}"}

    @classmethod
    def get_tool_hint(
        cls,
        question_text: str,
        question_classification: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Determine which tool (if any) should be invoked for a question.

        Parameters
        ----------
        question_text : str
            The survey / task question text.
        question_classification : dict, optional
            Pre-computed classification that may contain a ``tool_hint`` key.

        Returns
        -------
        str or None
            Tool name to invoke, or ``None`` if no tool is applicable.
        """
        # Priority 1: explicit hint from classification
        if question_classification and question_classification.get("tool_hint"):
            hint = question_classification["tool_hint"]
            if hint in cls._TOOL_MAP:
                return hint

        if not question_text:
            return None

        q = question_text.lower()

        # Letter / word counting
        if _LETTER_COUNT_RE.search(q) or _WORD_COUNT_RE.search(q) or _NTH_WORD_RE.search(q):
            return "analyze_text"
        if re.search(r"\bcount\s+(?:the\s+)?(?:letters?|characters?)\b", q, re.IGNORECASE):
            return "analyze_text"

        # Arithmetic
        if _ARITHMETIC_RE.search(q):
            return "compute_arithmetic"
        if re.search(r"\b(?:sum|product|difference|quotient)\s+of\s+\d", q, re.IGNORECASE):
            return "compute_arithmetic"

        # Date / time
        if _DATE_TIME_RE.search(q):
            return "get_current_datetime"

        # Factual: president
        if _PRESIDENT_RE.search(q):
            return "lookup_factual"

        # Factual: state capital
        if _CAPITAL_RE.search(q):
            return "lookup_factual"

        # Factual: continents / oceans
        if re.search(r"\bhow\s+many\s+(?:continents|oceans)\b", q, re.IGNORECASE):
            return "lookup_factual"

        # Object counting
        if re.search(r"\bhow\s+many\s+(?:objects?|items?|things?)\b", q, re.IGNORECASE):
            return "count_objects_description"

        return None
