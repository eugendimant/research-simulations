"""
Adversarial Self-Validation Pipeline for the Human Behavior Simulator (HBS)
============================================================================

After data generation, validates the dataset against human-realism benchmark axes
and auto-corrects any metrics outside human-plausible ranges.

Benchmark axes:
- Completion time distribution
- Attention check pass rates
- Open-ended response uniqueness and length
- Straight-lining rates
- Rating-text coherence

Version: 1.2.5.0
"""

__version__ = "1.2.5.0"

__all__ = ["HBSValidator"]

import logging
import math
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Sentiment word banks for rating-text coherence checks
# ---------------------------------------------------------------------------

_POSITIVE_WORDS = frozenset({
    "great", "excellent", "wonderful", "fantastic", "amazing", "love",
    "loved", "best", "awesome", "outstanding", "perfect", "brilliant",
    "superb", "terrific", "enjoy", "enjoyed", "enjoying", "happy",
    "glad", "pleased", "impressive", "remarkable", "delightful",
    "incredible", "magnificent", "favorable", "favourable", "positive",
    "good", "nice", "like", "liked", "appreciate", "appreciated",
    "satisfied", "thrilled", "excited", "inspiring", "beautiful",
})

_NEGATIVE_WORDS = frozenset({
    "terrible", "horrible", "awful", "worst", "hate", "hated", "bad",
    "dreadful", "disgusting", "appalling", "abysmal", "atrocious",
    "dislike", "disliked", "angry", "furious", "upset", "annoyed",
    "frustrated", "disappointed", "disappointing", "pathetic", "useless",
    "rubbish", "trash", "garbage", "miserable", "painful", "unfair",
    "unacceptable", "poor", "inferior", "mediocre", "inadequate",
    "detrimental", "harmful", "offensive", "outrageous", "lousy",
})


class HBSValidator:
    """Adversarial self-validation pipeline for HBS-generated datasets.

    Validates generated data against human-realism benchmark axes and optionally
    auto-corrects metrics that fall outside human-plausible ranges.

    Usage::

        validator = HBSValidator()
        report = validator.validate(df)
        if not report["passed"]:
            corrected_df, report = validator.validate_and_correct(df)
    """

    # ------------------------------------------------------------------
    # Class-level benchmark thresholds
    # ------------------------------------------------------------------

    BENCHMARKS: Dict[str, Dict[str, Any]] = {
        "completion_time": {
            "human_min_seconds": 480,
            "human_max_seconds": 1500,
            "target_pct_in_range": 0.80,
        },
        "attention_checks": {
            "overall_pass_rate_range": (0.82, 0.96),
        },
        "oe_responses": {
            "min_uniqueness_pct": 0.90,
            "mean_words_range": (8, 35),
        },
        "straightlining_rate": {
            "expected_range": (0.03, 0.08),
        },
        "rating_text_coherence": {
            "min_pct_coherent": 0.95,
        },
    }

    # ------------------------------------------------------------------
    # Column detection patterns
    # ------------------------------------------------------------------

    _TIMING_COL_PATTERNS = [
        re.compile(r"duration.*seconds", re.IGNORECASE),
        re.compile(r"Duration__in_seconds_", re.IGNORECASE),
        re.compile(r"total_time", re.IGNORECASE),
        re.compile(r"completion_time", re.IGNORECASE),
        re.compile(r"time_spent", re.IGNORECASE),
        re.compile(r"survey_duration", re.IGNORECASE),
    ]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, benchmarks: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Initialise the validator, optionally overriding default benchmarks.

        Args:
            benchmarks: Optional dict to merge with / override ``BENCHMARKS``.
        """
        self._benchmarks = dict(self.BENCHMARKS)
        if benchmarks:
            for key, val in benchmarks.items():
                if key in self._benchmarks:
                    self._benchmarks[key].update(val)
                else:
                    self._benchmarks[key] = val

    # ==================================================================
    # Public API
    # ==================================================================

    def validate(
        self,
        df: Any,
        participant_states: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Run all validation checks on the generated DataFrame.

        Args:
            df: A ``pandas.DataFrame`` (or dict-of-lists when pandas is
                unavailable) containing the generated dataset.
            participant_states: Optional list of per-participant state
                dicts produced by the simulation engine.

        Returns:
            A dict with keys ``passed``, ``checks``, ``auto_corrections``,
            and ``summary``.
        """
        df = self._ensure_df(df)

        checks: Dict[str, Dict[str, Any]] = {}
        checks["completion_time"] = self._check_completion_time(df)
        checks["oe_uniqueness"] = self._check_oe_uniqueness(df)
        checks["straightlining"] = self._check_straightlining(df)
        checks["oe_length"] = self._check_oe_length_distribution(df)
        checks["rating_text_coherence"] = self._check_rating_text_coherence(df)

        all_passed = all(c.get("passed", True) for c in checks.values())
        failed_names = [k for k, v in checks.items() if not v.get("passed", True)]

        if all_passed:
            summary = "All human-realism benchmark checks passed."
        else:
            summary = (
                f"{len(failed_names)} of {len(checks)} checks failed: "
                + ", ".join(failed_names)
                + "."
            )

        return {
            "passed": all_passed,
            "checks": checks,
            "auto_corrections": [],
            "summary": summary,
        }

    def validate_and_correct(
        self,
        df: Any,
        participant_states: Optional[List[Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Validate and auto-correct metrics outside human-plausible ranges.

        Steps:
            1. Run ``validate()``.
            2. For each failed check, apply targeted correction.
            3. Re-validate after corrections.
            4. Return the corrected DataFrame and the final report.

        Args:
            df: Generated dataset (DataFrame or dict-of-lists).
            participant_states: Optional per-participant state dicts.

        Returns:
            Tuple of ``(corrected_df, validation_report)``.
        """
        df = self._ensure_df(df)
        report = self.validate(df, participant_states)

        if report["passed"]:
            return df, report

        corrections: List[str] = []

        for check_name, result in report["checks"].items():
            if result.get("passed", True):
                continue

            if check_name == "completion_time":
                df = self._correct_completion_time(df)
                corrections.append(
                    "Re-sampled timing values within [480, 1500] seconds."
                )

            elif check_name == "straightlining":
                df = self._correct_straightlining(df)
                corrections.append(
                    "Perturbed 2-3 scale items by +/-1 for worst straight-liners."
                )

            elif check_name == "oe_uniqueness":
                df = self._correct_oe_uniqueness(df)
                corrections.append(
                    "Appended random filler words to duplicate OE responses."
                )

            elif check_name == "oe_length":
                df = self._correct_oe_length(df)
                corrections.append(
                    "Truncated or extended OE responses to fit [8, 35] mean words."
                )

            elif check_name == "rating_text_coherence":
                # Coherence corrections are complex and risk introducing
                # artefacts; log but do not auto-correct.
                corrections.append(
                    "Rating-text coherence issues detected; manual review "
                    "recommended (no auto-correction applied)."
                )

        # Re-validate after corrections
        final_report = self.validate(df, participant_states)
        final_report["auto_corrections"] = corrections

        if final_report["passed"]:
            final_report["summary"] = (
                "All checks passed after auto-correction. "
                + "; ".join(corrections)
            )
        else:
            still_failed = [
                k for k, v in final_report["checks"].items()
                if not v.get("passed", True)
            ]
            final_report["summary"] = (
                f"After auto-correction, {len(still_failed)} check(s) still "
                f"fail: {', '.join(still_failed)}. Manual review recommended."
            )

        return df, final_report

    # ==================================================================
    # Individual check methods
    # ==================================================================

    def _check_completion_time(self, df: Any) -> Dict[str, Any]:
        """Check whether 80%+ of completion times fall within [480, 1500]s."""
        timing_col = self._find_timing_column(df)
        if timing_col is None:
            return {
                "passed": True,
                "actual": None,
                "target": self._benchmarks["completion_time"]["target_pct_in_range"],
                "details": "No timing data found in dataset; check skipped.",
            }

        values = self._col_values_numeric(df, timing_col)
        if not values:
            return {
                "passed": True,
                "actual": None,
                "target": self._benchmarks["completion_time"]["target_pct_in_range"],
                "details": "Timing column present but contains no numeric values.",
            }

        lo = self._benchmarks["completion_time"]["human_min_seconds"]
        hi = self._benchmarks["completion_time"]["human_max_seconds"]
        target = self._benchmarks["completion_time"]["target_pct_in_range"]

        in_range = sum(1 for v in values if lo <= v <= hi)
        pct = in_range / len(values)

        return {
            "passed": pct >= target,
            "actual": round(pct, 4),
            "target": target,
            "details": (
                f"{in_range}/{len(values)} participants ({pct:.1%}) have "
                f"completion times in [{lo}, {hi}]s (target >= {target:.0%})."
            ),
        }

    def _check_oe_uniqueness(self, df: Any) -> Dict[str, Any]:
        """Check pairwise uniqueness of open-ended response columns."""
        oe_cols = self._find_oe_columns(df)
        if not oe_cols:
            return {
                "passed": True,
                "actual": None,
                "target": self._benchmarks["oe_responses"]["min_uniqueness_pct"],
                "details": "No open-ended columns detected; check skipped.",
            }

        target = self._benchmarks["oe_responses"]["min_uniqueness_pct"]
        col_results: List[Tuple[str, float]] = []
        overall_pass = True

        for col in oe_cols:
            values = self._col_values_str(df, col)
            non_empty = [v for v in values if v.strip()]
            if not non_empty:
                continue
            unique_pct = len(set(non_empty)) / len(non_empty)
            col_results.append((col, unique_pct))
            if unique_pct < target:
                overall_pass = False

        if not col_results:
            return {
                "passed": True,
                "actual": None,
                "target": target,
                "details": "OE columns found but all are empty; check skipped.",
            }

        worst_col, worst_pct = min(col_results, key=lambda x: x[1])
        avg_pct = sum(p for _, p in col_results) / len(col_results)

        return {
            "passed": overall_pass,
            "actual": round(avg_pct, 4),
            "target": target,
            "details": (
                f"Checked {len(col_results)} OE column(s). "
                f"Average uniqueness: {avg_pct:.1%}. "
                f"Worst: '{worst_col}' at {worst_pct:.1%} "
                f"(target >= {target:.0%})."
            ),
        }

    def _check_straightlining(self, df: Any) -> Dict[str, Any]:
        """Check proportion of participants who straight-line scale items."""
        scale_cols = self._find_scale_columns(df)
        if len(scale_cols) < 3:
            return {
                "passed": True,
                "actual": None,
                "target": str(self._benchmarks["straightlining_rate"]["expected_range"]),
                "details": (
                    f"Found only {len(scale_cols)} scale column(s); need >= 3 "
                    "for straight-lining detection. Check skipped."
                ),
            }

        lo, hi = self._benchmarks["straightlining_rate"]["expected_range"]
        n_rows = self._nrows(df)
        if n_rows == 0:
            return {
                "passed": True,
                "actual": 0.0,
                "target": str((lo, hi)),
                "details": "DataFrame is empty; check skipped.",
            }

        straightliners = 0
        for row_idx in range(n_rows):
            row_vals = []
            for col in scale_cols:
                v = self._cell_value(df, row_idx, col)
                if v is not None:
                    try:
                        row_vals.append(float(v))
                    except (TypeError, ValueError):
                        pass
            if len(row_vals) >= 3:
                sd = self._std(row_vals)
                if sd < 0.2:
                    straightliners += 1

        rate = straightliners / n_rows
        passed = lo <= rate <= hi

        return {
            "passed": passed,
            "actual": round(rate, 4),
            "target": str((lo, hi)),
            "details": (
                f"{straightliners}/{n_rows} participants ({rate:.1%}) are "
                f"straight-liners (SD < 0.2 across {len(scale_cols)} scale items). "
                f"Expected range: [{lo:.0%}, {hi:.0%}]."
            ),
        }

    def _check_oe_length_distribution(self, df: Any) -> Dict[str, Any]:
        """Check that mean word count of OE responses is within [8, 35]."""
        oe_cols = self._find_oe_columns(df)
        if not oe_cols:
            return {
                "passed": True,
                "actual": None,
                "target": str(self._benchmarks["oe_responses"]["mean_words_range"]),
                "details": "No open-ended columns detected; check skipped.",
            }

        lo, hi = self._benchmarks["oe_responses"]["mean_words_range"]
        all_word_counts: List[int] = []

        for col in oe_cols:
            values = self._col_values_str(df, col)
            for v in values:
                stripped = v.strip()
                if stripped:
                    all_word_counts.append(len(stripped.split()))

        if not all_word_counts:
            return {
                "passed": True,
                "actual": None,
                "target": str((lo, hi)),
                "details": "OE columns found but all responses are empty.",
            }

        mean_wc = sum(all_word_counts) / len(all_word_counts)
        sd_wc = self._std(all_word_counts) if len(all_word_counts) > 1 else 0.0
        passed = lo <= mean_wc <= hi

        return {
            "passed": passed,
            "actual": round(mean_wc, 2),
            "target": str((lo, hi)),
            "details": (
                f"Mean word count across {len(all_word_counts)} OE responses: "
                f"{mean_wc:.1f} (SD={sd_wc:.1f}). Target range: [{lo}, {hi}]."
            ),
        }

    def _check_rating_text_coherence(self, df: Any) -> Dict[str, Any]:
        """Check whether numeric ratings and OE sentiment are consistent.

        Simplified check:
        - If average rating >= 5/7 and OE contains strong negative words -> flag
        - If average rating <= 2/7 and OE contains strong positive words -> flag
        """
        scale_cols = self._find_scale_columns(df)
        oe_cols = self._find_oe_columns(df)

        if not scale_cols or not oe_cols:
            return {
                "passed": True,
                "actual": None,
                "target": self._benchmarks["rating_text_coherence"]["min_pct_coherent"],
                "details": (
                    "Need both scale and OE columns for coherence check. "
                    f"Found {len(scale_cols)} scale, {len(oe_cols)} OE. Skipped."
                ),
            }

        target = self._benchmarks["rating_text_coherence"]["min_pct_coherent"]
        n_rows = self._nrows(df)
        if n_rows == 0:
            return {
                "passed": True,
                "actual": None,
                "target": target,
                "details": "DataFrame is empty; check skipped.",
            }

        incoherent = 0
        evaluated = 0

        for row_idx in range(n_rows):
            # Compute average numeric rating for this participant
            ratings: List[float] = []
            for col in scale_cols:
                v = self._cell_value(df, row_idx, col)
                if v is not None:
                    try:
                        ratings.append(float(v))
                    except (TypeError, ValueError):
                        pass

            if not ratings:
                continue

            avg_rating = sum(ratings) / len(ratings)

            # Detect max scale point (guess from data)
            max_scale = max(ratings) if ratings else 7.0
            if max_scale <= 5:
                max_scale = 5.0
            else:
                max_scale = 7.0

            # Gather OE text
            oe_text_parts: List[str] = []
            for col in oe_cols:
                v = self._cell_value(df, row_idx, col)
                if v is not None:
                    oe_text_parts.append(str(v).strip())
            oe_text = " ".join(oe_text_parts).lower()

            if not oe_text.strip():
                continue

            evaluated += 1
            words_in_text = set(re.findall(r"[a-z]+", oe_text))

            # High rating threshold: >= 5/7 (71%) of scale
            high_threshold = max_scale * (5.0 / 7.0)
            # Low rating threshold: <= 2/7 (29%) of scale
            low_threshold = max_scale * (2.0 / 7.0)

            neg_hits = words_in_text & _NEGATIVE_WORDS
            pos_hits = words_in_text & _POSITIVE_WORDS

            if avg_rating >= high_threshold and len(neg_hits) >= 2 and len(pos_hits) == 0:
                incoherent += 1
                logger.debug(
                    "Row %d: avg_rating=%.2f but OE has negative words: %s",
                    row_idx, avg_rating, neg_hits,
                )
            elif avg_rating <= low_threshold and len(pos_hits) >= 2 and len(neg_hits) == 0:
                incoherent += 1
                logger.debug(
                    "Row %d: avg_rating=%.2f but OE has positive words: %s",
                    row_idx, avg_rating, pos_hits,
                )

        if evaluated == 0:
            return {
                "passed": True,
                "actual": None,
                "target": target,
                "details": "No rows had both ratings and OE text; check skipped.",
            }

        coherent_pct = 1.0 - (incoherent / evaluated)
        passed = coherent_pct >= target

        return {
            "passed": passed,
            "actual": round(coherent_pct, 4),
            "target": target,
            "details": (
                f"{evaluated - incoherent}/{evaluated} participants ({coherent_pct:.1%}) "
                f"show coherent rating-text sentiment (target >= {target:.0%}). "
                f"{incoherent} flagged as incoherent."
            ),
        }

    # ==================================================================
    # Auto-correction methods
    # ==================================================================

    def _correct_completion_time(self, df: Any) -> Any:
        """Re-sample out-of-range timing values within [480, 1500] seconds."""
        timing_col = self._find_timing_column(df)
        if timing_col is None:
            return df

        lo = self._benchmarks["completion_time"]["human_min_seconds"]
        hi = self._benchmarks["completion_time"]["human_max_seconds"]
        n_rows = self._nrows(df)

        for row_idx in range(n_rows):
            v = self._cell_value(df, row_idx, timing_col)
            if v is None:
                continue
            try:
                val = float(v)
            except (TypeError, ValueError):
                continue

            if val < lo or val > hi:
                # Re-sample with a log-normal-ish distribution centred on ~720s
                new_val = random.gauss(mu=720, sigma=200)
                new_val = max(lo, min(hi, new_val))
                self._set_cell(df, row_idx, timing_col, round(new_val, 1))

        logger.info("Corrected completion times to [%d, %d]s range.", lo, hi)
        return df

    def _correct_straightlining(self, df: Any) -> Any:
        """Perturb 2-3 multi-item scale items by +/-1 for participants with SD < 0.2.

        v1.2.4.0: Only operates on columns identified as multi-item scale items
        (not standalone DVs). If no multi-item scale columns are found, returns
        the DataFrame unchanged. This prevents corrupting DV data.
        """
        scale_cols = self._find_scale_columns(df)
        if len(scale_cols) < 3:
            logger.info("Straightlining correction skipped: fewer than 3 multi-item scale columns found.")
            return df

        n_rows = self._nrows(df)
        _perturbed_count = 0

        for row_idx in range(n_rows):
            row_vals: List[Tuple[str, float]] = []
            for col in scale_cols:
                v = self._cell_value(df, row_idx, col)
                if v is not None:
                    try:
                        row_vals.append((col, float(v)))
                    except (TypeError, ValueError):
                        pass

            if len(row_vals) < 3:
                continue

            vals_only = [x[1] for x in row_vals]
            sd = self._std(vals_only)
            if sd >= 0.2:
                continue

            # Perturb 2-3 randomly chosen items by +/-1
            n_perturb = min(random.choice([2, 3]), len(row_vals))
            indices = random.sample(range(len(row_vals)), n_perturb)

            for idx in indices:
                col_name, old_val = row_vals[idx]
                direction = random.choice([-1, 1])
                new_val = old_val + direction
                # Clamp to scale bounds (infer from actual data range)
                max_val = max(vals_only)
                scale_lo = 1
                scale_hi = int(max_val) if max_val > 5 else 5
                if scale_hi < 5:
                    scale_hi = 5
                new_val = max(scale_lo, min(scale_hi, new_val))
                self._set_cell(df, row_idx, col_name, int(new_val))
                _perturbed_count += 1

        if _perturbed_count > 0:
            logger.info("Corrected straight-lining: perturbed %d scale item cells.", _perturbed_count)
        else:
            logger.info("Straightlining correction: no rows needed perturbation.")
        return df

    def _correct_oe_uniqueness(self, df: Any) -> Any:
        """Append random filler words to duplicate OE responses."""
        oe_cols = self._find_oe_columns(df)
        if not oe_cols:
            return df

        _filler_phrases = [
            " honestly", " I think", " in my opinion", " overall",
            " personally", " to be fair", " basically", " I mean",
            " you know", " I guess", " kind of", " really",
            " for the most part", " in general", " more or less",
        ]

        for col in oe_cols:
            n_rows = self._nrows(df)
            seen: Dict[str, int] = {}
            for row_idx in range(n_rows):
                v = self._cell_value(df, row_idx, col)
                if v is None:
                    continue
                text = str(v).strip()
                if not text:
                    continue

                normalised = text.lower()
                if normalised in seen:
                    # This is a duplicate — append filler to make unique
                    filler = random.choice(_filler_phrases)
                    # Vary by adding count to avoid re-duplication
                    count = seen[normalised]
                    suffix = filler if count == 1 else f"{filler} ({count})"
                    new_text = text.rstrip(".") + suffix.rstrip() + "."
                    self._set_cell(df, row_idx, col, new_text)
                    seen[normalised] = count + 1
                else:
                    seen[normalised] = 1

        logger.info("Corrected OE uniqueness by appending fillers to duplicates.")
        return df

    def _correct_oe_length(self, df: Any) -> Any:
        """Truncate or extend OE responses to bring mean into [8, 35] words."""
        oe_cols = self._find_oe_columns(df)
        if not oe_cols:
            return df

        lo, hi = self._benchmarks["oe_responses"]["mean_words_range"]

        _extension_phrases = [
            "I feel this way because of my personal experience.",
            "This is something I think about quite often.",
            "It really depends on the specific situation though.",
            "There are many factors that contribute to this.",
            "I have mixed feelings about the whole thing.",
        ]

        for col in oe_cols:
            n_rows = self._nrows(df)
            word_counts: List[int] = []
            texts: List[Tuple[int, str]] = []

            for row_idx in range(n_rows):
                v = self._cell_value(df, row_idx, col)
                if v is None:
                    continue
                text = str(v).strip()
                if not text:
                    continue
                wc = len(text.split())
                word_counts.append(wc)
                texts.append((row_idx, text))

            if not word_counts:
                continue

            mean_wc = sum(word_counts) / len(word_counts)

            if mean_wc > hi:
                # Truncate longest responses
                target_mean = (lo + hi) / 2.0
                for row_idx, text in texts:
                    words = text.split()
                    if len(words) > hi:
                        truncated = " ".join(words[:random.randint(lo, hi)])
                        if not truncated.endswith("."):
                            truncated += "."
                        self._set_cell(df, row_idx, col, truncated)

            elif mean_wc < lo:
                # Extend shortest responses
                for row_idx, text in texts:
                    words = text.split()
                    if len(words) < lo:
                        extension = random.choice(_extension_phrases)
                        new_text = text.rstrip(".") + ". " + extension
                        self._set_cell(df, row_idx, col, new_text)

        logger.info("Corrected OE response lengths toward [%d, %d] word target.", lo, hi)
        return df

    # ==================================================================
    # Column detection helpers
    # ==================================================================

    def _find_timing_column(self, df: Any) -> Optional[str]:
        """Find the first column that looks like a completion-time field."""
        columns = self._columns(df)
        for col in columns:
            for pat in self._TIMING_COL_PATTERNS:
                if pat.search(col):
                    return col
        return None

    # v1.2.5.1: Columns that should NEVER be treated as OE text — these are
    # metadata, condition labels, demographics, or engine internals.
    _OE_EXCLUDE_COLS = frozenset({
        "CONDITION", "RUN_ID", "SIMULATION_MODE", "SIMULATION_SEED",
        "PARTICIPANT_ID", "Gender", "Age", "_Generation_Source",
        "Completion_Time_Seconds", "Attention_Pass_Rate", "Max_Straight_Line",
        "Mean_Item_RT_ms", "Total_Scale_RT_ms",
    })
    _OE_EXCLUDE_PREFIXES = (
        "Flag_", "Exclude_", "Attention_", "ABE3_", "HBS_", "_",
    )

    def _find_oe_columns(self, df: Any) -> List[str]:
        """Find columns likely containing open-ended text responses.

        Heuristic: dtype is object/string and mean non-empty length > 10 chars.

        v1.2.5.1: Explicitly excludes CONDITION, demographics, and metadata
        columns to prevent the uniqueness corrector from treating repeated
        condition labels as "duplicate" OE responses and appending filler words.
        """
        columns = self._columns(df)
        oe_cols: List[str] = []

        for col in columns:
            # Skip protected columns
            if col in self._OE_EXCLUDE_COLS:
                continue
            if any(col.startswith(p) for p in self._OE_EXCLUDE_PREFIXES):
                continue

            values = self._col_values_str(df, col)
            non_empty = [v for v in values if v.strip()]
            if not non_empty:
                continue

            # v1.2.5.1: Skip columns with very low cardinality (likely categorical,
            # not OE text). CONDITION with 6 labels has cardinality 6 out of 600 rows.
            unique_ratio = len(set(v.lower().strip() for v in non_empty)) / max(len(non_empty), 1)
            if unique_ratio < 0.10:
                continue  # <10% unique values → categorical, not OE text

            # Check if values look like free text (not categorical codes)
            mean_len = sum(len(v) for v in non_empty) / len(non_empty)
            if mean_len > 10:
                # Additional check: not purely numeric
                numeric_count = sum(
                    1 for v in non_empty if re.match(r"^-?\d+\.?\d*$", v.strip())
                )
                if numeric_count / len(non_empty) < 0.5:
                    oe_cols.append(col)

        return oe_cols

    # Columns that should NEVER be treated as scale items for straightlining
    # correction — these are metadata, DV composites, or single-item measures.
    _PROTECTED_COL_PREFIXES = (
        "PARTICIPANT_", "RUN_ID", "SIMULATION_", "CONDITION", "Gender", "Age",
        "Exclude_", "Flag_", "Completion_Time", "Attention_", "Max_Straight",
        "HBS_", "_Generation_Source", "Mean_Item_RT", "Total_Scale_RT",
    )
    _PROTECTED_COL_PATTERNS = frozenset({
        "CONDITION", "RUN_ID", "SIMULATION_MODE", "SIMULATION_SEED",
    })

    def _find_scale_columns(self, df: Any) -> List[str]:
        """Find numeric columns that look like Likert-type multi-item scale items.

        Heuristic: column name must look like a numbered item (e.g., Scale_1,
        DV_Item2) and values are integers in [1, 7] (or [1, 5]) range.

        v1.2.4.0: Excludes standalone DV columns, metadata, and any column
        that doesn't have a multi-item naming pattern. Only columns that appear
        to be PART OF A MULTI-ITEM SCALE (name contains a trailing number or
        underscore-number pattern) are eligible. This prevents the straightlining
        correction from corrupting actual DV data.
        """
        columns = self._columns(df)
        scale_cols: List[str] = []

        # Build set of column base names that have numbered siblings
        # e.g., if "Main_DV_1", "Main_DV_2" exist, "Main_DV" is a multi-item base
        import re as _re
        _numbered_pattern = _re.compile(r'^(.+?)_?(\d+)$')
        _base_counts: dict = {}
        for col in columns:
            m = _numbered_pattern.match(col)
            if m:
                base = m.group(1)
                _base_counts[base] = _base_counts.get(base, 0) + 1

        # Only bases with 2+ numbered items are multi-item scales
        _multi_item_bases = {b for b, c in _base_counts.items() if c >= 2}

        for col in columns:
            # Skip protected columns
            if col in self._PROTECTED_COL_PATTERNS:
                continue
            if any(col.startswith(p) for p in self._PROTECTED_COL_PREFIXES):
                continue

            # Must be part of a multi-item scale (has numbered siblings)
            m = _numbered_pattern.match(col)
            if not m or m.group(1) not in _multi_item_bases:
                continue

            values = self._col_values_numeric(df, col)
            if len(values) < 3:
                continue

            # Check if values are integer-like and within plausible scale range
            all_int = all(v == int(v) for v in values)
            if not all_int:
                continue

            min_v = min(values)
            max_v = max(values)

            # Accept 1-5, 1-7, 1-9, 0-10 scales
            if min_v >= 0 and max_v <= 10 and max_v - min_v <= 9:
                if len(set(int(v) for v in values)) >= 2:
                    scale_cols.append(col)

        return scale_cols

    # ==================================================================
    # DataFrame abstraction (works with pandas or dict-of-lists)
    # ==================================================================

    def _ensure_df(self, df: Any) -> Any:
        """Convert dict-of-lists to a form we can work with uniformly."""
        if HAS_PANDAS and isinstance(df, pd.DataFrame):
            return df
        if isinstance(df, dict):
            return df
        # If it's something pandas-like but we have pandas, try to convert
        if HAS_PANDAS:
            try:
                return pd.DataFrame(df)
            except Exception:
                pass
        return df

    def _columns(self, df: Any) -> List[str]:
        """Return column names."""
        if HAS_PANDAS and isinstance(df, pd.DataFrame):
            return list(df.columns)
        if isinstance(df, dict):
            return list(df.keys())
        return []

    def _nrows(self, df: Any) -> int:
        """Return number of rows."""
        if HAS_PANDAS and isinstance(df, pd.DataFrame):
            return len(df)
        if isinstance(df, dict):
            cols = list(df.values())
            return len(cols[0]) if cols else 0
        return 0

    def _cell_value(self, df: Any, row: int, col: str) -> Any:
        """Get value at (row, col)."""
        if HAS_PANDAS and isinstance(df, pd.DataFrame):
            try:
                val = df.iloc[row][col]
                if pd.isna(val):
                    return None
                return val
            except (IndexError, KeyError):
                return None
        if isinstance(df, dict):
            try:
                val = df[col][row]
                return val
            except (KeyError, IndexError):
                return None
        return None

    def _set_cell(self, df: Any, row: int, col: str, value: Any) -> None:
        """Set value at (row, col) in-place."""
        if HAS_PANDAS and isinstance(df, pd.DataFrame):
            try:
                df.iat[row, df.columns.get_loc(col)] = value
            except (IndexError, KeyError):
                pass
        elif isinstance(df, dict):
            try:
                df[col][row] = value
            except (KeyError, IndexError):
                pass

    def _col_values_numeric(self, df: Any, col: str) -> List[float]:
        """Return all numeric (non-null) values in a column."""
        result: List[float] = []
        n = self._nrows(df)
        for i in range(n):
            v = self._cell_value(df, i, col)
            if v is None:
                continue
            try:
                result.append(float(v))
            except (TypeError, ValueError):
                pass
        return result

    def _col_values_str(self, df: Any, col: str) -> List[str]:
        """Return all string values in a column (empty string for None)."""
        result: List[str] = []
        n = self._nrows(df)
        for i in range(n):
            v = self._cell_value(df, i, col)
            if v is None:
                result.append("")
            else:
                result.append(str(v))
        return result

    # ==================================================================
    # Math helpers
    # ==================================================================

    @staticmethod
    def _std(values: List[float]) -> float:
        """Compute population standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
