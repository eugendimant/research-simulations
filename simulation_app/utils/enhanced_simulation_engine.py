# simulation_app/utils/enhanced_simulation_engine.py
"""
Enhanced Simulation Engine for BDS5010 Behavioral Experiment Simulation Tool
=============================================================================
Advanced simulation engine with:
- Theory-grounded persona library integration
- Automatic domain detection
- Expected effect size handling
- Natural variation across runs (no identical outputs unless seed fixed)
- Text generation for open-ended responses
- Stimulus/image evaluation support
- Exclusion criteria simulation

Notes on reproducibility:
- Reproducibility is controlled by `seed`. If `seed` is None, the engine will
  generate a run-specific seed so repeated runs are different by default.
- Internal hashing uses stable (MD5-based) hashing rather than Python's built-in
  `hash()` (which is randomized per process).

This module is designed to run inside a `utils/` package (i.e., imported as
`utils.enhanced_simulation_engine`), so relative imports are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import hashlib
import json
import random

import numpy as np
import pandas as pd

from .persona_library import (
    PersonaLibrary,
    Persona,
    TextResponseGenerator,
    StimulusEvaluationHandler,
)


def _stable_int_hash(s: str) -> int:
    """Stable, cross-run integer hash for strings."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


@dataclass
class EffectSizeSpec:
    """Specification for an expected effect in the study."""
    variable: str
    factor: str
    level_high: str
    level_low: str
    cohens_d: float
    direction: str = "positive"  # "positive" or "negative"


@dataclass
class ExclusionCriteria:
    """Criteria for simulating participant exclusions."""
    attention_check_threshold: float = 0.0  # Min attention checks passed proportion
    completion_time_min_seconds: int = 60
    completion_time_max_seconds: int = 3600
    straight_line_threshold: int = 10  # Max consecutive identical responses across items
    duplicate_ip_check: bool = True
    exclude_careless_responders: bool = False  # If True, flags but doesn't exclude


class EnhancedSimulationEngine:
    """
    Advanced simulation engine for generating synthetic behavioral experiment data.
    """

    def __init__(
        self,
        # Study metadata
        study_title: str,
        study_description: str,
        sample_size: int,
        # Experimental design
        conditions: List[str],
        factors: List[Dict[str, Any]],
        # Measures
        scales: List[Dict[str, Any]],
        additional_vars: List[Dict[str, Any]],
        # Demographics
        demographics: Dict[str, Any],
        # Quality parameters
        attention_rate: float = 0.95,
        random_responder_rate: float = 0.05,
        # Effect sizes (optional)
        effect_sizes: Optional[List[EffectSizeSpec]] = None,
        # Exclusion criteria (optional)
        exclusion_criteria: Optional[ExclusionCriteria] = None,
        # Persona customization (optional)
        custom_persona_weights: Optional[Dict[str, float]] = None,
        # Open-ended response settings
        open_ended_questions: Optional[List[Dict[str, Any]]] = None,
        # Stimulus/image evaluation settings
        stimulus_evaluations: Optional[List[Dict[str, Any]]] = None,
        # Seed for reproducibility (optional)
        seed: Optional[int] = None,
        # Mode
        mode: str = "pilot",  # "pilot" or "final"
    ):
        self.study_title = str(study_title or "").strip()
        self.study_description = str(study_description or "").strip()
        self.sample_size = int(sample_size)
        self.conditions = [str(c).strip() for c in (conditions or []) if str(c).strip()]
        self.factors = factors or []
        self.scales = scales or []
        self.additional_vars = additional_vars or []
        self.demographics = demographics or {}
        self.attention_rate = float(attention_rate)
        self.random_responder_rate = float(random_responder_rate)
        self.effect_sizes = effect_sizes or []
        self.exclusion_criteria = exclusion_criteria or ExclusionCriteria()
        self.open_ended_questions = open_ended_questions or []
        self.stimulus_evaluations = stimulus_evaluations or []
        self.mode = (mode or "pilot").strip().lower()
        if self.mode not in ("pilot", "final"):
            self.mode = "pilot"

        if not self.conditions:
            self.conditions = ["Condition A"]

        if self.sample_size <= 0:
            raise ValueError("sample_size must be a positive integer")

        if seed is None:
            timestamp = int(datetime.now().timestamp() * 1_000_000)
            study_hash = int(
                hashlib.md5(f"{self.study_title}_{self.study_description}".encode("utf-8")).hexdigest()[:8],
                16,
            )
            self.seed = (timestamp + study_hash) % (2**31)
        else:
            self.seed = int(seed) % (2**31)

        self.run_id = f"{self.mode.upper()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.seed % 10000:04d}"

        np.random.seed(self.seed)
        random.seed(self.seed)

        self.persona_library = PersonaLibrary(seed=self.seed)

        self.detected_domains = self.persona_library.detect_domains(
            self.study_description, self.study_title
        )
        self.available_personas = self.persona_library.get_personas_for_domains(
            self.detected_domains
        )

        if custom_persona_weights:
            for name, weight in custom_persona_weights.items():
                if name in self.available_personas:
                    try:
                        self.available_personas[name].weight = float(weight)
                    except Exception:
                        pass

        total_weight = sum(p.weight for p in self.available_personas.values()) or 1.0
        for persona in self.available_personas.values():
            persona.weight = persona.weight / total_weight

        self.text_generator = TextResponseGenerator()
        self.stimulus_handler = StimulusEvaluationHandler()

        self.column_info: List[Tuple[str, str]] = []
        self.validation_log: List[str] = []

    def _assign_persona(self, participant_id: int) -> Tuple[str, Persona]:
        persona_names = list(self.available_personas.keys())
        weights = [self.available_personas[n].weight for n in persona_names]

        p_seed = (self.seed + participant_id * 7919) % (2**31)
        rng = np.random.RandomState(p_seed)

        name = str(rng.choice(persona_names, p=weights))
        return name, self.available_personas[name]

    def _generate_participant_traits(
        self, participant_id: int, persona: Persona
    ) -> Dict[str, float]:
        return self.persona_library.generate_participant_profile(
            persona, participant_id, self.seed
        )

    def _get_effect_for_condition(self, condition: str, variable: str) -> float:
        for effect in self.effect_sizes:
            if effect.variable == variable or str(variable).startswith(effect.variable):
                condition_lower = str(condition).lower()

                if str(effect.level_high).lower() in condition_lower:
                    d = effect.cohens_d if effect.direction == "positive" else -effect.cohens_d
                    return float(d) * 0.15
                if str(effect.level_low).lower() in condition_lower:
                    d = -effect.cohens_d if effect.direction == "positive" else effect.cohens_d
                    return float(d) * 0.15
        return 0.0

    def _generate_scale_response(
        self,
        scale_min: int,
        scale_max: int,
        traits: Dict[str, float],
        is_reverse: bool,
        condition: str,
        variable_name: str,
        participant_seed: int,
    ) -> int:
        rng = np.random.RandomState(participant_seed)

        scale_min = int(scale_min)
        scale_max = int(scale_max)
        if scale_max < scale_min:
            scale_min, scale_max = scale_max, scale_min
        scale_range = scale_max - scale_min

        base_tendency = float(traits.get("response_tendency", traits.get("scale_use_breadth", 0.5)))
        condition_effect = self._get_effect_for_condition(condition, variable_name)

        adjusted_tendency = float(np.clip(base_tendency + condition_effect, 0.05, 0.95))
        center = scale_min + (adjusted_tendency * scale_range)

        if is_reverse:
            center = scale_max - (center - scale_min)

        variance = float(traits.get("variance_tendency", traits.get("scale_use_breadth", 0.8)))
        sd = (scale_range / 4.0) * variance if scale_range > 0 else 0.5

        response = float(rng.normal(center, sd))

        extreme_tendency = float(traits.get("extreme_tendency", 0.2))
        if rng.random() < extreme_tendency * 0.5:
            if response > (scale_min + scale_max) / 2.0:
                response = scale_max - float(rng.uniform(0, 0.8))
            else:
                response = scale_min + float(rng.uniform(0, 0.8))

        acquiescence = float(traits.get("acquiescence", 0.5))
        if (not is_reverse) and acquiescence > 0.6 and scale_range > 0:
            response += (acquiescence - 0.5) * scale_range * 0.1

        response = max(scale_min, min(scale_max, round(response)))
        return int(response)

    def _generate_attention_check(
        self,
        condition: str,
        traits: Dict[str, float],
        check_type: str,
        participant_seed: int,
    ) -> Tuple[int, bool]:
        rng = np.random.RandomState(participant_seed)

        attention = float(traits.get("attention_level", 0.85))
        is_attentive = rng.random() < attention * self.attention_rate

        if check_type == "ai_manipulation":
            correct = 1 if ("ai" in str(condition).lower() and "no ai" not in str(condition).lower()) else 2
            if is_attentive:
                return int(correct), True
            return int(3 - correct), False

        if check_type == "product_type":
            cond = str(condition).lower()
            if "hedonic" in cond:
                correct = 7
            elif "utilitarian" in cond:
                correct = 1
            else:
                correct = 4

            if is_attentive:
                return int(round(correct + float(rng.normal(0, 0.8)))), True
            return int(rng.uniform(1, 7)), False

        if is_attentive:
            return 1, True
        return int(rng.randint(2, 5)), False

    def _generate_open_response(
        self,
        question_spec: Dict[str, Any],
        persona: Persona,
        traits: Dict[str, float],
        condition: str,
        participant_seed: int,
    ) -> str:
        response_type = str(question_spec.get("type", "task_summary"))

        if float(traits.get("attention_level", 0.8)) < 0.5:
            style = "careless"
        elif persona.name == "Satisficer":
            style = "satisficer"
        elif persona.name == "Extreme Responder":
            style = "extreme"
        elif persona.name == "Engaged Responder":
            style = "engaged"
        else:
            style = "default"

        rng = np.random.RandomState(participant_seed)

        context = {
            "topic": question_spec.get("topic", "the presented content"),
            "stimulus": question_spec.get("stimulus", "product recommendation"),
            "product": question_spec.get("product", "product"),
            "feature": question_spec.get("feature", "features"),
            "emotion": str(rng.choice(["pleased", "interested", "satisfied", "engaged"])),
        }

        cond = str(condition).lower()
        if "ai" in cond:
            context["stimulus"] = "AI-recommended " + str(context["stimulus"])
        if "hedonic" in cond:
            context["product"] = "hedonic " + str(context["product"])
        elif "utilitarian" in cond:
            context["product"] = "functional " + str(context["product"])

        return self.text_generator.generate_response(
            response_type, style, context, traits, participant_seed
        )

    def _generate_demographics(self, n: int) -> pd.DataFrame:
        rng = np.random.RandomState(self.seed + 1000)

        age_mean = float(self.demographics.get("age_mean", 35))
        age_sd = float(self.demographics.get("age_sd", 12))
        ages = rng.normal(age_mean, age_sd, int(n))
        ages = np.clip(ages, 18, 70).astype(int)

        male_pct = float(self.demographics.get("gender_quota", 50)) / 100.0
        male_pct = float(np.clip(male_pct, 0.0, 1.0))

        female_pct = (1.0 - male_pct) * 0.96
        nonbinary_pct = 0.025
        pnts_pct = 0.015

        total = male_pct + female_pct + nonbinary_pct + pnts_pct
        genders = rng.choice(
            [1, 2, 3, 4],
            size=int(n),
            p=[male_pct / total, female_pct / total, nonbinary_pct / total, pnts_pct / total],
        )

        return pd.DataFrame({"Age": ages, "Gender": genders})

    def _generate_condition_assignment(self, n: int) -> pd.Series:
        n_conditions = len(self.conditions)
        n_per = int(n) // n_conditions
        remainder = int(n) % n_conditions

        assignments: List[str] = []
        for i, cond in enumerate(self.conditions):
            count = n_per + (1 if i < remainder else 0)
            assignments.extend([cond] * count)

        rng = np.random.RandomState(self.seed + 2000)
        rng.shuffle(assignments)
        return pd.Series(assignments, name="CONDITION")

    def _simulate_exclusion_flags(
        self,
        attention_checks_passed: List[bool],
        traits: Dict[str, float],
        participant_item_responses: List[int],
        participant_seed: int,
    ) -> Dict[str, Any]:
        rng = np.random.RandomState(participant_seed)

        base_time = 300
        attention = float(traits.get("attention_level", 0.8))

        if attention < 0.5:
            completion_time = int(rng.uniform(45, 150))
        elif attention > 0.9:
            completion_time = int(rng.normal(base_time * 1.2, 60))
        else:
            completion_time = int(rng.normal(base_time, 90))

        completion_time = int(np.clip(completion_time, 30, 1800))

        total_checks = len(attention_checks_passed)
        passed_checks = int(sum(bool(x) for x in attention_checks_passed))
        pass_rate = (passed_checks / total_checks) if total_checks > 0 else 1.0

        max_straight_line = 0
        current_streak = 1
        vals = [int(v) for v in (participant_item_responses or [])]
        if len(vals) >= 2:
            for i in range(1, len(vals)):
                if vals[i] == vals[i - 1]:
                    current_streak += 1
                    max_straight_line = max(max_straight_line, current_streak)
                else:
                    current_streak = 1

        exclude_time = (
            completion_time < int(self.exclusion_criteria.completion_time_min_seconds)
            or completion_time > int(self.exclusion_criteria.completion_time_max_seconds)
        )
        exclude_attention = pass_rate < float(self.exclusion_criteria.attention_check_threshold)
        exclude_straightline = max_straight_line >= int(self.exclusion_criteria.straight_line_threshold)

        exclude_recommended = bool(exclude_time or exclude_attention or exclude_straightline)
        if bool(self.exclusion_criteria.exclude_careless_responders):
            exclude_recommended = False

        return {
            "completion_time_seconds": completion_time,
            "attention_check_pass_rate": round(float(pass_rate), 2),
            "max_straight_line": int(max_straight_line),
            "flag_completion_time": bool(exclude_time),
            "flag_attention": bool(exclude_attention),
            "flag_straight_line": bool(exclude_straightline),
            "exclude_recommended": bool(exclude_recommended),
        }

    def generate(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        n = self.sample_size
        data: Dict[str, Any] = {}

        data["PARTICIPANT_ID"] = list(range(1, n + 1))
        data["RUN_ID"] = [self.run_id] * n
        data["SIMULATION_MODE"] = [self.mode.upper()] * n
        data["SIMULATION_SEED"] = [self.seed] * n

        self.column_info.extend(
            [
                ("PARTICIPANT_ID", "Unique participant identifier (1-N)"),
                ("RUN_ID", "Simulation run identifier"),
                ("SIMULATION_MODE", "Simulation mode: PILOT or FINAL"),
                ("SIMULATION_SEED", "Random seed for reproducibility"),
            ]
        )

        conditions = self._generate_condition_assignment(n)
        data["CONDITION"] = conditions.tolist()
        self.column_info.append(("CONDITION", f'Experimental condition: {", ".join(self.conditions)}'))

        demographics_df = self._generate_demographics(n)
        data["Age"] = demographics_df["Age"].tolist()
        data["Gender"] = demographics_df["Gender"].tolist()
        self.column_info.extend(
            [
                ("Age", f"Participant age (18-70, mean ~ {self.demographics.get('age_mean', 35)})"),
                ("Gender", "Gender: 1=Male, 2=Female, 3=Non-binary, 4=Prefer not to say"),
            ]
        )

        assigned_personas: List[str] = []
        all_traits: List[Dict[str, float]] = []
        for i in range(n):
            persona_name, persona = self._assign_persona(i)
            traits = self._generate_participant_traits(i, persona)
            assigned_personas.append(persona_name)
            all_traits.append(traits)

        data["_PERSONA"] = assigned_personas

        participant_item_responses: List[List[int]] = [[] for _ in range(n)]

        attention_results: List[List[bool]] = []
        ai_check_values: List[int] = []
        for i in range(n):
            p_seed = (self.seed + i * 100) % (2**31)
            check_val, passed = self._generate_attention_check(
                conditions.iloc[i], all_traits[i], "ai_manipulation", p_seed
            )
            ai_check_values.append(int(check_val))
            attention_results.append([bool(passed)])

        data["AI_Mentioned_Check"] = ai_check_values
        self.column_info.append(("AI_Mentioned_Check", "Manipulation check: Was AI mentioned? 1=Yes, 2=No"))

        for scale in self.scales:
            scale_name_raw = str(scale.get("name", "Scale")).strip() or "Scale"
            scale_name = scale_name_raw.replace(" ", "_")
            scale_points = int(scale.get("scale_points", 7))
            num_items = int(scale.get("num_items", 5))
            reverse_items = set(int(x) for x in (scale.get("reverse_items", []) or []))

            for item_num in range(1, num_items + 1):
                col_name = f"{scale_name}_{item_num}"
                is_reverse = item_num in reverse_items

                item_values: List[int] = []
                col_hash = _stable_int_hash(col_name)
                for i in range(n):
                    p_seed = (self.seed + i * 100 + col_hash) % (2**31)
                    val = self._generate_scale_response(
                        1,
                        scale_points,
                        all_traits[i],
                        is_reverse,
                        conditions.iloc[i],
                        scale_name,
                        p_seed,
                    )
                    item_values.append(int(val))
                    participant_item_responses[i].append(int(val))

                data[col_name] = item_values

                reverse_note = " (reverse-coded)" if is_reverse else ""
                self.column_info.append(
                    (col_name, f'{scale_name_raw} item {item_num} (1-{scale_points}){reverse_note}')
                )

        for var in self.additional_vars:
            var_name_raw = str(var.get("name", "Variable")).strip() or "Variable"
            var_name = var_name_raw.replace(" ", "_")
            var_min = int(var.get("min", 0))
            var_max = int(var.get("max", 10))

            col_hash = _stable_int_hash(var_name)
            values: List[int] = []
            for i in range(n):
                p_seed = (self.seed + i * 100 + col_hash) % (2**31)
                val = self._generate_scale_response(
                    var_min, var_max, all_traits[i], False, conditions.iloc[i], var_name, p_seed
                )
                values.append(int(val))
                participant_item_responses[i].append(int(val))

            data[var_name] = values
            self.column_info.append((var_name, f"{var_name_raw} ({var_min}-{var_max})"))

        has_product_factor = any(
            ("hedonic" in str(f.get("levels", [])).lower()) or ("utilitarian" in str(f.get("levels", [])).lower())
            for f in (self.factors or [])
        )
        if has_product_factor:
            hedonic_values: List[int] = []
            for i in range(n):
                p_seed = (self.seed + i * 100 + 9999) % (2**31)
                val, passed = self._generate_attention_check(
                    conditions.iloc[i], all_traits[i], "product_type", p_seed
                )
                hedonic_values.append(int(np.clip(val, 1, 7)))
                attention_results[i].append(bool(passed))
                participant_item_responses[i].append(int(np.clip(val, 1, 7)))

            data["Hedonic_Utilitarian"] = hedonic_values
            self.column_info.append(("Hedonic_Utilitarian", "Product type perception: 1=Utilitarian, 7=Hedonic"))

        if not self.open_ended_questions:
            self.open_ended_questions = [
                {
                    "name": "Task_Summary",
                    "type": "task_summary",
                    "topic": "product recommendations",
                    "stimulus": "product recommendation",
                }
            ]

        for q in self.open_ended_questions:
            col_name = str(q.get("name", "Open_Response")).replace(" ", "_")
            col_hash = _stable_int_hash(col_name)
            responses: List[str] = []
            for i in range(n):
                p_seed = (self.seed + i * 100 + col_hash) % (2**31)
                persona_name = assigned_personas[i]
                persona = self.available_personas[persona_name]
                text = self._generate_open_response(q, persona, all_traits[i], conditions.iloc[i], p_seed)
                responses.append(str(text))

            data[col_name] = responses
            self.column_info.append((col_name, f"Open-ended response: {q.get('type', 'text')}"))

        exclusion_data: List[Dict[str, Any]] = []
        for i in range(n):
            p_seed = (self.seed + i * 100 + 88888) % (2**31)
            excl = self._simulate_exclusion_flags(
                attention_results[i], all_traits[i], participant_item_responses[i], p_seed
            )
            exclusion_data.append(excl)

        data["Completion_Time_Seconds"] = [e["completion_time_seconds"] for e in exclusion_data]
        data["Attention_Pass_Rate"] = [e["attention_check_pass_rate"] for e in exclusion_data]
        data["Max_Straight_Line"] = [e["max_straight_line"] for e in exclusion_data]
        data["Flag_Speed"] = [1 if e["flag_completion_time"] else 0 for e in exclusion_data]
        data["Flag_Attention"] = [1 if e["flag_attention"] else 0 for e in exclusion_data]
        data["Flag_StraightLine"] = [1 if e["flag_straight_line"] else 0 for e in exclusion_data]
        data["Exclude_Recommended"] = [1 if e["exclude_recommended"] else 0 for e in exclusion_data]

        self.column_info.extend(
            [
                ("Completion_Time_Seconds", "Survey completion time in seconds"),
                ("Attention_Pass_Rate", "Proportion of attention checks passed (0-1)"),
                ("Max_Straight_Line", "Maximum consecutive identical responses"),
                ("Flag_Speed", "Flagged for completion time: 1=Yes, 0=No"),
                ("Flag_Attention", "Flagged for attention checks: 1=Yes, 0=No"),
                ("Flag_StraightLine", "Flagged for straight-lining: 1=Yes, 0=No"),
                ("Exclude_Recommended", "Recommended for exclusion: 1=Yes, 0=No"),
            ]
        )

        if "_PERSONA" in data:
            del data["_PERSONA"]

        df = pd.DataFrame(data)

        metadata = {
            "run_id": self.run_id,
            "simulation_mode": self.mode,
            "seed": self.seed,
            "generation_timestamp": datetime.now().isoformat(),
            "study_title": self.study_title,
            "study_description": self.study_description,
            "detected_domains": self.detected_domains,
            "sample_size": self.sample_size,
            "conditions": self.conditions,
            "factors": self.factors,
            "scales": self.scales,
            "effect_sizes": [
                {"variable": e.variable, "factor": e.factor, "cohens_d": e.cohens_d, "direction": e.direction}
                for e in self.effect_sizes
            ],
            "personas_used": sorted(list(set(assigned_personas))),
            "persona_distribution": {
                p: assigned_personas.count(p) / len(assigned_personas) for p in set(assigned_personas)
            },
            "exclusion_summary": {
                "flagged_speed": int(sum(data["Flag_Speed"])),
                "flagged_attention": int(sum(data["Flag_Attention"])),
                "flagged_straightline": int(sum(data["Flag_StraightLine"])),
                "total_excluded": int(sum(data["Exclude_Recommended"])),
            },
        }
        return df, metadata

    def generate_explainer(self) -> str:
        lines = [
            "=" * 70,
            "COLUMN EXPLAINER - BDS5010 Simulated Behavioral Experiment Data",
            "=" * 70,
            "",
            f"Study: {self.study_title}",
            f"Run ID: {self.run_id}",
            f"Mode: {self.mode.upper()}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Sample Size: {self.sample_size}",
            f"Conditions: {len(self.conditions)}",
            f"Detected Domains: {', '.join(self.detected_domains[:5])}",
            "",
            "-" * 70,
            "VARIABLE DESCRIPTIONS",
            "-" * 70,
            "",
        ]

        for col_name, description in self.column_info:
            lines.append(f"{col_name}")
            lines.append(f"    {description}")
            lines.append("")

        lines.extend(["-" * 70, "EXPERIMENTAL CONDITIONS", "-" * 70, ""])
        n_per = self.sample_size // len(self.conditions)
        for cond in self.conditions:
            lines.append(f"  - {cond} (target n = {n_per})")

        if self.effect_sizes:
            lines.extend(["", "-" * 70, "EXPECTED EFFECT SIZES", "-" * 70, ""])
            for effect in self.effect_sizes:
                lines.append(
                    f"  {effect.variable}: {effect.level_high} > {effect.level_low}, Cohen's d = {effect.cohens_d}"
                )

        lines.extend(
            [
                "",
                "-" * 70,
                "EXCLUSION CRITERIA",
                "-" * 70,
                "",
                f"  Min completion time: {self.exclusion_criteria.completion_time_min_seconds}s",
                f"  Max completion time: {self.exclusion_criteria.completion_time_max_seconds}s",
                f"  Straight-line threshold: {self.exclusion_criteria.straight_line_threshold} items",
                "",
                "=" * 70,
                "END OF COLUMN EXPLAINER",
                "=" * 70,
            ]
        )
        return "\n".join(lines)

    def generate_r_export(self, df: pd.DataFrame) -> str:
        """
        Generate R-compatible export with proper factor coding.

        Returns an R script that loads and prepares Simulated.csv.
        """
        def _r_quote(x: str) -> str:
            x = str(x).replace("\\", "\\\\").replace('"', '\\"')
            return f'"{x}"'

        condition_levels = ", ".join([_r_quote(c) for c in self.conditions])

        lines: List[str] = [
            "# ============================================================",
            f"# R Data Preparation Script - {self.study_title}",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Run ID: {self.run_id}",
            "# ============================================================",
            "",
            "# Load packages",
            "suppressPackageStartupMessages({",
            "  library(readr)",
            "  library(dplyr)",
            "})",
            "",
            "# Load the data",
            'data <- read_csv("Simulated.csv", show_col_types = FALSE)',
            "",
            "# Convert CONDITION to factor with proper levels",
            f"data$CONDITION <- factor(data$CONDITION, levels = c({condition_levels}))",
            "",
            "# Convert Gender to factor",
            'data$Gender <- factor(data$Gender, levels = 1:4,',
            '                     labels = c("Male", "Female", "Non-binary", "Prefer not to say"))',
            "",
        ]

        for scale in self.scales:
            scale_name_raw = str(scale.get("name", "Scale")).strip() or "Scale"
            scale_name = scale_name_raw.replace(" ", "_")
            num_items = int(scale.get("num_items", 5))
            scale_points = int(scale.get("scale_points", 7))
            reverse_items = set(int(x) for x in (scale.get("reverse_items", []) or []))

            items = [f"{scale_name}_{i}" for i in range(1, num_items + 1)]

            if reverse_items:
                lines.append(f"# {scale_name_raw} - reverse code items {sorted(reverse_items)}")
                for r_item in sorted(reverse_items):
                    item_name = f"{scale_name}_{r_item}"
                    max_val = scale_points
                    lines.append(f"data${item_name}_R <- {max_val + 1} - data${item_name}")
                lines.append("")

            lines.append(f"# Create {scale_name_raw} composite")
            item_list = ", ".join([f"data${item}" for item in items])
            lines.append(f"data${scale_name}_composite <- rowMeans(cbind({item_list}), na.rm = TRUE)")
            lines.append("")

        lines.extend(
            [
                "# Filter excluded participants (optional)",
                "data_clean <- data[data$Exclude_Recommended == 0, ]",
                "",
                'cat("Total N:", nrow(data), "\\n")',
                'cat("Clean N:", nrow(data_clean), "\\n")',
                "",
                "# Ready for analysis",
            ]
        )

        return "\n".join(lines)


__all__ = ["EnhancedSimulationEngine", "EffectSizeSpec", "ExclusionCriteria"]
