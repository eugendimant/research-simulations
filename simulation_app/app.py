# simulation_app/utils/enhanced_simulation_engine.py

"""
Enhanced Simulation Engine for BDS5010 Behavioral Experiment Simulation Tool
=============================================================================
Advanced simulation engine with:
- Theory-grounded persona library integration
- Automatic domain detection
- Expected effect size handling
- Natural variation guarantees (no identical outputs)
- Text generation for open-ended responses
- Image/stimulus evaluation support
- Exclusion criteria simulation
"""

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .persona_library import (
    Persona,
    PersonaLibrary,
    StimulusEvaluationHandler,
    TextResponseGenerator,
)


@dataclass
class EffectSizeSpec:
    """Specification for an expected effect in the study."""
    variable: str
    factor: str
    level_high: str
    level_low: str
    cohens_d: float
    direction: str = "positive"  # 'positive' or 'negative'


@dataclass
class ExclusionCriteria:
    """Criteria for simulating participant exclusions."""
    attention_check_threshold: float = 0.0
    completion_time_min_seconds: int = 60
    completion_time_max_seconds: int = 3600
    straight_line_threshold: int = 10
    duplicate_ip_check: bool = True
    exclude_careless_responders: bool = False


class EnhancedSimulationEngine:
    """
    Advanced simulation engine for generating synthetic behavioral experiment data.
    """

    def __init__(
        self,
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
        custom_persona_weights: Optional[Dict[str, float]] = None,
        open_ended_questions: Optional[List[Dict[str, Any]]] = None,
        stimulus_evaluations: Optional[List[Dict[str, Any]]] = None,
        seed: Optional[int] = None,
        mode: str = "pilot",
    ):
        self.study_title = study_title
        self.study_description = study_description
        self.sample_size = int(sample_size)
        self.conditions = list(conditions)
        self.factors = list(factors)
        self.scales = list(scales)
        self.additional_vars = list(additional_vars)
        self.demographics = dict(demographics)
        self.attention_rate = float(attention_rate)
        self.random_responder_rate = float(random_responder_rate)
        self.effect_sizes = effect_sizes or []
        self.exclusion_criteria = exclusion_criteria or ExclusionCriteria()
        self.open_ended_questions = open_ended_questions or []
        self.stimulus_evaluations = stimulus_evaluations or []
        self.mode = mode

        if seed is None:
            timestamp = int(datetime.now().timestamp() * 1_000_000)
            study_hash = int(hashlib.md5(f"{study_title}_{study_description}".encode("utf-8")).hexdigest()[:8], 16)
            self.seed = (timestamp + study_hash) % (2**31)
        else:
            self.seed = int(seed)

        self.run_id = f"{mode.upper()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.seed % 10000:04d}"

        np.random.seed(self.seed)
        random.seed(self.seed)

        self.persona_library = PersonaLibrary(seed=self.seed)

        self.detected_domains = self.persona_library.detect_domains(study_description, study_title)
        self.available_personas = self.persona_library.get_personas_for_domains(self.detected_domains)

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

    def _generate_participant_traits(self, participant_id: int, persona: Persona) -> Dict[str, float]:
        return self.persona_library.generate_participant_profile(persona, participant_id, self.seed)

    def _get_effect_for_condition(self, condition: str, variable: str) -> float:
        for effect in self.effect_sizes:
            if effect.variable == variable or variable.startswith(effect.variable):
                condition_lower = condition.lower()
                if effect.level_high.lower() in condition_lower:
                    d = effect.cohens_d if effect.direction == "positive" else -effect.cohens_d
                    return float(d) * 0.15
                if effect.level_low.lower() in condition_lower:
                    d = -effect.cohens_d if effect.direction == "positive" else effect.cohens_d
                    return float(d) * 0.15
        return 0.0

    def _generate_scale_response(
        self,
        scale_min: float,
        scale_max: float,
        traits: Dict[str, float],
        is_reverse: bool,
        condition: str,
        variable_name: str,
        participant_seed: int,
    ) -> int:
        rng = np.random.RandomState(participant_seed)
        scale_range = float(scale_max) - float(scale_min)

        base_tendency = traits.get("response_tendency", traits.get("scale_use_breadth", 0.5))
        condition_effect = self._get_effect_for_condition(condition, variable_name)

        adjusted = np.clip(base_tendency + condition_effect, 0.05, 0.95)
        center = float(scale_min) + (adjusted * scale_range)

        if is_reverse:
            center = float(scale_max) - (center - float(scale_min))

        variance = traits.get("variance_tendency", traits.get("scale_use_breadth", 0.8))
        sd = (scale_range / 4.0) * float(variance)

        response = float(rng.normal(center, sd))

        extreme_tendency = float(traits.get("extreme_tendency", 0.2))
        if rng.random() < extreme_tendency * 0.5:
            midpoint = (float(scale_min) + float(scale_max)) / 2.0
            if response > midpoint:
                response = float(scale_max) - float(rng.uniform(0, 0.8))
            else:
                response = float(scale_min) + float(rng.uniform(0, 0.8))

        acquiescence = float(traits.get("acquiescence", 0.5))
        if (not is_reverse) and acquiescence > 0.6:
            response += (acquiescence - 0.5) * scale_range * 0.1

        response = max(float(scale_min), min(float(scale_max), round(response)))
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
        is_attentive = rng.random() < (attention * self.attention_rate)

        if check_type == "ai_manipulation":
            correct = 1 if ("ai" in condition.lower() and "no ai" not in condition.lower()) else 2
            if is_attentive:
                return int(correct), True
            return int(3 - correct), False

        if check_type == "product_type":
            if "hedonic" in condition.lower():
                correct = 7
            elif "utilitarian" in condition.lower():
                correct = 1
            else:
                correct = 4

            if is_attentive:
                return int(correct + rng.normal(0, 0.8)), True
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
        response_type = question_spec.get("type", "task_summary")

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

        context = {
            "topic": question_spec.get("topic", "the presented content"),
            "stimulus": question_spec.get("stimulus", "product recommendation"),
            "product": question_spec.get("product", "product"),
            "feature": question_spec.get("feature", "features"),
            "emotion": str(np.random.choice(["pleased", "interested", "satisfied", "engaged"])),
        }

        if "ai" in condition.lower():
            context["stimulus"] = "AI-recommended " + str(context["stimulus"])
        if "hedonic" in condition.lower():
            context["product"] = "hedonic " + str(context["product"])
        elif "utilitarian" in condition.lower():
            context["product"] = "functional " + str(context["product"])

        return self.text_generator.generate_response(response_type, style, context, traits, participant_seed)

    def _generate_demographics(self, n: int) -> pd.DataFrame:
        rng = np.random.RandomState(self.seed + 1000)

        age_mean = float(self.demographics.get("age_mean", 35))
        age_sd = float(self.demographics.get("age_sd", 12))
        ages = rng.normal(age_mean, age_sd, int(n))
        ages = np.clip(ages, 18, 70).astype(int)

        male_pct = float(self.demographics.get("gender_quota", 50)) / 100.0
        female_pct = (1.0 - male_pct) * 0.96
        nonbinary_pct = 0.025
        pnts_pct = 0.015

        total = male_pct + female_pct + nonbinary_pct + pnts_pct
        genders = rng.choice([1, 2, 3, 4], size=int(n), p=[male_pct / total, female_pct / total, nonbinary_pct / total, pnts_pct / total])

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
        responses: Dict[str, List[int]],
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

        completion_time = int(max(30, min(1800, completion_time)))

        total_checks = len(attention_checks_passed)
        passed_checks = int(sum(attention_checks_passed))
        pass_rate = (passed_checks / total_checks) if total_checks > 0 else 1.0

        max_straight_line = 0
        for col_values in responses.values():
            if col_values:
                current_streak = 1
                for i in range(1, len(col_values)):
                    if col_values[i] == col_values[i - 1]:
                        current_streak += 1
                        max_straight_line = max(max_straight_line, current_streak)
                    else:
                        current_streak = 1

        exclude_time = (completion_time < int(self.exclusion_criteria.completion_time_min_seconds)) or (
            completion_time > int(self.exclusion_criteria.completion_time_max_seconds)
        )
        exclude_attention = float(pass_rate) < float(self.exclusion_criteria.attention_check_threshold)
        exclude_straightline = int(max_straight_line) >= int(self.exclusion_criteria.straight_line_threshold)

        return {
            "completion_time_seconds": completion_time,
            "attention_check_pass_rate": round(float(pass_rate), 2),
            "max_straight_line": int(max_straight_line),
            "flag_completion_time": bool(exclude_time),
            "flag_attention": bool(exclude_attention),
            "flag_straight_line": bool(exclude_straightline),
            "exclude_recommended": bool(exclude_time or exclude_attention or exclude_straightline),
        }

    def generate(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        n = int(self.sample_size)
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
        self.column_info.append(("CONDITION", f"Experimental condition: {', '.join(self.conditions)}"))

        demographics_df = self._generate_demographics(n)
        data["Age"] = demographics_df["Age"].tolist()
        data["Gender"] = demographics_df["Gender"].tolist()
        self.column_info.extend(
            [
                ("Age", f"Participant age (18-70, M~{self.demographics.get('age_mean', 35)})"),
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

        attention_results: List[List[bool]] = []
        ai_check_values: List[int] = []
        for i in range(n):
            p_seed = (self.seed + i * 100) % (2**31)
            check_val, passed = self._generate_attention_check(conditions.iloc[i], all_traits[i], "ai_manipulation", p_seed)
            ai_check_values.append(int(check_val))
            attention_results.append([bool(passed)])

        data["AI_Mentioned_Check"] = ai_check_values
        self.column_info.append(("AI_Mentioned_Check", "Manipulation check: Was AI mentioned? 1=Yes, 2=No"))

        scale_responses: Dict[str, List[int]] = {}

        for scale in self.scales:
            scale_name = str(scale.get("name", "Scale")).replace(" ", "_")
            scale_points = int(scale.get("scale_points", 6))
            num_items = int(scale.get("num_items", 5))
            reverse_items = set(int(x) for x in (scale.get("reverse_items", []) or []) if isinstance(x, int) or str(x).isdigit())

            scale_responses[scale_name] = []

            for item_num in range(1, num_items + 1):
                col_name = f"{scale_name}_{item_num}"
                is_reverse = item_num in reverse_items

                item_values: List[int] = []
                for i in range(n):
                    # stable-ish seed without relying on Python's randomized hash
                    col_hash = int(hashlib.md5(col_name.encode("utf-8")).hexdigest()[:8], 16)
                    p_seed = (self.seed + i * 100 + col_hash) % (2**31)
                    val = self._generate_scale_response(1, scale_points, all_traits[i], is_reverse, conditions.iloc[i], scale_name, p_seed)
                    item_values.append(int(val))
                    scale_responses[scale_name].append(int(val))

                data[col_name] = item_values

                reverse_note = " (reverse-coded)" if is_reverse else ""
                self.column_info.append((col_name, f"{scale.get('name','Scale')} item {item_num} (1-{scale_points}){reverse_note}"))

        for var in self.additional_vars:
            var_name = str(var.get("name", "Var")).replace(" ", "_")
            vmin = float(var.get("min", 0))
            vmax = float(var.get("max", 10))

            values: List[int] = []
            for i in range(n):
                col_hash = int(hashlib.md5(var_name.encode("utf-8")).hexdigest()[:8], 16)
                p_seed = (self.seed + i * 100 + col_hash) % (2**31)
                val = self._generate_scale_response(vmin, vmax, all_traits[i], False, conditions.iloc[i], var_name, p_seed)
                values.append(int(val))

            data[var_name] = values
            self.column_info.append((var_name, f"{var.get('name','Var')} ({vmin}-{vmax})"))

        has_product_factor = any(
            ("hedonic" in str(f.get("levels", [])).lower()) or ("utilitarian" in str(f.get("levels", [])).lower())
            for f in self.factors
        )

        if has_product_factor:
            hedonic_values: List[int] = []
            for i in range(n):
                p_seed = (self.seed + i * 100 + 9999) % (2**31)
                val, passed = self._generate_attention_check(conditions.iloc[i], all_traits[i], "product_type", p_seed)
                hedonic_values.append(int(np.clip(val, 1, 7)))
                attention_results[i].append(bool(passed))

            data["Hedonic_Utilitarian"] = hedonic_values
            self.column_info.append(("Hedonic_Utilitarian", "Product type perception: 1=Utilitarian, 7=Hedonic"))

        if not self.open_ended_questions:
            self.open_ended_questions = [
                {"name": "Task_Summary", "type": "task_summary", "topic": "product recommendations", "stimulus": "product recommendation"}
            ]

        for q in self.open_ended_questions:
            col_name = str(q.get("name", "Open")).replace(" ", "_")
            responses_txt: List[str] = []
            for i in range(n):
                col_hash = int(hashlib.md5(col_name.encode("utf-8")).hexdigest()[:8], 16)
                p_seed = (self.seed + i * 100 + col_hash) % (2**31)
                persona_name = assigned_personas[i]
                persona = self.available_personas[persona_name]
                text = self._generate_open_response(q, persona, all_traits[i], conditions.iloc[i], p_seed)
                responses_txt.append(str(text))

            data[col_name] = responses_txt
            self.column_info.append((col_name, f"Open-ended response: {q.get('type','text')}"))

        exclusion_data: List[Dict[str, Any]] = []
        for i in range(n):
            p_seed = (self.seed + i * 100 + 88888) % (2**31)
            excl = self._simulate_exclusion_flags(attention_results[i], all_traits[i], scale_responses, p_seed)
            exclusion_data.append(excl)

        data["Completion_Time_Seconds"] = [int(e["completion_time_seconds"]) for e in exclusion_data]
        data["Attention_Pass_Rate"] = [float(e["attention_check_pass_rate"]) for e in exclusion_data]
        data["Max_Straight_Line"] = [int(e["max_straight_line"]) for e in exclusion_data]
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

        metadata: Dict[str, Any] = {
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
                {"variable": e.variable, "factor": e.factor, "cohens_d": e.cohens_d, "direction": e.direction} for e in self.effect_sizes
            ]
            if self.effect_sizes
            else [],
            "personas_used": list(set(assigned_personas)),
            "persona_distribution": {p: assigned_personas.count(p) / len(assigned_personas) for p in set(assigned_personas)},
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

        n_per = max(1, self.sample_size // max(1, len(self.conditions)))
        for cond in self.conditions:
            lines.append(f"  - {cond} (target n = {n_per})")

        if self.effect_sizes:
            lines.extend(["", "-" * 70, "EXPECTED EFFECT SIZES", "-" * 70, ""])
            for effect in self.effect_sizes:
                lines.append(f"  {effect.variable}: {effect.level_high} > {effect.level_low}, Cohen's d = {effect.cohens_d}")

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

        This is rewritten to avoid invalid nested f-strings that were causing SyntaxError on deploy.
        """
        # Use JSON to create valid quoted strings (double-quoted), which also escapes safely.
        r_levels = ", ".join(json.dumps(str(c)) for c in self.conditions)

        lines = [
            "# ============================================================",
            f"# R Data Preparation Script - {self.study_title}",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Run ID: {self.run_id}",
            "# ============================================================",
            "",
            "# Load the data",
            'data <- read.csv("Simulated.csv", stringsAsFactors = FALSE)',
            "",
            "# Convert CONDITION to factor with proper levels",
            f"data$CONDITION <- factor(data$CONDITION, levels = c({r_levels}))",
            "",
            "# Convert Gender to factor",
            'data$Gender <- factor(data$Gender, levels = 1:4,',
            '                       labels = c("Male", "Female", "Non-binary", "Prefer not to say"))',
            "",
        ]

        for scale in self.scales:
            scale_name = str(scale.get("name", "Scale")).replace(" ", "_")
            num_items = int(scale.get("num_items", 5))
            reverse_items = list(scale.get("reverse_items", []) or [])
            scale_points = int(scale.get("scale_points", 6))

            items = [f"{scale_name}_{i}" for i in range(1, num_items + 1)]

            if reverse_items:
                lines.append(f"# {scale.get('name','Scale')} - reverse code items {reverse_items}")
                for r_item in reverse_items:
                    try:
                        r_item_int = int(r_item)
                    except Exception:
                        continue
                    item_name = f"{scale_name}_{r_item_int}"
                    lines.append(f"data${item_name}_R <- {scale_points + 1} - data${item_name}")

            lines.append(f"# Create {scale.get('name','Scale')} composite")
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
                "# Ready for analysis!",
            ]
        )

        return "\n".join(lines)


__all__ = ["EnhancedSimulationEngine", "EffectSizeSpec", "ExclusionCriteria"]
