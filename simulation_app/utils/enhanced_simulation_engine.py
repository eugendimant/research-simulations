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

Based on methodology from:
- Manning & Horton (2025) - LLM simulation with personas
- Cohen (1988) - Effect size conventions
- Krosnick (1991) - Survey satisficing theory
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import random
import string
import hashlib
import json
from datetime import datetime
from dataclasses import dataclass

from .persona_library import (
    PersonaLibrary,
    Persona,
    TextResponseGenerator,
    StimulusEvaluationHandler
)


@dataclass
class EffectSizeSpec:
    """Specification for an expected effect in the study."""
    variable: str
    factor: str
    level_high: str
    level_low: str
    cohens_d: float
    direction: str = 'positive'  # 'positive' or 'negative'


@dataclass
class ExclusionCriteria:
    """Criteria for simulating participant exclusions."""
    attention_check_threshold: float = 0.0  # Min attention checks passed
    completion_time_min_seconds: int = 60  # Minimum completion time
    completion_time_max_seconds: int = 3600  # Maximum completion time
    straight_line_threshold: int = 10  # Max consecutive identical responses
    duplicate_ip_check: bool = True
    exclude_careless_responders: bool = False  # If True, flags but doesn't include


class EnhancedSimulationEngine:
    """
    Advanced simulation engine for generating synthetic behavioral experiment data.

    Key Features:
    1. Domain-aware persona selection based on study description
    2. Expected effect sizes to generate hypothesis-informed (but not deterministic) data
    3. Guaranteed unique outputs - same study never produces identical data
    4. Rich text generation for open-ended responses
    5. Realistic exclusion simulation
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
        effect_sizes: List[EffectSizeSpec] = None,
        # Exclusion criteria (optional)
        exclusion_criteria: ExclusionCriteria = None,
        # Persona customization (optional)
        custom_persona_weights: Dict[str, float] = None,
        # Open-ended response settings
        open_ended_questions: List[Dict[str, Any]] = None,
        # Stimulus/image evaluation settings
        stimulus_evaluations: List[Dict[str, Any]] = None,
        # Seed for reproducibility (optional - if None, uses timestamp)
        seed: int = None,
        # Mode
        mode: str = 'pilot'  # 'pilot' or 'final'
    ):
        """
        Initialize the enhanced simulation engine.

        Args:
            study_title: Title of the research study
            study_description: Description for automatic domain detection
            sample_size: Total N to generate
            conditions: List of condition names
            factors: List of factor dicts with 'name' and 'levels'
            scales: List of scale dicts with 'name', 'num_items', 'scale_points', etc.
            additional_vars: List of single-item variable dicts
            demographics: Dict with gender_quota, age_mean, age_sd
            attention_rate: Base probability of passing attention checks
            random_responder_rate: Proportion of random responders
            effect_sizes: List of EffectSizeSpec for expected effects
            exclusion_criteria: ExclusionCriteria for realistic exclusions
            custom_persona_weights: Override default persona weights
            open_ended_questions: List of open-ended question specs
            stimulus_evaluations: List of stimulus evaluation specs
            seed: Random seed (None = timestamp-based)
            mode: 'pilot' or 'final'
        """
        self.study_title = study_title
        self.study_description = study_description
        self.sample_size = sample_size
        self.conditions = conditions
        self.factors = factors
        self.scales = scales
        self.additional_vars = additional_vars
        self.demographics = demographics
        self.attention_rate = attention_rate
        self.random_responder_rate = random_responder_rate
        self.effect_sizes = effect_sizes or []
        self.exclusion_criteria = exclusion_criteria or ExclusionCriteria()
        self.open_ended_questions = open_ended_questions or []
        self.stimulus_evaluations = stimulus_evaluations or []
        self.mode = mode

        # Generate unique seed ensuring no identical outputs
        if seed is None:
            # Combine timestamp with study hash for uniqueness
            timestamp = int(datetime.now().timestamp() * 1000000)
            study_hash = int(hashlib.md5(
                f"{study_title}_{study_description}".encode()
            ).hexdigest()[:8], 16)
            self.seed = (timestamp + study_hash) % (2**31)
        else:
            self.seed = seed

        # Add run-specific entropy
        self.run_id = f"{mode.upper()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.seed % 10000:04d}"

        np.random.seed(self.seed)
        random.seed(self.seed)

        # Initialize persona library with seed
        self.persona_library = PersonaLibrary(seed=self.seed)

        # Detect domains and select personas
        self.detected_domains = self.persona_library.detect_domains(
            study_description, study_title
        )
        self.available_personas = self.persona_library.get_personas_for_domains(
            self.detected_domains
        )

        # Apply custom weights if provided
        if custom_persona_weights:
            for name, weight in custom_persona_weights.items():
                if name in self.available_personas:
                    self.available_personas[name].weight = weight

        # Normalize persona weights
        total_weight = sum(p.weight for p in self.available_personas.values())
        for persona in self.available_personas.values():
            persona.weight = persona.weight / total_weight

        # Initialize helpers
        self.text_generator = TextResponseGenerator()
        self.stimulus_handler = StimulusEvaluationHandler()

        # Track generated columns
        self.column_info = []

        # Validation log
        self.validation_log = []

    def _assign_persona(self, participant_id: int) -> Tuple[str, Persona]:
        """Assign a persona to participant with guaranteed variation."""
        persona_names = list(self.available_personas.keys())
        weights = [self.available_personas[n].weight for n in persona_names]

        # Use participant-specific randomization
        p_seed = (self.seed + participant_id * 7919) % (2**31)  # 7919 is prime
        rng = np.random.RandomState(p_seed)

        name = rng.choice(persona_names, p=weights)
        return name, self.available_personas[name]

    def _generate_participant_traits(
        self,
        participant_id: int,
        persona: Persona
    ) -> Dict[str, float]:
        """Generate unique trait profile for participant."""
        return self.persona_library.generate_participant_profile(
            persona, participant_id, self.seed
        )

    def _get_effect_for_condition(
        self,
        condition: str,
        variable: str
    ) -> float:
        """
        Get the effect modifier for a specific condition and variable.

        Returns a value to shift the distribution mean based on expected effect size.
        """
        for effect in self.effect_sizes:
            if effect.variable == variable or variable.startswith(effect.variable):
                # Check if this condition matches the high or low level
                condition_lower = condition.lower()

                if effect.level_high.lower() in condition_lower:
                    # This condition should show higher values
                    d = effect.cohens_d if effect.direction == 'positive' else -effect.cohens_d
                    return d * 0.15  # Convert d to proportion of scale
                elif effect.level_low.lower() in condition_lower:
                    # This condition should show lower values
                    d = -effect.cohens_d if effect.direction == 'positive' else effect.cohens_d
                    return d * 0.15

        return 0.0  # No effect specified

    def _generate_scale_response(
        self,
        scale_min: int,
        scale_max: int,
        traits: Dict[str, float],
        is_reverse: bool,
        condition: str,
        variable_name: str,
        participant_seed: int
    ) -> int:
        """
        Generate a single scale response with effect size consideration.

        Args:
            scale_min: Minimum scale value
            scale_max: Maximum scale value
            traits: Participant trait dict
            is_reverse: Whether item is reverse-coded
            condition: Participant's experimental condition
            variable_name: Name of the variable (for effect lookup)
            participant_seed: Unique seed for this participant

        Returns:
            Integer scale response
        """
        rng = np.random.RandomState(participant_seed)

        scale_range = scale_max - scale_min

        # Base tendency from traits
        base_tendency = traits.get('response_tendency',
                                   traits.get('scale_use_breadth', 0.5))

        # Get condition effect
        condition_effect = self._get_effect_for_condition(condition, variable_name)

        # Adjust tendency
        adjusted_tendency = base_tendency + condition_effect
        adjusted_tendency = np.clip(adjusted_tendency, 0.05, 0.95)

        # Calculate center point
        center = scale_min + (adjusted_tendency * scale_range)

        # Reverse for reverse-coded items
        if is_reverse:
            center = scale_max - (center - scale_min)

        # Variance from traits
        variance = traits.get('variance_tendency',
                             traits.get('scale_use_breadth', 0.8))
        sd = (scale_range / 4) * variance

        # Generate response
        response = rng.normal(center, sd)

        # Extreme responder adjustment
        extreme_tendency = traits.get('extreme_tendency', 0.2)
        if rng.random() < extreme_tendency * 0.5:
            if response > (scale_min + scale_max) / 2:
                response = scale_max - rng.uniform(0, 0.8)
            else:
                response = scale_min + rng.uniform(0, 0.8)

        # Acquiescence bias
        acquiescence = traits.get('acquiescence', 0.5)
        if not is_reverse and acquiescence > 0.6:
            response += (acquiescence - 0.5) * scale_range * 0.1

        # Truncate to valid range
        response = max(scale_min, min(scale_max, round(response)))

        return int(response)

    def _generate_attention_check(
        self,
        condition: str,
        traits: Dict[str, float],
        check_type: str,
        participant_seed: int
    ) -> Tuple[int, bool]:
        """
        Generate attention/manipulation check response.

        Returns:
            (response_value, passed_check)
        """
        rng = np.random.RandomState(participant_seed)

        attention = traits.get('attention_level', 0.85)
        is_attentive = rng.random() < attention * self.attention_rate

        if check_type == 'ai_manipulation':
            # AI mentioned check - correct depends on condition
            correct = 1 if 'ai' in condition.lower() and 'no ai' not in condition.lower() else 2
            if is_attentive:
                return correct, True
            else:
                return 3 - correct, False  # Wrong answer

        elif check_type == 'product_type':
            # Hedonic/utilitarian check
            if 'hedonic' in condition.lower():
                correct = 7  # Hedonic end of scale
            elif 'utilitarian' in condition.lower():
                correct = 1  # Utilitarian end of scale
            else:
                correct = 4  # Middle

            if is_attentive:
                return correct + int(rng.normal(0, 0.8)), True
            else:
                return int(rng.uniform(1, 7)), False

        else:
            # Generic attention check
            if is_attentive:
                return 1, True  # Correct answer
            else:
                return rng.randint(2, 5), False

    def _generate_open_response(
        self,
        question_spec: Dict[str, Any],
        persona: Persona,
        traits: Dict[str, float],
        condition: str,
        participant_seed: int
    ) -> str:
        """Generate open-ended text response based on persona and context."""
        response_type = question_spec.get('type', 'task_summary')

        # Determine persona style
        if traits.get('attention_level', 0.8) < 0.5:
            style = 'careless'
        elif persona.name == 'Satisficer':
            style = 'satisficer'
        elif persona.name == 'Extreme Responder':
            style = 'extreme'
        elif persona.name == 'Engaged Responder':
            style = 'engaged'
        else:
            style = 'default'

        # Build context
        context = {
            'topic': question_spec.get('topic', 'the presented content'),
            'stimulus': question_spec.get('stimulus', 'product recommendation'),
            'product': question_spec.get('product', 'product'),
            'feature': question_spec.get('feature', 'features'),
            'emotion': np.random.choice(['pleased', 'interested', 'satisfied', 'engaged'])
        }

        # Add condition-specific context
        if 'ai' in condition.lower():
            context['stimulus'] = 'AI-recommended ' + context['stimulus']
        if 'hedonic' in condition.lower():
            context['product'] = 'hedonic ' + context['product']
        elif 'utilitarian' in condition.lower():
            context['product'] = 'functional ' + context['product']

        return self.text_generator.generate_response(
            response_type, style, context, traits, participant_seed
        )

    def _generate_demographics(self, n: int) -> pd.DataFrame:
        """Generate demographic variables with realistic distributions."""
        rng = np.random.RandomState(self.seed + 1000)

        # Age: truncated normal
        age_mean = self.demographics.get('age_mean', 35)
        age_sd = self.demographics.get('age_sd', 12)
        ages = rng.normal(age_mean, age_sd, n)
        ages = np.clip(ages, 18, 70).astype(int)

        # Gender
        male_pct = self.demographics.get('gender_quota', 50) / 100
        # Realistic distribution: Male, Female, Non-binary, Prefer not to say
        female_pct = (1 - male_pct) * 0.96
        nonbinary_pct = 0.025
        pnts_pct = 0.015

        total = male_pct + female_pct + nonbinary_pct + pnts_pct
        genders = rng.choice(
            [1, 2, 3, 4],
            size=n,
            p=[male_pct/total, female_pct/total, nonbinary_pct/total, pnts_pct/total]
        )

        return pd.DataFrame({
            'Age': ages,
            'Gender': genders
        })

    def _generate_condition_assignment(self, n: int) -> pd.Series:
        """Balanced random assignment to conditions."""
        n_conditions = len(self.conditions)
        n_per = n // n_conditions
        remainder = n % n_conditions

        assignments = []
        for i, cond in enumerate(self.conditions):
            count = n_per + (1 if i < remainder else 0)
            assignments.extend([cond] * count)

        # Shuffle with seed
        rng = np.random.RandomState(self.seed + 2000)
        rng.shuffle(assignments)

        return pd.Series(assignments, name='CONDITION')

    def _simulate_exclusion_flags(
        self,
        attention_checks_passed: List[bool],
        traits: Dict[str, float],
        responses: Dict[str, List[int]],
        participant_seed: int
    ) -> Dict[str, Any]:
        """
        Simulate exclusion-relevant variables.

        Returns dict with exclusion flags and related variables.
        """
        rng = np.random.RandomState(participant_seed)

        # Completion time (seconds)
        base_time = 300  # 5 minutes base
        attention = traits.get('attention_level', 0.8)

        if attention < 0.5:
            # Careless = faster
            completion_time = int(rng.uniform(45, 150))
        elif attention > 0.9:
            # Very careful = slightly longer
            completion_time = int(rng.normal(base_time * 1.2, 60))
        else:
            completion_time = int(rng.normal(base_time, 90))

        completion_time = max(30, min(1800, completion_time))

        # Attention check pass rate
        total_checks = len(attention_checks_passed)
        passed_checks = sum(attention_checks_passed)
        pass_rate = passed_checks / total_checks if total_checks > 0 else 1.0

        # Straight-line detection (consecutive same responses)
        max_straight_line = 0
        for col_values in responses.values():
            if len(col_values) > 0:
                current_streak = 1
                for i in range(1, len(col_values)):
                    if col_values[i] == col_values[i-1]:
                        current_streak += 1
                        max_straight_line = max(max_straight_line, current_streak)
                    else:
                        current_streak = 1

        # Determine exclusion flags
        exclude_time = (completion_time < self.exclusion_criteria.completion_time_min_seconds or
                       completion_time > self.exclusion_criteria.completion_time_max_seconds)
        exclude_attention = pass_rate < self.exclusion_criteria.attention_check_threshold
        exclude_straightline = max_straight_line >= self.exclusion_criteria.straight_line_threshold

        return {
            'completion_time_seconds': completion_time,
            'attention_check_pass_rate': round(pass_rate, 2),
            'max_straight_line': max_straight_line,
            'flag_completion_time': exclude_time,
            'flag_attention': exclude_attention,
            'flag_straight_line': exclude_straightline,
            'exclude_recommended': exclude_time or exclude_attention or exclude_straightline
        }

    def generate(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generate the complete simulated dataset.

        Returns:
            Tuple of (DataFrame with all data, metadata dict)
        """
        n = self.sample_size
        data = {}

        # ================================================================
        # CORE IDENTIFIERS
        # ================================================================
        data['PARTICIPANT_ID'] = list(range(1, n + 1))
        data['RUN_ID'] = [self.run_id] * n
        data['SIMULATION_MODE'] = [self.mode.upper()] * n
        data['SIMULATION_SEED'] = [self.seed] * n

        self.column_info.extend([
            ('PARTICIPANT_ID', 'Unique participant identifier (1-N)'),
            ('RUN_ID', 'Simulation run identifier'),
            ('SIMULATION_MODE', 'Simulation mode: PILOT or FINAL'),
            ('SIMULATION_SEED', 'Random seed for reproducibility')
        ])

        # ================================================================
        # CONDITION ASSIGNMENT
        # ================================================================
        conditions = self._generate_condition_assignment(n)
        data['CONDITION'] = conditions.tolist()
        self.column_info.append(('CONDITION', f'Experimental condition: {", ".join(self.conditions)}'))

        # ================================================================
        # DEMOGRAPHICS
        # ================================================================
        demographics_df = self._generate_demographics(n)
        data['Age'] = demographics_df['Age'].tolist()
        data['Gender'] = demographics_df['Gender'].tolist()
        self.column_info.extend([
            ('Age', f'Participant age (18-70, M~{self.demographics.get("age_mean", 35)})'),
            ('Gender', 'Gender: 1=Male, 2=Female, 3=Non-binary, 4=Prefer not to say')
        ])

        # ================================================================
        # ASSIGN PERSONAS AND GENERATE TRAITS
        # ================================================================
        assigned_personas = []
        all_traits = []
        for i in range(n):
            persona_name, persona = self._assign_persona(i)
            traits = self._generate_participant_traits(i, persona)
            assigned_personas.append(persona_name)
            all_traits.append(traits)

        data['_PERSONA'] = assigned_personas  # Internal tracking

        # ================================================================
        # ATTENTION/MANIPULATION CHECKS
        # ================================================================
        attention_results = []
        ai_check_values = []
        for i in range(n):
            p_seed = (self.seed + i * 100) % (2**31)
            check_val, passed = self._generate_attention_check(
                conditions.iloc[i], all_traits[i], 'ai_manipulation', p_seed
            )
            ai_check_values.append(check_val)
            attention_results.append([passed])

        data['AI_Mentioned_Check'] = ai_check_values
        self.column_info.append(('AI_Mentioned_Check', 'Manipulation check: Was AI mentioned? 1=Yes, 2=No'))

        # ================================================================
        # SCALE RESPONSES
        # ================================================================
        scale_responses = {}  # Track for exclusion analysis

        for scale in self.scales:
            scale_name = scale['name'].replace(' ', '_')
            scale_points = scale.get('scale_points', 6)
            num_items = scale.get('num_items', 5)
            reverse_items = scale.get('reverse_items', [])

            scale_responses[scale_name] = []

            for item_num in range(1, num_items + 1):
                col_name = f"{scale_name}_{item_num}"
                is_reverse = item_num in reverse_items

                item_values = []
                for i in range(n):
                    p_seed = (self.seed + i * 100 + hash(col_name)) % (2**31)
                    val = self._generate_scale_response(
                        1, scale_points, all_traits[i], is_reverse,
                        conditions.iloc[i], scale_name, p_seed
                    )
                    item_values.append(val)
                    scale_responses[scale_name].append(val)

                data[col_name] = item_values

                reverse_note = " (reverse-coded)" if is_reverse else ""
                self.column_info.append(
                    (col_name, f'{scale["name"]} item {item_num} (1-{scale_points}){reverse_note}')
                )

        # ================================================================
        # ADDITIONAL VARIABLES
        # ================================================================
        for var in self.additional_vars:
            var_name = var['name'].replace(' ', '_')
            var_min = var.get('min', 0)
            var_max = var.get('max', 10)

            values = []
            for i in range(n):
                p_seed = (self.seed + i * 100 + hash(var_name)) % (2**31)
                val = self._generate_scale_response(
                    var_min, var_max, all_traits[i], False,
                    conditions.iloc[i], var_name, p_seed
                )
                values.append(val)

            data[var_name] = values
            self.column_info.append((var_name, f'{var["name"]} ({var_min}-{var_max})'))

        # ================================================================
        # PRODUCT TYPE MANIPULATION CHECK (if hedonic/utilitarian)
        # ================================================================
        has_product_factor = any(
            'hedonic' in str(f.get('levels', [])).lower() or
            'utilitarian' in str(f.get('levels', [])).lower()
            for f in self.factors
        )

        if has_product_factor:
            hedonic_values = []
            for i in range(n):
                p_seed = (self.seed + i * 100 + 9999) % (2**31)
                val, passed = self._generate_attention_check(
                    conditions.iloc[i], all_traits[i], 'product_type', p_seed
                )
                hedonic_values.append(int(np.clip(val, 1, 7)))
                attention_results[i].append(passed)

            data['Hedonic_Utilitarian'] = hedonic_values
            self.column_info.append(
                ('Hedonic_Utilitarian', 'Product type perception: 1=Utilitarian, 7=Hedonic')
            )

        # ================================================================
        # OPEN-ENDED RESPONSES
        # ================================================================
        # Default task summary if no custom questions specified
        if not self.open_ended_questions:
            self.open_ended_questions = [{
                'name': 'Task_Summary',
                'type': 'task_summary',
                'topic': 'product recommendations',
                'stimulus': 'product recommendation'
            }]

        for q in self.open_ended_questions:
            col_name = q['name'].replace(' ', '_')
            responses = []
            for i in range(n):
                p_seed = (self.seed + i * 100 + hash(col_name)) % (2**31)
                persona_name = assigned_personas[i]
                persona = self.available_personas[persona_name]
                text = self._generate_open_response(
                    q, persona, all_traits[i], conditions.iloc[i], p_seed
                )
                responses.append(text)

            data[col_name] = responses
            self.column_info.append((col_name, f'Open-ended response: {q.get("type", "text")}'))

        # ================================================================
        # EXCLUSION CRITERIA VARIABLES
        # ================================================================
        exclusion_data = []
        for i in range(n):
            p_seed = (self.seed + i * 100 + 88888) % (2**31)
            excl = self._simulate_exclusion_flags(
                attention_results[i], all_traits[i], scale_responses, p_seed
            )
            exclusion_data.append(excl)

        data['Completion_Time_Seconds'] = [e['completion_time_seconds'] for e in exclusion_data]
        data['Attention_Pass_Rate'] = [e['attention_check_pass_rate'] for e in exclusion_data]
        data['Max_Straight_Line'] = [e['max_straight_line'] for e in exclusion_data]
        data['Flag_Speed'] = [1 if e['flag_completion_time'] else 0 for e in exclusion_data]
        data['Flag_Attention'] = [1 if e['flag_attention'] else 0 for e in exclusion_data]
        data['Flag_StraightLine'] = [1 if e['flag_straight_line'] else 0 for e in exclusion_data]
        data['Exclude_Recommended'] = [1 if e['exclude_recommended'] else 0 for e in exclusion_data]

        self.column_info.extend([
            ('Completion_Time_Seconds', 'Survey completion time in seconds'),
            ('Attention_Pass_Rate', 'Proportion of attention checks passed (0-1)'),
            ('Max_Straight_Line', 'Maximum consecutive identical responses'),
            ('Flag_Speed', 'Flagged for completion time: 1=Yes, 0=No'),
            ('Flag_Attention', 'Flagged for attention checks: 1=Yes, 0=No'),
            ('Flag_StraightLine', 'Flagged for straight-lining: 1=Yes, 0=No'),
            ('Exclude_Recommended', 'Recommended for exclusion: 1=Yes, 0=No')
        ])

        # ================================================================
        # REMOVE INTERNAL COLUMNS FROM FINAL OUTPUT
        # ================================================================
        del data['_PERSONA']

        # ================================================================
        # CREATE DATAFRAME
        # ================================================================
        df = pd.DataFrame(data)

        # ================================================================
        # PREPARE METADATA
        # ================================================================
        metadata = {
            'run_id': self.run_id,
            'simulation_mode': self.mode,
            'seed': self.seed,
            'generation_timestamp': datetime.now().isoformat(),
            'study_title': self.study_title,
            'study_description': self.study_description,
            'detected_domains': self.detected_domains,
            'sample_size': self.sample_size,
            'conditions': self.conditions,
            'factors': self.factors,
            'scales': self.scales,
            'effect_sizes': [
                {
                    'variable': e.variable,
                    'factor': e.factor,
                    'cohens_d': e.cohens_d,
                    'direction': e.direction
                } for e in self.effect_sizes
            ] if self.effect_sizes else [],
            'personas_used': list(set(assigned_personas)),
            'persona_distribution': {
                p: assigned_personas.count(p) / len(assigned_personas)
                for p in set(assigned_personas)
            },
            'exclusion_summary': {
                'flagged_speed': sum(data['Flag_Speed']),
                'flagged_attention': sum(data['Flag_Attention']),
                'flagged_straightline': sum(data['Flag_StraightLine']),
                'total_excluded': sum(data['Exclude_Recommended'])
            }
        }

        return df, metadata

    def generate_explainer(self) -> str:
        """Generate column explainer document."""
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
            ""
        ]

        for col_name, description in self.column_info:
            lines.append(f"{col_name}")
            lines.append(f"    {description}")
            lines.append("")

        lines.extend([
            "-" * 70,
            "EXPERIMENTAL CONDITIONS",
            "-" * 70,
            ""
        ])

        n_per = self.sample_size // len(self.conditions)
        for cond in self.conditions:
            lines.append(f"  - {cond} (target n = {n_per})")

        if self.effect_sizes:
            lines.extend([
                "",
                "-" * 70,
                "EXPECTED EFFECT SIZES",
                "-" * 70,
                ""
            ])
            for effect in self.effect_sizes:
                lines.append(
                    f"  {effect.variable}: {effect.level_high} > {effect.level_low}, "
                    f"Cohen's d = {effect.cohens_d}"
                )

        lines.extend([
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
            "=" * 70
        ])

        return "\n".join(lines)

    def generate_r_export(self, df: pd.DataFrame) -> str:
        """
        Generate R-compatible export with proper factor coding.

        Returns R script that loads and prepares the data.
        """
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
            f'data$CONDITION <- factor(data$CONDITION, levels = c({", ".join([f\'"{c}\"\' for c in self.conditions])}))',
            "",
            "# Convert Gender to factor",
            'data$Gender <- factor(data$Gender, levels = 1:4,',
            '                       labels = c("Male", "Female", "Non-binary", "Prefer not to say"))',
            "",
        ]

        # Add scale composites
        for scale in self.scales:
            scale_name = scale['name'].replace(' ', '_')
            items = [f"{scale_name}_{i}" for i in range(1, scale['num_items'] + 1)]
            reverse_items = scale.get('reverse_items', [])

            if reverse_items:
                lines.append(f"# {scale['name']} - reverse code items {reverse_items}")
                for r_item in reverse_items:
                    item_name = f"{scale_name}_{r_item}"
                    max_val = scale['scale_points']
                    lines.append(
                        f'data${item_name}_R <- {max_val + 1} - data${item_name}'
                    )

            lines.append(f"# Create {scale['name']} composite")
            item_list = ', '.join([f'data${item}' for item in items])
            lines.append(f'data${scale_name}_composite <- rowMeans(cbind({item_list}), na.rm = TRUE)')
            lines.append("")

        lines.extend([
            "# Filter excluded participants (optional)",
            "data_clean <- data[data$Exclude_Recommended == 0, ]",
            "",
            f'cat("Total N:", nrow(data), "\\n")',
            f'cat("Clean N:", nrow(data_clean), "\\n")',
            "",
            "# Ready for analysis!",
        ])

        return "\n".join(lines)


# Export
__all__ = ['EnhancedSimulationEngine', 'EffectSizeSpec', 'ExclusionCriteria']
