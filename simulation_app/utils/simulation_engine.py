"""
Simulation Engine for BDS5010 Behavioral Experiment Simulation Tool
====================================================================
Core module for generating synthetic behavioral experiment data following
the methodology from "Simulating Behavioral Experiments with ChatGPT-5"

This engine uses theory-grounded personas to create realistic variance in responses.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import random
import string
from datetime import datetime


class SimulationEngine:
    """
    Main simulation engine that generates synthetic behavioral experiment data.

    The engine follows these principles:
    1. Theory-grounded personas for realistic heterogeneity
    2. Condition-balanced allocation
    3. Realistic attention/manipulation check failure rates
    4. Proper scale response patterns (no floor/ceiling effects)
    """

    def __init__(
        self,
        sample_size: int,
        conditions: List[str],
        factors: List[Dict[str, Any]],
        scales: List[Dict[str, Any]],
        additional_vars: List[Dict[str, Any]],
        demographics: Dict[str, Any],
        attention_rate: float = 0.95,
        random_responder_rate: float = 0.05,
        persona_weights: Dict[str, float] = None
    ):
        """
        Initialize the simulation engine.

        Args:
            sample_size: Total number of participants to simulate
            conditions: List of condition names (e.g., ["AI x Hedonic", "AI x Utilitarian", ...])
            factors: List of factor definitions with names and levels
            scales: List of scale definitions with names, items, and points
            additional_vars: List of single-item variables with min/max
            demographics: Dict with gender_quota, age_mean, age_sd
            attention_rate: Probability of passing attention checks (0-1)
            random_responder_rate: Proportion of random responders (0-1)
            persona_weights: Dict mapping persona names to weights
        """
        self.sample_size = sample_size
        self.conditions = conditions
        self.factors = factors
        self.scales = scales
        self.additional_vars = additional_vars
        self.demographics = demographics
        self.attention_rate = attention_rate
        self.random_responder_rate = random_responder_rate

        # Default persona weights if not provided
        self.persona_weights = persona_weights or {
            'engaged': 0.30,
            'satisficer': 0.25,
            'extreme': 0.10,
            'neutral': 0.35
        }

        # Normalize persona weights
        total = sum(self.persona_weights.values())
        if total > 0:
            self.persona_weights = {k: v/total for k, v in self.persona_weights.items()}

        # Set random seed for reproducibility within session
        self.seed = int(datetime.now().timestamp()) % 1000000
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Track generated columns for explainer
        self.column_info = []

    def _assign_persona(self) -> str:
        """Assign a behavioral persona to a participant."""
        personas = list(self.persona_weights.keys())
        weights = list(self.persona_weights.values())
        return np.random.choice(personas, p=weights)

    def _generate_participant_traits(self, persona: str) -> Dict[str, float]:
        """
        Generate trait parameters for a participant based on their persona.

        Returns dict with:
        - response_tendency: Center point preference (0.3-0.7)
        - variance_tendency: Response variability (0.5-1.5)
        - attention_level: Probability of attending to checks (0.7-1.0)
        - extreme_tendency: Likelihood of using scale endpoints (0-1)
        """
        if persona == 'engaged':
            return {
                'response_tendency': np.random.uniform(0.4, 0.6),
                'variance_tendency': np.random.uniform(0.8, 1.2),
                'attention_level': np.random.uniform(0.92, 0.99),
                'extreme_tendency': np.random.uniform(0.1, 0.3)
            }
        elif persona == 'satisficer':
            return {
                'response_tendency': np.random.uniform(0.45, 0.55),  # More central
                'variance_tendency': np.random.uniform(0.4, 0.7),   # Lower variance
                'attention_level': np.random.uniform(0.75, 0.90),
                'extreme_tendency': np.random.uniform(0.0, 0.15)
            }
        elif persona == 'extreme':
            return {
                'response_tendency': np.random.choice([0.2, 0.8]),  # Skewed
                'variance_tendency': np.random.uniform(1.0, 1.5),
                'attention_level': np.random.uniform(0.80, 0.95),
                'extreme_tendency': np.random.uniform(0.5, 0.8)
            }
        else:  # neutral
            return {
                'response_tendency': np.random.uniform(0.35, 0.65),
                'variance_tendency': np.random.uniform(0.7, 1.1),
                'attention_level': np.random.uniform(0.85, 0.95),
                'extreme_tendency': np.random.uniform(0.15, 0.35)
            }

    def _generate_scale_response(
        self,
        scale_min: int,
        scale_max: int,
        traits: Dict[str, float],
        is_reverse: bool = False
    ) -> int:
        """
        Generate a single scale response based on participant traits.

        Uses a truncated normal distribution centered on the participant's
        response tendency, with variance determined by their traits.
        """
        scale_range = scale_max - scale_min

        # Calculate center point based on traits
        center = scale_min + (traits['response_tendency'] * scale_range)

        # Reverse for reverse-coded items
        if is_reverse:
            center = scale_max - (center - scale_min)

        # Add noise based on variance tendency
        sd = (scale_range / 4) * traits['variance_tendency']

        # Generate response
        response = np.random.normal(center, sd)

        # Extreme responder adjustment
        if np.random.random() < traits['extreme_tendency']:
            if response > (scale_min + scale_max) / 2:
                response = scale_max - np.random.uniform(0, 1)
            else:
                response = scale_min + np.random.uniform(0, 1)

        # Truncate to valid range and round
        response = max(scale_min, min(scale_max, round(response)))

        return int(response)

    def _generate_demographics(self, n: int) -> pd.DataFrame:
        """Generate demographic data for n participants."""

        # Age: truncated normal
        ages = np.random.normal(
            self.demographics.get('age_mean', 35),
            self.demographics.get('age_sd', 12),
            n
        )
        ages = np.clip(ages, 18, 65).astype(int)

        # Gender: based on quota
        male_pct = self.demographics.get('gender_quota', 50) / 100
        female_pct = (1 - male_pct) * 0.98  # Leave 2% for other options
        nonbinary_pct = 0.01
        prefer_not_pct = 0.01

        # Normalize
        total = male_pct + female_pct + nonbinary_pct + prefer_not_pct
        genders = np.random.choice(
            [1, 2, 3, 4],  # 1=Male, 2=Female, 3=Non-binary, 4=Prefer not to say
            size=n,
            p=[male_pct/total, female_pct/total, nonbinary_pct/total, prefer_not_pct/total]
        )

        return pd.DataFrame({
            'Age': ages,
            'Gender': genders
        })

    def _generate_condition_assignment(self, n: int) -> pd.Series:
        """Assign participants to conditions (balanced)."""
        n_conditions = len(self.conditions)
        n_per_condition = n // n_conditions
        remainder = n % n_conditions

        assignments = []
        for i, cond in enumerate(self.conditions):
            count = n_per_condition + (1 if i < remainder else 0)
            assignments.extend([cond] * count)

        np.random.shuffle(assignments)
        return pd.Series(assignments, name='CONDITION')

    def _generate_attention_check(
        self,
        condition: str,
        traits: Dict[str, float],
        check_type: str = 'manipulation'
    ) -> int:
        """
        Generate attention/manipulation check response.

        For manipulation checks, correct answer depends on condition.
        For general attention checks, there's one correct answer.
        """
        # Determine if participant is attentive
        is_attentive = np.random.random() < traits['attention_level'] * self.attention_rate

        if check_type == 'manipulation':
            # Correct answer depends on condition
            # Assume "AI" in condition name means they should answer "Yes" (1)
            correct_answer = 1 if 'AI' in condition.upper() and 'NO' not in condition.upper() else 2

            if is_attentive:
                return correct_answer
            else:
                return 3 - correct_answer  # Wrong answer
        else:
            # General attention check - correct answer is typically a specific value
            if is_attentive:
                return 1  # Correct
            else:
                return np.random.randint(2, 5)  # Random wrong answer

    def _generate_open_response(self) -> str:
        """Generate a plausible open-ended task summary response."""
        templates = [
            "I looked at a product recommendation and rated my feelings about it.",
            "The study was about product recommendations and purchasing decisions.",
            "I saw a product and answered questions about my interest in buying it.",
            "I evaluated a recommended product and answered questions about ownership feelings.",
            "This survey asked about my reactions to a product suggestion.",
            "I rated my feelings about a product that was recommended to me.",
            "The task involved viewing a product and rating my purchase intentions.",
            "I saw product recommendations and rated how I felt about them.",
            "I answered questions about a product shown to me during shopping.",
            "The study measured my responses to online product recommendations.",
            "I evaluated a product and answered questions about wanting to buy it.",
            "This was about product recommendations and consumer attitudes.",
            "I looked at products and answered questions about ownership and purchase.",
            "The survey asked about my feelings toward recommended products.",
            "I rated products and answered questions about my shopping preferences.",
            "This study examined reactions to product suggestions online.",
            "I saw a shopping scenario and rated my interest in the product.",
            "The task involved evaluating products and stating purchase intentions.",
            "I answered questions about how product recommendations affect me.",
            "This was a study about online shopping and product attitudes."
        ]
        return np.random.choice(templates)

    def generate(self) -> pd.DataFrame:
        """
        Generate the complete simulated dataset.

        Returns:
            DataFrame with all simulated data
        """
        n = self.sample_size

        # Initialize data dictionary
        data = {}

        # Core identifiers
        data['PARTICIPANT_ID'] = list(range(1, n + 1))
        data['RUN_ID'] = [f"SIM_{datetime.now().strftime('%Y%m%d_%H%M%S')}"] * n
        data['CHUNK'] = [1] * n

        self.column_info.extend([
            ('PARTICIPANT_ID', 'Unique participant identifier (1-N)'),
            ('RUN_ID', 'Simulation run identifier with timestamp'),
            ('CHUNK', 'Data chunk number (always 1 for single-run)')
        ])

        # Condition assignment
        conditions = self._generate_condition_assignment(n)
        data['CONDITION'] = conditions.tolist()
        self.column_info.append(('CONDITION', f'Experimental condition: {", ".join(self.conditions)}'))

        # Random ID (simulated embedded data)
        data['Random_ID'] = [np.random.randint(10000, 99999) for _ in range(n)]
        self.column_info.append(('Random_ID', 'Simulated Qualtrics embedded random ID'))

        # Demographics
        demographics_df = self._generate_demographics(n)
        data['Age'] = demographics_df['Age'].tolist()
        data['Gender'] = demographics_df['Gender'].tolist()
        self.column_info.extend([
            ('Age', f'Participant age (18-65, M~{self.demographics.get("age_mean", 35)})'),
            ('Gender', 'Gender: 1=Male, 2=Female, 3=Non-binary, 4=Prefer not to say')
        ])

        # Generate persona and traits for each participant
        personas = [self._assign_persona() for _ in range(n)]
        all_traits = [self._generate_participant_traits(p) for p in personas]

        # Manipulation check
        data['AI_Mentioned_Check'] = [
            self._generate_attention_check(conditions.iloc[i], all_traits[i], 'manipulation')
            for i in range(n)
        ]
        self.column_info.append(('AI_Mentioned_Check', 'Manipulation check: Was AI mentioned? 1=Yes, 2=No'))

        # Generate scale responses
        for scale in self.scales:
            scale_name = scale['name'].replace(' ', '_')
            scale_points = scale.get('scale_points', 6)
            num_items = scale.get('num_items', 5)
            reverse_items = scale.get('reverse_items', [])

            for item_num in range(1, num_items + 1):
                col_name = f"{scale_name}_{item_num}"
                is_reverse = item_num in reverse_items

                data[col_name] = [
                    self._generate_scale_response(1, scale_points, all_traits[i], is_reverse)
                    for i in range(n)
                ]

                reverse_note = " (reverse-coded)" if is_reverse else ""
                self.column_info.append(
                    (col_name, f'{scale["name"]} item {item_num} (1-{scale_points}){reverse_note}')
                )

        # Generate additional single-item variables
        for var in self.additional_vars:
            var_name = var['name'].replace(' ', '_')
            var_min = var.get('min', 0)
            var_max = var.get('max', 10)

            # Generate with persona-based variance
            values = []
            for i in range(n):
                traits = all_traits[i]
                center = var_min + (traits['response_tendency'] * (var_max - var_min))
                sd = (var_max - var_min) / 4 * traits['variance_tendency']
                val = np.random.normal(center, sd)
                val = int(max(var_min, min(var_max, round(val))))
                values.append(val)

            data[var_name] = values
            self.column_info.append((var_name, f'{var["name"]} ({var_min}-{var_max})'))

        # Comprehension check (for WTP auction format)
        data['WTP_CompCheck'] = [
            1 if np.random.random() < all_traits[i]['attention_level'] * 0.87
            else np.random.randint(2, 5)
            for i in range(n)
        ]
        self.column_info.append(('WTP_CompCheck', 'WTP comprehension check: 1=Correct answer'))

        # Hedonic/Utilitarian rating (if conditions suggest product type manipulation)
        has_product_factor = any(
            'hedonic' in str(f.get('levels', [])).lower() or
            'utilitarian' in str(f.get('levels', [])).lower()
            for f in self.factors
        )

        if has_product_factor:
            hedonic_ratings = []
            for i in range(n):
                cond = conditions.iloc[i].lower()
                if 'hedonic' in cond:
                    # Hedonic products rated higher (toward 7)
                    base = 5.5 + np.random.normal(0, 1.1)
                else:
                    # Utilitarian products rated lower (toward 1)
                    base = 2.5 + np.random.normal(0, 1.0)
                hedonic_ratings.append(int(max(1, min(7, round(base)))))

            data['Hedonic_Utilitarian'] = hedonic_ratings
            self.column_info.append(
                ('Hedonic_Utilitarian', 'Product type perception: 1=Utilitarian, 7=Hedonic')
            )

        # Task summary (open-ended)
        data['Task_Summary'] = [self._generate_open_response() for _ in range(n)]
        self.column_info.append(('Task_Summary', 'Open-ended task summary response'))

        # MTurk confirmation
        data['MTurkID_Confirmed'] = [2] * n  # 2 = Yes (confirmed)
        self.column_info.append(('MTurkID_Confirmed', 'MTurk ID confirmation: 1=No, 2=Yes'))

        # Create DataFrame
        df = pd.DataFrame(data)

        return df

    def generate_explainer(self) -> str:
        """
        Generate a column explainer document describing all variables.

        Returns:
            String with formatted column descriptions
        """
        lines = [
            "=" * 70,
            "COLUMN EXPLAINER - Simulated Behavioral Experiment Data",
            "=" * 70,
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Sample Size: {self.sample_size}",
            f"Conditions: {len(self.conditions)}",
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
            lines.append(f"  - {cond} (n = {n_per})")

        lines.extend([
            "",
            "-" * 70,
            "FACTORS",
            "-" * 70,
            ""
        ])

        for factor in self.factors:
            lines.append(f"  {factor['name']}: {', '.join(factor['levels'])}")

        lines.extend([
            "",
            "-" * 70,
            "MEASUREMENT SCALES",
            "-" * 70,
            ""
        ])

        for scale in self.scales:
            reverse_str = ""
            if scale.get('reverse_items'):
                reverse_str = f" (items {scale['reverse_items']} reverse-coded)"
            lines.append(
                f"  {scale['name']}: {scale['num_items']} items, "
                f"{scale['scale_points']}-point scale{reverse_str}"
            )

        lines.extend([
            "",
            "-" * 70,
            "SIMULATION PARAMETERS",
            "-" * 70,
            "",
            f"  Attention check pass rate: {self.attention_rate * 100:.0f}%",
            f"  Random responder rate: {self.random_responder_rate * 100:.0f}%",
            f"  Random seed: {self.seed}",
            "",
            "  Persona weights:",
        ])

        for persona, weight in self.persona_weights.items():
            lines.append(f"    - {persona}: {weight * 100:.0f}%")

        lines.extend([
            "",
            "=" * 70,
            "END OF COLUMN EXPLAINER",
            "=" * 70
        ])

        return "\n".join(lines)
