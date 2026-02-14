"""Game-specific prompt templates for LLM strategy modules.

Each prompt template:
  - Describes the game rules and persona traits
  - Requests structured JSON output with action probabilities
  - Includes parsing and repair rules for malformed responses
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from ....persona import Persona
from ....agents.backend import GameState


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are simulating a participant in a behavioral economics experiment. "
    "You must respond with ONLY a JSON object — no explanation, no preamble. "
    "The JSON must map action names to probabilities that sum to 1.0."
)

# ---------------------------------------------------------------------------
# Game-specific templates
# ---------------------------------------------------------------------------

def dictator_prompt(state: GameState, persona: Persona) -> str:
    endow = state.game_params.get("endowment", 10)
    step = state.game_params.get("step", 1)
    p = persona.params
    return (
        f"GAME: Dictator Game\n"
        f"You have ${endow} and must decide how much to give to another person.\n"
        f"You can give any amount from $0 to ${endow} in steps of ${step}.\n\n"
        f"YOUR PERSONALITY:\n"
        f"- Fairness concern (alpha): {p.get('fairness_alpha', 0.8):.2f}\n"
        f"- Guilt from advantageous inequity (beta): {p.get('fairness_beta', 0.2):.2f}\n"
        f"- General prosociality: {p.get('prosociality', 0.0):.2f}\n"
        f"- Norm sensitivity: {p.get('norm_weight', 0.0):.2f}\n\n"
        f"Return a JSON object mapping each possible giving amount to a probability.\n"
        f'Example: {{"0": 0.1, "1": 0.1, "2": 0.15, ...}}\n'
        f"Probabilities must sum to 1.0."
    )


def trust_prompt(state: GameState, persona: Persona) -> str:
    endow = state.game_params.get("endowment", 10)
    mult = state.game_params.get("multiplier", 3)
    p = persona.params
    return (
        f"GAME: Trust Game\n"
        f"You are the SENDER. You have ${endow}.\n"
        f"Any amount you send is multiplied by {mult} and received by the other player.\n"
        f"The other player can then return any amount back to you.\n\n"
        f"YOUR PERSONALITY:\n"
        f"- Trust/prosociality: {p.get('prosociality', 0.0):.2f}\n"
        f"- Reciprocity expectation: {p.get('reciprocity', 0.5):.2f}\n"
        f"- Risk aversion: {p.get('risk_aversion', 0.5):.2f}\n\n"
        f"Return a JSON mapping each possible send amount to a probability.\n"
        f'Example: {{"0": 0.1, "5": 0.3, "10": 0.6}}\n'
        f"Probabilities must sum to 1.0."
    )


def money_request_prompt(state: GameState, persona: Persona) -> str:
    min_r = state.game_params.get("min_request", 11)
    max_r = state.game_params.get("max_request", 20)
    bonus = state.game_params.get("bonus", 20)
    p = persona.params
    return (
        f"GAME: Money Request Game\n"
        f"You and another player each request an amount from ${min_r} to ${max_r}.\n"
        f"You receive whatever you request.\n"
        f"BONUS: If you request exactly $1 less than the other player, you get an extra ${bonus}.\n\n"
        f"YOUR PERSONALITY:\n"
        f"- Strategic thinking depth: {p.get('strategic_depth', 1.0):.2f}\n"
        f"- Risk aversion: {p.get('risk_aversion', 0.5):.2f}\n"
        f"- Noise/randomness: {p.get('noise_lambda', 1.0):.2f}\n\n"
        f"Return a JSON mapping each possible request to a probability.\n"
        f'Example: {{"{min_r}": 0.1, "{min_r+1}": 0.2, ...}}\n'
        f"Probabilities must sum to 1.0."
    )


def generic_prompt(state: GameState, persona: Persona) -> str:
    actions = [a.name for a in state.available_actions]
    p_desc = ", ".join(f"{k}={v:.2f}" for k, v in sorted(persona.params.items()))
    return (
        f"GAME: {state.game_name}\n"
        f"Parameters: {json.dumps(state.game_params)}\n"
        f"Available actions: {actions}\n\n"
        f"YOUR PERSONALITY: {p_desc}\n\n"
        f"Return a JSON mapping each action to a probability. Sum must be 1.0.\n"
        f'Example: {{"{actions[0]}": 0.5, "{actions[-1] if len(actions) > 1 else actions[0]}": 0.5}}'
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

PROMPT_REGISTRY: Dict[str, Any] = {
    "dictator": dictator_prompt,
    "trust": trust_prompt,
    "money_request_11_20": money_request_prompt,
}


def get_game_prompt(state: GameState, persona: Persona) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a game.

    Returns
    -------
    tuple of (str, str)
        System prompt and user prompt.
    """
    template_fn = PROMPT_REGISTRY.get(state.game_name, generic_prompt)
    return SYSTEM_PROMPT, template_fn(state, persona)
