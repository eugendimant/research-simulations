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


def ultimatum_prompt(state: GameState, persona: Persona) -> str:
    endow = state.game_params.get("endowment", 10)
    role = state.game_params.get("role", "proposer")
    p = persona.params
    if role == "proposer":
        return (
            f"GAME: Ultimatum Game (You are the PROPOSER)\n"
            f"You have ${endow} and must propose a split with another player.\n"
            f"If they accept, both get the proposed amounts.\n"
            f"If they reject, BOTH get $0.\n\n"
            f"YOUR PERSONALITY:\n"
            f"- Fairness concern: {p.get('fairness_alpha', 0.8):.2f}\n"
            f"- Strategic thinking: {p.get('strategic_depth', 1.0):.2f}\n"
            f"- Prosociality: {p.get('prosociality', 0.0):.2f}\n\n"
            f"Return a JSON mapping each possible offer to a probability.\n"
            f'Example: {{"0": 0.05, "3": 0.15, "5": 0.50, "7": 0.30}}\n'
            f"Probabilities must sum to 1.0."
        )
    return (
        f"GAME: Ultimatum Game (You are the RESPONDER)\n"
        f"The other player has ${endow} and will propose a split.\n"
        f"You can ACCEPT (both get proposed amounts) or REJECT (both get $0).\n\n"
        f"YOUR PERSONALITY:\n"
        f"- Fairness concern: {p.get('fairness_alpha', 0.8):.2f}\n"
        f"- Rejection threshold: proportional to alpha\n\n"
        f"Return a JSON mapping 'accept' and 'reject' to probabilities.\n"
        f'Example: {{"accept": 0.7, "reject": 0.3}}\n'
        f"Probabilities must sum to 1.0."
    )


def public_goods_prompt(state: GameState, persona: Persona) -> str:
    endow = state.game_params.get("endowment", 20)
    mpcr = state.game_params.get("mpcr", 0.4)
    group_size = state.game_params.get("group_size", 4)
    p = persona.params
    return (
        f"GAME: Public Goods Game\n"
        f"You have ${endow}. You are in a group of {group_size} people.\n"
        f"Each person decides how much to contribute to a shared pot.\n"
        f"The pot is multiplied by {mpcr * group_size:.1f} and split equally.\n"
        f"(MPCR = {mpcr}, so each $1 contributed returns ${mpcr:.2f} to each person)\n\n"
        f"YOUR PERSONALITY:\n"
        f"- Prosociality: {p.get('prosociality', 0.0):.2f}\n"
        f"- Norm sensitivity: {p.get('norm_weight', 0.0):.2f}\n"
        f"- Conformity: {p.get('conformity', 0.0):.2f}\n\n"
        f"Return a JSON mapping each possible contribution to a probability.\n"
        f'Example: {{"0": 0.1, "5": 0.2, "10": 0.3, "20": 0.4}}\n'
        f"Probabilities must sum to 1.0."
    )


def prisoners_dilemma_prompt(state: GameState, persona: Persona) -> str:
    payoffs = state.game_params.get("payoffs", {"CC": 3, "CD": 0, "DC": 5, "DD": 1})
    p = persona.params
    return (
        f"GAME: Prisoner's Dilemma\n"
        f"You and another player simultaneously choose to COOPERATE or DEFECT.\n"
        f"Payoffs: Both cooperate={payoffs.get('CC', 3)}, "
        f"You defect/they cooperate={payoffs.get('DC', 5)}, "
        f"You cooperate/they defect={payoffs.get('CD', 0)}, "
        f"Both defect={payoffs.get('DD', 1)}.\n\n"
        f"YOUR PERSONALITY:\n"
        f"- Prosociality: {p.get('prosociality', 0.0):.2f}\n"
        f"- Reciprocity: {p.get('reciprocity', 0.5):.2f}\n"
        f"- Risk aversion: {p.get('risk_aversion', 0.5):.2f}\n\n"
        f"Return a JSON: {{\"cooperate\": probability, \"defect\": probability}}\n"
        f"Probabilities must sum to 1.0."
    )


def beauty_contest_prompt(state: GameState, persona: Persona) -> str:
    p_frac = state.game_params.get("p_fraction", 2.0 / 3.0)
    max_guess = state.game_params.get("max_guess", 100)
    p = persona.params
    return (
        f"GAME: Beauty Contest (Guessing Game)\n"
        f"Everyone picks a number from 0 to {max_guess}.\n"
        f"The winner is whoever is closest to {p_frac:.2f} × the average of all guesses.\n\n"
        f"YOUR PERSONALITY:\n"
        f"- Strategic depth: {p.get('strategic_depth', 1.0):.2f}\n"
        f"- Level-0 guess would be ~{max_guess/2}.\n"
        f"- Level-1 guess: ~{max_guess/2 * p_frac:.0f}\n"
        f"- Level-2 guess: ~{max_guess/2 * p_frac**2:.0f}\n\n"
        f"Return a JSON mapping possible guesses to probabilities.\n"
        f'Example: {{"30": 0.3, "33": 0.4, "22": 0.3}}\n'
        f"Probabilities must sum to 1.0."
    )


def generic_prompt(state: GameState, persona: Persona) -> str:
    actions = [a.name for a in state.available_actions]
    p_desc = ", ".join(f"{k}={v:.2f}" for k, v in sorted(persona.params.items()))
    game_desc = f"GAME: {state.game_name}\n"
    if state.game_params:
        game_desc += f"Rules: {json.dumps(state.game_params, indent=2)}\n"
    return (
        f"{game_desc}"
        f"Available actions: {actions}\n"
        f"Round: {state.round_number}\n\n"
        f"YOUR PERSONALITY: {p_desc}\n\n"
        f"Think about how someone with these personality traits would play this game.\n"
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
    "ultimatum": ultimatum_prompt,
    "public_goods": public_goods_prompt,
    "pd": prisoners_dilemma_prompt,
    "prisoners_dilemma": prisoners_dilemma_prompt,
    "beauty_contest": beauty_contest_prompt,
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
