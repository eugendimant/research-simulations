from __future__ import annotations
from typing import Dict, Any
import numpy as np

from .base import Game, GameOutcome
from ..decision import logit_choice

class SenderReceiverGame(Game):
    name = "sender_receiver"

    def simulate_one(self, rng: np.random.Generator, a, b, spec: Dict[str, Any]) -> GameOutcome:
        if b is None:
            raise ValueError("Sender-receiver requires two personas (sender a, receiver b).")

        # Two states; sender observes state. Each option gives different payoffs to sender/receiver.
        # state 0: option A good for receiver; state 1: option B good for receiver
        state = int(rng.integers(0, 2))
        # payoffs: (sender, receiver)
        A = tuple(spec.get("A", (5.0, 8.0)))
        B = tuple(spec.get("B", (8.0, 5.0)))
        # in state 0, receiver-optimal is A; in state 1, receiver-optimal is B
        receiver_opt = 0 if state == 0 else 1  # 0=A, 1=B

        lam_s = float(a.params.get("noise_lambda", 2.0))
        honesty_cost = float(a.params.get("honesty_cost", 1.0))

        # messages: recommend A (0) or recommend B (1)
        # sender prefers whichever gives higher sender payoff
        sender_pref = 0 if A[0] >= B[0] else 1
        truthful = receiver_opt

        utilities_msg = []
        for msg in [0, 1]:
            base = 0.0
            lie = (msg != truthful)
            base -= honesty_cost * (1.0 if lie else 0.0)
            # slight preference for steering to sender_pref
            base += 1.0 if msg == sender_pref else 0.0
            utilities_msg.append(base)
        idx_msg, probs_msg = logit_choice(rng, np.array(utilities_msg, dtype=float), lam=lam_s)
        msg = int(idx_msg)

        # receiver chooses action based on message, moderated by gullibility
        lam_r = float(b.params.get("noise_lambda", 2.0))
        gull = float(b.params.get("gullibility", 0.5))

        # receiver utility is own payoff; follows message with probability increasing in gullibility
        # Use logit on utilities that incorporate message alignment
        uA = float(A[1]) + gull * (1.0 if msg == 0 else -1.0)
        uB = float(B[1]) + gull * (1.0 if msg == 1 else -1.0)
        idx_act, probs_act = logit_choice(rng, np.array([uA, uB], dtype=float), lam=lam_r)
        act = int(idx_act)

        if act == 0:
            pay_s, pay_r = float(A[0]), float(A[1])
        else:
            pay_s, pay_r = float(B[0]), float(B[1])

        return GameOutcome(
            actions={"state": state, "message": "A" if msg == 0 else "B", "action": "A" if act == 0 else "B", "lied": bool(msg != truthful)},
            payoffs={"a": pay_s, "b": pay_r},
            trace={
                "A": A, "B": B, "receiver_opt": "A" if receiver_opt == 0 else "B",
                "sender_pref": "A" if sender_pref == 0 else "B",
                "msg_probs": probs_msg.tolist(), "act_probs": probs_act.tolist(),
                "params": {"honesty_cost_sender": honesty_cost, "gullibility_receiver": gull},
            },
        )
