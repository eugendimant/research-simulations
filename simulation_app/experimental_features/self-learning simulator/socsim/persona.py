from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict
import numpy as np
from .utils import ParamPrior
from .causal import DoIntervention, draw_params

@dataclass
class Persona:
    id: str
    params: Dict[str, float]
    latent_class: str
    attributes: Dict[str, Any] = field(default_factory=dict)

class PersonaGenerator:
    def __init__(self, priors: Dict[str, ParamPrior], latent_classes: Dict[str, Any] | None = None):
        self.priors = priors
        self.latent_classes = latent_classes or {}

    def _class_weights(self) -> tuple[list[str], np.ndarray]:
        names, ws = [], []
        for k, v in self.latent_classes.items():
            names.append(k)
            ws.append(float(v.get("weight", 0.0)))
        if not names:
            return ["default"], np.array([1.0])
        w = np.array(ws, dtype=float)
        if w.sum() <= 0:
            w = np.ones_like(w)
        return names, w / w.sum()

    def sample(
        self,
        rng: np.random.Generator,
        persona_id: str,
        mean_shifts: Dict[str, float],
        extra_sd: Dict[str, float],
        intervention: DoIntervention | None = None,
    ) -> Persona:
        class_names, w = self._class_weights()
        cls = str(rng.choice(class_names, p=w))
        class_shift = self.latent_classes.get(cls, {}).get("shifts", {}) if cls != "default" else {}
        params = draw_params(
            rng=rng,
            priors=self.priors,
            mean_shifts=mean_shifts,
            extra_sd=extra_sd,
            class_shift=class_shift,
            intervention=intervention,
        )
        return Persona(id=persona_id, params=params, latent_class=cls)
