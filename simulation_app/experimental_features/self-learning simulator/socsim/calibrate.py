from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json
import numpy as np
from sklearn.linear_model import Ridge

@dataclass
class CalibrationResult:
    updated_priors: Dict
    coefs: Dict[str, float]
    intercepts: Dict[str, float]
    r2_mean: float

def calibrate_ridge_per_moment(pred_csv: Path, obs_csv: Path, cols: List[str], priors_path: Path, alpha: float = 1.0) -> CalibrationResult:
    import pandas as pd
    pred = pd.read_csv(pred_csv)
    obs = pd.read_csv(obs_csv)

    coefs: Dict[str, float] = {}
    intercepts: Dict[str, float] = {}
    r2s: List[float] = []
    for c in cols:
        X = pred[[c]].to_numpy()
        y = obs[c].to_numpy()
        model = Ridge(alpha=float(alpha))
        model.fit(X, y)
        coefs[c] = float(model.coef_[0])
        intercepts[c] = float(model.intercept_)
        r2s.append(float(model.score(X, y)))

    pri = json.loads(priors_path.read_text(encoding="utf-8"))
    pri.setdefault("_calibration", {})
    pri["_calibration"]["method"] = "ridge_per_moment"
    pri["_calibration"]["alpha"] = float(alpha)
    pri["_calibration"]["coefs"] = coefs
    pri["_calibration"]["intercepts"] = intercepts
    pri["_calibration"]["r2_mean"] = float(np.mean(r2s)) if r2s else 0.0
    return CalibrationResult(updated_priors=pri, coefs=coefs, intercepts=intercepts, r2_mean=pri["_calibration"]["r2_mean"])
