"""Low-dimensional recalibration: p' ∝ p^T * exp(b). Fixes systematic over/under-confidence and
per-outcome bias (e.g. draws under-predicted) with only 4 parameters, so it is hard to overfit."""
from __future__ import annotations
import json
import numpy as np
from scipy.optimize import minimize


class Calibrator:
    def __init__(self, T: float = 1.0, b=None):
        self.T = float(T)
        self.b = np.zeros(3) if b is None else np.asarray(b, float)

    def apply(self, p):
        p = np.asarray(p, float)
        z = self.T * np.log(np.clip(p, 1e-9, 1)) + self.b
        z = z - z.max(axis=-1, keepdims=True)
        q = np.exp(z)
        return q / q.sum(axis=-1, keepdims=True)

    @classmethod
    def fit(cls, P, y, l2: float = 20.0) -> "Calibrator":
        P, y = np.asarray(P), np.asarray(y)
        n = len(y)
        if n < 50:
            return cls()

        def f(x):
            q = cls(x[0], x[1:]).apply(P)
            ll = -np.log(np.clip(q[np.arange(n), y], 1e-9, 1)).sum()
            return ll + l2 * ((x[0] - 1) ** 2 + (x[1:] ** 2).sum())  # shrink toward "no change"

        r = minimize(f, np.array([1.0, 0, 0, 0]), method="L-BFGS-B")
        return cls(r.x[0], r.x[1:])

    def to_dict(self):
        return {"T": self.T, "b": [float(v) for v in self.b]}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("T", 1.0), d.get("b"))
