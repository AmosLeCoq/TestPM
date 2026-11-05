"""Modèle logit conditionnel (type Bolton & Chapman / Benter) : softmax des scores par course."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _segments(race_ids: np.ndarray) -> np.ndarray:
    """Indices de début de chaque course (les lignes doivent être triées par course)."""
    change = np.r_[True, race_ids[1:] != race_ids[:-1]]
    return np.flatnonzero(change)


def race_softmax(scores: np.ndarray, starts: np.ndarray) -> np.ndarray:
    mx = np.maximum.reduceat(scores, starts)
    sizes = np.diff(np.r_[starts, len(scores)])
    e = np.exp(scores - np.repeat(mx, sizes))
    return e / np.repeat(np.add.reduceat(e, starts), sizes)


class ConditionalLogit:
    def __init__(self, features: list[str], l2: float = 1e-3, name: str = "model"):
        self.features = features
        self.l2 = l2
        self.name = name
        self.w = np.zeros(len(features))
        self.mu = np.zeros(len(features))
        self.sd = np.ones(len(features))

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        return (df[self.features].to_numpy(float) - self.mu) / self.sd

    def fit(self, df: pd.DataFrame, max_iter: int = 50, verbose: bool = False) -> "ConditionalLogit":
        """df trié par race_id ; colonne 'win' (1 gagnant par course)."""
        self.mu = df[self.features].mean().to_numpy(float)
        self.sd = df[self.features].std().replace(0, 1).fillna(1).to_numpy(float)
        X = self._X(df)
        y = df["win"].to_numpy(float)
        starts = _segments(df["race_id"].to_numpy())
        sizes = np.diff(np.r_[starts, len(df)])
        y = y / np.repeat(np.add.reduceat(y, starts), sizes)   # ex-aequo
        n_races = len(starts)
        d = X.shape[1]
        w = np.zeros(d)
        prev = np.inf
        for it in range(max_iter):
            p = race_softmax(X @ w, starts)
            loss = -np.sum(y * np.log(p + 1e-15)) / n_races + 0.5 * self.l2 * w @ w
            grad = X.T @ (p - y) / n_races + self.l2 * w
            Xp = X * p[:, None]
            m = np.add.reduceat(Xp, starts)
            H = (X.T @ Xp - m.T @ m) / n_races + self.l2 * np.eye(d)
            step = np.linalg.solve(H, grad)
            t = 1.0                                   # recherche linéaire simple
            while t > 1e-4:
                w_new = w - t * step
                p2 = race_softmax(X @ w_new, starts)
                l2 = -np.sum(y * np.log(p2 + 1e-15)) / n_races + 0.5 * self.l2 * w_new @ w_new
                if l2 <= loss:
                    break
                t /= 2
            w = w_new
            if verbose:
                print(f"iter {it}: logloss={l2:.5f}")
            if abs(prev - l2) < 1e-8:
                break
            prev = l2
        self.w = w
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """df trié par race_id."""
        starts = _segments(df["race_id"].to_numpy())
        return race_softmax(self._X(df) @ self.w, starts)

    def coefficients(self) -> pd.Series:
        return pd.Series(self.w, index=self.features).sort_values(key=np.abs, ascending=False)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": self.name, "features": self.features, "l2": self.l2,
                                    "w": self.w.tolist(), "mu": self.mu.tolist(),
                                    "sd": self.sd.tolist()}, indent=1))

    @classmethod
    def load(cls, path: Path) -> "ConditionalLogit":
        d = json.loads(path.read_text())
        m = cls(d["features"], d["l2"], d["name"])
        m.w, m.mu, m.sd = (np.array(d[k]) for k in ("w", "mu", "sd"))
        return m
