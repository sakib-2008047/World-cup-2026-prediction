"""Poisson scoreline model.

Two Poisson GLMs (log link) estimate expected goals for each side from
pre-match features; an independence product gives the scoreline matrix;
a Dixon-Coles correction patches the known low-score dependence.

Why this model exists alongside the ML classifier:
  - it predicts SCORELINES, which the knockout simulator needs (extra time
    happens at 'level after 90', not at 'draw-ish vibes');
  - it is interpretable: exp(coef) is a multiplicative effect on goal rate;
  - it is the classical baseline every football model answers to.

Usage:
    model = PoissonGoalModel().fit(train_features)
    lh, la = model.predict_lambdas(match_row)
    M = model.score_matrix(lh, la)            # P(i-j) grid
    outcome_probs(M)                          # {home_win, draw, away_win}
    top_scorelines(M, 5)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

MAX_GOALS = 10  # truncation; P(>10 goals) is ~0 at international rates

#: features for the HOME goal rate; AWAY swaps the perspective
HOME_RATE_FEATURES = [
    "elo_pre_diff",
    "home_avg_goals_scored", "away_avg_goals_conceded",
    "form_pts_last5_diff",
    "home_continent_advantage",
    "neutral",
]
AWAY_RATE_FEATURES = [
    "elo_pre_diff",            # sign carries the information
    "away_avg_goals_scored", "home_avg_goals_conceded",
    "form_pts_last5_diff",
    "away_continent_advantage",
    "neutral",
]


def _make_glm(alpha: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("glm", PoissonRegressor(alpha=alpha, max_iter=1000)),
    ])


@dataclass
class PoissonGoalModel:
    """Two GLMs + a Dixon-Coles rho. fit() then predict_lambdas()/score_matrix()."""

    alpha: float = 1e-3                  # L2 strength for the GLMs
    fit_rho: bool = True                 # estimate Dixon-Coles correction?
    rho: float = 0.0
    home_glm: Pipeline = field(default=None, repr=False)
    away_glm: Pipeline = field(default=None, repr=False)

    # ----------------------------------------------------------- fitting ----
    def fit(self, features: pd.DataFrame) -> "PoissonGoalModel":
        self.home_glm = _make_glm(self.alpha)
        self.away_glm = _make_glm(self.alpha)
        self.home_glm.fit(features[HOME_RATE_FEATURES].astype(float),
                          features["home_score"].astype(int))
        self.away_glm.fit(features[AWAY_RATE_FEATURES].astype(float),
                          features["away_score"].astype(int))
        if self.fit_rho:
            self.rho = self._estimate_rho(features)
        return self

    def _estimate_rho(self, features: pd.DataFrame) -> float:
        """Maximum likelihood for the single Dixon-Coles parameter, holding
        the GLMs fixed. Only the four low-score cells depend on rho, so the
        likelihood is cheap. Bounded to keep all tau factors positive."""
        lh, la = self.predict_lambdas(features)
        hs = features["home_score"].to_numpy(dtype=int)
        as_ = features["away_score"].to_numpy(dtype=int)

        def neg_ll(rho: float) -> float:
            tau = dixon_coles_tau(hs, as_, lh, la, rho)
            ll = (np.log(tau)
                  + poisson.logpmf(hs, lh) + poisson.logpmf(as_, la))
            return -float(ll.sum())

        res = minimize_scalar(neg_ll, bounds=(-0.2, 0.2), method="bounded")
        return float(res.x)

    # -------------------------------------------------------- prediction ----
    def predict_lambdas(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        f = features if isinstance(features, pd.DataFrame) else pd.DataFrame([features])
        lh = self.home_glm.predict(f[HOME_RATE_FEATURES].astype(float))
        la = self.away_glm.predict(f[AWAY_RATE_FEATURES].astype(float))
        return lh, la

    def score_matrix(self, lam_home: float, lam_away: float,
                     max_goals: int = MAX_GOALS) -> np.ndarray:
        """P(home=i, away=j) grid, Dixon-Coles corrected, renormalized.
        Rows = home goals 0..max_goals, columns = away goals."""
        i = np.arange(max_goals + 1)
        ph = poisson.pmf(i, lam_home)
        pa = poisson.pmf(i, lam_away)
        M = np.outer(ph, pa)
        if self.rho:
            for hi, ai in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                M[hi, ai] *= dixon_coles_tau(
                    np.array([hi]), np.array([ai]),
                    np.array([lam_home]), np.array([lam_away]), self.rho)[0]
        return M / M.sum()

    def coefficients(self) -> pd.DataFrame:
        """exp(coef) per standardized feature = multiplicative effect on the
        goal rate per 1-sigma feature increase. The interpretability payoff."""
        rows = []
        for side, glm, feats in [("home", self.home_glm, HOME_RATE_FEATURES),
                                 ("away", self.away_glm, AWAY_RATE_FEATURES)]:
            for f, c in zip(feats, glm.named_steps["glm"].coef_):
                rows.append({"side": side, "feature": f,
                             "coef": c, "rate_multiplier_per_sigma": np.exp(c)})
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Dixon-Coles correction
# ----------------------------------------------------------------------------
def dixon_coles_tau(hs: np.ndarray, as_: np.ndarray,
                    lh: np.ndarray, la: np.ndarray, rho: float) -> np.ndarray:
    """Correction factor for the four low-score cells; 1 elsewhere.
    Negative rho inflates 0-0 and 1-1 (more draws than independence predicts)
    and deflates 1-0 / 0-1 - the empirical pattern in real football."""
    tau = np.ones_like(lh, dtype=float)
    m00 = (hs == 0) & (as_ == 0)
    m10 = (hs == 1) & (as_ == 0)
    m01 = (hs == 0) & (as_ == 1)
    m11 = (hs == 1) & (as_ == 1)
    tau[m00] = 1 - lh[m00] * la[m00] * rho
    tau[m10] = 1 + la[m10] * rho
    tau[m01] = 1 + lh[m01] * rho
    tau[m11] = 1 - rho
    return np.clip(tau, 1e-10, None)


# ----------------------------------------------------------------------------
# Matrix consumers
# ----------------------------------------------------------------------------
def outcome_probs(M: np.ndarray) -> dict[str, float]:
    return {"home_win": float(np.tril(M, -1).sum()),
            "draw": float(np.trace(M)),
            "away_win": float(np.triu(M, 1).sum())}


def top_scorelines(M: np.ndarray, n: int = 5) -> list[tuple[str, float]]:
    flat = [((i, j), M[i, j]) for i in range(M.shape[0]) for j in range(M.shape[1])]
    flat.sort(key=lambda t: -t[1])
    return [(f"{i}-{j}", round(p, 4)) for (i, j), p in flat[:n]]


def expected_points(M: np.ndarray) -> tuple[float, float]:
    """Expected group-stage points (home, away) - useful for group sims."""
    p = outcome_probs(M)
    return (3 * p["home_win"] + p["draw"], 3 * p["away_win"] + p["draw"])
