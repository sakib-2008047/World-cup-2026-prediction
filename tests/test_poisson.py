"""Poisson model tests: the math first, then the learning."""

import numpy as np
import pandas as pd
import pytest

from src.models.poisson_model import (
    HOME_RATE_FEATURES,
    PoissonGoalModel,
    dixon_coles_tau,
    expected_points,
    outcome_probs,
    top_scorelines,
)


def make_features(n=800, seed=3):
    """Synthetic matches where goals truly come from feature-driven Poisson
    rates - so the GLM has a real signal to recover."""
    rng = np.random.default_rng(seed)
    elo_diff = rng.normal(0, 150, n)
    h_att = rng.uniform(0.8, 2.0, n)
    a_def = rng.uniform(0.8, 2.0, n)
    a_att = rng.uniform(0.8, 2.0, n)
    h_def = rng.uniform(0.8, 2.0, n)
    neutral = rng.integers(0, 2, n)
    lam_h = np.exp(np.log(1.2) + 0.0015 * elo_diff + 0.25 * (h_att - 1.4)
                   + 0.20 * (a_def - 1.4) + 0.18 * (1 - neutral))
    lam_a = np.exp(np.log(1.1) - 0.0015 * elo_diff + 0.25 * (a_att - 1.4)
                   + 0.20 * (h_def - 1.4))
    return pd.DataFrame({
        "elo_pre_diff": elo_diff,
        "home_avg_goals_scored": h_att, "away_avg_goals_conceded": a_def,
        "away_avg_goals_scored": a_att, "home_avg_goals_conceded": h_def,
        "form_pts_last5_diff": rng.normal(0, 1, n),
        "home_continent_advantage": rng.integers(0, 2, n),
        "away_continent_advantage": rng.integers(0, 2, n),
        "neutral": neutral,
        "home_score": rng.poisson(lam_h),
        "away_score": rng.poisson(lam_a),
    })


@pytest.fixture(scope="module")
def fitted():
    return PoissonGoalModel().fit(make_features())


# ------------------------------------------------------------- matrix math --
def test_score_matrix_sums_to_one(fitted):
    M = fitted.score_matrix(1.5, 1.1)
    assert M.sum() == pytest.approx(1.0)
    assert (M >= 0).all()


def test_outcome_probs_partition(fitted):
    p = outcome_probs(fitted.score_matrix(1.7, 0.9))
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["home_win"] > p["away_win"]  # higher lambda must favor home


def test_equal_lambdas_symmetric_without_rho():
    m = PoissonGoalModel(fit_rho=False)
    M = m.score_matrix(1.3, 1.3)
    p = outcome_probs(M)
    assert p["home_win"] == pytest.approx(p["away_win"])


def test_top_scorelines_ordering(fitted):
    top = top_scorelines(fitted.score_matrix(2.4, 0.6), n=3)
    probs = [p for _, p in top]
    assert probs == sorted(probs, reverse=True)
    # with lambdas 2.4 vs 0.6 the modal score should be a home win
    h, a = map(int, top[0][0].split("-"))
    assert h > a


def test_expected_points_bounds(fitted):
    eh, ea = expected_points(fitted.score_matrix(1.5, 1.5))
    assert 0 < eh < 3 and 0 < ea < 3


# ------------------------------------------------------------- Dixon-Coles --
def test_tau_identity_when_rho_zero():
    t = dixon_coles_tau(np.array([0, 1, 5]), np.array([0, 1, 2]),
                        np.full(3, 1.5), np.full(3, 1.2), rho=0.0)
    np.testing.assert_allclose(t, 1.0)


def test_negative_rho_inflates_draws():
    base = PoissonGoalModel(fit_rho=False, rho=0.0)
    corrected = PoissonGoalModel(fit_rho=False, rho=-0.08)
    p0 = outcome_probs(base.score_matrix(1.3, 1.1))
    p1 = outcome_probs(corrected.score_matrix(1.3, 1.1))
    assert p1["draw"] > p0["draw"]


def test_tau_only_touches_low_score_cells():
    m0 = PoissonGoalModel(fit_rho=False, rho=0.0).score_matrix(1.4, 1.2)
    m1 = PoissonGoalModel(fit_rho=False, rho=-0.08).score_matrix(1.4, 1.2)
    # ratio differs in the 2x2 corner; elsewhere only via renormalization
    raw_ratio = m1[2:, 2:] / m0[2:, 2:]
    assert np.allclose(raw_ratio, raw_ratio.flat[0])  # uniform rescale only


# ---------------------------------------------------------------- learning --
def test_glm_recovers_elo_effect(fitted):
    """More positive elo_diff must raise home lambda, monotonically."""
    base = make_features(n=3).iloc[[0]].copy()
    lams = []
    for d in (-300, 0, 300):
        row = base.copy()
        row["elo_pre_diff"] = d
        lh, _ = fitted.predict_lambdas(row)
        lams.append(lh[0])
    assert lams[0] < lams[1] < lams[2]


def test_rho_estimated_within_bounds(fitted):
    assert -0.2 <= fitted.rho <= 0.2


def test_coefficients_table_shape(fitted):
    coefs = fitted.coefficients()
    assert len(coefs) == 2 * len(HOME_RATE_FEATURES)
    assert {"side", "feature", "rate_multiplier_per_sigma"} <= set(coefs.columns)
