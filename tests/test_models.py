"""Tests for the model training harness. These don't test model QUALITY
(that's the backtest's job) - they test the harness can't lie to us."""

import numpy as np
import pandas as pd
import pytest

from src.models.train import (
    CLASSES,
    baseline_results,
    multiclass_brier,
    run_experiment,
    time_split,
    xy,
)


@pytest.fixture(scope="module")
def synthetic_features():
    """600 matches over 12 years where elo_diff genuinely drives outcomes -
    so a working harness must find signal, and a leaky one would be caught
    by the time-split assertions."""
    rng = np.random.default_rng(0)
    n = 600
    dates = pd.date_range("2010-01-01", periods=n, freq="7D")
    elo_diff = rng.normal(0, 150, n)
    p_home = 1 / (1 + 10 ** (-(elo_diff + 60) / 400))
    u = rng.random(n)
    target = np.where(u < p_home * 0.75, "home_win",
                      np.where(u < p_home * 0.75 + 0.25, "draw", "away_win"))
    df = pd.DataFrame({
        "date": dates,
        "home_elo_pre": 1500 + elo_diff / 2,
        "away_elo_pre": 1500 - elo_diff / 2,
        "elo_pre_diff": elo_diff,
        "form_pts_last5_diff": rng.normal(0, 1, n),
        "neutral": rng.integers(0, 2, n),
        "target": target,
    })
    # sprinkle NaN like real data has
    df.loc[df.sample(50, random_state=1).index, "form_pts_last5_diff"] = np.nan
    return df


def test_time_split_no_overlap(synthetic_features):
    train, test = time_split(synthetic_features, test_from="2018-01-01")
    assert train["date"].max() < test["date"].min()
    assert len(train) + len(test) <= len(synthetic_features)  # train_from trims


def test_time_split_rejects_reversed_config(synthetic_features):
    with pytest.raises(AssertionError):
        # test_from before train_from -> empty/overlapping windows must fail loudly
        time_split(synthetic_features, test_from="1989-01-01")


def test_xy_selects_only_known_features(synthetic_features):
    X, y, feats = xy(synthetic_features)
    assert "target" not in feats and "date" not in feats
    assert set(feats) <= set(synthetic_features.columns)


def test_brier_bounds_and_perfection():
    y = pd.Series([2, 1])  # codes for home_win, draw
    perfect = np.array([[0, 0, 1], [0, 1, 0]], dtype=float)
    worst = np.array([[1, 0, 0], [0, 0, 1]], dtype=float)
    assert multiclass_brier(y, perfect) == 0.0
    assert multiclass_brier(y, worst) == 2.0


def test_baseline_priors_match_train_distribution(synthetic_features):
    train, test = time_split(synthetic_features, test_from="2018-01-01")
    from src.models.train import encode_target
    res = baseline_results(encode_target(train["target"]),
                           encode_target(test["target"]))[0]
    assert res.name == "baseline_priors"
    assert 0 < res.log_loss < 1.5


@pytest.mark.slow
def test_experiment_end_to_end(synthetic_features):
    exp = run_experiment(synthetic_features, test_from="2018-01-01",
                         n_iter=8, n_cv_splits=3)
    table = exp["results"]
    # every model produces valid probabilities on the test window
    _, test = exp["split"]
    X_test, _, _ = xy(test, exp["features"])
    for name, model in exp["models"].items():
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(test), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)
        assert (proba >= 0).all()
    # with genuine elo signal, every learner must beat the priors baseline
    prior_ll = table.loc[table.model == "baseline_priors", "log_loss"].iloc[0]
    for name in exp["models"]:
        model_ll = table.loc[table.model == name, "log_loss"].iloc[0]
        assert model_ll < prior_ll, f"{name} failed to beat priors"
    # importances exist and elo_pre_diff should matter for tree models
    assert exp["importances"]["xgboost"].index[0] in (
        "elo_pre_diff", "home_elo_pre", "away_elo_pre")
