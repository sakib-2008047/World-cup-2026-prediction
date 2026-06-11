"""Match outcome models: home_win / draw / away_win.

Usage:
    from src.models.train import run_experiment
    results = run_experiment(features_df)

Three models, one harness:
    logistic   - calibrated, interpretable linear baseline (the bar to beat)
    rf         - random forest: nonlinear interactions, no tuning fragility
    xgboost    - gradient boosting: usually the winner, needs tuning + care

Anti-leakage rules enforced here:
    1. TIME-ORDERED split: train on the past, test on the future. Never shuffle.
    2. TimeSeriesSplit for CV and tuning - every validation fold is later
       than its training folds.
    3. The test window (default: from 2018-06-01, i.e. the 2018 + 2022 World
       Cups and everything between) is touched exactly once, after tuning.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # keep the rest of the harness usable without xgboost
    HAS_XGB = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
CLASSES = ["away_win", "draw", "home_win"]  # fixed order; code i = CLASSES[i]
CLASS_TO_CODE = {c: i for i, c in enumerate(CLASSES)}


def encode_target(y: pd.Series) -> pd.Series:
    """String labels -> integer codes, in fixed CLASSES order. Centralized
    because XGBoost requires numeric labels and every consumer must agree on
    the same code <-> class mapping (predict_proba column order follows it)."""
    codes = y.map(CLASS_TO_CODE)
    assert codes.notna().all(), f"unknown labels: {set(y) - set(CLASSES)}"
    return codes.astype(int)

#: model inputs - every one of these is a PRE-match quantity by construction
FEATURE_COLS = [
    "home_elo_pre", "away_elo_pre", "elo_pre_diff",
    "fifa_rank_diff",
    "form_pts_last5_diff", "form_pts_last10_diff",
    "win_pct_last20_diff",
    "goal_diff_per_game_diff",
    "home_avg_goals_scored", "home_avg_goals_conceded",
    "away_avg_goals_scored", "away_avg_goals_conceded",
    "opp_strength_last10_diff",
    "home_wc_matches_played", "away_wc_matches_played",
    "home_continent_advantage", "away_continent_advantage",
    "neutral",
]


# ----------------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------------
def time_split(df: pd.DataFrame, test_from: str = "2018-06-01",
               train_from: str = "1990-01-01"):
    """Train strictly before `test_from`; test from `test_from` onward.
    `train_from` trims the very deep past: 1950s friendlies tell us little
    about modern international football and dilute the signal."""
    df = df.sort_values("date")
    train = df[(df["date"] >= train_from) & (df["date"] < test_from)]
    test = df[df["date"] >= test_from]
    assert train["date"].max() < test["date"].min(), "temporal overlap!"
    return train, test


def xy(df: pd.DataFrame, features: list[str] | None = None):
    features = features or [c for c in FEATURE_COLS if c in df.columns]
    X = df[features].astype(float)
    y = encode_target(df["target"])
    return X, y, features


# ----------------------------------------------------------------------------
# Models + search spaces
# ----------------------------------------------------------------------------
def make_models(seed: int = 42) -> dict[str, tuple[Pipeline, dict]]:
    """Each entry: (pipeline, hyperparameter distributions for random search).

    Logistic regression needs imputation (no native NaN handling) and scaling
    (regularization is scale-sensitive). Trees need neither; imputation is
    still applied for RF (sklearn RFs reject NaN), while XGBoost consumes
    NaN natively - its 'missing' routing is itself learned, a real advantage
    with our NaN-by-design features (no FIFA rank pre-1993, no form for
    debutant nations).
    """
    models: dict[str, tuple[Pipeline, dict]] = {}

    models["logistic"] = (
        Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
        ]),
        {"clf__C": loguniform(1e-3, 1e2)},
    )

    models["rf"] = (
        Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=400, n_jobs=-1, random_state=seed)),
        ]),
        {
            "clf__max_depth": randint(4, 16),
            "clf__min_samples_leaf": randint(5, 60),
            "clf__max_features": uniform(0.3, 0.6),
        },
    )

    if HAS_XGB:
        models["xgboost"] = (
            Pipeline([
                ("clf", XGBClassifier(
                    objective="multi:softprob", num_class=3,
                    tree_method="hist",
                    random_state=seed, n_jobs=-1,
                    eval_metric="mlogloss")),
            ]),
            {
                "clf__n_estimators": randint(100, 600),
                "clf__learning_rate": loguniform(0.01, 0.3),
                "clf__max_depth": randint(2, 8),
                "clf__min_child_weight": randint(1, 20),
                "clf__subsample": uniform(0.6, 0.4),
                "clf__colsample_bytree": uniform(0.6, 0.4),
                "clf__reg_lambda": loguniform(0.1, 50),
            },
        )
    return models


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def multiclass_brier(y_true: pd.Series, proba: np.ndarray) -> float:
    """Mean squared distance between predicted probability vector and the
    one-hot outcome (y_true as integer codes). Like log loss it rewards
    calibration, but it's bounded ([0, 2] for 3 classes) and far less
    hysterical about confident mistakes. np.eye indexing is class-complete
    even when a class is absent from the sample (unlike get_dummies)."""
    onehot = np.eye(len(CLASSES))[np.asarray(y_true, dtype=int)]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


@dataclass
class EvalResult:
    name: str
    log_loss: float
    brier: float
    accuracy: float
    best_params: dict

    def row(self) -> dict:
        return {"model": self.name, "log_loss": round(self.log_loss, 4),
                "brier": round(self.brier, 4), "accuracy": round(self.accuracy, 4)}


def evaluate(name: str, model, X_test, y_test, best_params=None) -> EvalResult:
    proba = model.predict_proba(X_test)
    return EvalResult(
        name=name,
        log_loss=float(log_loss(y_test, proba, labels=range(len(CLASSES)))),
        brier=multiclass_brier(y_test, proba),
        accuracy=float(accuracy_score(y_test, model.predict(X_test))),
        best_params=best_params or {},
    )


def baseline_results(y_train: pd.Series, y_test: pd.Series) -> list[EvalResult]:
    """Two baselines every result table needs:
    - class priors: predicts the training distribution for every match
    - home-win always: the naive pundit
    A tuned model that doesn't clearly beat these is decoration."""
    prior = (y_train.value_counts(normalize=True)
             .reindex(range(len(CLASSES)), fill_value=0.0).to_numpy())
    proba_prior = np.tile(prior, (len(y_test), 1))
    out = [EvalResult(
        "baseline_priors",
        float(log_loss(y_test, proba_prior, labels=range(len(CLASSES)))),
        multiclass_brier(y_test, proba_prior),
        float((y_test == int(np.argmax(prior))).mean()),
        {},
    )]
    return out


# ----------------------------------------------------------------------------
# Experiment driver
# ----------------------------------------------------------------------------
def run_experiment(features_df: pd.DataFrame,
                   test_from: str = "2018-06-01",
                   n_iter: int = 25,
                   n_cv_splits: int = 4,
                   seed: int = 42) -> dict:
    """Tune each model with time-series CV on the training window, then
    evaluate ONCE on the held-out future. Returns models + metrics + importances."""
    train, test = time_split(features_df, test_from=test_from)
    X_train, y_train, feats = xy(train)
    X_test, y_test, _ = xy(test, feats)
    print(f"train: {len(train):,} matches  (to {train['date'].max().date()})")
    print(f"test:  {len(test):,} matches  (from {test['date'].min().date()})")

    cv = TimeSeriesSplit(n_splits=n_cv_splits)
    results: list[EvalResult] = baseline_results(y_train, y_test)
    fitted, importances = {}, {}

    for name, (pipe, space) in make_models(seed).items():
        search = RandomizedSearchCV(
            pipe, space, n_iter=n_iter, cv=cv,
            scoring="neg_log_loss", n_jobs=-1, random_state=seed, refit=True)
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        results.append(evaluate(name, search.best_estimator_, X_test, y_test,
                                search.best_params_))
        importances[name] = feature_importance(name, search.best_estimator_, feats)

    table = pd.DataFrame([r.row() for r in results]).sort_values("log_loss")
    return {"results": table, "models": fitted, "importances": importances,
            "features": feats, "split": (train, test)}


def feature_importance(name: str, model: Pipeline,
                       features: list[str]) -> pd.Series:
    """Comparable importance per model family.
    logistic: mean |coefficient| across classes (on STANDARDIZED inputs,
              so magnitudes are comparable across features).
    trees:    impurity-based importances. Caveat noted in docs: impurity
              importance inflates high-cardinality features; for the report,
              prefer permutation importance on the test set."""
    clf = model.named_steps["clf"]
    if name == "logistic":
        vals = np.abs(clf.coef_).mean(axis=0)
    else:
        vals = clf.feature_importances_
    return pd.Series(vals, index=features).sort_values(ascending=False)


def save_experiment(exp: dict, out_dir: Path = MODELS_DIR) -> None:
    import joblib
    out_dir.mkdir(parents=True, exist_ok=True)
    best_name = exp["results"].iloc[0]["model"]
    if best_name in exp["models"]:
        joblib.dump({"model": exp["models"][best_name],
                     "features": exp["features"]},
                    out_dir / "best_model.joblib")
    (out_dir / "metrics.json").write_text(
        json.dumps(exp["results"].to_dict(orient="records"), indent=2))
    print(f"saved best model ({best_name}) + metrics to {out_dir}")
