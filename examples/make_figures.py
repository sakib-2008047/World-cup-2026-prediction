"""Generate all project figures.

    python -m examples.make_figures

Inputs: simulation summary (reruns a small batch if missing), demo Elo
ratings, and a quick model fit for importances. Outputs to reports/figures/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from examples.simulate_2026 import DEMO_ELO  # noqa: E402
from src.data.normalize import load_wc2026_teams  # noqa: E402
from src.features.build_features import build_features  # noqa: E402
from src.models.train import run_experiment  # noqa: E402
from src.simulation.monte_carlo import EloScoreSampler, TournamentSimulator  # noqa: E402
from src.visualization.plots import (  # noqa: E402
    plot_elo_rankings, plot_feature_importance, plot_stage_heatmap,
    plot_top_contenders, plot_win_probability,
)

SUMMARY_CSV = PROJECT_ROOT / "reports" / "simulation_summary.csv"
N_SIMS = 10_000


def get_summary(teams: pd.DataFrame) -> pd.DataFrame:
    if SUMMARY_CSV.exists():
        return pd.read_csv(SUMMARY_CSV, index_col=0)
    groups = {g: sub["team"].tolist() for g, sub in teams.groupby("group")}
    return TournamentSimulator(DEMO_ELO, groups).run(n_sims=N_SIMS, seed=2026)


def quick_importances() -> dict[str, pd.Series]:
    """Small synthetic-world model fit, just to drive the chart."""
    rng = np.random.default_rng(11)
    names = [f"T{i:02d}" for i in range(24)]
    strength = dict(zip(names, rng.normal(1.3, 0.45, 24).clip(0.3)))
    rows, date = [], pd.Timestamp("2012-01-01")
    for i in range(3000):
        h, a = rng.choice(names, 2, replace=False)
        date += pd.Timedelta(days=2)
        t = rng.choice(["Friendly", "FIFA World Cup qualification"])
        hs, as_ = rng.poisson(strength[h] + 0.3), rng.poisson(strength[a])
        rows.append(dict(date=date, home_team=h, away_team=a, home_score=hs,
                         away_score=as_, tournament=t, country=h, neutral=False))
    X = build_features(pd.DataFrame(rows))
    exp = run_experiment(X, test_from="2019-01-01", n_iter=8)
    return {k: exp["importances"][k] for k in ("xgboost", "logistic")
            if k in exp["importances"]}


def example_matchups(summary: pd.DataFrame) -> list[dict]:
    """W/D/L for a few marquee group fixtures via the Elo sampler + Poisson."""
    from scipy.stats import poisson
    sampler = EloScoreSampler(DEMO_ELO)
    pairs = [("Mexico", "South Africa", False), ("Brazil", "Morocco", True),
             ("Spain", "Uruguay", True), ("United States", "Paraguay", False),
             ("England", "Croatia", True), ("Argentina", "Algeria", True)]
    out = []
    for h, a, neutral in pairs:
        lh, la = sampler.lambdas(h, a, neutral=neutral)
        i = np.arange(11)
        M = np.outer(poisson.pmf(i, lh), poisson.pmf(i, la))
        M /= M.sum()
        out.append(dict(home=h, away=a,
                        p_home=float(np.tril(M, -1).sum()),
                        p_draw=float(np.trace(M)),
                        p_away=float(np.triu(M, 1).sum())))
    return out


def main() -> None:
    teams = load_wc2026_teams()
    confeds = dict(zip(teams["team"], teams["confederation"]))
    summary = get_summary(teams)
    summary["confederation"] = [confeds.get(t, "unknown") for t in summary.index]

    paths = [
        plot_elo_rankings(DEMO_ELO, confeds, asof="June 2026 (demo ratings)"),
        plot_feature_importance(quick_importances()),
        plot_win_probability(example_matchups(summary)),
        plot_stage_heatmap(summary.drop(columns="confederation"), N_SIMS),
        plot_top_contenders(summary, N_SIMS),
    ]
    for p in paths:
        print("wrote", p.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
