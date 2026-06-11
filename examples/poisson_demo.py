"""Poisson model demo: example predictions + scoreline heatmap.

    python -m examples.poisson_demo     # writes reports/figures/poisson_demo.png

Three example matchups (favorite vs underdog, even match, defensive pair),
each with expected goals, W/D/L probabilities, and top scorelines; the figure
shows the full scoreline matrix for the marquee match with marginals.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.poisson_model import (  # noqa: E402
    PoissonGoalModel, outcome_probs, top_scorelines,
)
from tests.test_poisson import make_features  # noqa: E402  (reuse the generator)

FIG_DIR = PROJECT_ROOT / "reports" / "figures"

EXAMPLES = [
    ("Strong favorite at home", dict(elo_pre_diff=280, home_avg_goals_scored=2.1,
     away_avg_goals_conceded=1.6, away_avg_goals_scored=0.9,
     home_avg_goals_conceded=0.7, form_pts_last5_diff=1.2,
     home_continent_advantage=1, away_continent_advantage=0, neutral=0)),
    ("Even match, neutral venue", dict(elo_pre_diff=10, home_avg_goals_scored=1.5,
     away_avg_goals_conceded=1.1, away_avg_goals_scored=1.4,
     home_avg_goals_conceded=1.0, form_pts_last5_diff=0.1,
     home_continent_advantage=0, away_continent_advantage=0, neutral=1)),
    ("Two defensive sides", dict(elo_pre_diff=-40, home_avg_goals_scored=0.9,
     away_avg_goals_conceded=0.7, away_avg_goals_scored=1.0,
     home_avg_goals_conceded=0.8, form_pts_last5_diff=-0.3,
     home_continent_advantage=0, away_continent_advantage=0, neutral=1)),
]


def main() -> Path:
    model = PoissonGoalModel().fit(make_features(n=2000))
    print(f"fitted Dixon-Coles rho = {model.rho:+.4f}\n")

    matrices = {}
    for name, feats in EXAMPLES:
        row = pd.DataFrame([feats])
        lh, la = model.predict_lambdas(row)
        M = model.score_matrix(lh[0], la[0])
        matrices[name] = (M, lh[0], la[0])
        p = outcome_probs(M)
        print(f"--- {name} ---")
        print(f"  expected goals: {lh[0]:.2f} vs {la[0]:.2f}")
        print(f"  P(home win) {p['home_win']:.1%} | P(draw) {p['draw']:.1%} "
              f"| P(away win) {p['away_win']:.1%}")
        print(f"  top scorelines: {top_scorelines(M, 4)}\n")

    # ---- figure: heatmap + marginals for the marquee match ----
    name = EXAMPLES[0][0]
    M, lh, la = matrices[name]
    show = 6  # display 0..5 goals
    sub = M[:show, :show]

    fig = plt.figure(figsize=(8.5, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1.1), height_ratios=(1.1, 4),
                          hspace=0.06, wspace=0.06)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

    im = ax.imshow(sub, origin="lower", cmap="viridis")
    for i in range(show):
        for j in range(show):
            ax.text(j, i, f"{sub[i, j]:.1%}", ha="center", va="center",
                    fontsize=8, color="white" if sub[i, j] < sub.max() * 0.6 else "black")
    best = np.unravel_index(np.argmax(sub), sub.shape)
    ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1,
                               fill=False, edgecolor="red", lw=2))
    ax.set_xlabel("Away goals")
    ax.set_ylabel("Home goals")
    ax.set_xticks(range(show)); ax.set_yticks(range(show))

    ax_top.bar(range(show), sub.sum(axis=0), color="tab:blue", alpha=0.7)
    ax_top.set_ylabel("P(away)")
    ax_top.tick_params(labelbottom=False)
    ax_right.barh(range(show), sub.sum(axis=1), color="tab:green", alpha=0.7)
    ax_right.set_xlabel("P(home)")
    ax_right.tick_params(labelleft=False)

    p = outcome_probs(M)
    fig.suptitle(
        f"{name}:  λ_home={lh:.2f}, λ_away={la:.2f}, ρ={model.rho:+.3f}\n"
        f"home {p['home_win']:.0%} / draw {p['draw']:.0%} / away {p['away_win']:.0%}"
        f"  —  modal score {best[0]}-{best[1]} (red box)",
        fontsize=10)
    fig.colorbar(im, ax=ax_right, shrink=0.8, label="P(scoreline)")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "poisson_demo.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"figure written: {out}")
    return out


if __name__ == "__main__":
    main()
