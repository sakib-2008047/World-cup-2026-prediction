"""Elo demo: example usage + visualization of rating dynamics.

    python -m examples.elo_demo            # writes reports/figures/elo_demo.png

Panel 1 - the logistic expected-score curve (the heart of the system).
Panel 2 - rating trajectories over a simulated 4-year cycle, with the
          biggest single-match swing annotated (always an upset - that IS Elo).
Panel 3 - distribution of per-match rating changes by tournament type,
          showing the K-factor doing its job.
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

from src.features.elo import EloEngine, add_elo  # noqa: E402

FIG_DIR = PROJECT_ROOT / "reports" / "figures"

#: true (hidden) strengths the simulation draws from - Elo must discover these
TRUE_STRENGTH = {
    "Brazil": 2.0, "France": 1.9, "Argentina": 1.9, "Germany": 1.6,
    "Mexico": 1.3, "Japan": 1.3, "Morocco": 1.4, "New Zealand": 0.8,
}


def simulate_history(n_matches: int = 600, seed: int = 42) -> pd.DataFrame:
    """Synthetic but realistic schedule: mostly friendlies/qualifiers with a
    World Cup burst every 4 years. Goals ~ Poisson with strength-driven rates."""
    rng = np.random.default_rng(seed)
    teams = list(TRUE_STRENGTH)
    rows = []
    start = pd.Timestamp("2018-01-01")
    for i in range(n_matches):
        date = start + pd.Timedelta(days=int(i * 2.4))
        wc_window = date.month in (6, 7) and date.year in (2018, 2022)
        tournament = ("FIFA World Cup" if wc_window and rng.random() < 0.7
                      else rng.choice(["Friendly", "FIFA World Cup qualification"],
                                      p=[0.45, 0.55]))
        h, a = rng.choice(teams, size=2, replace=False)
        neutral = tournament == "FIFA World Cup"
        adv = 0.0 if neutral else 0.25
        hs = rng.poisson(TRUE_STRENGTH[h] + adv)
        as_ = rng.poisson(TRUE_STRENGTH[a])
        rows.append({"date": date, "home_team": h, "away_team": a,
                     "home_score": hs, "away_score": as_,
                     "tournament": tournament, "neutral": neutral})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def trajectories(matches_with_elo: pd.DataFrame) -> pd.DataFrame:
    """Long frame of (team, date, rating_pre) built from add_elo output."""
    home = matches_with_elo[["date", "home_team", "home_elo_pre"]].rename(
        columns={"home_team": "team", "home_elo_pre": "elo"})
    away = matches_with_elo[["date", "away_team", "away_elo_pre"]].rename(
        columns={"away_team": "team", "away_elo_pre": "elo"})
    return pd.concat([home, away]).sort_values("date")


def main() -> Path:
    matches = simulate_history()
    rated = add_elo(matches)
    engine: EloEngine = rated.attrs["elo_engine"]

    # per-match swing (always symmetric, so home delta suffices)
    post_home = rated["home_elo_pre"].shift(-1)  # not reliable per-team; recompute:
    deltas = []
    replay = EloEngine()
    for m in matches.itertuples(index=False):
        ra = replay.get(m.home_team)
        replay.update(m.home_team, m.away_team, int(m.home_score),
                      int(m.away_score), m.tournament, bool(m.neutral))
        deltas.append(replay.get(m.home_team) - ra)
    rated = rated.assign(home_delta=deltas)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # Panel 1: the logistic curve
    ax = axes[0]
    gap = np.linspace(-800, 800, 400)
    ax.plot(gap, [EloEngine.expected(1500 + g, 1500) for g in gap], lw=2)
    for g, label in [(100, "+100 (home adv)"), (400, "+400 (10:1)")]:
        e = EloEngine.expected(1500 + g, 1500)
        ax.scatter([g], [e], zorder=3)
        ax.annotate(f"{label}\nE={e:.2f}", (g, e), textcoords="offset points",
                    xytext=(8, -18), fontsize=8)
    ax.axhline(0.5, color="grey", lw=0.5, ls="--")
    ax.set_xlabel("Rating difference (R_A − R_B)")
    ax.set_ylabel("Expected score E_A")
    ax.set_title("Expected score curve")

    # Panel 2: trajectories
    ax = axes[1]
    traj = trajectories(rated)
    for team, sub in traj.groupby("team"):
        ax.plot(sub["date"], sub["elo"], lw=1.4,
                label=f"{team} ({engine.get(team):.0f})")
    # annotate the biggest single-match swing
    big = rated.loc[rated["home_delta"].abs().idxmax()]
    winner = big.home_team if big.home_delta > 0 else big.away_team
    loser = big.away_team if big.home_delta > 0 else big.home_team
    ax.annotate(
        f"biggest swing: {winner} beats {loser}\n"
        f"{big.home_score}-{big.away_score}, {big.tournament}, "
        f"Δ={abs(big.home_delta):.0f}",
        xy=(big.date, big.home_elo_pre), fontsize=8,
        textcoords="offset points", xytext=(10, 18),
        arrowprops={"arrowstyle": "->", "lw": 0.8})
    ax.set_title("Rating trajectories (sim. 2018-2022)")
    ax.set_ylabel("Elo")
    ax.legend(fontsize=7, ncol=2, loc="lower left")

    # Panel 3: K-factor in action
    ax = axes[2]
    for t, color in [("Friendly", "tab:grey"),
                     ("FIFA World Cup qualification", "tab:blue"),
                     ("FIFA World Cup", "tab:red")]:
        d = rated.loc[rated.tournament == t, "home_delta"].abs()
        ax.hist(d, bins=24, alpha=0.55, label=f"{t} (n={len(d)})",
                color=color, density=True)
    ax.set_xlabel("|rating change| per match")
    ax.set_title("K-factor: importance scales movement")
    ax.legend(fontsize=7)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "elo_demo.png"
    fig.savefig(out, dpi=150)

    print("Final ratings (true strength in parens):")
    for team, r in sorted(engine.ratings.items(), key=lambda kv: -kv[1]):
        print(f"  {team:<12} {r:7.1f}   (λ={TRUE_STRENGTH[team]})")
    print(f"\nFigure written: {out}")
    return out


if __name__ == "__main__":
    main()
