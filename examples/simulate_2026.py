"""Run the full 2026 Monte Carlo: 10,000 tournaments, real groups.

    python -m examples.simulate_2026

RATINGS NOTE: the dict below contains APPROXIMATE Elo ratings (eloratings.net
scale, ~mid-2026) so the engine can be demonstrated standalone. They are demo
inputs, not pipeline outputs. For publishable predictions, replace with the
ratings produced by your own Elo engine over the real match data:

    rated = add_elo(matches_clean)
    ratings = rated.attrs["elo_engine"].ratings
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.normalize import load_wc2026_teams  # noqa: E402
from src.simulation.monte_carlo import TournamentSimulator  # noqa: E402

#: APPROXIMATE Elo (demo only - see module docstring)
DEMO_ELO = {
    "Spain": 2170, "Argentina": 2120, "France": 2060, "England": 2010,
    "Brazil": 2000, "Portugal": 2010, "Netherlands": 1980, "Germany": 1950,
    "Colombia": 1940, "Uruguay": 1920, "Croatia": 1880, "Belgium": 1890,
    "Morocco": 1900, "Japan": 1850, "Ecuador": 1880, "Switzerland": 1830,
    "Mexico": 1830, "United States": 1790, "Turkey": 1840, "Senegal": 1820,
    "Iran": 1800, "Austria": 1810, "South Korea": 1780, "Australia": 1760,
    "Norway": 1820, "Sweden": 1740, "Egypt": 1750, "Algeria": 1760,
    "Paraguay": 1770, "Tunisia": 1720, "Ivory Coast": 1740, "Canada": 1760,
    "Scotland": 1720, "Czech Republic": 1700, "Panama": 1700, "Ghana": 1680,
    "Qatar": 1640, "Saudi Arabia": 1660, "South Africa": 1650, "Jordan": 1640,
    "Uzbekistan": 1640, "Iraq": 1630, "Cape Verde": 1600, "Haiti": 1560,
    "Bosnia and Herzegovina": 1700, "New Zealand": 1590, "Curaçao": 1560,
    "DR Congo": 1640,
}


def main(n_sims: int = 10_000, seed: int = 2026) -> pd.DataFrame:
    teams = load_wc2026_teams()
    groups = {g: sub["team"].tolist() for g, sub in teams.groupby("group")}

    sim = TournamentSimulator(DEMO_ELO, groups)
    t0 = time.perf_counter()
    summary = sim.run(n_sims=n_sims, seed=seed)
    dt = time.perf_counter() - t0

    print(f"{n_sims:,} tournaments in {dt:.1f}s "
          f"({n_sims / dt:,.0f} tournaments/sec)\n")
    out = (summary.head(15) * 100).round(1)
    out.columns = [f"P({c}) %" for c in out.columns]
    print(out.to_string())

    dest = PROJECT_ROOT / "reports" / "simulation_summary.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(dest)
    print(f"\nfull table written: {dest}")
    return summary


if __name__ == "__main__":
    main()
