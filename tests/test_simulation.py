"""Simulation tests. The conservation invariants are the key ones: every
simulated tournament must produce exactly one champion, one beaten finalist,
two beaten semifinalists, and so on - any bracket-logic bug breaks them."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from src.simulation.bracket_2026 import GROUPS, THIRD_SLOTS
from src.simulation.monte_carlo import (
    EloScoreSampler,
    TournamentSimulator,
    allocate_thirds,
    rank_group,
)


def demo_groups():
    return {g: [f"{g}{i}" for i in range(1, 5)] for g in GROUPS}


def demo_ratings(spread=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return {t: 1700 + spread * rng.standard_normal()
            for g in demo_groups().values() for t in g}


# -------------------------------------------------------------- standings ---
def test_rank_group_points_first():
    res = [("A", "B", 1, 0), ("A", "C", 1, 0), ("A", "D", 0, 1),
           ("B", "C", 2, 0), ("B", "D", 2, 0), ("C", "D", 1, 0)]
    rows = rank_group(["A", "B", "C", "D"], res, np.random.default_rng(0))
    assert [r["team"] for r in rows][:2] == ["B", "A"]  # B 6pts +3GD? B beats C,D; lost? B lost to A: B has 6 pts; A has 6 pts; A GD: +1+1-1=+1, B GD: -1+2+2=+3 -> B first


def test_rank_group_h2h_breaks_exact_ties():
    """A and B tied on pts/GD/GF, but A beat B head-to-head -> A above B."""
    res = [("A", "B", 1, 0), ("B", "C", 2, 1), ("A", "D", 0, 1),
           ("C", "A", 0, 1), ("D", "B", 0, 2), ("C", "D", 9, 0)]
    # A: beat B 1-0, beat C 1-0, lost D 0-1 -> 6 pts, GF 2, GA 1, GD +1
    # B: lost A 0-1, beat C 2-1, beat D 2-0 -> 6 pts, GF 4, GA 2, GD +2 (not tied)
    rows = rank_group(["A", "B", "C", "D"], res, np.random.default_rng(0))
    teams = [r["team"] for r in rows]
    assert teams.index("B") < teams.index("A")  # GD separates before h2h

    # construct an EXACT tie: identical pts/GD/GF, A beat B
    res2 = [("A", "B", 1, 0), ("B", "C", 1, 0), ("C", "A", 1, 0),
            ("A", "D", 2, 0), ("B", "D", 2, 0), ("C", "D", 0, 2)]
    # A: W B, L C, W D -> 6pts GF 3 GA 1 GD +2
    # B: L A, W C, W D -> 6pts GF 3 GA 1 GD +2  (exact tie, A beat B)
    rows2 = rank_group(["A", "B", "C", "D"], res2, np.random.default_rng(0))
    teams2 = [r["team"] for r in rows2]
    assert teams2.index("A") < teams2.index("B")


# ---------------------------------------------------------- third matching --
def test_allocate_thirds_every_combination_solvable():
    """All C(12,8)=495 scenarios must admit a valid assignment - the same
    guarantee FIFA's Annex C provides."""
    for combo in combinations(GROUPS, 8):
        qualified = {g: f"team_{g}" for g in combo}
        sol = allocate_thirds(qualified)
        assert sol is not None, f"unsolvable scenario {combo}"
        assert len(sol) == 8 and len(set(sol.values())) == 8
        for match_no, team in sol.items():
            g = team.split("_")[1]
            assert g in THIRD_SLOTS[match_no], \
                f"{team} assigned to slot {match_no} outside pool"


# ------------------------------------------------------------ conservation --
@pytest.fixture(scope="module")
def sim():
    return TournamentSimulator(demo_ratings(spread=120, seed=1), demo_groups())


def test_simulate_once_conservation(sim):
    rng = np.random.default_rng(7)
    stage = sim.simulate_once(rng)
    counts = pd.Series(stage).value_counts()
    assert counts["champion"] == 1
    assert counts["F"] == 1          # beaten finalist
    assert counts["SF"] == 2
    assert counts["QF"] == 4
    assert counts["R16"] == 8
    assert counts["R32"] == 16
    assert counts["group"] == 16
    assert counts.sum() == 48


def test_run_probabilities_consistent(sim):
    summary = sim.run(n_sims=300, seed=5)
    assert summary["champion"].sum() == pytest.approx(1.0)
    assert summary["final"].sum() == pytest.approx(2.0)
    assert summary["semifinal"].sum() == pytest.approx(4.0)
    # monotone nesting: champion <= final <= semifinal <= QF <= knockout
    m = summary
    assert (m["champion"] <= m["final"] + 1e-12).all()
    assert (m["final"] <= m["semifinal"] + 1e-12).all()
    assert (m["semifinal"] <= m["quarterfinal"] + 1e-12).all()


def test_seed_reproducibility(sim):
    a = sim.run(n_sims=100, seed=11)
    b = sim.run(n_sims=100, seed=11)
    pd.testing.assert_frame_equal(a, b)


def test_stronger_team_wins_more():
    ratings = demo_ratings(spread=0)
    ratings["A1"] = 2200  # one giant in a flat field
    s = TournamentSimulator(ratings, demo_groups())
    summary = s.run(n_sims=400, seed=3)
    assert summary.index[0] == "A1"
    assert summary.loc["A1", "champion"] > 0.25


def test_sampler_host_advantage():
    sampler = EloScoreSampler({"H": 1700, "V": 1700})
    la_n, lb_n = sampler.lambdas("H", "V", neutral=True)
    la_h, lb_h = sampler.lambdas("H", "V", neutral=False)
    assert la_n == pytest.approx(lb_n)
    assert la_h > la_n and lb_h < lb_n
