"""Tests for the OO tournament model. Two themes:
1. the OO model obeys the same conservation invariants as the fast engine
   and agrees with it statistically;
2. result LOCKING works - the new capability this layer exists for."""

import numpy as np
import pytest

from src.data.normalize import load_wc2026_teams
from src.simulation.monte_carlo import EloScoreSampler, TournamentSimulator
from src.simulation.tournament import Match, Team, Tournament, monte_carlo


@pytest.fixture(scope="module")
def field_():
    teams_df = load_wc2026_teams()
    rng = np.random.default_rng(9)
    ratings = {row.team: 1500 + 200 * rng.standard_normal()
               for row in teams_df.itertuples()}
    ratings["Spain"] = 2150  # one clear favorite for signal tests
    teams = {row.team: Team(row.team, row.group, ratings[row.team],
                            row.confederation)
             for row in teams_df.itertuples()}
    return teams, ratings


def sampler_for(ratings):
    return EloScoreSampler(ratings).lambdas


# ----------------------------------------------------------------- match ----
def test_match_regulation_winner():
    a, b = Team("A", "A", 1500), Team("B", "A", 1500)
    m = Match(a, b, stage="group").lock(2, 1)
    assert m.winner is a and m.decided_by == "locked"


def test_drawn_knockout_lock_requires_winner():
    a, b = Team("A", "A", 1500), Team("B", "B", 1500)
    with pytest.raises(AssertionError):
        Match(a, b, stage="R32", knockout=True).lock(1, 1)
    m = Match(a, b, stage="R32", knockout=True).lock(1, 1, winner_name="B")
    assert m.winner is b


def test_knockout_match_always_has_winner(field_):
    teams, ratings = field_
    rng = np.random.default_rng(0)
    s = sampler_for(ratings)
    for _ in range(200):
        m = Match(teams["Haiti"], teams["Spain"], stage="R32", knockout=True)
        m.play(s, rng)
        assert m.winner is not None
        assert m.decided_by in ("regulation", "extra_time", "penalties")


def test_locked_match_never_rerolled(field_):
    teams, ratings = field_
    m = Match(teams["Mexico"], teams["South Africa"], stage="group").lock(2, 1)
    m.play(sampler_for(ratings), np.random.default_rng(0))
    assert (m.home_goals, m.away_goals) == (2, 1)


# ------------------------------------------------------------- tournament ---
def test_outcome_invariants(field_):
    teams, ratings = field_
    t = Tournament.build_2026(teams)
    out = t.play(sampler_for(ratings), np.random.default_rng(4))
    assert out.champion in out.finalists
    assert set(out.finalists) <= set(out.semifinalists)
    assert len(out.finalists) == 2 and len(out.semifinalists) == 4
    assert len(out.matches) == 72 + 31           # groups + bracket (no 3rd-place match)
    from collections import Counter
    c = Counter(out.stage_reached.values())
    assert c["champion"] == 1 and c["group"] == 16 and c["R32"] == 16


def test_group_lock_respected(field_):
    teams, ratings = field_
    t = Tournament.build_2026(teams)
    t.lock_result("Mexico", "South Africa", 2, 1)
    out = t.play(sampler_for(ratings), np.random.default_rng(1))
    m = next(m for m in out.matches
             if {m.home.name, m.away.name} == {"Mexico", "South Africa"})
    assert (m.home_goals, m.away_goals) == (2, 1)
    assert m.decided_by == "locked"


def test_locking_a_loss_shifts_probabilities(field_):
    """Conditioning sanity: forcing Spain to lose all three group matches
    must crater their title probability vs the unconditioned run."""
    teams, ratings = field_
    s = sampler_for(ratings)
    base = monte_carlo(teams, s, n_sims=150, seed=7)
    locked = [("Spain", "Uruguay", 0, 2), ("Spain", "Saudi Arabia", 0, 2),
              ("Spain", "Cape Verde", 0, 2)]
    cond = monte_carlo(teams, s, n_sims=150, seed=7, locked=locked)
    assert cond["Spain"] < base["Spain"]
    assert cond["Spain"] <= 5  # group exit ~guaranteed; tiny residual via thirds? no: 0 pts


def test_oo_and_fast_engine_agree(field_):
    """Same ratings, same rules -> champion distributions should correlate.
    Loose statistical check, not exact equality (different RNG consumption)."""
    teams, ratings = field_
    groups = {}
    for t in teams.values():
        groups.setdefault(t.group, []).append(t.name)
    fast = TournamentSimulator(ratings, groups).run(n_sims=300, seed=2)
    slow = monte_carlo(teams, sampler_for(ratings), n_sims=300, seed=2)
    fast_top = set(fast.head(5).index)
    slow_top = {t for t, _ in slow.most_common(5)}
    assert len(fast_top & slow_top) >= 3
    assert fast.index[0] == "Spain" and slow.most_common(1)[0][0] == "Spain"
