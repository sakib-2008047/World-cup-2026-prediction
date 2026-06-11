"""Unit tests for the Elo engine's mathematical properties.

Complements tests/test_features.py (which tests Elo in pipeline context);
these tests pin down the math itself, property by property.
"""

import pandas as pd
import pytest

from src.features.elo import DEFAULT_K, EloEngine, add_elo


# ----------------------------------------------------------- expected score
def test_expected_score_symmetry():
    """E_A + E_B = 1 for any rating pair."""
    for ra, rb in [(1500, 1500), (1800, 1500), (1234, 2100)]:
        assert EloEngine.expected(ra, rb) + EloEngine.expected(rb, ra) == pytest.approx(1.0)


def test_expected_score_equal_ratings_is_half():
    assert EloEngine.expected(1500, 1500) == pytest.approx(0.5)


def test_expected_score_400_points_is_10_to_1():
    """The defining property of the 400 scale constant."""
    e = EloEngine.expected(1900, 1500)
    assert e == pytest.approx(10 / 11, abs=1e-9)


def test_expected_score_monotonic_in_rating_gap():
    gaps = [-400, -200, 0, 200, 400]
    probs = [EloEngine.expected(1500 + g, 1500) for g in gaps]
    assert probs == sorted(probs)


# ----------------------------------------------------------- goal multiplier
def test_goal_multiplier_values_and_monotonicity():
    g = EloEngine.goal_multiplier
    assert g(0) == g(1) == 1.0
    assert g(2) == 1.5
    assert g(3) == pytest.approx(1.75)
    assert g(4) == pytest.approx(1.875)
    margins = list(range(0, 10))
    vals = [g(m) for m in margins]
    assert vals == sorted(vals)


def test_goal_multiplier_sublinear():
    """Doubling the margin must NOT double the multiplier (anti rating-farming)."""
    g = EloEngine.goal_multiplier
    assert g(6) < 2 * g(3)


# ----------------------------------------------------------- update dynamics
def _play(eng, home, away, hs, as_, tournament="Friendly", neutral=True):
    m = pd.DataFrame([{
        "date": pd.Timestamp("2024-01-01"), "home_team": home, "away_team": away,
        "home_score": hs, "away_score": as_, "tournament": tournament,
        "neutral": neutral,
    }])
    add_elo(m, eng)


def test_zero_sum():
    eng = EloEngine()
    _play(eng, "A", "B", 3, 1)
    _play(eng, "B", "C", 0, 2, tournament="FIFA World Cup")
    assert sum(eng.ratings.values()) == pytest.approx(3 * eng.start)


def test_draw_moves_points_from_favorite_to_underdog():
    eng = EloEngine()
    eng.ratings = {"Fav": 1800.0, "Dog": 1500.0}
    _play(eng, "Fav", "Dog", 1, 1)
    assert eng.get("Fav") < 1800 and eng.get("Dog") > 1500


def test_k_factor_scales_update():
    def gain(tournament):
        eng = EloEngine()
        _play(eng, "A", "B", 1, 0, tournament=tournament)
        return eng.get("A") - eng.start

    assert gain("FIFA World Cup") == pytest.approx(3 * gain("Friendly"))  # K 60 vs 20
    assert gain("UnknownCup") == pytest.approx(gain("Friendly") * DEFAULT_K / 20)


def test_convergence_to_true_strength_ordering():
    """Round-robin where A always beats B, B always beats C: after enough
    cycles the rating order must reflect the true order."""
    eng = EloEngine()
    for _ in range(30):
        _play(eng, "A", "B", 2, 0)
        _play(eng, "B", "C", 2, 0)
        _play(eng, "A", "C", 2, 0)
    assert eng.get("A") > eng.get("B") > eng.get("C")


def test_home_advantage_only_affects_expectation_not_stored_rating():
    """After a 'fair' home result (home team wins as expected at the home-adv
    rate), stored ratings should drift LESS than after the same neutral win."""
    eng_home, eng_neutral = EloEngine(), EloEngine()
    _play(eng_home, "A", "B", 1, 0, neutral=False)
    _play(eng_neutral, "A", "B", 1, 0, neutral=True)
    assert (eng_home.get("A") - 1500) < (eng_neutral.get("A") - 1500)
    # and no engine ever stores the +100 in the rating itself
    assert eng_home.get("A") + eng_home.get("B") == pytest.approx(3000)
