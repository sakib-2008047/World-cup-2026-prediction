"""Feature engineering tests. The leakage test at the bottom is the one that
matters most - it is the proof that no feature sees its own match's result."""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    add_continent_advantage,
    add_fifa_rank,
    add_rolling_features,
    build_features,
    to_long,
)
from src.features.elo import EloEngine, add_elo


def make_matches(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "home_team": "Brazil", "away_team": "Argentina",
        "home_score": 1, "away_score": 0,
        "tournament": "Friendly", "country": "Brazil", "neutral": False,
    }
    df = pd.DataFrame([{**defaults, **r} for r in rows])
    df["date"] = pd.to_datetime(df.get("date", pd.Series(["2024-01-01"] * len(df))))
    if "date" not in rows[0]:
        df["date"] = pd.date_range("2024-01-01", periods=len(df), freq="7D")
    return df.sort_values("date").reset_index(drop=True)


# ------------------------------------------------------------------- Elo ----
def test_elo_zero_sum():
    eng = EloEngine()
    m = make_matches([{}, {}, {"home_score": 0, "away_score": 2}])
    add_elo(m, eng)
    assert sum(eng.ratings.values()) == pytest.approx(2 * EloEngine().start)


def test_elo_winner_gains():
    m = make_matches([{"home_score": 3, "away_score": 0}])
    out = add_elo(m)
    eng = out.attrs["elo_engine"]
    assert eng.get("Brazil") > 1500 > eng.get("Argentina")
    # pre-match columns must be the pre-update values
    assert out["home_elo_pre"].iloc[0] == 1500 == out["away_elo_pre"].iloc[0]


def test_elo_upset_moves_more_than_expected_result():
    """Beating a stronger opponent must move ratings more than beating a
    weaker one - the defining property of Elo."""
    eng = EloEngine()
    eng.ratings = {"Giant": 2000.0, "Minnow": 1400.0}
    m = make_matches([{"home_team": "Minnow", "away_team": "Giant",
                       "home_score": 1, "away_score": 0, "neutral": True}])
    add_elo(m, eng)
    upset_gain = eng.get("Minnow") - 1400

    eng2 = EloEngine()
    eng2.ratings = {"Giant": 2000.0, "Minnow": 1400.0}
    m2 = make_matches([{"home_team": "Giant", "away_team": "Minnow",
                        "home_score": 1, "away_score": 0, "neutral": True}])
    add_elo(m2, eng2)
    expected_gain = eng2.get("Giant") - 2000
    assert upset_gain > expected_gain > 0


def test_elo_home_advantage_dampens_home_win_gain():
    """A home win is less surprising than a neutral-venue win, so it earns less."""
    def gain(neutral):
        eng = EloEngine()
        m = make_matches([{"neutral": neutral}])
        add_elo(m, eng)
        return eng.get("Brazil") - 1500
    assert gain(neutral=False) < gain(neutral=True)


# --------------------------------------------------------------- rolling ----
def _featurize(rows):
    m = add_elo(make_matches(rows))
    return add_rolling_features(to_long(m))


def test_form_window_and_shift():
    # Brazil: W W L (3,3,0 points). Before match 3, last-5 form = (3+3)/2 = 3.0
    rows = [{}, {}, {"home_score": 0, "away_score": 1}]
    long = _featurize(rows)
    brazil = long[long.team == "Brazil"].sort_values("date")
    assert np.isnan(brazil["form_pts_last5"].iloc[0])      # no history yet
    assert brazil["form_pts_last5"].iloc[1] == 3.0
    assert brazil["form_pts_last5"].iloc[2] == 3.0          # result of match 3 NOT included


def test_win_pct_and_goals():
    rows = [{"home_score": 2, "away_score": 0},
            {"home_score": 0, "away_score": 1},
            {}]
    long = _featurize(rows)
    b = long[long.team == "Brazil"].sort_values("date")
    assert b["win_pct_last20"].iloc[2] == pytest.approx(0.5)
    assert b["avg_goals_scored"].iloc[2] == pytest.approx(1.0)   # (2+0)/2
    assert b["avg_goals_conceded"].iloc[2] == pytest.approx(0.5)  # (0+1)/2
    assert b["goal_diff_per_game"].iloc[2] == pytest.approx(0.5)


def test_wc_experience_counts_only_prior_wc_matches():
    rows = [{"tournament": "FIFA World Cup"},
            {"tournament": "Friendly"},
            {"tournament": "FIFA World Cup"}]
    long = _featurize(rows)
    b = long[long.team == "Brazil"].sort_values("date")
    assert b["wc_matches_played"].tolist() == [0, 1, 1]


def test_strength_of_schedule_uses_pre_match_opponent_elo():
    long = _featurize([{}, {}])
    b = long[long.team == "Brazil"].sort_values("date")
    # Before match 2, Brazil's only prior opponent (Argentina) had Elo 1500 pre-match.
    assert b["opp_strength_last10"].iloc[1] == pytest.approx(1500.0)


# ------------------------------------------------------------- continent ----
def test_continent_advantage():
    m = add_elo(make_matches([
        {"country": "Brazil"},                                  # Brazil home in CONMEBOL
        {"home_team": "Germany", "away_team": "Brazil",
         "country": "United States", "neutral": True},          # neither side's confed
        {"home_team": "Mexico", "away_team": "Germany",
         "country": "United States", "neutral": True},          # Mexico in CONCACAF
    ]))
    long = add_continent_advantage(to_long(m))
    get = lambda team, mid: int(long[(long.team == team) & (long.match_id == mid)]
                                ["continent_advantage"].iloc[0])
    assert get("Brazil", 0) == 1
    assert get("Germany", 1) == 0 and get("Brazil", 1) == 0
    assert get("Mexico", 2) == 1 and get("Germany", 2) == 0


# ------------------------------------------------------------ FIFA ranks ----
def test_fifa_rank_asof_is_strictly_before_match():
    m = add_elo(make_matches([{"date": "2024-06-15"}]))
    long = to_long(m)
    rankings = pd.DataFrame({
        "country_full": ["Brazil", "Brazil", "Argentina"],
        "rank": [5, 3, 1],
        "rank_date": pd.to_datetime(["2024-05-01", "2024-06-15", "2024-05-01"]),
    })
    out = add_fifa_rank(long, rankings)
    b = out[out.team == "Brazil"]
    # The 2024-06-15 release (same day) is EXCLUDED -> rank 5 from May, not 3.
    assert b["fifa_rank"].iloc[0] == 5


# ------------------------------------------------- THE LEAKAGE TEST ---------
def test_no_leakage_flipping_last_result_changes_no_feature():
    """Build features; then flip the final match's result and rebuild.
    Every feature OF THAT MATCH must be bit-identical. If any feature moves,
    it saw its own result - the bug this whole architecture exists to prevent."""
    rows = [{"home_score": (i * 7) % 4, "away_score": (i * 5) % 3,
             "tournament": "Friendly" if i % 2 else "FIFA World Cup qualification"}
            for i in range(30)]
    base = make_matches(rows)

    flipped = base.copy()
    flipped.loc[flipped.index[-1], ["home_score", "away_score"]] = [9, 0]

    fa = build_features(base)
    fb = build_features(flipped)

    feature_cols = [c for c in fa.columns
                    if c.startswith(("home_", "away_")) or c.endswith("_diff")]
    feature_cols = [c for c in feature_cols
                    if c not in ("home_team", "away_team", "home_score", "away_score",
                                 "home_team_id", "away_team_id")]
    last = fa.index[-1]
    pd.testing.assert_series_equal(
        fa.loc[last, feature_cols], fb.loc[last, feature_cols],
        check_names=False)


def test_match_level_shape_and_diffs():
    out = build_features(make_matches([{}, {}, {}]))
    assert {"home_elo_pre", "away_elo_pre", "elo_pre_diff",
            "form_pts_last5_diff", "target"} <= set(out.columns)
    assert len(out) == 3
    assert out["target"].iloc[0] == "home_win"
    assert out["elo_pre_diff"].iloc[1] == pytest.approx(
        out["home_elo_pre"].iloc[1] - out["away_elo_pre"].iloc[1])
