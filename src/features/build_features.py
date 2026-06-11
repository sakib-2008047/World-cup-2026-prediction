"""Feature engineering: clean matches -> model-ready feature matrix.

ARCHITECTURE - the "long format" pattern:

    match rows (1 per match)
        -> team-match rows (2 per match: each team's perspective)
        -> per-team rolling features, ALL through shift(1)
        -> pivot back to match rows with home_* / away_* / *_diff columns

Every rolling feature is computed as  group.shift(1).rolling(...)  - the
shift(1) guarantees the current match's result can never leak into its own
features. One defense, applied uniformly, instead of eleven separate chances
to make a mistake. tests/test_features.py::test_no_leakage proves it by
flipping a result and asserting that match's features don't move.

Usage:
    from src.features.build_features import build_features
    X = build_features(matches_clean, rankings=fifa_rankings)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.normalize import load_wc2026_teams  # noqa: E402
from src.features.elo import EloEngine, add_elo  # noqa: E402

CONFED_PATH = PROJECT_ROOT / "data" / "external" / "confederations.csv"

WORLD_CUP_TOURNAMENTS = {"FIFA World Cup"}
FORM_WINDOWS = (5, 10)
ROLLING_WINDOW = 10       # goals / goal-diff / strength-of-schedule window
WIN_PCT_WINDOW = 20       # win% over a longer horizon - more stable


# ----------------------------------------------------------------------------
# Long format: one row per (team, match)
# ----------------------------------------------------------------------------
def to_long(matches: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match - the home team's view and the away team's view.
    All per-team time series logic (rolling form, streaks, experience) becomes
    a plain groupby('team') instead of awkward home/away case handling."""
    m = matches.reset_index(drop=True).rename_axis("match_id").reset_index()

    def view(side: str, other: str) -> pd.DataFrame:
        return pd.DataFrame({
            "match_id": m["match_id"],
            "date": m["date"],
            "team": m[f"{side}_team"],
            "opponent": m[f"{other}_team"],
            "gf": m[f"{side}_score"].astype(int),
            "ga": m[f"{other}_score"].astype(int),
            "side": side,
            "tournament": m["tournament"],
            "host_country": m["country"],
            "elo_pre": m[f"{side}_elo_pre"],
            "opp_elo_pre": m[f"{other}_elo_pre"],
        })

    long = pd.concat([view("home", "away"), view("away", "home")],
                     ignore_index=True)
    long["points"] = np.select(
        [long.gf > long.ga, long.gf == long.ga], [3, 1], default=0)
    long["won"] = (long.gf > long.ga).astype(int)
    long["is_world_cup"] = long["tournament"].isin(WORLD_CUP_TOURNAMENTS).astype(int)
    # Stable order: by team, then date, then match_id (same-day tiebreak).
    return long.sort_values(["team", "date", "match_id"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Rolling features (all leakage-safe via shift(1))
# ----------------------------------------------------------------------------
def add_rolling_features(long: pd.DataFrame) -> pd.DataFrame:
    long = long.copy()
    g = long.groupby("team", sort=False)

    def lagged_roll(col: str, window: int, agg: str = "mean") -> pd.Series:
        return g[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).agg(agg))

    # Form: average points per game over the last N matches.
    for w in FORM_WINDOWS:
        long[f"form_pts_last{w}"] = lagged_roll("points", w)

    # Win percentage over a longer horizon.
    long[f"win_pct_last{WIN_PCT_WINDOW}"] = lagged_roll("won", WIN_PCT_WINDOW)

    # Attack / defense / balance over the rolling window.
    long["goal_diff_per_game"] = lagged_roll("gf", ROLLING_WINDOW) - \
        lagged_roll("ga", ROLLING_WINDOW)
    long["avg_goals_scored"] = lagged_roll("gf", ROLLING_WINDOW)
    long["avg_goals_conceded"] = lagged_roll("ga", ROLLING_WINDOW)

    # Strength of schedule: mean PRE-match Elo of recent opponents.
    long["opp_strength_last10"] = lagged_roll("opp_elo_pre", ROLLING_WINDOW)

    # Tournament experience: World Cup finals matches played BEFORE this one.
    long["wc_matches_played"] = g["is_world_cup"].transform(
        lambda s: s.shift(1).cumsum()).fillna(0)

    # Matches played at all (rookie-nation indicator / minutes of history).
    long["career_matches"] = g.cumcount()
    return long


# ----------------------------------------------------------------------------
# Confederation / home-continent advantage
# ----------------------------------------------------------------------------
def load_confederations(path: Path = CONFED_PATH) -> dict[str, str]:
    """team -> confederation. Seeded from wc2026_teams.csv, extended by
    data/external/confederations.csv. Teams absent from both map to 'unknown'
    and the advantage feature degrades to 0 - graceful, and logged by tests."""
    confed = dict(zip(*load_wc2026_teams()[["team", "confederation"]].T.values))
    if path.exists():
        extra = pd.read_csv(path, comment="#")
        confed.update(dict(zip(extra["team"], extra["confederation"])))
    return confed


def add_continent_advantage(long: pd.DataFrame,
                            confed: dict[str, str] | None = None) -> pd.DataFrame:
    """1 if the match is played in the team's own confederation.

    The host country column holds a country name, which is (almost always)
    itself a national team - so one map serves both lookups. European teams
    historically over-perform at European World Cups and under-perform in the
    Americas; in 2026 this flag lights up for all CONCACAF sides."""
    confed = confed or load_confederations()
    long = long.copy()
    team_conf = long["team"].map(confed)
    host_conf = long["host_country"].map(confed)
    long["continent_advantage"] = (
        team_conf.notna() & (team_conf == host_conf)).astype(int)
    return long


# ----------------------------------------------------------------------------
# FIFA rankings (as-of merge)
# ----------------------------------------------------------------------------
def add_fifa_rank(long: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """Latest FIFA rank published STRICTLY BEFORE the match date.

    allow_exact_matches=False: a ranking released on matchday could already
    embed that match's result under the post-2018 formula - exclude it.
    Pre-1993 matches get NaN (rankings didn't exist): we keep NaN rather than
    invent a number; gradient-boosted trees treat NaN as 'missing' natively.
    Use ordinal rank, not points - FIFA changed the points formula in 1999,
    2006, and 2018, so points are not comparable across eras."""
    r = rankings.rename(columns={"country_full": "team"})
    r = r[["team", "rank", "rank_date"]].sort_values("rank_date")
    long = long.sort_values("date")
    out = pd.merge_asof(
        long, r,
        left_on="date", right_on="rank_date", by="team",
        direction="backward", allow_exact_matches=False,
    ).drop(columns="rank_date").rename(columns={"rank": "fifa_rank"})
    return out.sort_values(["team", "date", "match_id"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Pivot back to match level
# ----------------------------------------------------------------------------
TEAM_FEATURES = [
    "elo_pre", "fifa_rank",
    "form_pts_last5", "form_pts_last10", f"win_pct_last{WIN_PCT_WINDOW}",
    "goal_diff_per_game", "avg_goals_scored", "avg_goals_conceded",
    "opp_strength_last10", "wc_matches_played", "career_matches",
    "continent_advantage",
]


def to_match_level(matches: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    """home_* and away_* columns plus *_diff contrasts. Differences matter
    more than levels: 'Elo 1900 vs 1700' carries the signal, not '1900'."""
    feats = [f for f in TEAM_FEATURES if f in long.columns]
    wide = long.set_index(["match_id", "side"])[feats].unstack("side")
    wide.columns = [f"{side}_{feat}" for feat, side in wide.columns]

    out = matches.reset_index(drop=True).rename_axis("match_id").reset_index()
    # add_elo already wrote home/away_elo_pre on the match frame; the pivot
    # regenerates them from the long format - drop originals to avoid _x/_y.
    out = out.drop(columns=[c for c in ("home_elo_pre", "away_elo_pre")
                            if c in out.columns])
    out = out.merge(wide.reset_index(), on="match_id", how="left")

    for f in feats:
        if f == "continent_advantage":
            continue  # a flag, not a magnitude - diff is less meaningful
        out[f"{f}_diff"] = out[f"home_{f}"] - out[f"away_{f}"]
    # FIFA rank: lower is better, so flip for an intuitive sign
    if "fifa_rank_diff" in out.columns:
        out["fifa_rank_diff"] = -out["fifa_rank_diff"]

    out["target"] = np.select(
        [out.home_score > out.away_score, out.home_score == out.away_score],
        ["home_win", "draw"], default="away_win")
    return out


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------
def build_features(matches: pd.DataFrame,
                   rankings: pd.DataFrame | None = None,
                   elo_engine: EloEngine | None = None) -> pd.DataFrame:
    """Clean matches in, feature matrix out. Pure function of its inputs."""
    matches = add_elo(matches, elo_engine)
    long = to_long(matches)
    long = add_rolling_features(long)
    long = add_continent_advantage(long)
    if rankings is not None:
        long = add_fifa_rank(long, rankings)
    out = to_match_level(matches, long)
    # NOTE: deliberately NOT propagating the EloEngine onto out.attrs -
    # pandas serializes attrs into parquet metadata (json), and a live
    # object can't survive that. Get final ratings from add_elo's frame:
    #     rated = add_elo(matches); rated.attrs["elo_engine"].ratings
    return out
