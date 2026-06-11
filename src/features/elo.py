"""Chronological Elo rating engine.

WHY ELO MATTERS: a single number that integrates a team's entire match history,
weighted by opponent quality, match importance, and recency. It answers the
question every other feature only approximates: "how strong is this team
right now?" Empirically, Elo difference alone predicts international match
outcomes better than FIFA rank difference, especially pre-2018 (when the FIFA
formula was famously gameable - teams avoided friendlies to protect points).

LEAKAGE DESIGN: ratings are recorded BEFORE each match is processed, then
updated with its result. The pre-match snapshot is the feature; the update
only affects FUTURE rows. Processing strictly in date order makes leakage
structurally impossible rather than something to remember.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

#: K-factor by competition: how fast ratings move. A World Cup match carries
#: 3x the information of a friendly (teams at full strength, full motivation).
K_BY_TOURNAMENT = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 40,
    "Copa América": 50,
    "UEFA Euro": 50,
    "UEFA Euro qualification": 40,
    "African Cup of Nations": 50,
    "African Cup of Nations qualification": 40,
    "AFC Asian Cup": 50,
    "AFC Asian Cup qualification": 40,
    "CONCACAF Championship": 50,
    "Confederations Cup": 40,
    "UEFA Nations League": 40,
    "CONCACAF Nations League": 40,
    "Friendly": 20,
}
DEFAULT_K = 30
HOME_ADVANTAGE_ELO = 100  # ≈ 64% expected score for an otherwise equal home team
START_ELO = 1500.0


@dataclass
class EloEngine:
    """Stateful Elo calculator. Feed matches in chronological order."""

    k_map: dict[str, int] = field(default_factory=lambda: dict(K_BY_TOURNAMENT))
    home_adv: float = HOME_ADVANTAGE_ELO
    start: float = START_ELO
    ratings: dict[str, float] = field(default_factory=dict)

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.start)

    @staticmethod
    def expected(r_a: float, r_b: float) -> float:
        """Probability-like expected score of A vs B (win=1, draw=0.5)."""
        return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400))

    @staticmethod
    def goal_multiplier(margin: int) -> float:
        """Margin-of-victory scaling (World Football Elo convention):
        a 4-0 should move ratings more than a 1-0, but sublinearly -
        running up the score against minnows shouldn't mint rating points."""
        if margin <= 1:
            return 1.0
        if margin == 2:
            return 1.5
        return (11 + margin) / 8

    def update(self, home: str, away: str, home_score: int, away_score: int,
               tournament: str, neutral: bool) -> tuple[float, float]:
        """Process one match. Returns the PRE-match ratings (the features),
        then updates internal state with the result."""
        ra, rb = self.get(home), self.get(away)

        adv = 0.0 if neutral else self.home_adv
        exp_home = self.expected(ra + adv, rb)
        actual = 1.0 if home_score > away_score else (
            0.0 if home_score < away_score else 0.5)

        k = self.k_map.get(tournament, DEFAULT_K)
        delta = k * self.goal_multiplier(abs(home_score - away_score)) * (actual - exp_home)

        self.ratings[home] = ra + delta
        self.ratings[away] = rb - delta
        return ra, rb


def add_elo(matches: pd.DataFrame, engine: EloEngine | None = None) -> pd.DataFrame:
    """Append home_elo_pre / away_elo_pre to a date-sorted match dataframe.

    Returns a copy. The engine (with final post-data ratings) is attached as
    df.attrs['elo_engine'] - those final ratings are what the tournament
    simulator starts from.
    """
    assert matches["date"].is_monotonic_increasing, \
        "Elo requires chronological order - run the cleaning pipeline first"
    engine = engine or EloEngine()
    home_pre, away_pre = [], []
    for m in matches.itertuples(index=False):
        ra, rb = engine.update(m.home_team, m.away_team,
                               int(m.home_score), int(m.away_score),
                               m.tournament, bool(m.neutral))
        home_pre.append(ra)
        away_pre.append(rb)
    out = matches.copy()
    out["home_elo_pre"] = home_pre
    out["away_elo_pre"] = away_pre
    out.attrs["elo_engine"] = engine
    return out
