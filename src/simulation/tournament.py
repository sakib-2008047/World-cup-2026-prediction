"""Object-oriented domain model of the 2026 World Cup.

Relationship to monte_carlo.py: that module is the FAST path (vectorized,
hundreds of tournaments/sec) for bulk Monte Carlo. This module is the
EXPRESSIVE path: explicit Team/Match/Group/KnockoutStage/Tournament objects
that make one tournament inspectable and - crucially - support LOCKING real
results, so once actual matches are played you simulate only what remains.
Both share rank_group() and allocate_thirds(), so the rules live in one place.

    teams = {name: Team(name, group, rating) ...}
    t = Tournament.build_2026(teams)
    t.lock_result("Mexico", "South Africa", 2, 1)      # real result
    outcome = t.play(sampler, rng)                      # simulate the rest
    outcome.champion, outcome.finalists, outcome.semifinalists

Monte Carlo over the OO model:
    results = [Tournament.build_2026(teams, locked).play(sampler, rng)
               for _ in range(n)]
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.bracket_2026 import (  # noqa: E402
    GROUPS, HOST_COUNTRIES, KNOCKOUT_ROUNDS, ROUND_NAMES, ROUND_OF_32,
    THIRD_SLOTS,
)
from src.simulation.monte_carlo import allocate_thirds, rank_group  # noqa: E402

#: (team_a, team_b, neutral) -> (lambda_a, lambda_b)
ScoreSampler = Callable[[str, str, bool], tuple[float, float]]


# ----------------------------------------------------------------------------
# Team
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Team:
    name: str
    group: str
    rating: float
    confederation: str = "unknown"

    @property
    def is_host(self) -> bool:
        return self.name in HOST_COUNTRIES


# ----------------------------------------------------------------------------
# Match
# ----------------------------------------------------------------------------
@dataclass
class Match:
    """One fixture. A Match can be LOCKED (real result injected) or sampled.

    Knockout resolution lives here: regulation -> extra time (rates / 3)
    -> penalties (coin flip with small rating tilt). `winner` is None for
    drawn group matches, never None for knockouts."""

    home: Team
    away: Team
    stage: str                       # 'group' | 'R32' | 'R16' | 'QF' | 'SF' | 'F'
    knockout: bool = False
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    decided_by: str = "unplayed"     # 'regulation' | 'extra_time' | 'penalties' | 'locked'
    winner: Optional[Team] = None
    PENALTY_TILT: float = 0.0002

    @property
    def played(self) -> bool:
        return self.home_goals is not None

    @property
    def neutral(self) -> bool:
        return not (self.home.is_host and not self.away.is_host)

    def lock(self, home_goals: int, away_goals: int,
             winner_name: str | None = None) -> "Match":
        """Inject a real-world result. For drawn knockouts, winner_name
        identifies who progressed (ET/pens already resolved in reality)."""
        self.home_goals, self.away_goals = home_goals, away_goals
        self.decided_by = "locked"
        if home_goals != away_goals:
            self.winner = self.home if home_goals > away_goals else self.away
        elif self.knockout:
            assert winner_name in (self.home.name, self.away.name), \
                "drawn knockout lock requires winner_name"
            self.winner = self.home if winner_name == self.home.name else self.away
        return self

    def play(self, sampler: ScoreSampler, rng: np.random.Generator) -> "Match":
        if self.played:                      # locked results are never re-rolled
            return self
        la, lb = sampler(self.home.name, self.away.name, self.neutral)
        self.home_goals = int(rng.poisson(la))
        self.away_goals = int(rng.poisson(lb))
        self.decided_by = "regulation"
        if self.home_goals != self.away_goals:
            self.winner = self.home if self.home_goals > self.away_goals else self.away
        elif self.knockout:
            et_h, et_a = int(rng.poisson(la / 3)), int(rng.poisson(lb / 3))
            self.home_goals += et_h
            self.away_goals += et_a
            if et_h != et_a:
                self.decided_by = "extra_time"
                self.winner = self.home if et_h > et_a else self.away
            else:
                self.decided_by = "penalties"
                d = np.clip(self.home.rating - self.away.rating, -400, 400)
                p_home = 0.5 + d * self.PENALTY_TILT
                self.winner = self.home if rng.random() < p_home else self.away
        return self

    def __repr__(self) -> str:
        score = (f"{self.home_goals}-{self.away_goals}" if self.played else "v")
        suffix = f" ({self.decided_by})" if self.decided_by in (
            "extra_time", "penalties") else ""
        return f"[{self.stage}] {self.home.name} {score} {self.away.name}{suffix}"


# ----------------------------------------------------------------------------
# Group
# ----------------------------------------------------------------------------
@dataclass
class Group:
    letter: str
    teams: list[Team]
    matches: list[Match] = field(default_factory=list)

    def __post_init__(self):
        assert len(self.teams) == 4, f"group {self.letter} needs 4 teams"
        if not self.matches:
            for i in range(4):
                for j in range(i + 1, 4):
                    a, b = self.teams[i], self.teams[j]
                    if b.is_host and not a.is_host:
                        a, b = b, a          # hosts at home
                    self.matches.append(Match(a, b, stage="group"))

    def play(self, sampler: ScoreSampler, rng: np.random.Generator) -> None:
        for m in self.matches:
            m.play(sampler, rng)

    def standings(self, rng: np.random.Generator) -> list[dict]:
        results = [(m.home.name, m.away.name, m.home_goals, m.away_goals)
                   for m in self.matches]
        return rank_group([t.name for t in self.teams], results, rng)

    def find_match(self, a: str, b: str) -> Match | None:
        for m in self.matches:
            if {m.home.name, m.away.name} == {a, b}:
                return m
        return None


# ----------------------------------------------------------------------------
# Knockout stage
# ----------------------------------------------------------------------------
@dataclass
class KnockoutStage:
    """One round of the bracket. R32 resolves from group-position slots and
    the third-place assignment; later rounds resolve from feeder winners."""

    name: str                                # 'R32', 'R16', ...
    spec: dict                               # match_no -> slots or feeders
    matches: dict[int, Match] = field(default_factory=dict)

    def resolve_and_play(self, ctx: "Tournament", sampler: ScoreSampler,
                         rng: np.random.Generator) -> None:
        for match_no, (sa, sb) in self.spec.items():
            if isinstance(sa, str):          # R32 slot notation
                a = (ctx.third_assignment[match_no] if sa.startswith("3rd")
                     else ctx.slot_team[sa])
                b = (ctx.third_assignment[match_no] if sb.startswith("3rd")
                     else ctx.slot_team[sb])
            else:                            # feeder match numbers
                a = ctx.winners[sa].name
                b = ctx.winners[sb].name
            home, away = ctx.team(a), ctx.team(b)
            if away.is_host and not home.is_host:
                home, away = away, home
            m = ctx.locked_knockouts.pop(frozenset((a, b)), None)
            if m is None:
                m = Match(home, away, stage=self.name, knockout=True)
            else:
                m = Match(home, away, stage=self.name, knockout=True).lock(
                    *m)  # m is (hg, ag, winner_name)
            self.matches[match_no] = m.play(sampler, rng)
            ctx.winners[match_no] = m.winner


# ----------------------------------------------------------------------------
# Tournament
# ----------------------------------------------------------------------------
@dataclass
class TournamentOutcome:
    champion: str
    finalists: list[str]
    semifinalists: list[str]
    stage_reached: dict[str, str]
    matches: list[Match]


@dataclass
class Tournament:
    """The aggregate root: 12 Groups + 5 KnockoutStages + the glue state
    (slot assignments, third allocation, winners by match number)."""

    teams_by_name: dict[str, Team]
    groups: dict[str, Group]
    knockout_stages: list[KnockoutStage]
    # conditioning state
    locked_knockouts: dict[frozenset, tuple] = field(default_factory=dict)
    # per-run resolution state
    slot_team: dict[str, str] = field(default_factory=dict)
    third_assignment: dict[int, str] = field(default_factory=dict)
    winners: dict[int, Team] = field(default_factory=dict)

    # -------------------------------------------------------- construction --
    @classmethod
    def build_2026(cls, teams: dict[str, Team]) -> "Tournament":
        groups = {}
        for g in GROUPS:
            members = [t for t in teams.values() if t.group == g]
            groups[g] = Group(g, members)
        stages = [KnockoutStage(name, spec)
                  for name, spec in zip(ROUND_NAMES, KNOCKOUT_ROUNDS)]
        return cls(teams_by_name=teams, groups=groups, knockout_stages=stages)

    def team(self, name: str) -> Team:
        return self.teams_by_name[name]

    # -------------------------------------------------------- conditioning --
    def lock_result(self, team_a: str, team_b: str, goals_a: int, goals_b: int,
                    winner: str | None = None) -> None:
        """Pin a real result. Group matches lock immediately on the fixture;
        knockout locks are stored by team-pair and applied when the bracket
        produces that pairing."""
        ga = self.team(team_a).group
        if ga == self.team(team_b).group:
            m = self.groups[ga].find_match(team_a, team_b)
            assert m is not None
            # orient goals to the fixture's home/away
            if m.home.name == team_a:
                m.lock(goals_a, goals_b, winner)
            else:
                m.lock(goals_b, goals_a, winner)
        else:
            self.locked_knockouts[frozenset((team_a, team_b))] = (
                goals_a, goals_b, winner)

    # ---------------------------------------------------------------- play --
    def play(self, sampler: ScoreSampler, rng: np.random.Generator
             ) -> TournamentOutcome:
        stage_reached = {name: "group" for name in self.teams_by_name}

        for g in GROUPS:
            group = self.groups[g]
            group.play(sampler, rng)
            rows = group.standings(rng)
            self.slot_team[f"{g}1"] = rows[0]["team"]
            self.slot_team[f"{g}2"] = rows[1]["team"]
            self.groups[g].third_row = {**rows[2], "group": g}

        thirds = [self.groups[g].third_row for g in GROUPS]
        thirds.sort(key=lambda r: (-r["pts"], -r["gd"], -r["gf"], rng.random()))
        qualified = {r["group"]: r["team"] for r in thirds[:8]}
        self.third_assignment = allocate_thirds(qualified) or {}

        for stage in self.knockout_stages:
            stage.resolve_and_play(self, sampler, rng)
            for m in stage.matches.values():
                stage_reached[m.home.name] = stage.name
                stage_reached[m.away.name] = stage.name

        final = self.knockout_stages[-1].matches
        final_match = next(iter(final.values()))
        champion = final_match.winner.name
        stage_reached[champion] = "champion"
        semis = self.knockout_stages[-2].matches.values()

        all_matches = [m for g in self.groups.values() for m in g.matches]
        all_matches += [m for s in self.knockout_stages for m in s.matches.values()]
        return TournamentOutcome(
            champion=champion,
            finalists=[final_match.home.name, final_match.away.name],
            semifinalists=sorted({m.home.name for m in semis}
                                 | {m.away.name for m in semis}),
            stage_reached=stage_reached,
            matches=all_matches,
        )


# ----------------------------------------------------------------------------
# Monte Carlo over the OO model
# ----------------------------------------------------------------------------
def monte_carlo(teams: dict[str, Team], sampler: ScoreSampler,
                n_sims: int = 1000, seed: int = 42,
                locked: list[tuple] | None = None) -> Counter:
    """Champion frequency over n_sims, honoring locked real results.
    locked: list of (team_a, team_b, goals_a, goals_b[, winner]) tuples."""
    rng = np.random.default_rng(seed)
    champions: Counter = Counter()
    for _ in range(n_sims):
        t = Tournament.build_2026(teams)
        for row in (locked or []):
            t.lock_result(*row)
        champions[t.play(sampler, rng).champion] += 1
    return champions
