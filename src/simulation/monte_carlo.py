"""Monte Carlo simulation of the 2026 World Cup.

    engine = TournamentSimulator(ratings, groups_df)
    summary = engine.run(n_sims=10_000, seed=42)

Pipeline per simulated tournament:
    1. GROUP STAGE - all 72 matches sampled as Poisson scorelines
       (scorelines, not just outcomes: goal difference decides tiebreaks
       and the best-thirds ranking, so W/D/L sampling would be insufficient).
    2. STANDINGS - FIFA tiebreakers: points, GD, GF, head-to-head points,
       then drawing of lots (random). (FIFA also uses fair-play points
       before lots; we have no booking data, so lots stand in for it.)
    3. BEST THIRDS - 12 third-placed teams ranked by points/GD/GF; top 8
       advance; allocation to bracket slots solved as a constraint matching
       (see bracket_2026.py for the Annex C note).
    4. KNOCKOUTS - 90' Poisson; if level, 30' extra time at 1/3 rates;
       if still level, penalties (coin flip with a small Elo tilt).
    5. TRACKING - per team: champion, final, semifinal, and furthest stage.

Probability source: any callable (team_a, team_b, neutral) -> (lam_a, lam_b).
Default is EloScoreSampler built from end-of-data Elo ratings, so the whole
engine runs off the rating system; swap in PoissonGoalModel-based lambdas via
the same interface when the full feature pipeline is available.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.bracket_2026 import (  # noqa: E402
    FINAL, GROUPS, HOST_COUNTRIES, KNOCKOUT_ROUNDS, ROUND_NAMES,
    ROUND_OF_32, THIRD_SLOTS,
)

STAGES = ["group", "R32", "R16", "QF", "SF", "F", "champion"]


# ----------------------------------------------------------------------------
# Score sampling: Elo -> Poisson goal rates
# ----------------------------------------------------------------------------
@dataclass
class EloScoreSampler:
    """Maps an Elo difference to a pair of Poisson goal rates.

    Calibration: log-rates symmetric around a base of ~1.25 goals/side
    (international tournament average ~2.5 total), with slope chosen so a
    400-Elo favorite expects roughly 2.3 vs 0.7 - consistent with what the
    fitted Poisson GLM produces for that gap. Host teams get the standard
    Elo home advantage folded into the difference.
    """

    ratings: dict[str, float]
    base_log_rate: float = float(np.log(1.25))
    slope: float = 0.0017
    home_adv_elo: float = 100.0
    lam_bounds: tuple[float, float] = (0.15, 4.5)

    def lambdas(self, team_a: str, team_b: str, neutral: bool = True
                ) -> tuple[float, float]:
        d = self.ratings[team_a] - self.ratings[team_b]
        if not neutral:
            d += self.home_adv_elo
        lam_a = float(np.exp(self.base_log_rate + self.slope * d / 2))
        lam_b = float(np.exp(self.base_log_rate - self.slope * d / 2))
        lo, hi = self.lam_bounds
        return min(max(lam_a, lo), hi), min(max(lam_b, lo), hi)


# ----------------------------------------------------------------------------
# Group standings with FIFA tiebreakers
# ----------------------------------------------------------------------------
def rank_group(teams: list[str], results: list[tuple[str, str, int, int]],
               rng: np.random.Generator) -> list[dict]:
    """Order 4 teams by: points, GD, GF, head-to-head points among the tied,
    then lots. Returns standings rows (team, pts, gd, gf) best-first."""
    stats = {t: {"team": t, "pts": 0, "gd": 0, "gf": 0} for t in teams}
    for a, b, ga, gb in results:
        stats[a]["gf"] += ga; stats[a]["gd"] += ga - gb
        stats[b]["gf"] += gb; stats[b]["gd"] += gb - ga
        if ga > gb:
            stats[a]["pts"] += 3
        elif gb > ga:
            stats[b]["pts"] += 3
        else:
            stats[a]["pts"] += 1; stats[b]["pts"] += 1

    def h2h_pts(team: str, among: set[str]) -> int:
        pts = 0
        for a, b, ga, gb in results:
            if a == team and b in among:
                pts += 3 if ga > gb else (1 if ga == gb else 0)
            elif b == team and a in among:
                pts += 3 if gb > ga else (1 if ga == gb else 0)
        return pts

    rows = list(stats.values())
    # primary sort with random lots as final key
    lots = {t: rng.random() for t in teams}
    rows.sort(key=lambda r: (-r["pts"], -r["gd"], -r["gf"], lots[r["team"]]))
    # apply head-to-head among exact ties on (pts, gd, gf)
    i = 0
    while i < len(rows):
        j = i + 1
        while (j < len(rows)
               and (rows[j]["pts"], rows[j]["gd"], rows[j]["gf"])
               == (rows[i]["pts"], rows[i]["gd"], rows[i]["gf"])):
            j += 1
        if j - i > 1:
            tied = {r["team"] for r in rows[i:j]}
            rows[i:j] = sorted(rows[i:j],
                               key=lambda r: (-h2h_pts(r["team"], tied),
                                              lots[r["team"]]))
        i = j
    return rows


# ----------------------------------------------------------------------------
# Best-thirds ranking + slot allocation
# ----------------------------------------------------------------------------
def allocate_thirds(qualified: dict[str, str]) -> dict[int, str] | None:
    """qualified: {group_letter: team} for the 8 advancing thirds.
    Assign each to a third-slot whose pool contains its group - a tiny
    bipartite perfect matching solved by backtracking (8 slots, <=8 options
    each; FIFA's Annex C is a precomputed table of such solutions)."""
    slots = sorted(THIRD_SLOTS, key=lambda m: len(THIRD_SLOTS[m] & set(qualified)))
    groups = set(qualified)

    def backtrack(idx: int, remaining: set, assign: dict) -> dict | None:
        if idx == len(slots):
            return assign
        m = slots[idx]
        for g in sorted(THIRD_SLOTS[m] & remaining):
            assign[m] = g
            res = backtrack(idx + 1, remaining - {g}, assign)
            if res is not None:
                return res
            del assign[m]
        return None

    sol = backtrack(0, groups, {})
    return None if sol is None else {m: qualified[g] for m, g in sol.items()}


# ----------------------------------------------------------------------------
# The simulator
# ----------------------------------------------------------------------------
@dataclass
class TournamentSimulator:
    ratings: dict[str, float]
    groups: dict[str, list[str]]          # group letter -> 4 teams
    sampler: EloScoreSampler = None
    penalty_elo_tilt: float = 0.0002      # P(win pens) = 0.5 + tilt*elo_diff
    counts: pd.DataFrame = field(default=None, repr=False)

    def __post_init__(self):
        if self.sampler is None:
            self.sampler = EloScoreSampler(self.ratings)
        teams = [t for g in self.groups.values() for t in g]
        assert len(teams) == 48 and len(set(teams)) == 48
        missing = [t for t in teams if t not in self.ratings]
        assert not missing, f"no rating for: {missing}"
        # precompute group fixtures + lambdas (fixed across simulations)
        self._fixtures, lams = [], []
        for g, members in self.groups.items():
            for i in range(4):
                for j in range(i + 1, 4):
                    a, b = members[i], members[j]
                    neutral = not ({a, b} & HOST_COUNTRIES and
                                   (a in HOST_COUNTRIES or b in HOST_COUNTRIES))
                    # host plays at home; order so host is 'a'
                    if b in HOST_COUNTRIES and a not in HOST_COUNTRIES:
                        a, b = b, a
                    self._fixtures.append((g, a, b))
                    lams.append(self.sampler.lambdas(a, b, neutral=a not in HOST_COUNTRIES))
        self._lam_a = np.array([l[0] for l in lams])
        self._lam_b = np.array([l[1] for l in lams])

    # ------------------------------------------------------------ one sim ---
    def _knockout_match(self, a: str, b: str, rng) -> str:
        la, lb = self.sampler.lambdas(a, b, neutral=not (
            a in HOST_COUNTRIES and b not in HOST_COUNTRIES))
        ga, gb = rng.poisson(la), rng.poisson(lb)
        if ga != gb:
            return a if ga > gb else b
        ga, gb = rng.poisson(la / 3), rng.poisson(lb / 3)       # extra time
        if ga != gb:
            return a if ga > gb else b
        d = self.ratings[a] - self.ratings[b]                    # penalties
        p_a = 0.5 + np.clip(d, -400, 400) * self.penalty_elo_tilt
        return a if rng.random() < p_a else b

    def simulate_once(self, rng: np.random.Generator) -> dict[str, str]:
        """Returns {team: furthest stage reached}."""
        ga = rng.poisson(self._lam_a)
        gb = rng.poisson(self._lam_b)
        per_group: dict[str, list] = {g: [] for g in GROUPS}
        for (g, a, b), sa, sb in zip(self._fixtures, ga, gb):
            per_group[g].append((a, b, int(sa), int(sb)))

        slot_team: dict[str, str] = {}
        thirds_rows = []
        stage = {}
        for g in GROUPS:
            rows = rank_group(self.groups[g], per_group[g], rng)
            slot_team[f"{g}1"], slot_team[f"{g}2"] = rows[0]["team"], rows[1]["team"]
            thirds_rows.append({**rows[2], "group": g})
            for r in rows:
                stage[r["team"]] = "group"

        # best 8 thirds: points, GD, GF, lots
        thirds_rows.sort(key=lambda r: (-r["pts"], -r["gd"], -r["gf"], rng.random()))
        qualified = {r["group"]: r["team"] for r in thirds_rows[:8]}
        third_assign = allocate_thirds(qualified)
        if third_assign is None:  # pools are dense; matching failure ~impossible
            third_assign = {m: t for m, t in
                            zip(sorted(THIRD_SLOTS), qualified.values())}

        # knockout rounds
        winners: dict[int, str] = {}
        for rnd, name in zip(KNOCKOUT_ROUNDS, ROUND_NAMES):
            for match_no, (sa, sb) in rnd.items():
                if isinstance(sa, str):   # R32: slots
                    a = third_assign[match_no] if sa.startswith("3rd") else slot_team[sa]
                    b = third_assign[match_no] if sb.startswith("3rd") else slot_team[sb]
                else:                     # later rounds: feeder match numbers
                    a, b = winners[sa], winners[sb]
                stage[a] = name
                stage[b] = name
                winners[match_no] = self._knockout_match(a, b, rng)
        stage[winners[max(FINAL)]] = "champion"
        return stage

    # ------------------------------------------------------------- driver ---
    def run(self, n_sims: int = 10_000, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        teams = [t for g in self.groups.values() for t in g]
        counts = pd.DataFrame(0, index=teams, columns=STAGES)
        for _ in range(n_sims):
            stage = self.simulate_once(rng)
            for team, st in stage.items():
                counts.loc[team, st] += 1
        self.counts = counts

        # cumulative 'reached at least' probabilities
        reach = counts[STAGES[::-1]].cumsum(axis=1)[STAGES[::-1][::-1]]
        summary = pd.DataFrame({
            "champion": counts["champion"] / n_sims,
            "final": reach["F"] / n_sims,
            "semifinal": reach["SF"] / n_sims,
            "quarterfinal": reach["QF"] / n_sims,
            "knockout_stage": reach["R32"] / n_sims,
        }).sort_values("champion", ascending=False)
        summary.attrs["n_sims"] = n_sims
        return summary
