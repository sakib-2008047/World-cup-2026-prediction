"""Team name normalization.

All raw team names from every source pass through `normalize_names` BEFORE
any join. The mapping lives in data/external/team_name_map.csv so it can be
extended without code changes. Names not in the map pass through unchanged
(they are assumed already canonical).
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NAME_MAP_PATH = PROJECT_ROOT / "data" / "external" / "team_name_map.csv"
WC2026_TEAMS_PATH = PROJECT_ROOT / "data" / "external" / "wc2026_teams.csv"


def _clean(s: str) -> str:
    """Light cleanup applied before map lookup: trim and collapse whitespace,
    normalize unicode to NFC so 'Curaçao' compares equal regardless of how
    the accent was encoded."""
    s = unicodedata.normalize("NFC", str(s)).strip()
    return " ".join(s.split())


def load_name_map(path: Path = NAME_MAP_PATH) -> dict[str, str]:
    df = pd.read_csv(path, comment="#")
    df["raw_name"] = df["raw_name"].map(_clean)
    df["canonical_name"] = df["canonical_name"].map(_clean)
    dupes = df[df.duplicated("raw_name", keep=False)]
    if not dupes.empty:
        raise ValueError(
            f"Duplicate raw_name entries in name map: {sorted(dupes['raw_name'].unique())}"
        )
    return dict(zip(df["raw_name"], df["canonical_name"]))


def load_wc2026_teams(path: Path = WC2026_TEAMS_PATH) -> pd.DataFrame:
    teams = pd.read_csv(path)
    assert len(teams) == 48, f"Expected 48 teams, got {len(teams)}"
    assert teams["team"].is_unique, "Duplicate team in wc2026_teams.csv"
    counts = teams.groupby("group").size()
    assert (counts == 4).all(), f"Every group must have 4 teams:\n{counts}"
    return teams


def normalize_names(
    df: pd.DataFrame, columns: list[str], name_map: dict[str, str] | None = None
) -> pd.DataFrame:
    """Return a copy of df with team-name columns mapped to canonical names."""
    if name_map is None:
        name_map = load_name_map()
    out = df.copy()
    for col in columns:
        cleaned = out[col].map(_clean)
        out[col] = cleaned.map(lambda x: name_map.get(x, x))
    return out


def validate_coverage(results: pd.DataFrame, since: str = "2022-01-01") -> list[str]:
    """Return any 2026 World Cup teams that do NOT appear in recent normalized
    results - the canary for name-mapping gaps. Empty list == all good."""
    teams_2026 = set(load_wc2026_teams()["team"])
    recent = results[results["date"] >= pd.Timestamp(since)]
    seen = set(recent["home_team"]) | set(recent["away_team"])
    return sorted(teams_2026 - seen)
