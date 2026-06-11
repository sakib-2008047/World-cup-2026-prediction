"""Tests for team name normalization - the layer most likely to silently break joins."""

import unicodedata

import pandas as pd
import pytest

from src.data.normalize import (
    load_name_map,
    load_wc2026_teams,
    normalize_names,
    validate_coverage,
)


@pytest.fixture(scope="module")
def name_map():
    return load_name_map()


def test_wc2026_structure():
    teams = load_wc2026_teams()
    assert len(teams) == 48
    assert teams.groupby("group").size().eq(4).all()
    assert set(teams["confederation"]) == {
        "UEFA", "CONMEBOL", "CONCACAF", "CAF", "AFC", "OFC"
    }


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("Korea Republic", "South Korea"),
        ("USA", "United States"),
        ("U.S.", "United States"),
        ("IR Iran", "Iran"),
        ("Côte d'Ivoire", "Ivory Coast"),
        ("Cabo Verde", "Cape Verde"),
        ("Türkiye", "Turkey"),
        ("Czechia", "Czech Republic"),
        ("Congo (DRC)", "DR Congo"),
        ("Zaire", "DR Congo"),
        ("West Germany", "Germany"),
        ("Bosnia & Herzegovina", "Bosnia and Herzegovina"),
    ],
)
def test_known_variants(name_map, raw, canonical):
    assert name_map[raw] == canonical


def test_unicode_nfd_accent_normalizes(name_map):
    """'Curaçao' typed with a combining cedilla (NFD) must still resolve."""
    nfd = unicodedata.normalize("NFD", "Curaçao")
    df = pd.DataFrame({"home_team": [nfd]})
    out = normalize_names(df, ["home_team"], name_map)
    assert out.loc[0, "home_team"] == "Curaçao"


def test_unknown_names_pass_through(name_map):
    df = pd.DataFrame({"home_team": ["Brazil"]})
    out = normalize_names(df, ["home_team"], name_map)
    assert out.loc[0, "home_team"] == "Brazil"


def test_coverage_validator_flags_missing_team():
    teams = set(load_wc2026_teams()["team"])
    present = sorted(teams - {"Jordan"})
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-06-01"] * len(present)),
            "home_team": present,
            "away_team": ["Brazil"] * len(present),
        }
    )
    assert validate_coverage(df) == ["Jordan"]
