"""Tests for the cleaning pipeline. Each test feeds a known dirt pattern and
asserts the pipeline removes exactly that dirt and nothing else."""

import pandas as pd
import pytest

from src.data.clean import (
    CleaningReport,
    ValidationError,
    assign_team_ids,
    handle_missing,
    parse_dtypes,
    remove_duplicates,
    standardize_names,
    team_slug,
    trim_strings,
    validate,
)


def make_raw(rows: list[dict]) -> pd.DataFrame:
    """Build a raw-shaped dataframe with sensible defaults."""
    defaults = {
        "date": "2024-06-01", "home_team": "Brazil", "away_team": "Argentina",
        "home_score": 1, "away_score": 0, "tournament": "Friendly",
        "city": "Rio de Janeiro", "country": "Brazil", "neutral": False,
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


@pytest.fixture
def report():
    return CleaningReport()


# ---------------------------------------------------------------- dtypes ----
def test_unparseable_dates_dropped_and_counted(report):
    df = make_raw([{"date": "2024-06-01"}, {"date": "not a date"}])
    out = parse_dtypes(df, report)
    assert len(out) == 1
    assert report.warnings and "unparseable dates" in report.warnings[0]


def test_scores_become_nullable_int(report):
    df = make_raw([{"home_score": "2"}, {"home_score": None}])
    out = parse_dtypes(df, report)
    assert str(out["home_score"].dtype) == "Int64"
    assert out["home_score"].iloc[0] == 2 and pd.isna(out["home_score"].iloc[1])


def test_neutral_flag_variants(report):
    df = make_raw([{"neutral": "TRUE"}, {"neutral": "false"},
                   {"neutral": 1}, {"neutral": True}])
    out = parse_dtypes(df, report)
    assert out["neutral"].tolist() == [True, False, True, True]


# ------------------------------------------------------------ whitespace ----
def test_whitespace_and_nbsp_stripped(report):
    df = make_raw([{"home_team": "  Brazil\xa0", "away_team": "Argen  tina"}])
    out = trim_strings(df, report)
    assert out["home_team"].iloc[0] == "Brazil"
    assert out["away_team"].iloc[0] == "Argen tina"  # inner runs collapse to one


# ------------------------------------------------------ standardization -----
def test_names_standardized_and_counted(report):
    df = make_raw([{"home_team": "Korea Republic", "away_team": "USA"},
                   {"home_team": "Brazil"}])
    out = standardize_names(df, report)
    assert out["home_team"].iloc[0] == "South Korea"
    assert out["away_team"].iloc[0] == "United States"
    assert "1 rows" in report.steps[-1]["detail"]


# --------------------------------------------------------------- missing ----
def test_missing_split_played_fixture_dropped(report):
    future = (pd.Timestamp.now() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    df = parse_dtypes(make_raw([
        {},                                                  # played
        {"date": future, "home_score": None, "away_score": None},   # fixture
        {"date": "2010-01-01", "home_score": None, "away_score": None},  # bad row
    ]), report)
    played, fixtures = handle_missing(df, report)
    assert len(played) == 1 and len(fixtures) == 1


def test_scores_never_imputed(report):
    """The policy test: a past match with no score must vanish, not become 0-0."""
    df = parse_dtypes(make_raw(
        [{"date": "2015-05-05", "home_score": None, "away_score": None}]), report)
    played, _ = handle_missing(df, report)
    assert len(played) == 0


def test_missing_city_becomes_unknown(report):
    df = parse_dtypes(make_raw([{"city": None}]), report)
    played, _ = handle_missing(df, report)
    assert played["city"].iloc[0] == "unknown"


# ------------------------------------------------------------- duplicates ---
def test_exact_duplicates_removed(report):
    df = parse_dtypes(make_raw([{}, {}]), report)
    out = remove_duplicates(df, report)
    assert len(out) == 1


def test_logical_duplicate_keeps_named_tournament(report):
    df = parse_dtypes(make_raw([
        {"tournament": "Friendly"},
        {"tournament": "Copa América"},
    ]), report)
    out = remove_duplicates(df, report)
    assert len(out) == 1
    assert out["tournament"].iloc[0] == "Copa América"


def test_spelling_variant_duplicate_collides_after_standardization(report):
    """'USA vs Mexico' and 'United States vs Mexico' on the same date are one
    match - but only once names are canonical. Tests the step ORDER."""
    df = parse_dtypes(make_raw([
        {"home_team": "USA", "away_team": "Mexico"},
        {"home_team": "United States", "away_team": "Mexico"},
    ]), report)
    df = standardize_names(df, report)
    out = remove_duplicates(df, report)
    assert len(out) == 1


# ---------------------------------------------------------------- team ids --
@pytest.mark.parametrize(("name", "slug"), [
    ("Brazil", "brazil"),
    ("Curaçao", "curacao"),
    ("Bosnia and Herzegovina", "bosnia_and_herzegovina"),
    ("Côte d'Ivoire", "cote_d_ivoire"),  # accents fold to base letters via NFKD
    ("Ivory Coast", "ivory_coast"),
])
def test_team_slug(name, slug):
    assert team_slug(name) == slug


def test_slug_stability_is_order_independent(report):
    a = assign_team_ids(parse_dtypes(make_raw([{}]), report), report)
    b = assign_team_ids(parse_dtypes(make_raw(
        [{"home_team": "Zambia", "away_team": "Brazil", "date": "2020-01-01"}, {}]
    ), report), report)
    id_a = a.loc[a.home_team == "Brazil", "home_team_id"].iloc[0]
    id_b = b.loc[b.home_team == "Brazil", "home_team_id"].iloc[0]
    assert id_a == id_b == "brazil"


# -------------------------------------------------------------- validation --
def _valid_df(report):
    df = parse_dtypes(make_raw([
        {"date": "2024-06-01"},
        {"date": "2024-06-02", "home_team": "France", "away_team": "Spain"},
    ]), report)
    df, _ = handle_missing(df, report)
    df = remove_duplicates(df, report)
    return assign_team_ids(df, report)


def test_validate_passes_clean_data(report):
    validate(_valid_df(report), report)


def test_validate_catches_self_play(report):
    df = _valid_df(report)
    df.loc[0, "away_team"] = df.loc[0, "home_team"]
    df.loc[0, "away_team_id"] = df.loc[0, "home_team_id"]
    with pytest.raises(ValidationError, match="plays itself"):
        validate(df, report)


def test_validate_catches_unsorted_dates(report):
    df = _valid_df(report).iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValidationError, match="sorted"):
        validate(df, report)


def test_validate_catches_absurd_score(report):
    df = _valid_df(report)
    df.loc[0, "home_score"] = 105
    with pytest.raises(ValidationError, match="below 40"):
        validate(df, report)
