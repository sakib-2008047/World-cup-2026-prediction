"""Data cleaning pipeline: data/raw -> data/processed.

Usage:
    python -m src.data.clean                      # full pipeline
    python -m src.data.clean --raw path/to.csv    # clean a specific file

Architecture: a linear sequence of small, named, individually-testable steps.
Each step takes a DataFrame and the shared CleaningReport, returns a DataFrame,
and records what it changed. The report is written next to the output so every
processed dataset carries a record of exactly how it was produced.

Step order matters and is deliberate:
    1. parse/repair dtypes        (everything downstream needs correct types)
    2. trim whitespace            (before any string comparison)
    3. standardize team names     (before dedup - duplicates may differ only in spelling)
    4. handle missing values      (split played vs unplayed before integer casting)
    5. remove duplicates          (after names are canonical)
    6. assign team identifiers    (after names are final)
    7. validate                   (last - the gate before anything is written)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.normalize import load_name_map, load_wc2026_teams, normalize_names  # noqa: E402

RAW_RESULTS = PROJECT_ROOT / "data" / "raw" / "international_results" / "results.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

log = logging.getLogger("clean")

TEAM_COLS = ["home_team", "away_team"]
SCORE_COLS = ["home_score", "away_score"]


# ----------------------------------------------------------------------------
# Cleaning report: the pipeline's flight recorder
# ----------------------------------------------------------------------------
@dataclass
class CleaningReport:
    """Accumulates what each step did. Written as JSON beside the output file
    so 'how was this dataset produced?' always has an answer."""

    input_path: str = ""
    input_rows: int = 0
    steps: list[dict] = field(default_factory=list)
    output_rows: int = 0
    warnings: list[str] = field(default_factory=list)

    def record(self, step: str, before: int, after: int, detail: str = "") -> None:
        entry = {"step": step, "rows_before": before, "rows_after": after,
                 "rows_removed": before - after, "detail": detail}
        self.steps.append(entry)
        log.info("%-28s %7d -> %7d  (%+d)  %s",
                 step, before, after, after - before, detail)

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        log.warning(message)

    def to_json(self) -> str:
        return json.dumps(
            {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             **self.__dict__},
            indent=2,
        )


# ----------------------------------------------------------------------------
# Step 1 - dtypes: parse dates, coerce scores to nullable integers
# ----------------------------------------------------------------------------
def parse_dtypes(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    before = len(df)
    df = df.copy()

    # errors="coerce": an unparseable date becomes NaT instead of crashing the
    # pipeline; rows with NaT are dropped *visibly* and counted in the report.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = df["date"].isna().sum()
    if bad_dates:
        report.warn(f"{bad_dates} rows had unparseable dates and were dropped")
        df = df.dropna(subset=["date"])

    # Scores: use pandas' nullable Int64, not float64. A 2-1 result stored as
    # 2.0-1.0 invites equality bugs; Int64 keeps integers AND tolerates NaN
    # for matches without a result (fixtures, abandonments).
    for col in SCORE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # 'neutral' arrives as bool, "TRUE"/"FALSE" strings, or 0/1 depending on
    # the CSV reader's mood. Normalize through a string round-trip.
    df["neutral"] = (
        df["neutral"].astype(str).str.strip().str.lower()
        .map({"true": True, "1": True, "false": False, "0": False})
        .fillna(False)
        .astype(bool)
    )

    report.record("parse_dtypes", before, len(df),
                  f"dropped {bad_dates} unparseable dates")
    return df


# ----------------------------------------------------------------------------
# Step 2 - whitespace and invisible-character hygiene
# ----------------------------------------------------------------------------
def trim_strings(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """' Brazil' != 'Brazil'. Trailing spaces, double spaces, non-breaking
    spaces (\\xa0, common in scraped data) all silently break joins. Fix them
    once, here, before any string is compared to anything."""
    before = len(df)
    df = df.copy()
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("\xa0", " ", regex=False)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
    report.record("trim_strings", before, len(df), f"{len(str_cols)} text columns")
    return df


# ----------------------------------------------------------------------------
# Step 3 - standardize country/team names (delegates to the name map)
# ----------------------------------------------------------------------------
def standardize_names(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    before = len(df)
    name_map = load_name_map()
    original = df[TEAM_COLS].copy()
    df = normalize_names(df, TEAM_COLS, name_map)
    changed = int((df[TEAM_COLS] != original).any(axis=1).sum())
    report.record("standardize_names", before, len(df),
                  f"{changed} rows had at least one name remapped")
    return df


# ----------------------------------------------------------------------------
# Step 4 - missing values: split, don't impute
# ----------------------------------------------------------------------------
def handle_missing(
    df: pd.DataFrame, report: CleaningReport
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Policy: NEVER impute scores. A match without a score is not a data
    error to be filled in - it is either a future fixture (useful! that's
    what we predict) or an abandoned/annulled match (drop). We split:

        played   -> matches with both scores  -> training data
        fixtures -> matches dated in the future, no score -> prediction targets

    Past-dated matches with missing scores are dropped and counted.
    Categorical gaps (city/country) become explicit 'unknown' rather than NaN,
    because NaN in a groupby silently vanishes rows.
    """
    before = len(df)
    has_score = df[SCORE_COLS].notna().all(axis=1)
    is_future = df["date"] >= pd.Timestamp.now().normalize()

    played = df[has_score].copy()
    fixtures = df[~has_score & is_future].copy()
    dropped = before - len(played) - len(fixtures)

    for col in ("city", "country"):
        if col in played.columns:
            n = played[col].isin(["nan", "None", ""]).sum() + played[col].isna().sum()
            if n:
                played[col] = played[col].replace(["nan", "None", ""], pd.NA).fillna("unknown")
                report.warn(f"{n} missing '{col}' values set to 'unknown'")

    report.record("handle_missing", before, len(played),
                  f"{len(fixtures)} future fixtures split off; "
                  f"{dropped} past matches without scores dropped")
    return played, fixtures


# ----------------------------------------------------------------------------
# Step 5 - duplicates: exact first, then logical
# ----------------------------------------------------------------------------
def remove_duplicates(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Two passes.

    Pass 1, exact: identical rows across all columns - classic double-append
    from re-running a scraper. drop_duplicates(keep='first').

    Pass 2, logical: same (date, home_team, away_team) but differing in some
    other column (e.g. one row says 'Friendly', the other names the actual
    cup). Two international teams cannot play each other twice on the same
    day, so these are the same match recorded twice. We keep the row with the
    most non-null information, preferring a named tournament over 'Friendly'.

    This pass runs AFTER name standardization on purpose: 'USA vs Mexico' and
    'United States vs Mexico' on the same date only collide once both have
    become 'United States'.
    """
    before = len(df)
    df = df.drop_duplicates(keep="first")
    exact_removed = before - len(df)

    key = ["date", "home_team", "away_team"]
    df = df.copy()
    df["_completeness"] = df.notna().sum(axis=1) + (df["tournament"] != "Friendly")
    df = (
        df.sort_values("_completeness", ascending=False)
        .drop_duplicates(subset=key, keep="first")
        .drop(columns="_completeness")
        .sort_values("date")
        .reset_index(drop=True)
    )
    logical_removed = before - exact_removed - len(df)

    report.record("remove_duplicates", before, len(df),
                  f"{exact_removed} exact, {logical_removed} logical (same date+teams)")
    return df


# ----------------------------------------------------------------------------
# Step 6 - stable team identifiers
# ----------------------------------------------------------------------------
def team_slug(name: str) -> str:
    """Deterministic ASCII identifier: 'Curaçao' -> 'curacao',
    'Bosnia and Herzegovina' -> 'bosnia_and_herzegovina'.

    Why slugs and not integers? An integer ID's meaning depends on the
    enumeration order of a particular dataframe - rerun with one extra team
    and every ID shifts. A slug is a pure function of the canonical name:
    stable across runs, machines, and dataset versions, and human-readable
    in logs and bug reports.
    """
    ascii_name = (
        unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    )
    return "".join(c if c.isalnum() else "_" for c in ascii_name.lower()).strip("_")


def assign_team_ids(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    before = len(df)
    df = df.copy()
    for col in TEAM_COLS:
        df[col.replace("_team", "_team_id")] = df[col].map(team_slug)
    report.record("assign_team_ids", before, len(df),
                  f"{df['home_team_id'].nunique() + 0:,} distinct home ids")
    return df


# ----------------------------------------------------------------------------
# Step 7 - validation gate
# ----------------------------------------------------------------------------
class ValidationError(AssertionError):
    pass


def validate(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Hard invariants. Any failure aborts the pipeline - nothing is written.
    Cheap to run, and each check exists because its violation has a known
    failure mode downstream (noted inline)."""
    checks: list[tuple[str, bool]] = [
        # Chronology: the Elo engine and time-based splits assume sorted dates.
        ("dates sorted ascending", df["date"].is_monotonic_increasing),
        ("no future-dated played matches",
         bool((df["date"] <= pd.Timestamp.now()).all())),
        # Score sanity: negative scores poison goal-based features.
        ("scores non-negative",
         bool((df[SCORE_COLS] >= 0).all().all())),
        # 31-0 (Australia-American Samoa 2001) is real; 100+ is corruption.
        ("scores below 40", bool((df[SCORE_COLS] < 40).all().all())),
        # Self-play: a join bug signature, not a football result.
        ("no team plays itself", bool((df["home_team"] != df["away_team"]).all())),
        # Key completeness: NaN in any key column breaks groupbys silently.
        ("no nulls in key columns",
         bool(df[["date", *TEAM_COLS, *SCORE_COLS, "tournament"]].notna().all().all())),
        # Uniqueness: the dedup invariant, restated as a gate.
        ("(date, home, away) unique",
         bool(~df.duplicated(["date", "home_team", "away_team"]).any())),
        # ID bijection: each canonical name <-> exactly one slug.
        ("team ids bijective",
         df.groupby("home_team")["home_team_id"].nunique().le(1).all()
         and df.groupby("home_team_id")["home_team"].nunique().le(1).all()),
    ]
    failed = [name for name, ok in checks if not ok]
    if failed:
        raise ValidationError(f"Validation failed: {failed}")

    # Soft checks: suspicious but not fatal - logged for human review.
    teams_2026 = set(load_wc2026_teams()["team"])
    recent = df[df["date"] >= df["date"].max() - pd.Timedelta(days=730)]
    seen = set(recent["home_team"]) | set(recent["away_team"])
    missing = sorted(teams_2026 - seen)
    if missing:
        report.warn(f"2026 teams with no matches in final 2 years of data: {missing}")

    report.record("validate", len(df), len(df), f"{len(checks)} hard checks passed")
    return df


# ----------------------------------------------------------------------------
# Pipeline driver
# ----------------------------------------------------------------------------
def clean_results(raw_path: Path = RAW_RESULTS) -> tuple[pd.DataFrame, pd.DataFrame]:
    report = CleaningReport(input_path=str(raw_path))
    df = pd.read_csv(raw_path)
    report.input_rows = len(df)
    log.info("Loaded %s: %d rows, %d columns", raw_path.name, len(df), df.shape[1])

    df = parse_dtypes(df, report)
    df = trim_strings(df, report)
    df = standardize_names(df, report)
    df, fixtures = handle_missing(df, report)
    df = remove_duplicates(df, report)
    df = assign_team_ids(df, report)
    df = validate(df, report)
    report.output_rows = len(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "matches_clean.parquet"
    df.to_parquet(out_path, index=False)
    if len(fixtures):
        fixtures = assign_team_ids(fixtures, report)
        fixtures.to_parquet(PROCESSED_DIR / "fixtures.parquet", index=False)

    report_path = PROCESSED_DIR / "cleaning_report.json"
    report_path.write_text(report.to_json())
    log.info("Wrote %s (%d rows) + %s", out_path.name, len(df), report_path.name)
    return df, fixtures


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_RESULTS)
    args = parser.parse_args()
    try:
        clean_results(args.raw)
    except (ValidationError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
