"""Acquire all raw datasets for the World Cup 2026 prediction project.

Usage:
    python -m src.data.download            # download everything + validate
    python -m src.data.download --check    # validate existing files only

Prerequisites:
    pip install kaggle pandas pyyaml
    Kaggle API token at ~/.kaggle/kaggle.json  (kaggle.com -> Account -> Create API Token)

Design principles:
    1. data/raw/ is immutable: we never edit downloaded files, only re-download.
    2. Every acquisition is recorded in data/raw/data_manifest.yaml with the
       source, license, retrieval timestamp, row count, and sha256 - the audit
       trail that makes published predictions reproducible.
    3. Validation runs at ingest and fails loudly. A name-mapping gap caught
       here costs one minute; caught after model training it costs a day.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

# Allow running both as a module (python -m src.data.download) and as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.normalize import (  # noqa: E402
    load_name_map,
    load_wc2026_teams,
    normalize_names,
    validate_coverage,
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "data_manifest.yaml"

# ----------------------------------------------------------------------------
# Source registry: one entry per dataset. Adding a source = adding a dict.
# ----------------------------------------------------------------------------
SOURCES = [
    {
        "name": "international_results",
        "kaggle_ref": "martj42/international-football-results-from-1872-to-2017",
        "license": "CC0: Public Domain",
        "files": ["results.csv", "shootouts.csv", "goalscorers.csv"],
        "description": "Full international matches 1872-present (title is stale; data is current).",
    },
    {
        "name": "fifa_rankings",
        "kaggle_ref": "cashncarry/fifaworldranking",
        "license": "CC0: Public Domain",
        "files": ["fifa_ranking-2024-06-20.csv"],  # filename drifts; resolved at runtime
        "description": "FIFA ranking history, Dec 1992-present, all release dates.",
    },
    {
        "name": "player_market_values",
        "kaggle_ref": "davidcariboo/player-scores",
        "license": "CC0: Public Domain",
        "files": ["players.csv"],
        "description": "Transfermarkt mirror: player market values (squad-strength proxy).",
        "optional": True,  # garnish, not backbone - failure here is a warning, not an error
    },
]


# ----------------------------------------------------------------------------
# Acquisition
# ----------------------------------------------------------------------------
def kaggle_download(ref: str, dest: Path) -> None:
    """Download and unzip a Kaggle dataset via the official CLI."""
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", ref, "-p", str(dest), "--unzip", "--force"]
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle download failed for {ref}.\n"
            f"stderr: {result.stderr.strip()}\n"
            "Check that ~/.kaggle/kaggle.json exists and `pip install kaggle` is done."
        )


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_files(source: dict, source_dir: Path) -> list[Path]:
    """Return actual CSV paths for a source. Kaggle datasets sometimes rename
    files between versions (the FIFA ranking file carries a date stamp), so
    fall back to 'all CSVs in the folder' when an expected name is missing."""
    expected = [source_dir / f for f in source["files"]]
    found = [p for p in expected if p.exists()]
    if len(found) == len(expected):
        return found
    all_csvs = sorted(source_dir.glob("*.csv"))
    if not all_csvs:
        raise FileNotFoundError(f"No CSV files found in {source_dir}")
    print(f"  note: expected filenames drifted; using {[p.name for p in all_csvs]}")
    return all_csvs


# ----------------------------------------------------------------------------
# Validation (cheap, loud, at ingest)
# ----------------------------------------------------------------------------
def validate_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "home_team", "away_team", "home_score", "away_score",
                "tournament", "neutral"}
    missing = required - set(df.columns)
    assert not missing, f"results.csv missing columns: {missing}"
    played = df.dropna(subset=["home_score", "away_score"])
    n_fixtures = len(df) - len(played)
    assert played["date"].max() <= pd.Timestamp.now() + pd.Timedelta(days=1), \
        "results.csv contains future-dated PLAYED matches (scores on unplayed games)"
    if n_fixtures:
        print(f"  note: {n_fixtures:,} scheduled fixtures without scores "
              "(prediction targets - split off by the cleaning pipeline)")
    assert (played["home_score"] >= 0).all() and (played["away_score"] >= 0).all(), \
        "Negative scores found"
    assert df["date"].min().year <= 1900, "History looks truncated (no pre-1900 matches)"
    span = (df["date"].min().date(), df["date"].max().date())
    print(f"  results.csv: {len(df):,} matches, {span[0]} to {span[1]}")

    # The canary: do all 48 qualified teams resolve after normalization?
    name_map = load_name_map()
    norm = normalize_names(played, ["home_team", "away_team"], name_map)
    unmatched = validate_coverage(norm)
    if unmatched:
        raise AssertionError(
            f"2026 teams with NO recent matches after normalization: {unmatched}. "
            "Almost certainly a name-map gap - inspect raw spellings with:\n"
            "  df[df.home_team.str.contains('<fragment>', case=False)]"
        )
    print("  name-map coverage: all 48 qualified teams resolve OK")
    return df


def validate_fifa_rankings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = next((c for c in ("rank_date", "date") if c in df.columns), None)
    assert date_col, f"No date column found in {path.name}; columns: {list(df.columns)}"
    df[date_col] = pd.to_datetime(df[date_col])
    ranks = pd.to_numeric(df["rank"], errors="coerce")
    n_bad = int((~ranks.between(1, 250)).sum())
    assert n_bad / len(df) < 0.01, \
        f"{n_bad} rows with missing/implausible rank - more than 1% of file, investigate"
    if n_bad:
        print(f"  note: {n_bad} rows with missing/implausible rank tolerated (<1%)")
    assert df[date_col].min().year <= 1993, "Ranking history looks truncated"
    print(f"  {path.name}: {len(df):,} rows, "
          f"{df[date_col].min().date()} to {df[date_col].max().date()}")
    return df


def validate_players(path: Path) -> None:
    df = pd.read_csv(path, usecols=lambda c: c in
                     {"name", "market_value_in_eur", "country_of_citizenship"})
    assert "market_value_in_eur" in df.columns, "players.csv missing market values"
    print(f"  {path.name}: {len(df):,} players")


VALIDATORS = {
    "results.csv": validate_results,
    "players.csv": validate_players,
}


def validate_source(source: dict, files: list[Path]) -> None:
    for path in files:
        if path.name in VALIDATORS:
            VALIDATORS[path.name](path)
        elif source["name"] == "fifa_rankings" and path.suffix == ".csv":
            validate_fifa_rankings(path)


# ----------------------------------------------------------------------------
# Manifest
# ----------------------------------------------------------------------------
def manifest_entry(source: dict, files: list[Path]) -> dict:
    return {
        "name": source["name"],
        "kaggle_ref": source["kaggle_ref"],
        "license": source["license"],
        "description": source["description"],
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": [
            {
                "filename": p.name,
                "rows": sum(1 for _ in open(p, "rb")) - 1,
                "sha256": sha256_of(p),
            }
            for p in files
        ],
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(check_only: bool = False) -> int:
    print(f"Project root: {PROJECT_ROOT}")
    load_wc2026_teams()  # fail fast if the 48-team config itself is broken
    print("wc2026_teams.csv: 48 teams, 12 groups of 4 - OK\n")

    manifest = []
    failures = []

    for source in SOURCES:
        source_dir = RAW_DIR / source["name"]
        print(f"[{source['name']}] {source['description']}")
        try:
            if not check_only:
                kaggle_download(source["kaggle_ref"], source_dir)
            files = resolve_files(source, source_dir)
            validate_source(source, files)
            manifest.append(manifest_entry(source, files))
        except Exception as exc:  # noqa: BLE001 - we want a full report, not first-failure
            if source.get("optional"):
                print(f"  WARNING (optional source skipped): {exc}")
            else:
                print(f"  ERROR: {exc}")
                failures.append(source["name"])
        print()

    if manifest:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MANIFEST_PATH, "w") as f:
            yaml.safe_dump({"generated_utc": datetime.now(timezone.utc).isoformat(),
                            "sources": manifest}, f, sort_keys=False)
        print(f"Manifest written: {MANIFEST_PATH}")

    if failures:
        print(f"\nFAILED sources: {failures}")
        return 1
    print("\nAll required sources acquired and validated.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Validate already-downloaded files without re-downloading")
    args = parser.parse_args()
    sys.exit(main(check_only=args.check))
