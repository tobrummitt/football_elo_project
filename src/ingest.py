from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass(frozen=True)
class IngestConfig:
    raw_glob: str = "raw_data/*.csv"
    dayfirst: bool = True
    column_aliases: dict[str, str] = None

def _default_aliases() -> dict[str, str]:
    """
    Aliases to map various file schemas onto a standard schema.
    Football-data.co.uk typically uses: Date, HomeTeam, AwayTeam, FTHG, FTAG.
    """
    return {
        # Dates
        "Date": "Date",
        "MatchDate": "Date",

        # Teams
        "HomeTeam": "HomeTeam",
        "AwayTeam": "AwayTeam",
        "Home": "HomeTeam",
        "Away": "AwayTeam",

        # Goals (football-data)
        "FTHG": "HomeGoals",
        "FTAG": "AwayGoals",

        # Other common abbreviations
        "HG": "HomeGoals",
        "AG": "AwayGoals",
        "HomeGoals": "HomeGoals",
        "AwayGoals": "AwayGoals",
    }


def load_matches(config: IngestConfig = IngestConfig()) -> pd.DataFrame:
    """
    Load and clean all raw match CSVs into a single standard DataFrame.

    Returns a DataFrame with columns:
      Date (datetime64), HomeTeam (str), AwayTeam (str), HomeGoals (float), AwayGoals (float)

    Raises:
      FileNotFoundError if no files match.
      ValueError if no valid rows are loaded.
    """
    aliases = config.column_aliases or _default_aliases()

    files = sorted(Path().glob(config.raw_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {config.raw_glob}")

    required = {"Date", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"}
    loaded: list[pd.DataFrame] = []
    skipped: list[tuple[str, list[str]]] = []

    total_files = len(files)
    total_rows_read = 0
    total_rows_kept = 0

    for fp in files:
        print(f"Reading: {fp}")

        try:
            try:
                df = pd.read_csv(
                    fp,
                    encoding="utf-8",
                    engine="python",
                    on_bad_lines="skip"
                )
            except UnicodeDecodeError:
                df = pd.read_csv(
                    fp,
                    encoding="latin1",
                    engine="python",
                    on_bad_lines="skip"
                )
        
        except Exception as e:
            print(f"[SKIP] Failed to read {fp}: {e}")
            continue
        
        original_rows = len(df)
        total_rows_read += original_rows

        df = df.rename(columns=aliases)

        missing = sorted(list(required - set(df.columns)))
        if missing:
            skipped.append((str(fp), missing))
            continue

        # Keep only what we need (extra columns are irrelevant)
        df = df[["Date", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"]].copy()

        # Parse date safely
        df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=config.dayfirst, errors="coerce")
        df["HomeGoals"] = pd.to_numeric(df["HomeGoals"], errors="coerce")
        df["AwayGoals"] = pd.to_numeric(df["AwayGoals"], errors="coerce")

        # Clean team names
        df["HomeTeam"] = df["HomeTeam"].astype(str).str.replace("\xa0", " ", regex=False).str.strip()
        df["AwayTeam"] = df["AwayTeam"].astype(str).str.replace("\xa0", " ", regex=False).str.strip()

        # Drop invalid rows
        before_drop = len(df)
        df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"])
        df = df[(df["HomeTeam"] != "") & (df["AwayTeam"] != "")]
        after_drop = len(df)
        total_rows_kept += after_drop

        print(f"Loaded {original_rows} rows, kept {after_drop} valid rows")

        if after_drop > 0:
            loaded.append(df)

        else:
            skipped.append((str(fp), "no valid rows after cleaning"))

    if skipped:
        print("[INGEST] Skipped files due to missing columns:")
        for path, miss in skipped:
            print(f"  - {path}: missing {miss}")

    if not loaded:
        raise ValueError("No valid data loaded. Check your CSV formats and column aliases.")

    matches = pd.concat(loaded, ignore_index=True)

    # Sort chronologically (Elo requires time order)
    matches = matches.sort_values("Date").reset_index(drop=True)

    # Optional: drop exact duplicate matches (can happen if you accidentally include files twice)
    matches = matches.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"])

    # INGESTION SUMMARY
    seasons = matches["Date"].dt.year.unique()
    teams = pd.concat([matches["HomeTeam"], matches["AwayTeam"]]).nunique()

    print("\n==============================")
    print("INGESTION SUMMARY")
    print("==============================")

    print(f"Files processed: {total_files}")
    print(f"Matches loaded: {len(matches)}")
    print(f"Rows read from CSVs: {total_rows_read}")
    print(f"Valid rows kept: {total_rows_kept}")
    print(f"Rows dropped during cleaning: {total_rows_read - total_rows_kept}")

    print(f"Seasons detected: {seasons.min()}–{seasons.max()}")
    print(f"Unique teams detected: {teams}")

    print("==============================\n")
    print(f"\n[INGEST] Final dataset: {len(matches)} matches")

    
    return matches
