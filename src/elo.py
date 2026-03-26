from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class EloConfig:
    initial_elo: float = 1500.0
    k: float = 20.0
    home_advantage: float = 60.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Classic Elo expected score."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def run_elo(matches: pd.DataFrame, config: EloConfig = EloConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Elo sequentially over matches and return:
      1) elo_df: match-level table with pre/post Elo + expected probs
      2) elo_long: team-match long table (2 rows per match) with pre/post Elo
    Expects matches columns:
      Date, HomeTeam, AwayTeam, HomeGoals, AwayGoals
    """
    # Returns error if the matches dataframe does not contain the correct columns
    required = {"Date", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"matches is missing required columns: {sorted(missing)}")
    

    # Chronologically order matches dataframe
    df = matches.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Initialise team elos
    teams = pd.concat([df["HomeTeam"], df["AwayTeam"]]).dropna().unique()
    elo: dict[str, float] = {str(t): config.initial_elo for t in teams}

    rows: list[dict] = []

    # Using itertuples for speed
    for row in df.itertuples(index=False):
        date = row.Date
        home, away = str(row.HomeTeam), str(row.AwayTeam)
        hg, ag = float(row.HomeGoals), float(row.AwayGoals)

        # Pre match elo ratings
        home_elo_pre = elo.get(home, config.initial_elo)
        away_elo_pre = elo.get(away, config.initial_elo)

        # Calculate probabilities
        exp_home = expected_score(home_elo_pre + config.home_advantage, away_elo_pre)
        exp_away = 1.0 - exp_home

        # Actual outcome (home perspective)
        if hg > ag:
            act_home, act_away = 1.0, 0.0
        elif hg < ag:
            act_home, act_away = 0.0, 1.0
        else:
            act_home, act_away = 0.5, 0.5

        # Post-match updates
        home_elo_post = home_elo_pre + config.k * (act_home - exp_home)
        away_elo_post = away_elo_pre + config.k * (act_away - exp_away)

        # Persist ratings
        elo[home] = home_elo_post
        elo[away] = away_elo_post

        rows.append({
            "Date": date,
            "HomeTeam": home,
            "AwayTeam": away,
            "HomeGoals": hg,
            "AwayGoals": ag,
            "HomeEloPre": home_elo_pre,
            "AwayEloPre": away_elo_pre,
            "HomeEloPost": home_elo_post,
            "AwayEloPost": away_elo_post,
            "ExpectedHomeWinProb": exp_home,
            "ExpectedAwayWinProb": exp_away,
        })

    elo_df = pd.DataFrame(rows)

    # Build long table (2 rows per match)
    home_long = elo_df[[
        "Date", "HomeTeam", "AwayTeam",
        "HomeGoals", "AwayGoals",
        "HomeEloPre", "HomeEloPost",
        "ExpectedHomeWinProb"
    ]].rename(columns={
        "HomeTeam": "Team",
        "AwayTeam": "Opponent",
        "HomeGoals": "GoalsFor",
        "AwayGoals": "GoalsAgainst",
        "HomeEloPre": "EloPre",
        "HomeEloPost": "EloPost",
        "ExpectedHomeWinProb": "ExpWinProb",
    })
    home_long["IsHome"] = 1

    away_long = elo_df[[
        "Date", "AwayTeam", "HomeTeam",
        "AwayGoals", "HomeGoals",
        "AwayEloPre", "AwayEloPost",
        "ExpectedAwayWinProb"
    ]].rename(columns={
        "AwayTeam": "Team",
        "HomeTeam": "Opponent",
        "AwayGoals": "GoalsFor",
        "HomeGoals": "GoalsAgainst",
        "AwayEloPre": "EloPre",
        "AwayEloPost": "EloPost",
        "ExpectedAwayWinProb": "ExpWinProb",
    })
    away_long["IsHome"] = 0

    elo_long = (
        pd.concat([home_long, away_long], ignore_index=True)
          .sort_values(["Team", "Date"])
          .reset_index(drop=True)
    )

    return elo_df, elo_long
