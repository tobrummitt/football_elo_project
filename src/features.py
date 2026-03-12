from __future__ import annotations

import pandas as pd


def add_season(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out["Season"] = out[date_col].dt.year
    out.loc[out[date_col].dt.month < 8, "Season"] -= 1
    return out


def add_rest_days_team_level(elo_long: pd.DataFrame) -> pd.DataFrame:
    """
    Adds RestDays for each Team (days since previous match for that team).
    Expects columns: Date, Team
    """
    df = elo_long.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Team", "Date"]).reset_index(drop=True)
    df["RestDays"] = df.groupby("Team")["Date"].diff().dt.days
    return df