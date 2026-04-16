import pandas as pd
from src.plotting import plot_elo_over_time

# Big 6: "Man United","Chelsea","Arsenal", "Man City", "Liverpool", "Tottenham"

def main():
    elo_long = pd.read_csv("data/elo_long.csv", parse_dates=["Date"])

    plot_df = filter_elo_data(elo_long, start_season=1993, end_season=2024)

    plot_elo_over_time(
        elo_long=plot_df,
        highlight_teams=["Arsenal","Chelsea","Liverpool","Man City", "Man United", "Tottenham"],
        smooth_games=5,
        resample_rule="MS",  # Options: "MS" monthly, "W" weekly
        show_all_teams_faint=True,
        title="The Big 6 All Time Premier League Elo",
        outpath="figures/big6_alltime_pl_elo.png",
    )

    print("[OK] Wrote figures/big6_alltime_pl_elo.png")

def filter_elo_data(
    elo_long: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    start_season: int | None = None,
    end_season: int | None = None,
) -> pd.DataFrame:
    df = elo_long.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["SeasonStart"] = df["SeasonStart"].astype(int)

    if start_date is not None:
        df = df[df["Date"] >= pd.Timestamp(start_date)]

    if end_date is not None:
        df = df[df["Date"] <= pd.Timestamp(end_date)]

    if start_season is not None:
        df = df[df["SeasonStart"] >= start_season]

    if end_season is not None:
        df = df[df["SeasonStart"] <= end_season]

    return df

if __name__ == "__main__":
    main()