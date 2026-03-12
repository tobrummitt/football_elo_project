from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

def add_season_column(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Premier League season runs Aug–May so create a Season Start and Season column
    Example: 2022-09-01 -> Season Start: 2022, Season: 2022-2023
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])

    season_start = out[date_col].dt.year - (out[date_col].dt.month < 8)

    out["SeasonStart"] = season_start.astype("int16")
    out["Season"] = season_start.astype(str) + "-" + (season_start + 1).astype(str).str[-2:]

    return out

def add_spell_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a Spell column so lines stay connected across consecutive seasons
    and break when a team misses one or more seasons.
    """
    out = df.copy()

    # Unique team-season combinations
    team_seasons = (
        out[["Team", "SeasonStart"]]
        .drop_duplicates()
        .sort_values(["Team", "SeasonStart"])
    )

    # New spell whenever season gap > 1
    team_seasons["LeagueSpell"] = (
        team_seasons.groupby("Team")["SeasonStart"]
        .diff()
        .gt(1)
        .groupby(team_seasons["Team"])
        .cumsum()
        .astype(int)
    )

    out = out.merge(
        team_seasons[["Team", "SeasonStart", "LeagueSpell"]],
        on=["Team","SeasonStart"],
        how="left",
    )

    return out

def prep_monthly_smoothed_elo(
    elo_long: pd.DataFrame,
    smooth_games: int = 5,
    resample_rule: str = "W",  # Week start. Alternatives: "MS" month start, "M" month end
) -> pd.DataFrame:
    """
    Smooth Elo per team, resample within each team-season,
    and assign a Spell so plotting breaks across missed seasons
    but stays joined across consecutive seasons in the league.
    """
    df = elo_long[["Date", "Team", "EloPost"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Team", "Date"]).reset_index(drop=True)

    # 5-game rolling average per team (match-based smoothing)
    df["EloSmooth"] = (
        df.groupby("Team", sort=False)["EloPost"]
          .transform(lambda s: s.rolling(window=smooth_games, min_periods=1).mean())
    )

    # Add season metadata and continuous participation blocks
    df = add_season_column(df)
    df = add_spell_column(df)

    # Resample only within each continous spell
    out = (
        df.set_index("Date")
          .groupby(["Team", "LeagueSpell"])["EloSmooth"]
          .resample(resample_rule)
          .last()
          .ffill()
          .reset_index()
          .rename(columns={"EloSmooth": "Elo"})
    )

    # Re-attach season labels after resampling
    out = add_season_column(out)

    out = out.sort_values(["Team", "LeagueSpell", "Date"]).reset_index(drop=True)
    return out


def plot_elo_over_time(
    elo_long: pd.DataFrame,
    highlight_teams: list[str],
    smooth_games: int = 5,
    resample_rule: str = "W",
    show_all_teams_faint: bool = True,
    title: str = "Premier League Elo over time (smoothed)",
    outpath: str | None = "figures/elo_over_time.png",
):
    """
    Creates a clean multi-season Elo chart:
    - 5-game rolling smoothing (match-based)
    - resampled snapshots (weekly, monthly etc.)
    - alternating grey season shading
    - line breaks when teams relegated, lines reconnect across consecutive seasons
    - optional faint lines for all other teams + highlighted teams on top
    """
    df = prep_monthly_smoothed_elo(
        elo_long=elo_long,
        smooth_games=smooth_games,
        resample_rule=resample_rule,
    )

    fig, ax = plt.subplots(figsize=(13, 6))

    # Graph Theming
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_alpha(0.25)

    # --- Season shading (alternating) ---
    season_starts = df.groupby("Season")["Date"].min().sort_values()
    seasons = season_starts.index.to_list()
    start_dates = season_starts.to_list()
    last_date = df["Date"].max()

    season_midpoints = []
    for i, start in enumerate(start_dates):
        end = start_dates[i + 1] if i + 1 < len(start_dates) else last_date

        if i % 2 == 0:
            ax.axvspan(start, end, color="white", alpha=0.10)

        midpoint = start + ((end - start) / 2)
        season_midpoints.append(midpoint)

    ax.set_xticks(season_midpoints)
    ax.set_xticklabels(seasons, rotation=45, ha="right", color="white")

    ax.tick_params(axis="x", direction="out", length=5)

    highlight_set = set(highlight_teams)

    # Plot non-highlighted teams faintly, split by LeagueSpell
    if show_all_teams_faint:
        for (team, _spell), tdf in df.groupby(["Team", "LeagueSpell"], sort=False):
            if team in highlight_set:
                continue
            ax.plot(tdf["Date"], tdf["Elo"], linewidth=1, alpha=0.12, color="white")

    # Set team colours
    team_colors = {
        # add/adjust as you like
        "Arsenal": "#EF0107",
        "Chelsea": "#034694",
        "Liverpool": "#C8102E",
        "Man City": "#6CABDD",
        "Man United": "#DA291C",
        "Newcastle": "#FFFFFF",
        "Tottenham": "#132257"
    }

    # Plot highlighted teams, split by LeagueSpell
    for team in highlight_teams:
        team_df = df[df["Team"] == team]
        if team_df.empty:
            continue

        color = team_colors.get(team, None)
        first_segment = True

        for _, seg in team_df.groupby("LeagueSpell", sort=True):
            ax.plot(
                seg["Date"],
                seg["Elo"], 
                linewidth=2.6, 
                label=team if first_segment else None,
                color=color
            )
            first_segment = False

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Elo (rolling {smooth_games} matches, {resample_rule} snapshots)")
    ax.grid(axis="y", alpha=0.18, linewidth=0.8, color="white")

    leg = ax.legend(frameon=False)
    for t in leg.get_texts():
        t.set_color("white")

    fig.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())

    plt.show()
