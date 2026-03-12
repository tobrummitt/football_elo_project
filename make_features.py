import pandas as pd
from src.features import add_rest_days_team_level


def main():
    elo_long = pd.read_csv("data/elo_long.csv", parse_dates=["Date"])
    elo_long = add_rest_days_team_level(elo_long)

    elo_long.to_csv("data/elo_long_with_rest.csv", index=False)
    print("[OK] Wrote data/elo_long_with_rest.csv")


if __name__ == "__main__":
    main()