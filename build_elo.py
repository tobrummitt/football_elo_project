from src.ingest import load_matches
from src.elo import save_elo_long_csv, EloConfig

def main():
    matches = load_matches()
    cfg = EloConfig(initial_elo=1500, k=20, home_advantage=60)

    save_elo_long_csv(matches, outpath="data/elo_long.csv", config=cfg)

    print("Saved data/elo_long.csv")
    print("Matches loaded:", len(matches))

if __name__ == "__main__":
    main()