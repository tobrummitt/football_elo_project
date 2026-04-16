from src.ingest import load_matches
from src.elo import run_elo, EloConfig
from src.evaluation import evaluate_model, save_metrics

def main():
    matches = load_matches()
    cfg = EloConfig(initial_elo=1500, k=20, home_advantage=60)

    # Build Elo outputs
    elo_df, elo_long = run_elo(matches, config=cfg)

    # Save datasets
    elo_df.to_csv("data/elo_df.csv", index=False)
    elo_long.to_csv("data/elo_long.csv", index=False)

    print("Saved data/elo_long.csv")
    print("Matches loaded:", len(matches))

    # Evaluate Model and save the metrics to a csv file
    metrics = evaluate_model(elo_df,exclude_first_n_seasons=2)
    print(metrics)

    save_metrics(metrics, model_name="baseline_cutoff_2")


if __name__ == "__main__":
    main()