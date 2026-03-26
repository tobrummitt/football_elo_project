import pandas as pd
import numpy as np
from pathlib import Path

def evaluate_model(df):

    df = df.copy()

    df["ActualHomeWin"] = (df["HomeGoals"] > df["AwayGoals"]).astype(int)
    df["PredHomeWin"] = (df["ExpectedHomeWinProb"] > 0.5).astype(int)

    accuracy = (df["PredHomeWin"] == df["ActualHomeWin"]).mean()

    brier = np.mean(
        (df["ExpectedHomeWinProb"] - df["ActualHomeWin"]) ** 2
    )

    return {
        "accuracy": round(accuracy, 4),
        "brier_score": round(brier, 4),
    }


def save_metrics(metrics, model_name):

    row = pd.DataFrame([{
        "model": model_name,
        **metrics
    }])

    path = Path("data/model_results.csv")

    if path.exists():
        existing = pd.read_csv(path)
        df = pd.concat([existing, row], ignore_index=True)
    else:
        df = row

    df.to_csv(path, index=False)