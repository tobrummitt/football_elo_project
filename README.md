# Premier League Elo Ratings (with Rest Days extension)

This project builds an Elo rating system for Premier League teams using match results and produces:
- Match-level Elo with pre- and post-match ratings
- A team-match long dataset (`data/elo_long.csv`) suitable for plotting and feature engineering
- A visualisation of team elos over time

## How to run

1) Put your football-data.co.uk CSVs in `raw_data/`

2) Build Elo dataset:

```bash
python build_elo.py
```
3) Produce plots (plot format edited in file):

```bash
python plot_elo.py
```

## Baseline Version
The baseline Elo model updates team ratings sequentially across Premier League matches.
Baseline Model Spec:
- Initial rating: 1500 for all teams
- K-factor: 20
- Home advantage: 60 Elo points
- Match result scoring:
    Home win = 1
    Draw = 0.5
    Home loss = 0
- Expected result:
    E = 1 / (1 + 10^(-(home_elo + home_adv - away_elo)/400))
- Rating update:
    new_elo = old_elo + K * (actual - expected)
- No margin-of-victory adjustment
- No season regression
- No promoted-team adjustment
