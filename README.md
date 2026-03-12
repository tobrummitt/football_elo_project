# Premier League Elo Ratings (with Rest Days extension)

This project builds an Elo rating system for Premier League teams using match results and produces:
- Match-level Elo with pre- and post-match ratings
- A team-match long dataset (`data/elo_long.csv`) suitable for plotting and feature engineering
- A clean dark-theme visualization with season shading

## How to run

1) Put your football-data.co.uk CSVs in `raw_data/`

2) Build Elo dataset:

```bash
python build_elo.py
