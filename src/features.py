
import pandas as pd
from pathlib import Path
import yaml

from src.elo import EloParams, compute_match_elos
from src.utils import outcome_to_label

PROCESSED_DIR = Path("data/processed")

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def build_features():
    cfg = load_config()
    matches = pd.read_csv(PROCESSED_DIR / "matches_normalized.csv", parse_dates=["date"])
    params = EloParams(k_factor=cfg["elo"]["k_factor"], home_advantage=cfg["elo"]["home_advantage"])
    matches = compute_match_elos(matches, params)

    window = cfg["features"]["rolling_window"]
    # rolling means per team
    matches = matches.sort_values("date").reset_index(drop=True)

    def team_rolls(df, gf_col, ga_col, prefix):
        df = df.sort_values("date")
        df[f"{prefix}_gf"] = df[gf_col].rolling(window, min_periods=1).mean()
        df[f"{prefix}_ga"] = df[ga_col].rolling(window, min_periods=1).mean()
        return df

    home_hist = matches.groupby("home_team", group_keys=False).apply(lambda g: team_rolls(g, "home_goals","away_goals","home"))
    away_hist = matches.groupby("away_team", group_keys=False).apply(lambda g: team_rolls(g, "away_goals","home_goals","away"))

    feats = matches[["date","home_team","away_team","home_goals","away_goals","result","elo_home_pre","elo_away_pre"]].copy()
    feats = feats.merge(home_hist[["date","home_team","away_team","home_gf","home_ga"]], on=["date","home_team","away_team"])
    feats = feats.merge(away_hist[["date","home_team","away_team","away_gf","away_ga"]], on=["date","home_team","away_team"])

    feats["y"] = feats["result"].apply(outcome_to_label)
    feats["elo_diff"] = feats["elo_home_pre"] - feats["elo_away_pre"]
    feats["gf_diff"] = feats["home_gf"] - feats["away_gf"]
    feats["ga_diff"] = feats["away_ga"] - feats["home_ga"]

    feats.to_csv(PROCESSED_DIR / "features.csv", index=False)
    return feats

if __name__ == "__main__":
    feats = build_features()
    print(f"Saved {PROCESSED_DIR / 'features.csv'} with {len(feats)} rows.")
