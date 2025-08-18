
import argparse, os, requests
from pathlib import Path
import pandas as pd
import joblib
from dotenv import load_dotenv
from src.features import PROCESSED_DIR
from src.utils import clean_team_name

PRED_DIR = Path("predictions")
PRED_DIR.mkdir(parents=True, exist_ok=True)

def fetch_fixtures_football_data_org(season: int, matchday: int):
    load_dotenv()
    token = os.getenv("FOOTBALL_DATA_API_TOKEN")
    if not token:
        raise RuntimeError("FOOTBALL_DATA_API_TOKEN not set")
    url = f"https://api.football-data.org/v4/competitions/PL/matches?season={season}&matchday={matchday}"
    headers = {"X-Auth-Token": token}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = []
    for m in data.get("matches", []):
        status = m.get("status")
        if status in {"SCHEDULED", "TIMED"}:
            rows.append({
                "matchday": matchday,
                "date": m.get("utcDate"),
                "home_team": clean_team_name(m["homeTeam"]["name"]),
                "away_team": clean_team_name(m["awayTeam"]["name"]),
            })
    return pd.DataFrame(rows)

def build_prediction_table(fixtures: pd.DataFrame, features_df: pd.DataFrame, model_bundle):
    last = features_df.sort_values("date").copy()
    home_last = last.groupby("home_team")[["elo_home_pre","home_gf","home_ga"]].last().rename(columns={
        "elo_home_pre":"elo_pre", "home_gf":"gf", "home_ga":"ga"
    })
    away_last = last.groupby("away_team")[["elo_away_pre","away_gf","away_ga"]].last().rename(columns={
        "elo_away_pre":"elo_pre", "away_gf":"gf", "away_ga":"ga"
    })
    latest = home_last.combine_first(away_last)
    rows = []
    for _, row in fixtures.iterrows():
        h, a = row["home_team"], row["away_team"]
        median_vals = latest.median(numeric_only=True).to_dict()
        h_stats = latest.loc[h].to_dict() if h in latest.index else median_vals
        a_stats = latest.loc[a].to_dict() if a in latest.index else median_vals
        feat = {
            "elo_home_pre": h_stats.get("elo_pre", 1500.0),
            "elo_away_pre": a_stats.get("elo_pre", 1500.0),
            "home_gf": h_stats.get("gf", 1.2),
            "home_ga": h_stats.get("ga", 1.2),
            "away_gf": a_stats.get("gf", 1.2),
            "away_ga": a_stats.get("ga", 1.2),
        }
        feat["elo_diff"] = feat["elo_home_pre"] - feat["elo_away_pre"]
        feat["gf_diff"] = feat["home_gf"] - feat["away_gf"]
        feat["ga_diff"] = feat["away_ga"] - feat["home_ga"]
        rows.append({**row.to_dict(), **feat})
    pred_df = pd.DataFrame(rows)
    pipe = model_bundle["pipeline"]
    X = pred_df[model_bundle["feature_cols"]]
    proba = pipe.predict_proba(X)
    pred_df["p_home"] = proba[:,0]
    pred_df["p_draw"] = proba[:,1]
    pred_df["p_away"] = proba[:,2]
    pred_df["prediction"] = pred_df[["p_home","p_draw","p_away"]].idxmax(axis=1).str.replace("p_","").str.upper()
    return pred_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--matchday", type=int, default=2)
    parser.add_argument("--fixtures_csv", type=str, default=None)
    parser.add_argument("--out", type=str, default="predictions/predictions.csv")
    args = parser.parse_args()

    if args.fixtures_csv:
        fixtures = pd.read_csv(args.fixtures_csv)
    else:
        fixtures = fetch_fixtures_football_data_org(args.season, args.matchday)

    features_df = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])
    model_bundle = joblib.load("models/model.joblib")

    pred_df = build_prediction_table(fixtures, features_df, model_bundle)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")
