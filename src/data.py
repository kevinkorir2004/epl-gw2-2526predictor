
import argparse
import os
from pathlib import Path
from typing import List
import pandas as pd
import requests
from tqdm import tqdm
import yaml

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

FD_BASE = "https://www.football-data.co.uk"
FD_MMZ = FD_BASE + "/mmz4281/{season}/{league}.csv"
FD_CURRENT = FD_BASE + "/new/{league}.csv"

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def season_code(year: int) -> str:
    return f"{year % 100:02d}{(year + 1) % 100:02d}"

def download_season(league_code: str, season_year: int):
    season = season_code(season_year)
    url = FD_MMZ.format(season=season, league=league_code)
    out = RAW_DIR / f"{league_code}_{season}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and len(r.content) > 1000:
            out.write_bytes(r.content)
            return out
        alt = FD_CURRENT.format(league=league_code)
        r2 = requests.get(alt, timeout=30)
        if r2.status_code == 200 and len(r2.content) > 1000:
            out.write_bytes(r2.content)
            return out
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    return None

def download_all(league_code: str, seasons: List[int]):
    files = []
    for y in tqdm(seasons, desc="Downloading seasons"):
        p = download_season(league_code, y)
        if p:
            files.append(p)
    return files

def load_all_raw() -> pd.DataFrame:
    files = list(RAW_DIR.glob("*.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["src_file"] = f.name
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No raw CSVs found. Run with --download first.")
    return pd.concat(dfs, ignore_index=True)

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "Date": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_goals",
        "FTAG": "away_goals",
        "FTR": "result",
        "HS": "home_shots",
        "AS": "away_shots",
        "HST": "home_shots_on",
        "AST": "away_shots_on",
        "AvgH": "odds_home_avg",
        "AvgD": "odds_draw_avg",
        "AvgA": "odds_away_avg",
    }
    out = {}
    for k, v in cols.items():
        out[v] = df[k] if k in df.columns else None
    out = pd.DataFrame(out)
    out["date"] = pd.to_datetime(out["date"], dayfirst=True, errors="coerce")
    out = out.dropna(subset=["date", "home_team", "away_team"])
    if "src_file" in df.columns:
        out["season"] = df["src_file"].str.extract(r"(\d{4})").squeeze()
    from src.utils import clean_team_name
    out["home_team"] = out["home_team"].apply(clean_team_name)
    out["away_team"] = out["away_team"].apply(clean_team_name)
    for c in ["home_goals","away_goals","home_shots","away_shots","home_shots_on","away_shots_on"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    if args.download:
        download_all(cfg["data"]["league_code"], cfg["data"]["seasons"] + [cfg["predict"]["season"]])
    raw_df = load_all_raw()
    norm = normalize(raw_df).sort_values("date").reset_index(drop=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    norm.to_csv(PROCESSED_DIR / "matches_normalized.csv", index=False)
    print(f"Saved {PROCESSED_DIR / 'matches_normalized.csv'} with {len(norm)} rows.")

if __name__ == "__main__":
    main()
