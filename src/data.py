# src/data.py
"""
Download & normalize real EPL match data from football-data.co.uk
Saves:
  - data/raw/E0_YYYY.csv (one per season)
  - data/processed/matches_normalized.csv (combined, tidy format)
Run:
  python src/data.py --download
Optional:
  python src/data.py --seasons 2018,2019,2020,2021,2022,2023,2024,2025
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Optional
import pandas as pd
import requests
from tqdm import tqdm

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")

LEAGUE = "E0"  # Premier League on football-data.co.uk
FD_BASE = "https://www.football-data.co.uk"
FD_URL = FD_BASE + "/mmz4281/{season}/{league}.csv"  # season like 2425 for 2024/25

DEFAULT_SEASONS = list(range(2015, 2026))  # 2015/16 ... 2025/26


def season_code(year: int) -> str:
    """Convert start year to football-data season code: 2024 -> '2425'."""
    return f"{year % 100:02d}{(year + 1) % 100:02d}"


def clean_team_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    name = name.strip()
    replacements = {
        "Man United": "Manchester United",
        "Man Utd": "Manchester United",
        "Manchester Utd": "Manchester United",
        "Man City": "Manchester City",
        "Spurs": "Tottenham Hotspur",
        "Wolves": "Wolverhampton Wanderers",
        "Newcastle Utd": "Newcastle United",
        "Nott'm Forest": "Nottingham Forest",
        "Nott Forest": "Nottingham Forest",
        "Sheffield Utd": "Sheffield United",
        "West Brom": "West Bromwich Albion",
        "Bournemouth": "AFC Bournemouth",
        "Brighton": "Brighton & Hove Albion",
        "Leeds Utd": "Leeds United",
        "Ipswich": "Ipswich Town",
    }
    return replacements.get(name, name)


def download_season(season_start_year: int, league: str = LEAGUE) -> Optional[Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    code = season_code(season_start_year)
    url = FD_URL.format(season=code, league=league)
    out = RAW_DIR / f"{league}_{code}.csv"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and len(r.content) > 1000:
            out.write_bytes(r.content)
            return out
        else:
            print(f"⚠️  No data for {season_start_year}/{season_start_year+1} (HTTP {r.status_code}).")
            return None
    except Exception as e:
        print(f"⚠️  Error downloading {url}: {e}")
        return None


def download_all(seasons: List[int]) -> List[Path]:
    files = []
    for y in tqdm(seasons, desc="Downloading seasons"):
        p = download_season(y)
        if p:
            files.append(p)
    print(f"✅ Downloaded {len(files)} seasons into {RAW_DIR}")
    return files


def normalize_all() -> pd.DataFrame:
    """Read every raw CSV and unify to one tidy table."""
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW_DIR.glob("E0_*.csv"))
    if not files:
        raise FileNotFoundError("No raw CSVs found in data/raw. Run with --download first.")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["src_file"] = f.name

        # Map common columns to tidy names
        colmap = {
            "Date": "date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "home_goals",
            "FTAG": "away_goals",
            "FTR": "result",
            # (optional odds if present)
            "AvgH": "odds_home_avg",
            "AvgD": "odds_draw_avg",
            "AvgA": "odds_away_avg",
        }
        out = pd.DataFrame({v: df[k] if k in df.columns else None for k, v in colmap.items()})
        out["date"] = pd.to_datetime(out["date"], dayfirst=True, errors="coerce")
        out["home_team"] = out["home_team"].astype(str).map(clean_team_name)
        out["away_team"] = out["away_team"].astype(str).map(clean_team_name)
        out["season_code"] = df["src_file"].str.extract(r"(\d{4})")
        out = out.dropna(subset=["date", "home_team", "away_team"])
        dfs.append(out)

    all_matches = pd.concat(dfs, ignore_index=True).sort_values("date").reset_index(drop=True)

    # Ensure numeric types
    for c in ["home_goals", "away_goals", "odds_home_avg", "odds_draw_avg", "odds_away_avg"]:
        if c in all_matches:
            all_matches[c] = pd.to_numeric(all_matches[c], errors="coerce")

    out_path = PROC_DIR / "matches_normalized.csv"
    all_matches.to_csv(out_path, index=False)
    print(f"✅ Saved normalized dataset: {out_path} ({len(all_matches)} rows)")
    return all_matches


def parse_cli_seasons(s: Optional[str]) -> List[int]:
    if not s:
        return DEFAULT_SEASONS
    return [int(x) for x in s.split(",") if x.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download raw CSVs before normalizing")
    parser.add_argument("--seasons", type=str, default=None,
                        help="Comma-separated start years, e.g. 2018,2019,...,2025")
    args = parser.parse_args()

    seasons = parse_cli_seasons(args.seasons)

    if args.download:
        download_all(seasons)

    normalize_all()

