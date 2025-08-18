
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class EloParams:
    k_factor: float = 20.0
    home_advantage: float = 60.0

def expected_score(rating_a, rating_b):
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

def update_elo(ra, rb, score_a, score_b, params: EloParams):
    ea = expected_score(ra + params.home_advantage, rb)
    eb = 1 - ea
    ra_new = ra + params.k_factor * (score_a - ea)
    rb_new = rb + params.k_factor * (score_b - eb)
    return ra_new, rb_new

def compute_match_elos(df: pd.DataFrame, params: EloParams, base_rating=1500):
    df = df.copy().sort_values("date")
    teams = pd.unique(pd.concat([df["home_team"], df["away_team"]], ignore_index=True))
    ratings = {t: base_rating for t in teams}
    elo_home_pre, elo_away_pre = [], []
    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        rh, ra = ratings.get(h, base_rating), ratings.get(a, base_rating)
        elo_home_pre.append(rh)
        elo_away_pre.append(ra)
        r = str(row.get("result", "")).upper()
        if r not in {"H","D","A"}:
            continue
        if r == "H": sh, sa = 1.0, 0.0
        elif r == "A": sh, sa = 0.0, 1.0
        else: sh, sa = 0.5, 0.5
        rh_new, ra_new = update_elo(rh, ra, sh, sa, params)
        ratings[h], ratings[a] = rh_new, ra_new
    df["elo_home_pre"] = elo_home_pre
    df["elo_away_pre"] = elo_away_pre
    return df
