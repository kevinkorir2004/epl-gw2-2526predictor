
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import TimeSeriesSplit

@dataclass
class TrainResult:
    model: Pipeline
    metrics: dict

def make_pipeline(random_state=42):
    features = ["elo_diff","gf_diff","ga_diff","elo_home_pre","elo_away_pre","home_gf","home_ga","away_gf","away_ga"]
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(multi_class="multinomial", max_iter=200, random_state=random_state))
    ])
    return pipe, features

def time_series_cv(feats: pd.DataFrame, pipe: Pipeline, feature_cols, n_splits=5):
    feats = feats.dropna(subset=feature_cols + ["y"]).copy()
    X = feats[feature_cols].values
    y = feats["y"].values.astype(int)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    logs, accs = [], []
    for train_idx, test_idx in tscv.split(X):
        pipe.fit(X[train_idx], y[train_idx])
        proba = pipe.predict_proba(X[test_idx])
        y_pred = proba.argmax(axis=1)
        logs.append(log_loss(y[test_idx], proba, labels=[0,1,2]))
        accs.append(accuracy_score(y[test_idx], y_pred))
    return {"cv_log_loss_mean": float(np.mean(logs)), "cv_accuracy_mean": float(np.mean(accs))}

def train_final(feats: pd.DataFrame, random_state=42, cv_folds=5):
    pipe, feature_cols = make_pipeline(random_state=random_state)
    metrics = time_series_cv(feats, pipe, feature_cols, n_splits=cv_folds)
    train_df = feats.dropna(subset=feature_cols + ["y"]).copy()
    X = train_df[feature_cols].values
    y = train_df["y"].values.astype(int)
    pipe.fit(X, y)
    return TrainResult(model=pipe, metrics=metrics), feature_cols
