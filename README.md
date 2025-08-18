
# EPL Gameweek 2 Predictor (2025/26)

Baseline ML project to predict **Premier League Gameweek 2** outcomes.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env  # add your football-data.org token (optional)

python src/data.py --download
python src/features.py
python src/train.py

# Predict GW2 via API
python src/predict.py --season 2025 --matchday 2 --out predictions/prem_gw2_2025_26.csv

# Or with manual fixtures CSV
python src/predict.py --fixtures_csv data/upcoming_fixtures.csv --out predictions/prem_gw2_manual.csv
```

**Data:** football-data.co.uk (historical results/odds), football-data.org (fixtures). Please respect licensing/TOS.
