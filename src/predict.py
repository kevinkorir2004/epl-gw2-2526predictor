import os
import sys
import joblib
import pandas as pd

# ===========================
# Load trained model + encoder
# ===========================
MODEL_PATH = os.path.join("models", "random_forest.pkl")
ENCODER_PATH = os.path.join("models", "label_encoder.pkl")

print("📂 Loading model + encoder...")
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
print("✅ Model and encoder loaded")

# ===========================
# Prediction function
# ===========================
def predict_match(home_team, away_team, odds_home, odds_draw, odds_away):
    try:
        # Encode teams
        home_enc = encoder.transform([home_team])[0]
        away_enc = encoder.transform([away_team])[0]
    except ValueError:
        raise ValueError(f"❌ One of the teams ({home_team}, {away_team}) is not in the encoder training data")

    # Build input row
    X_new = pd.DataFrame([{
        "home_team_enc": home_enc,
        "away_team_enc": away_enc,
        "odds_home_avg": odds_home,
        "odds_draw_avg": odds_draw,
        "odds_away_avg": odds_away
    }])

    # Predict
    prediction = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0]

    return prediction, {
        "Home": proba[0],
        "Draw": proba[1],
        "Away": proba[2]
    }

# ===========================
# CLI mode
# ===========================
if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("⚠️ Usage: python predict.py <HomeTeam> <AwayTeam> <OddsHome> <OddsDraw> <OddsAway>")
        print("👉 Example: python predict.py Arsenal Chelsea 1.85 3.40 4.20")
        sys.exit(1)

    home_team = sys.argv[1]
    away_team = sys.argv[2]
    odds_home = float(sys.argv[3])
    odds_draw = float(sys.argv[4])
    odds_away = float(sys.argv[5])

    result, probabilities = predict_match(home_team, away_team, odds_home, odds_draw, odds_away)

    print(f"\n🔮 Prediction: {home_team} vs {away_team}")
    print(f"➡️ Result: {result}")
    print(f"📊 Probabilities: {probabilities}")
