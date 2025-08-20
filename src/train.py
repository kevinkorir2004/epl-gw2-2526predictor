
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# ===========================
# File paths
# ===========================
DATA_PATH = os.path.join("data", "processed", "matches_normalized.csv")
MODEL_PATH = os.path.join("models", "random_forest.pkl")
ENCODER_PATH = os.path.join("models", "label_encoder.pkl")

print(f"📂 Loading dataset from: {DATA_PATH}")

# ===========================
# Load dataset
# ===========================
df = pd.read_csv(DATA_PATH)

print("✅ Dataset loaded. Columns available:", list(df.columns)[:10])

# ===========================
# Encode teams
# ===========================
encoder = LabelEncoder()
all_teams = pd.concat([df["home_team"], df["away_team"]]).unique()
encoder.fit(all_teams)

df["home_team_enc"] = encoder.transform(df["home_team"])
df["away_team_enc"] = encoder.transform(df["away_team"])

# ===========================
# Features & Target
# ===========================
features = ["home_team_enc", "away_team_enc", "odds_home_avg", "odds_draw_avg", "odds_away_avg"]
target = "result"

X = df[features]
y = df[target]

# ===========================
# Train/test split
# ===========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"📊 Training on {len(X_train)} matches, testing on {len(X_test)} matches")

# ===========================
# Train model
# ===========================
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# ===========================
# Evaluate
# ===========================
y_pred = model.predict(X_test)
print("\n✅ Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ===========================
# Save model + encoder
# ===========================
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(encoder, ENCODER_PATH)

print(f"\n💾 Model saved to {MODEL_PATH}")
print(f"💾 Encoder saved to {ENCODER_PATH}")
