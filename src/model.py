import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ===========================
# Load dataset
# ===========================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "matches_normalized.csv")

print(f"📂 Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# ===========================
# Encode categorical features
# ===========================
encoder = LabelEncoder()
df["home_team_enc"] = encoder.fit_transform(df["home_team"])
df["away_team_enc"] = encoder.fit_transform(df["away_team"])

# ===========================
# Features + Target
# ===========================
X = df[["home_team_enc", "away_team_enc", "odds_home_avg", "odds_draw_avg", "odds_away_avg"]]
y = df["result"]  # values like "H", "A", "D"

# ===========================
# Train/test split
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===========================
# Model
# ===========================
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ===========================
# Evaluate
# ===========================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained. Accuracy: {acc:.2%}")

# ===========================
# Save artifacts
# ===========================
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/random_forest.pkl")
joblib.dump(encoder, "models/label_encoder.pkl")
print("💾 Model + encoder saved in models/")

