
import joblib
from pathlib import Path
from src.features import build_features, load_config
from src.model import train_final

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    cfg = load_config()
    feats = build_features()
    result, feature_cols = train_final(feats, random_state=cfg["model"]["random_state"], cv_folds=cfg["model"]["cv_folds"])
    joblib.dump({"pipeline": result.model, "feature_cols": feature_cols}, MODELS_DIR / "model.joblib")
    print("Saved model to", MODELS_DIR / "model.joblib")
    print("CV metrics:", result.metrics)
