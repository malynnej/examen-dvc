import joblib
import pandas as pd
from src.config.config import get_path, load_config
from sklearn.ensemble import GradientBoostingRegressor

def train_model_gb(X_train: pd.DataFrame, y_train:pd.Series):

    models_dir = get_path("models.models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)
    best_params = joblib.load(models_dir / "best_params_gbr.pkl")
    best_params.pop("random_state", None)

    model_config = load_config("params")

    random_state=model_config["general"].get("random_state", 42)
    model = GradientBoostingRegressor(random_state=random_state, **best_params)
    model.fit(X_train, y_train)

    joblib.dump(model, models_dir / "model_gbr.pkl")

if __name__ == "__main__":
    import logging

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Load raw data
    data_dir = get_path("data.preprocessed")
    X_train= pd.read_parquet(data_dir / "X_train_scaled.parquet")
    if "date" in X_train.columns:
        X_train = X_train.drop(columns=["date"])
    y_train = pd.read_parquet(data_dir / "y_train.parquet").squeeze()

    print(f"Loaded X_train: {len(X_train)} samples")
    print(f"Loaded y_train: {len(y_train)} samples")

    # Preprocess
    output_paths = train_model_gb(X_train,y_train)

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE!")
    print("=" * 60)
