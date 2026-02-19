import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from src.config.config import get_path

def evaluate(X_test: pd.DataFrame, y_test:pd.Series, y_train:pd.Series):

    models_dir = get_path("models.models_dir")
    model = joblib.load(models_dir/"model.pkl")

    y_pred = model.predict(X_test)

    baseline_pred = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
    print("Baseline MSE:", mean_squared_error(y_test, baseline_pred))
    print("Baseline R2:", r2_score(y_test, baseline_pred))

    print("Model MSE:", mean_squared_error(y_test, y_pred))
    print("Model R2:", r2_score(y_test, y_pred))

    results_dir= get_path("results.results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)
    # save predictions
    preds = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    preds.to_csv(results_dir / "predictions.csv", index=False)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    
    with open(results_dir / "scores.json", "w") as f:
        json.dump({"mse": mse, "r2": r2}, f, indent=2)

if __name__ == "__main__":
    import logging

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Load raw data
    data_dir = get_path("data.preprocessed")
    X_test= pd.read_parquet(data_dir / "X_test_scaled.parquet")
    if "date" in X_test.columns:
        X_test = X_test.drop(columns=["date"])
    y_test = pd.read_parquet(data_dir / "y_test.parquet").squeeze()
    y_train = pd.read_parquet(data_dir / "y_train.parquet").squeeze()

    print(f"Loaded X_test: {len(X_test)} samples")
    print(f"Loaded y_test: {len(y_test)} samples")

    # Preprocess
    output_paths = evaluate(X_test,y_test, y_train)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
