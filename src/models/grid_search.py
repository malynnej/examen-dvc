import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from src.config.config import get_path, load_config

def grid_search(X_train: pd.DataFrame, y_train:pd.Series):

    grid_config = load_config("params")

    model = RandomForestRegressor(
        random_state=grid_config["general"].get("random_state", 42)
        )

    param_grid = {
        "n_estimators": grid_config["gridsearch"].get("n_estimators", [50, 100, 200]),
        "max_depth": grid_config["gridsearch"].get("max_depth", [None, 5, 10]),
        "min_samples_split": grid_config["gridsearch"].get("min_samples_split", [2,5]),
        "min_samples_leaf": grid_config["gridsearch"].get("min_samples_leaf", [1, 2, 4]),
        "max_features": grid_config["gridsearch"].get("max_features", ["auto", "sqrt"]),
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv= grid_config["gridsearch"].get("cv", 5),
        # scoring=grid_config["gridsearch"].get("scoring","neg_mean_squared_error"),
        n_jobs= grid_config["gridsearch"].get("n_jobs",-1),
        verbose= grid_config["gridsearch"].get("verbose",1),
    )

    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    print("Best params:", best_params)

    models_dir = get_path("models.models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)

    # save best params
    joblib.dump(best_params, models_dir / "best_params.pkl")

    # optional: save cv results as json or csv
    with open(models_dir / "grid_results.json", "w") as f:
        json.dump(
            {
                "best_score": grid.best_score_,
                "best_params": best_params,
            },
            f,
            indent=2,
        )

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
    output_paths = grid_search(X_train,y_train)

    print("\n" + "=" * 60)
    print("GRID SEARCH COMPLETE!")
    print("=" * 60)
