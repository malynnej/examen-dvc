import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config.config import get_path

def normalize_data(X_train: pd.DataFrame, X_test:pd.DataFrame)->dict:

    preprocessed_dir = get_path("data.preprocessed")

    # select numeric feature columns (float/int)
    num_cols = X_train.select_dtypes(include=["float64", "int64"]).columns

    scaler = StandardScaler()

    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    print("Columns:", X_train.columns.tolist())
    print(X_train.dtypes)   
    print("Columns:", X_test.columns.tolist())
    print(X_test.dtypes) 

    X_train_scaled_path = preprocessed_dir/"X_train_scaled.parquet"
    X_test_scaled_path = preprocessed_dir/"X_test_scaled.parquet"

    # Save normalized data
    X_train.to_parquet(X_train_scaled_path, index=False)
    X_test.to_parquet(X_test_scaled_path, index=False)

    result = {
        "X_train_scaled": str(X_train_scaled_path),
        "X_test_scaled": str(X_test_scaled_path),
        "num_X_train_scaled": len(X_train),
        "num_X_test_scaled": len(X_test),
    }

    return result

if __name__ == "__main__":
    import logging

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Load raw data
    data_dir = get_path("data.preprocessed")
    X_train= pd.read_parquet(data_dir / "X_train.parquet")
    X_test = pd.read_parquet(data_dir / "X_test.parquet")

    print(f"Loaded X_train: {len(X_train)} samples")
    print(f"Loaded X_test: {len(X_test)} samples")

    # Preprocess
    output_paths = normalize_data(X_train,X_test)

    print("\n" + "=" * 60)
    print("NORMALIZING COMPLETE!")
    print("=" * 60)
