import pandas as pd

from src.config.config import get_path, load_config

def split_data(df:pd.DataFrame)-> dict:
    """
    Splits data into train and test set.
    """

    preprocessed_dir = get_path("data.preprocessed")
    data_config = load_config("params")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    cutoff_quantile = data_config["data"].get("cutoff_quantile", 0.8)
    cutoff = df["date"].quantile(cutoff_quantile)  # 80% oldest rows → train

    train_df = df[df["date"] <= cutoff]
    test_df = df[df["date"] >  cutoff]
    
    target = data_config["data"].get("target", "silica_concentrate")
    X_train = train_df.drop(target, axis=1)
    y_train = train_df[target].to_frame()
    X_test = test_df.drop(target, axis=1)
    y_test = test_df[target].to_frame()

    X_train_path = preprocessed_dir/"X_train.parquet"
    y_train_path = preprocessed_dir/"y_train.parquet"
    X_test_path = preprocessed_dir/"X_test.parquet"
    y_test_path = preprocessed_dir/"y_test.parquet"

    # Save splits
    X_train.to_parquet(X_train_path, index=False)
    y_train.to_parquet(y_train_path, index=False)
    X_test.to_parquet(X_test_path, index=False)
    y_test.to_parquet(y_test_path, index=False)

    print("y_train.describe():")
    print(y_train.describe())
    print("\ny_test.describe():")
    print(y_test.describe())

    result = {
        "X_train": str(X_train_path),
        "y_train": str(y_train_path),
        "X_test": str(X_test_path),
        "y_test": str(y_test_path),
        "num_X_train": len(X_train),
        "num_y_train": len(y_train),
        "num_X_test": len(X_test),
        "num_y_test": len(y_test),
    }

    return result

# Standalone execution for testing
if __name__ == "__main__":
    import logging

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Load raw data
    raw_dir = get_path("data.raw")
    paths = load_config("paths")
    raw_data = paths["data"]["csv"]
    df = pd.read_csv(raw_dir / raw_data)

    print(f"Loaded {len(df)} samples")

    # Preprocess
    output_paths = split_data(df)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)