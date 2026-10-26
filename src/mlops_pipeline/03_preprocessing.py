import os
import sys
import json
import joblib
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def clean_dataframe(df, config):
    """Apply deterministic dataset cleaning shared by training and inference."""
    df = df.copy()
    target_column = config["data"]["target_column"]
    id_column = config["data"]["id_column"]

    if id_column in df.columns:
        df = df.drop(columns=[id_column])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if target_column in df.columns:
        df[target_column] = df[target_column].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0).astype(int)

    return df


def create_initial_test_ids(df, config, processed_dir):
    """Recreate the original test split and permanently save its customer IDs."""
    id_column = config["data"]["id_column"]
    target_column = config["data"]["target_column"]
    test_ids_path = os.path.join(processed_dir, "test_customer_ids.csv")

    if os.path.exists(test_ids_path):
        test_ids = pd.read_csv(test_ids_path)

        if id_column not in test_ids.columns:
            raise ValueError(f"{test_ids_path} does not contain {id_column}.")

        return test_ids[id_column].astype(str).tolist()

    print("[INFO] test_customer_ids.csv not found.")
    print("[INFO] Recreating the original train/test split to recover test customer IDs...")

    working_df = df.copy()
    working_df[id_column] = working_df[id_column].astype(str)
    working_df["TotalCharges"] = pd.to_numeric(working_df["TotalCharges"], errors="coerce")
    working_df[target_column] = working_df[target_column].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0).astype(int)

    X = working_df.drop(columns=[id_column, target_column])
    y = working_df[target_column]

    _, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=y,
    )

    test_ids = working_df.loc[X_test.index, id_column].astype(str).tolist()

    pd.DataFrame({id_column: test_ids}).to_csv(test_ids_path, index=False)

    print(f"[SUCCESS] Original test-set IDs saved to {test_ids_path}")
    print(f"[INFO] Fixed test set size: {len(test_ids)}")

    return test_ids


def split_raw_data(df, config, processed_dir):
    """Keep the original test set fixed while using all other records for training."""
    target_column = config["data"]["target_column"]
    id_column = config["data"]["id_column"]

    test_ids_path = os.path.join(processed_dir, "test_customer_ids.csv")

    if os.path.exists(test_ids_path):
        test_ids_df = pd.read_csv(test_ids_path)
        test_ids = set(test_ids_df[id_column].astype(str))
        print(f"[INFO] Using fixed test set from {test_ids_path}")
    else:
        test_ids = set(create_initial_test_ids(df, config, processed_dir))
        print("[INFO] Original test set has been permanently fixed.")

    working_df = df.copy()
    working_df[id_column] = working_df[id_column].astype(str)

    test_mask = working_df[id_column].isin(test_ids)

    if not test_mask.any():
        raise ValueError("None of the fixed test customer IDs were found in the current dataset.")

    test_df = working_df.loc[test_mask].copy()
    train_df = working_df.loc[~test_mask].copy()

    y_test = test_df[target_column].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0).astype(int)
    y_train = train_df[target_column].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0).astype(int)

    X_test = clean_dataframe(test_df, config).drop(columns=[target_column])
    X_train = clean_dataframe(train_df, config).drop(columns=[target_column])

    print(f"[INFO] Training records: {len(X_train)}")
    print(f"[INFO] Fixed test records: {len(X_test)}")

    return X_train, X_test, y_train, y_test


def build_preprocessor(X_train):
    """Create the single canonical feature-preprocessing pipeline used by ML Pipeline."""
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    ), cat_cols, num_cols


def run_preprocessing():
    print("[INFO] Starting Preprocessing...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = config["data"]["raw_path"]
    processed_dir = config["data"]["processed_dir"]
    target_column = config["data"]["target_column"]

    if not os.path.exists(data_path):
        print(f"[ERROR] Raw dataset not found at: {data_path}")
        return False

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    raw_df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = split_raw_data(raw_df, config, processed_dir)

    preprocessor, cat_cols, num_cols = build_preprocessor(X_train)

    print("[INFO] Fitting preprocessing pipeline on training data...")
    X_train_final = preprocessor.fit_transform(X_train)
    X_test_final = preprocessor.transform(X_test)

    np.save(os.path.join(processed_dir, "X_train_final.npy"), X_train_final)
    np.save(os.path.join(processed_dir, "X_test_final.npy"), X_test_final)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train.to_numpy(dtype=np.int64))
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test.to_numpy(dtype=np.int64))

    X_train.to_csv(os.path.join(processed_dir, "X_train_raw.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test_raw.csv"), index=False)

    joblib.dump(preprocessor, config["paths"]["preprocessor"])

    metadata = {
        "dataset_name": "Telco Customer Churn",
        "target_column": target_column,
        "train_shape_raw": list(X_train.shape),
        "test_shape_raw": list(X_test.shape),
        "train_shape_transformed": list(X_train_final.shape),
        "test_shape_transformed": list(X_test_final.shape),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "split_test_size": config["split"]["test_size"],
        "split_random_state": config["split"]["random_state"],
        "fixed_test_set": True,
        "test_customer_ids_path": os.path.join(processed_dir, "test_customer_ids.csv"),
        "raw_train_path": os.path.join(processed_dir, "X_train_raw.csv"),
        "raw_test_path": os.path.join(processed_dir, "X_test_raw.csv"),
    }

    with open(os.path.join(processed_dir, "dataset_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[SUCCESS] Preprocessing complete. Preprocessor saved to {config['paths']['preprocessor']}")
    return True


if __name__ == "__main__":
    passed = run_preprocessing()
    sys.exit(0 if passed else 1)