from pathlib import Path
from typing import Any
import json
import joblib
import yaml


class ModelLoader:
    """
    Loads the ML model and preprocessing pipeline from the
    verified MLflow deployment bundle.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = self.project_root / config_path

        self.config: dict[str, Any] = {}

        self.model = None
        self.preprocessor = None

        self.model_uri = None
        self.model_name = None
        self.model_version = None
        self.run_id = None
        self.model_id = None
        self.metadata = {}

    def load_config(self) -> None:
        """Load project configuration from config.yaml."""

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

        with open(self.config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file) or {}

    def load_deployment_bundle(self) -> None:
        """Load model, preprocessor, and metadata from deployment bundle."""

        deployment_dir = self.project_root / "deployment"

        model_path = deployment_dir / "model.pkl"
        preprocessor_path = deployment_dir / "preprocessor.pkl"
        metadata_path = deployment_dir / "model_metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        if not preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor not found: {preprocessor_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Model metadata not found: {metadata_path}"
            )

        print(f"[INFO] Loading deployment metadata from: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        self.model_name = self.metadata.get("model_name")
        self.model_version = self.metadata.get("model_version")
        self.run_id = self.metadata.get("run_id")
        self.model_id = self.metadata.get("model_id")
        self.model_uri = str(model_path)

        print(f"[INFO] Model name    : {self.model_name}")
        print(f"[INFO] Model version : {self.model_version}")
        print(f"[INFO] MLflow Run ID : {self.run_id}")
        print(f"[INFO] Model ID      : {self.model_id}")

        print(f"[INFO] Loading model from: {model_path}")

        self.model = joblib.load(model_path)

        print("[SUCCESS] Model loaded successfully.")

        print(f"[INFO] Loading preprocessor from: {preprocessor_path}")

        self.preprocessor = joblib.load(preprocessor_path)

        print("[SUCCESS] Preprocessor loaded successfully.")

    def load(self) -> None:
        """Load configuration and deployment artifacts."""

        print("[INFO] Initializing ModelLoader...")

        self.load_config()
        self.load_deployment_bundle()

        print("[SUCCESS] ModelLoader initialization completed.")

    def is_ready(self) -> bool:
        """Return True when both model and preprocessor are loaded."""

        return self.model is not None and self.preprocessor is not None


if __name__ == "__main__":
    loader = ModelLoader()

    try:
        loader.load()

        print("\n========== MODEL LOADER STATUS ==========")
        print(f"Model loaded       : {loader.model is not None}")
        print(f"Preprocessor loaded: {loader.preprocessor is not None}")
        print(f"Model name         : {loader.model_name}")
        print(f"Model version      : {loader.model_version}")
        print(f"MLflow Run ID      : {loader.run_id}")
        print(f"Model ID           : {loader.model_id}")
        print(f"Model path         : {loader.model_uri}")
        print("Ready              :", loader.is_ready())
        print("=========================================\n")

    except Exception as exc:
        print(f"[ERROR] Model loading failed: {exc}")
        raise