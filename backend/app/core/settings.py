import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

MODEL_NAME = "churn-model"
MODEL_VERSION = 1.0
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = f"{MODEL_DIR}/{MODEL_NAME}.dat.gz"
MODEL_VERSION_PATH = f"{MODEL_DIR}/{MODEL_NAME}-latest-version.txt"
MODEL_METRIC_PATH = f"{MODEL_DIR}/{MODEL_NAME}-metrics.txt"
DATASET_PATH = BASE_DIR / "datasets/churn.csv"

BACKEND_CORS_ORIGINS = [
    "http://localhost",
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://127.0.0.1:8020",
    "http://localhost:8020"
]