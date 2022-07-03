from core.settings import MODEL_VERSION_PATH, MODEL_METRIC_PATH, MODEL_PATH, DATASET_PATH
from helpers.functions import prepare_churn_dataset, train_and_save_model, execute_pipeline

execute_pipeline(DATASET_PATH, MODEL_PATH, MODEL_METRIC_PATH, MODEL_VERSION_PATH)