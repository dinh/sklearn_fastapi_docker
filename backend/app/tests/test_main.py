# To resolve ModuleNotFoundError
# https://techwithtech.com/importerror-attempted-relative-import-with-no-known-parent-package/
import pathlib
import sys
from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from main import app, load_model_artifacts
from main import common_parameters

from starlette.testclient import TestClient

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATASET_DIR = f"{BASE_DIR}/fixtures/datasets"
MODEL_DIR = f"{BASE_DIR}/models"
DATASET_PATH = f"{DATASET_DIR}/churn.csv"
MODEL_PATH = f"{MODEL_DIR}/churn-model.dat.gz"
MODEL_METRIC_PATH = f"{MODEL_DIR}/churn-model.metrics.txt"
MODEL_VERSION_PATH = f"{MODEL_DIR}/churn-model.version.txt"


async def override_common_parameters():
    return {
        "dataset_path": DATASET_PATH,
        "model_path": MODEL_PATH,
        "model_metric_path": MODEL_METRIC_PATH,
        "model_version_path": MODEL_VERSION_PATH
    }


app.dependency_overrides[common_parameters] = override_common_parameters


def test_docs_redirect():
    client = TestClient(app)
    response = client.get("/")
    assert response.history[0].status_code == 307
    assert response.status_code == 200
    # http://testserver is an httpx "magic" url that tells the client to query the
    # given app instance.
    assert response.url == "http://testserver/docs"


def test_model_info():
    client = TestClient(app)
    response = client.get("/info")
    assert response.status_code == 200
    assert response.url == "http://testserver/info"
    assert response.json() == {
        "status_code": 200,
        "name": "churn-model",
        "version": 1,
        "description": "Predict churn based on customer data",
        "is_artifacts_available": "True"
    }


def test_healthcheck():
    client = TestClient(app)
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.url == "http://testserver/healthcheck"


def test_predict():
    # We need the startup event handlers in these tests, so let's use TestClient with a with statement
    # https://fastapi.tiangolo.com/advanced/testing-events/
    with TestClient(app) as client:
        payload = {
            "tenure": 2,
            "no_internet_service": False,
            "internet_service_fiber_optic": False,
            "online_security": False,
            "device_protection": False,
            "contract_month_to_month": True,
            "payment_method_electronic_check": True,
            "paperless_billing": True
        }
        response = client.post('/predict', json=payload)
        assert response.status_code == 200
        # data = response.json()
        # assert data["prediction"] == 0
        assert "label" in response.json()
        assert "prediction" in response.json()
        assert "label", "probability" in response.json()
        # Warning: don't change the test model artifact or this will fail
        # assert response.json() == {
        #     "label": "churner",
        #     "prediction": 1,
        #     "probability": 0.65
        # }


def test_predict_batch():
    client = TestClient(app)
    fpath = f"{DATASET_DIR}/churn-small.csv"
    with open(fpath, "rb") as f:
        response = client.post("/batch-predict", files={"file": ("filename", f, "text/csv")})
        assert response.status_code == 200, response.content
        # The b character before a string produces a variable of byte type instead of string
        assert response.content == b"customerID,Churn,Prediction\n7010-BRBUU,No,No\n9688-YGXVR,No,No\n9286-DOJGF,Yes,Yes\n6994-KERXL,No,Yes\n2181-UAESM,No,No\n4312-GVYNH,No,No\n"


def test_train():
    with TestClient(app) as client:
        response = client.get("/train")
    assert response.status_code == 200
    assert response.url == "http://testserver/train"
    assert response.json() == {"status_code": 200, "detail": "Model training job has been created!"}


def test_limit_upload_size():
    client = TestClient(app)
    fpath = f"{DATASET_DIR}/churn-12MB.csv"
    with open(fpath, "rb") as f:
        response = client.post("/batch-predict", files={"file": ("filename", f, "text/csv")})
        assert response.status_code == 200
        assert response.json() == {
            "status_code": 412,
            "detail": "Request entity is too large. Payload must be less than 10MB"
        }
