# To resolve ModuleNotFoundError
# https://techwithtech.com/importerror-attempted-relative-import-with-no-known-parent-package/
import hashlib
import shutil
import sys
from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

import pytest
from starlette.testclient import TestClient

import main
from main import app

# The tracked fixtures. Tests must never write to these: every test works on copies in tmp_path.
FIXTURES_DIR = file.parent
FIXTURE_DATASET_DIR = FIXTURES_DIR / "fixtures" / "datasets"
FIXTURE_MODEL_DIR = FIXTURES_DIR / "models"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Hashes recorded before any test runs, checked by test_tracked_fixtures_are_untouched
_FIXTURE_HASHES = {p.name: _sha256(p) for p in sorted(FIXTURE_MODEL_DIR.iterdir()) if p.is_file()}


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    """
    Points every artifact path of the app at a temporary copy of the test fixtures.
    The API reads these paths at startup and when it serves /info and /train, so one
    monkeypatch covers the whole app.
    """
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()

    paths = {
        "MODEL_PATH": model_dir / "churn-model.dat.gz",
        "MODEL_METRIC_PATH": model_dir / "churn-model-metrics.txt",
        "MODEL_VERSION_PATH": model_dir / "churn-model-latest-version.txt",
        "MODEL_MANIFEST_PATH": model_dir / "churn-model.manifest.json",
        "DATASET_PATH": dataset_dir / "churn.csv",
    }
    shutil.copy(FIXTURE_MODEL_DIR / "churn-model.dat.gz", paths["MODEL_PATH"])
    shutil.copy(FIXTURE_MODEL_DIR / "churn-model.metrics.txt", paths["MODEL_METRIC_PATH"])
    shutil.copy(FIXTURE_MODEL_DIR / "churn-model.version.txt", paths["MODEL_VERSION_PATH"])
    shutil.copy(FIXTURE_DATASET_DIR / "churn.csv", paths["DATASET_PATH"])

    # Start every test without a model in memory, and restore None afterwards
    monkeypatch.setattr(main, "model_artifacts", None)
    for name, path in paths.items():
        monkeypatch.setattr(main, name, str(path))
    return paths


@pytest.fixture
def client(artifacts):
    # The `with` block runs the startup event, which loads the model from artifacts["MODEL_PATH"]
    with TestClient(app) as test_client:
        yield test_client


def _expected_version(artifacts):
    return int(artifacts["MODEL_VERSION_PATH"].read_text().strip())


def test_docs_redirect(client):
    response = client.get("/")
    assert response.history[0].status_code == 307
    assert response.status_code == 200
    # http://testserver is an httpx "magic" url that tells the client to query the
    # given app instance.
    assert response.url == "http://testserver/docs"


def test_model_info(client, artifacts):
    response = client.get("/info")
    assert response.status_code == 200
    assert response.url == "http://testserver/info"
    body = response.json()
    assert body["name"] == "churn-model"
    assert body["description"] == "Predict churn based on customer data"
    assert body["is_artifacts_available"] == "True"
    assert body["model_loaded"] is True
    assert body["version"] == _expected_version(artifacts)
    assert body["manifest"] is None  # no training has run in this test


def test_healthcheck(client):
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.url == "http://testserver/healthcheck"
    assert response.json() == {"detail": "OK", "model_loaded": True}


def test_predict(client):
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
    body = response.json()
    assert set(body) == {"label", "prediction", "probability"}
    assert body["label"] in {"churner", "non churner"}
    assert body["prediction"] in {0, 1}
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label"] == ("churner" if body["prediction"] == 1 else "non churner")


def test_predict_rejects_negative_tenure(client):
    response = client.post('/predict', json={"tenure": -1})
    assert response.status_code == 422


def test_predict_batch(client):
    fpath = FIXTURE_DATASET_DIR / "churn-small.csv"
    with open(fpath, "rb") as f:
        response = client.post("/batch-predict", files={"file": ("filename", f, "text/csv")})
    assert response.status_code == 200, response.content
    assert response.headers["content-type"].startswith("text/csv")

    lines = response.text.splitlines()
    assert lines[0] == "customerID,Churn,Prediction"
    rows = [line.split(",") for line in lines[1:]]
    assert [row[0] for row in rows] == [
        "7010-BRBUU", "9688-YGXVR", "9286-DOJGF", "6994-KERXL", "2181-UAESM", "4312-GVYNH"
    ]
    for row in rows:
        assert row[1] in {"Yes", "No"}  # the actual churn status from the input file
        assert row[2] in {"Yes", "No"}  # the prediction

    # Prediction is deterministic for a given model and input
    with open(fpath, "rb") as f:
        again = client.post("/batch-predict", files={"file": ("filename", f, "text/csv")})
    assert again.content == response.content


def test_predict_without_model_returns_503(artifacts, monkeypatch):
    monkeypatch.setattr(main, "MODEL_PATH", str(artifacts["MODEL_PATH"].parent / "missing.dat.gz"))
    with TestClient(app) as client:
        assert client.get("/healthcheck").json() == {"detail": "OK", "model_loaded": False}
        payload = {"tenure": 1}
        assert client.post("/predict", json=payload).status_code == 503
        with open(FIXTURE_DATASET_DIR / "churn-small.csv", "rb") as f:
            batch = client.post("/batch-predict", files={"file": ("filename", f, "text/csv")})
        assert batch.status_code == 503


def test_train_reloads_model(client, artifacts):
    old_model = main.model_artifacts
    old_version = _expected_version(artifacts)
    old_metric_lines = artifacts["MODEL_METRIC_PATH"].read_text().splitlines()

    response = client.get("/train")
    # The background training task has finished by the time the TestClient returns
    assert response.status_code == 200
    assert response.url == "http://testserver/train"
    assert response.json() == {"detail": "Model training job has been created!"}

    assert _expected_version(artifacts) == old_version + 1
    assert len(artifacts["MODEL_METRIC_PATH"].read_text().splitlines()) == len(old_metric_lines) + 1
    assert main.model_artifacts is not old_model  # the in-memory model was replaced

    info = client.get("/info").json()
    assert info["version"] == old_version + 1
    assert info["manifest"]["model_version"] == old_version + 1
    assert set(info["manifest"]["libraries"]) == {"numpy", "pandas", "scikit-learn", "joblib"}

    payload = {"tenure": 2, "contract_month_to_month": True}
    assert client.post("/predict", json=payload).status_code == 200


def test_train_from_missing_model_creates_it(artifacts, monkeypatch):
    """The API starts without a model, and /train creates the first one."""
    missing = artifacts["MODEL_PATH"].parent / "first.dat.gz"
    monkeypatch.setattr(main, "MODEL_PATH", str(missing))
    with TestClient(app) as client:
        assert client.get("/healthcheck").json()["model_loaded"] is False
        assert client.get("/train").status_code == 200
        assert missing.exists()
        assert client.get("/healthcheck").json()["model_loaded"] is True
        assert client.post("/predict", json={"tenure": 2}).status_code == 200


def test_limit_upload_size(client):
    fpath = FIXTURE_DATASET_DIR / "churn-12MB.csv"
    with open(fpath, "rb") as f:
        response = client.post("/batch-predict", files={"file": ("filename", f, "text/csv")})
    assert response.status_code == 413
    assert response.json() == {
        "detail": "Request entity is too large. Payload must be less than 10MB"
    }


def test_tracked_fixtures_are_untouched():
    # Runs last in this module; catches any test that writes into the tracked fixtures
    current = {p.name: _sha256(p) for p in sorted(FIXTURE_MODEL_DIR.iterdir()) if p.is_file()}
    assert current == _FIXTURE_HASHES
