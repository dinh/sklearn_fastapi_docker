import errno
import io
import os

import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTasks
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import RedirectResponse, StreamingResponse, JSONResponse, Response
from starlette.types import ASGIApp

from uvicorn import run

from core.schemas.schema import ChurnPrediction, CustomerData
from core.settings import *
from helpers.functions import get_churn_prediction, batch_file_predict, execute_pipeline

description = """
The Churn API helps you predict churners. 🚀

## Predict

You can predict the churn status of a user based on his data.

## Train model

Whenever the datasets change, you can retrain the model. The training is done in the backgound.
"""

tags_metadata = [
    {
        "name": "predict",
        "description": "Given the customer data, predict if he will churn or not.",
    },
    {
        "name": "train",
        "description": "Train the model.",
    },
]

app = FastAPI(
    title="Churn API",
    description=description,
    version="1.0.0",
    openapi_tags=tags_metadata
)


async def common_parameters():
    return {
        "dataset_path": DATASET_PATH,
        "model_path": MODEL_PATH,
        "model_metric_path": MODEL_METRIC_PATH,
        "model_version_path": MODEL_VERSION_PATH
    }


app.add_middleware(
    CORSMiddleware,
    allow_origins=BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# A middleware that limits the size of the uploaded file
class LimitUploadSize(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, max_upload_size: int) -> None:
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method == 'POST':
            if 'content-length' not in request.headers:
                return JSONResponse(content={
                    "detail": "Length Required"
                }, status_code=411)

            content_length = int(request.headers['content-length'])
            if content_length > self.max_upload_size:
                return JSONResponse(content={
                    "detail": "Request entity is too large. Payload must be less than 10MB"
                }, status_code=412)

        return await call_next(request)


app.add_middleware(
    LimitUploadSize,
    max_upload_size=10_000_000  # ~10MB
)

model_artifacts = None


@app.on_event("startup")
async def load_model_artifacts():
    """
    > Load the model artifacts from the model directory
    """
    global model_artifacts

    if model_artifacts is None:
        if os.path.exists(MODEL_PATH):
            model_artifacts = joblib.load(MODEL_PATH)
            print("Ready for inference!")
        else:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), MODEL_PATH)


@app.get("/", include_in_schema=False)
def docs_redirect():
    return RedirectResponse("/docs")


@app.get('/info')
async def model_info():
    """Return model information"""
    return JSONResponse(content={
        "name": MODEL_NAME,
        "version": MODEL_VERSION,
        "description": "Predict churn based on customer data",
        "is_artifacts_available": str(os.path.exists(MODEL_PATH))
    })


@app.get('/healthcheck')
def healthcheck():
    return JSONResponse(content={"detail": "OK"})


@app.post('/predict', response_model=ChurnPrediction, name='Predict churner', tags=['predict'])
async def predict(request: Request, payload: CustomerData):
    """
    Takes in a `CustomerData` object, and returns a prediction
    """
    global model_artifacts
    if request.method == "POST":
        return get_churn_prediction(payload, model_artifacts)


@app.post('/batch-predict', name='Moke predictions in batch', tags=['predict'])
async def predict_batch(request: Request, file: UploadFile = File(...)):
    """Predict with file input"""
    global model_artifacts
    if request.method == "POST":
        # Ensure that the file is a CSV
        print(file.content_type)
        if not file.content_type.startswith("text/csv") and not file.content_type.startswith("application/vnd.ms-excel"):
            raise HTTPException(status_code=415, detail="File must be in CSV format with comma separators")

        customer_data = await file.read()
        # TODO: check file fields

        if not customer_data:
            raise HTTPException(status_code=204, detail="No content")

    df_customer_data = batch_file_predict(customer_data, model_artifacts)
    if not isinstance(df_customer_data, pd.DataFrame):
        raise HTTPException(status_code=422, detail=df_customer_data)

    # https://stackoverflow.com/questions/61140398/fastapi-return-a-file-response-with-the-output-of-a-sql-query
    stream = io.StringIO()
    df_customer_data.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv"
                                 )
    response.headers["Content-Disposition"] = "attachment; filename=predictions-export.csv"
    response.headers["Access-Control-Expose-Headers"] = "Content-Disposition"

    return response


@app.get('/train', name='Train model', tags=['train'])
async def train(background_tasks: BackgroundTasks, commons: dict = Depends(common_parameters)):
    background_tasks.add_task(execute_pipeline, commons["dataset_path"], commons["model_path"],
                              commons["model_metric_path"],
                              commons["model_version_path"], message="Model created")

    return JSONResponse(content={"detail": "Model training job has been created!"})


if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8000)
