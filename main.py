import uvicorn
import pandas as pd
import os
import io
from fastapi import FastAPI,File,UploadFile,HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
# Third party imports
from pydantic import BaseModel, Field, validator
from typing import List
from starlette.responses import RedirectResponse, StreamingResponse, JSONResponse, Response

from ms.functions import get_model_response,batch_file_predict,prepare_data

model_name = "Churn model"
version = "v1.0.0"

api = FastAPI(
             title="Customer Churn Prediction API",
             description="API powered by FastAPI.",
             version="1.0.1",
             openapi_tags=[
    {
        'name': 'API Check',
        'description': 'default API check functions'
    },
    {
        'name': 'Prediction Functions',
        'description': 'functions that are used to predict customer churn'
    }
                           ])

# Input for data validation


class Input(BaseModel):
    tenure: int
    no_internet_service: bool = False
    internet_service_fiber_optic: bool = False
    online_security: bool = False
    device_protection: bool = False
    contract_month_to_month: bool = False
    payment_method_electronic_check: bool = False
    paperless_billing: bool = False

    @validator("tenure")
    def tenure_must_be_int_positive(cls, value):
        if value < 0:
            raise ValueError("Tenure value must be greater or equal to zero.")
        return value

    class Config:
        schema_extra = {
            "example": {
                'tenure': 2,
                'no_internet_service': False,
                'internet_service_fiber_optic': False,
                'online_security': False,
                'device_protection': False,
                'contract_month_to_month': True,
                'payment_method_electronic_check': True,
                'paperless_billing': True
            }
        }


# Ouput for data validation
class Output(BaseModel):
    label: str
    prediction: int
    probability: float


@api.get('/',tags=['API Check'])
async def model_info():
    """Return model information, version, how to call"""
    return {
        "name": model_name,
        "version": version
    }


@api.get('/health', tags=['API Check'])
async def service_health():
    """Return service health"""
    return {
        "health": "ok"
    }


@api.post('/predict', response_model=Output, tags=['Prediction Functions'])
async def model_predict(input: Input):
    """Predict with input"""
    return get_model_response(input)

# Define the response JSON
class Result(BaseModel):
     filename: str
     content_type: str
     predictions: str
 
responses = {
    415: {"description": "File Format is invalid"},
    #400: {"description": "the file is empty"},
    #410: {"description": "There is 1 missing column in file input"},
    #403: {"description": "Not enough privileges"},
}

@api.post('/batch_predict',name="Batch File Churn Predict", tags=['Prediction Functions'],response_model=Result,responses=responses )
async def batch_predict(file: UploadFile = File(...)):
    """Predict with file input"""
    # Ensure that the file is a CSV
    if not file.content_type.startswith("text/csv") and not file.content_type.startswith("application/vnd.ms-excel"):
            raise HTTPException(status_code=415, detail="File must be in CSV format with comma separators")

    contents = await file.read()
    #buffer = BytesIO(contents)
    #df = pd.read_csv(buffer)
    #if os.path.exists('data/{}'.format(name)) and os.stat('data/{}'.format(name)).st_size == 0:
    if not contents:
            raise HTTPException(status_code=415, detail="No content")
    buffer = BytesIO(contents)
    df = pd.read_csv(buffer)
    buffer.close()
    df_initial=df
    expected_columns = [
        'customerID',
        'gender',
        'SeniorCitizen',
        'Partner',
        'Dependents',
        'tenure',
        'PhoneService',
        'MultipleLines',
        'InternetService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
        'Contract',
        'PaperlessBilling',
        'PaymentMethod',
        'MonthlyCharges',
        'TotalCharges'
    ]

    # Create required features columns if missing
    for column in expected_columns:
        if column not in df:
            raise HTTPException(
                status_code=415,
                detail=f"Mandatory Columns are customerID,gender,SeniorCitizen,Partner, Dependents, tenure, PhoneService, MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection, TechSupport,StreamingTV,StreamingMovies, Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges.Missing column {column}",
            )


    data_clean = prepare_data(df)
    output = batch_file_predict(data_clean,df_initial)
    stream = io.StringIO()
    output.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv"
                                 )
    response.headers["Content-Disposition"] = "attachment; filename=predictions-export.csv"
    response.headers["Access-Control-Expose-Headers"] = "Content-Disposition"
    response.headers["predictions"]= output.to_json()
    return response
