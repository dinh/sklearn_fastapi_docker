import uvicorn
import pandas as pd
from fastapi import FastAPI,File,UploadFile,HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
# Third party imports
from pydantic import BaseModel, Field
from typing import List

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
    paperless_billing: int
    internet_service_fiber_optic: int
    no_internet_service: int
    online_security: int
    device_protection: int
    contract_month_to_month: int
    payment_method_electronic_check: int

    

    class Config:
        schema_extra = {
            'tenure': 29,
            'paperless_billing': 1,
            'internet_service_fiber_optic': 0,
            'no_internet_service': 1,
            'online_security': 0,
            'device_protection': 0,
            'contract_month_to_month': 0,
            'payment_method_electronic_check': 1
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
 

@api.post('/batch_predict',name="Batch File Churn Predict", tags=['Prediction Functions'],response_model=Result )
async def batch_predict(file: UploadFile = File(...)):
    """Predict with file input"""
    # Ensure that the file is a CSV
    if not file.content_type.startswith("application/vnd.ms-excel"):
        raise HTTPException(status_code=400, detail="File format provided is not valid.")
    contents = await file.read()
    buffer = BytesIO(contents)
    df = pd.read_csv(buffer)
    buffer.close()
    df_initial=df
    data_clean = prepare_data(df)
    response = batch_file_predict(data_clean,df_initial)
    name=file.filename
    result='data/result'
    prediction_result = f'{result}_{name}'
    response.to_csv(prediction_result,sep='\t')
    #return response.to_json()
    return {
        "filename": prediction_result,
        "content_type": 'CSV file',
        #"filename": file.filename,
        #"content_type": file.content_type,  
        "predictions": response.to_json()
    }


   
   

