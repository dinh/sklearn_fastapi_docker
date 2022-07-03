from pydantic import BaseModel, validator


class CustomerData(BaseModel):
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

class ChurnPrediction(BaseModel):
    label: str
    prediction: int
    probability: float
