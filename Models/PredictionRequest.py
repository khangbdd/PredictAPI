from pydantic import BaseModel
from typing import Optional

class PredictionRequest(BaseModel):
    predictionId: int
    frequency: str
    horizon_date: str
    horizon: int
    error_function: str
    approaches = []
    time_series = []
    


