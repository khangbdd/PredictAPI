from pydantic import BaseModel

class PredictionResponse(BaseModel):
    predictionId: int
    results = []