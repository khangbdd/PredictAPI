from fastapi import FastAPI
from Constant import *
from Logics.PredictionLogic import PredictionPresenter
from Models.PredictionRequest import PredictionRequest


import uvicorn
import asyncio

app = FastAPI()

@app.post("/api/v1/prediction")
async def createNewPrediction(request: PredictionRequest):
    presenter = PredictionPresenter()
    if(isValidRequest(request)):
        task = asyncio.create_task(presenter.predict(request))
        return 1
    return 0

def isValidRequest(request: PredictionRequest) -> bool:
    if(request.approaches.count == 0):
        return False
    if(request.time_series.count == 0):
        return False
    if(request.error_function != default_error_function):
        return False
    if(request.horizon <= 0):
        return False
    return True

if __name__ == "__main__":
    uvicorn.run(app, host = "127.0.0.1", port = 8000)