from fastapi import FastAPI
from Models.PredictionRequest import PredictionRequest
import uvicorn

app = FastAPI()

@app.post("/api/v1/prediction")
async def createNewPrediction(request: PredictionRequest):
    print(request)
    return 1

if __name__ == "__main__":
    uvicorn.run(app, host = "127.0.0.1", port = 8000)