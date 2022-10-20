from fastapi import FastAPI

app = FastAPI()

@app.get("/hello")
async def readFirst():
    result = {"message":"Hello World!", "tax": 123 }
    return result