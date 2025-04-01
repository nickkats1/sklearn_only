import joblib
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()


model = joblib.load("models/lasso.joblib")
features = joblib.load("models/features.joblib")

class InputData(BaseModel):
    log_age: float
    pas: int
    wtop: int
    fixgear: int
    tdrag: int
    log_horse: float
    log_fuel: float
    log_ceiling: float

@app.get("/")
async def route_message():
    return {"message": "Airplane Price Prediction"}

@app.post("/predict")
def predict(data: InputData):

    input_data = pd.DataFrame([data.dict()])
    

    input_data = input_data[features.columns]


    prediction = model.predict(input_data)[0]
    print(prediction)
    return str(prediction)