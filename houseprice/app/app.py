from pydantic import BaseModel
from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()


model = joblib.load("models/gbr.joblib")
features = joblib.load("models/features.joblib")

class InputData(BaseModel):
    year: int
    age: int
    beds: int
    baths: float
    home_size: int
    parcel_size: int
    pool: float
    dist_cbd: float
    dist_lakes: float
    x_coord: float
    y_coord: int

@app.get('/')
def index():
    return {'message': 'house price Prediction'}

@app.post("/predict/")
def predict(data: InputData):

    input_data = pd.DataFrame([data.dict()])

    pred = model.predict(input_data)[0] 


    return {
        "Predicted Price": float(pred),
    }