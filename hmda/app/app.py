from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel


model = joblib.load("models/lr_model.joblib")
scaler = joblib.load("models/scaler.joblib")



app = FastAPI()

class InputData(BaseModel):
    occupancy: int
    race: int
    sex: int
    income: float
    married: int
    credit_history: int
    di_ratio: float
    pmi_denied: int
    unverifiable: int
    pmi_sought: int
    vr: int






@app.get('/')
def index():
    return {'message': 'HMDA Default Prediction'}

@app.post("/predict/")
def predict(data: InputData):

    input_data = pd.DataFrame([data.dict()])
    data_scaled = scaler.transform(input_data)

    pred = model.predict(data_scaled)[0] 
    pred_prob = model.predict_proba(data_scaled)[0][1]

    return {
        "Default Prediction": int(pred),
        "Default Probability": round(pred_prob, 2)
    }