from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel


model = joblib.load("models/gbr_model.joblib")

features = joblib.load("models/features.joblib")


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



class DependentVariable(BaseModel):
    predID: str
    prediction: float



@app.get('/')
def index():
    return {'message': 'HMDA Default Prediction'}

@app.post("/predict/")
def predict(data: InputData):

    input_data = pd.DataFrame([data.dict()])

    pred = model.predict(input_data)[0] 
    pred_prob = model.predict_proba(input_data)[0][1]

    return {
        "Default Prediction": int(pred),
        "Default Probability": round(pred_prob, 2)
    }