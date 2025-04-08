from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel


with open("models/gbc_clf.pkl","rb") as f:
    model = pickle.load(f)


with open("models/scaler.pkl","rb") as f:
    scaler = pickle.load(f)



app = FastAPI()

class InputData(BaseModel):
    AccountWeeks:int
    ContractRenewal:int
    DataPlan:int
    DataUsage:float
    CustServCalls:int
    DayMins:float
    DayCalls:int
    MonthlyCharge:float
    OverageFee:float
    RoamMins:float


class DependentVariable(BaseModel):
    predID: str
    prediction: float



@app.get('/')
async def index():
    return {'message': 'Churn Prediction'}

@app.post("/predict/")
async def predict(data: InputData):

    input_data = pd.DataFrame([data.dict()])
    input_data_scaled = scaler.fit_transform(input_data)

    pred = model.predict(input_data_scaled)[0] 
    pred_prob = model.predict_proba(input_data_scaled)[0][1]

    return {
        "Customer Churn Prediction": int(pred),
        "Custom Churn Probability": round(pred_prob, 2)
}
