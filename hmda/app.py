from flask import Flask, render_template, request
import pandas as pd
from src.pipeline.Predict_Pipeline import PredictPipeline
from src.preprocess.data_ingestion import DataIngestion
from src.preprocess.data_transformation import DataTransformation
from src.config import load_config

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:

        data = {
            'occupancy': float(request.form['occupancy']),
            'race': float(request.form['race']),
            'sex': float(request.form['sex']),
            'income': float(request.form['income']),
            'married':float(request.form['married']),
            'credit_history':float(request.form['credit_history']),
            'di_ratio':float(request.form['di_ratio']),
            'pmi_denied':float(request.form['pmi_denied']),
            'unverifiable':float(request.form['unverifiable']),
            'pmi_sought':float(request.form['pmi_sought']),
            'vr':float(request.form['vr'])

        }


        input_data = pd.DataFrame([data])


        config = load_config()

 
        predict_pipeline = PredictPipeline(config)
        pred_prob = predict_pipeline.predict(features=input_data)[0][0]
 


        return render_template('result.html', pred_prob=pred_prob)

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True,port=88)