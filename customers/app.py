from flask import Flask, render_template, request
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline
from src.data_processing.data_ingestion import DataIngestion
from src.data_processing.data_transformation import DataTransformation
from src.config import load_config

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:

        data = {
            'app_id': float(request.form['app_id']),
            'ssn': float(request.form['ssn']),
            'zip_code': float(request.form['zip_code']),
            'income':float(request.form['income']),
            'homeownership':float(request.form['homeownership']),
            'purchases':float(request.form['purchases']),
            'credit_limit':float(request.form['credit_limit']),
            'fico':float(request.form['fico']),
            'num_late':float(request.form['num_late']),
            'past_def':float(request.form['past_def']),
            'num_bankruptcy':float(request.form['num_bankruptcy']),
            'avg_income':float(request.form['avg_income']),
            'density':float(request.form['density'])

        }


        input_data = pd.DataFrame([data])


        config = load_config()

 
        predict_pipeline = PredictPipeline(config)
        prediction = predict_pipeline.predict_pipeline(features=input_data)


        return render_template('result.html', prediction=prediction)

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(port=80,debug=True)