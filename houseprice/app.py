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
            'year': float(request.form['year']),
            'age': float(request.form['age']),
            'beds': float(request.form['beds']),
            'baths':float(request.form['baths']),
            'home_size':float(request.form['home_size']),
            'parcel_size':float(request.form['parcel_size']),
            'pool':float(request.form['pool']),
            'dist_cbd':float(request.form['dist_cbd']),
            'dist_lakes':float(request.form['dist_lakes']),
            'x_coord':float(request.form['x_coord']),
            'y_coord':float(request.form['y_coord'])

        }


        input_data = pd.DataFrame([data])


        config = load_config()

 
        predict_pipeline = PredictPipeline(config)
        prediction = predict_pipeline.predict_pipeline(features=input_data)


        return render_template('results.html', prediction=prediction)

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
