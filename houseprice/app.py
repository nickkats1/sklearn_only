from flask import Flask,render_template,request
import joblib
import numpy as np
import os






app = Flask(__name__)
scaler = joblib.load("models/scaler.joblib")

model = joblib.load("models/model.joblib")


@app.route("/")
def home():
    return render_template("index.html")



@app.route("/predict", methods=["POST"])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    features_scaled = scaler.transform(final_features)
    prediction = model.predict(features_scaled)[0]
    return render_template("results.html", prediction=prediction)


if __name__ == "__main__":
    app.run(port=80)


