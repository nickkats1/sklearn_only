from flask import Flask,render_template,request
import joblib
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)


model = joblib.load("models/model.joblib")


app = Flask(__name__)



@app.route("/")
def home():
    return render_template("home.html")




@app.route("/predict",methods=["POST"])
def predict():
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    pred_prob = model.predict_proba(final_features)[:,1]
    return render_template('result.html',pred_prob=(np.round(pred_prob,2)))


if __name__ == "__main__":
    app.run(debug=True)