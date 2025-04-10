from flask import Flask,render_template,request
import pickle
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

scaler = pickle.load(open("models/scaler.pkl","rb"))
model = pickle.load(open("models/gbc.pkl","rb"))


app = Flask(__name__)



@app.route("/")
def home():
    return render_template("home.html")




@app.route("/predict",methods=["POST"])
def predict():
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    features_scaled = scaler.transform(final_features)
    pred_prob = model.predict_proba(features_scaled)[:,1]
    return render_template('result.html',pred_prob=(np.round(pred_prob,2)))


if __name__ == "__main__":
    app.run(debug=True)
    









