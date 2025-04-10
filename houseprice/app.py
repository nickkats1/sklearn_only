from flask import Flask,render_template,request
import pickle
import numpy as np
import os


with open("models/gbr.pkl","rb") as f:
    model = pickle.load(f)


with open("models/scaler.pkl","rb") as f:
    scaler = pickle.load(f)



app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")



@app.route("/predict", methods=["POST"])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    features_scaled = scaler.transform(final_features)
    prediction = model.predict(features_scaled)[0]



    return render_template(
        "result.html", prediction='The Predicted Price: ${}'.format(np.round(prediction,2)))


if __name__ == "__main__":
    app.run(debug=True)


