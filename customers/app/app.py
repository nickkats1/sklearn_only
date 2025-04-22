from flask import Flask,render_template,jsonify,request
import pickle
import numpy as np



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
    pred = model.predict(features_scaled)[0]



    return render_template(
        "index.html", prediction=pred)


if __name__ == "__main__":
    app.run(debug=True)




