from flask import Flask,render_template,request
import numpy as np
import joblib



app = Flask(__name__)

model = joblib.load("models/model.joblib")

@app.route("/")
def home():
    return render_template("index.html")




@app.route("/predict", methods=["POST"])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)[0]
    
    return render_template("result.html",prediction=prediction)

if __name__ == "__main__":
    app.run(port=8080)