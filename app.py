import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__)

# load de pickle model
# rb is the argument to read the file
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def Home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # convert integers to float
    float_features = [float(x) for x in request.form.values()]
    # convert floats to array
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text=f"The flower species is {prediction}")


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000)
