import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/loading")
def loading():
    return render_template("loading.html")

@flask_app.route("/name")
def name():
    return render_template("name.html")

@flask_app.route("/form", methods=["GET", "POST"])
def form():
    return render_template("form.html")



@flask_app.route("/predict", methods = ["GET","POST"])
def predict():
        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        prediction = model.predict(features)
        return render_template("predictt.html", prediction_text = "Your Predicted SGPA is {}".format(prediction))


if __name__ == "__main__":
    flask_app.run(debug=True)