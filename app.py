from flask import Flask, request, jsonify, render_template
import os
import joblib

from iris_model import train_and_save_model

app = Flask(__name__)

FLOWER_NAMES = ["Setosa", "Versicolor", "Virginica"]


def load_model():
    model_path = "iris_model.pkl"
    if not os.path.exists(model_path):
        train_and_save_model()
    return joblib.load(model_path)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = None
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form

    try:
        sepal_length = float(data.get("sepal_length"))
        sepal_width = float(data.get("sepal_width"))
        petal_length = float(data.get("petal_length"))
        petal_width = float(data.get("petal_width"))
    except Exception:
        return jsonify({"error": "Invalid or missing input values"}), 400

    sample = [sepal_length, sepal_width, petal_length, petal_width]
    model = load_model()
    pred = int(model.predict([sample])[0])
    return jsonify({"prediction": pred, "name": FLOWER_NAMES[pred]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
