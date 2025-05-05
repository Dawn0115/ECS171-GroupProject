from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load("example_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data["features"]
    prediction = model.predict([features])
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(debug=True)

