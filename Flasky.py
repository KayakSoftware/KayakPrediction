from flask import Flask
from flask import request
from flask_cors import CORS
import json
from HAR_BRAIN import Har_Brain
from flask import jsonify

brain = Har_Brain()
app = Flask(__name__)
CORS(app)

class PredictionResult:

    def __init__(self, activity, confidence):
        self.activity = activity;
        self.confidence = confidence;

@app.route("/predict", methods=["GET", "POST"])
def predict():
    json = request.json
    requestData = json["data"]

    dataList = []
    for data in requestData:
        dataList.append([data["xAxis"], data["yAxis"], data["zAxis"]])

    prediction = brain.predict(dataList)

    result = [];

    for key in prediction:
        value = prediction[key]
        result.append([key, str(value)])

    return jsonify({"predictions": result})


@app.route("/test", methods=["GET", "POST"])
def test():
    return "Det virker!!!"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)