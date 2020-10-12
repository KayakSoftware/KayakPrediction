from flask import Flask
from flask import request
from flask_cors import CORS
import json
from HAR_BRAIN import Har_Brain

brain = Har_Brain()
app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    json = request.json
    requestData = json["data"]

    dataList = []
    for data in requestData:
        dataList.append([data["xAxis"], data["yAxis"], data["zAxis"]])

    prediction = brain.predict(dataList)
    return prediction


@app.route("/test", methods=["GET", "POST"])
def test():
    return "Det virker!!!"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)