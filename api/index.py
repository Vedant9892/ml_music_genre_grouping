from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "ML Music Genre App is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    # load model here
    return jsonify({"genre": "rock"})
