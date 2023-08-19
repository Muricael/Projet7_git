import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
from my_utilities import custom_loss

import numpy as np
import pandas as pd

app = Flask(__name__)

##Load the model
model = pickle.load(open('xgb_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])

def predict_api():
    #data=request.json['data']
    #print(data)
    #output = model.predict(data)
    #print(output[0])
    #return jsonify(output[0])
    data = request.json.get('data')
    if not data:
        return jsonify({"error": "Data not provided"}), 400

    output = model.predict(data)
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)
