from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from category_encoders import CatBoostEncoder
from tensorflow.keras.models import load_model
import json
import pickle
import xgboost as xgb
import pprint
from config import MODEL_PATH, ENCODER_PATH, PROCESSED_DATA_PATH, CAT_FEATURES, COLUMNS_TO_SCALE, SCALAR_PATH

from inference import predict_default_wrapper

# Load the saved model
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

app = Flask(__name__)

data = pd.read_csv(PROCESSED_DATA_PATH)
encoder = CatBoostEncoder()
with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)
scaler = MinMaxScaler()
with open(SCALAR_PATH, "rb") as f:
    scaler = pickle.load(f)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)


@app.route('/predict', methods=['POST'])
def predict_default_status():
    account_ids = request.get_json()['account_ids']

    prediction = predict_default_wrapper(account_ids, data, CAT_FEATURES, COLUMNS_TO_SCALE, encoder, scaler, model)
    print(prediction)
    # return jsonify({k: v for k, v in prediction.items()})
    import json
    return json.dumps(prediction, cls=NumpyEncoder)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)

"""
import requests
import pprint
url = 'http://127.0.0.1:5000/predict'
account_ids = [
        30160400547, 
        101000000030, 
        201000000002, 
        101000000001
    ]
data = {'account_ids': account_ids}
response = requests.post(url, json=data)
pprint.pprint(response.json())
"""