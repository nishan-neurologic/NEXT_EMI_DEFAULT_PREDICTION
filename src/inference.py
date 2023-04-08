import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from category_encoders import CatBoostEncoder
from tensorflow.keras.models import load_model
import pickle
from collections import Counter

import xgboost as xgb

from config import MODEL_PATH, PROCESSED_DATA_PATH, MODEL_PATH, COLUMNS_TO_SCALE, CAT_FEATURES, ENCODER_PATH, SCALAR_PATH


loaded_model = xgb.XGBClassifier()
loaded_model.load_model(MODEL_PATH)

def preprocess_data(data, cat_features, COLUMNS_TO_SCALE, encoder, scaler):
    # Encode categorical features
    data = encoder.transform(data)

    # Convert dates to ordinal values
    reference_date = pd.Timestamp('2000-01-01')
    data['VALUE_DATE'] = (pd.to_datetime(data['VALUE_DATE']) - reference_date).dt.days
    data['TRANSACTION_DATE'] = (pd.to_datetime(data['TRANSACTION_DATE']) - reference_date).dt.days

    # Scale numerical features
    scaled_data = data.copy()
    scaled_data[COLUMNS_TO_SCALE] = scaler.transform(scaled_data[COLUMNS_TO_SCALE])

    return scaled_data

def prepare_input_sequences(data, unique_account_ids, maxlen=5):
    padded_data = []
    for account_id in unique_account_ids:
        account_data = data[data['ACCOUNT_ID'] == account_id].values
        
        if len(account_data) < maxlen:
            padded_account_data = pad_sequences([account_data], maxlen=maxlen, dtype='float32', padding='post')[0]
        else:
            padded_account_data = account_data
        
        padded_data.append(padded_account_data)

    return np.array(padded_data)

def predict_default(account_id, data, cat_features, COLUMNS_TO_SCALE, encoder, scaler, model):
    # Filter data by account_id
    result = dict()
    account_data = data[data['ACCOUNT_ID'] == account_id]

    # Preprocess the account data
    preprocessed_data = preprocess_data(account_data, cat_features, COLUMNS_TO_SCALE, encoder, scaler)

    # Pad the sequence if it has less than 5 entries
    account_sequence = preprocessed_data.values
    if len(account_sequence) < 5:
        account_sequence = pad_sequences([account_sequence], maxlen=5, dtype='float32', padding='post')[0]

    # Prepare the input data for the model by selecting the last 5 timesteps
    input_data = account_sequence[-5:, :-1]

    
    # Add an extra dimension to match the model's input shape (batch_size, timesteps, features)
    input_data = np.expand_dims(input_data, axis=0).astype('float32')
    input_data = input_data.reshape(input_data.shape[0], -1)

    prediction = model.predict_proba(input_data)
    
    result["PREDICTION"] = bool(np.argmax(prediction))
    result["ACCOUNT_ID"] = account_id
    result["LATEST_REPAYMENT_NUM"] = list(account_data["REPAYMENT_NUM"])[-1]
    # count = Counter(account_data["Default"][-5:])
    # max_occuring_value = max(count, key=count.get)
    # print(f"MAX VALUE: {max_occuring_value}")
    # result["ACTUAL"] = True if count[True]>=2 else False
    return np.argmax(prediction), result

def predict_default_wrapper(account_ids, data, CAT_FEATURES, COLUMNS_TO_SCALE, encoder, scaler, model):
    result = dict()
    for account_id in account_ids:
        _, result[account_id] = predict_default(account_id, data, CAT_FEATURES, COLUMNS_TO_SCALE, encoder, scaler, model)
    return result

if __name__=="__main__":
    account_ids = [
        30160400547, 
        101000000030, 
        201000000002, 
        101000000001
    ]
    # model = load_model(MODEL_PATH)
    loaded_model = xgb.XGBClassifier()
    loaded_model.load_model(MODEL_PATH)

    data = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"DATA: {data.head()}")
    print(f"COLUMNS: {data.columns}")
    encoder = CatBoostEncoder()
    # encoder.load(ENCODER_PATH)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    scaler = MinMaxScaler()
    with open(SCALAR_PATH, "rb") as f:
        scaler = pickle.load(f)
    prediction = predict_default_wrapper(account_ids, data, CAT_FEATURES, COLUMNS_TO_SCALE, encoder, scaler, loaded_model)
    print(f"The predicted default status for account ID {account_ids} is: {prediction}")