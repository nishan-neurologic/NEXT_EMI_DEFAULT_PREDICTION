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
    data = data.drop("Default", axis=1)
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
            padded_account_data = account_data[-maxlen:, :]
        
        padded_data.append(padded_account_data)

    return np.array(padded_data)



def predict_default_for_csv(input_csv, output_csv, CAT_FEATURES, COLUMNS_TO_SCALE, encoder, scaler, model):
    # Read the input CSV file
    data = pd.read_csv(input_csv)
    
    # Preprocess the data
    preprocessed_data = preprocess_data(data, CAT_FEATURES, COLUMNS_TO_SCALE, encoder, scaler)
    
    # Get unique account IDs
    unique_account_ids = data['ACCOUNT_ID'].unique()
    
    # Prepare input sequences
    input_sequences = prepare_input_sequences(preprocessed_data, unique_account_ids)
    
    # Predict default status for each account
    predictions = []
    for account_id, input_data in zip(unique_account_ids, input_sequences):
        input_data = np.expand_dims(input_data, axis=0).astype('float32')
        input_data = input_data.reshape(input_data.shape[0], -1)
        prediction = model.predict_proba(input_data)
        predictions.append(bool(np.argmax(prediction)))
    
    # Add the predictions to the DataFrame
    account_id_to_prediction = dict(zip(unique_account_ids, predictions))
    data['PREDICTED_DEFAULT'] = data['ACCOUNT_ID'].map(account_id_to_prediction)
    
    # Save the DataFrame with the new column to a new CSV file
    data.to_csv(output_csv, index=False)

if __name__ == "__main__":
    input_csv = "/Users/nishanali/WorkSpace/arthan_emi_default_prediction/data/processed/merged.csv"
    output_csv = "output_data.csv"
    
    # Load the necessary components
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    with open(SCALAR_PATH, "rb") as f:
        scaler = pickle.load(f)
    
    # Call the function
    predict_default_for_csv(input_csv, output_csv, CAT_FEATURES, COLUMNS_TO_SCALE, encoder, scaler, loaded_model)
