import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from category_encoders import CatBoostEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
import pickle

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LeakyReLU, Bidirectional, LSTM, TimeDistributed
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import Adam


from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import Precision, Recall, AUC
from sklearn.utils import class_weight

import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from pre_processing import pre_process_data
from config import CAT_FEATURES, MODEL_PATH, COLUMNS_TO_SCALE, ENCODER_PATH, SCALAR_PATH

# Data preparation
def create_train_data():
    data = pre_process_data()
    print(f"DATA: {data.head()}")
    y = data['Default']
    # Encode categorical features
    encoder = CatBoostEncoder(cols=CAT_FEATURES)
    encoder.fit(data, y)
    # encoder.save(ENCODER_PATH)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)
    data = encoder.transform(data)

    # Get unique account ids
    unique_account_ids = data['ACCOUNT_ID'].unique()

    # Convert dates to ordinal values
    reference_date = pd.Timestamp('2000-01-01')
    data['VALUE_DATE'] = (pd.to_datetime(data['VALUE_DATE']) - reference_date).dt.days
    data['TRANSACTION_DATE'] = (pd.to_datetime(data['TRANSACTION_DATE']) - reference_date).dt.days

    # Calculate sequence lengths for each account
    sequence_lengths = data.groupby('ACCOUNT_ID').size()

    # Create masks for accounts with less than 5 entries
    mask_less_than_5 = sequence_lengths[sequence_lengths < 5].index
    mask_greater_than_equal_5 = sequence_lengths[sequence_lengths >= 5].index

    # Filter data using masks
    data_less_than_5 = data[data['ACCOUNT_ID'].isin(mask_less_than_5)]
    data_greater_than_equal_5 = data[data['ACCOUNT_ID'].isin(mask_greater_than_equal_5)]

    # Scale numerical features
    scaler = MinMaxScaler()

    scaled_data = data.copy()
    scaled_data[COLUMNS_TO_SCALE] = scaler.fit_transform(scaled_data[COLUMNS_TO_SCALE])
    with open(SCALAR_PATH, "wb") as f:
        pickle.dump(scaler, f)
    # Pad sequences with less than 5 entries
    padded_data = []
    for account_id in unique_account_ids:
        account_data = scaled_data[scaled_data['ACCOUNT_ID'] == account_id].values
        
        if account_id in mask_less_than_5:
            padded_account_data = pad_sequences([account_data], maxlen=5, dtype='float32', padding='post')[0]
        else:
            padded_account_data = account_data
        
        padded_data.append(padded_account_data)

    padded_data = np.array(padded_data)

    # Prepare the input and output data
    input_timesteps = 5
    output_timestep = 1

    train_x = []
    train_y = []

    for sequence in padded_data:
        for i in range(len(sequence) - input_timesteps - output_timestep + 1):
            train_x.append(sequence[i:i + input_timesteps, :-1])
            train_y.append(sequence[i + input_timesteps, -1])

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    return train_x, train_y


def pre_training():
    # Split the data into training and testing sets
    train_x, train_y = create_train_data()
    train_x = train_x.reshape(train_x.shape[0], -1)
    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.15, random_state=42)

    # Convert the data to TensorFlow tensors
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    # One-hot encode the target variable
    # num_classes = 2
    # y_train = to_categorical(y_train, num_classes=num_classes)
    # y_test = to_categorical(y_test, num_classes=num_classes)

    return train_y, X_train, X_test, y_train, y_test

def define_model_architecture(train_y):
    # model = Sequential()
    # model.add(LSTM(100, activation='relu', input_shape=input_shape, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    # model.add(LSTM(50, activation='relu', dropout=0.2, recurrent_dropout=0.2))
    # model.add(Dense(num_classes, activation='softmax'))

    # weights = class_weight.compute_sample_weight('balanced', train_y)
    # class_weights = {0: weights[0], 1: weights[1]}
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    neg_count = np.sum(train_y == 0)
    pos_count = np.sum(train_y == 1)
    ratio = neg_count / pos_count
    model = xgb.XGBClassifier(scale_pos_weight=ratio, n_jobs=-1)
    return model

def model_fit(persist=False):
    train_y, X_train, X_test, y_train, y_test = pre_training()
    model = define_model_architecture(train_y)
    # model.fit(X_train, y_train, epochs=50, batch_size=128, verbose=1, validation_data=(X_test, y_test), class_weight=class_weights)
    model.fit(X_train, y_train)
    if persist:
        model.save_model(MODEL_PATH)
        # model.save(MODEL_PATH)
    # _, accuracy = model.evaluate(X_test, y_test)
    # print('Accuracy: %.2f' % (accuracy * 100))
    
    y_pred = model.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred)
    # auc = roc_auc_score(y_test, np.argmax(y_pred_proba))
    print(f"TRAINING PERFORMANCE MEASURES:\n=================================")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-score: ", f1)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    # auc = roc_auc_score(y_test, np.argmax(y_pred_proba))
    print(f"TEST PERFORMANCE MEASURES:\n=================================")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-score: ", f1)

if __name__=="__main__":
    model_fit(True)