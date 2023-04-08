import os

DATA_ROOT="../data/raw_data"
MODEL_PATH = "../data/model/xgb_model.model"
PROCESSED_DATA_PATH = "../data/processed/merged.csv"
ENCODER_PATH = "../data/encoder/cat_boost_encoder.pkl"
SCALAR_PATH = "../data/encoder/scalar.pkl"

os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs("../data/model", exist_ok=True)
os.makedirs("../data/processed", exist_ok=True)
os.makedirs("../data/encoder", exist_ok=True)

IMPORTANT_COLUMNS=['ACCOUNT_ID', 'REPAYMENT_NUM',
       'REPAYMENT_TYPE', 'AMOUNT_MAGNITUDE','BALANCE_MAGNITUDE','PRINCIPAL_MAGNITUDE','NORMAL_INTEREST_MAGNITUDE',
       'PENAL_INTEREST_MAGNITUDE',
       'PRINCIPAL_ADJUSTMENT_MAGNITUDE',
       'NORMAL_INTEREST_ADJUSTMENT_MAGNITUDE',
       'PENAL_INTEREST_ADJUSTMENT_MAGNITUDE',
       'NORMAL_INTEREST_WAIVER_MAGNITUDE',
       'PENAL_INTEREST_WAIVER_MAGNITUDE',
       'INTEREST_TDS_MAGNITUDE',
       'NORMAL_INTEREST_TDS_MAGNITUDE', 
       'PENAL_INTEREST_TDS_MAGNITUDE', 
       'LAST_SATISFIED_DEMAND_NUM', 'TRANSACTION_ID', 'VALUE_DATE',
       'TRANSACTION_DATE', 'INSTRUMENT', 'OLD_DPD', 'DPD', 'USER_ID']

CAT_FEATURES = ['TRANSACTION_ID', 'USER_ID', 'BRANCH_CODE', 'PRODUCT_TYPE', 'PRODUCT_CODE', 'BRANCH_NAME', 'IS_HQ']

COLUMNS_TO_SCALE = ['AMOUNT_MAGNITUDE', 'BALANCE_MAGNITUDE', 'PRINCIPAL_MAGNITUDE', 'NORMAL_INTEREST_MAGNITUDE',
                    'PENAL_INTEREST_MAGNITUDE', 'PRINCIPAL_ADJUSTMENT_MAGNITUDE',
                    'NORMAL_INTEREST_ADJUSTMENT_MAGNITUDE', 'PENAL_INTEREST_ADJUSTMENT_MAGNITUDE',
                    'NORMAL_INTEREST_WAIVER_MAGNITUDE', 'PENAL_INTEREST_WAIVER_MAGNITUDE',
                    'INTEREST_TDS_MAGNITUDE', 'NORMAL_INTEREST_TDS_MAGNITUDE', 'PENAL_INTEREST_TDS_MAGNITUDE']
