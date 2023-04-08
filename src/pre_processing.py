from config import DATA_ROOT, IMPORTANT_COLUMNS, PROCESSED_DATA_PATH
import pandas as pd


def pre_process_data():
    products=pd.read_csv(f'{DATA_ROOT}/datasetsloan_od_products.csv')
    ap=pd.read_csv(f'{DATA_ROOT}/datasetsaccount_profiles.csv')
    branches=pd.read_csv(f'{DATA_ROOT}/datasetsbranches.csv')
    branches=branches[['BRANCH_CODE','BRANCH_NAME','IS_HQ']]
    ap2=pd.merge(ap,branches,on=['BRANCH_CODE'],how='left')[['ACCOUNT_ID','BRANCH_CODE','PRODUCT_TYPE','PRODUCT_CODE','BRANCH_NAME','IS_HQ']]
    lof=pd.read_csv(f'{DATA_ROOT}/datasetsloan_od_fulfillments.csv')
    lor=pd.read_csv(f'{DATA_ROOT}/datasetsloan_od_repayments.csv')

    lor2=lor[IMPORTANT_COLUMNS]
    merged=pd.merge(lor2,ap2,on=['ACCOUNT_ID'],how='left')
    merged=merged.fillna('Blank')
    merged['IS_HQ']=merged['IS_HQ'].astype('str')
    merged['TRANSACTION_DATE']=pd.to_datetime(merged['TRANSACTION_DATE'])
    merged['VALUE_DATE']=pd.to_datetime(merged['VALUE_DATE'])
    merged['Default']=(merged['TRANSACTION_DATE']-merged['VALUE_DATE']).dt.days>2
    merged.to_csv(PROCESSED_DATA_PATH, index=False)
    return merged