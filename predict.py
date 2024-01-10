import sys
import pandas as pd 
import pickle
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import RobustScaler
from utils import *

pd.set_option('display.max_columns', 100)

VERBOSE=False
CONF_MATR=False
METRICS=False
SPLIT_DATASET=False

TEST_SIZE=0.20
RANDOM_STATE=2018

def main():
    #READ THE DATA
    data_df = pd.read_csv("creditcard.csv")
    if VERBOSE:
        print("Credit Card Fraud Detection data -  rows:",data_df.shape[0]," columns:", data_df.shape[1])
    with open('final_model.pickle', 'rb') as model_f:
        final_model = pickle.load(model_f)
    
    #PREPROCESS THE DATA
        
    #SCALING OF TIME AND AMOUNT FEATURES
    # RobustScaler is less prone to outliers.

    rob_scaler = RobustScaler()

    data_df['scaled_amount'] = rob_scaler.fit_transform(data_df['Amount'].values.reshape(-1,1))
    data_df['scaled_time'] = rob_scaler.fit_transform(data_df['Time'].values.reshape(-1,1))

    data_df.drop(['Time','Amount'], axis=1, inplace=True)
    scaled_amount = data_df['scaled_amount']
    scaled_time = data_df['scaled_time']

    data_df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
    data_df.insert(0, 'scaled_amount', scaled_amount)
    data_df.insert(1, 'scaled_time', scaled_time)
        #print(data_df.head())
    
    #DROP UNUSEFUL FEATURES
    data_df=data_df.drop('V15', axis=1)
    data_df=data_df.drop('V26', axis=1)
    #OPTIONAL SPLIT OF THE DATASET
    if SPLIT_DATASET:
        _, test_df=train_test_split(data_df,test_size=TEST_SIZE,random_state=RANDOM_STATE)
        if VERBOSE:
            print(f'Number of test samples: {test_df.shape[0]}')
        X_test=test_df.drop('Class', axis=1)
        Y_test=test_df['Class']
    else:
        X_test=data_df.drop('Class', axis=1)
        Y_test=data_df['Class']

    final_model_preds = final_model.predict(X_test)
    final_model_preds_proba=final_model.predict_proba(X_test)[:, 1]
    if CONF_MATR:
        #CONFUSION MATRIX
        create_confusion_matrix(Y_test, final_model_preds)
    if METRICS:
        compute_metrics(Y_test, final_model_preds, final_model_preds_proba)
    prediction = pd.DataFrame(final_model_preds, columns=['predictions']).to_csv('prediction.csv')
      
        
        
    
        
if __name__ == "__main__":
    main()

