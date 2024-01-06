import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, plot


import gc
from datetime import datetime 
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn import svm
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb
import os
pd.set_option('display.max_columns', 100)

DEBUG=False
CHECK_DATA=False
PREPROCESS_DATA=False
RANDOMIZED_SEARCH=False

RFC=False
ABC=True

NUM_ESTIMATORS=100

#TRAIN/VALIDATION/TEST SPLIT
#VALIDATION
VALID_SIZE = 0.20 # simple validation using train_test_split
TEST_SIZE = 0.20 # test size using_train_test_split

#CROSS-VALIDATION
NUMBER_KFOLDS = 5 #number of KFolds for cross-validation



RANDOM_STATE = 2018

MAX_ROUNDS = 1000 #lgb iterations
EARLY_STOP = 50 #lgb early stop 
OPT_ROUNDS = 1000  #To be adjusted based on best validation rounds
VERBOSE_EVAL = 50 #Print out metric result

def create_confusion_matrix(Y,preds):
    cm = pd.crosstab(Y, preds, rownames=['Actual'], colnames=['Predicted'])
    fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
    sns.heatmap(cm, 
                xticklabels=['Not Fraud', 'Fraud'],
                yticklabels=['Not Fraud', 'Fraud'],
                annot=True,ax=ax1,
                linewidths=.2,linecolor="Darkblue", cmap="Blues")
    plt.title('Confusion Matrix', fontsize=14)
    plt.show()
    plt.close()

def compute_metrics(Y,preds,proba):
    #AUROC METRIC
    rfc_auroc=roc_auc_score(Y, proba)
    print(f'RFC AUROC score: {rfc_auroc}\n')
    #AUPRC OR AVERAGE-PRECISION METRIC (SUITABLE FOR UNBALANCED DATASETS)
    rfc_auprc=average_precision_score(Y, proba)
    print(f'RFC AUPRC score: {rfc_auprc}\n')
    #ACCURACY METRIC
    rfc_accuracy=accuracy_score(Y, preds)
    print(f'Accuracy score: {rfc_accuracy}\n')

def main():
    #READ THE DATA
    data_df = pd.read_csv("creditcard.csv")
    if CHECK_DATA:
        #CHECH THE DATA
        print("Credit Card Fraud Detection data -  rows:",data_df.shape[0]," columns:", data_df.shape[1])
        print(data_df.head())
        print(data_df.describe())
        #CHECH EVENTUAL NULL DATA (NOT FOUND)
        total = data_df.isnull().sum().sort_values(ascending = False)
        percent = (data_df.isnull().sum()/data_df.isnull().count()*100).sort_values(ascending = False)
        print(pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose())
        #CHECK DATA UNBALANCE
        print('No Frauds', round(data_df['Class'].value_counts()[0]/len(data_df) * 100,2), '% of the dataset')
        print('Frauds', round(data_df['Class'].value_counts()[1]/len(data_df) * 100,2), '% of the dataset')

        #PER CLASS AMOUNT
        tmp = data_df[['Amount','Class']].copy()
        class_0 = tmp.loc[tmp['Class'] == 0]['Amount']
        class_1 = tmp.loc[tmp['Class'] == 1]['Amount']
        print(f'Class 0:\n{class_0.describe()}\n')
        print(f'Class 1:\n{class_1.describe()}')
        '''
        As expected, the mean of the amounts of the Class 1 (fraudolent transactions) is higher than Class 0's one
        '''
        #FEATURES CORRELATION
        plt.figure(figsize = (14,14))
        plt.title('Credit Card Transactions features correlation plot (Pearson)')
        corr = data_df.corr()
        sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="Reds")
        plt.show()
        plt.close()
        '''
        As expected, there is no notable correlation between features V1-V28. 
        There are certain correlations between some of these features and Time 
        (inverse correlation with V3) and Amount (direct correlation with V7 and V20, 
        inverse correlation with V1 and V5).
        '''
        #FEATURES DENSITY
        #This one it takes a long time!
        var = data_df.columns.values
        i = 0
        t0 = data_df.loc[data_df['Class'] == 0]
        t1 = data_df.loc[data_df['Class'] == 1]

        sns.set_style('whitegrid')
        plt.figure()
        fig, ax = plt.subplots(8,4,figsize=(16,28))

        for feature in var:
            i += 1
            plt.subplot(8,4,i)
            sns.kdeplot(t0[feature], bw_method=0.5,label="Class = 0")
            sns.kdeplot(t1[feature], bw_method=0.5,label="Class = 1")
            plt.xlabel(feature, fontsize=12)
            locs, labels = plt.xticks()
            plt.tick_params(axis='both', which='major', labelsize=12)
        plt.show()
        '''
        For some of the features we can observe a good selectivity in terms of distribution for the two values of Class:
        V4, V11 have clearly separated distributions for Class values 0 and 1, V12, V14, V18 are partially separated,
        V1, V2, V3, V10 have a quite distinct profile, whilse V25, V26, V28 have similar profiles for the two values of Class.
        In general, with just few exceptions (Time and Amount), the features distribution for legitimate transactions (values of Class = 0)
        is centered around 0, sometime with a long queue at one of the extremities. In the same time, the fraudulent transactions (values of Class = 1) have a skewed (asymmetric) distribution.
        '''
    
    if PREPROCESS_DATA:
        #SCALING OF TIME AND AMOUNT FEATURES
        # RobustScaler is less prone to outliers.

        std_scaler = StandardScaler()
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
    
    X = data_df.drop('Class', axis=1)
    Y = data_df['Class']
    #SIMPLE SPLIT BETWEEN TRAINING SET AND TESTING SET
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=TEST_SIZE,random_state=RANDOM_STATE)
    #CHECK DATA UNBALANCE AFTER THE SPLIT
    print('Train No Frauds', round(Y_train.value_counts()[0]/len(Y_train) * 100,2), '% of the Train Set')
    print('Train Frauds', round(Y_train.value_counts()[1]/len(Y_train) * 100,2), '% of the Train Set\n')
    print('Test No Frauds', round(Y_test.value_counts()[0]/len(Y_test) * 100,2), '% of the Test Set')
    print('Test Frauds', round(Y_test.value_counts()[1]/len(Y_test) * 100,2), '% of the Test Set\n')
    if RFC:
        if RANDOMIZED_SEARCH:
            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
            # Number of features to consider at every split
            max_features = ['auto', 'sqrt']
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 4]
            # Method of selecting samples for training each tree
            bootstrap = [True, False]
            # Create the random grid
            random_grid = {'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'bootstrap': bootstrap}
            random_rfc = RandomizedSearchCV(estimator = RandomForestClassifier(), param_distributions = random_grid,
            n_iter = 100, cv = 3, verbose=2, random_state=RANDOM_STATE, n_jobs = -1)
            random_rfc.fit(X_train, Y_train)

            # tree best estimator
            rfc = random_rfc.best_estimator_
            print(random_rfc.best_params_)
            rfc_preds = rfc.predict(X_test)
            rfc_preds_proba=rfc.predict_proba(X_test)[:, 1]
            compute_metrics(Y_test, rfc_preds, rfc_preds_proba)
        else:
            #RANDOM FOREST CLASSIFIER
            rfc = RandomForestClassifier(verbose=1, n_estimators=NUM_ESTIMATORS)
            rfc.fit(X_train, Y_train)
            rfc_preds = rfc.predict(X_test)
            rfc_preds_proba=rfc.predict_proba(X_test)[:, 1]
            if DEBUG:
                #CONFUSION MATRIX
                create_confusion_matrix(Y_test, rfc_preds)
            compute_metrics(Y_test, rfc_preds, rfc_preds_proba)
        if RANDOMIZED_SEARCH:
            pass
        else:
            #RANDOM BALANCED FOREST CLASSIFIER
            rfc_bal = RandomForestClassifier(class_weight='balanced',verbose=1, n_estimators=NUM_ESTIMATORS)
            rfc_bal.fit(X_train, Y_train)
            rfc_bal_preds = rfc_bal.predict(X_test)
            rfc_bal_preds_proba=rfc_bal.predict_proba(X_test)[:, 1]
            if DEBUG:
                #CONFUSION MATRIX
                create_confusion_matrix(Y_test, rfc_bal_preds) 
            compute_metrics(Y_test, rfc_bal_preds, rfc_bal_preds_proba)
    if ABC:
        abc = AdaBoostClassifier(random_state=RANDOM_STATE, n_estimators=NUM_ESTIMATORS)
        abc.fit(X_train, Y_train)
        abc_preds = abc.predict(X_test)
        abc_preds_proba=abc.predict_proba(X_test)[:, 1]
        if DEBUG:
            #CONFUSION MATRIX
            create_confusion_matrix(Y_test, abc_preds)
        compute_metrics(Y_test, abc_preds, abc_preds_proba)
        
if __name__ == "__main__":
    main()

