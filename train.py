import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, plot


import gc
from datetime import datetime 
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn import svm
import os
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
pd.set_option('display.max_columns', 100)

#COSE DA FA: K-FOLD, DOWNSAMPLING, UPSAMPLING, ENSEMBLE STRANI (CON C0 CLUSTER), 
#TUNING CLASSIFICATORI IPERPARAMETRI, SALVARE IL MODELLO FINALE, 
#AGGIORNARE IL README, FARE L'EVAL.PY, SCRIVERE E FARE COME SE FOSSE UN DEPLOYMENT

#BESTS: CBC -> RFC -> ABC
CHECK_DATA=False
CONF_MATR=False
K_FOLD=False
SMT=True

SCALE_TIME_AMOUNT=False
DROP_FEATURES=False #always a good idea
REMOVE_OUTLIERS=False
RANDOMIZED_SEARCH=False
PCAN=False
TSVD=False

RFC=False
ABC=False
CBC=True
LGBMC=False #improve with preprocessing, all parameters needed
LRC=False
SVMC=False#improve with preprocessing
KNNC=False

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
    print(f'Number of initial total samples: {data_df.size}')
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
    if SCALE_TIME_AMOUNT:
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
    if DROP_FEATURES:
        data_df=data_df.drop('V15', axis=1)
        data_df=data_df.drop('V26', axis=1)

    train_df, test_df=train_test_split(data_df,test_size=TEST_SIZE,random_state=RANDOM_STATE)
    print(f'Number of train samples: {train_df.size}')
    print(f'Number of test samples: {test_df.size}')

    if REMOVE_OUTLIERS:
        #  -----> Removing Outliers
        for k in train_df.keys() :
            if k not in ['Time', 'Amount', 'Class']:
                feat = train_df[k].loc[train_df['Class'] == 0].values
                q25, q75 = np.percentile(feat, 25), np.percentile(feat, 75)
                #print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
                feat_iqr = q75 - q25
                #print('iqr: {}'.format(feat_iqr))

                feat_cut_off = feat_iqr * 2.5
                feat_lower, feat_upper = q25 - feat_cut_off, q75 + feat_cut_off
                #print('Cut Off: {}'.format(feat_cut_off))
                #print('feat Lower: {}'.format(feat_lower))
                #print('feat Upper: {}'.format(feat_upper))

                train_df = train_df.drop(train_df[((train_df[k] > feat_upper) | (train_df[k] < feat_lower)) & (train_df['Class']==0)].index)
                #print('----' * 44)
        print(f'Number of train samples after removing Class 0 outliers: {train_df.size}')


    #SIMPLE SPLIT BETWEEN TRAINING SET AND TESTING SET
    X_train=train_df.drop('Class', axis=1)
    X_test=test_df.drop('Class', axis=1)
    Y_train=train_df['Class']
    Y_test=test_df['Class']
    #CHECK DATA UNBALANCE AFTER THE SPLIT AND PREPROCESSING
    print('Train No Frauds', round(Y_train.value_counts()[0]/len(Y_train) * 100,2), '% of the Train Set,',Y_train.value_counts()[0],' samples\n')
    print('Train Frauds', round(Y_train.value_counts()[1]/len(Y_train) * 100,2), '% of the Train Set,',Y_train.value_counts()[1],' samples\n')
    print('Test No Frauds', round(Y_test.value_counts()[0]/len(Y_test) * 100,2), '% of the Test Set,',Y_test.value_counts()[0],' samples\n')
    print('Test Frauds', round(Y_test.value_counts()[1]/len(Y_test) * 100,2), '% of the Test Set,',Y_test.value_counts()[1],' samples\n')
    if PCAN:
        # PCA Implementation
        X_train = PCA(n_components=(len(X_train.keys())-2), random_state=RANDOM_STATE).fit_transform(X_train)
        X_test = PCA(n_components=(len(X_test.keys())-2), random_state=RANDOM_STATE).fit_transform(X_test)
    if TSVD:
        # TruncatedSVD
        X_train= TruncatedSVD(n_components=(len(X_train.keys())-2), algorithm='randomized', random_state=RANDOM_STATE).fit_transform(X_train)
        X_test= TruncatedSVD(n_components=(len(X_test.keys())-2), algorithm='randomized', random_state=RANDOM_STATE).fit_transform(X_test)
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
            if CONF_MATR:
                #CONFUSION MATRIX
                create_confusion_matrix(Y_test, rfc_preds)
            compute_metrics(Y_test, rfc_preds, rfc_preds_proba)

    if ABC:
        abc = AdaBoostClassifier(random_state=RANDOM_STATE, n_estimators=NUM_ESTIMATORS)
        abc.fit(X_train, Y_train)
        abc_preds = abc.predict(X_test)
        abc_preds_proba=abc.predict_proba(X_test)[:, 1]
        if CONF_MATR:
            #CONFUSION MATRIX
            create_confusion_matrix(Y_test, abc_preds)
        compute_metrics(Y_test, abc_preds, abc_preds_proba)

    if CBC:
        cbc = CatBoostClassifier(random_seed = RANDOM_STATE, metric_period = VERBOSE_EVAL)
        '''
        cbc = CatBoostClassifier(iterations=1000,
                             learning_rate=0.02,
                             depth=12,
                             eval_metric='AUC',
                             random_seed = RANDOM_STATE,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = VERBOSE_EVAL,
                             od_wait=100)
        '''
        if K_FOLD:
            sss = StratifiedKFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
            for train_index, val_index in sss.split(X_train, Y_train):
                #print("Train:", train_index, "Val:", val_index)
                sss_X_train, sss_X_val = X_train.iloc[train_index], X_train.iloc[val_index]
                sss_Y_train, sss_Y_val = Y_train.iloc[train_index], Y_train.iloc[val_index]
                if SMT:
                    smt=SMOTE(sampling_strategy='minority', random_state=RANDOM_STATE)
                    sss_smt_X_train, sss_smt_Y_train = smt.fit_resample(sss_X_train, sss_Y_train)
                    cbc.fit(sss_smt_X_train, sss_smt_Y_train)
                else:
                    cbc.fit(sss_X_train, sss_Y_train)
        else:
            if SMT:
                smt=SMOTE(sampling_strategy='minority', random_state=RANDOM_STATE)
                smt_X_train, smt_Y_train = smt.fit_resample(X_train, Y_train)
                cbc.fit(smt_X_train, smt_Y_train)
            else:
                cbc.fit(X_train, Y_train)
        cbc_preds = cbc.predict(X_test)
        cbc_preds_proba=cbc.predict_proba(X_test)[:, 1]
        if CONF_MATR:
            #CONFUSION MATRIX
            create_confusion_matrix(Y_test, cbc_preds)
        compute_metrics(Y_test, cbc_preds, cbc_preds_proba)
    if LGBMC:
        lgbmc = LGBMClassifier(
                  nthread=-1,
                  n_estimators=2000,
                  learning_rate=0.01,
                  num_leaves=80,
                  colsample_bytree=0.98,
                  subsample=0.78,
                  reg_alpha=0.04,
                  reg_lambda=0.073,
                  subsample_for_bin=50,
                  boosting_type='gbdt',
                  is_unbalance=False,
                  min_split_gain=0.025,
                  min_child_weight=40,
                  min_child_samples=510,
                  objective='binary',
                  metric='auc',
                  silent=-1,
                  verbose=-1,
                  feval=None)
        lgbmc.fit(X_train, Y_train)
        lgbmc_preds = lgbmc.predict(X_test)
        lgbmc_preds_proba=lgbmc.predict_proba(X_test)[:, 1]
        if CONF_MATR:
            #CONFUSION MATRIX
            create_confusion_matrix(Y_test, lgbmc_preds)
        compute_metrics(Y_test, lgbmc_preds, lgbmc_preds_proba)
    if LRC:
        lrc = LogisticRegression()
        lrc.fit(X_train, Y_train)
        lrc_preds = lrc.predict(X_test)
        lrc_preds_proba=lrc.predict_proba(X_test)[:, 1]
        if CONF_MATR:
            #CONFUSION MATRIX
            create_confusion_matrix(Y_test, lrc_preds)
        compute_metrics(Y_test, lrc_preds, lrc_preds_proba)
    if SVMC:#RBF
        svmc = svm.SVC(probability=True, kernel='rbf')
        svmc.fit(X_train, Y_train)
        svmc_preds = svmc.predict(X_test)
        svmc_preds_proba=svmc.predict_proba(X_test)[:, 1]
        if CONF_MATR:
            #CONFUSION MATRIX
            create_confusion_matrix(Y_test, svmc_preds)
        compute_metrics(Y_test, svmc_preds, svmc_preds_proba)
    if KNNC:
        knnc = KNeighborsClassifier()
        knnc.fit(X_train, Y_train)
        knnc_preds = knnc.predict(X_test)
        knnc_preds_proba=knnc.predict_proba(X_test)[:, 1]
        if CONF_MATR:
            #CONFUSION MATRIX
            create_confusion_matrix(Y_test, knnc_preds)
        compute_metrics(Y_test, knnc_preds, knnc_preds_proba)     
        
        
    
        
if __name__ == "__main__":
    main()

