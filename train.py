import sys
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
from sklearn.neural_network import MLPClassifier
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, plot


import gc
from datetime import datetime 
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, silhouette_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn import svm
import os
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE,KMeansSMOTE,ADASYN
from imblearn.under_sampling import NearMiss
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import EasyEnsembleClassifier, RUSBoostClassifier, BalancedRandomForestClassifier, BalancedBaggingClassifier
from imblearn.metrics import classification_report_imbalanced
pd.set_option('display.max_columns', 100)

#COSE DA FA: K-FOLD, DOWNSAMPLING, UPSAMPLING, ENSEMBLE STRANI (CON C0 CLUSTER), 
#TUNING CLASSIFICATORI IPERPARAMETRI, SALVARE IL MODELLO FINALE, 
#AGGIORNARE IL README, FARE L'EVAL.PY, SCRIVERE E FARE COME SE FOSSE UN DEPLOYMENT

#BESTS: CBC -> XGBC -> RFC -> MLPC -> LGBMC 
CHECK_DATA=False
CONF_MATR=True
UPSAMP=False
DOWNSAMP=False
UPDOWNSAMP=False #it takes a little long

SCALE_TIME_AMOUNT=True
DROP_FEATURES=True #always a good idea
REMOVE_OUTLIERS=False
PCAN=False
TSVD=False

RANDOMIZED_SEARCH=True
K_FOLD=False

NC=False
RC=False

RFC=False
ABC=False
CBC=False #20 - 2
XGBC=True #20 - 3
LGBMC=False#improve with preprocessing, all parameters needed
LRC=False
SVMC=False#improve with preprocessing
KNNC=False#improve with preprocessing
MLPC=False#improve with preprocessing

EEC=False
RBC=False
BRFC=False
BBC=False

INTER_WRONG_PRED=False
ENSEMBLE=False #19 3

#TRAIN/TEST SPLIT
TEST_SIZE = 0.20 # test size using_train_test_split

#CROSS-VALIDATION
NUMBER_KFOLDS = 5 #number of KFolds for cross-validation

#Common Classifiers parameters
NUM_ESTIMATORS=100
RANDOM_STATE = 2018
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

def find_intersection_of_wrong_predictions(ground_truth, pred1, pred2):
    # Find indices where both classifiers make incorrect predictions
    incorrect_pred1 = np.where(pred1 != ground_truth)[0]
    print(f'Number of wrong predictions of Classifier1: {len(incorrect_pred1)}')
    incorrect_pred2 = np.where(pred2 != ground_truth)[0]
    print(f'Number of wrong predictions of Classifier2: {len(incorrect_pred2)}')

    # Find the intersection of the sets of incorrect predictions
    intersection = set(incorrect_pred1) & set(incorrect_pred2)
    print(f'Number of the intersection of the wrong predictions of both(the less the better): {len(intersection)}')


def main():
    #READ THE DATA
    data_df = pd.read_csv("creditcard.csv")
    print("Credit Card Fraud Detection data -  rows:",data_df.shape[0]," columns:", data_df.shape[1])
    if CHECK_DATA:
        #CHECH THE DATA
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
    print(f'Number of train samples: {train_df.shape[0]}')
    print(f'Number of test samples: {test_df.shape[0]}')

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
    if PCAN:
        # PCA Implementation
        X_train = PCA(n_components=(len(X_train.keys())-2), random_state=RANDOM_STATE).fit_transform(X_train)
        X_test = PCA(n_components=(len(X_test.keys())-2), random_state=RANDOM_STATE).fit_transform(X_test)
    if TSVD:
        # TruncatedSVD
        X_train= TruncatedSVD(n_components=(len(X_train.keys())-2), algorithm='randomized', random_state=RANDOM_STATE).fit_transform(X_train)
        X_test= TruncatedSVD(n_components=(len(X_test.keys())-2), algorithm='randomized', random_state=RANDOM_STATE).fit_transform(X_test)
    if CHECK_DATA:
        #SILHOUETTE SCORE FOR K-MEANS OF THE CLASS 0 OF TRAIN SET
        #This one it takes a long time!
        X_train_0=X_train[Y_train==0]
        scores = []
        for k in range(2, 10):
            sc = KMeans(random_state=RANDOM_STATE,n_clusters=k)
            labels=sc.fit_predict(X_train_0)
            score = silhouette_score(X_train_0, labels)#sc.labels_
            scores.append(score)
        
        plt.plot(range(2, 10), scores)
        plt.title('Silhouette Score Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.show()
    #CHECK DATA UNBALANCE AFTER THE SPLIT AND PREPROCESSING
    print('Train No Frauds', round(Y_train.value_counts()[0]/len(Y_train) * 100,2), '% of the Train Set,',Y_train.value_counts()[0],' samples\n')
    print('Train Frauds', round(Y_train.value_counts()[1]/len(Y_train) * 100,2), '% of the Train Set,',Y_train.value_counts()[1],' samples\n')
    print('Test No Frauds', round(Y_test.value_counts()[0]/len(Y_test) * 100,2), '% of the Test Set,',Y_test.value_counts()[0],' samples\n')
    print('Test Frauds', round(Y_test.value_counts()[1]/len(Y_test) * 100,2), '% of the Test Set,',Y_test.value_counts()[1],' samples\n')
    classificators=list()
    predictions=list()
    probabilities=list()
    if NC:
        #FOR THE BASELINE
        class AlwaysClassZeroClassifier:
            def fit(self, X, Y):
                pass  # No training needed

            def predict(self, X):
                return [0] * len(X)
            
            def predict_proba(self, X):
                probs=list()
                for i in range(len(X)):
                    probs.append([1.0,0.0])
                probs=np.array(probs)
                return probs
            
        nc = AlwaysClassZeroClassifier()
        nc.fit(X_train, Y_train)
        nc_preds = nc.predict(X_test)
        nc_preds_proba=nc.predict_proba(X_test)[:, 1]
        if CONF_MATR:
            #CONFUSION MATRIX
            create_confusion_matrix(Y_test, nc_preds)
        compute_metrics(Y_test, nc_preds, nc_preds_proba)

    if RC:
        class RandomClassifier:
            def fit(self, X, Y):
                self.probs_=None

            def predict(self, X):
                prob_one = np.random.rand(len(X))
                prob_zero = 1.0 - prob_one 
                self.probs_ = np.column_stack((prob_zero, prob_one))
                return np.argmax(self.probs_, axis=1)

            def predict_proba(self, X):
                if self.probs_ is not None:
                    return self.probs_
                else:
                    prob_one = np.random.rand(len(X))
                    prob_zero = 1.0 - prob_one 
                    self.probs_ = np.column_stack((prob_zero, prob_one))
                    return self.probs_
        rc = RandomClassifier()
        rc.fit(X_train, Y_train)
        rc_preds = rc.predict(X_test)
        rc_preds_proba=rc.predict_proba(X_test)[:, 1]
        if CONF_MATR:
            #CONFUSION MATRIX
            create_confusion_matrix(Y_test, rc_preds)
        compute_metrics(Y_test, rc_preds, rc_preds_proba)

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
            space = {'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'bootstrap': bootstrap}
            random_rfc = RandomizedSearchCV(estimator = RandomForestClassifier(), param_distributions = space,
            n_iter = 100, cv = 3, verbose=2, random_state=RANDOM_STATE, n_jobs = -1)
            random_rfc.fit(X_train, Y_train)

            # tree best estimator
            rfc = random_rfc.best_estimator_
            print(random_rfc.best_params_)
            rfc_preds = rfc.predict(X_test)
            rfc_preds_proba=rfc.predict_proba(X_test)[:, 1]
            compute_metrics(Y_test, rfc_preds, rfc_preds_proba)
            predictions.append(('RFC',rfc_preds))
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
            predictions.append(('RFC',rfc_preds))

    if ABC:
        model = CatBoostClassifier(random_seed = RANDOM_STATE, metric_period = VERBOSE_EVAL)
        '''
        model = LogisticRegression()
        model = RandomForestClassifier(verbose=1, n_estimators=NUM_ESTIMATORS)
        
        model = LGBMClassifier(
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
        
        model = CatBoostClassifier(iterations=1000,
                             learning_rate=0.02,
                             depth=12,
                             eval_metric='AUC',
                             random_seed = RANDOM_STATE,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = VERBOSE_EVAL,
                             od_wait=100)
        '''
        abc = AdaBoostClassifier(estimator=model,random_state=RANDOM_STATE)#, n_estimators=NUM_ESTIMATORS
        abc.fit(X_train, Y_train)
        abc_preds = abc.predict(X_test)
        abc_preds_proba=abc.predict_proba(X_test)[:, 1]
        if CONF_MATR:
            #CONFUSION MATRIX
            create_confusion_matrix(Y_test, abc_preds)
        compute_metrics(Y_test, abc_preds, abc_preds_proba)
        predictions.append(('ABC',abc_preds))

    if CBC:
        '''
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
        
        if K_FOLD:
            sss = StratifiedKFold(n_splits=NUMBER_KFOLDS, random_state=RANDOM_STATE, shuffle=True)
            for train_index, val_index in sss.split(X_train, Y_train):
                #print("Train:", train_index, "Val:", val_index)
                sss_X_train, sss_X_val = X_train.iloc[train_index], X_train.iloc[val_index]
                sss_Y_train, sss_Y_val = Y_train.iloc[train_index], Y_train.iloc[val_index]
                if UPSAMP:
                    upsamp=ADASYN(sampling_strategy='minority', random_state=RANDOM_STATE)
                    sss_upsamp_X_train, sss_upsamp_Y_train = upsamp.fit_resample(sss_X_train, sss_Y_train)
                    cbc.fit(sss_upsamp_X_train, sss_upsamp_Y_train)
                elif DOWNSAMP:
                    downsamp=NearMiss(sampling_strategy='majority', version=3)
                    sss_downsamp_X_train, sss_downsamp_Y_train = downsamp.fit_resample(sss_X_train, sss_Y_train)
                    cbc.fit(sss_downsamp_X_train, sss_downsamp_Y_train)
                elif UPDOWNSAMP:
                    updownsamp=SMOTEENN(sampling_strategy='minority', random_state=RANDOM_STATE)
                    sss_updownsamp_X_train, sss_updownsamp_Y_train = updownsamp.fit_resample(sss_X_train, sss_Y_train)
                    cbc.fit(sss_updownsamp_X_train, sss_updownsamp_Y_train)
                else:
                    cbc.fit(sss_X_train, sss_Y_train)
        else:
            if UPSAMP:
                upsamp=ADASYN(sampling_strategy='minority', random_state=RANDOM_STATE)
                upsamp_X_train, upsamp_Y_train = upsamp.fit_resample(X_train, Y_train)
                cbc.fit(upsamp_X_train, upsamp_Y_train)
            elif DOWNSAMP:
                downsamp=NearMiss(sampling_strategy='majority',version=3)
                downsamp_X_train, downsamp_Y_train = downsamp.fit_resample(X_train, Y_train)
                cbc.fit(downsamp_X_train, downsamp_Y_train)
            elif UPDOWNSAMP:
                updownsamp=SMOTEENN(sampling_strategy='minority', random_state=RANDOM_STATE)#, smote=upsamp
                updownsamp_X_train, updownsamp_Y_train = updownsamp.fit_resample(X_train, Y_train)
                cbc.fit(updownsamp_X_train, updownsamp_Y_train)
            else:
                cbc.fit(X_train, Y_train)
        cbc_preds = cbc.predict(X_test)
        cbc_preds_proba=cbc.predict_proba(X_test)[:, 1]
        if CONF_MATR:
            #CONFUSION MATRIX
            create_confusion_matrix(Y_test, cbc_preds)
        compute_metrics(Y_test, cbc_preds, cbc_preds_proba)
        predictions.append(('CBC',cbc_preds))
    if XGBC:
        if RANDOMIZED_SEARCH:
            space = {
            'max_depth':range(3,10,2),
            'min_child_weight':range(1,6,2),
            'gamma':[i/10.0 for i in range(0,5)],
            'subsample':[i/10.0 for i in range(6,10)],
            'colsample_bytree':[i/10.0 for i in range(6,10)],
            'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
            }
            random_xgbc = RandomizedSearchCV(estimator = XGBClassifier(), param_distributions = space,
            n_iter = 100, cv = 3, verbose=2, random_state=RANDOM_STATE, n_jobs = -1)
            random_xgbc.fit(X_train, Y_train)

            # tree best estimator
            xgbc = random_xgbc.best_estimator_
            print(random_xgbc.best_params_)
        else:
            xgbc = XGBClassifier(random_state=RANDOM_STATE,verbosity=2)#n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic'
            xgbc.fit(X_train, Y_train)
        xgbc_preds = xgbc.predict(X_test)
        xgbc_preds_proba=xgbc.predict_proba(X_test)[:, 1]
        if CONF_MATR:
            #CONFUSION MATRIX
            create_confusion_matrix(Y_test, xgbc_preds)
        compute_metrics(Y_test, xgbc_preds, xgbc_preds_proba)
        predictions.append(('XGBC',xgbc_preds))
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
        predictions.append(('LGBMC',lgbmc_preds))
    if LRC:
        lrc = LogisticRegression()
        lrc.fit(X_train, Y_train)
        lrc_preds = lrc.predict(X_test)
        lrc_preds_proba=lrc.predict_proba(X_test)[:, 1]
        if CONF_MATR:
            #CONFUSION MATRIX
            create_confusion_matrix(Y_test, lrc_preds)
        compute_metrics(Y_test, lrc_preds, lrc_preds_proba)
        predictions.append(('LRC',lrc_preds))
    if SVMC:
        svmc = svm.SVC(probability=True, kernel='rbf', verbose=True)
        svmc.fit(X_train, Y_train)
        svmc_preds = svmc.predict(X_test)
        svmc_preds_proba=svmc.predict_proba(X_test)[:, 1]
        if CONF_MATR:
            #CONFUSION MATRIX
            create_confusion_matrix(Y_test, svmc_preds)
        compute_metrics(Y_test, svmc_preds, svmc_preds_proba)
        predictions.append(('SVMC',svmc_preds))
    if KNNC:
        knnc = KNeighborsClassifier()
        knnc.fit(X_train, Y_train)
        knnc_preds = knnc.predict(X_test)
        knnc_preds_proba=knnc.predict_proba(X_test)[:, 1]
        if CONF_MATR:
            #CONFUSION MATRIX
            create_confusion_matrix(Y_test, knnc_preds)
        compute_metrics(Y_test, knnc_preds, knnc_preds_proba)
        predictions.append(('KNNC',knnc_preds))
    if MLPC:
        mlpc = MLPClassifier(hidden_layer_sizes=(200,),random_state=RANDOM_STATE, max_iter=20000)
        mlpc.fit(X_train, Y_train)
        mlpc_preds = mlpc.predict(X_test)
        mlpc_preds_proba=mlpc.predict_proba(X_test)[:, 1]
        if CONF_MATR:
            #CONFUSION MATRIX
            create_confusion_matrix(Y_test, mlpc_preds)
        compute_metrics(Y_test, mlpc_preds, mlpc_preds_proba)
        predictions.append(('MLPC',mlpc_preds))

    if EEC:
        model=CatBoostClassifier(random_seed = RANDOM_STATE, metric_period = VERBOSE_EVAL)
        eec = EasyEnsembleClassifier(estimator=model,random_state=RANDOM_STATE, verbose=1, n_estimators=NUM_ESTIMATORS, sampling_strategy='majority') 
        eec.fit(X_train, Y_train)
        eec_preds = eec.predict(X_test)
        eec_preds_proba=eec.predict_proba(X_test)[:, 1]
        if CONF_MATR:
            #CONFUSION MATRIX
            create_confusion_matrix(Y_test, eec_preds)
        compute_metrics(Y_test, eec_preds, eec_preds_proba) 
    if RBC:
        model=CatBoostClassifier(random_seed = RANDOM_STATE, metric_period = VERBOSE_EVAL)
        rbc = RUSBoostClassifier(estimator=model, random_state=RANDOM_STATE,  sampling_strategy='majority') #n_estimators=NUM_ESTIMATORS,
        rbc.fit(X_train, Y_train)
        rbc_preds = rbc.predict(X_test)
        rbc_preds_proba=rbc.predict_proba(X_test)[:, 1]
        if CONF_MATR:
            #CONFUSION MATRIX
            create_confusion_matrix(Y_test, rbc_preds)
        compute_metrics(Y_test, rbc_preds, rbc_preds_proba) 
    if BRFC:
        brfc = BalancedRandomForestClassifier(random_state=RANDOM_STATE, n_estimators=(NUM_ESTIMATORS*4), replacement=True, sampling_strategy='all',verbose=1) #
        brfc.fit(X_train, Y_train)
        brfc_preds = brfc.predict(X_test)
        brfc_preds_proba=brfc.predict_proba(X_test)[:, 1]
        if CONF_MATR:
            #CONFUSION MATRIX
            create_confusion_matrix(Y_test, brfc_preds)
        compute_metrics(Y_test, brfc_preds, brfc_preds_proba) 
    if BBC:
        if UPSAMP:
            samp=ADASYN(sampling_strategy='minority', random_state=RANDOM_STATE)
        elif DOWNSAMP:
            samp=NearMiss(sampling_strategy='majority', version=3)
        cbc = CatBoostClassifier(random_seed = RANDOM_STATE)#, metric_period = VERBOSE_EVAL
        bbc = BalancedBaggingClassifier(sampler=samp, n_estimators=NUM_ESTIMATORS,estimator=cbc, random_state=RANDOM_STATE, verbose=1)
        bbc.fit(X_train, Y_train)
        bbc_preds = bbc.predict(X_test)
        bbc_preds_proba=bbc.predict_proba(X_test)[:, 1]
        if CONF_MATR:
            #CONFUSION MATRIX
            create_confusion_matrix(Y_test, bbc_preds)
        compute_metrics(Y_test, bbc_preds, bbc_preds_proba) 

    if INTER_WRONG_PRED:
        main_pred_index=None
        for i,pred in enumerate(predictions):
            if pred[0]=='CBC':
                main_pred_index=i
        if main_pred_index==None:
            print('ERROR: IMPOSSIBLE TO MEASURE INTERSECTION BECAUSE THERE ISN\'T THE MAIN CLASSIFIER CBC')
            sys.exit()
        main_pred=predictions.pop(main_pred_index)
        for i,pred in enumerate(predictions):
            print(pred[0])
            find_intersection_of_wrong_predictions(Y_test, main_pred[1], pred[1])
        

    if ENSEMBLE:
        cbc = CatBoostClassifier(iterations=1000,
                             learning_rate=0.02,
                             depth=12,
                             eval_metric='AUC',
                             random_seed = RANDOM_STATE,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             #metric_period = VERBOSE_EVAL,
                             od_wait=100)
        xgbc = XGBClassifier(random_state=RANDOM_STATE,verbosity=0)
        
        ensemble = VotingClassifier(estimators=[
    
                                     ('CBC', cbc), 
                                     ('XGBC', xgbc), 
    
                                    ], verbose=True, voting='soft')
        if K_FOLD:
            sss = StratifiedKFold(n_splits=NUMBER_KFOLDS, random_state=RANDOM_STATE, shuffle=True)
            for train_index, val_index in sss.split(X_train, Y_train):
                #print("Train:", train_index, "Val:", val_index)
                sss_X_train, sss_X_val = X_train.iloc[train_index], X_train.iloc[val_index]
                sss_Y_train, sss_Y_val = Y_train.iloc[train_index], Y_train.iloc[val_index]
                ensemble.fit(sss_X_train, sss_Y_train)
        else:
            ensemble.fit(X_train, Y_train)
        ensemble_preds = ensemble.predict(X_test)
        ensemble_preds_proba=ensemble.predict_proba(X_test)[:, 1]
        if CONF_MATR:
            #CONFUSION MATRIX
            create_confusion_matrix(Y_test, ensemble_preds)
        compute_metrics(Y_test, ensemble_preds, ensemble_preds_proba)
      
        
        
    
        
if __name__ == "__main__":
    main()

