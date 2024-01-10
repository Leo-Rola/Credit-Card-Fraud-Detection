from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score


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