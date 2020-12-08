import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import sklearn
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection
import sklearn.metrics as metrics
import sklearn.svm as svm
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from main import openFiles

def cross(train_dataset):
    x = train_dataset.iloc[:, 0:20].values
    y = train_dataset.iloc[:, 20].values
    print(" ----> x= ", len(x))
    print(" ----> y= ", len(y))

    cv = model_selection.KFold(n_splits=5, random_state=0, shuffle=True)
    for train_index, test_index in cv.split(train_dataset):
       print("TRAIN:", train_index, "TEST:", test_index)


# k: numero di fold per k-fold cross validation\n",
# C: iperparametro per SVM\n",
# kernel: tipologia di kernel per SVM\
# train_x = train_dataset
def k_fold_cross_validation_svm(train_x, k=5, C=1, kernel='linear', degree=3, gamma='auto'):
    avg_score = 0
    cv = model_selection.KFold(n_splits=k, random_state=0)
    classifier = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
    for train_index, test_index in cv.split(train_x):
        fold_train_x, fold_test_x = x[train_index], x[test_index]
        fold_train_y, fold_test_y = y[train_index], y[test_index]
        classifier.fit(fold_train_x, fold_train_y)
        fold_pred_y = classifier.predict(fold_test_x)
        score = metrics.accuracy_score(fold_test_y, fold_pred_y)
        print(score)
        avg_score += score
    avg_score = avg_score / k
    return avg_score

    score = k_fold_cross_validation_svm(train_x, k=5, C=1, kernel='linear')
    print('k-fold score:', score)



def main():
    print("merda MERDA")
    datasetPath = './training_set.csv'
    train_dataset = openFiles(datasetPath)
    cross(train_dataset)

if __name__ == '__main__':
    main()