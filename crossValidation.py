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



def cross(train_x, test_x, train_y, test_y, x, y,method):
    if method == "IQR":
        np.savetxt("train_x.data_FINALE_iqr.csv", train_x.data, delimiter=",")
        np.savetxt("train_y.data_FINALE_iqr.csv", train_y.data, delimiter=",")
    else:

        np.savetxt("train_x.data_FINALE_z.csv", train_x.data, delimiter=",")
        np.savetxt("train_y.data_FINALE_z.csv", train_y.data, delimiter=",")

    x = train_x.data.iloc[:,0:8].values
    np.append(x, test_x.data.iloc[:,0:8].values)

    print(" ---------> x: ", x.shape)



    print(" ----> train_x name= ", train_x.name)

    cv = model_selection.KFold(n_splits=5, random_state=0, shuffle=True)
    for train_index, test_index in cv.split(train_x.data):
       print("TRAIN:", train_index, "TEST:", test_index)

    degree = 3
    gamma = "auto"
    kernel = "linear"

    #score = k_fold_cross_validation_svm(train_x.data, 5, 1, kernel, degree, gamma, x, y)
    score = k_fold_cross_validation_svm(train_x, 5, 1, kernel, degree, gamma, x, y,test_x, test_y)

    #TODO: LEGGERE!!!!!
    '''
    funziona con train_x e y solo perché FORTUNATAMENTE y non ha NaN
    con x e y non funziona perché x così ha dei NaN 
    con train_x e train_y non funziona a causa di bug in pandas (parquet index:
    https://github.com/modin-project/modin/pull/1397 )
    '''

    print('k-fold score:', score)




def k_fold_cross_validation_svm(train_x, k, C, kernel, degree, gamma, x, train_y,test_x, test_y):

    '''
    :param train_x: train dataset senza colonna target y
    :param k: numero di fold per k-fold cross validation
    :param C: iperparametro per SVM
    :param kernel: tipologia di kernel per SVM
    :param degree: grado kernel polinomiale
    :param gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
    :param x: dataset features
    :param y: dataset target y
    :return:
    '''

    avg_score = 0
    cv = model_selection.KFold(n_splits=k, random_state=0, shuffle=True)
    classifier = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
    for train_index, test_index in cv.split(train_x.data):
        fold_train_x, fold_test_x = train_x.data[train_index], test_x.data[test_index]
        fold_train_y, fold_test_y = train_y.data[train_index], test_y.data[test_index]
        classifier.fit(fold_train_x, fold_train_y)
        fold_pred_y = classifier.predict(fold_test_x)
        # ora calcola accuracy (%di esempi classificati correttamente):
        score = metrics.accuracy_score(fold_test_y, fold_pred_y)
        print(score)
        avg_score += score
    avg_score = avg_score / k   #risultato finale
    return avg_score





def main():
    print("merda MERDA")

if __name__ == '__main__':
    main()