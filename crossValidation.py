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
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

from main import openFiles



def cross(train_x, test_x, train_y, test_y,method):
    if method == "IQR":
        np.savetxt("train_x.data_FINALE_iqr.csv", train_x.data, delimiter=",")
        np.savetxt("train_y.data_FINALE_iqr.csv", train_y.data, delimiter=",")
    else:

        np.savetxt("train_x.data_FINALE_z.csv", train_x.data, delimiter=",")
        np.savetxt("train_y.data_FINALE_z.csv", train_y.data, delimiter=",")

    #np.ravel().estimator.fit(X_train, y_train, )

    randomForest(train_x, train_y)
    svm_param_selection(train_x, train_y, n_folds=5, metric='f1_macro')


def randomForest(train_x, train_y):
    # scelgo algoritmo/classificatore
    #classifier = RandomForestClassifier(n_estimators=600, random_state=0)
    classifier = RandomForestClassifier()

    #pipe = Pipeline(['classifier', classifier])

    # calcolo accuracy di tutti i folds
    all_accuracies = cross_val_score(estimator=classifier, X=train_x.data, y=train_y.data.ravel(), cv=5)

    print("all_accuracies: ", all_accuracies)
    print("all_accuracies.mean: ", all_accuracies.mean())
    print("all_accuracies.std: ", all_accuracies.std())

    # Adesso facciamo Grid Search
    grid_param = {
        # 'n_estimators': [100, 300, 500, 800, 1000],
        'n_estimators': [1000, 1500, 2000, 2500],
        'criterion': ['gini', 'entropy'],
        'bootstrap': [True, False]
    }

    grid_param2 = {
        # 'n_estimators': [100, 300, 500, 800, 1000],
        'n_estimators': [2500, 3000, 3500, 4000],
        'criterion': ['gini', 'entropy'],
        'bootstrap': [True, False]
    }

    gd_sr = model_selection.GridSearchCV(classifier,
                                         param_grid=grid_param2,
                                         scoring='f1_macro',
                                         cv=5,
                                         refit=True,
                                         n_jobs=-1)


    gd_sr.fit(train_x.data, train_y.data.ravel())
    best_parameters = gd_sr.best_params_
    print("best_parameters RANDOM FOREST: ",best_parameters)
    best_result = gd_sr.best_score_
    print("best_result RANDOM FOREST: ", best_result)



    '''
    gd_sr.fit(train_x.data, train_y.data.ravel())
    print("Best parameters:")
    print()
    print(gd_sr.best_params_)
    print()
    print("Grid scores:")
    print()
    means = gd_sr.cv_results_['mean_test_score']
    stds = gd_sr.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, gd_sr.cv_results_['params']):
        print("%0.4f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    '''

def svm_param_selection(train_x, train_y, n_folds, metric):

    # griglia degli iperparametri\n",
    param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [0.1, 1, 10]},
                    {'kernel': ['linear'], 'C': [0.1, 1, 10]}]

    clf = model_selection.GridSearchCV(svm.SVC(), param_grid, scoring=metric, cv=n_folds, refit=True)


    clf.fit(train_x.data, train_y.data.ravel())
    best_parameters = clf.best_params_
    print("\n\nbest_parameters SVM : ", best_parameters)
    best_result = clf.best_score_
    print("best_result SVM: ", best_result)

    '''
    clf.fit(train_x.data, train_y.data.ravel())
    
    print("Best parameters:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    '''

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