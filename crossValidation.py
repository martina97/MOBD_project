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




def cross(train_x, test_x, train_y, test_y,method):
    if method == "IQR":
        np.savetxt("train_x.data_FINALE_iqr.csv", train_x.data, delimiter=",")
        np.savetxt("train_y.data_FINALE_iqr.csv", train_y.data, delimiter=",")
    else:

        np.savetxt("train_x.data_FINALE_z.csv", train_x.data, delimiter=",")
        np.savetxt("train_y.data_FINALE_z.csv", train_y.data, delimiter=",")

    #np.ravel().estimator.fit(X_train, y_train, )

    #randomForest(train_x, train_y)
    #svm_param_selection(train_x, train_y, n_folds=5, metric='f1_macro')
    decisionTree(train_x, train_y, n_folds=5, metric='f1_macro')


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
    # griglia degli iperparametri
    grid_param = {
        # 'n_estimators': [100, 300, 500, 800, 1000],
        'max_depth': [80, 90],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4],
        'n_estimators': [1000, 1500, 2000, 2500],
        'criterion': ['gini', 'entropy'],
        'bootstrap': [True, False]
    }



    gd_sr = model_selection.GridSearchCV(classifier,
                                         param_grid=grid_param,
                                         scoring='f1_macro',
                                         cv=5,
                                         refit=True,
                                         n_jobs=-1)


    gd_sr.fit(train_x.data, train_y.data.ravel())
    best_parameters = gd_sr.best_params_
    print("best_parameters RANDOM FOREST: ",best_parameters)
    best_result = gd_sr.best_score_
    print("best_result RANDOM FOREST: ", best_result)





def svm_param_selection(train_x, train_y, n_folds, metric):

    # griglia degli iperparametri\n",
    c_svc = [1, 1.5, 2, 2.5, 2.75, 3, 3.5, 5, 10]
    gamma_svc = [0.03, 0.05, 0.07, 0.1, 0.5]
    c_svc_log10 = 10. ** np.arange(-3,3)
    gamma_svc_log10 = 10. ** np.arange(-5,4)

    c_svc_log2 = 2. ** np.arange(-5, 5)
    gamma_svc_log2 = 2. ** np.arange(-3, 3)



    param_grid = [  #{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [0.1, 1, 10]},
                    #{'kernel': ['linear'], 'C': [0.1, 1, 10]},
                    #{'kernel': ['rbf'], 'gamma': 2. ** np.arange(-3,3), 'C': 2. ** np.arange(-5,5), 'class_weight': [None, 'balanced']},
                    #{'kernel': ['rbf'], 'gamma': [0.01], 'C': [50], 'class_weight': [None]},
                    {'kernel': ['rbf'], 'gamma': gamma_svc_log10, 'C':c_svc_log10, 'class_weight': [None, 'balanced']},
                    {'kernel': ['rbf'], 'gamma': gamma_svc, 'C': c_svc, 'class_weight': [None, 'balanced']},
                    {'kernel': ['linear'], 'C': c_svc},
                    {'kernel': ['linear'], 'C': c_svc_log10},
                    {'kernel': ['linear'], 'C': c_svc_log2}

    ]

    clf = model_selection.GridSearchCV(svm.SVC(), param_grid, scoring=metric, cv=n_folds, refit=True)


    clf.fit(train_x.data, train_y.data.ravel())
    best_parameters = clf.best_params_
    print("\n\nbest_parameters SVM : ", best_parameters)
    best_result = clf.best_score_
    print("best_result SVM: ", best_result)



def decisionTree(train_x, train_y, n_folds, metric):
    classifier = (DecisionTreeClassifier())

    param_grid = {
                    'criterion':['gini', 'entropy'],
                    'splitter':['best', 'random'],
                    #'max_depth':[np.arange(3, 200, 10), None],
                    'max_leaf_nodes': np.arange(2, 100),
                    'min_samples_split': [2, 3, 4],
                    'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]
                    # 'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)}}
    }
    clf = model_selection.GridSearchCV(classifier, param_grid, scoring=metric, cv=n_folds, refit=True)
    clf.fit(train_x.data, train_y.data.ravel())
    best_parameters = clf.best_params_
    print("\n\nbest_parameters DECISION TREE : ", best_parameters)
    best_result = clf.best_score_
    print("best_result DECISION TREE: ", best_result)


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

def main():
    print("merda MERDA")

if __name__ == '__main__':
    main()