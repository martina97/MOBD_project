import numpy as np
import sklearn.model_selection as model_selection
import sklearn.metrics as metrics
import sklearn.svm as svm
from imblearn.under_sampling import CondensedNearestNeighbour, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, \
    AllKNN, InstanceHardnessThreshold, NearMiss
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.metrics import f1_score, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

def cross4(train_x, test_x, train_y, test_y,method):
    classifier = QuadraticDiscriminantAnalysis()
    parameters= {
        'reg_param': (0.0000000001, 0.0001, 0.001, 0.01, 0.1),
        'store_covariance': (True, False),
        'tol': (0.0000000001, 0.001, 0.01, 0.1),
    }
    clf = model_selection.GridSearchCV(classifier, parameters,  scoring='f1_macro', cv=5, refit=True, n_jobs=-1)

    clf.fit(train_x.data, train_y.data.ravel())
    best_parameters = clf.best_params_
    print("\n\nbest_parameters MLP : ", best_parameters)
    best_result = clf.best_score_
    print("best_result MLP: ", best_result)
    evaluate_classifier(clf, test_x, test_y)

def cross3(train_x, test_x, train_y, test_y,method):
    #clf = QuadraticDiscriminantAnalysis(store_covariance=True)
    #clf = GaussianNB()
    classifier = KNeighborsClassifier(n_neighbors=3)
    #kernel = 1.0 * RBF(1.0)
    #clf = GaussianProcessClassifier(kernel=kernel,random_state=0)

    parameter_space = {
        'weights' : ['uniform', 'distance'],
        'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']

    }
    clf = model_selection.GridSearchCV(classifier, parameter_space, scoring='f1_macro', cv=5, refit=True, n_jobs=-1)

    clf.fit(train_x.data, train_y.data.ravel())
    best_parameters = clf.best_params_
    print("\n\nbest_parameters MLP : ", best_parameters)
    best_result = clf.best_score_
    print("best_result MLP: ", best_result)
    #best_result = clf.best_score_
    #print("best_result QuadraticDiscriminantAnalysis: ", best_result)
    evaluate_classifier(clf, test_x, test_y)

def cross2(train_x, test_x, train_y, test_y,method):


    clfs = {
        'gnb': GaussianNB(),
        'svm_linear': SVC(kernel='linear'),
        'svm_rbf': SVC(kernel='rbf'),
        'svm_sigmoid': SVC(kernel='sigmoid'),
        'svm_poly': SVC(kernel='poly'),
        'mlp1': MLPClassifier(),
        'mlp2': MLPClassifier(hidden_layer_sizes=[20,20,20]),
        'ada': AdaBoostClassifier(),
        'dtc': DecisionTreeClassifier(),
        'rfc': RandomForestClassifier(),
        'gbc': GradientBoostingClassifier(),
        'lr': LogisticRegression(),
        'qda': QuadraticDiscriminantAnalysis(reg_param= 0.001, store_covariance= True, tol= 1e-10)
    }

    f1_scores = dict()
    for clf_name in clfs:
        #print(clf_name)
        clf = clfs[clf_name]
        clf.fit(train_x.data, train_y.data.ravel())
        pred_y = clf.predict(test_x.data)
        f1_scores[clf_name] = f1_score(test_y.data, pred_y,  average='macro')

    print('Classifier\t\tF1')
    for name, score in f1_scores.items():
        print('{:<15} {:<15}'.format(name, score))


def cross_underSampl(train_x, test_x, train_y, test_y):


    classifier = QuadraticDiscriminantAnalysis()
    parameters = {
        'reg_param': (0.0000000001, 0.0001, 0.001, 0.01, 0.1),
        'store_covariance': (True, False),
        'tol': (0.0000000001, 0.001, 0.01, 0.1),
    }


    underSamplings = {
        #'CondensedNearestNeighbour' : CondensedNearestNeighbour(random_state=42),
        #'CondensedNearestNeighbour2': CondensedNearestNeighbour(),
        'EditedNearestNeighbours': EditedNearestNeighbours(n_neighbors=7, kind_sel = 'mode', n_jobs = -1),
        'EditedNearestNeighbours2': EditedNearestNeighbours(n_neighbors=7, kind_sel = 'mode', n_jobs = -1),
         'RepeatedEditedNearestNeighbours': RepeatedEditedNearestNeighbours(),
        #'AllKNN': AllKNN(allow_minority=True, n_neighbors=5, kind_sel='mode', n_jobs=-1),

        #'RepeatedEditedNearestNeighbours2': RepeatedEditedNearestNeighbours(n_neighbors=6, max_iter = 900000, kind_sel = 'mode', n_jobs = -1)

        #'AllKNN': AllKNN(allow_minority=True, n_neighbors=10, kind_sel='mode', n_jobs=-1),
        #'AllKNN2': AllKNN(allow_minority=True, n_neighbors=40, kind_sel='mode', n_jobs=-1),
        #'NearMiss': NearMiss(n_neighbors=10, version=1, sampling_strategy='majority'),
        #'NearMiss2': NearMiss(n_neighbors=10, version=1, sampling_strategy='not minority'),
        #'NearMiss3': NearMiss(n_neighbors=10, version=1, sampling_strategy='not majority'),
        #'NearMiss4': NearMiss(n_neighbors=10, version=1, sampling_strategy='all'),

    }

    f1_scores = dict()
    for underSampl_names in underSamplings:
        # print(clf_name)
        undersample = underSamplings[underSampl_names]
        #train_xPROVA, train_yPROVA = undersample.fit_resample(train_x.data, train_y.data)
        train_x.data, train_y.data = undersample.fit_resample(train_x.data, train_y.data)
        clf = model_selection.GridSearchCV(classifier, parameters, scoring='f1_macro', cv=5, refit=True, n_jobs=-1)
        #clf.fit(train_xPROVA, train_yPROVA.ravel())
        clf.fit(train_x.data, train_y.data.ravel())
        pred_y = clf.predict(test_x.data)
        f1_scores[underSampl_names] = f1_score(test_y.data, pred_y, average='macro')

    print('Classifier\t\tF1')
    for name, score in f1_scores.items():
        print('{:<30} {:<15}'.format(name, score))




def cross(train_x, test_x, train_y, test_y,method):
    if method == "IQR":
        np.savetxt("train_x.data_FINALE_iqr.csv", train_x.data, delimiter=",")
        np.savetxt("train_y.data_FINALE_iqr.csv", train_y.data, delimiter=",")
    else:

        np.savetxt("train_x.data_FINALE_z.csv", train_x.data, delimiter=",")
        np.savetxt("train_y.data_FINALE_z.csv", train_y.data, delimiter=",")

    #randomForest(train_x, train_y)
    #svm_param_selection(train_x, train_y, n_folds=5, metric='f1_macro')
    #decisionTree(train_x, train_y, n_folds=5, metric='f1_macro')

    clf = mlp(train_x, train_y, n_folds=5, metric='f1_macro')
    evaluate_classifier(clf, test_x, test_y)


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
    c_svc_log10 = 10. ** np.arange(-3, 3)
    gamma_svc_log10 = 10. ** np.arange(-5, 4)

    c_svc_log2 = 2. ** np.arange(-5, 5)
    gamma_svc_log2 = 2. ** np.arange(-3, 3)

    param_grid = [  # {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [0.1, 1, 10]},
        # {'kernel': ['linear'], 'C': [0.1, 1, 10]},
        # {'kernel': ['rbf'], 'gamma': 2. ** np.arange(-3,3), 'C': 2. ** np.arange(-5,5), 'class_weight': [None, 'balanced']},
        # {'kernel': ['rbf'], 'gamma': [0.01], 'C': [50], 'class_weight': [None]},
        {'kernel': ['rbf'], 'gamma': gamma_svc_log10, 'C': c_svc_log10, 'class_weight': [None, 'balanced']},
        {'kernel': ['rbf'], 'gamma': gamma_svc, 'C': c_svc, 'class_weight': [None, 'balanced']},
        # {'kernel': ['linear'], 'C': c_svc},
        # {'kernel': ['linear'], 'C': c_svc_log10},
        # {'kernel': ['linear'], 'C': c_svc_log2}

    ]

    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    gamma = [0.1, 1, 10, 100, 1e-2, 1e-3, 1e-4, 1e-5]
    c = [0.001, 0.10, 0.1, 1, 10, 25, 50, 100, 1000]
    poly_degree = [0, 1, 2, 3, 4, 5, 6]

    param_grid2 = [{'kernel': ['rbf'], 'gamma': gamma_range, 'C': C_range},
                   {'kernel': ['sigmoid'], 'gamma': gamma_range, 'C': C_range},
                   {'kernel': ['linear'], 'C': C_range},
                   {'kernel': ['poly'], 'C': C_range, 'gamma': gamma_range, 'degree': poly_degree},
                   ]

    param_grid3 = {'C': 100.0, 'class_weight': 'balanced', 'gamma': 0.1, 'kernel': 'rbf'}
    param_grid4 = {'kernel': ['rbf'], 'gamma': gamma_range, 'C': C_range}
    param_grid5 = {'kernel': ['sigmoid'], 'gamma': gamma_range, 'C': C_range}
    param_grid6 = {'kernel': ['linear'], 'C': C_range}
    param_grid7 = {'kernel': ['poly'], 'C': C_range, 'gamma': gamma_range, 'degree': poly_degree}

    clf = model_selection.GridSearchCV(svm.SVC(), param_grid, scoring=metric, cv=n_folds, refit=True)

    print("\n\nSVM")
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


def mlp(train_x, train_y, n_folds, metric):

    max_iter = 500
    classifier = MLPClassifier(max_iter = 500)
    #classifier = MLPClassifier()

    alpha1 = 1e-4
    alpha2 = [0.00005, 0.0005]
    alpha3 = 10.0 ** -np.arange(1, 10)
    alpha4 = [0.0001, 0.05]
    alpha5 = 10.0 ** -np.arange(1, 7)

    hidden_layer_sizes1 = (50, 50, 50)
    hidden_layer_sizes2 = [1, 50]
    hidden_layer_sizes3 = np.arange(10, 15)
    hidden_layer_sizes4 = [(10, 30, 10), (20,)]
    hidden_layer_sizes5 = [(7, 7), (128,), (128, 7)]
    hidden_layer_sizes6 = [(100, 1), (100, 2), (100, 3)]
    hidden_layer_sizes7 = [(50, 50, 50), (50, 100, 50), (100,)]

    '''
     'learning_rate': ["constant", "invscaling", "adaptive"],
        'activation': ["logistic", "relu", "tanh", "identity"],
                'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'max_iter':  [(200)],

        '''

    param_grid = {
        'random_state': [0, 1],
        'solver': ['adam'],
        'alpha': np.any(alpha3),
        'hidden_layer_sizes': hidden_layer_sizes3,
        'max_iter': np.array([200]),
        'tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
        'epsilon': [1e-3, 1e-7, 1e-8, 1e-9],

        'learning_rate': ['constant'],
        'activation': ['relu']
    }

    '''
    param_grid2={
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'hidden_layer_sizes': [(100,1), (100,2), (100,3)],
        'alpha': [10.0 ** -np.arange(1, 7)],
        'activation': ['logistic', 'relu', 'tanh', 'identity'],
        'solver' : ['adam']
    }
    '''

    parameter_space = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant'],
    }

    parameter_space2 =  {'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
                         'alpha': [0.3, 0.5, 0.7, 1, 1.2],
                         'early_stopping' : [True],
                         'activation': ['logistic', 'relu', 'tanh', 'identity'],
                         'learning_rate': ['constant', 'invscaling', 'adaptive'],
                         'tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                         'epsilon': [1e-3, 1e-7, 1e-8, 1e-9]

                         #'activation': ["logistic", "relu", "tanh", "identity"],
                         }

    parameter_space3 = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,150,100),(200,250,200)],
        #'hidden_layer_sizes': [ (50, 100, 50), (100, 150, 100), (200, 250, 200)],
        'activation': ['relu'],
        'alpha': [0.5, 1, 1.5],
        'learning_rate_init' : [0.02, 0.01, 0.001],
        'solver': ['sgd'],
        'learning_rate': ['invscaling']
        #'epsilon': [1e-08],
        #'tol': [1e-05],
        #'activation': ['logistic', 'relu', 'tanh', 'identity'],
        #'learning_rate': ['constant', 'invscaling', 'adaptive']
        
        #'tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
        #'epsilon': [1e-3, 1e-7, 1e-8, 1e-9]

    }
    parameter_space4 = {
        'hidden_layer_sizes': [(20,20,20), (50, 100, 50), (200, 150, 200)],
        'alpha': [0.5, 1, 1.2],
        # 'epsilon': [1e-08],
        # 'tol': [1e-05],
         'activation': ['relu'],
        'learning_rate': ['adaptive']

        # 'tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
        # 'epsilon': [1e-3, 1e-7, 1e-8, 1e-9]

    }

    print("parameter_space4 : ", parameter_space4)

    clf = model_selection.GridSearchCV(classifier, parameter_space4, scoring=metric, cv=n_folds, refit=True, n_jobs=-1)

    print("MLP")
    print("max_iter = ", max_iter)
    clf.fit(train_x.data, train_y.data.ravel())
    best_parameters = clf.best_params_
    print("\n\nbest_parameters MLP : ", best_parameters)
    best_result = clf.best_score_
    print("best_result MLP: ", best_result)

    return clf






# utilizziamo ora il miglior modello ottenuto al termine della cross-validation per fare previsioni sui dati di test\n",
def evaluate_classifier(classifier, test_x, test_y):
    pred_y = classifier.predict(test_x.data)
    confusion_matrix = metrics.confusion_matrix(test_y.data, pred_y)
    print(confusion_matrix)
    f1_score = metrics.f1_score(test_y.data, pred_y, average='macro')
    acc_score = metrics.accuracy_score(test_y.data, pred_y)
    print('F1: ', f1_score)
    print('Accuracy: ', acc_score)
    report=classification_report( test_y.data, pred_y)
    print(report)






def main():
    print("merda MERDA")

if __name__ == '__main__':
    main()