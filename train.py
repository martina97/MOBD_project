import csv

import sklearn
import sklearn.preprocessing as prep
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
import csv

import pickle
import dataPreparation


class Dataset:
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.naCount = None
        self.outliers = None  # lista outliers di una colonna
        self.dataColumn = None  # elementi di una colonna
        self.result = None
        self.outliersDict = {}


def preProcessing_train(trainingSet_x, trainingSet_y, train_x, train_y):
    trainingSet_x.data = train_x
    trainingSet_y.data = train_y

    print('SHAPE : Train_x:', train_x.data.shape, "   train_y:", train_y.data.shape)
    print('Train_x:', trainingSet_x.data, "   train_y:", trainingSet_y.data)

    trainingSet_x.data = pd.DataFrame(trainingSet_x.data)
    trainingSet_y.data = pd.DataFrame(trainingSet_y.data, columns=['CLASS'])

    dataPreparation.changeColNames(trainingSet_x.data)

    dataPreparation.naKNN(trainingSet_x, None)

    outliers_train(trainingSet_x)

    dataPreparation.matrix(trainingSet_x, None, trainingSet_y, None)
    dataPreparation.standardScaler(trainingSet_x, None)

    dataPreparation.pca(trainingSet_x, None)

    dataPreparation.Resampling(trainingSet_x, trainingSet_y)

    dataPreparation.save_object(trainingSet_x.outliersDict, 'dict.pkl')

    saveDataInCSV(trainingSet_x)


def evaluation_train(trainingSet_x, trainingSet_y):
    n_folds = 5
    metric = 'f1_macro'

    clf = QuadraticDiscriminantAnalysis(reg_param= 0.0001,store_covariance= True, tol=0.1)

    ''' 
    parameters = {
        'reg_param': (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10),
        'store_covariance': [True, False],
        'tol': (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10),
    }
    '''

    #clf = model_selection.GridSearchCV(classifier, parameters, scoring=metric, cv=n_folds, refit=True, n_jobs=-1)
    print("QuadraticDiscriminantAnalysis")
    clf.fit(trainingSet_x.data, trainingSet_y.data.ravel())
    #best_parameters = clf.best_params_
    #print("\n\nbest_parameters QuadraticDiscriminantAnalysis : ", best_parameters)
    #best_result = clf.best_score_
    #print("best_result QuadraticDiscriminantAnalysis: ", best_result)

    return clf





def outliers_train(trainingSet_x):
    print("dictionary outliers: ", trainingSet_x.outliersDict)
    for colName in trainingSet_x.data.columns:
        print("\n\ncolName = ", colName)


        print("\n\nOUTLIERS WITH ZSCORE\n")
        res = dataPreparation.outZSCORE(trainingSet_x, None, colName)
        print("dictionary outliers: ", trainingSet_x.outliersDict)

        #mean = res[2]
        #std = res[3]
        #print("mean = ", mean, "\nstd = ", std)

        # aggiungo mean e std al dizionario, che servirà nella parte successiva in test.py
        #dataPreparation.appendDict(colName, mean, trainingSet_x)
        #dataPreparation.appendDict(colName, std, trainingSet_x)

        # una volta che ho la lista di outliers, li sostituisco con il metodo KNN, che avrà come input sia
        # il training che il test, poichè devo modificarli entrambi colonna x colonna
        dataPreparation.knnDetectionTRAIN(trainingSet_x, None, colName)
        print("dictionary outliers: ", trainingSet_x.outliersDict)

        # sostuituiamo i risultati con gli outliers nel dataset originario
        substituteOutliersTrain(trainingSet_x, colName)
        print("dictionary outliers: ", trainingSet_x.outliersDict)

        # controllo outliers dopo aver applicato KNN
        checkOutliersAfterReplacementTrain(trainingSet_x, colName)
        print("dictionary outliers: ", trainingSet_x.outliersDict)

    print("dictionary outliers: ", trainingSet_x.outliersDict)


def substituteOutliersTrain(trainingSet_x, colName):
    if len(trainingSet_x.result) == 1:
        for i in trainingSet_x.outliers:
            trainingSet_x.data[colName][trainingSet_x.data[colName] == i] = (trainingSet_x.result[0][0])
    if len(trainingSet_x.result) > 1:
        for i in trainingSet_x.outliers:
            res = dataPreparation.checkClosestOutlier(i, trainingSet_x.result)
            trainingSet_x.data[colName][trainingSet_x.data[colName] == i] = (res)


def checkOutliersAfterReplacementTrain(trainingSet_x, colName):
    outliers = dataPreparation.outZSCORE(trainingSet_x, None, colName)[0]
    if len(outliers) == 0:
        print(colName, ": Tutti gli outliers nel training set sono stati sostituiti\n\n")
        return 0


def saveDataInCSV(trainingSet_x):
    file_path = './preProcessingValues.csv'

    df = pd.DataFrame(trainingSet_x.outliersDict)
    df.to_csv(file_path)

    f = open(file_path, 'r')
    reader = csv.reader(f)
    mylist = list(reader)
    f.close()
    mylist[1][0] = 'ZSCOREMean'
    mylist[2][0] = 'ZSCOREStd'
    mylist[3][0] = 'ReplacementOutliers'
    my_new_list = open(file_path, 'w', newline='')
    csv_writer = csv.writer(my_new_list)
    csv_writer.writerows(mylist)
    my_new_list.close()

    preProcDataset = pd.read_csv('./preProcessingValues.csv')


def main():
    datasetPath = './training_set.csv'
    dataset = pd.read_csv(datasetPath)

    trainingSet_x = Dataset("trainingSet_x", None)  # feature x
    trainingSet_y = Dataset("trainingSet_y", None)  # target y

    # separiamo le features x dal target y
    train_x = dataset.iloc[:, 0:20].values
    train_y = dataset.iloc[:, 20].values

    preProcessing_train(trainingSet_x, trainingSet_y, train_x, train_y)

    clf = evaluation_train(trainingSet_x, trainingSet_y)
    dataPreparation.save_object(clf, 'returned_clf.pkl')


if __name__ == '__main__':
    main()
