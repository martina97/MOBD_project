import pickle

import sklearn
import sklearn.preprocessing as prep
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
import dataPreparation
import crossValidation


class Dataset:
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.naCount = None
        self.outliers = None  # lista outliers di una colonna
        self.dataColumn = None  # elementi di una colonna
        self.result = None
        self.outliersDict = {}


def preProcessing_test(testSet_x, testSet_y, test_x, test_y):
    testSet_x.data = test_x
    testSet_y.data = test_y

    print('SHAPE : test_x:', testSet_x.data.shape, "   test_y:", testSet_y.data.shape)

    testSet_x.data = pd.DataFrame(testSet_x.data)
    testSet_y.data = pd.DataFrame(testSet_y.data, columns=['CLASS'])

    dataPreparation.changeColNames(testSet_x.data)

    # sostituisco i NaN con il valore della media della colonna calcolato precedentemente nel preProcessing
    replaceNaN(testSet_x)

    # sostituisco gli outliers con il valore calcolato precedentemente nel preProcessing
    outliersDetection(testSet_x)

    matrixTest(testSet_x, testSet_y)
    # scaler
    standardScalerTest(testSet_x)

    # pca
    pcaTest(testSet_x)


def matrixTest(testSet_x, testSet_y):
    testSet_x.data = np.float64(testSet_x.data)
    testSet_y.data = np.float64(testSet_y.data)
    testSet_y.data = testSet_y.data.reshape((len(testSet_y.data), 1))


def standardScalerTest(testSet_x):
    scaler = getObject('scaler.pkl')
    testSet_x.data = scaler.transform(testSet_x.data)


def pcaTest(testSet_x):
    pca = getObject('pca1.pkl')
    testSet_x.data = pca.transform(testSet_x.data)
    pca = getObject('pca2.pkl')
    testSet_x.data = pca.transform(testSet_x.data)


def replaceNaN(testSet_x):
    dataPreparation.getNaCount(testSet_x)
    imputer = getObject('imputer.pkl')
    imputed_test = imputer.transform(testSet_x.data)
    testSet_x.data = pd.DataFrame(imputed_test, columns=testSet_x.data.columns)
    dataPreparation.getNaCount(testSet_x)


def outliersDetection(testSet_x):
    for colName in testSet_x.data.columns:
        outZScoreTest(testSet_x, colName)

        # sostituisco gli outliers con il valore trovato precedentemente nel preProcessing
        replaceOutliers(testSet_x, colName)

        # controllo outliers dopo averli sostituiti
        checkOutliers(testSet_x, colName)


def outZScoreTest(testSet_x, colName):
    testSet_x.dataColumn = np.array([])

    for colElement in testSet_x.data[colName]:
        testSet_x.dataColumn = np.append(testSet_x.dataColumn, colElement)

    mean = testSet_x.outliersDict[colName][0]
    std = testSet_x.outliersDict[colName][1]

    count = 0
    threshold = 3
    testSet_x.outliers = []
    for i in testSet_x.dataColumn:
        z = (i - mean) / std

        if z > threshold:
            count = count + 1
            testSet_x.outliers.append(i)
            # print("-- outlier n ", count, ":  ", testSet_x.outliers[count - 1])

    return testSet_x.outliers


def createDataColumn(dataset, colName):
    dataset.dataColumn = np.array([])

    for colElement in dataset.data[colName]:
        value = float(colElement)
        dataset.dataColumn = np.append(dataset.dataColumn, value)


def replaceOutliers(testSet_x, colName):
    substitution = testSet_x.outliersDict[colName][2]
    for i in testSet_x.outliers:
        testSet_x.data[colName][testSet_x.data[colName] == i] = (substitution)


def checkOutliers(testSet_x, colName):
    outliers = outZScoreTest(testSet_x, colName)
    if len(outliers) == 0:
        print(colName, ": KNN terminato, outliers sostituiti\n\n")


def getClf():
    with open('returned_clf.pkl', 'rb') as input:
        clf = pickle.load(input)

    return clf


def getObject(path):
    with open(path, 'rb') as input:
        object = pickle.load(input)

    return object


def main():
    # apro il file di test
    print("Inserire path del test set:")
    path_test = input()
    testSet = pd.read_csv(path_test)

    testSet_x = Dataset("testSet_x", None)
    testSet_y = Dataset("testSet_y", None)

    # separiamo le features x dal target y
    test_x = testSet.iloc[:, 0:20].values
    test_y = testSet.iloc[:, 20].values

    testSet_x.outliersDict = getObject('./dict.pkl')

    preProcessing_test(testSet_x, testSet_y, test_x, test_y)
    clf = getClf()
    crossValidation.evaluate_classifier(clf, testSet_x, testSet_y)


if __name__ == '__main__':
    main()
