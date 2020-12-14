import pickle

import sklearn
import sklearn.preprocessing as prep
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
import dataPreparation
import crossValidation

class Dataset:
  def __init__(self, name, data):
    self.name = name
    self.data = data
    self.naCount = None
    self.outliers = None    #lista outliers di una colonna
    self.dataColumn = None  #elementi di una colonna
    self.result = None
    self.outliersDict = {}


def preProcessing_test(testSet_x,testSet_y, test_x, test_y, preProcDataset):

    testSet_x.data = test_x
    testSet_y.data = test_y
    print('SHAPE : test_x:', testSet_x.data.shape, "   test_y:", testSet_y.data.shape)
    print(' test_x:', testSet_x.data,"   test_y:", testSet_y.data)

    testSet_x.data = pd.DataFrame(testSet_x.data)
    testSet_y.data = pd.DataFrame(testSet_y.data, columns=['CLASS'])

    dataPreparation.changeColNames(testSet_x.data)

    #sostituisco i NaN con il valore della media della colonna calcolato precedentemente nel preProcessing
    replaceNaN(testSet_x, preProcDataset)

    print("\n\ntest_x dopo nan: ", testSet_x.data)
    #sostituisco gli outliers con il valore calcolato precedentemente nel preProcessing
    outliersDetection(testSet_x, preProcDataset)


def replaceNaN(testSet_x, preProcDataset):

    dataPreparation.getNaCount(testSet_x)
    print("train x na count : ", testSet_x.naCount)

    for colName in testSet_x.data.columns:
        #print("\n\ncolName = ", colName)
        valore = float(preProcDataset[colName][0])
        print("valore:", valore )
        testSet_x.data[colName] = testSet_x.data[colName].fillna(valore)

    dataPreparation.getNaCount(testSet_x)
    print("train x na count : ", testSet_x.naCount)


def outliersDetection(testSet_x, preProcDataset):

    for colName in testSet_x.data.columns:

        print("colName:", colName)
        #createDataColumn(testSet_x,colName)

        #poichè sto leggendo da csv devo convertire  tutti i valori in float
        #testSet_x.dataColumn = np.array(testSet_x.dataColumn).astype(np.float)

        dataPreparation.outZSCORE(testSet_x,colName)

        #sostituisco gli outliers con il valore trovato precedentemente nel preProcessing
        replaceOutliers(testSet_x, colName,preProcDataset)

        #testSet_x.dataColumn = np.array(testSet_x.dataColumn).astype(np.float)

        # controllo outliers dopo averli sostituiti
        checkOutliers(testSet_x, colName)


def createDataColumn(dataset,colName):

    dataset.dataColumn = np.array([])

    for colElement in dataset.data[colName]:
        valore = float(colElement)
        dataset.dataColumn = np.append(dataset.dataColumn, valore)


def replaceOutliers(testSet_x, colName,preProcDataset):

    for i in testSet_x.outliers:
        valore = float(preProcDataset[colName][1])
        testSet_x.data[colName][testSet_x.data[colName] == i] = (valore)


def checkOutliers(testSet_x, colName):
    #dataPreparation.createDataColumn(testSet_x, colName)

    # poichè sto leggendo da csv devo convertire  tutti i valori in float
    #testSet_x.dataColumn = np.array(testSet_x.dataColumn).astype(np.float)
    print("\n\n---")
    outliers = dataPreparation.outZSCORE(testSet_x, colName)
    if len(outliers) == 0:
        print(colName, ": KNN terminato, outliers sostituiti\n\n")
        return 0



def getClf():

    with open ('returned_clf.pkl','rb') as input:
        clf = pickle.load(input)

    best_parameters = clf.best_params_
    print("\n\nbest_parameters MLP : ", best_parameters)
    best_result = clf.best_score_
    print("best_result MLP: ", best_result)

    return clf

def main():

    #apro il file di test
    print("Inserisci path del test set:")
    path_test =input()
    testSet = pd.read_csv(path_test)
    print(testSet)

    #apro il file in cui ci sono i valori salvati del preProcessing
    preProcDataset = pd.read_csv('./preProcessingValues.csv')
    print("preProcDataset", preProcDataset)

    #print("ciaoooooooooooooo",preProcDataset['F1'][1][0])
    #print(testSet['CLASS'])

    testSet_x = Dataset("testSet_x", None)
    testSet_y = Dataset("testSet_y", None)

    # separiamo le features x dal target y
    test_x = testSet.iloc[:, 0:20].values
    test_y = testSet.iloc[:, 20].values

    preProcessing_test(testSet_x,testSet_y, test_x, test_y,preProcDataset)
    clf = getClf()
    crossValidation.evaluate_classifier(testSet_x, testSet_y)



if __name__ == '__main__':
    main()