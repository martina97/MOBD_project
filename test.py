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
    self.outliers = None    #lista outliers di una colonna
    self.dataColumn = None  #elementi di una colonna
    self.result = None
    self.outliersDict = {}


def preProcessing_test(testSet_x,testSet_y, test_x, test_y, preProcDataset):

    testSet_x.data = test_x
    testSet_y.data = test_y

    np.savetxt("PROVAA TEST_X 1.csv", testSet_x.data, delimiter=",")
    np.savetxt("PROVAA TEST_Y 1.csv", testSet_y.data, delimiter=",")

    print('SHAPE : test_x:', testSet_x.data.shape, "   test_y:", testSet_y.data.shape)
    print(' test_x:', testSet_x.data,"   test_y:", testSet_y.data)

    testSet_x.data = pd.DataFrame(testSet_x.data)
    testSet_y.data = pd.DataFrame(testSet_y.data, columns=['CLASS'])

    print("testSet_x = ", testSet_x.data)
    print("testSet_y = ", testSet_y.data)

    dataPreparation.changeColNames(testSet_x.data)

    #np.savetxt("PROVAA TEST_X 2.csv", testSet_x.data, delimiter=",")
    #np.savetxt("PROVAA TEST_Y 2.csv", testSet_y.data, delimiter=",")

    #sostituisco i NaN con il valore della media della colonna calcolato precedentemente nel preProcessing
    replaceNaN(testSet_x)

    #np.savetxt("PROVAA TEST_X 3.csv", testSet_x.data, delimiter=",")
    #np.savetxt("PROVAA TEST_Y 3.csv", testSet_y.data, delimiter=",")

    print("\n\ntest_x dopo nan: ", testSet_x.data)
    #sostituisco gli outliers con il valore calcolato precedentemente nel preProcessing
    outliersDetection(testSet_x, preProcDataset)
    #np.savetxt("PROVAA TEST_X 4.csv", testSet_x.data, delimiter=",")
    #np.savetxt("PROVAA TEST_Y 4.csv", testSet_y.data, delimiter=",")

    matrixTest(testSet_x, testSet_y)
    #scaler
    standardScalerTest(testSet_x)
    #np.savetxt("PROVAA TEST_X 5.csv", testSet_x.data, delimiter=",")
    #np.savetxt("PROVAA TEST_Y 5.csv", testSet_y.data, delimiter=",")
    #pca
    pcaTest(testSet_x)
    #np.savetxt("PROVAA TEST_X 6.csv", testSet_x.data, delimiter=",")
    #np.savetxt("PROVAA TEST_Y 6.csv", testSet_y.data, delimiter=",")

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

    '''
    for colName in testSet_x.data.columns:
        #print("\n\ncolName = ", colName)
        valore = float(preProcDataset[colName][0])
        print("valore:", valore )
        testSet_x.data[colName] = testSet_x.data[colName].fillna(valore)

    dataPreparation.getNaCount(testSet_x)
    print("train x na count : ", testSet_x.naCount)
    '''

def outliersDetection(testSet_x, preProcDataset):

    for colName in testSet_x.data.columns:

        print("colName:", colName)
        #createDataColumn(testSet_x,colName)

        #poichè sto leggendo da csv devo convertire  tutti i valori in float
        #testSet_x.dataColumn = np.array(testSet_x.dataColumn).astype(np.float)

        outZScoreTest(testSet_x, colName, preProcDataset)
        #sostituisco gli outliers con il valore trovato precedentemente nel preProcessing
        replaceOutliers(testSet_x, colName,preProcDataset)

        #testSet_x.dataColumn = np.array(testSet_x.dataColumn).astype(np.float)

        # controllo outliers dopo averli sostituiti
        checkOutliers(testSet_x, colName, preProcDataset)

def outZScoreTest(testSet_x, colName, preProcDataset):
    testSet_x.dataColumn = np.array([])

    for colElement in testSet_x.data[colName]:
        testSet_x.dataColumn = np.append(testSet_x.dataColumn, colElement)

    #mean = float(preProcDataset[colName][0])
    #std = float(preProcDataset[colName][1])

    mean = testSet_x.outliersDict[colName][0]
    std = testSet_x.outliersDict[colName][1]


    print("mean = ", mean, "std = ", std)

    count = 0
    threshold = 3
    testSet_x.outliers = []
    for i in testSet_x.dataColumn:
        z = (i - mean) / std

        if z > threshold:
            count = count + 1
            testSet_x.outliers.append(i)
            print("-- outlier n ", count, ":  ", testSet_x.outliers[count - 1])

    return testSet_x.outliers





def createDataColumn(dataset,colName):

    dataset.dataColumn = np.array([])

    for colElement in dataset.data[colName]:
        valore = float(colElement)
        dataset.dataColumn = np.append(dataset.dataColumn, valore)


def replaceOutliers(testSet_x, colName,preProcDataset):

    substitution = testSet_x.outliersDict[colName][2]
    print("substitution = ", substitution)
    for i in testSet_x.outliers:
        testSet_x.data[colName][testSet_x.data[colName] == i] = (substitution)


def checkOutliers(testSet_x, colName, preProcDataset):
    #dataPreparation.createDataColumn(testSet_x, colName)

    # poichè sto leggendo da csv devo convertire  tutti i valori in float
    #testSet_x.dataColumn = np.array(testSet_x.dataColumn).astype(np.float)
    print("\n\n---")
    outliers = outZScoreTest(testSet_x, colName, preProcDataset)
    if len(outliers) == 0:
        print(colName, ": KNN terminato, outliers sostituiti\n\n")
        return 0



def getClf():

    with open ('returned_clf.pkl','rb') as input:
        clf = pickle.load(input)

    #best_parameters = clf.best_params_
    #print("\n\nbest_parameters  : ", best_parameters)
    #best_result = clf.best_score_
    #print("best_result : ", best_result)

    return clf

def getObject(path):
    with open (path,'rb') as input:
        object = pickle.load(input)

    return object


def main():

    #apro il file di test
    print("Inserisci path del test set:")
    path_test =input()
    testSet = pd.read_csv(path_test)
    print(testSet)

    #apro il file in cui ci sono i valori salvati del preProcessing
    preProcDataset = pd.read_csv('./preProcessingValues.csv')
    print("preProcDataset", preProcDataset)

    print("ciaoooooooooooooo",preProcDataset['F1'][1], "\n\n")
    #print(testSet['CLASS'])

    testSet_x = Dataset("testSet_x", None)
    testSet_y = Dataset("testSet_y", None)

    # separiamo le features x dal target y
    test_x = testSet.iloc[:, 0:20].values
    test_y = testSet.iloc[:, 20].values

    testSet_x.outliersDict = getObject('./dict.pkl')
    print("prima colonna", testSet_x.outliersDict['F1'])
    print("prima colonna", testSet_x.outliersDict['F1'][0])
    print("prima colonna", testSet_x.outliersDict['F1'][1])
    print("prima colonna", testSet_x.outliersDict['F1'][2])
    #print("dict = " , dict)
    preProcessing_test(testSet_x,testSet_y, test_x, test_y,preProcDataset)
    clf = getClf()
    print("clf = ", clf)
    crossValidation.evaluate_classifier(clf, testSet_x, testSet_y)



if __name__ == '__main__':
    main()