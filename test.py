import pickle

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.metrics import f1_score, classification_report




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

    changeColNames(testSet_x.data)

    # sostituisco i NaN con il valore della media della colonna calcolato precedentemente nel preProcessing
    replaceNaN(testSet_x)

    # sostituisco gli outliers con il valore calcolato precedentemente nel preProcessing
    outliersDetection(testSet_x)

    # conversione dei DataFrame in vettori 2D
    matrixTest(testSet_x, testSet_y)

    # scaling del test set
    standardScalerTest(testSet_x)

    # Principal Component Analysis
    pcaTest(testSet_x)


def matrixTest(testSet_x, testSet_y):
    """
    Converte i DataFrame per l'input e per l'output in vettori 2D (matrici)
    :param testSet_x: test set x
    :param testSet_y: test set y
    :return: None
    """
    testSet_x.data = np.float64(testSet_x.data)
    testSet_y.data = np.float64(testSet_y.data)
    testSet_y.data = testSet_y.data.reshape((len(testSet_y.data), 1))


def standardScalerTest(testSet_x):
    """
   Effettua lo scaling del test set tramite lo scaler salvato precedentemente
   nella fase di addestramento del training set.
   :param testSet_x: test set
   :return: None
   """
    scaler = getObject('scaler.pkl')
    testSet_x.data = scaler.transform(testSet_x.data)


def pcaTest(testSet_x):
    """
    Effettua la Principal Component Analysis sul test set tramite la pca salvata precedentemente
    nella fase di addestramento del training set.
    :param testSet_x: test set
    :return: None
    """
    pca = getObject('pca1.pkl')
    testSet_x.data = pca.transform(testSet_x.data)
    pca = getObject('pca2.pkl')
    testSet_x.data = pca.transform(testSet_x.data)


def replaceNaN(testSet_x):
    """
    Sostituisce i valori mancanti nel test set tramite l'imputer KNNImputer salvato precedentemente
    nella fase di addestramento del training set.
    :param testSet_x:
    :return:
    """
    getNaCount(testSet_x)
    imputer = getObject('imputer.pkl')
    imputed_test = imputer.transform(testSet_x.data)
    testSet_x.data = pd.DataFrame(imputed_test, columns=testSet_x.data.columns)
    getNaCount(testSet_x)


def outliersDetection(testSet_x):
    """
    Individua e sostituisce gli outliers presenti test set
    :param testSet_x: test set
    :return: None
    """
    for colName in testSet_x.data.columns:
        outZScoreTest(testSet_x, colName)

        # sostituisco gli outliers con il valore trovato precedentemente nel preProcessing
        replaceOutliers(testSet_x, colName)

        # controllo outliers dopo averli sostituiti
        checkOutliers(testSet_x, colName)


def outZScoreTest(testSet_x, colName):
    """
    Inserisce in testSet_x.outliers gli outliers contenuti nella colonna colName, individuati tramite
    il metodo ZScore.
    In particolare, la media e la deviazione standard della colonna sono, rispettivamente,
    il primo ed il secondo valore associati alla chiave 'colName' del dizionario testSet_x.outliersDict.
    :param testSet_x: test set
    :param colName: colonna interessata
    :return: lista degli outliers presenti nella colonna colName del test set
    """
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

def changeColNames(dataset):
    """
    Funzione che rinomina le colonne del dataset
    :param dataset: dataset
    :return: None
    """
    string = "F"
    for i in range(1, 21):
        currColumn = string + str(i)
        index = i - 1
        dataset.rename(columns={index: currColumn}, inplace=True)


def getNaCount(dataset):
    """
    Funzione che conta il numero di valori mancanti all'interno del dataset.
    :param dataset: dataset
    :return: numero di valori mancanti
    """
    # per ogni elemento (i,j) del dataset, isna() restituisce
    # TRUE/FALSE se il valore corrispondente è mancante/presente
    boolean_mask = dataset.data.isna()
    # contiamo il numero di TRUE per ogni attributo sul dataset
    count = boolean_mask.sum(axis=0)
    # print("count NaN: ", count)
    dataset.naCount = count
    return count

def replaceOutliers(testSet_x, colName):
    """
    Sostituisce, colonna x colonna, gli outliers presenti in testSet_x.outliers con il terzo valore
    contenuto nel dizionario testSet_x.outliersDict avente come chiave la colonna colName
    :param testSet_x: test set
    :param colName: colonna desiderata
    :return: None
    """
    substitution = testSet_x.outliersDict[colName][2]
    for i in testSet_x.outliers:
        testSet_x.data[colName][testSet_x.data[colName] == i] = (substitution)


def checkOutliers(testSet_x, colName):
    """
    Controlla, colonna x colonna, se sono ancora presenti outliers nel test set
    :param testSet_x: test set
    :param colName: colonna interessata
    :return: None
    """
    outliers = outZScoreTest(testSet_x, colName)
    if len(outliers) == 0:
        print(colName, ": KNN terminato, outliers sostituiti\n\n")


def getClf():
    """
    Salva in clf il classificatore contenuto nel file 'returned_clf.pkl'
    :return: l'oggetto clf
    """
    with open('returned_clf.pkl', 'rb') as input:
        clf = pickle.load(input)

    return clf


def getObject(path):
    """
    Prende i file.pkl salvati durante l'addestramento del training set
    :param path: path del file contenente l'oggetto che si vuole caricare
    :return: oggetto caricato
    """
    with open(path, 'rb') as input:
        object = pickle.load(input)

    return object

def evaluate_classifier(classifier, test_x, test_y):
    """
    Permette di calcolare la metrica F1 score.
    :param classifier: classificatore
    :param test_x: test set x
    :param test_y: test set y
    :return: None
    """

    pred_y = classifier.predict(test_x.data)
    confusion_matrix = metrics.confusion_matrix(test_y.data, pred_y)
    print("confusion_matrix:\n", confusion_matrix)
    f1_score = metrics.f1_score(test_y.data, pred_y, average='macro')
    acc_score = metrics.accuracy_score(test_y.data, pred_y)
    print('\nF1: ', f1_score)
    print('Accuracy: ', acc_score)
    report = classification_report(test_y.data, pred_y)
    print(report)


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

    # prendo il dizionario salvato durante l'addestramento del training set
    # Tale dizionario avrà come chiave il nome della feature, e come valori, in ordine,
    # la media e la deviazione standard della colonna (utili per individuare gli
    # outliers), e il valore con cui gli outliers appartenenti alla colonna devono essere sostituiti
    testSet_x.outliersDict = getObject('./dict.pkl')

    # preprocessing sul test set
    preProcessing_test(testSet_x, testSet_y, test_x, test_y)
    clf = getClf()

    # Calcolo della metrica F1 score
    evaluate_classifier(clf, testSet_x, testSet_y)


if __name__ == '__main__':
    main()
