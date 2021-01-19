import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

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

    trainingSet_x.data = pd.DataFrame(trainingSet_x.data)
    trainingSet_y.data = pd.DataFrame(trainingSet_y.data, columns=['CLASS'])

    dataPreparation.changeColNames(trainingSet_x.data)

    # sostituisco i valori mancanti nel training set
    dataPreparation.naKNN(trainingSet_x, None)

    # sostituisco gli outliers nel training set
    outliers_train(trainingSet_x)

    # conversione dei DataFrame in vettori 2D
    dataPreparation.matrix(trainingSet_x, None, trainingSet_y, None)

    # scaling del training set
    dataPreparation.standardScaler(trainingSet_x, None)

    # Principal Component Analysis
    dataPreparation.principalComponentAnalysis(trainingSet_x, None)

    # resampling
    dataPreparation.Resampling(trainingSet_x, trainingSet_y, "AllKNN")

    # salvataggio del dizionario trainingSet_x.outliersDict in un file (utile per la valutazione successiva)
    dataPreparation.save_object(trainingSet_x.outliersDict, 'dict.pkl')


def evaluation_train(trainingSet_x, trainingSet_y):
    clf = QuadraticDiscriminantAnalysis(reg_param=0.0001, store_covariance=True, tol=0.1)

    print("QuadraticDiscriminantAnalysis")
    clf.fit(trainingSet_x.data, trainingSet_y.data.ravel())

    return clf


def outliers_train(trainingSet_x):
    """
    Individuazione e sostituzione degli outliers con i metodi ZScore e KNN.
    :param trainingSet_x: training set x
    """
    for colName in trainingSet_x.data.columns:
        # individuo gli outliers con ZScore
        dataPreparation.outZSCORE(trainingSet_x, None, colName)

        # una volta che ho la lista di outliers, trovo i valori con cui sostituirli
        dataPreparation.knnDetectionTRAIN(trainingSet_x, None, colName)

        # sostuituisco gli outliers
        substituteOutliersTrain(trainingSet_x, colName)

        # controllo outliers dopo aver applicato KNN
        checkOutliersAfterReplacementTrain(trainingSet_x, colName)


def substituteOutliersTrain(trainingSet_x, colName):
    """
    Sostituzione degli outliers appartenenti alla colonna 'colName'
    :param trainingSet_x: training set x
    :param colName: colonna interessata
    """
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


def main():
    datasetPath = './training_set.csv'
    dataset = pd.read_csv(datasetPath)

    trainingSet_x = Dataset("trainingSet_x", None)  # feature x
    trainingSet_y = Dataset("trainingSet_y", None)  # target y

    # separiamo le features x dal target y
    train_x = dataset.iloc[:, 0:20].values
    train_y = dataset.iloc[:, 20].values

    # Pre Proceccing del training set
    preProcessing_train(trainingSet_x, trainingSet_y, train_x, train_y)

    # Evaluation training set
    clf = evaluation_train(trainingSet_x, trainingSet_y)

    dataPreparation.save_object(clf, 'returned_clf.pkl')


if __name__ == '__main__':
    main()
