import sklearn
import sklearn.preprocessing as prep
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier

import pickle
import dataPreparation

find_method = dataPreparation.find_method
substitute_method = dataPreparation.substitute_method
scaleType = dataPreparation.scaleType


class Dataset:
  def __init__(self, name, data):
    self.name = name
    self.data = data
    self.naCount = None
    self.outliers = None    #lista outliers di una colonna
    self.dataColumn = None  #elementi di una colonna
    self.result = None
    self.outliersDict = {}


def preProcessing_train(trainingSet_x, trainingSet_y, train_x, train_y):

    trainingSet_x.data = train_x
    trainingSet_y.data = train_y

    print('SHAPE : Train_x:', train_x.data.shape,"   train_y:", train_y.data.shape)
    print('Train_x:', trainingSet_x.data, "   train_y:", trainingSet_y.data)

    trainingSet_x.data = pd.DataFrame(trainingSet_x.data)
    trainingSet_y.data = pd.DataFrame(trainingSet_y.data, columns=['CLASS'])

    dataPreparation.changeColNames(trainingSet_x.data)

    naDict = dataPreparation.naMean2(trainingSet_x, None)
    print ("dictionary medie: ", naDict)

    outliers_train(trainingSet_x)
    print("dictionary outliers: ", trainingSet_x.outliersDict)

    dataPreparation.scale(trainingSet_x, None, trainingSet_y, None)

    dataPreparation.pca(trainingSet_x, None)


def evaluation_train(trainingSet_x, trainingSet_y):

    n_folds = 5
    metric = 'f1_macro'

    classifier = MLPClassifier(max_iter=200)

    parameter_space = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        'alpha': [0.5, 1, 1.2],
    }


    clf = model_selection.GridSearchCV(classifier, parameter_space, scoring=metric, cv=n_folds, refit=True, n_jobs=-1)
    print("MLP")
    clf.fit(trainingSet_x.data, trainingSet_y.data.ravel())
    best_parameters = clf.best_params_
    print("\n\nbest_parameters MLP : ", best_parameters)
    best_result = clf.best_score_
    print("best_result MLP: ", best_result)

    return clf

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)



def outliers_train(trainingSet_x):

    for colName in trainingSet_x.data.columns:
        print("\n\ncolName = ", colName)

        title = colName + ' before KNN'

        #metodi diversi per il calcolo di outliers sia per train_x che per test_x
        if find_method == "IQR":

            print("\nOUTLIERS WITH IQR")
            print("\n------ train ------")
            dataPreparation.outIQR(trainingSet_x, "train" + title, colName)

            '''
            fig1, ax = plt.subplots()
            ax.set_title(colName + " before KNN")
            ax.set_xticklabels(['TRAIN', 'TEST'])

            ax.boxplot([train_x.dataColumn, test_x.dataColumn])
            plt.show()
            '''

        if find_method == "ZSCORE":

            print("\n\nOUTLIERS WITH ZSCORE\n")
            print("\n------ train ------")
            dataPreparation.outZSCORE(trainingSet_x, colName)


        if substitute_method == "KNN":

            # una volta che ho la lista di outliers, li sostituisco con il metodo KNN, che avrà come input sia
            # il training che il test, poichè devo modificarli entrambi colonna x colonna
            dataPreparation.knnDetectionTRAIN(trainingSet_x, None, colName)

        else:
            dataPreparation.outlierMean(trainingSet_x, None, colName)

        # sostuituiamo i risultati con gli outliers nel dataset originario
        dataPreparation.substituteOutliers(trainingSet_x, colName)
        # controllo outliers dopo aver applicato KNN
        dataPreparation.checkOutliersAfterReplacement(trainingSet_x, colName)

        #print("dictionary outliers: ",outliersDict)

        '''
        fig1, ax = plt.subplots()
        ax.set_title(colName + " after KNN")
        ax.set_xticklabels(["TRAIN", "TEST"])

        ax.boxplot([train_x.dataColumn, test_x.dataColumn])
        # fig1.tight_layout()

        plt.show()
        '''



def main():
    datasetPath = './training_set.csv'
    dataset = pd.read_csv(datasetPath)

    trainingSet_x = Dataset("trainingSet_x", None)    #feature x
    trainingSet_y = Dataset("trainingSet_y", None)        #target y

    # separiamo le features x dal target y
    train_x = dataset.iloc[:, 0:20].values
    train_y = dataset.iloc[:, 20].values

    preProcessing_train(trainingSet_x, trainingSet_y, train_x, train_y)
    clf = evaluation_train(trainingSet_x, trainingSet_y)
    save_object(clf,'returned_clf.pkl')

if __name__ == '__main__':
    main()