
import sklearn
import sklearn.preprocessing as prep
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsRegressor

import seaborn as sns

#method = "IQR"
method = "ZSCORE"

class Dataset:
  def __init__(self, name, data):
    self.name = name
    self.data = data
    self.naCount = None
    #self.dataAfterIQR = None
    #self.dataAfterZSCORE = None
    self.outliers = None    #lista outliers di una colonna
    self.dataColumn = None #elementi di una colonna
    self.result = None


def openFiles(dataset, train_x, test_x, train_y, test_y):

    x = dataset.iloc[:, 0:20].values
    y = dataset.iloc[:, 20].values

    train_x.data, test_x.data, train_y.data, test_y.data = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=0)
    print('Train:', train_x.data.shape, train_y.data.shape)
    print('Test:', test_x.data.shape, test_y.data.shape)

    # traformo train_x.data da numpy.ndarray in un DataFrame
    # altrimenti non si può utilizzare attributo .isna() nella def get_na_count()
    train_x.data = pd.DataFrame(train_x.data)
    test_x.data = pd.DataFrame(test_x.data)
    train_y.data = pd.DataFrame(train_y.data)
    test_y.data = pd.DataFrame(test_y.data)

    #aggiungiamo i nomi alle colonne in tutti e 4 i data frames
    changeColNames(train_x.data)
    changeColNames(test_x.data)
    changeColNames(train_y.data)
    changeColNames(test_y.data)

    # calcoliamo il numero di valori mancanti su train e test (n/a)
    naMean(train_x,test_x)

    '''
    #method = "IQR"
    method = "ZSCORE"
    '''

    #outliers detection
    outlierDetection(train_x, test_x)


#questa funzione copia in dataColumn tutti gli elementi di una colonna, e per ogni colonna
#si gestiscono gli outlier con metodi opportuni

def outlierDetection(train_x, test_x):

    for colName in train_x.data.columns:
        print("\n\ncolName = ", colName)

        '''
        #aggiungo gli elementi del training set apaprtenenti alla colonna 'colName' nell'array dataColumnTrain
        train_x.dataColumn = np.array([])

        for colElement in train_x.data[colName]:
            train_x.dataColumn = np.append(train_x.dataColumn, colElement)


        # aggiungo gli elementi del test set apaprtenenti alla colonna 'colName' nell'array dataColumnTest
        test_x.dataColumn = np.array([])

        for colElement in test_x.data[colName]:
            test_x.dataColumn = np.append(test_x.dataColumn, colElement)
        
        '''

        #print("\n\ndata2: ", dataColumn, "\n\n")

        title = colName + ' before KNN'

        #metodi diversi per il calcolo di outliers sia per train_x che per test_x
        if(method == "IQR"):

            print("\nOUTLIERS WITH IQR")
            print("\n------ train ------")
            outIQR(train_x,"train" + title,colName)
            print("\n------ test ------")
            outIQR(test_x,"test" + title,colName)


        if(method == "ZSCORE"):

            print("\n\nOUTLIERS WITH ZSCORE\n")
            print("\n------ train ------")
            outZSCORE(train_x, colName)
            print("\n------ test ------")
            outZSCORE(test_x,colName)



        #una volta che ho la lista di outliers, li sostituisco con il metodo KNN, che avrà come input sia
        #il training che il test, poichè devo modificarli entrambi colonna x colonna
        knnDetectionTRAIN(train_x,test_x,colName)
        # sostuituiamo i risultati con gli outliers nel dataset originario
        substituteOutliers(train_x, colName)
        # controllo outliers dopo aver applicato KNN
        checkOutliersAfterKNN(train_x, colName)

        print("\n\n--------- KNN TEST ------ ")
        substituteOutliers(test_x, colName)
        checkOutliersAfterKNN(test_x, colName)




#calcola gli outliers con il metodo IQR e stampa il boxplot
#input: colonna training set della quale troviamo gli outliers e titolo boxplot
#output: lista outliers per la colonna

def outIQR(dataset,title,colName):

    dataset.dataColumn = np.array([])

    for colElement in dataset.data[colName]:
        dataset.dataColumn = np.append(dataset.dataColumn, colElement)


    fig1, ax = plt.subplots()
    ax.set_title(title)
    ax.boxplot(dataset.dataColumn)

    median = np.median(dataset.dataColumn)
    q3 = np.percentile(dataset.dataColumn, 75)  # upper_quartile
    q1 = np.percentile(dataset.dataColumn, 25)  # lower_quartile
    iqr = q3 - q1

    print("mediana: ", median)
    print("q1: ", q1)
    print("q3: ", q3)
    print("iqr: ", iqr)

    l = q1 - 1.5 * iqr
    r = q3 + 1.5 * iqr
    print("l: ", l, "    r:", r)

    # trovo gli outliers e li inserisco in una lista

    dataset.outliers = []
    count = 0
    for i in dataset.dataColumn:
        if i < l or i > r:
            count = count + 1
            dataset.outliers.append(i)
            print("-- outlier n ", count, ":  ", dataset.outliers[count - 1])

    ax.set_xlim(right=1.5)
    plt.show()

    return dataset.outliers

#calcola gli outliers con il metodo ZSCORE
#input: colonna training set della quale troviamo gli outliers
#output: lista outliers per la colonna

def outZSCORE(dataset,colName):

    dataset.dataColumn = np.array([])

    for colElement in dataset.data[colName]:
        dataset.dataColumn = np.append(dataset.dataColumn, colElement)

    count = 0
    threshold = 3
    mean = np.mean(dataset.dataColumn)
    std = np.std(dataset.dataColumn)
    dataset.outliers = []
    for i in dataset.dataColumn:
        # print("vaffanculo2")

        z = (i - mean) / std
        if z > threshold:
            # print("vaffanculo3")

            count = count + 1
            dataset.outliers.append(i)
            print("-- outlier n ", count, ":  ", dataset.outliers[count - 1])

    return dataset.outliers


#data la lsita di outliers di una colonna, li sostituisco con il metodo KNN, che avrà come input sia
#il training che il test, poichè devo modificarli entrambi colonna x colonna
def knnDetectionTRAIN(train_x, test_x, colName):

    # copio dataset in lista y e tolgo outliers
    y = train_x.data[colName].copy()
    # print("y = ", y)
    for i in train_x.outliers:
        # print ("i= ",i)
        # y.remove(i)
        y = y[y != i]
        # print("y = ", y)

    # print("y = ", y)
    # print("data2: ", data2)


    # ORA ABBIAMO TOLTO E SOSTITUITO OUTLIER : USIAMO KNN !!!!!

    lenX = len(train_x.data[colName]) - len(train_x.outliers)
    rows = lenX
    col = 1
    X = [[0 for i in range(col)] for j in range(rows)]  # inizializzo X come lista 2D
    count_X_position = 0

    # metto dati nella lista 2D "X"

    # per creare lista 2D "X" per poterla usare in KNN in cui devono andarci tutti i valori di data2 tranne outliers
    # così poi a KNN gli do X senza outlier che gli passo separatamente, in modo da calcolare media dei k vicini e sostituirli
    for i in y:
        # print("count X = ", count_X_position,"     data2_elem = ",i)
        X[count_X_position][0] = i
        # print("X[count_X_position][0] = ", X[count_X_position][0])
        count_X_position = count_X_position + 1


    print("\n\n--------- KNN TRAIN ------ ")

    # fit
    neigh = KNeighborsRegressor(n_neighbors=3)
    neigh.fit(X, y)

    # predict
    result = []
    for i in train_x.outliers:
        result.append(neigh.predict([[i]]))
    # print("result: ", result[0][0],result[1][0])
    #print("\n\nresult: ", result)


    #poichè nel test set non ho una corrispondenza 1 a 1 tra result e numero di outliers, rimuovo da result
    #i duplicati, poichè avrò un solo result per gli oulliers inferiori e un solo resutl per quelli superiori
    result = np.unique(result, axis=0)
    print("result senza duplicati: ", result)

    if len(result) > 2:
        print("CECILIA AIUTOOOOOOOOOOOOOOOOOOOOO")

    train_x.result = result
    test_x.result = result


def substituteOutliers(dataset, colName):

    #print("\n\n--------- KNN TEST ------ ")


    '''
    # result = [[-2.71536111] [ 2.65369323]]
    #   outliers =      2.8855370482008214
    -- outlier n  2 :   2.9115293876047064
    -- outlier n  3 :   3.1141434886609813
    -- outlier n  4 :   -3.1585339273483033
    -- outlier n  5 :   -3.3372981576055034
    -- outlier n  6 :   -3.3824899824206835
    '''

    if method == "IQR" :

        # sostuituiamo i risultati con gli outliers nel dataset originario
        for i in dataset.outliers:
            res = checkClosestOutlier(i, dataset.result)
            dataset.data[colName][dataset.data[colName] == i] = (res)

    if method == "ZSCORE" :

        for i in dataset.outliers:
            dataset.data[colName][dataset.data[colName] == i] = (dataset.result[0][0])

    #checkOutliersAfterKNN(dataset,colName)

def checkClosestOutlier(outlier,resultList):

    '''
    resultList[0][0] = -2.71536111
    resultList[1][0] = 2.65369323
    outlier n  1 :    2.8855370482008214
    outlier n  6 :   -3.3824899824206835

    diff1 e diff2 sono le distanze in valore assoluto dall'outlier a entrambi i valori di resultList

    Nel caso di outlier n 1 bisogna sostituirlo con resultList[1][0], che è il valore più vicino
    quindi calcolo la distanza dell'outlier dai due valori contenuti in resulList,
    e prendo la distanza minore (valore assoluto)
    '''

    diff1 = abs(outlier - resultList[0][0])
    diff2 =abs(outlier - resultList[1][0])
    #print("diff1 : ", diff1, "  diff2: ",diff2)


    if diff2 < diff1 :
        return diff2
    else :
        return diff1

def checkOutliersAfterKNN(dataset,colName):

    if method == "IQR":
        # CALCOLO OUTLIERS DEL TRAINING SET FINALE DELLA SIGNOLA FEATURE
        title = colName + "after KNN"
        outliers = outIQR(dataset, title, colName)

    if method == "ZSCORE":
        outliers = outZSCORE(dataset,colName)

    if len(outliers) == 0:
        print(colName, ": KNN terminato, outliers sostituiti\n\n")
        return 0


def changeColNames(dfDataset):
    # print("TOTALE PRIMA-   ", df_imputed)
    # print("COLONNA 0-   ", df_imputed[0])

    string = "F"
    for i in range(1, 21):
        currColumn = string + str(i)
        index = i - 1
        # print("index: ", index)
        # print(df_imputed.rename(columns={index: 'F1'}))
        dfDataset.rename(columns={index: currColumn}, inplace=True)

    print("TOTALE DOPO-      ", dfDataset)



#sostuisce NaN con media per ogni colonna
def naMean(train_x, test_x):

    getNaCount(train_x)
    print("train x na count : ", train_x.naCount)
    getNaCount(test_x)
    print("test x na count : ", test_x.naCount)

    # print(train_dataset['F1'].mean())

    print("\n\nMEDIA PER OGNI ATTRIBUTO: ")

    string = "F"
    for i in range(1, 21):
        currColumn = string + str(i)
        currMean = train_x.data[currColumn].mean()

        print(currColumn, ": ", currMean)

        train_x.data[currColumn] = train_x.data[currColumn].fillna(currMean)
        test_x.data[currColumn] = test_x.data[currColumn].fillna(currMean)

    # controlliamo nuovamente che train e test siano senza n/a
    getNaCount(train_x)
    print("train x na count : ", train_x.naCount)
    getNaCount(test_x)
    print("test x na count : ", test_x.naCount)

def getNaCount(dataset):
    # per ogni elemento (i,j) del dataset, isna() restituisce
    # TRUE/FALSE se il valore corrispondente è mancante/presente
    boolean_mask = dataset.data.isna()
    # contiamo il numero di TRUE per ogni attributo sul dataset
    count = boolean_mask.sum(axis=0)
    # print("count NaN: ",count)
    dataset.naCount=count

def main():
    datasetPath = './training_set.csv'
    dataset = pd.read_csv(datasetPath)
    train_x= Dataset("train_x",None)
    test_x=Dataset("test_x",None)

    train_y= Dataset("train_y",None)
    test_y = Dataset("test_y", None)

    openFiles(dataset, train_x, test_x, train_y, test_y)

'''
    print("\n\n DOPO ----------------")
    print('Train:', train_x.data.shape, train_y.data.shape, "   col: ")
    print('Test:', test_x.data.shape, test_y.data.shape)

    print("Shape:", train_x.data)
    train_x.naCount=2
    print("na:", train_x.naCount)

'''

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()