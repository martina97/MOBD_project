from collections import Counter

import sklearn
import sklearn.preprocessing as prep
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, NearMiss, CondensedNearestNeighbour, TomekLinks, \
    ClusterCentroids, EditedNearestNeighbours, OneSidedSelection, NeighbourhoodCleaningRule, \
    RepeatedEditedNearestNeighbours, AllKNN, InstanceHardnessThreshold
from matplotlib import pyplot as plt
from pyod.models.knn import KNN
from scipy.stats import stats
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold, RFE, SelectKBest, mutual_info_classif, f_classif
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer, Normalizer

import crossValidation
from sklearn.cluster import DBSCAN
from matplotlib import cm
import winsound

#outliers:

#find_method = "IQR"
find_method = "ZSCORE"
#find_method = "ZSCORE2"
#find_method = "DBSCAN"

substitute_method = "KNN"
#substitute_method = "MEAN"

scaleType = "STANDARD"
#scaleType = "MINMAX"
#scaleType = "MAX_ABS"
#scaleType = "ROBUST"

class Dataset:
  def __init__(self, name, data):
    self.name = name
    self.data = data
    self.naCount = None
    self.outliers = None    #lista outliers di una colonna
    self.dataColumn = None  #elementi di una colonna
    self.result = None
    self.outliersDict = {} # { F1: [meanNA, meanZSCORE, stdZSCORE, resultOUTLIERS]


def preProcessing(train_x, test_x, train_y, test_y, x, y):

    train_x.data, test_x.data, train_y.data, test_y.data = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=42)
    print('Train:', train_x.data.shape, train_y.data.shape)
    print('Test:', test_x.data.shape, test_y.data.shape)

    # traformo train_x.data da numpy.ndarray in un DataFrame
    # altrimenti non si può utilizzare attributo .isna() nella def get_na_count()
    train_x.data = pd.DataFrame(train_x.data)
    test_x.data = pd.DataFrame(test_x.data)
    train_y.data = pd.DataFrame(train_y.data, columns=['CLASS'])
    test_y.data = pd.DataFrame(test_y.data, columns=['CLASS'])

    #aggiungiamo i nomi alle colonne in tutti e 4 i data frames
    changeColNames(train_x.data)
    changeColNames(test_x.data)

    '''
        if find_method == "IQR":
        np.savetxt("train_x.data_INIZIALE_iqr.csv", train_x.data, delimiter=",")
        np.savetxt("train_y.data_INIZIALE_iqr.csv", train_y.data, delimiter=",")
    else:

        np.savetxt("train_x.data_INIZIALE_z.csv", train_x.data, delimiter=",")
        np.savetxt("train_y.data_INIZIALE_z.csv", train_y.data, delimiter=",")
        
    '''




    # calcoliamo il numero di valori mancanti su train e test (n/a)

    #naMean(train_x,test_x)


    imputer = KNNImputer(n_neighbors=1)
    imputed = imputer.fit_transform(train_x.data)
    imputed_test = imputer.transform(test_x.data)
    train_x.data = pd.DataFrame(imputed, columns=train_x.data.columns)
    test_x.data = pd.DataFrame(imputed_test, columns=test_x.data.columns)

    '''
    imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
    #imputer.fit(train_x.data)
    #imputer.fit(test_x.data)
    getNaCount(train_x)
    train_x.data = pd.DataFrame(imputer.fit_transform(train_x.data))
    getNaCount(train_x)
    test_x.data = pd.DataFrame(imputer.fit_transform(test_x.data))
    '''

    #dbScan(train_x)
    #zScore(train_x)


    #outliers detection
    if find_method == "ZSCORE2":
        outlierDetection_zScoreGlobal(train_x, test_x)


    else:
        outlierDetection(train_x, test_x)

    #dbScan(train_x, train_y)
    #dbScan(test_x, test_y)



    #normalizziamo i dati
    scale(train_x, test_x, train_y, test_y)

    #dbScan(train_x)

    #applichiamo PCA
    pca(train_x, test_x)

    '''
    fs = SelectKBest(score_func=mutual_info_classif, k=13)
    # fs = SelectKBest(score_func=f_classif, k='all')

    fs.fit(train_x.data, train_y.data.ravel())
    train_x.data = fs.transform(train_x.data)
    test_x.data = fs.transform(test_x.data)
    '''
    #SMOTE

    '''
    # Under-sampling:
    ros = RandomUnderSampler(random_state=42)
    #ros = RandomOverSampler(random_state=42)

    (train_x.data, train_y.data) = ros.fit_sample(train_x.data, train_y.data)
    (test_x.data, test_y.data) = ros.fit_sample(test_x.data, test_y.data)
    '''

    # define the undersampling method
    #undersample = TomekLinks()
    #undersample =EditedNearestNeighbours(n_neighbors=1)
    #undersample = NeighbourhoodCleaningRule(n_neighbors=2, threshold_cleaning=0.5)
    #undersample = NeighbourhoodCleaningRule(n_neighbors=2,threshold_cleaning=0)

    #migliori
    #undersample = RepeatedEditedNearestNeighbours(n_neighbors=10, max_iter = 900000, kind_sel = 'mode', n_jobs = -1)
    #undersample = NeighbourhoodCleaningRule(n_neighbors=10,threshold_cleaning=0, n_jobs = -1,  kind_sel = 'mode')
    #undersample =EditedNearestNeighbours(n_neighbors=10, kind_sel = 'mode', n_jobs = -1)

    undersample =AllKNN(allow_minority=True, n_neighbors=10, kind_sel = 'mode', n_jobs = -1) #migliore!!!!
    #undersample =InstanceHardnessThreshold(random_state=42)


    # transform the dataset
    train_x.data, train_y.data = undersample.fit_resample(train_x.data, train_y.data)
    #test_x.data, test_y.data = undersample.fit_resample(test_x.data, test_y.data)


    '''
    oversample = SMOTE()
    train_x.data, train_y.data = oversample.fit_resample(train_x.data, train_y.data)
    counter = Counter(train_y.data)
    for k,v in counter.items():
        per = v / len(y) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
    '''

    '''
    #Under-sampling:
    ros = RandomUnderSampler(random_state=0)
    (train_x.data, train_y.data) = ros.fit_sample(train_x.data, train_y.data)
    
    
    #Over-sampling:
    ros = RandomOverSampler(random_state=0)
    (train_x.data, train_y.data) = ros.fit_sample(train_x.data, train_y.data)
    (test_x.data, test_y.data) = ros.fit_sample(test_x.data, test_y.data)
    counter = Counter(test_y.data)
    for k, v in counter.items():
        per = v / len(y) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k, v, per))

    '''

def zScore(dataset):
    z = np.abs(stats.zscore(dataset.data))
    print(z)
    threshold = 3
    print(np.where(z > 3))
    print(len(np.where(z > 3)[0]))
    return z



def pca(train_x, test_x):
    pca = PCA()
    train_x.data = pca.fit_transform(train_x.data)

    if test_x is not None:
        test_x.data = pca.transform(test_x.data)
    explained_variance = pca.explained_variance_ratio_
    count = 0
    for i in explained_variance:
        count = count + 1
        print("explained_variance", count, "--->", i)

    print("explained_variance:", explained_variance)

    pca = PCA(n_components=15)
    train_x.data = pca.fit_transform(train_x.data)

    if test_x is not None:
        test_x.data = pca.transform(test_x.data)





def scale(train_x, test_x, train_y, test_y):

        matrix(train_x, test_x, train_y, test_y)

        if scaleType=="STANDARD":
            standardScaler(train_x, test_x)
        if scaleType == "MINMAX":
            minMaxScaler(train_x, test_x)
        if scaleType == "MAX_ABS":
            maxAbsScaler(train_x, test_x)
        if scaleType == "ROBUST":
            robustScaler(train_x, test_x)



def matrix(train_x, test_x, train_y, test_y):
    #convertiamo i DataFrame per l'input e per l'output in vettori 2D (matrici)
    train_x.data = np.float64(train_x.data)
    train_y.data = np.float64(train_y.data)
    train_y.data = train_y.data.reshape((len(train_y.data), 1))

    if test_x is not None or test_y is not None:
        test_x.data = np.float64(test_x.data)
        test_y.data = np.float64(test_y.data)
        test_y.data = test_y.data.reshape((len(test_y.data), 1))
        print("MATRIX Y: ", test_x.data.shape, test_y.data.shape)

    print("MATRIX X: ",train_x.data.shape, train_y.data.shape)



def  standardScaler(train_x, test_x):
    '''
    È buona norma normalizzare le feature che utilizzano scale e intervalli diversi.
    Non normalizzare i dati rende l'allenamento più difficile e rende il modello risultante
    dipendente dalla scelta delle unità nell'input.
    '''

    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(train_x.data) # ATTENTIONE! SI USA LA MEDIA E VARIANZA DEL TRAINING SET
    train_x.data = scaler.transform(train_x.data)

    if test_x is not None:

        test_x.data = scaler.transform(test_x.data)


def minMaxScaler(train_x, test_x):

    #feature_range=(0, 2)
    scaler_x = prep.MinMaxScaler(feature_range=(-2, 2))
    scaler_x.fit(train_x.data)

    train_x.data = scaler_x.transform(train_x.data)
    if test_x is not None:
        test_x.data = scaler_x.transform(test_x.data)

def maxAbsScaler(train_x, test_x):
    scaler = MaxAbsScaler()
    scaler.fit(train_x.data)
    train_x.data = scaler.transform(train_x.data)

    if test_x is not None:
        test_x.data = scaler.transform(test_x.data)

def robustScaler(train_x, test_x):
    scaler = RobustScaler(quantile_range=(25, 75))
    scaler.fit(train_x.data)
    train_x.data = scaler.transform(train_x.data)

    if test_x is not None:
        test_x.data = scaler.transform(test_x.data)



#sostuisce outliers con media per ogni colonna
def outlierMean(train_x, test_x, colName):

    # copio dataset in lista y e tolgo outliers
    y = train_x.data[colName].copy()
    for i in train_x.outliers:
        y = y[y != i]

    mean = y.mean()
    print("\n\n -- Media di ",colName," : ", mean)
    train_x.result = mean
    appendDict(colName,mean,train_x)
    #train_x.outliersDict[colName] = mean

    if test_x is not None:
        test_x.result = mean


#questa funzione copia in dataColumn tutti gli elementi di una colonna, e per ogni colonna
#si gestiscono gli outlier con metodi opportuni


def outlierMedian(train_x, test_x, colName):

    # copio dataset in lista y e tolgo outliers
    y = train_x.data[colName].copy()
    for i in train_x.outliers:
        y = y[y != i]

    median = y.median()
    print("\n\n -- Media di ", colName, " : ", median)
    train_x.result = median
    appendDict(colName, median, train_x)
    # train_x.outliersDict[colName] = mean

    if test_x is not None:
        test_x.result = median



def outlierDetection_zScoreGlobal(train_x, test_x):
    if find_method == "ZSCORE2":

        # z = z-Score del dataset
        z_train = zScore(train_x)
        z_test = zScore(test_x)
        print("z_train_len == ", len(z_train[0]))

        #serve per scorrere le colonne quando uso ZSCORE2
        col_z = 0

        for colName in train_x.data.columns:
            print("\n\ncolName = ", colName)

            z_array_train = np.where(z_train > 3)
            print("\n\nOUTLIERS WITH ZSCORE_global\n")
            print("\n------ train ------")
            outZSCORE_global(train_x, z_array_train, col_z, z_train, colName)
            print("\n------ test ------")
            z_array_test = np.where(z_test > 3)

            outZSCORE_global(test_x, z_array_test, col_z, z_test, colName)

            col_z = col_z +1

            if substitute_method == "KNN":

                # una volta che ho la lista di outliers, li sostituisco con il metodo KNN, che avrà come input sia
                # il training che il test, poichè devo modificarli entrambi colonna x colonna
                knnDetectionTRAIN(train_x, test_x, colName)

            else:
                outlierMean(train_x, test_x, colName)
                # outlierMedian(train_x,test_x,colName)

            # sostuituiamo i risultati con gli outliers nel dataset originario
            substituteOutliers(train_x, colName)

            print("\n\n--------- KNN TEST ------ ")
            substituteOutliers(test_x, colName)


        #abbiamo modificato il dataset sostituendo gli outliers
        z_train = zScore(train_x)
        z_test = zScore(test_x)
        col_z = 0

        for colName in train_x.data.columns:
            z_array_train = np.where(z_train > 3)
            print("\n\nOUTLIERS WITH ZSCORE_global\n")
            print("\n------ train ------")
            outliers = outZSCORE_global(train_x, z_array_train, col_z, z_train, colName)
            if len(outliers) == 0:
                print(colName, ": KNN terminato, outliers sostituiti\n\n")
            else:
                print("no")


            print("\n------ test ------")
            z_array_test = np.where(z_test > 3)
            outliers = outZSCORE_global(test_x, z_array_test, col_z, z_test, colName)
            if len(outliers) == 0:
                print(colName, ": KNN terminato, outliers sostituiti\n\n")

            else:
                #todo: errore da fare
                print("no")

            col_z = col_z + 1

def outlierDetection(train_x, test_x):

    for colName in train_x.data.columns:
        print("\n\ncolName = ", colName)

        title = colName + ' before KNN'

        #metodi diversi per il calcolo di outliers sia per train_x che per test_x
        if find_method == "IQR":

            print("\nOUTLIERS WITH IQR")
            print("\n------ train ------")
            #outIQR(train_x,"train" + title,colName)
            outIQR2(train_x, test_x, "train" + title,colName)

            #print("\n------ test ------")
            #outIQR(test_x,"test" + title,colName)

            fig1, ax = plt.subplots()
            ax.set_title(colName + " before KNN")
            ax.set_xticklabels(['TRAIN', 'TEST'])

            ax.boxplot([train_x.dataColumn, test_x.dataColumn])
            plt.show()




        if find_method == "ZSCORE":

            print("\n\nOUTLIERS WITH ZSCORE\n")
            #print("\n------ train ------")
            #outliers, mean, std  = outZSCORE(train_x, colName)
            outliers, mean, std = outZSCORE2(train_x,test_x, colName)
            #outliers, mean, std = outZSCORE3(train_x,test_x, colName)

            appendDict(colName, mean, train_x)
            appendDict(colName, std, train_x)
            #print("\n------ test ------")
            #outZSCORE(test_x,colName)





        '''
        if find_method == "DBSCAN":
            print("\n\nOUTLIERS WITH DBSCAN\n")
            print("\n------ train ------")
            dbScan(train_x, colName)
            print("\n------ test ------")
            dbScan(test_x, colName)
        '''

        if substitute_method == "KNN" :

            #una volta che ho la lista di outliers, li sostituisco con il metodo KNN, che avrà come input sia
            #il training che il test, poichè devo modificarli entrambi colonna x colonna
            #knnDetectionTRAIN(train_x,test_x,colName)
            knnDetectionTRAIN(train_x,test_x,colName)



        else:
            outlierMean(train_x,test_x,colName)
            #outlierMedian(train_x,test_x,colName)



        # sostuituiamo i risultati con gli outliers nel dataset originario
        substituteOutliers(train_x, colName)
        # controllo outliers dopo aver applicato KNN
        checkOutliersAfterReplacement(train_x, colName)


        print("\n\n--------- KNN TEST ------ ")
        substituteOutliers(test_x, colName)
        checkOutliersAfterReplacement(test_x, colName)

        fig1, ax = plt.subplots()
        ax.set_title(colName + " after KNN")
        ax.set_xticklabels(["TRAIN", "TEST"])

        ax.boxplot([train_x.dataColumn, test_x.dataColumn])
        #fig1.tight_layout()

        plt.show()


def dbScan(dataset_x, dataset_y):

    model = DBSCAN(
     eps = 5,
     metric='euclidean',
     min_samples = 5,
     n_jobs = -1).fit_predict(dataset_x.data)

    #clusters = model.fit_predict(train_x.data)
    mask = model != -1

    outliers = dataset_x.data[mask]
    print("model ==== ", model, "\n\n")

    print("outliers ==== ", outliers, "\n\n")

    print("mask------------", mask)

    dataset_x.data  = dataset_x.data.iloc[mask, :]
    dataset_y.data =  dataset_y.data.iloc[mask]
    print( dataset_x.data.shape,  dataset_y.data.shape)


#calcola gli outliers con il metodo IQR e stampa il boxplot
#input: colonna training set della quale troviamo gli outliers e titolo boxplot
#output: lista outliers per la colonna

def outIQR(dataset,title,colName):

    dataset.dataColumn = np.array([])

    for colElement in dataset.data[colName]:
        dataset.dataColumn = np.append(dataset.dataColumn, colElement)

    '''
    fig1, ax = plt.subplots()
    ax.set_title(title)
    ax.boxplot(dataset.dataColumn)
    '''
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

    '''
    ax.set_xlim(right=1.5)
    plt.show()
    '''
    return dataset.outliers



def outIQR2(train_x,test_x, title,colName):

    print("\n------ train ------")

    train_x.dataColumn = np.array([])

    for colElement in train_x.data[colName]:
        train_x.dataColumn = np.append(train_x.dataColumn, colElement)

    '''
    fig1, ax = plt.subplots()
    ax.set_title(title)
    ax.boxplot(dataset.dataColumn)
    '''
    median = np.median(train_x.dataColumn)
    q3 = np.percentile(train_x.dataColumn, 75)  # upper_quartile
    q1 = np.percentile(train_x.dataColumn, 25)  # lower_quartile
    iqr = q3 - q1

    print("mediana: ", median)
    print("q1: ", q1)
    print("q3: ", q3)
    print("iqr: ", iqr)

    l = q1 - 1.5 * iqr
    r = q3 + 1.5 * iqr
    print("l: ", l, "    r:", r)

    # trovo gli outliers e li inserisco in una lista

    train_x.outliers = []
    count = 0
    for i in train_x.dataColumn:
        if i < l or i > r:
            count = count + 1
            train_x.outliers.append(i)
            print("-- outlier n ", count, ":  ", train_x.outliers[count - 1])




    '''
    ax.set_xlim(right=1.5)
    plt.show()
    '''

    print("\n------ test ------")
    test_x.dataColumn = np.array([])

    for colElement in train_x.data[colName]:
        test_x.dataColumn = np.append(test_x.dataColumn, colElement)

    test_x.outliers = []
    count = 0
    for j in test_x.dataColumn:
        if j < l or j > r:
            count = count + 1
            test_x.outliers.append(j)
            print("-- outlier n ", count, ":  ", test_x.outliers[count - 1])



    return train_x.outliers


def outIQR3(train_x,test_x, title,colName):

    print("\n------ train ------")

    train_x.dataColumn = np.array([])

    for colElement in train_x.data[colName]:
        train_x.dataColumn = np.append(train_x.dataColumn, colElement)

    '''
    fig1, ax = plt.subplots()
    ax.set_title(title)
    ax.boxplot(dataset.dataColumn)
    '''
    median = np.median(train_x.dataColumn)
    q3 = np.percentile(train_x.dataColumn, 75)  # upper_quartile
    q1 = np.percentile(train_x.dataColumn, 25)  # lower_quartile
    iqr = q3 - q1

    print("mediana: ", median)
    print("q1: ", q1)
    print("q3: ", q3)
    print("iqr: ", iqr)

    l = q1 - 1.5 * iqr
    r = q3 + 1.5 * iqr
    print("l: ", l, "    r:", r)

    # trovo gli outliers e li inserisco in una lista

    train_x.outliers = []
    count = 0
    for i in train_x.dataColumn:
        if i < l or i > r:
            count = count + 1
            #train_x.outliers.append(i)
            train_x.data[colName][train_x.data[colName] == i] = np.nan

            print("-- outlier n ", count, ":  ", i)

    '''
    ax.set_xlim(right=1.5)
    plt.show()
    '''

    print("\n------ test ------cacca")
    test_x.dataColumn = np.array([])

    for colElement in train_x.data[colName]:
        test_x.dataColumn = np.append(test_x.dataColumn, colElement)

    test_x.outliers = []
    count = 0
    for j in test_x.dataColumn:
        if j < l or j > r:
            count = count + 1
            #test_x.outliers.append(j)
            test_x.data[colName][test_x.data[colName] == j] = np.nan

            print("-- outlier n ", count, ":  ", test_x.outliers[count - 1])


    return train_x.outliers

#calcola gli outliers con il metodo ZSCORE
#input: colonna training set della quale troviamo gli outliers
#output: lista outliers per la colonna


def createDataColumn(dataset,colName):

    dataset.dataColumn = np.array([])

    for colElement in dataset.data[colName]:
        dataset.dataColumn = np.append(dataset.dataColumn, colElement)

def outZSCORE(dataset,colName):
    '''
    calcola gli outliers con il metodo ZSCORE
    :param dataset:
    :param colName:
    :return:
    '''


    dataset.dataColumn = np.array([])

    for colElement in dataset.data[colName]:
        dataset.dataColumn = np.append(dataset.dataColumn, colElement)


    #print("dataColumn: ", dataset.dataColumn)
    count = 0
    threshold = 3
    mean = np.mean(dataset.dataColumn)
    std = np.std(dataset.dataColumn)
    dataset.outliers = []
    for i in dataset.dataColumn:
        z = (i - mean) / std

        if z > threshold:
            count = count + 1
            dataset.outliers.append(i)
            print("-- outlier n ", count, ":  ", dataset.outliers[count - 1])

    return dataset.outliers, mean, std


def outZSCORE2(train_x,test_x, colName):
    '''
    calcola gli outliers con il metodo ZSCORE
    :param dataset:
    :param colName:
    :return:
    '''

    # train
    print("\n------ train ------")

    train_x.dataColumn = np.array([])

    for colElement in train_x.data[colName]:
        train_x.dataColumn = np.append(train_x.dataColumn, colElement)


    #print("dataColumn: ", dataset.dataColumn)
    count = 0
    threshold = 3
    mean = np.mean(train_x.dataColumn)
    std = np.std(train_x.dataColumn)
    train_x.outliers = []


    for i in train_x.dataColumn:
        z = (i - mean) / std

        if z > threshold :
            count = count + 1
            train_x.outliers.append(i)
            print("-- outlier n ", count, ":  ", train_x.outliers[count - 1])


    print("\n------ test ------")

    # test
    test_x.dataColumn = np.array([])

    for colElement in test_x.data[colName]:
        test_x.dataColumn = np.append(test_x.dataColumn, colElement)

    count = 0
    test_x.outliers = []
    for j in test_x.dataColumn:
        #print("j==", j)
        z = (j - mean) / std

        if z > threshold:
            count = count + 1
            test_x.outliers.append(j)
            print("-- outlier n ", count, ":  ", test_x.outliers[count - 1])


    return train_x.outliers, mean, std

def outZSCORE3(train_x,test_x, colName):
    '''
    calcola gli outliers con il metodo ZSCORE
    :param dataset:
    :param colName:
    :return:
    '''

    # train
    print("\n------ train ------")

    train_x.dataColumn = np.array([])

    for colElement in train_x.data[colName]:
        train_x.dataColumn = np.append(train_x.dataColumn, colElement)


    #print("dataColumn: ", dataset.dataColumn)
    count = 0
    threshold = 3
    mean = np.mean(train_x.dataColumn)
    std = np.std(train_x.dataColumn)
    train_x.outliers = []


    for i in train_x.dataColumn:
        z = (i - mean) / std

        if z > threshold :
            count = count + 1
            #train_x.outliers.append(i)
            train_x.data[colName][train_x.data[colName] == i] = np.nan
            print("-- outlier n ", count, ":  ", i)


    print("\n------ test ------")

    # test
    test_x.dataColumn = np.array([])

    for colElement in test_x.data[colName]:
        test_x.dataColumn = np.append(test_x.dataColumn, colElement)

    count = 0
    test_x.outliers = []
    for j in test_x.dataColumn:
        #print("j==", j)
        z = (j - mean) / std

        if z > threshold:
            count = count + 1
            #test_x.outliers.append(j)
            test_x.data[colName][test_x.data[colName] == j] = np.nan
            print("-- outlier n ", count, ":  ", j)


    return train_x.outliers, mean, std


def outZSCORE_global(dataset, z_array, col_z, z,colName):
    # sto in F1 e devo prendere z_array[1] che hanno valore 0


    col_index = np.where(z_array[1] == col_z)  # contiene gli indici in z_array[1] che indicano la colonna
    row_num = []
    for i in col_index:
        row_num.append(z_array[0][i])

    print("col_index ===== ", col_index)
    print("row_index ===== ", row_num)

    dataset.outliers = []


    for j in row_num[0]:
        # print("j=",j)
        #dataset.outliers.append(z[j][col_z])
        dataset.outliers.append(dataset.data[colName][j])
        print("outlier == ",dataset.data[colName][j])



    print(dataset.outliers)



    #print("cccccccccccc=== ", dataset.outliers[0])





    return dataset.outliers

#data la lsita di outliers di una colonna, li sostituisco con il metodo KNN, che avrà come input sia
#il training che il test, poichè devo modificarli entrambi colonna x colonna
def knnDetectionTRAIN(train_x, test_x, colName):



    # copio dataset in lista y e tolgo outliers
    y = train_x.data[colName].copy()
    for i in train_x.outliers:
        #print("i ==== ", i)
        y = y[y != i]

    print("y ==== ", y)
    # ORA ABBIAMO TOLTO E SOSTITUITO OUTLIER : USIAMO KNN !!!!!
    print("len(train_x.outliers) === ", len(train_x.outliers))
    lenX = len(train_x.data[colName]) - len(train_x.outliers)
    print("lenX == ", lenX)
    print("lenY == ", len(y))

    rows = lenX
    col = 1
    X = [[0 for i in range(col)] for j in range(rows)]  # inizializzo X come lista 2D
    count_X_position = 0

    print("X ==== ", X)

    # metto dati nella lista 2D "X"

    # per creare lista 2D "X" per poterla usare in KNN in cui devono andarci tutti i valori di data2 tranne outliers
    # così poi a KNN gli do X senza outlier che gli passo separatamente, in modo da calcolare media dei k vicini e sostituirli
    for k in y:
        # print("count X = ", count_X_position,"     data2_elem = ",i)
        #print("k ==== ", k)

        X[count_X_position][0] = k
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
    appendDict(colName,result[0][0],train_x)
    #train_x.outliersDict[colName] = result
    #outliersDict[colName] = result

    if len(result) > 2:
        print("Lenght result >2")
        return -1

    train_x.result = result
    if test_x is not None:
        test_x.result = result



def knnDetectionTRAIN2(train_x, test_x, colName):

    imputer = KNNImputer(n_neighbors=2)
    imputed = imputer.fit_transform(train_x.data)
    imputed_test = imputer.transform(test_x.data)
    train_x.data = pd.DataFrame(imputed, columns=train_x.data.columns)
    test_x.data = pd.DataFrame(imputed_test, columns=test_x.data.columns)


def substituteOutliers(dataset, colName):


    '''
    # result = [[-2.71536111] [ 2.65369323]]
    #   outliers =      2.8855370482008214
    -- outlier n  2 :   2.9115293876047064
    -- outlier n  3 :   3.1141434886609813
    -- outlier n  4 :   -3.1585339273483033
    -- outlier n  5 :   -3.3372981576055034
    -- outlier n  6 :   -3.3824899824206835
    '''
    if substitute_method == "KNN":
        if find_method == "IQR":

            # sostuituiamo i risultati con gli outliers nel dataset originario
            for i in dataset.outliers:
                res = checkClosestOutlier(i, dataset.result)
                dataset.data[colName][dataset.data[colName] == i] = (res)

        if find_method == "ZSCORE" :
            for i in dataset.outliers:
                dataset.data[colName][dataset.data[colName] == i] = (dataset.result[0][0])

        if find_method == "ZSCORE2":
            if len(dataset.outliers) == 1:
                for i in dataset.outliers:
                    dataset.data[colName][dataset.data[colName] == i] = (dataset.result[0][0])
            if len(dataset.outliers) == 2:
                for i in dataset.outliers:
                    res = checkClosestOutlier(i, dataset.result)
                    dataset.data[colName][dataset.data[colName] == i] = (res)




    if substitute_method == "MEAN":
        for i in dataset.outliers:
            dataset.data[colName][dataset.data[colName] == i] = dataset.result

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

def checkOutliersAfterReplacement(dataset,colName):

    if find_method == "IQR":
        # CALCOLO OUTLIERS DEL TRAINING SET FINALE DELLA SIGNOLA FEATURE
        title = colName + "after KNN"
        outliers = outIQR(dataset, title, colName)

    if find_method == "ZSCORE":
        outliers = outZSCORE(dataset,colName)[0]

    if find_method == "ZSCORE2":
        outliers = outZSCORE_global(dataset,colName)

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
def naMean2(train_x, test_x):


    getNaCount(train_x)
    print("train x na count : ", train_x.naCount)
    if test_x is not None:
        getNaCount(test_x)
        print("test x na count : ", test_x.naCount)

    # print(train_dataset['F1'].mean())

    print("\n\nMEDIA PER OGNI ATTRIBUTO: ")

    string = "F"
    for i in range(1, 21):
        currColumn = string + str(i)
        currMean = train_x.data[currColumn].mean()

        print(currColumn, ": ", currMean)
        appendDict(currColumn, currMean, train_x)
        #naDict[currColumn] = currMean
        #train_x.outliersDict[currColumn] = currMean

        train_x.data[currColumn] = train_x.data[currColumn].fillna(currMean)
        if test_x is not None:
            test_x.data[currColumn] = test_x.data[currColumn].fillna(currMean)

    # controlliamo nuovamente che train e test siano senza n/a
    getNaCount(train_x)
    print("train x na count : ", train_x.naCount)
    if test_x is not None:
        getNaCount(test_x)
        print("test x na count : ", test_x.naCount)



def appendDict(key, value, train_x):

        if key in train_x.outliersDict:
            # append the new number to the existing array at this slot
            train_x.outliersDict[key].append(value)
        else:
            # create a new array in this slot
            train_x.outliersDict[key] = [value]





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
        appendDict(currColumn, currMean, train_x)

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
    print("count NaN: ",count)
    dataset.naCount=count

def main():
    datasetPath = './training_set.csv'
    dataset = pd.read_csv(datasetPath)

    train_x= Dataset("train_x",None)
    test_x=Dataset("test_x",None)

    train_y= Dataset("train_y",None)
    test_y = Dataset("test_y", None)

    #separiamo le features x dal target y
    x = dataset.iloc[:, 0:20].values
    y = dataset.iloc[:, 20].values

    preProcessing(train_x, test_x, train_y, test_y, x, y)
    print(find_method, "---", substitute_method, "---", scaleType)



    crossValidation.cross4(train_x, test_x, train_y, test_y, find_method)

    print("dict === ", train_x.outliersDict)
    #suono quando finisce
    duration = 1000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()