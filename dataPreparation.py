import pickle
import sklearn
import sklearn.preprocessing as prep
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler, BorderlineSMOTE, KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss, CondensedNearestNeighbour, TomekLinks, \
    ClusterCentroids, EditedNearestNeighbours, NeighbourhoodCleaningRule, \
    RepeatedEditedNearestNeighbours, AllKNN
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, LabelEncoder
import chooseInputs
import crossValidation
import winsound

import plotGraphics


class Dataset:
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.naCount = None
        self.outliers = None  # lista outliers di una colonna
        self.dataColumn = None  # elementi di una colonna
        self.result = None
        self.outliersDict = {}  # dict che contiene i valori necessari per la fase di Outliers Detection


def preProcessing(train_x, test_x, train_y, test_y, x, y, na_method, find_method, substitute_method, scale_type,
                  resampling_method):
    # Dividiamo il dataset in training set e test set secondo le proporzioni 80:20
    train_x.data, test_x.data, train_y.data, test_y.data = sklearn.model_selection.train_test_split(x, y, test_size=0.2,
                                                                                                    random_state=42)
    print('Train:', train_x.data.shape, train_y.data.shape)
    print('Test:', test_x.data.shape, test_y.data.shape)

    # Trasformiamo i vari dataset da numpy.ndarray a un DataFrame
    train_x.data = pd.DataFrame(train_x.data)
    test_x.data = pd.DataFrame(test_x.data)
    train_y.data = pd.DataFrame(train_y.data, columns=['CLASS'])
    test_y.data = pd.DataFrame(test_y.data, columns=['CLASS'])

    # Aggiungiamo i nomi alle colonne (F1-F20) in tutti e 4 i DataFrames
    changeColNames(train_x.data)
    changeColNames(test_x.data)

    # Sostituiamo i valori mancanti (NaN) del train e del test con opportuni valori (il metodo da utilizzare è scelto
    # dal parametro 'na_method' )
    naDetection(train_x, test_x, na_method)

    # outliers detection
    outlierDetection(train_x, test_x, find_method, substitute_method)

    # normalizziamo i dati
    scale(train_x, test_x, train_y, test_y, scale_type)

    # applichiamo PCA
    principalComponentAnalysis(train_x, test_x)

    # applichiamo resampling
    Resampling(train_x, train_y, resampling_method)


def naDetection(train_x, test_x, na_method):
    """
    Funzione che sostituisce i valori mancanti all'interno del dataset in due
    diversi modi, determinati dal parametro na_method.
    :param train_x: training set x
    :param test_x: test set x
    :param na_method: metodo con cui si vogliono sostituire i valori mancanti nei dataset
    """
    if na_method == "MEAN":
        naMean(train_x, test_x)  # sostituisce i NaN con la media
    if na_method == "KNN":
        naKNN(train_x, test_x)  # sostituisce i NaN con KNNImputer()


def principalComponentAnalysis(train_x, test_x):
    """
    Principal Component Analysis
    :param train_x: training set x
    :param test_x: test set x
    :return: None
    """

    pca = PCA()
    train_x.data = pca.fit_transform(train_x.data)
    save_object(pca, "pca1.pkl")

    if test_x is not None:
        test_x.data = pca.transform(test_x.data)

    # plotGraphics.plotVariance(pca)      #scommentare se si desidera vedere il grafico

    exp_var_cumsum = pd.Series(np.round(pca.explained_variance_ratio_.cumsum(), 4) * 100)
    for index, var in enumerate(exp_var_cumsum):
        print('if n_components= %d,   variance=%f' % (index, np.round(var, 3)))

    pca = PCA(n_components=15)
    train_x.data = pca.fit_transform(train_x.data)
    save_object(pca, "pca2.pkl")

    if test_x is not None:
        test_x.data = pca.transform(test_x.data)


def scale(train_x, test_x, train_y, test_y, scale_type):
    """
    Normalizzazione delle feature che utilizzano scale ed intervalli diversi.
    :param train_x: training set x
    :param test_x: test set x
    :param train_y: training set y
    :param test_y: test set y
    :param scale_type: stringa che specifica il tipo di scaler da utilizzare
    """

    matrix(train_x, test_x, train_y,
           test_y)  # convertiamo i DataFrame per l'input e per l'output in vettori 2D (matrici)

    train_xBefore = np.copy(train_x.data)  # utile solo per i grafici

    if scale_type == "STANDARD":
        standardScaler(train_x, test_x)
    if scale_type == "MINMAX":
        minMaxScaler(train_x, test_x)
    if scale_type == "MAX_ABS":
        maxAbsScaler(train_x, test_x)
    if scale_type == "ROBUST":
        robustScaler(train_x, test_x)

    # scommentare le prossime 2 righe se si vogliono visualizzare i grafici
    # plotGraphics.plotHistAfterScaler(train_xBefore, train_x.data)
    # plotGraphics.plotDistributionScaler(train_xBefore, train_x.data)


def Resampling(train_x, train_y, resampling_method):
    train_y.data = LabelEncoder().fit_transform(train_y.data)
    # summarize distribution

    # scommentare la riga di seguito se si vuole visualizzare il grafico a torta della distribuzione delle classi prima di resampling
    #plotGraphics.piePlot(train_y, "Before Resampling")

    # ---- UNDER-SAMPLING ------ #
    if resampling_method == "ClusterCentroids":
        resample = ClusterCentroids(voting='hard', random_state=42)

    if resampling_method == "CondensedNearestNeighbour":
        resample = CondensedNearestNeighbour(n_neighbors=7, random_state=42)

    if resampling_method == "EditedNearestNeighbours":
        resample = EditedNearestNeighbours(n_neighbors=7, kind_sel='mode', n_jobs=-1)

    if resampling_method == "RepeatedEditedNearestNeighbours":
        resample = RepeatedEditedNearestNeighbours(n_neighbors=7, kind_sel='mode', n_jobs=-1)

    if resampling_method == "AllKNN":
        resample = AllKNN(n_neighbors=7, kind_sel='mode', allow_minority=True, n_jobs=-1)

    if resampling_method == "NearMiss":
        resample = NearMiss(n_neighbors=7, n_jobs=-1)

    if resampling_method == "NeighbourhoodCleaningRule":
        resample = NeighbourhoodCleaningRule(n_neighbors=7, kind_sel='all')

    if resampling_method == "RandomUnderSampler":
        resample = RandomUnderSampler(random_state=42)

    if resampling_method == "TomekLinks":
        resample = TomekLinks(n_jobs=-1)

    # ---- OVER-SAMPLING ------ #
    if resampling_method == "BorderlineSMOTE":
        resample = BorderlineSMOTE(random_state=42, n_jobs=-1)

    if resampling_method == "KMeansSMOTE":
        resample = KMeansSMOTE(random_state=42)

    if resampling_method == "RandomUnderSampler":
        resample = RandomOverSampler(random_state=42)

    if resampling_method == "SMOTE":
        resample = SMOTE(random_state=42, n_jobs=-1)

    # transform the dataset
    train_x.data, train_y.data = resample.fit_resample(train_x.data, train_y.data)

    # scommentare la riga di seguito se si vuole visualizzare il grafico a torta della distribuzione delle classi dopo resampling
    #plotGraphics.piePlot(train_y, "After Resampling with ALLKNN")


def matrix(train_x, test_x, train_y, test_y):
    """
    Conversione dei DataFrame per l'input e per l'output in vettori 2D (matrici).
    :param train_x: training set x
    :param test_x: test set x
    :param train_y: training set y
    :param test_y: test set y
    :return None
    """

    train_x.data = np.float64(train_x.data)
    train_y.data = np.float64(train_y.data)
    train_y.data = train_y.data.reshape((len(train_y.data), 1))

    if test_x is not None or test_y is not None:
        test_x.data = np.float64(test_x.data)
        test_y.data = np.float64(test_y.data)
        test_y.data = test_y.data.reshape((len(test_y.data), 1))


def standardScaler(train_x, test_x):
    scaler = prep.StandardScaler()
    scaler.fit(train_x.data)
    train_x.data = scaler.transform(train_x.data)
    save_object(scaler, 'scaler.pkl')

    if test_x is not None:
        test_x.data = scaler.transform(test_x.data)

    print(pd.DataFrame(train_x.data).describe())
    return train_x.data


def minMaxScaler(train_x, test_x):
    # feature_range=(0, 2)
    scaler_x = prep.MinMaxScaler(feature_range=(-1, 1))

    scaler_x.fit(train_x.data)

    train_x.data = scaler_x.transform(train_x.data)

    if test_x is not None:
        test_x.data = scaler_x.transform(test_x.data)

    print(pd.DataFrame(train_x.data).describe())
    return train_x.data


def maxAbsScaler(train_x, test_x):
    scaler = MaxAbsScaler()
    scaler.fit(train_x.data)
    train_x.data = scaler.transform(train_x.data)

    if test_x is not None:
        test_x.data = scaler.transform(test_x.data)

    print(pd.DataFrame(train_x.data).describe())
    return train_x.data


def robustScaler(train_x, test_x):
    scaler = RobustScaler(quantile_range=(25, 75), with_centering=False)
    scaler.fit(train_x.data)
    train_x.data = scaler.transform(train_x.data)

    if test_x is not None:
        test_x.data = scaler.transform(test_x.data)
    return train_x.data


def outlierMean(train_x, test_x, colName):
    """
   Sostuzione degli outliers con la media per ogni colonna.
   :param train_x: training set
   :param test_x: test set
   :param colName: colonna interessata
   :return: None
   """

    # copio dataset in lista y e tolgo outliers
    y = train_x.data[colName].copy()
    for i in train_x.outliers:
        y = y[y != i]

    mean = y.mean()
    train_x.result = mean
    appendDict(colName, mean, train_x)

    if test_x is not None:
        test_x.result = mean


def outlierDetection(train_x, test_x, find_method, substitute_method):
    """
    Effettua il rilevamento e la sostituzione degli outliers presenti nel training set e nel test set,
    scegliendo tra i diversi metodi indicati da 'find_method' e 'substitute_method'.
    :param train_x: training set
    :param test_x: test set
    :param find_method: stringa che rappresenta il metodo con cui individuare gli outliers nel dataset
    :param substitute_method: stringa che rappresenta il metodo con cui sostrituire gli outliers nel dataset
    :return: None
    """

    # plotGraphics.printBoxplot(train_x, 'Before Outliers Detection')       #scommentare se si vuole visualizzare il boxlot

    for colName in train_x.data.columns:
        train_xBefore = np.copy(train_x.data[
                                    colName])  # utile se si vuole visualizzare la distribuzione della feature prima e dopo KNNDetection

        # metodi diversi per individuare gli outliers per il training set e per il test set
        if find_method == "IQR":
            outIQR(train_x, test_x, colName)

        if find_method == "ZSCORE":
            outZSCORE(train_x, test_x, colName)

        # metodi diversi per sostituire gli outliers per il training set e per il test set
        if substitute_method == "KNN":
            knnDetectionTRAIN(train_x, test_x, colName)

        else:
            outlierMean(train_x, test_x, colName)

        # una volta individuati i valori con cui gli outliers devono essere sostituiti, si procede alla sostituzione
        # sia per il training set sia per il test set
        substituteOutliers(train_x, colName, substitute_method)

        substituteOutliers(test_x, colName, substitute_method)

        # controllo degli outliers dopo averli sostituiti
        checkOutliersAfterReplacement(train_x, test_x, colName, find_method)

        # Scommentare la seguente riga se si vuole visualizzare l'istogramma della feature 'colName' prima e dopo KnnDetection
        # printHist(train_xBefore, train_x, colName)

    # plotGraphics.printBoxplot(train_x, 'After Outliers Detection') #scommentare se si vuole visualizzare il boxlot


def outIQR(train_x, test_x, colName):
    """
    Funzione che individua gli outliers della feature 'colName' con il metodo IQR.
    :param train_x: training set
    :param test_x: test set
    :param colName: colonna considerata
    :return: la lista degli outliers del training set e del test set nella colonna 'colName'
    """

    # --------  TRAINING SET    ------- #

    train_x.dataColumn = np.array([])

    for colElement in train_x.data[colName]:
        train_x.dataColumn = np.append(train_x.dataColumn, colElement)

    q3 = np.percentile(train_x.dataColumn, 75)  # upper_quartile
    q1 = np.percentile(train_x.dataColumn, 25)  # lower_quartile
    iqr = q3 - q1

    l = q1 - 1.5 * iqr
    r = q3 + 1.5 * iqr

    # trovo gli outliers e li inserisco in train_x.outliers
    train_x.outliers = []
    count = 0
    for i in train_x.dataColumn:
        if i < l or i > r:
            count = count + 1
            train_x.outliers.append(i)
            # print("-- outlier n ", count, ":  ", train_x.outliers[count - 1])

    # --------  TEST SET    ------- #
    test_x.dataColumn = np.array([])

    for colElement in train_x.data[colName]:
        test_x.dataColumn = np.append(test_x.dataColumn, colElement)

    # trovo gli outliers e li inserisco in test_x.outliers
    test_x.outliers = []
    count = 0
    for j in test_x.dataColumn:
        if j < l or j > r:
            count = count + 1
            test_x.outliers.append(j)
            # print("-- outlier n ", count, ":  ", test_x.outliers[count - 1])

    return train_x.outliers, test_x.outliers


def createDataColumn(dataset, colName):
    dataset.dataColumn = np.array([])

    for colElement in dataset.data[colName]:
        dataset.dataColumn = np.append(dataset.dataColumn, colElement)


def outZSCORE(train_x, test_x, colName):
    """
    Funzione che individua gli outliers della feature 'colName' con il metodo ZSCORE.
    :param train_x: training set
    :param test_x: test set
    :param colName: colonna considerata
    :return: la lista degli outliers del training set e del test set nella colonna colName
    """

    # --------  TRAINING SET    -------- #

    train_x.dataColumn = np.array([])

    for colElement in train_x.data[colName]:
        train_x.dataColumn = np.append(train_x.dataColumn, colElement)

    count = 0
    threshold = 3
    mean = np.mean(train_x.dataColumn)
    std = np.std(train_x.dataColumn)
    appendDict(colName, mean, train_x)
    appendDict(colName, std, train_x)

    # trovo gli outliers e li inserisco in train_x.outliers
    train_x.outliers = []

    for i in train_x.dataColumn:
        z = (i - mean) / std

        if z > threshold:
            count = count + 1
            train_x.outliers.append(i)
            # print("-- outlier n ", count, ":  ", train_x.outliers[count - 1])

    # --------  TEST SET    -------- #

    # trovo gli outliers e li inserisco in test_x.outliers
    outliers_test = []

    if test_x is not None:
        test_x.dataColumn = np.array([])

        for colElement in test_x.data[colName]:
            test_x.dataColumn = np.append(test_x.dataColumn, colElement)

        count = 0
        test_x.outliers = []
        for j in test_x.dataColumn:
            z = (j - mean) / std

            if z > threshold:
                count = count + 1
                test_x.outliers.append(j)
                # print("-- outlier n ", count, ":  ", test_x.outliers[count - 1])

        outliers_test = test_x.outliers

    return train_x.outliers, outliers_test


def knnDetectionTRAIN(train_x, test_x, colName):
    """
    Data la lista di outliers di una colonna, calcola il valore con cui essi devono essere sostituiti,
    e lo inserisce in train_x.result.
    :param train_x: training set
    :param test_x: test set
    :param colName: colonna considerata
    :return: None
    """

    # Copio il contenuto della feature 'colName', senza gli outliers, in una lista y
    y = train_x.data[colName].copy()
    for i in train_x.outliers:
        y = y[y != i]

    lenX = len(train_x.data[colName]) - len(train_x.outliers)

    rows = lenX
    col = 1
    X = [[0 for i in range(col)] for j in range(rows)]  # inizializzo X come lista 2D
    count_X_position = 0

    # metto dati nella lista 2D "X", che conterrà i valori della feature 'colName' senza gli outliers, come y
    for k in y:
        X[count_X_position][0] = k
        count_X_position = count_X_position + 1

    # --------     TRAINING SET  -----------

    # fit
    neigh = KNeighborsRegressor(n_neighbors=3, n_jobs=-1)

    neigh.fit(X, y)

    # predict
    result = []
    for i in train_x.outliers:
        result.append(neigh.predict([[i]]))

    # Nella lista result ci sono alcuni duplicati, quindi li eliminiamo
    result = np.unique(result, axis=0)
    # print("result senza duplicati: ", result)

    appendDict(colName, result[0][0], train_x)

    if len(result) > 2:
        print("Lenght result >2")
        return -1

    train_x.result = result
    if test_x is not None:
        test_x.result = result


def substituteOutliers(dataset, colName, substitute_method):
    """
    Sostituisce, in base al metodo dichiarato con la stringa 'substitute_method', gli outliers presenti nella colonna
    colName.
    :param dataset: dataset interessato
    :param colName: colonna interessata
    :param substitute_method: stringa che rappresenta il metodo con cui sostituire gli outliers nel dataset
    :return: None
    """
    if substitute_method == "KNN":
        if len(dataset.result) == 1:
            for i in dataset.outliers:
                dataset.data[colName][dataset.data[colName] == i] = (dataset.result[0][0])
        if len(dataset.result) > 1:
            for i in dataset.outliers:
                res = checkClosestOutlier(i, dataset.result)
                dataset.data[colName][dataset.data[colName] == i] = (res)

    if substitute_method == "MEAN":
        for i in dataset.outliers:
            dataset.data[colName][dataset.data[colName] == i] = dataset.result


def checkClosestOutlier(outlier, resultList):
    """
    Se la lista che contiene i valori con cui gli outliers devono essere sostituiti ha una lunghezza pari a 2
    (nel caso di IQR), per ogni outlier si trova il valore più vicino ad esso presente in resultList.
    Si calcola quindi la distanza dell'outlier dai due valori contenuti in resulList e si
    prende la distanza minore (valore assoluto)
    :param outlier: il singolo outlier
    :param resultList: lista che contiene i valori con cui gli outliers vengono sostituiti
    :return: il valore in resultList più vicino a outlier
    """
    
    # diff1 e diff2 sono le distanze in valore assoluto dall'outlier a entrambi i valori contenuti in resultList
    diff1 = abs(outlier - resultList[0][0])
    diff2 = abs(outlier - resultList[1][0])
    # print("diff1 : ", diff1, "  diff2: ",diff2)

    if diff2 < diff1:
        return diff2
    else:
        return diff1


def checkOutliersAfterReplacement(train_x, test_x, colName, find_method):
    """
   Controlla gli outliers appartenenti alla feature 'colName' dopo averli sostituiti
   :param train_x: training set
   :param test_x: test set
   :param colName: colonna (feature) interessata
   :param find_method: metodo per individuare gli outliers
   :return: None
   """
    if find_method == "IQR":
        title = colName + "after KNN"
        returnIQR = outIQR(train_x, test_x, colName)
        outliers_train = returnIQR[0]
        outliers_test = returnIQR[1]

    if find_method == "ZSCORE":
        returnZ = outZSCORE(train_x, test_x, colName)
        outliers_train = returnZ[0]
        outliers_test = returnZ[1]

    if len(outliers_train) == 0:
        print(colName, ": Tutti gli outliers nel training set sono stati sostituiti\n\n")
    if len(outliers_test) == 0:
        print(colName, ": Tutti gli outliers nel test set sono stati sostituiti\n\n")


def changeColNames(dataset):
    """
    Rinomina le colonne del dataset.
    :param dataset: dataset
    :return: None
    """

    string = "F"
    for i in range(1, 21):
        currColumn = string + str(i)
        index = i - 1
        dataset.rename(columns={index: currColumn}, inplace=True)


def appendDict(key, value, train_x):
    """
    Inserisce nel dizionario i valori associati alla chiave 'key'.
    :param key: chiave
    :param value: valore da inserire
    :param train_x:  training set
    :return: None
    """
    if key in train_x.outliersDict:
        # Se la chiave è già presente del dict, aggiunge il nuovo valore associato alla chiave (senza sostituire quelli esistenti)
        train_x.outliersDict[key].append(value)
    else:
        # Se la chiave non è presente nel dict, si inserisce la chiave e il valore associato
        train_x.outliersDict[key] = [value]


def naMean(train_x, test_x):
    """
    Sostuisce i NaN con la media per ogni colonna.
    :param train_x: training set
    :param test_x: test set
    :return: None
    """
    getNaCount(train_x)  # calcola i NaN nel training set
    getNaCount(test_x)  # calcola i NaN nel test set

    string = "F"
    for i in range(1, 21):
        currColumn = string + str(i)
        currMean = train_x.data[currColumn].mean()

        train_x.data[currColumn] = train_x.data[currColumn].fillna(currMean)
        test_x.data[currColumn] = test_x.data[currColumn].fillna(currMean)

    # controlliamo nuovamente che train e test siano senza n/a
    getNaCount(train_x)
    getNaCount(test_x)


def naKNN(train_x, test_x):
    """
    Sostituisce i valori mancanti nel training set e nel test set con KNNImputer().
    :param train_x: training set
    :param test_x: test set
    :return: None
    """
    getNaCount(train_x)  # calcola il numero di NaN per il training set
    imputer = KNNImputer(n_neighbors=3)

    imputed_train = imputer.fit_transform(train_x.data)
    train_x.data = pd.DataFrame(imputed_train, columns=train_x.data.columns)
    save_object(imputer,
                'imputer.pkl')  # salva imputer nel file 'imputer.pkl' (serve successivamente per il test finale)

    if test_x is not None:
        imputed_test = imputer.transform(test_x.data)
        test_x.data = pd.DataFrame(imputed_test, columns=test_x.data.columns)


def getNaCount(dataset):
    """
    Conta il numero di valori mancanti all'interno del dataset.
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


def save_object(obj, filename):
    """
    Salva l'oggetto obj in un file. Utile per la parte di valutazione del progetto.
    :param obj: oggetto da salvare
    :param filename: nome del file desiderato
    :return: None
    """
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def sound():
    duration = 400  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)


def main():
    datasetPath = './training_set.csv'
    dataset = pd.read_csv(datasetPath)

    train_x = Dataset("train_x", None)
    test_x = Dataset("test_x", None)

    train_y = Dataset("train_y", None)
    test_y = Dataset("test_y", None)

    # separiamo le features x dal target y
    x = dataset.iloc[:, 0:20].values
    y = dataset.iloc[:, 20].values

    # scelgo i metodi che voglio utilizzare per la parte di Data Preparation
    na_method, find_method, substitute_method, scale_type, resampling_method, classifier = chooseInputs.chooseMethods()

    preProcessing(train_x, test_x, train_y, test_y, x, y, na_method, find_method, substitute_method, scale_type,
                  resampling_method)
    print(find_method, "---", substitute_method, "---", scale_type, "---", resampling_method, "---", classifier)

    # cross validation e scelta degli iperparametri migliori
    crossValidation.cross(train_x, test_x, train_y, test_y, classifier)

    # avviso sonoro quando termina l'esecuzione
    sound()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
