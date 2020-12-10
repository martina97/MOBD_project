# Thispath=Noneple Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


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


def openFiles(dataset):
    # leggiamo i dati specificando le colonne opportune
    # TODO: dataframe = read_csv(url, header=None, na_values='?')
    #dataset = pd.read_csv(datasetPath)

    print("Shape:", dataset.shape)
    print(dataset.tail())



    # separiamo le feature x dal target y
    x = dataset.iloc[:,0:20].values
    y = dataset.iloc[:,20].values


    # dividiamo i dati in training e test
    train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=0)
    print('Train:', train_x.shape, train_y.shape, "   col: ")
    print('Test:', test_x.shape, test_y.shape)

    print("\n\nPROVAAAAAAAA train_x: ", train_x)

    #train_x = trsining set SENZA colonna target -> (6400, 20)
    #test_x = test set SENZA colonna target -> (1600, 20)
    #train_y = training set della colonna target -> (6400,)
    #test_y = test set della colonna target -> (1600,)

    # TODO: DA CANCELLARE
    '''
    
    
     
    # Ora dividiamo il dataset in training set e test set secondo le proporzioni 80-20
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)


    # separiamo le caratteristiche dalla variabile che vogliamo prevedere 'CLASS'
    train_labels = train_dataset.pop('CLASS')   #stampa elementi della colonna CLASS estrapolata da train_dataset
    test_labels = test_dataset.pop('CLASS')
    print("train_dataset shape:", train_dataset.shape, "train_labels shape:", train_labels.shape)
    print("test_dataset shape:", test_dataset.shape, "test_labels shape:", test_labels.shape)

    print("train_labels :", train_labels)
    print("test_labels :", test_labels)
    '''


    # calcoliamo il numero di valori mancanti su train e test (n/a)
    train_dataset = naDetection(train_x)
    test_dataset = naDetection(test_x)

    #train_dataset.to_csv(r'D:\Universita\magistrale\MOBD\csv\TrainingSet iniziale.csv', header=True, sep=';')

    # controlliamo nuovamente che train e test siano senza n/a
    summary_train = get_na_count(train_dataset)
    print(summary_train)
    summary_test = get_na_count(test_dataset)
    print(summary_test)

    # ORA FACCIAMO BOX PLOT !!!!!!!!!!!
    # data2 = [18, 19, 19, 20, 20, 20, 20, 20, 21, 21, 21, 22, 23, 25, 28, 30, 31, 37]
    # data2 = [5, 39, 75, 79, 85, 90, 91, 93, 93, 98]

    # inserisco istanze della colonna in una lista
    print("size colonna: ", train_dataset['F1'].size)

    for j in train_dataset.columns:
        print("j = ", j)
        dataColumn = np.array([])

        for i in train_dataset[j]:
            dataColumn = np.append(dataColumn, i)

        title = j + '  before KNN'

        # print("\n\ndata2: ", dataColumn, "\n\n")
        #outliers = createBoxplot(dataColumn, title)
        # outliers = prova_box(train_dataset,j, "before KNN")

        outliers = prova_box2(dataColumn, "  before KNN", j)

        # knnDetection(dataColumn,outliers,j,train_dataset)
        knnDetection2(outliers, j, train_dataset)

        # knnDetection3(dataColumn, outliers, j, train_dataset)

    #train_dataset.to_csv(r'D:\Universita\magistrale\MOBD\csv\TrainingSet finale.csv', header=True, sep=';')

    '''  
    title = 'F1 before KNN'
    print("\n\ndata2: ",dataColumn,"\n\n")
    createBoxplot(dataColumn,title)
    '''
    #return train_dataset




def prova_box(train_dataset, colName, title):
    print("\n\nSONO IN PROVA BOX PORCO DUE")
    sns.boxplot(x=train_dataset[colName]).set_title(title)
    print("vaffanculo")
    count = 0
    threshold = 3
    mean = np.mean(train_dataset[colName])
    std = np.std(train_dataset[colName])
    outliers = []
    for i in train_dataset[colName]:
        # print("vaffanculo2")

        z = (i - mean) / std
        if z > threshold:
            print("vaffanculo3")

            count = count + 1
            outliers.append(i)
            print("-- outlier n ", count, ":  ", outliers[count - 1])
    return outliers


def prova_box2(dataColumn, title, colName):
    # print("\n\nSONO IN PROVA BOX PORCO DUE")

    sns.boxplot(x=dataColumn).set_title(colName + title)
    '''
    fig, ax = plt.subplots()
    ax.set_title(colName+title)
    ax.scatter(dataColumn,dataColumn)
    '''

    # print("vaffanculo")

    '''
    fig1, ax = plt.subplots()
    ax.set_title(colName+title)
    ax.boxplot(dataColumn)
    '''

    count = 0
    threshold = 3
    mean = np.mean(dataColumn)
    std = np.std(dataColumn)
    outliers = []
    for i in dataColumn:
        # print("vaffanculo2")

        z = (i - mean) / std
        if z > threshold:
            # print("vaffanculo3")

            count = count + 1
            outliers.append(i)
            print("-- outlier n ", count, ":  ", outliers[count - 1])

    plt.show()
    return outliers


def knnDetection2(outliers, colName, train_dataset):
    # data2 = np.array([18, 19, 19, 20, 20, 20, 20, 20, 21, 21, 21, 22, 23, 25, 28, 30, 31, 37])
    # data2 = np.array([5, 39, 75, 79, 85, 90, 91, 93, 93, 98])
    title = 'before KNN'

    # outliers = createBoxplot(data2, title)

    # copio dataset in lista y e tolgo outliers
    y = train_dataset[colName].copy()
    # print("y = ", y)
    for i in outliers:
        # print ("i= ",i)
        # y.remove(i)
        y = y[y != i]
        # print("y = ", y)

    # print("y = ", y)
    # print("data2: ", data2)

    # ============ FINORA ABBIAMO BOXPLOT ==========
    # TODO: capire come sceglire K
    # TODO: cambiare anche N/A con funzione che implementa algoritmo KNN

    # ORA ABBIAMO TOLTO E SOSTITUITO OUTLIER : USIAMO KNN !!!!!

    lenX = len(train_dataset[colName]) - len(outliers)
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

    # print("X = ", X)

    '''

      X = [[18],  [19], [19], [20], [20], [20], [20], [20], [21], [21], [21], [22], [23], [25], [28], [30], [31]]
    y = [18,19,19,20,20,20,20,20,21,21,21,22,23,25,28,30,31]

    '''

    # fit
    neigh = KNeighborsRegressor(n_neighbors=3)
    neigh.fit(X, y)

    # predict
    result = []
    for i in outliers:
        result.append(neigh.predict([[i]]))
    # print("result: ", result[0][0],result[1][0])
    print("result: ", result)

    # train_dataset[colName].to_csv(r'D:\Universita\magistrale\MOBD\csv\TrainingSetPrimaKNN' +colName+'.csv', header=True, sep = ';')

    # sostuituiamo i risultati con gli outliers nel dataset originario
    for i in outliers:
        for j in result:
            train_dataset[colName][train_dataset[colName] == i] = (j)
            # print("data2: ", data2)

    # print("data2: ", data2)

    # ============= BOXPLOT X VEDERE CHE FUNZIONA ===========
    ''' 
    title = colName + '  after KNN'
    if len(createBoxplot(data2, title)) == 0:
        print("KNN terminato, outliers sostituiti")
        return 0

    '''

    # train_dataset[colName].to_csv(r'D:\Universita\magistrale\MOBD\csv\TrainingSetDopoKNN' +colName+'.csv', header=True, sep = ';')

    # CALCOLO OUTLIERS DEL TRAINING SET FINALE DELLA SIGNOLA FEATURE
    outliers = prova_box2(train_dataset[colName], "  after KNN", colName)
    if len(outliers) == 0:
        print(colName, ": KNN terminato, outliers sostituiti\n\n")
        return 0

    '''
    if len(prova_box2(train_dataset[colName], "  after KNN", colName)) == 0:
        print(colName, ": KNN terminato, outliers sostituiti\n\n")
        return 0
    '''

def naMean(train_dataset,test_dataset):
    # calcoliamo il numero di valori mancanti su train e test
    summary_train = get_na_count(train_dataset)
    print("count NaN TRAINING: ", summary_train, "\n\n\n")
    summary_test = get_na_count(test_dataset)
    print("count NaN TESTING: ", summary_test, "\n\n\n")

    # print(train_dataset['F1'].mean())

    print("\n\nMEDIA PER OGNI ATTRIBUTO: ")

    string = "F"
    for i in range(1, 21):
        currColumn = string + str(i)
        currMean = train_dataset[currColumn].mean()

        print(currColumn, ": ", currMean)

        train_dataset[currColumn] = train_dataset[currColumn].fillna(currMean)
        test_dataset[currColumn] = test_dataset[currColumn].fillna(currMean)




def naDetection(dataset):
    df = pd.DataFrame(dataset)
    imputer = KNNImputer(n_neighbors=2)
    imputed = imputer.fit_transform(df)
    df_imputed = pd.DataFrame(imputed)

    # print(" ------->", get_na_count(df_imputed))

    return changeColNames(df_imputed)





def changeColNames(df_imputed):
    # print("TOTALE PRIMA-   ", df_imputed)
    # print("COLONNA 0-   ", df_imputed[0])

    string = "F"
    for i in range(1, 21):
        currColumn = string + str(i)
        index = i - 1
        # print("index: ", index)
        # print(df_imputed.rename(columns={index: 'F1'}))
        df_imputed.rename(columns={index: currColumn}, inplace=True)

    print("TOTALE DOPO-   ", df_imputed)
    return df_imputed


def createBoxplot(dataset, title):
    # data2 = np.array([18, 19, 19, 20, 20, 20, 20, 20, 21, 21, 21, 22, 23, 25, 28, 30, 31, 37])
    # data2 = np.array([5, 39, 75, 79, 85, 90, 91, 93, 93, 98])

    fig1, ax = plt.subplots()
    ax.set_title(title)
    ax.boxplot(dataset)

    median = np.median(dataset)
    q3 = np.percentile(dataset, 75)  # upper_quartile
    q1 = np.percentile(dataset, 25)  # lower_quartile
    iqr = q3 - q1

    print("mediana: ", median)
    print("q1: ", q1)
    print("q3: ", q3)
    print("iqr: ", iqr)

    l = q1 - 1.5 * iqr
    r = q3 + 1.5 * iqr
    print("l: ", l, "    r:", r)

    # trovo gli outliers e li inserisco in una lista

    outliers = []
    count = 0
    for i in dataset:
        if i < l or i > r:
            count = count + 1
            outliers.append(i)
            print("-- outlier n ", count, ":  ", outliers[count - 1])
    '''
    
    
    count = 0
    threshold = 10
    mean = np.mean(dataset)
    std = np.std(dataset)
    outliers = []
    for i in dataset:
        z = (i - mean) / std
        if z > threshold:
            count = count + 1
            outliers.append(i)
            print("-- outlier n ", count, ":  ", outliers[count - 1])

    # print('outlier in dataset is', outliers)
    '''

    ax.set_xlim(right=1.5)
    plt.show()

    return outliers


def knnDetection(data2, outliers, colName, train_dataset):
    # data2 = np.array([18, 19, 19, 20, 20, 20, 20, 20, 21, 21, 21, 22, 23, 25, 28, 30, 31, 37])
    # data2 = np.array([5, 39, 75, 79, 85, 90, 91, 93, 93, 98])
    title = 'before KNN'

    # outliers = createBoxplot(data2, title)

    # copio dataset in lista y e tolgo outliers
    y = data2.copy()
    # print("y = ", y)
    for i in outliers:
        # print ("i= ",i)
        # y.remove(i)
        y = y[y != i]
        # print("y = ", y)

    # print("y = ", y)
    # print("data2: ", data2)

    # ============ FINORA ABBIAMO BOXPLOT ==========
    # TODO: capire come sceglire K
    # TODO: cambiare anche N/A con funzione che implementa algoritmo KNN

    # ORA ABBIAMO TOLTO E SOSTITUITO OUTLIER : USIAMO KNN !!!!!

    lenX = len(data2) - len(outliers)
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

    # print("X = ", X)

    '''

      X = [[18],  [19], [19], [20], [20], [20], [20], [20], [21], [21], [21], [22], [23], [25], [28], [30], [31]]
    y = [18,19,19,20,20,20,20,20,21,21,21,22,23,25,28,30,31]

    '''

    # fit
    neigh = KNeighborsRegressor(n_neighbors=3)
    neigh.fit(X, y)

    # predict
    result = []
    for i in outliers:
        result.append(neigh.predict([[i]]))
    # print("result: ", result[0][0],result[1][0])
    print("result: ", result)

    # sostuituiamo i risultati con gli outliers nel dataset originario
    for i in outliers:
        for j in result:
            data2[data2 == i] = (j)
            # print("data2: ", data2)

    print("data2: ", data2)

    # ============= BOXPLOT X VEDERE CHE FUNZIONA ===========

    title = colName + '  after KNN'
    if len(createBoxplot(data2, title)) == 0:
        print("KNN terminato, outliers sostituiti")
        return 0

    '''
    if len(prova_box(train_dataset, colName)) == 0:
        print("KNN terminato, outliers sostituiti")
        return 0
    '''


def get_na_count(dataset):
    # per ogni elemento (i,j) del dataset, isna() restituisce
    # TRUE/FALSE se il valore corrispondente è mancante/presente
    boolean_mask = dataset.isna()
    # contiamo il numero di TRUE per ogni attributo sul dataset
    count = boolean_mask.sum(axis=0)
    # print("count NaN: ",count)
    return count


def main():
    datasetPath = './training_set.csv'
    dataset = pd.read_csv(datasetPath)
    # knnDetection()
    # naDetection()
    openFiles(dataset)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
