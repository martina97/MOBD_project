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



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def openFiles(datasetPath):

    # leggiamo i dati specificando le colonne opportune
    #TODO: dataframe = read_csv(url, header=None, na_values='?')
    dataset = pd.read_csv(datasetPath)

    print("Shape:", dataset.shape)
    print(dataset.tail())


    # Ora dividiamo il dataset in training set e test set secondo le proporzioni 80-20
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    # separiamo le caratteristiche dalla variabile che vogliamo prevedere 'CLASS'
    train_labels = train_dataset.pop('CLASS')
    test_labels = test_dataset.pop('CLASS')
    print("train_dataset shape:", train_dataset.shape, "train_labels shape:", train_labels.shape)
    print("test_dataset shape:", test_dataset.shape, "test_labels shape:", test_labels.shape)

    # calcoliamo il numero di valori mancanti su train e test (n/a)
    train_dataset=naDetection(train_dataset)
    test_dataset=naDetection(test_dataset)

    # controlliamo nuovamente che train e test siano senza n/a
    summary_train = get_na_count(train_dataset)
    print(summary_train)
    summary_test = get_na_count(test_dataset)
    print(summary_test)



    #ORA FACCIAMO BOX PLOT !!!!!!!!!!!
    #data2 = [18, 19, 19, 20, 20, 20, 20, 20, 21, 21, 21, 22, 23, 25, 28, 30, 31, 37]
    #data2 = [5, 39, 75, 79, 85, 90, 91, 93, 93, 98]


    dataColumn = np.array([])

    #inserisco istanze della colonna in una lista
    print("size colonna: ",train_dataset['F1'].size)

    for i in train_dataset['F1']:
        dataColumn = np.append(dataColumn,i)

    title = 'F1 before KNN'
    print("\n\ndata2: ",dataColumn,"\n\n")
    createBoxplot(dataColumn,title)



def naDetection(dataset):

    df = pd.DataFrame(dataset)
    imputer = KNNImputer(n_neighbors=2)
    imputed = imputer.fit_transform(df)
    df_imputed = pd.DataFrame(imputed)

    #print(" ------->", get_na_count(df_imputed))

    return changeColNames(df_imputed)


def changeColNames(df_imputed):
    #print("TOTALE PRIMA-   ", df_imputed)
    #print("COLONNA 0-   ", df_imputed[0])

    string = "F"
    for i in range(1, 21):
        currColumn = string + str(i)
        index = i - 1
        #print("index: ", index)
        # print(df_imputed.rename(columns={index: 'F1'}))
        df_imputed.rename(columns={index: currColumn}, inplace=True)

    print("TOTALE DOPO-   ", df_imputed)
    return df_imputed



def createBoxplot(dataset, title):
    #data2 = np.array([18, 19, 19, 20, 20, 20, 20, 20, 21, 21, 21, 22, 23, 25, 28, 30, 31, 37])
    #data2 = np.array([5, 39, 75, 79, 85, 90, 91, 93, 93, 98])

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

    #trovo gli outliers e li inserisco in una lista
    outliers = []
    count = 0
    for i in dataset:
        if i < l or i > r:
            count = count + 1
            outliers.append(i)
            print("-- outlier n ", count, ":  ", outliers[count - 1])

    ax.set_xlim(right=1.5)
    plt.show()

    return outliers



def knnDetection():

    #data2 = np.array([18, 19, 19, 20, 20, 20, 20, 20, 21, 21, 21, 22, 23, 25, 28, 30, 31, 37])
    data2 = np.array([5, 39, 75, 79, 85, 90, 91, 93, 93, 98])
    title = 'before KNN'

    outliers = createBoxplot(data2, title)

    # copio dataset in lista y e tolgo outliers
    y = data2.copy()
    #print("y = ", y)
    for i in outliers:
        #print ("i= ",i)
        #y.remove(i)
        y = y[y != i]
        #print("y = ", y)

    print("y = ", y)
    print("data2: ", data2)

    # ============ FINORA ABBIAMO BOXPLOT ==========
    # TODO: capire come sceglire K
    # TODO: cambiare anche N/A con funzione che implementa algoritmo KNN

    # ORA ABBIAMO TOLTO E SOSTITUITO OUTLIER : USIAMO KNN !!!!!

    lenX = len(data2)-len(outliers)
    rows = lenX
    col = 1
    X = [[0 for i in range(col)] for j in range(rows)]      #inizializzo X come lista 2D
    count_X_position = 0

    #metto dati nella lista 2D "X"

    #per creare lista 2D "X" per poterla usare in KNN in cui devono andarci tutti i valori di data2 tranne outliers
    #così poi a KNN gli do X senza outlier che gli passo separatamente, in modo da calcolare media dei k vicini e sostituirli
    for i in y:

        #print("count X = ", count_X_position,"     data2_elem = ",i)
        X[count_X_position][0]=i
        #print("X[count_X_position][0] = ", X[count_X_position][0])
        count_X_position = count_X_position + 1


    print("X = ", X)




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
    print("result: ", result[0][0],result[1][0])


    #sostuituiamo i risultati con gli outliers nel dataset originario
    for i in outliers:
        for j in result:
            data2[data2 == i] = (j)
            print("data2: ", data2)



    # ============= BOXPLOT X VEDERE CHE FUNZIONA ===========
    title = 'after KNN'
    if len(createBoxplot(data2,title)) == 0:
        print("KNN terminato, outliers sostituiti")
        return 0






def get_na_count(dataset):
    # per ogni elemento (i,j) del dataset, isna() restituisce
    # TRUE/FALSE se il valore corrispondente è mancante/presente
    boolean_mask = dataset.isna()
    # contiamo il numero di TRUE per ogni attributo sul dataset
    count = boolean_mask.sum(axis=0)
    #print("count NaN: ",count)
    return count




def main():

    datasetPath = './training_set.csv'
    #knnDetection()
    #naDetection()
    openFiles(datasetPath)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
