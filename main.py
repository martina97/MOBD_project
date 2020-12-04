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
from matplotlib.cbook import boxplot_stats
from sklearn.neighbors import NearestNeighbors

import seaborn as sns



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def openFiles(datasetPath):

    '''
    column_names2 = []
    string = "F"
    for i in range(1,21):
        column_names2.append(string + str(i))

    print("colonne: " , column_names2)
    '''
    
    # leggiamo i dati specificando le colonne opportune
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

    # calcoliamo il numero di valori mancanti su train e test
    summary_train = get_na_count(train_dataset)
    print("count NaN TRAINING: ",summary_train,"\n\n\n")
    summary_test = get_na_count(test_dataset)
    print("count NaN TESTING: ",summary_test,"\n\n\n")

    #print(train_dataset['F1'].mean())

    print("\n\nMEDIA PER OGNI ATTRIBUTO: ")
    # calcoliamo media di ogni colonna e la sostituiamo ai valori mancanti

    string = "F"
    for i in range(1, 21):
        currColumn = string + str(i)
        currMean = train_dataset[currColumn].mean()

        print(currColumn,": ", currMean)

        train_dataset[currColumn] = train_dataset[currColumn].fillna(currMean)
        test_dataset[currColumn] = test_dataset[currColumn].fillna(currMean)

    # controlliamo nuovamente train e test
    summary_train = get_na_count(train_dataset)
    print(summary_train)
    summary_test = get_na_count(test_dataset)
    print(summary_test)


    # Visualizziamo i dati tramite il pairplot seaborn le due classi di colore diverso usando l'attributo 'hue'


    # data2 = [18, 19, 19, 20, 20, 20, 20, 20, 21, 21, 21, 22, 23, 25, 28, 30, 31, 37]
    #data2 = [5, 39, 75, 79, 85, 90, 91, 93, 93, 98]
    data2 = []

    #inserisco istanze della colonna in una lista
    print("size colonna: ",train_dataset['F1'].size)
    for i in train_dataset['F1']:
        data2.append(i)


    df = pd.DataFrame(data2)
    print("df: ", df)
    #ax = sns.boxplot(data2=df)

    fig1, ax = plt.subplots()
    ax.set_title('Basic Plot')
    ax.boxplot(data2)



    median = np.median(data2)
    q3 = np.percentile(data2, 75)   #upper_quartile
    q1 = np.percentile(data2, 25)   #lower_quartile
    iqr = q3 - q1

    print("mediana: ", median)
    print("q1: ", q1)
    print("q3: ", q3)
    print("iqr: ", iqr)

    l= q1-1.5*iqr
    r=q3+1.5*iqr
    print("l: ", l,"    r:",r)
    count =0

    for i in data2:
        if i <l or i > r:
            count=count+1
            print ("-- outlier n ",count,":  ",i)

    ax.set_xlim(right=1.5)
    plt.show()



from sklearn.neighbors import KNeighborsRegressor

def prova_knn():

    # dataset (X=m^2, y=rental price)

    data2 = np.array([18, 19, 19, 20, 20, 20, 20, 20, 21, 21, 21, 22, 23, 25, 28, 30, 31, 37])

    df = pd.DataFrame(data2)
    #print("df: ", df)

    fig1, ax = plt.subplots()
    ax.set_title('before KNN ')
    ax.boxplot(data2)

    median = np.median(data2)
    q3 = np.percentile(data2, 75)  # upper_quartile
    q1 = np.percentile(data2, 25)  # lower_quartile
    iqr = q3 - q1

    print("mediana: ", median)
    print("q1: ", q1)
    print("q3: ", q3)
    print("iqr: ", iqr)

    l = q1 - 1.5 * iqr
    r = q3 + 1.5 * iqr
    print("l: ", l, "    r:", r)



    #trovo l'outlier
    out = 0
    count = 0
    for i in data2:
        if i < l or i > r:
            count = count + 1
            out=i
            print("-- outlier n ", count, ":  ", out)

    ax.set_xlim(right=1.5)
    plt.show()

    # ============ FINORA ABBIAMO BOXPLOT ==========
    # TODO: in funzione a parte che ritorna lista di outliers
    # TODO: capire come sceglire K
    # TODO: cambiare anche N/A con funzione che implementa algoritmo KNN
    # ORA ABBIAMO TOLTO E SOSTITUITO OUTLIER : USIAMO KNN !!!!!

    lenX = len(data2)-count
    rows = lenX
    col = 1

    X = [[0 for i in range(col)] for j in range(rows)]
    X[0][0] = 33333

    count_X_position = 0

    #metto dati nella lista 2D

    for i in data2:
        if i!=out:

            #print("count X = ", count_X_position,"     i = ",i)
            X[count_X_position][0]=i
            #print("X[count_X_position][0] = ", X[count_X_position][0])
            count_X_position = count_X_position + 1



    print("X = ", X)

    #copio dataset in y e tolgo outliers
    y = data2.copy()
    y = y[y!=out]
    #y.remove(out)
    print("y = ", y)
    print("data2: ", data2)


    '''
    
      X = [[18],  [19], [19], [20], [20], [20], [20], [20], [21], [21], [21], [22], [23], [25], [28], [30], [31]]
    y = [18,19,19,20,20,20,20,20,21,21,21,22,23,25,28,30,31]
    
    '''

    # fit
    neigh = KNeighborsRegressor(n_neighbors=3)
    neigh.fit(X, y)

    # predict
    result = neigh.predict([[37]])
    print("result: ", result[0])


    #il risultato va al posto di 37 in data2
    data2[data2 == out] = round(result[0])
    print("data2: ", data2)

    # ============= BOXPLOT X VEDERE CHE FUNZIONA ===========
    #TODO : da eliminare !
    df = pd.DataFrame(data2)
    # print("df: ", df)

    fig1, ax = plt.subplots()
    ax.set_title('after KNN ')
    ax.boxplot(data2)

    median = np.median(data2)
    q3 = np.percentile(data2, 75)  # upper_quartile
    q1 = np.percentile(data2, 25)  # lower_quartile
    iqr = q3 - q1

    print("mediana: ", median)
    print("q1: ", q1)
    print("q3: ", q3)
    print("iqr: ", iqr)

    l = q1 - 1.5 * iqr
    r = q3 + 1.5 * iqr
    print("l: ", l, "    r:", r)

    # trovo l'outlier
    out = 0
    count = 0
    for i in data2:
        if i < l or i > r:
            count = count + 1
            out = i
            print("-- outlier n ", count, ":  ", out)

    ax.set_xlim(right=1.5)
    plt.show()









def get_na_count(dataset):
    # per ogni elemento (i,j) del dataset, isna() restituisce
    # TRUE/FALSE se il valore corrispondente è mancante/presente
    boolean_mask = dataset.isna()
    # contiamo il numero di TRUE per ogni attributo sul dataset
    count = boolean_mask.sum(axis=0)
    #print("count NaN: ",count)
    return count




def main():
    print ("ciaomerda")
    print("tensorflow: ", tf.__version__)

    dataset_path = './auto-mpg.data'
    datasetPath = './training_set.csv'
    prova_knn()
    #openFiles(datasetPath)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
