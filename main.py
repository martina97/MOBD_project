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
    openFiles(datasetPath)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
