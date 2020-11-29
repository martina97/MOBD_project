# Thispath=Noneple Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#import learn as learn
#pipi
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
    
    # leggiamo i dati specificando le colonne opportune\n",
    dataset = pd.read_csv(datasetPath)

    print("Shape:", dataset.shape)
    print(dataset.tail())



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
