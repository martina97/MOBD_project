import pickle

import sklearn
import sklearn.preprocessing as prep
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsRegressor


def main():
    datasetPath = './training_set.csv'
    dataset = pd.read_csv(datasetPath)
    # knnDetection()
    # naDetection()
    with open ('returned_clf.pkl','rb') as input:
        clf = pickle.load(input)

    best_parameters = clf.best_params_
    print("\n\nbest_parameters MLP : ", best_parameters)
    best_result = clf.best_score_
    print("best_result MLP: ", best_result)





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

