def chooseMethods():
    na_methods = ["MEAN", "KNN"]
    find_methods = ["IQR", "ZSCORE"]
    substitute_methods = ["KNN", "MEAN"]
    scale_types = ["STANDARD", "MINMAX", "MAX_ABS", "ROBUST"]
    resampling_methods = ["ClusterCentroids", "CondensedNearestNeighbour", "EditedNearestNeighbours",
                          "RepeatedEditedNearestNeighbours", "AllKNN", "NearMiss", "NeighbourhoodCleaningRule",
                          "RandomUnderSampler", "TomekLinks", "BorderlineSMOTE", "KMeansSMOTE",
                          "RandomUnderSampler", "SMOTE"]
    classifiers = ["MLP", "KNeighbors", "SVC", "DecisionTree", "RandomForest", "QuadraticDiscriminantAnalysis"]

    print("\n\nInserire i metodi con cui si vuole effettuare la classificazione:\n"
          "- na_method: metodo con cui si desiderano sostituire i valori mancanti nel dataset\n"
          "- find_method: metodo con cui si desiderano individuare gli outliers\n"
          "- substitute_method: metodo con cui si desiderano sostituire gli outliers\n"
          "- scale_type: tipo di scaler che si desidera\n"
          "- resampling_method: metodo con cui si vuole effettuare il resampling\n"
          "- classifier: classificatore che si vuole utilizzare\n"
          "\n\n")

    print("Scegliere e inserire una tra le seguenti stringhe:\n MEAN / KNN\nna_method: ")
    na_method = input()
    while na_method not in na_methods:
        print("ATTENZIONE: Inserire una tra le stringhe a disposizione, facendo attenzione a maiuscole e minuscole!")
        print("na_method:")
        na_method = input()

    print("\nScegliere e inserire una tra le seguenti stringhe:\n IQR / ZSCORE\nfind_method: ")
    find_method = input()
    while find_method not in find_methods:
        print("ATTENZIONE: Inserire una tra le stringhe a disposizione, facendo attenzione a maiuscole e minuscole!")
        print("find_method:")
        find_method = input()

    print("\nScegliere e inserire una tra le seguenti stringhe:\n KNN / MEAN\nsubstitute_method: ")
    substitute_method = input()
    while substitute_method not in substitute_methods:
        print("ATTENZIONE: Inserire una tra le stringhe a disposizione, facendo attenzione a maiuscole e minuscole!")
        print("substitute_method:")
        substitute_method = input()

    print("\nScegliere e inserire una tra le seguenti stringhe:\n STANDARD / MINMAX / MAX_ABS / ROBUST\nscale_type: ")
    scale_type = input()
    while scale_type not in scale_types:
        print("ATTENZIONE: Inserire una tra le stringhe a disposizione, facendo attenzione a maiuscole e minuscole!")
        print("scale_type:")
        scale_type = input()

    print("\nScegliere e inserire una tra le seguenti stringhe:\n ClusterCentroids / CondensedNearestNeighbour / EditedNearestNeighbours /\n"
                          "RepeatedEditedNearestNeighbours / AllKNN / NearMiss / NeighbourhoodCleaningRule /\n" 
                          "RandomUnderSampler / TomekLinks / BorderlineSMOTE / KMeansSMOTE / " 
                          "RandomUnderSampler /  SMOTE\nresampling_method: ")
    resampling_method = input()
    while resampling_method not in resampling_methods:
        print("ATTENZIONE: Inserire una tra le stringhe a disposizione, facendo attenzione a maiuscole e minuscole!")
        print("resampling_method:")
        resampling_method = input()

    print(
        "\nScegliere e inserire una tra le seguenti stringhe:\n MLP / KNeighbors / SVC / DecisionTree / RandomForest / QuadraticDiscriminantAnalysis\nclassifier: ")
    classifier = input()
    while classifier not in classifiers:
        print("ATTENZIONE: Inserire una tra le stringhe a disposizione, facendo attenzione a maiuscole e minuscole!")
        print("classifier:")
        classifier = input()

    return na_method, find_method, substitute_method, scale_type, resampling_method, classifier


def main():
    print("chooseInputs.py")


if __name__ == '__main__':
    main()
