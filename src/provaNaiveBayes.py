#Per addestrare un modello Naive Bayes utilizzando un dataset e un file JSON in Python, puoi seguire un approccio simile al seguente utilizzando la libreria scikit-learn per il modello Naive Bayes (Gaussian Naive Bayes in questo caso):

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from aif360.sklearn.preprocessing import Reweighing

def trainNaiveBayes(X_train, X_test, y_train, y_test, json_config_path, prot_attr: list[str]):

    # Dividi il dataset in feature e target
    # Vengono importate dai parametri

    # Inizializza il modello Naive Bayes (Gaussian)
    naive_bayes_model = GaussianNB()

    # Addestra il modello sul set di addestramento
    naive_bayes_model.fit(X_train, y_train)

    # Fai previsioni sul set di test
    predictions = naive_bayes_model.predict(X_test)

    # Valuta le prestazioni del modello
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")

    return naive_bayes_model