#Per addestrare un modello Naive Bayes utilizzando un dataset e un file JSON in Python, puoi seguire un approccio simile al seguente utilizzando la libreria scikit-learn per il modello Naive Bayes (Gaussian Naive Bayes in questo caso):
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import json

def train_naive_bayes(dataset_path, json_config_path):
    # Carica il dataset
    dataset = pd.read_csv(dataset_path)

    # Carica la configurazione dal file JSON
    with open(json_config_path, 'r') as json_file:
        config = json.load(json_file)

    # Seleziona le colonne desiderate
    selected_columns = config['selected_columns']
    target_column = config['target_column']

    # Dividi il dataset in feature e target
    X = dataset[selected_columns]
    y = dataset[target_column]

    # Dividi il dataset in set di addestramento e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Esempio di utilizzo della funzione
dataset_path = "path_al_tuo_dataset.csv"
json_config_path = "path_al_tuo_file_json.json"
trained_model = train_naive_bayes(dataset_path, json_config_path)
#Assicurati di personalizzare il percorso al tuo dataset CSV e al tuo file JSON di configurazione. Nel file JSON, puoi specificare le colonne desiderate e la colonna bersaglio per l'addestramento del modello.