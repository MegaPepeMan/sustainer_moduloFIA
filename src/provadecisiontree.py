#Per addestrare un modello Decision Tree utilizzando un dataset e un file JSON in Python, puoi seguire un approccio simile al seguente utilizzando la libreria scikit-learn per il modello Decision Tree:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import json

def train_decision_tree(dataset_path, json_config_path):
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

    # Inizializza il modello Decision Tree
    decision_tree_model = DecisionTreeClassifier()

    # Addestra il modello sul set di addestramento
    decision_tree_model.fit(X_train, y_train)

    # Fai previsioni sul set di test
    predictions = decision_tree_model.predict(X_test)

    # Valuta le prestazioni del modello
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")

    return decision_tree_model

# Esempio di utilizzo della funzione
dataset_path = "path_al_tuo_dataset.csv"
json_config_path = "path_al_tuo_file_json.json"
trained_model = train_decision_tree(dataset_path, json_config_path)

#Assicurati di personalizzare il percorso al tuo dataset CSV e al tuo file JSON di configurazione. Nel file JSON, puoi specificare le colonne desiderate e la colonna bersaglio per l'addestramento del modello.