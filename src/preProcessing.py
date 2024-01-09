import pandas as pd
import numpy as np
import re

# pd.set_option('display.max_columns', None) # Mostra tutte le colonne

def letturaDataset(path):
    df=pd.read_csv(path)
    return df
def dataCleaning(dataset):

    # ----------1° parte----------
    # Tramite questa print abbiamo una overview sul complessivo di attirbuti mancanti del dataset
    null_percentage=dataset.isnull().mean()
    print(dataset.isnull().mean())

    # Trova le colonne con più di 1/3 di valori nulli
    print('Le vecchie colonne sono: ', dataset.columns.to_list())
    columns_to_drop = null_percentage[null_percentage > 1 / 3].index

    # Elimina le colonne che superano i valori nulli per oltre il 33%
    print('Le colonne da eliminare sono: ', columns_to_drop)

    nuovoDataframe = dataset.drop(columns=columns_to_drop)

    # Stampo il dataset ripulito
    print('Le nuove colonne sono: ',nuovoDataframe.columns.to_list())

    # ----------2° parte----------
    # Andiamo a gestire i valori che sono nulli e che vogliamo conservare
    # Identifica le colonne di tipo numerico
    attributiNumerici = nuovoDataframe.select_dtypes(include=[np.number]).columns
    print('Gli attributi numerici sono: ',attributiNumerici.tolist())

    # Imputa i valori nulli solo per le colonne numeriche utilizzando la media
    for att in attributiNumerici:
        nuovoDataframe[att].fillna(nuovoDataframe[att].mean(), inplace=True)

    print('Questa è la nuova distribuzione dei valori null: ',nuovoDataframe.isnull().mean())
    print(nuovoDataframe)

    # ----------3° parte----------
    # Gestione della normalizzazione dei dati:
    # Considerando le 2 pipeline da implementare, ovvero DecsionTree e NaiveBayes,
    # teniamo conto che non è utile la normalizzazione dei dataset data la natura
    # di questi 2 algoritmi

    # Uniformiamo le date
    # Cerca le colonne che potrebbero contenere date (pattern basato su una data ipotetica)
    date_columns = []
    date_pattern = re.compile(r'\b\d{4}-\d{2}-\d{2}\b')  # Pattern per data in formato YYYY-MM-DD

    for column in nuovoDataframe.columns:
        if any(nuovoDataframe[column].astype(str).str.match(date_pattern)):
            date_columns.append(column)

    for attData in date_columns:
        nuovoDataframe[attData] = pd.to_datetime(nuovoDataframe[attData])

    # Stampare le colonne contenenti date
    print("Colonne contenenti date:", date_columns)

    # ----------4° parte----------
    # Mostra i valori unici per ogni colonna

    # Stabiliamo una soglia per il numero di categorie degli attributi categorici
    soglia = 10
    attributiCategorici = []

    # Individuiamo quali sono gli attributi categorici
    for col in nuovoDataframe.columns:
        num_unique_values = nuovoDataframe[col].nunique()
        if num_unique_values <= soglia:
            attributiCategorici.append(col)

    print('I seguenti attributi sono stati evidenziati come categorici: ',attributiCategorici)

    # Usiamo One Hot Encoding per dividere gli attributi categorici
    for col in attributiCategorici:
        nuovoDataframe = pd.get_dummies(nuovoDataframe, columns=[col])

    print(nuovoDataframe)
