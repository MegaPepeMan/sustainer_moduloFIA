import pandas as pd
import numpy as np
import re


# pd.set_option('display.max_columns', None) # Mostra tutte le colonne

def letturaDataset(path):
    df = pd.read_csv(path)
    return df


def dataCleaning(dataset):

    print(dataset)

    # ----------1° parte----------
    # Tramite questa print abbiamo una overview sul complessivo di attirbuti mancanti del dataset
    null_percentage = dataset.isnull().mean()
    print(dataset.isnull().mean())

    # Trova le colonne con più di 1/3 di valori nulli
    print('Le vecchie colonne sono: ', dataset.columns.to_list())
    columns_to_drop = null_percentage[null_percentage > 1 / 3].index

    # Elimina le colonne che superano i valori nulli per oltre il 33%
    print('Le colonne da eliminare sono: ', columns_to_drop)

    nuovoDataframe = dataset.drop(columns=columns_to_drop)

    # Stampo il dataset ripulito
    print('Le nuove colonne sono: ', nuovoDataframe.columns.to_list())

    # ----------2° parte----------
    # Eliminiamo i record che hanno troppi valori nulli

    # Abbiamo stabilito, data la mancanza di un valore universalmente accettato per la soglia dei valori nulli,
    # che un buon compromesso sia considerare record con almeno il 15% di valori nulli da eliminare
    soglia_valori_nulli = int(0.15 * len(nuovoDataframe.columns))  # 15% dei valori nulli rispetto al numero totale di colonne

    # Conta i valori nulli in ciascuna riga
    print('La soglia dei valori nulli è: ', soglia_valori_nulli)
    conteggio_valori_nulli = nuovoDataframe.isnull().sum(axis=1)
    print('Conteggio dei valori nulli: ',conteggio_valori_nulli)
    indiciDaRimuovere = conteggio_valori_nulli[conteggio_valori_nulli >= soglia_valori_nulli].index
    print('Gli indici da rimuovere sono: ', indiciDaRimuovere)

    nuovoDataframe = nuovoDataframe.drop(indiciDaRimuovere)


    #

    # ----------3° parte----------
    # Andiamo a gestire i valori che sono nulli e che vogliamo conservare
    # Identifica le colonne di tipo numerico
    attributiNumerici = nuovoDataframe.select_dtypes(include=[np.number]).columns
    print('Gli attributi numerici sono: ', attributiNumerici.tolist())

    # Imputa i valori nulli solo per le colonne numeriche utilizzando la media
    for att in attributiNumerici:
        nuovoDataframe[att].fillna(nuovoDataframe[att].mean(), inplace=True)

    print('Questa è la nuova distribuzione dei valori null: ', nuovoDataframe.isnull().mean())
    print(nuovoDataframe)

    #
    # # ----------4° parte----------
    # # Gestione della normalizzazione dei dati:
    # # Considerando le 2 pipeline da implementare, ovvero DecsionTree e NaiveBayes,
    # # teniamo conto che non è utile la normalizzazione dei dataset data la natura
    # # di questi 2 algoritmi
    #
    # # Uniformiamo le date
    # # Cerca le colonne che potrebbero contenere date (pattern basato su una data ipotetica)
    # date_columns = []
    # date_pattern = re.compile(r'\b\d{4}-\d{2}-\d{2}\b')  # Pattern per data in formato YYYY-MM-DD
    #
    # for column in nuovoDataframe.columns:
    #     if any(nuovoDataframe[column].astype(str).str.match(date_pattern)):
    #         date_columns.append(column)
    #
    # for attData in date_columns:
    #     nuovoDataframe[attData] = pd.to_datetime(nuovoDataframe[attData])
    #
    # # Stampare le colonne contenenti date
    # print("Colonne contenenti date:", date_columns)

    # ----------4° parte----------
    # Mostra i valori unici per ogni colonna

    # Stabiliamo una soglia per il numero di categorie degli attributi categorici
    sogliaMin = 3 # 3 perchè non ci interessa fare il one hot encoding di valori booleani
    sogliaMax = 10 # oltre i 10 valori diventa incategorizzabile gestire l'attributo come categorico
    attributiCategorici = []

    # Individuiamo quali sono gli attributi categorici
    for col in nuovoDataframe.columns:
        num_unique_values = nuovoDataframe[col].nunique()
        if num_unique_values <= sogliaMax and num_unique_values>= sogliaMin:
            attributiCategorici.append(col)

    print('I seguenti attributi sono stati evidenziati come categorici: ', attributiCategorici)

    # Usiamo One Hot Encoding per dividere gli attributi categorici
    for col in attributiCategorici:
        nuovoDataframe = pd.get_dummies(nuovoDataframe, columns=[col])

    print(nuovoDataframe)


    # Feature scaling (z-score normalization o min-max normalization)

    # Feature selection (Selezioniamo le feature che hanno più potere predittivo dato che
    # avere un dataset con troppi attributi produrrà un modello overfittato) - opzionale

    # Data balancing (Sia undersampling che oversampling (possiamo usare smote))

    # aif360

    # Se vuoi stampare il dataset:
    # nuovoDataframe.to_csv('nome_file.csv', index=False)
