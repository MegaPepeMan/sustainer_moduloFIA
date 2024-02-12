import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from src.addestramentoDecisionTree import train_decision_tree
from src.addestramentoNaiveBayes import train_naive_bayes

from loguru import logger


def converti_in_numero(valore):
    try:
        if valore:
            valore_convertito = int(valore) if valore.isnumeric() else float(valore)
            return valore_convertito
        else:
            return valore
    except ValueError:
        raise ValueError  # Mantieni il valore originale se non può essere convertito


def lettura_json(path) -> dict:
    with open(path, 'r') as file:
        # Carica il contenuto del file JSON in un dizionario
        dizionario_da_json = json.load(file)
        return dizionario_da_json


def lettura_dataset(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(e)


def lettura_gruppo_privilegiato(protAttr: dict):
    attributi = []
    for label, valore in protAttr.items():
        if valore == 'true':
            attributi.append(label)
    print(attributi)
    return attributi


def data_preparation(dataset, parametri_addestramento, attributi_protetti):

    # Configura il logger per visualizzare i messaggi di debug
    logger.remove()
    logger.add(sink="addestramento.log", level="DEBUG")

    logger.debug('Dataset: ', dataset)


    target = parametri_addestramento['target']

    # ----------1° parte----------
    # Tramite questa print abbiamo una overview sul complessivo di attributi mancanti del dataset
    percentuale_nulli = dataset.isnull().mean()
    logger.debug(dataset.isnull().mean())

    # Trova le colonne con più di 1/3 di valori nulli
    logger.debug('Le vecchie colonne sono: ', dataset.columns.to_list())
    colonne_da_eliminare = percentuale_nulli[percentuale_nulli > 1 / 3].index

    # Elimina le colonne che superano i valori nulli per oltre il 33%
    logger.debug('Le colonne da eliminare sono: ', colonne_da_eliminare)

    df = dataset.drop(columns=colonne_da_eliminare)

    # Stampo il dataset ripulito
    logger.debug('Le nuove colonne sono: ', df.columns.to_list())

    # ----------2° parte----------
    # Eliminiamo i record che hanno troppi valori nulli

    # 15% dei valori nulli rispetto al numero totale di colonne
    soglia_valori_nulli = int(0.15 * len(df.columns))

    # Conta i valori nulli in ciascuna riga
    logger.debug('La soglia dei valori nulli è: ', soglia_valori_nulli)

    conteggio_valori_nulli = df.isnull().sum(axis=1)  # axis=1 per contare i valori nulli per riga, axis=0 per colonna
    logger.debug('Conteggio dei valori nulli: ', conteggio_valori_nulli)
    indici_da_rimuovere = conteggio_valori_nulli[conteggio_valori_nulli >= soglia_valori_nulli].index
    logger.debug('Gli indici da rimuovere sono: ', indici_da_rimuovere)

    df = df.drop(indici_da_rimuovere)

    # Resettiamo gli indici del dataframe dopo la cancellazione
    df = df.reset_index(drop=True)

    # ----------3° parte----------
    # converti gli attributi che contengono solo numeri in attributi di tipo float o int

    # Salviamo tutte le colonne non numeriche
    colonne_non_numeriche = df.select_dtypes(exclude=['float64', 'int64', 'int32', 'float32']).columns.tolist()

    for col in colonne_non_numeriche:
        # Salviamo la colonna in una variabile temporanea in modo da farne il rollback
        # nel caso questa non sia solo numerica

        df['colTmp'] = df[col]
        try:
            logger.debug('Ora controllo col:', col)
            try:
                df[col] = df[col].apply(converti_in_numero)
            except ValueError as e:
                logger.debug(e)
                # Se si verifica un'eccezione, ripristina la colonna 'col' alla sua copia originale, 'colTmp'
                logger.debug('Questa è la colonna di cui annullo le modifiche: ', col)
                df[col] = df['colTmp']

                # Rimuovi la colonna temporanea utilizzata per il rollback
                df.drop('colTmp', axis=1, inplace=True)
        except Exception:
            logger.debug('Conversione finita, nessun errore')

    logger.debug(df.dtypes)

    # ----------4° parte----------
    # Andiamo a gestire i valori che sono nulli e che vogliamo conservare

    # Identifica le colonne di tipo numerico
    attributi_numerici = df.select_dtypes(include=[np.number]).columns
    logger.debug('Gli attributi numerici sono: ', attributi_numerici.tolist())

    # Imputa i valori nulli solo per le colonne numeriche utilizzando la media
    for att in attributi_numerici:
        df[att].fillna(df[att].mean(), inplace=True)
        # inplace=True per modificare il dataframe originale senza doverlo assegnare a una nuova variabile

    logger.debug('Questa è la nuova distribuzione dei valori null: ', df.isnull().mean())
    logger.debug(df)

    # ----------5° parte----------
    # Mostra i valori unici per ogni colonna

    # Stabiliamo delle soglie per il numero di categorie degli attributi categorici
    soglia_minima = 3
    soglia_massima = 10  # oltre i 10 valori diventa difficile gestire l'attributo come categorico
    attributi_categorici = []
    attributi_categorici_binari = []
    label_encoder = LabelEncoder()

    # Individuiamo quali sono gli attributi categorici
    for col in df.columns:
        numero_valori_unici = df[col].nunique()

        if soglia_massima >= numero_valori_unici >= soglia_minima and col != target:
            attributi_categorici.append(col)
        elif numero_valori_unici == 2:
            if col != target:
                # Usiamo LabelEncoder() per dividere gli attributi categorici binari
                df[col] = label_encoder.fit_transform(df[col])
                attributi_categorici_binari.append(col)
            else:
                logger.debug("Attributo categorico target: ", col)

    logger.debug('I seguenti attributi sono stati evidenziati come categorici: ', attributi_categorici)

    # Usiamo LabelEncoder() per dividere gli attributi categorici
    for col in attributi_categorici:
        df[col] = label_encoder.fit_transform(df[col])

    try:
        attributi_categorici.append(*attributi_categorici_binari)
    except TypeError:
        pass

    logger.debug('Gli unici attributi categorici sono: ', attributi_categorici)

    # Qui convertiamo i valori booleani True/False in valori 0 e 1 degli attributi categorici
    for col in attributi_categorici:
        df[col] = df[col].astype(int)

    # ----------6° parte----------
    # Feature scaling (min-max normalization)

    # distinguiamo le colonne da numeriche da quelle non numeriche
    colonne_numeriche = df.select_dtypes(include=['float64', 'int64', 'int32', 'float32']).columns.tolist()
    colonne_non_numeriche = df.select_dtypes(exclude=['float64', 'int64', 'int32', 'float32']).columns.tolist()

    logger.debug(colonne_numeriche)
    logger.debug(colonne_non_numeriche)

    # Inizializzazione del MinMaxScaler
    scaler = MinMaxScaler()

    # Normalizzazione delle features
    dati_normalizzati = scaler.fit_transform(df[colonne_numeriche])

    # Creazione di un nuovo DataFrame con i dati normalizzati
    df_normalizzato = pd.DataFrame(dati_normalizzati, columns=colonne_numeriche)

    logger.debug(df_normalizzato)
    # Aggiunta delle colonne non numeriche al DataFrame normalizzato e della colonna target
    for col in colonne_non_numeriche:
        df_normalizzato[col] = df[col]

    logger.debug("Dati normalizzati:")
    logger.debug(df_normalizzato)

    # ----------7° parte----------
    # Data balancing (oversampling usando SMOTE-NC)

    logger.debug(df_normalizzato.dtypes)

    # Trasformiamo la colonna target in numeri
    df_normalizzato[target] = label_encoder.fit_transform(df_normalizzato[target])

    X = df_normalizzato.select_dtypes(include=['float64', 'int64'])
    X = X.drop(target, axis=1)
    y = df_normalizzato[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # test_size=0.2 indica che il 20% dei dati verrà utilizzato per il test
    # random_state=42 indica che i dati verranno divisi casualmente(42 è il seed per il generatore di numeri casuali)

    smote_nc = SMOTENC(categorical_features=attributi_categorici, random_state=42)

    X_resampled, y_resampled = smote_nc.fit_resample(X_train, y_train)

    # ----------8° parte----------
    # Creiamo un grafico a torta per visualizzare la distribuzione delle classi
    # nella variabile target y prima e dopo l'oversampling

    # Creare il primo grafico per y
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)  # 1 riga, 2 colonne, primo grafico
    y.value_counts().plot.pie(autopct='%.2f%%', colors=['#2467D1', '#E2E2E2'])

    plt.title('Distribuzione di y')

    # Creare il secondo grafico per y_resampled
    plt.subplot(1, 2, 2)  # 1 riga, 2 colonne, secondo grafico
    y_resampled.value_counts().plot.pie(autopct='%.2f%%', colors=['#2467D1', '#E2E2E2'])
    plt.title('Distribuzione di y_resampled')

    plt.show()

    if parametri_addestramento['tipoModello'] == 'decisiontree':
        return train_decision_tree(X_resampled, X_test, y_resampled, y_test, parametri_addestramento,
                                   attributi_protetti)
    elif parametri_addestramento['tipoModello'] == 'naivebayes':
        return train_naive_bayes(X_resampled, X_test, y_resampled, y_test, parametri_addestramento, attributi_protetti)

    return None
