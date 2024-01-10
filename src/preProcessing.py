import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def converti_in_numero(valore):
    try:
        if (valore):
            valore_convertito = int(valore) if valore.isnumeric() else float(valore)
            return valore_convertito
        else:
            return valore
    except ValueError:
        raise ValueError  # Mantieni il valore originale se non può essere convertito

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

    df = dataset.drop(columns=columns_to_drop)

    # Stampo il dataset ripulito
    print('Le nuove colonne sono: ', df.columns.to_list())

    # ----------2° parte----------
    # Eliminiamo i record che hanno troppi valori nulli

    # Abbiamo stabilito, data la mancanza di un valore universalmente accettato per la soglia dei valori nulli,
    # che un buon compromesso sia considerare record con almeno il 15% di valori nulli da eliminare
    soglia_valori_nulli = int(0.15 * len(df.columns))  # 15% dei valori nulli rispetto al numero totale di colonne

    # Conta i valori nulli in ciascuna riga
    print('La soglia dei valori nulli è: ', soglia_valori_nulli)
    conteggio_valori_nulli = df.isnull().sum(axis=1)
    print('Conteggio dei valori nulli: ',conteggio_valori_nulli)
    indiciDaRimuovere = conteggio_valori_nulli[conteggio_valori_nulli >= soglia_valori_nulli].index
    print('Gli indici da rimuovere sono: ', indiciDaRimuovere)

    df = df.drop(indiciDaRimuovere)

    # Resettiamo gli indici del dataframe dopo la cancellazione
    df = df.reset_index(drop=True)


    # ----------3° parte----------
    # Andiamo a gestire i valori che sono nulli e che vogliamo conservare

    # Identifica le colonne di tipo numerico
    attributiNumerici = df.select_dtypes(include=[np.number]).columns
    print('Gli attributi numerici sono: ', attributiNumerici.tolist())

    # Imputa i valori nulli solo per le colonne numeriche utilizzando la media
    for att in attributiNumerici:
        df[att].fillna(df[att].mean(), inplace=True)

    print('Questa è la nuova distribuzione dei valori null: ', df.isnull().mean())
    print(df)

    # ----------5° parte----------
    # converti gli attributi che contengono solo numeri in attributi di tipo float64 o int64

    # Salviamo tutte le colonne non numeriche
    colonneNonNumeriche = df.select_dtypes(exclude=['float64', 'int64']).columns.tolist()

    for col in colonneNonNumeriche:
        # Salviamo la colonna in una variabile temporanea in modo da farne il rollback nel caso questa non sia solo numerica
        df['colTmp'] = df[col]
        try:
            print('Ora controllo col:', col)
            try:
                df[col] = df[col].apply(converti_in_numero)
            except ValueError as e:
                print(e)
                # Se si verifica un'eccezione, ripristina la colonna 'col' alla sua copia originale, 'colTmp'
                print('Questa è la colonna di cui annullo le modifiche: ', col)
                df[col] = df['colTmp']

                # Rimuovi la colonna temporanea utilizzata per il rollback
                df.drop('colTmp', axis=1, inplace=True)
        except Exception as e:
            print('Conversione finita, nessun errore')

    print(df.dtypes)


    #
    # # ----------X° parte----------
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
    # for column in df.columns:
    #     if any(df[column].astype(str).str.match(date_pattern)):
    #         date_columns.append(column)
    #
    # for attData in date_columns:
    #     df[attData] = pd.to_datetime(df[attData])
    #
    # # Stampare le colonne contenenti date
    # print("Colonne contenenti date:", date_columns)


    # ----------6° parte----------
    # Mostra i valori unici per ogni colonna

    # Stabiliamo una soglia per il numero di categorie degli attributi categorici
    sogliaMin = 3 # 3 perchè non ci interessa fare il one hot encoding di valori booleani
    sogliaMax = 10 # oltre i 10 valori diventa incategorizzabile gestire l'attributo come categorico
    attributiCategorici = []

    # Individuiamo quali sono gli attributi categorici
    for col in df.columns:
        num_unique_values = df[col].nunique()
        if num_unique_values <= sogliaMax and num_unique_values>= sogliaMin:
            attributiCategorici.append(col)
        elif num_unique_values == 2:
            df[col] = df[col].apply(lambda x: x != 0)

    print('I seguenti attributi sono stati evidenziati come categorici: ', attributiCategorici)

    # Usiamo One Hot Encoding per dividere gli attributi categorici
    for col in attributiCategorici:
        df = pd.get_dummies(df, columns=[col])

    print(df)

    # ----------7° parte----------
    # Feature scaling (min-max normalization)

    # distinguiamo le colonne da numeriche da quelle non numeriche
    colonneNumeriche = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    colonneNonNumeriche = df.select_dtypes(exclude=['float64', 'int64']).columns.tolist()

    print(colonneNumeriche)
    print(colonneNonNumeriche)
    # Inizializzazione del MinMaxScaler
    scaler = MinMaxScaler()

    # Normalizzazione delle features
    datiNormalizzati = scaler.fit_transform(df[colonneNumeriche])

    # Creazione di un nuovo DataFrame con i dati normalizzati
    dfNormalizzato = pd.DataFrame(datiNormalizzati, columns=colonneNumeriche )

    print(dfNormalizzato)
    # Aggiunta delle colonne non numeriche al DataFrame normalizzato
    for col in colonneNonNumeriche:
        dfNormalizzato[col] = df[col]

    print("Dati normalizzati:")
    print(dfNormalizzato)



    # Feature selection (Selezioniamo le feature che hanno più potere predittivo dato che
    # avere un dataset con troppi attributi produrrà un modello overfittato) - opzionale

    # Data balancing (Sia undersampling che oversampling (possiamo usare smote))

    # aif360

    # Se vuoi stampare il dataset:
    # dfNormalizzato.to_csv('nome_file.csv', index=False)