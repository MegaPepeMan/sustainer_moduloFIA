from src.Server import gestisci_addestramento
from src.preProcessing import *
import os

cartella_attuale = os.getcwd() # cartella tests
cartella_dataset = os.path.abspath(os.path.join(cartella_attuale, 'Dataset')) # cartella tests/Dataset


# pathdataset = "Carburanti.csv"
# dfLetto = letturaDataset(pathdataset)
# dataCleaning(dfLetto,"NOME_PRODOTTO", 'PREZZO')

path_dataset = os.path.join(cartella_dataset, "Titanic-Dataset.csv")
# file_json = os.path.join(cartella_dataset, "config-Titanic.json")
file_json = os.path.join(cartella_dataset, "naiveBayes.json")
dfLetto = lettura_dataset(path_dataset)
gestisci_addestramento(path_dataset, file_json, {"Sex": "true"})

# pathdataset = "myntra_products_catalog.csv"
# dfLetto = lettura_dataset(pathdataset)
# data_preparation(dfLetto, "NumImages", 'Gender')

# pathdataset = "ideal-dataset_1.CSV"
# dfLetto = letturaDataset(pathdataset)
# print(dfLetto)