from src.preProcessing import letturaDataset
from src.preProcessing import *

# pathdataset = "Carburanti.csv"
# dfLetto = letturaDataset(pathdataset)
# dataCleaning(dfLetto,"NOME_PRODOTTO")

pathdataset = "Titanic-Dataset.csv"
dfLetto = letturaDataset(pathdataset)
dataCleaning(dfLetto,"Survived")

# pathdataset = "myntra_products_catalog.csv"
# dfLetto = letturaDataset(pathdataset)
# dataCleaning(dfLetto,"Gender")