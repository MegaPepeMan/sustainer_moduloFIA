from src.preProcessing import *

#pathdataset = "Carburanti.csv"
#dfLetto = letturaDataset(pathdataset)
#dataCleaning(dfLetto,"NOME_PRODOTTO")

pathdataset = "Titanic-Dataset.csv"
dfLetto = letturaDataset(pathdataset)
dataCleaning(dfLetto,"Survived", "Sex")

# pathdataset = "myntra_products_catalog.csv"
# dfLetto = letturaDataset(pathdataset)
# dataCleaning(dfLetto,"Gender")

# pathdataset = "ideal-dataset_1.CSV"
# dfLetto = letturaDataset(pathdataset)
# print(dfLetto)