from src.preProcessing import letturaDataset
from src.preProcessing import *

pathdataset = "Carburanti.csv"
dfLetto = letturaDataset(pathdataset)
dataCleaning(dfLetto,"VARIAZIONE")

#pathdataset = "Titanic-Dataset.csv"
#dfLetto = letturaDataset(pathdataset)
#dataCleaning(dfLetto,"Survived")