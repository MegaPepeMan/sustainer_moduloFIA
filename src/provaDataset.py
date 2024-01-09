from src.preProcessing import letturaDataset
from src.preProcessing import *

pathdataset = "Titanic-dataset.csv"
dfLetto = letturaDataset(pathdataset)
dataCleaning(dfLetto)