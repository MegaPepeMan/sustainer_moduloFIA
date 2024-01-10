from src.preProcessing import letturaDataset
from src.preProcessing import *

pathdataset = "Titanic-Dataset.csv"
dfLetto = letturaDataset(pathdataset)
dataCleaning(dfLetto)