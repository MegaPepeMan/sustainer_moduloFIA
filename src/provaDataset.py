from src.preProcessing import letturaDataset
from src.preProcessing import *

pathdataset = "Carburanti.csv"
dfLetto = letturaDataset(pathdataset)
dataCleaning(dfLetto)

Co-authored-by: MegaPepeMan <MegaPepeMan@users.noreply.github.com>