import pandas as pd
def columnsFromFile(filename):
    df = pd.read_csv(filename)
    colonnes = df.columns
    colonnes_json = df.columns.to_list()
    return colonnes_json
