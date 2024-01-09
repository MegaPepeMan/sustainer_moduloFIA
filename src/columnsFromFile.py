import pandas as pd
def columnsFromFile(filename):
    df = pd.read_csv(filename)
    colonnes = df.columns
    colonnes_json = df.columns.to_list()
    print(colonnes_json)

columnsFromFile('Carburanti.csv')