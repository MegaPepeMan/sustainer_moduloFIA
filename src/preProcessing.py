import pandas as pd
import numpy as np
columns=[]
def parametersSelection(path,list):
    df=pd.read_csv(path)
    columns = list
    return df
dataset=parametersSelection('Carburanti.csv',columns)
def data_cleaning(dataset, selected_columns):

    cleaned_dataset = dataset.dropna(subset=selected_columns)

    return cleaned_dataset

df_pulito = data_cleaning(dataset, columns)


print(df_pulito)
