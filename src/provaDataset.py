import pandas as pd

# importa il file csv
dataset = pd.read_csv("Titanic-Dataset.csv")

print("Sopravvissuti: ", len(dataset[(dataset['Survived'] == 1)]))
print("Non sopravvissuti: ", len(dataset[(dataset['Survived'] == 0)]))
print("Sopravvissuti + non sopravvissuti: ", len(dataset))

valoriNulli = dataset.isna()

print()
print("CONTO DEI VALORI NULLI")
print(valoriNulli.sum())
print()

print(dataset.info())
print(dataset.describe())

dataset.drop('Cabin', axis=1, inplace=True)
