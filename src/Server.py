import os
import pickle

import flask
from flask import request, jsonify
from src.preProcessing import data_preparation, lettura_json, lettura_gruppo_privilegiato, lettura_dataset

app = flask.Flask(__name__)

port_number = 8000


def gestisci_addestramento(path_dataset: str, path_json: str, prot_attr: dict):
    parametri_addestramento = lettura_json(path_json)
    dataset_addestramento = lettura_dataset(path_dataset)
    prot_attr = lettura_gruppo_privilegiato(prot_attr)
    print('Il parametro di addestramento è:', parametri_addestramento['target'])
    return data_preparation(dataset_addestramento, parametri_addestramento, prot_attr)


@app.route('/', methods=['GET', 'POST'])
def handle_resource():
    try:
        # Verifica se la richiesta contiene dati JSON
        if request.is_json:
            # Estrai i dati JSON dalla richiesta
            dati_json = request.json

            # Puoi ora lavorare con i dati JSON come un dizionario Python
            parametri_get = request.values.to_dict()
            print(parametri_get)
            print(dati_json['pathDataset'])
            print(dati_json['pathJson'])
            df = lettura_dataset(dati_json['pathDataset'])
            print(df)
            modello_addestrato, accuracy, recall, precision, emissions = gestisci_addestramento(
                dati_json['pathDataset'], dati_json['pathJson'], parametri_get)

            print(accuracy, recall, precision, emissions)
            with open('modelloAddestrato.pkl', 'wb') as file:
                pickle.dump(modello_addestrato, file)

            # Esempio di risposta JSON
            path_modello = os.path.abspath('modelloAddestrato.pkl')
            risposta = {'messaggio': 'Addestramento concluso con successo',
                        'path_modello': path_modello,
                        'accuracy': accuracy,
                        'recall': recall,
                        'precision': precision,
                        'emissions': emissions}
            return jsonify(risposta), 200
        else:
            # Se la richiesta non contiene dati JSON, restituisci un errore
            return jsonify({'errore': 'Richiesta non valida. Assicurati di inviare dati in formato JSON.'}), 400

    except Exception as e:
        # Gestione delle eccezioni in caso di errori
        return jsonify({'errore': f'Errore durante l\'elaborazione della richiesta: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=port_number)
