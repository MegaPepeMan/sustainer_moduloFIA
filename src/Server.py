import os
import pickle

import flask
from flask import request, jsonify
from src.preProcessing import dataCleaning, letturaJson, letturaGruppoPrivilegiato, letturaDataset

app = flask.Flask(__name__)

port_number = 8000

def gestisciAddestramento(pathDataset: str, pathJson: str, prot_attr: dict):
    parametriAddestramento = letturaJson(pathJson)
    datasetAddestramento = letturaDataset(pathDataset)
    prot_attr = letturaGruppoPrivilegiato(prot_attr)
    print('Ho letto tutto')
    print('Il parametro di addestramento è:',parametriAddestramento['target'])
    return dataCleaning(datasetAddestramento,parametriAddestramento,prot_attr)



@app.route('/', methods=['GET', 'POST'])
def handle_resource():
    try:
        # Verifica se la richiesta contiene dati JSON
        if request.is_json:
            # Estrai i dati JSON dalla richiesta
            dati_json = request.json

            # Puoi ora lavorare con i dati JSON come un dizionario Python
            parametriGet = request.values.to_dict()
            print(parametriGet)
            print(dati_json['pathDataset'])
            print(dati_json['pathJson'])
            df = letturaDataset(dati_json['pathDataset'])
            print(df)
            modelloAddestrato, accuracy, recall, precision, emissions = gestisciAddestramento(dati_json['pathDataset'],dati_json['pathJson'],parametriGet)

            print(accuracy, recall, precision, emissions)
            with open('modelloAddestrato.pkl', 'wb') as file:
                pickle.dump(modelloAddestrato, file)

            # Esempio di risposta JSON
            pathModello = os.path.abspath('modelloAddestrato.pkl')
            risposta = {'messaggio': 'Addestramento concluso con successo',
                        'pathModello': pathModello,
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

@app.route('/prova', methods=['GET', 'POST'])
def prova():
    try:
        # Verifica se la richiesta contiene dati JSON
        print(request.headers['Content-Type'])
        if request.headers['Content-Type'] == 'application/json':

            risposta = {'message':'Collegamento riuscito'}, 200

            return jsonify(risposta)
        else:
            # Se la richiesta non contiene dati JSON, restituisci un errore
            return jsonify({'errore': 'Richiesta non valida. Assicurati di inviare dati in formato JSON.'}), 400

    except Exception as e:
        # Gestione delle eccezioni in caso di errori
        return jsonify({'errore': f'Errore durante l\'elaborazione della richiesta: {str(e)}'}), 500



if __name__ == '__main__':
    app.run(debug=True, port=port_number)
