import warnings
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from aif360.sklearn.preprocessing import Reweighing
from aif360.sklearn.preprocessing import ReweighingMeta
from codecarbon import OfflineEmissionsTracker
import logging


def train_naive_bayes(X_train, X_test, y_train, y_test, json_config_path, prot_attr: list[str]):

    # Dato che nella libreria di scikit-learn la funzione 'if_has_delegate_method' verrà deprecato
    # dunque sopprimiamo l'errore
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Inizializza il tracker delle emissioni di carbonio
    tracker = OfflineEmissionsTracker(country_iso_code="ITA")

    # Tipo di distribuzione: 'multinomial' e 'gaussian'
    distribution_type = json_config_path['naiveBayesCriterioDiSuddivisione']

    # Imposta gli iperparametri del modello
    alpha_smoothing = float(json_config_path['naiveBayesSmoothing'])
    priori = json_config_path['naiveBayesDistribuzione']

    naive_bayes_model = None
    if distribution_type == 'gaussian':
        naive_bayes_model = GaussianNB(var_smoothing=alpha_smoothing, priors=priori)
    elif distribution_type == 'multinomial':
        naive_bayes_model = MultinomialNB(alpha=alpha_smoothing)

    X_train = X_train.set_index(prot_attr)
    X_test = X_test.set_index(prot_attr)

    # Avvia il tracker delle emissioni di carbonio
    tracker.start()

    rew = ReweighingMeta(estimator=naive_bayes_model, reweigher=Reweighing(prot_attr))
    rew.fit(X_train, y_train)

    # Ferma il tracker delle emissioni di carbonio
    emissions: float = tracker.stop()

    # Accurancy
    y_pred = rew.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accurancy: ', accuracy)

    # Recall
    try:
        recall = recall_score(y_test, y_pred)
        print('Recall: ', recall)
    except ValueError:
        recall = recall_score(y_test, y_pred, average=None)
        print('Recall: ', recall)

    # Precision
    try:
        precision = precision_score(y_test, y_pred)
        print('Precision: ', precision)
    except ValueError:
        precision = precision_score(y_test, y_pred, average=None)
        print('Precision: ', precision)

    # Sustainability
    emissions = "{:.10f}".format(emissions)
    print('CodeCarbon - Sustainability (CO2): ', emissions, 'kg')

    return rew, accuracy, recall, precision, emissions
