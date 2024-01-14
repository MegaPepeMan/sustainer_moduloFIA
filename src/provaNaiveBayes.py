import warnings
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from aif360.sklearn.preprocessing import Reweighing
from aif360.sklearn.preprocessing import ReweighingMeta
from codecarbon import OfflineEmissionsTracker


def trainNaiveBayes(X_train, X_test, y_train, y_test, json_config_path, prot_attr: list[str]):
    # Dividi il dataset in feature e target
    # Vengono importate dai parametri

    # # Inizializza il modello Naive Bayes (Gaussian)
    # naive_bayes_model = GaussianNB()
    #
    # # Addestra il modello sul set di addestramento
    # naive_bayes_model.fit(X_train, y_train)
    #
    # # Fai previsioni sul set di test
    # predictions = naive_bayes_model.predict(X_test)
    #
    # # Valuta le prestazioni del modello
    # accuracy = accuracy_score(y_test, predictions)
    # print(f"Accuracy: {accuracy}")
    #
    # # Il punteggio riflette il successo con cui il nostro modello Sklearn Gaussian Naive Bayes ha predetto utilizzando i dati del test.
    # score = naive_bayes_model.score(X_test, y_test)
    # print("Naive Bayes score: ", score)






    warnings.filterwarnings("ignore", category=FutureWarning)

    tracker = OfflineEmissionsTracker(country_iso_code="ITA")

    # Tipo di distribuzione: 'multinomial' e 'gaussian'
    distribution_type = 'gaussian'

    # Imposta il parametro di smoothing (Laplace smoothing)
    alpha_smoothing = 1.0

    naive_bayes_model = None
    if distribution_type == 'gaussian':
        naive_bayes_model = GaussianNB()
    elif distribution_type == 'multinomial':
        naive_bayes_model = MultinomialNB(alpha=alpha_smoothing)

    X_train = X_train.set_index(prot_attr)
    X_test = X_test.set_index(prot_attr)

    tracker.start()
    rew = ReweighingMeta(estimator=naive_bayes_model, reweigher=Reweighing(prot_attr))
    rew.fit(X_train, y_train)
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
        recall = recall_score(y_test, y_pred,average=None)
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
    print('CodeCarbon - Sustainability (CO2): ', emissions,'kg')
















    # # Calcola le probabilità di appartenenza alle classi
    # y_prob = naive_bayes_model.predict_proba(X_test)
    #
    # # Calcola la curva ROC
    # fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    # roc_auc = auc(fpr, tpr)
    #
    # # Plotta la curva ROC
    # plt.figure(figsize=(10, 6))
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")
    # plt.show()
    return rew, accuracy, recall, precision, emissions
