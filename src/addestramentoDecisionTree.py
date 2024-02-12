from aif360.sklearn.metrics import disparate_impact_ratio
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from aif360.sklearn.preprocessing import ReweighingMeta
from aif360.sklearn.preprocessing import Reweighing
import warnings
import logging
from codecarbon import OfflineEmissionsTracker


def train_decision_tree(X_train, X_test, y_train, y_test, json_config_path, prot_attr: list[str]):

    logging.basicConfig(level=logging.WARNING)

    # Dato che nella libreria di scikit-learn la funzione 'if_has_delegate_method' verrà deprecato
    # dunque sopprimiamo l'errore
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Inizializza il tracker delle emissioni di carbonio
    tracker = OfflineEmissionsTracker(country_iso_code="ITA")

    # Inizializza il classificatore Decision Tree
    tree_classifier = DecisionTreeClassifier(
        criterion=json_config_path['decisionTreeCriterioDiSuddivisione'],
        max_depth=int(json_config_path['decisionTreeProfondita']),
        min_samples_leaf=int(json_config_path['decisionTreeCampioniFoglia'])
    )

    X_train = X_train.set_index(prot_attr)
    X_test = X_test.set_index(prot_attr)

    # Avvia il tracker delle emissioni di carbonio
    tracker.start()

    # Addestra il modello Decision Tree sul set di addestramento
    rew = ReweighingMeta(estimator=tree_classifier, reweigher=Reweighing(prot_attr))
    rew.fit(X_train, y_train)

    # Ferma il tracker delle emissioni di carbonio
    emissions: float = tracker.stop()

    # Accurancy
    y_pred = rew.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accurancy: ', accuracy)

    # Recall
    recall = recall_score(y_test, y_pred)
    print('Recall: ', recall)

    # Precision
    precision = precision_score(y_test, y_pred)
    print('Precision: ', precision)

    # Sustainability
    emissions = "{:.10f}".format(emissions)
    print('CodeCarbon - Sustainability (CO2): ', emissions, 'kg')

    # Disperate_Impact
    # Il valore del primo parametro deve contenere la label del gruppo privilegiato
    disparate = disparate_impact_ratio(X_test, rew.predict(X_test), prot_attr=prot_attr)
    print('Disparate impact ratio: ', disparate)
    return rew, accuracy, recall, precision, emissions
