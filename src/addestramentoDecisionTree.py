# Per addestrare un modello Decision Tree utilizzando un dataset e un file JSON in Python, puoi seguire un approccio simile al seguente utilizzando la libreria scikit-learn per il modello Decision Tree:
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from aif360.sklearn.preprocessing import ReweighingMeta
from aif360.sklearn.preprocessing import Reweighing
from aif360.sklearn.metrics import disparate_impact_ratio
import warnings
from codecarbon import OfflineEmissionsTracker


def train_decision_tree(X_train, X_test, y_train, y_test, json_config_path, prot_attr: list[str]):

    # Dato che nella libreria di scikit-learn la funzione 'if_has_delegate_method' verrà deprecato
    # sopprimiamo l'errore

    warnings.filterwarnings("ignore", category=FutureWarning)

    tracker = OfflineEmissionsTracker(country_iso_code="ITA")

    tree_classifier = DecisionTreeClassifier(random_state=42)

    X_train = X_train.set_index(prot_attr)
    X_test = X_test.set_index(prot_attr)

    tracker.start()
    rew = ReweighingMeta(estimator=tree_classifier, reweigher=Reweighing(prot_attr))
    rew.fit(X_train, y_train)
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
    print('CodeCarbon - Sustainability (CO2): ', emissions,'kg')

    # Disperate_Impact
    # disparate = disparate_impact_ratio(y_test)
    # print('Disparate impact ratio: ', disparate)
    return rew
