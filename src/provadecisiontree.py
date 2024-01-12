# Per addestrare un modello Decision Tree utilizzando un dataset e un file JSON in Python, puoi seguire un approccio simile al seguente utilizzando la libreria scikit-learn per il modello Decision Tree:
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from aif360.sklearn.preprocessing import ReweighingMeta
from aif360.sklearn.preprocessing import Reweighing


def train_decision_tree(X_train, X_test, y_train, y_test, json_config_path, prot_attr: list[str]):
    # # Addestrare il modello Decision Tree
    # clf = DecisionTreeClassifier()
    # clf.fit(X_train, y_train)
    #
    # # Valutare il modello sul test set
    # y_pred = clf.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    #
    # print("Accuracy on test set: {:.2f}".format(accuracy))
    #
    # return clf

    tree_classifier = DecisionTreeClassifier(random_state=42)

    originalRew = Reweighing('Sex')
    originalRew.fit_transform(X_train, y_train)
    #rew = ReweighingMeta(estimator=tree_classifier, reweigher=Reweighing('Sex'))

    X_train.to_csv('X_train.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)

    #rew.fit(X_train, y_train.values)
    #
    # # Valutare il modello sul test set
    # y_pred = rew.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    #
    # print("Accuracy on test set: {:.2f}".format(accuracy))

    return originalRew
