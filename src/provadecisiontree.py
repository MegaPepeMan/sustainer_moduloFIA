# Per addestrare un modello Decision Tree utilizzando un dataset e un file JSON in Python, puoi seguire un approccio simile al seguente utilizzando la libreria scikit-learn per il modello Decision Tree:

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



    unprivileged_groups = []

    tree_classifier = DecisionTreeClassifier(random_state=42)

    rew = ReweighingMeta(estimator=tree_classifier, reweigher=Reweighing('Sex'))
    print("Prova X TRAIN")
    print(X_train.columns)
    rew.fit(X_train, X_train['Sex'])
    #
    # # Valutare il modello sul test set
    # y_pred = rew.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    #
    # print("Accuracy on test set: {:.2f}".format(accuracy))

    return rew