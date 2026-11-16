from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_supervised_model(X_train, y_train):
    """Trains a Decision Tree Classifier and a Baseline Dummy Classifier."""
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    
    clf = DecisionTreeClassifier(random_state=42, max_depth=4)
    clf.fit(X_train, y_train)
    
    return clf, baseline

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and returns metrics."""
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    return acc, cm, report