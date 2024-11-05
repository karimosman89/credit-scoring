from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, classification_report
import pickle



def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    """
    Train a Decision Tree model.
    """
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data and return ROC-AUC score and classification report.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        "roc_auc": roc_auc,
        "classification_report": report
    }

def save_model(model, file_path="model.pkl"):
    """
    Save the trained model to a file.
    """
    with open(file_path, "wb") as f:
        pickle.dump(model, f)

def load_model(file_path="model.pkl"):
    """
    Load a saved model from a file.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)

