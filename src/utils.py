import pandas as pd
import json

def save_metrics(metrics, file_path="metrics.json"):
    """
    Save evaluation metrics as a JSON file.
    """
    with open(file_path, "w") as f:
        json.dump(metrics, f)

def load_metrics(file_path="metrics.json"):
    """
    Load evaluation metrics from a JSON file.
    """
    with open(file_path, "r") as f:
        return json.load(f)

def generate_report(report_data):
    """
    Print the classification report from the evaluation.
    """
    print("Classification Report:")
    print(f"Precision: {report_data['precision']}")
    print(f"Recall: {report_data['recall']}")
    print(f"F1 Score: {report_data['f1-score']}")

