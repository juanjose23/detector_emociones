# classification_report.py
from sklearn.metrics import classification_report

def generate_classification_report(y_true, y_pred, labels):
    report = classification_report(y_true, y_pred, target_names=labels)
    print(report)
