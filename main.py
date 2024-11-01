# main.py
from confusion_matrix import plot_confusion_matrix
from roc_auc_curves import plot_roc_auc
from classification_report import generate_classification_report


y_true = ['happy', 'sad', 'angry', 'surprise', 'happy', 'sad']
y_pred = ['happy', 'sad', 'happy', 'angry', 'happy', 'surprise']
labels = ['happy', 'sad', 'angry', 'surprise']

# Ejecutar cada función
plot_confusion_matrix(y_true, y_pred, labels)
plot_roc_auc(y_true, y_pred, labels)
generate_classification_report(y_true, y_pred, labels)
