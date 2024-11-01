# roc_auc_curves.py
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

def plot_roc_auc(y_true, y_pred, labels):
    y_true_binary = label_binarize(y_true, classes=labels)
    y_pred_binary = label_binarize(y_pred, classes=labels)
    
    plt.figure()
    for i, emotion in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_pred_binary[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC de {emotion} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curvas ROC para cada emoción')
    plt.legend(loc="lower right")
    plt.show()
