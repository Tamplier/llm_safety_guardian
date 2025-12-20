import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from src.util import PathHelper

def roc_plot(y_test, y_proba):
    roc_auc = roc_auc_score(y_test, y_proba[:, 1])

    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig(PathHelper.notebooks.roc_curve_plot)
