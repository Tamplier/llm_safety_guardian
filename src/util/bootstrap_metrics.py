import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def bootstrap_metrics(y_true, y_pred, n_resamples=10_000, confidence_level=0.95):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    metrics_boot = {'accuracy': [], 'precision': [], 'recall': []}
    for _ in range(n_resamples):
        idx = np.random.choice(n, size=n, replace=True)
        yt, yp = y_true[idx], y_pred[idx]

        metrics_boot['accuracy'].append(accuracy_score(yt, yp))
        metrics_boot['precision'].append(precision_score(yt, yp, zero_division=0))
        metrics_boot['recall'].append(recall_score(yt, yp, zero_division=0))
    alpha = (1 - confidence_level) / 2
    results = {}
    for name, values in metrics_boot.items():
        results[name] = {
            'median': np.median(values),
            'interval': (np.percentile(values, alpha*100),
                        np.percentile(values, (1-alpha)*100))
        }
    results['confusion'] = confusion_matrix(y_true, y_pred)
    return results
