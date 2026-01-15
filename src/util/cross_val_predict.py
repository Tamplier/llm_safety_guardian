import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

def cross_val_predict(model, X, y, cv=5, random_state=42, sample_weight=None):
    """
    Yeah, I know there is cross_val_predict in sklearn.model_selection
    But I caught an issue and it seems it's not possible to use it with skorch models
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    pred_probs = np.zeros((len(y), 2))

    X = np.asarray(X)
    y = np.asarray(y)
    for train_idx, val_idx in skf.split(X, y):
        model_clone = clone(model)
        X_filtered = X[train_idx]
        if sample_weight is not None:
            X_filtered = {
                'data': X_filtered,
                'sample_weight': sample_weight[train_idx]
            }
        model_clone.fit(
            X_filtered,
            y[train_idx]
        )
        pred_probs[val_idx] = model_clone.predict_proba(X[val_idx])

    return pred_probs
