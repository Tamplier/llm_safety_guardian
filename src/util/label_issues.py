import math
import logging
from cleanlab.filter import find_label_issues
from src.util import cross_val_predict

logger = logging.getLogger(__name__)

def remove_label_issues(
        classifier,
        X, y,
        frac_noise,
        removeable_percetage=0.15, fraction_stop_signal=0.005
):
    total_removed = 0
    initial_objects = X.shape[0]
    max_removed = math.ceil(initial_objects * removeable_percetage)
    issues_fraction = 1
    while issues_fraction > fraction_stop_signal:
        current_counter = X.shape[0]
        pred_probs = cross_val_predict(classifier, X, y)
        issues_mask = find_label_issues(
            labels=y.values.astype(int),
            pred_probs=pred_probs,
            filter_by='prune_by_noise_rate',
            frac_noise=frac_noise
        )
        total_removed += issues_mask.sum()
        issues_fraction = issues_mask.sum() / current_counter
        if total_removed > max_removed:
            break
        X = X[~issues_mask]
        y = y[~issues_mask]
    logger.info('Label issues: %d', issues_mask.sum())
    return X, y
