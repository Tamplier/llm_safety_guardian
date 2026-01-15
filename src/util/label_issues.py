import math
import logging
import numpy as np
from cleanlab.count import compute_confident_joint
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
        y_array = y.values.astype(int)
        confident_joint = compute_confident_joint(
            labels=y_array,
            pred_probs=pred_probs,
            calibrate=True
        )
        estimated_noise_rate = 1 - np.trace(confident_joint) / len(y_array)
        logger.info('Estimated noise rate from confident joint: %.2f%%', estimated_noise_rate * 100)

        issues_mask = find_label_issues(
            labels=y_array,
            pred_probs=pred_probs,
            filter_by='both',
            frac_noise=frac_noise,
            confident_joint=confident_joint
        )
        total_removed += issues_mask.sum()
        issues_fraction = issues_mask.sum() / current_counter
        if total_removed > max_removed:
            total_removed -= issues_mask.sum()
            break
        X = X[~issues_mask]
        y = y[~issues_mask]
    logger.info('Label issues: %d', total_removed)
    return X, y
