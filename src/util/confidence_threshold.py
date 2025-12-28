import numpy as np

thresholds = np.linspace(0.5, 1, 15)

def find_threshold(y, probs, score_function):
    p1 = probs[:, 1]
    confidence = np.max(probs, axis=1)
    best_t = 0
    best_score = 0
    best_coverage = 0

    for t in thresholds:
        mask = confidence >= t
        coverage = mask.mean()
        if coverage < 0.6:
            continue

        score = score_function(y[mask], (p1[mask] > 0.5).astype(int))
        if score > best_score:
            best_score, best_t, best_coverage = score, t, coverage

    return best_t, best_coverage

def filter_by_threshold(probs, threshold):
    confidence = np.max(probs, axis=1)
    mask = confidence >= threshold
    return mask, mask.mean()
