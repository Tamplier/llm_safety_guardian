from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

def fit_or_transform(estimator, data):
    try:
        check_is_fitted(estimator)
        return estimator.transform(data)
    except NotFittedError:
        return estimator.fit_transform(data)
