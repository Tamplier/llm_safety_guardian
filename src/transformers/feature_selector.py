import warnings
from collections.abc import Iterable
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif

def correlation_selection(X, y, feature_names):
    combined_df = X.copy()
    combined_df['class'] = y
    corr_matrix = combined_df.corr(method='pearson')
    results = {}
    label_correlations = corr_matrix['class'][feature_names]

    top_features = label_correlations.abs().sort_values(ascending=False)

    results = [
        (feature, label_correlations[feature])
        for feature in top_features.index
    ]
    return results

def mutual_information_selection(X, y, feature_names):
    results = {}

    mi_scores = mutual_info_classif(X, y, random_state=42)
    top_indices = np.argsort(mi_scores)[::-1]
    results = [
        (feature_names[i], mi_scores[i])
        for i in top_indices
    ]

    return results

def random_forest_importance(X, y, feature_names):
    results = {}

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        n_jobs=-1
    )
    rf.fit(X, y)

    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[::-1]

    results = [
        (feature_names[i], importances[i])
        for i in top_indices
    ]

    return results

def k_best_selection(X, y, _feature_names):
    results = {}
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    scores = selector.scores_
    mask = selector.get_support()
    feat_names = X.columns[mask]
    feat_scores = scores[mask]
    results = list(zip(feat_names, feat_scores))
    return results

def comprehensive_feature_analysis(X, y, feature_names):
    methods = {
        'Mutual Information': mutual_information_selection,
        'Random Forest': random_forest_importance,
        'Correlation': correlation_selection,
        'K Best': k_best_selection,
    }
    all_results = {}
    for method_name, method_func in methods.items():
        try:
            all_results[method_name] = method_func(X, y, feature_names)
        except Exception as e:
            print(f"Error in {method_name}: {e}")
            continue

    return all_results

def compare_methods_consensus(all_results, top_k=15):
    feature_votes = {}
    for _method_name, method_results in all_results.items():
        for rank, (feature, _score) in enumerate(method_results[:top_k]):
            if feature not in feature_votes:
                feature_votes[feature] = 0
            feature_votes[feature] += (top_k - rank) / top_k

    consensus = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)

    return consensus[:top_k]

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, top_k=15):
        self._important_features = None
        self.top_k = top_k

    def get_feature_names_out(self, input_features=None):
        return self._important_features.copy()

    def fit(self, X, y=None):
        if not isinstance(y, Iterable):
            raise ValueError('y should be iterable')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
            extra_feature_names = list(X.columns)
            results = comprehensive_feature_analysis(X, y, extra_feature_names)
            important_features = []
            consensus = compare_methods_consensus(results, top_k=self.top_k)
            features = [feature for feature, rating in consensus]
            important_features.extend(features)
        self._important_features = list(set(important_features))
        return self

    def transform(self, X):
        return X[self._important_features]
