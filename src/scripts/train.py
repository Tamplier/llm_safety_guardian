import json
import logging
import argparse
import optuna
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from sklearn.calibration import CalibratedClassifierCV
from src.util import (
    PathHelper, set_log_file, flush_all_loggers,
    bootstrap_metrics, cross_val_predict, find_threshold, filter_by_threshold
)
from src.util.label_issues import remove_label_issues
from src.pipelines import classification_pipeline, calibration_pipeline
from src.scripts.reports import loss_plot, roc_plot, importance_plot

parser = argparse.ArgumentParser(description='A script that retrains a model.')
parser.add_argument(
    '--optimization_trials',
    type=int,
    default=30,
    help='Amount of trials for optuna to optimize classification parameters.'
)
parser.add_argument(
    '--frac_noise',
    type=float,
    default=0.4,
    help='Use to only remove the “top” frac_noise * num_label_issues'
)

set_log_file(PathHelper.logs.train)
logger = logging.getLogger(__name__)
args = parser.parse_args()

X_train = np.load(PathHelper.data.processed.get_path('X_train_vectorized.npz'))['X'].astype('float32')
X_test = np.load(PathHelper.data.processed.get_path('X_test_vectorized.npz'))['X'].astype('float32')
y_train = pd.read_csv(PathHelper.data.processed.y_train)['0'].astype('float32')
y_test = pd.read_csv(PathHelper.data.processed.y_test)['0'].astype('float32')

X_train, X_cal, y_train, y_cal = train_test_split(
    X_train, y_train,
    test_size=0.15,
    random_state=42,
    stratify=y_train
)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

def objective_nn(trial):
    params = {
        'residual': trial.suggest_categorical('residual', [True, False]),
        'dim1': trial.suggest_int('dim1', 256, 768, step=64),
        'dim2': trial.suggest_float('dim2', 0.3, 0.6),
        'dim3': trial.suggest_float('dim3', 0.3, 0.6),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.1),
        'weight_decay': trial.suggest_float('weight_decay', 1e-4, 0.1, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
    }
    pipeline = classification_pipeline(params)
    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=skf,
        scoring='neg_log_loss'
    )

    return scores.mean()

best_params = None
if args.optimization_trials > 0:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_nn, n_trials=args.optimization_trials, timeout=60*60*5) # 5 hours

    logger.info('Best accuracy: %f', study.best_value)
    logger.info('Best params: %s', study.best_params)

    best_params = study.best_params
else:
    best_params = {
        'dim1': 512,
        'dim2': 0.5,
        'dim3': 0.5,
        'residual': True,
        'dropout': 0.3,
        'learning_rate': 1e-4,
        'weight_decay': 1e-2,
        'batch_size': 32
    }

classifier = classification_pipeline(best_params)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

logger.info('Accuracy before cleaning: %s', bootstrap_metrics(y_test, y_pred))

if args.frac_noise > 0:
    lr = LogisticRegression(
        max_iter=10_000,
        class_weight='balanced',
        C=0.01,
        penalty='l2'
    )
    calibrated_model = CalibratedClassifierCV(lr, method='sigmoid', cv=3)
    X_train, y_train, weights = remove_label_issues(calibrated_model, X_train, y_train, args.frac_noise)
    X_train = {'data': X_train, 'sample_weight': weights}

# New t optimization for clean data
classifier = classification_pipeline(best_params)
classifier.fit(X_train, y_train)
calibrated_model = CalibratedClassifierCV(classifier, method='sigmoid', cv='prefit')
calibrated_model.fit(X_cal, y_cal)
calibrator = calibrated_model.calibrated_classifiers_[0].calibrators[0]
a = calibrator.a_
b = calibrator.b_

logger.info('Best calibration params: %s, %s', a, b)

calibrated_classifier = calibration_pipeline(classifier, a, b)

# There is a third class, the "high-risk zone."
# It is not so easy to add it, because there is no source of "ambiguous messages",
# unlike thematic subreddits.
# Therefore, the following is an attempt to distinguish them
# based on the confidence of the classifier.

pred_probs = cross_val_predict(
    calibrated_classifier,
    X_train['data'], y_train, sample_weight=X_train['sample_weight']
)
ct, _ = find_threshold(y_train, pred_probs, f1_score)

best_params['a_cal'] = a
best_params['b_cal'] = b
best_params['confidence_threshold'] = ct
with open(PathHelper.models.sbert_classifier_params, 'w', encoding='utf-8') as f:
    json.dump(best_params, f)

calibrated_classifier.fit(X_train, y_train)

calibrated_classifier.base_model.save_params(f_params=PathHelper.models.sbert_classifier_weights)

y_pred = calibrated_classifier.predict(X_test)
y_prob = calibrated_classifier.predict_proba(X_test)

logger.info('Accuracy after cleaning: %s', bootstrap_metrics(y_test, y_pred))

loss_plot(calibrated_classifier.base_model.history)
roc_plot(y_test, y_prob)
importance_plot(calibrated_classifier.base_model)

mask, coverage = filter_by_threshold(y_prob, ct)
logger.info('Applying confidence threshold: %f with coverage: %f', ct, coverage)
logger.info('High confidence accuracy: %s', bootstrap_metrics(y_test[mask], y_pred[mask]))

logger.info('Low confidence accuracy: %s', bootstrap_metrics(y_test[~mask], y_pred[~mask]))

flush_all_loggers()
