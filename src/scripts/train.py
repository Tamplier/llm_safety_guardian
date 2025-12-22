import json
import logging
import argparse
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from cleanlab.filter import find_label_issues
from src.util import PathHelper, set_log_file, flush_all_loggers, bootstrap_metrics, cross_val_predict
from src.pipelines import classification_pipeline
from src.scripts.reports import loss_plot, roc_plot, importance_plot

parser = argparse.ArgumentParser(description='A script that retrains a model.')
parser.add_argument(
    '--skip_label_fix',
    action='store_true',
    help='Skip CleanLab fixing step.'
)
parser.add_argument(
    '--optimization_trials',
    type=int,
    default=30,
    help='Amount of trials for optuna to optimize classification parameters.'
)

set_log_file(PathHelper.logs.train)
logger = logging.getLogger(__name__)
args = parser.parse_args()

X_train = np.load(PathHelper.data.processed.get_path('X_train_vectorized.npz'))['X'].astype('float32')
X_test = np.load(PathHelper.data.processed.get_path('X_test_vectorized.npz'))['X'].astype('float32')
y_train = pd.read_csv(PathHelper.data.processed.y_train)['0'].astype('float32')
y_test = pd.read_csv(PathHelper.data.processed.y_test)['0'].astype('float32')

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

def objective(trial):
    params = {
        'residual': trial.suggest_categorical('residual', [True, False]),
        'dim1': trial.suggest_int('dim1', 256, 768, step=64),
        'dim2': trial.suggest_float('dim2', 0.3, 0.6),
        'dim3': trial.suggest_float('dim3', 0.3, 0.6),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.1),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
    }
    pipeline = classification_pipeline(params)
    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=skf,
        scoring='accuracy'
    )

    return scores.mean()

best_params = None
if args.optimization_trials > 0:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.optimization_trials, timeout=60*60*5) # 5 hours

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

with open(PathHelper.models.sbert_classifier_params, 'w', encoding='utf-8') as f:
    json.dump(best_params, f)

classifier = classification_pipeline(best_params)
if not args.skip_label_fix:
    pred_probs = cross_val_predict(classifier, X_train, y_train)
    issues_mask = find_label_issues(
        labels=y_train.values.astype(int),
        pred_probs=pred_probs
    )
    y_train.loc[issues_mask] = 1 - y_train.loc[issues_mask]
    logger.info('Labels fixed: %d', issues_mask.sum())

classifier.fit(X_train, y_train)

classifier.save_params(f_params=PathHelper.models.sbert_classifier_weights)

y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)

logger.info('Final accuracy: %s', bootstrap_metrics(y_test, y_pred))

loss_plot(classifier.history)
roc_plot(y_test, y_prob)
importance_plot(classifier)

flush_all_loggers()
