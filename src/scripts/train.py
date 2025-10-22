import logging
import math
import argparse
import joblib
import optuna
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from src.transformers import fix_feature_names
from src.util import PathHelper, set_log_file, flush_all_handlers
from src.pipelines import (
    preprocessing_pieline,
    text_vecrotization_pipeline,
    classification_pipeline
)

parser = argparse.ArgumentParser(description='A script that retrains a model.')
parser.add_argument(
    '--skip_preprocessing',
    action='store_true',
    help='Skip preprocessing step and load previous results.'
)
parser.add_argument(
    '--sample_n',
    type=int,
    default=None,
    help='Use small sample instead of whole data set.'
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

df = pd.read_csv(PathHelper.data.raw.data_set)
if args.sample_n:
    df = df.sample(n=args.sample_n)
X, y = df['text'], df['class']

le = LabelEncoder()
y = le.fit_transform(y)
joblib.dump(le, PathHelper.models.label_encoder)

TEST_SIZE = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42, stratify=y)

text_vecrotization = text_vecrotization_pipeline()

if not args.skip_preprocessing:
    preprocessing = preprocessing_pieline()
    X_train_transformed = preprocessing.fit_transform(X_train, y_train)
    joblib.dump(preprocessing, PathHelper.models.light_text_preprocessor)
    fix_feature_names(X_train_transformed)
    X_train_transformed.to_csv(PathHelper.data.processed.x_train)
    y_train.to_csv(PathHelper.data.processed.y_train)

    X_test_transformed = preprocessing.transform(X_test)
    fix_feature_names(X_test_transformed)
    X_test_transformed.to_csv(PathHelper.data.processed.x_test)
    y_test.to_csv(PathHelper.data.processed.y_test)
else:
    X_train_transformed = pd.read_csv(PathHelper.data.processed.x_train)
    X_test_transformed = pd.read_csv(PathHelper.data.processed.x_test)
    y_train = pd.read_csv(PathHelper.data.processed.y_train)['class']
    y_test = pd.read_csv(PathHelper.data.processed.y_test)['class']
    if args.sample_n:
        train_n = math.ceil(args.sample_n * (1 - TEST_SIZE))
        test_n = math.ceil(args.sample_n * TEST_SIZE)
        X_train_transformed = X_train_transformed.sample(n=train_n)
        X_test_transformed = X_test_transformed.sample(n=test_n)
        y_train = y_train.loc[X_train_transformed.index]
        y_test = y_test.loc[X_test_transformed.index]

X_train_vectorized = text_vecrotization.fit_transform(X_train_transformed)
joblib.dump(text_vecrotization, PathHelper.models.vectorizer)
X_test_vectorized = text_vecrotization.transform(X_test_transformed)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 100.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 100.0, log=True),
        "use_label_encoder": False,
        "eval_metric": "logloss",
        'tree_method': 'hist',
        "n_jobs": 2,
    }
    pipeline = classification_pipeline(params)
    scores = cross_val_score(
        pipeline, X_train_vectorized, y_train,
        cv=skf,
        scoring='accuracy'
    )

    return scores.mean()

best_params = None
if args.optimization_trials > 0:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.optimization_trials)

    logger.info('Best accuracy: %f', study.best_value)
    logger.info('Best params: %s', study.best_params)

    best_params = study.best_params
else:
    best_params = {
        'n_estimators': 1000,
        'max_depth': 7,
        'learning_rate': 0.09834664138374917,
        'subsample': 0.5786264590243144,
        'reg_lambda': 0.04831147344367189,
        'reg_alpha': 2.141586285223651
    }

best_params['use_label_encoder'] = False
best_params['eval_metric'] = 'logloss'
best_params['tree_method'] = 'hist'
best_params['n_jobs'] = 2

xgb = classification_pipeline(best_params)
xgb.fit(X_train_vectorized, y_train)
joblib.dump(xgb, PathHelper.models.sbert_classifier)

y_pred = xgb.predict(X_test_vectorized)

logger.info('Final accuracy: %f', accuracy_score(y_test, y_pred))
flush_all_handlers()
