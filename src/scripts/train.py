import json
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
from src.util import PathHelper, set_log_file, flush_all_loggers
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

TEST_SIZE = 0.3

text_vecrotization = text_vecrotization_pipeline()

if not args.skip_preprocessing:
    df = pd.read_csv(PathHelper.data.raw.data_set)
    if args.sample_n:
        df = df.sample(n=args.sample_n)
    X, y = df['text'], df['class']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y = pd.Series(y_enc, index=y.index)
    del y_enc
    joblib.dump(le, PathHelper.models.label_encoder)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42, stratify=y)

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
    X_train_transformed = pd.read_csv(PathHelper.data.processed.x_train, index_col=0)
    X_test_transformed = pd.read_csv(PathHelper.data.processed.x_test, index_col=0)
    y_train = pd.read_csv(PathHelper.data.processed.y_train)['0']
    y_test = pd.read_csv(PathHelper.data.processed.y_test)['0']
    if args.sample_n:
        train_n = math.ceil(args.sample_n * (1 - TEST_SIZE))
        test_n = math.ceil(args.sample_n * TEST_SIZE)
        X_train_transformed = X_train_transformed.sample(n=train_n)
        X_test_transformed = X_test_transformed.sample(n=test_n)
        y_train = y_train.loc[X_train_transformed.index]
        y_test = y_test.loc[X_test_transformed.index]

X_train_vectorized = text_vecrotization.fit_transform(X_train_transformed)
X_test_vectorized = text_vecrotization.transform(X_test_transformed)

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
        X_train_vectorized.astype('float32'),
        y_train.astype('float32'),
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
classifier.fit(X_train_vectorized.astype('float32'), y_train.astype('float32'))

classifier.save_params(f_params=PathHelper.models.sbert_classifier_weights)

y_pred = classifier.predict(X_test_vectorized.astype('float32'))

logger.info('Final accuracy: %f', accuracy_score(y_test, y_pred))
flush_all_loggers()
