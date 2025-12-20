import logging
import math
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.util import PathHelper, set_log_file
from src.pipelines import (
    preprocessing_pieline,
    feature_processing_pipeline
)

parser = argparse.ArgumentParser(description='A script that prepares a model input.')
parser.add_argument(
    '--skip_preprocessing',
    action='store_true',
    help='Skip preprocessing step and load previous results.'
)
parser.add_argument(
    '--skip_vectorization',
    action='store_true',
    help='Skip vectorization step and load previous results.'
)
parser.add_argument(
    '--disale_new_sources',
    action='store_true',
    help='Use original dataset only.'
)
parser.add_argument(
    '--sample_n',
    type=int,
    default=None,
    help='Use small sample instead of whole data set.'
)

set_log_file(PathHelper.logs.get_path('prepare_input.log'))
logger = logging.getLogger(__name__)
args = parser.parse_args()

TEST_SIZE = 0.3

f_processing = feature_processing_pipeline()
if not args.skip_preprocessing:
    df = pd.read_csv(PathHelper.data.raw.data_set)
    df_test = pd.read_csv(PathHelper.data.raw.get_path('test_set.csv'))
    ask = pd.read_csv(PathHelper.data.raw.get_path('AskReddit_comments.csv'))
    true_ask = pd.read_csv(PathHelper.data.raw.get_path('TrueAskReddit_comments.csv'))
    casual = pd.read_csv(PathHelper.data.raw.get_path('CasualConversation_comments.csv'))

    """
    There are a lot of comments from r/teenagers, and it's a problem
    because teenagers have specific patterns in their speech, punctuation, and areas of interest.
    This can lead to false signals during the training process.
    That's why we need to add some more messages from neutral sources.
    """
    if not args.disale_new_sources:
        teenagers_indexes = df['class'] == 'non-suicide'
        drop_indices = df.loc[teenagers_indexes].sample(frac=0.66, random_state=42).index
        df = df.drop(drop_indices)
        df = pd.concat([df, ask, true_ask, casual], ignore_index=True)

    if args.sample_n:
        df = df.sample(n=args.sample_n)
    X_train, y_train = df['text'], df['class']
    X_test, y_test = df['text'], df['class']

    le = LabelEncoder()
    y_train = pd.Series(le.fit_transform(y_train), index=y_train.index)
    y_test = pd.Series(le.transform(y_test), index=y_test.index)
    joblib.dump(le, PathHelper.models.label_encoder)

    preprocessing = preprocessing_pieline()
    X_train_transformed = preprocessing.fit_transform(X_train, y_train)
    joblib.dump(preprocessing, PathHelper.models.light_text_preprocessor)
    X_train_transformed.to_csv(PathHelper.data.processed.x_train)
    y_train.to_csv(PathHelper.data.processed.y_train)

    X_test_transformed = preprocessing.transform(X_test)
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
if not args.skip_vectorization:
    X_train_vectorized = f_processing.fit_transform(X_train_transformed, y_train)
    joblib.dump(f_processing, PathHelper.models.vectorizer)
    X_test_vectorized = f_processing.transform(X_test_transformed)
    train_path = PathHelper.data.processed.get_path('X_train_vectorized.npz')
    test_path = PathHelper.data.processed.get_path('X_test_vectorized.npz')
    np.savez_compressed(train_path, X=X_train_vectorized.astype('float32'))
    np.savez_compressed(test_path, X=X_test_vectorized.astype('float32'))
