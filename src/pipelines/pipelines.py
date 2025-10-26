import math
import torch
from skorch import NeuralNetBinaryClassifier
from skorch.dataset import ValidSplit
from skorch.callbacks import EarlyStopping, EpochScoring, LRScheduler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from src.transformers import (
    fix_concatenated_words, SpacyTokenizer, ExtraFeatures,
    FeatureSelector, SbertVectorizer, fix_feature_names
)
from src.neural_network import DeepClassifier
from src.util import GPUManager

def preprocessing_pieline(top_k_feat=15):
    extra_features_routine = Pipeline([
        ('selector', FeatureSelector(top_k_feat)),
        ('scaler', StandardScaler().set_output(transform="pandas")),
    ])
    col_transformer = ColumnTransformer([
        ('extra_features_routine', extra_features_routine, selector(dtype_include='number'))
    ], remainder='passthrough')
    extra_features_routine.set_output(transform='pandas')
    col_transformer.set_output(transform='pandas')
    return Pipeline([
        ('splitter', FunctionTransformer(fix_concatenated_words, validate=False)),
        ('tokenizer', SpacyTokenizer()),
        ('features_extractor', ExtraFeatures()),
        ('column_transformer', col_transformer)
    ])

def text_vecrotization_pipeline():
    vectorizer = ColumnTransformer([
        ('sbert_vectorize', SbertVectorizer(), 'text')
    ], remainder='passthrough')
    return Pipeline([
        ('fix_column_names', FunctionTransformer(fix_feature_names, validate=False)),
        ('vectorize', vectorizer)
    ])

def classification_pipeline(params):
    first_layers = [params['dim1']] * 2 if params['residual'] else [params['dim1']]
    dim2 = math.ceil(params['dim1'] * params['dim2'])
    dim3 = math.ceil(dim2 * params['dim3'])
    dims = [783, *first_layers, dim2, dim3]

    return NeuralNetBinaryClassifier(
        DeepClassifier,
        module__dims=dims,
        module__dropout=params['dropout'],
        optimizer__weight_decay=params['weight_decay'],
        max_epochs=100,
        lr=params['learning_rate'],
        optimizer=torch.optim.AdamW,
        batch_size=params['batch_size'],
        train_split=ValidSplit(0.2, stratified=True),
        callbacks=[
            EarlyStopping(patience=15, monitor='valid_loss'),
            EpochScoring(scoring='accuracy', name='valid_acc', on_train=False),
            LRScheduler(
                policy=torch.optim.lr_scheduler.ReduceLROnPlateau,
                monitor='valid_loss',
                patience=5,
                factor=0.1
            )
        ],
        device=GPUManager.device(),
    )
