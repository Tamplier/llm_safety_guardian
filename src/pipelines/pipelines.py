import math
import torch
from skorch.dataset import ValidSplit
from skorch.callbacks import EarlyStopping, EpochScoring, LRScheduler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from src.transformers import (
    fix_concatenated_words, decode_html_entities,
    SpacyTokenizer, ExtraFeatures, FeatureSelector, SbertVectorizer
)
from src.neural_network import (
    DeepClassifier,
    CalibratedClassifier,
    WeightedNeuralNetBinaryClassifier
)
from src.util import GPUManager

def preprocessing_pipeline():
    return Pipeline([
        ('html_decoder', FunctionTransformer(decode_html_entities, validate=False)),
        ('splitter', FunctionTransformer(fix_concatenated_words, validate=False)),
        ('tokenizer', SpacyTokenizer()),
        ('features_extractor', ExtraFeatures()),
    ])

def feature_processing_pipeline(top_k_feat=15):
    numeric_pipe = Pipeline([
        ('selector', FeatureSelector(top_k_feat)),
        ('scaler', StandardScaler())
    ])

    return Pipeline([
        ('numeric_processing', ColumnTransformer([
            ('numeric', numeric_pipe, selector(dtype_include='number')),
            ('text', 'passthrough', ['text'])
        ], verbose_feature_names_out=False).set_output(transform='pandas')),

        ('sbert_vectorize', ColumnTransformer([
            ('sbert', SbertVectorizer(), 'text')
        ], remainder='passthrough'))
    ])

def classification_pipeline(params):
    first_layers = [params['dim1']] * 2 if params['residual'] else [params['dim1']]
    dim2 = math.ceil(params['dim1'] * params['dim2'])
    dim3 = math.ceil(dim2 * params['dim3'])
    dims = [783, *first_layers, dim2, dim3]

    return WeightedNeuralNetBinaryClassifier(
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
            EarlyStopping(patience=10, monitor='valid_loss'),
            EpochScoring(scoring='f1', name='valid_acc', on_train=False),
            LRScheduler(
                policy=torch.optim.lr_scheduler.ReduceLROnPlateau,
                monitor='valid_loss',
                patience=5,
                factor=0.1
            )
        ],
        device=GPUManager.device(),
    )

def calibration_pipeline(base_model, a, b):
    return CalibratedClassifier(base_model, a, b)
