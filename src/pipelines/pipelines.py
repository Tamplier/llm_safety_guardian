import math
import torch
from skorch import NeuralNetBinaryClassifier
from skorch.dataset import ValidSplit
from skorch.callbacks import EarlyStopping, EpochScoring, LRScheduler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from src.transformers import (
    fix_concatenated_words, SpacyTokenizer, ExtraFeatures,
    FeatureSelector, SbertVectorizer
)
from src.neural_network import DeepClassifier
from src.util import GPUManager

def _select_numeric(df):
    return df.select_dtypes(include='number')

def _select_text(df):
    return df['text']

def preprocessing_pieline():
    return Pipeline([
        ('splitter', FunctionTransformer(fix_concatenated_words, validate=False)),
        ('tokenizer', SpacyTokenizer()),
        ('features_extractor', ExtraFeatures()),
    ])

def feature_processing_pipeline(top_k_feat=15):
    numeric_features = Pipeline([
        ('select_numeric', FunctionTransformer(_select_numeric)),
        ('selector', FeatureSelector(top_k_feat)),
        ('scaler', StandardScaler())
    ])
    text_features = Pipeline([
        ('select_text', FunctionTransformer(_select_text)),
        ('sbert', SbertVectorizer())
    ])

    return FeatureUnion([
        ('numeric', numeric_features),
        ('text', text_features)
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
