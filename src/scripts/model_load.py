# pylint: disable=unused-import

import os
import json
import joblib
import torch
from sklearn.preprocessing import LabelEncoder
from src.util import PathHelper, GPUManager, fit_or_transform
from src.pipelines import (
    preprocessing_pieline,
    text_vecrotization_pipeline,
    classification_pipeline
)

# Hack to load model without GPU
original_torch_load = torch.load
devie = GPUManager.device()
def torch_load_monkey_path(f, *args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = devie
    return original_torch_load(f, *args, **kwargs)
torch.load = torch_load_monkey_path

classifier_params = None
with open(PathHelper.models.sbert_classifier_params, 'r', encoding='utf-8') as f:
    classifier_params = json.load(f)

label_encoder = joblib.load(PathHelper.models.label_encoder)
preprocessor = joblib.load(PathHelper.models.light_text_preprocessor)
vectorizer = text_vecrotization_pipeline()
classifier = classification_pipeline(classifier_params)
classifier.initialize()
classifier.load_params(f_params=PathHelper.models.sbert_classifier_weights)

def predict(X):
    preprocessed = preprocessor.transform(X)
    # Fitting is not required by actual steps, but required by wrappers
    vectorized = fit_or_transform(vectorizer, preprocessed)
    predicted = classifier.predict(vectorized.astype('float32'))
    decoded = label_encoder.inverse_transform(predicted)
    return decoded
