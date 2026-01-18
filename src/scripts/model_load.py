import json
import joblib
import torch
import numpy as np
import pandas as pd
from src.util import PathHelper, GPUManager
from src.pipelines import (
    classification_pipeline,
    calibration_pipeline
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
vectorizer = joblib.load(PathHelper.models.vectorizer)
classifier = classification_pipeline(classifier_params)
classifier.initialize()
classifier.load_params(f_params=PathHelper.models.sbert_classifier_weights)

a = classifier_params['a_cal']
b = classifier_params['b_cal']
calibrated_classifier = calibration_pipeline(classifier, a, b)
confidence_threshold = classifier_params['confidence_threshold']

def predict_with_proba(X):
    preprocessed = preprocessor.transform(X)
    vectorized = vectorizer.transform(preprocessed)
    probs = calibrated_classifier.predict_proba(vectorized.astype('float32'))
    confidences = np.max(probs, axis=1)
    pred_classes = np.argmax(probs, axis=1)
    decoded = label_encoder.inverse_transform(pred_classes).astype(object)

    mask = confidences < confidence_threshold
    decoded = np.where(mask, decoded + ' (low confidence)', decoded)

    return pd.DataFrame({
        'confidence': confidences,
        'class': decoded
    })

def predict(X):
    result = predict_with_proba(X)
    return result['class']
