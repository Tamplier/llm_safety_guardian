# pylint: disable=unused-import

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
device = GPUManager.device()
torch.load = lambda f, *args, **kwargs: original_torch_load(f, map_location=device, *args, **kwargs)

label_encoder = joblib.load(PathHelper.models.label_encoder)
preprocessor = joblib.load(PathHelper.models.light_text_preprocessor)
vectorizer = text_vecrotization_pipeline()
classifier = joblib.load(PathHelper.models.sbert_classifier, mmap_mode=None)

def predict(X):
    preprocessed = preprocessor.transform(X)
    # Fitting is not required by actual steps, but required by wrappers
    vectorized = fit_or_transform(vectorizer, preprocessed)
    predicted = classifier.predict(vectorized)
    decoded = label_encoder.inverse_transform(predicted)
    return decoded
