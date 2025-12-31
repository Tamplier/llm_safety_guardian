import torch
import numpy as np
from sklearn.base import BaseEstimator

class CalibratedClassifier(BaseEstimator):
    def __init__(self, base_model, temperature):
        self.base_model = base_model
        self.temperature = temperature

    def fit(self, X, y=None):
        self.base_model.fit(X, y)
        return self

    def predict_proba(self, X):
        self.base_model.module_.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.base_model.device)
            logits = self.base_model.module_.forward(X_tensor)
            scaled_logits = logits / self.temperature
            probs = torch.sigmoid(scaled_logits).cpu().numpy()
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
