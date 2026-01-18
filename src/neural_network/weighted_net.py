import skorch
import torch
from skorch import NeuralNetBinaryClassifier

class WeightedNeuralNetBinaryClassifier(NeuralNetBinaryClassifier):
    def __init__(self, *args, criterion__reduction='none', **kwargs):
        super().__init__(*args, criterion__reduction=criterion__reduction, **kwargs)

    def get_loss(self, y_pred, y_true, X, *args, **kwargs):
        loss_unreduced = super().get_loss(y_pred, y_true, X, *args, **kwargs)
        if not isinstance(X, dict) or 'sample_weight' not in X or X['sample_weight'] is None:
            return loss_unreduced.mean()

        sample_weight = skorch.utils.to_tensor(X['sample_weight'], device=self.device)
        loss_reduced = (sample_weight * loss_unreduced).mean()
        return loss_reduced

    def decision_function(self, X):
        self.module_.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.module_(X_tensor)
            return logits.detach().cpu().numpy().squeeze()
