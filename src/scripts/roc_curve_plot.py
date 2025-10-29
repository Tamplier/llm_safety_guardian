import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from src.util import PathHelper
from src.scripts.model_load import vectorizer, classifier

X_test_transformed = pd.read_csv(PathHelper.data.processed.x_test, index_col=0)
y_test = pd.read_csv(PathHelper.data.processed.y_test)['0']

X_test_vectorized = vectorizer.fit_transform(X_test_transformed)
y_proba = classifier.predict_proba(X_test_vectorized.astype('float32'))
roc_auc = roc_auc_score(y_test, y_proba[:, 1])

fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.savefig(PathHelper.notebooks.roc_curve_plot)
plt.show()
