import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.transformers import SbertVectorizer
from src.util import PathHelper

def importance_plot(classifier):
    n_sbert = SbertVectorizer().model.get_sentence_embedding_dimension()
    vectorizer = joblib.load(PathHelper.models.vectorizer)
    selector = (
        vectorizer
        .named_steps['numeric_processing']
        .named_transformers_['numeric']
        .named_steps['selector']
    )
    extra_features = selector._important_features

    torch_model = classifier.module_
    first_layer = None
    for _, module in torch_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            first_layer = module
            break
    weights = first_layer.weight.data.cpu().numpy()

    importance_per_input = np.abs(weights).mean(axis=0)

    all_feature_names = [f'feat_{i}' for i in range(n_sbert)]
    all_feature_names.extend(extra_features)

    results = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importance_per_input,
        'type': ['sbert'] * n_sbert + ['extra'] * len(extra_features)
    }).sort_values('importance', ascending=False)

    top_20 = results.head(20)

    _, axes = plt.subplots(2, 1, figsize=(5, 10))

    ax = axes[0]
    top_20 = results.head(20)
    ax.barh(range(len(top_20)), top_20['importance'], color='steelblue')
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20['feature'])
    ax.set_title('Top 20 features by importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.invert_yaxis()

    top_extra = results[results['type'] == 'extra'].head(5)
    ax = axes[1]
    ax.barh(range(len(top_extra)), top_extra['importance'], color='steelblue')
    ax.set_yticks(range(len(top_extra)))
    ax.set_yticklabels(top_extra['feature'])
    ax.set_title('Top 5 extra features by importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(PathHelper.notebooks.feature_importance_plot)
