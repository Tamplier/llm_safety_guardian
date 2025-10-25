import sys
from unittest.mock import MagicMock
import pandas as pd

# Don't load any models in test env
fake_model_module = MagicMock()
fake_model_module.predict_with_proba = lambda x: pd.DataFrame({'class': x, 'confidence': 1.0})
sys.modules['src.scripts.model_load'] = fake_model_module
