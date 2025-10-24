import gc
import logging
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.util import PathHelper, set_log_file
from src.pipelines import (
    preprocessing_pieline,
    text_vecrotization_pipeline,
    classification_pipeline
)

set_log_file(PathHelper.logs.loss_plot)
logger = logging.getLogger(__name__)

df = pd.read_csv(PathHelper.data.raw.data_set)
df = df.sample(n=10_000)

X, y = df['text'], df['class']
y = LabelEncoder().fit_transform(y)

preprocessing = preprocessing_pieline()
vecrotization = text_vecrotization_pipeline()
classifier = classification_pipeline({
    'dim1': 512,
    'dim2': 0.5,
    'dim3': 0.5,
    'residual': True,
    'dropout': 0.3,
    'learning_rate': 1e-4,
    'weight_decay': 1e-2,
    'batch_size': 32
})

X = preprocessing.fit_transform(X, y)
X = vecrotization.fit_transform(X)
logger.debug('X_train size: %.0f MB', X.nbytes / 1024**2)
gc.collect()

classifier.fit(X.astype('float32'), y.astype('float32'))

train_loss = classifier.history[:, 'train_loss']
valid_loss = classifier.history[:, 'valid_loss']
valid_acc = classifier.history[:, 'valid_acc']

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(train_loss, label='train_loss')
plt.plot(valid_loss, label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(valid_acc, label='val_acc')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.savefig(PathHelper.notebooks.loss_plot)
plt.show()
