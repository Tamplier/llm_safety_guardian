import matplotlib.pyplot as plt
from src.util import PathHelper

def loss_plot(history):
    train_loss = history[:, 'train_loss']
    valid_loss = history[:, 'valid_loss']
    valid_acc = history[:, 'valid_acc']

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
