import torch
from torch import nn
from .residual_block import ResidualBlock

class DeepClassifier(nn.Module):
    def __init__(self, dims=[783, 512, 512, 256, 128], outputs=1, dropout=0.3, temperature=1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.layers = nn.ModuleList()
        layers_amount = len(dims)
        for i in range(layers_amount - 1):
            input_dim = dims[i]
            output_dim = dims[i+1]
            if input_dim == output_dim:
                layer = ResidualBlock(input_dim, dropout)
            else:
                layer = nn.Sequential(
                    nn.Linear(input_dim, output_dim),
                    nn.LayerNorm(output_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            self.layers.append(layer)
        self.output_layer = nn.Linear(dims[-1], outputs)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x / self.temperature
