import torch
import torch.nn as nn


class TorchDNN(nn.Module):
    """Create a DNN to extract posteriors that can be used for HMM decoding
    Parameters:
        input_dim (int): Input features dimension
        output_dim (int): Number of classes
        num_layers (int): Number of hidden layers
        batch_norm (bool): Whether to use BatchNorm1d after each hidden layer
        hidden_dim (int): Number of neurons in each hidden layer
        dropout_p (float): Dropout probability for regularization
    """
    def __init__(
        self, input_dim, output_dim, num_layers=2, batch_norm=True, hidden_dim=256, dropout_p=0.2
    ):
        super(TorchDNN, self).__init__()
        self.layers = nn.ModuleList()

        # Ensure dimensions are integers
        input_dim = int(input_dim)
        output_dim = int(output_dim)
        hidden_dim = int(hidden_dim)

        # Initial layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        if batch_norm:
            self.layers.append(nn.BatchNorm1d(hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_p))

        # Intermediate layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_p))

        # Output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
