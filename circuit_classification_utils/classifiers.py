import torch
import torch.nn as nn
from torch.utils.data import Dataset

class Circuit_Dataset(Dataset):
    def __init__(self, X, y, weights=None):
        super().__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        if weights is not None:
            self.weights = torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        if self.weights is not None:
            return (self.X[index], self.y[index], self.weights[index])
        return (self.X[index], self.y[index])


class MLP(nn.Module):
    def __init__(self, n_layers, hidden_dim, input_dim, output_dim):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)
        y = nn.Sigmoid()(y)
        return y