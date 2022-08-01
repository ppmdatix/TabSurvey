import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basemodel_torch import BaseModelTorch

'''
    Custom implementation for the standard multi-layer perceptron
'''


class MLP(BaseModelTorch):

    def __init__(self, params, args):
        super().__init__(params, args)

        hidden_dims = [4,8,4]
        self.model = MLP_Model(n_layers=self.params["n_layers"], input_dim=self.args.num_features,
                               hidden_dim=self.args.hidden_dim[0], # self.params["hidden_dim"], # self.params["hidden_dim"],
                               output_dim=self.args.num_classes,
                               task=self.args.objective,
                               hidden_dims=hidden_dims
                               )

        self.to_device()

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.array(X, dtype=np.float)
        X_val = np.array(X_val, dtype=np.float)

        return super().fit(X, y, X_val, y_val)

    def predict_helper(self, X):
        X = np.array(X, dtype=np.float)
        return super().predict_helper(X)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "hidden_dim": args.hidden_dim,
            "n_layers": trial.suggest_int("n_layers", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.0005, 0.001)
        }
        return params


class MLP_Model(nn.Module):

    def __init__(self, n_layers, input_dim, hidden_dim, output_dim, task, hidden_dims=None):
        super().__init__()

        self.task = task

        self.layers = nn.ModuleList()

        first_hidden_dim = hidden_dim
        last_hidden_dim = hidden_dim

        if hidden_dims is not None:
            first_hidden_dim = hidden_dims[0]
            last_hidden_dim = hidden_dims[-1]

        # Input Layer (= first hidden layer)
        self.input_layer = nn.Linear(input_dim, first_hidden_dim)

        # Hidden Layers (number specified by n_layers)
        if hidden_dims is None:
            self.layers.extend([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 1)])
        else:

            self.layers.extend([nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)])

        # Output Layer
        self.output_layer = nn.Linear(last_hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.input_layer(x))

        # Use ReLU as activation for all hidden layers
        for layer in self.layers:
            x = F.relu(layer(x))

        # No activation function on the output
        x = self.output_layer(x)

        if self.task == "classification":
            x = F.softmax(x, dim=1)

        return x
