import torch
import torch.nn as nn


class FCNN(nn.Module):  #Fully Connected Neural Network 
    def __init__(self, input_dimension: int, layers: list, output_dimension):
        super().__init__()
        self.linear_layers = nn.ModuleList([nn.Linear(input_dimension, layers[0])])
        self.linear_layers.extend([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.linear_layers.append(nn.Linear(layers[-1], output_dimension))
        self.rl = nn.ReLU()

    def forward(self, x):
        for layer in self.linear_layers:
            x = layer(x)
            x = self.rl(x)
        return x
    