import torch
import torch.nn as nn


class FCNN(nn.Module):  #Fully Connected Neural Network 
    def __init__(self, input_dimension: int, layers: list, output_dimension: int):
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

    def prediction(self, x):
        out = self.forward(x)
        return torch.argmax(out)
    
class CNN(nn.Module):  # Convolutional Neural Network 
    def __init__(self, CNN_layers: list, output_dimension: int):
        super().__init__()
        self.linear_layers = nn.ModuleList()
        self.linear_layers.extend([nn.Conv2d(CNN_layers[i][0], CNN_layers[i][1], CNN_layers[i][2]) for i in range(len(CNN_layers)-1)])
        self.linear_layers.append(nn.AvgPool2d(2))
        self.linear_layers.append(nn.Flatten(1))
        self.linear_layers.append(nn.Linear(CNN_layers[-1], output_dimension))
        self.rl = nn.ReLU()

    def forward(self, x):
        for l in range(len(self.linear_layers)):
            if l < len(self.linear_layers) - 2: # Aviods to apply the activation after pooling and the last layer
                x = self.linear_layers[l](x)
                x = self.rl(x)
            else:
                x = self.linear_layers[l](x)
        return x

    def prediction(self, x):
        out = self.forward(x)
        return torch.argmax(out)
    
