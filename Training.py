import os
import torch
import torch.nn as nn
import numpy as np
from Models import FCNN, CNN
from pathlib import Path
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# Define different Fully Connected NN models and push these to the GPU for faster training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# The first FCNN model
hidden_layers_FCNN_1 = [100,100]
FCNN_1 = FCNN(28*28, hidden_layers_FCNN_1, 10)
FCNN_1.to(device)

# The second FCNN model
hidden_layers_FCNN_2 = [100, 100, 100]
FCNN_2 = FCNN(28*28, hidden_layers_FCNN_1, 10)
FCNN_2.to(device)

# The third FCNN model
hidden_layers_FCNN_3 = [150, 150]
FCNN_3 = FCNN(28*28, hidden_layers_FCNN_1, 10)
FCNN_3.to(device)

# The fourth FCNN model
hidden_layers_FCNN_4 = [150, 150, 150]
FCNN_4 = FCNN(28*28, hidden_layers_FCNN_1, 10)
FCNN_4.to(device)

# Combine all the FCNN models in a list
FCNN_Models = [FCNN_1, FCNN_2, FCNN_3, FCNN_4]

# Define different Fully Connected NN models and push these to the GPU for faster training

# The first CNN model
layers_CNN_1 = [[1,16,3],[16,1,3],100]
CNN_1 = CNN(layers_CNN_1, 10)
CNN_1.to(device)

# The second CNN model
layers_CNN_2 = [[1,16,3],[16,16,3],[16,1,3],100]
CNN_2 = CNN(layers_CNN_2, 10)
CNN_2.to(device)

# The third CNN model
layers_CNN_3 = [[1,32,3],[32,1,3],100]
CNN_3 = CNN(layers_CNN_3, 10)
CNN_3.to(device)

# The fourth CNN model
layers_CNN_4 = [[1,32,3],[32,32,3],[32,1,3],100]
CNN_4 = CNN(layers_CNN_4, 10)
CNN_4.to(device)


# Combine all the FCNN models in a list
FCNN_Models = [FCNN_1, FCNN_2, FCNN_3, FCNN_4]

# Combine all the FCNN models in a list
CNN_Models = [CNN_1, CNN_2, CNN_3, CNN_4]

# Determine the hyperparameters for the training
batch_size = 32
epochs = 27
learning_rates = [(i)/100 for i in range(1,10,2)]


# The data set is the MNIST data set
train_dataset = torchvision.datasets.MNIST(root='./data', train = True, transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train = False, transform=transforms.ToTensor())

train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define the training algorithm for the FCNNs
for i in range(len(FCNN_Models)):
    for lr in range(len(learning_rates)):
        # Specify the current directory
        curr_direcotry = os.path.dirname(os.path.realpath(__file__))
        model_directory = curr_direcotry + "/Model_Parameters" # A directory for the model parameters
        loss_directory = curr_direcotry + "/Loss"                    # A directory for the model losses
        model_dir_path = Path(model_directory)
        loss_dir_path = Path(loss_directory)
        current_model_path = model_directory + f"/FCNN_{i+1}_lr_{learning_rates[lr]}"
        current_model_loss_path = loss_directory + f"/FCNN_{i+1}_lr_{learning_rates[lr]}_loss"


        # Create the directories if they do not exist
        if not os.path.isdir(model_dir_path):
            model_dir_path.mkdir(parents=True, exist_ok=True)
        
        if not os.path.isdir(loss_dir_path):
            loss_dir_path.mkdir(parents=True, exist_ok=True)

    
        # Check whether model parameters of a trained model already exist if not start training
        if not os.path.isfile(current_model_path):
        
        
            # Define the optimizer and the loss function and create a loss tracking list
            optimizer = torch.optim.Adam(FCNN_Models[i].parameters(), lr=learning_rates[lr])
            criterion = nn.CrossEntropyLoss()
            loss_tracker = []

            # Define the training loop
            for epoch in range(epochs):
                for j, (images, targets) in enumerate(train_dataloader):
                    
                    # Push the images and targets to the device
                    images = images.to(device)
                    targets = targets.to(device)
                    images = torch.flatten(images, start_dim=1)

                    # Forward pass
                    outputs = FCNN_Models[i].forward(images)
                    loss = criterion(outputs, targets)
                    loss_tracker.append(loss.cpu().detach().numpy())

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                # Report over the training progress after certain epoch
                print(f"FCNN_{i+1}: In epoch {epoch+1} with loss = {loss.item()}.")
            
            torch.save(FCNN_Models[i].state_dict(), current_model_path)

            with open(current_model_loss_path, "w") as f:
                for item in loss_tracker:
                    f.write("%s\n" % item)



