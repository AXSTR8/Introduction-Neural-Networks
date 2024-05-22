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

# Set the seed for random numbers
torch.manual_seed(142)

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


# Define different Fully Connected NN models and push these to the GPU for faster training
# The first CNN model
layers_CNN_1 = [[1,32,3],[32,1,3],144]
CNN_1 = CNN(layers_CNN_1, 10)
CNN_1.to(device)

# The second CNN model
layers_CNN_2 = [[1,32,3],[32,32,3],[32,1,3],121]
CNN_2 = CNN(layers_CNN_2, 10)
CNN_2.to(device)

# The third CNN model
layers_CNN_3 = [[1,64,3],[64,1,3],144]
CNN_3 = CNN(layers_CNN_3, 10)
CNN_3.to(device)

# The fourth CNN model
layers_CNN_4 = [[1,64,3],[64,64,3],[64,1,3],121]
CNN_4 = CNN(layers_CNN_4, 10)
CNN_4.to(device)


# Combine all the FCNN models in a list
FCNN_Models = [FCNN_1, FCNN_2, FCNN_3, FCNN_4]

# Combine all the FCNN models in a list
CNN_Models = [CNN_1, CNN_2, CNN_3, CNN_4]

# Save the untrained FCNN model parameters
for model in FCNN_Models:
    curr_directory = os.path.dirname(os.path.realpath(__file__))
    inintial_model_directory = Path(curr_directory + f"/FCNN_Model_Parameters_Initial")
    # Create the directories if they do not exist
    if not os.path.isdir(inintial_model_directory):
        inintial_model_directory.mkdir(parents=True, exist_ok=True)
    initial_model_path = Path(inintial_model_directory / f"{str(model).split('(')[0]}_{FCNN_Models.index(model)}")
    torch.save(model.state_dict(), initial_model_path)

# Save the untrained CNN model parameters
for model in CNN_Models:
    curr_directory = os.path.dirname(os.path.realpath(__file__))
    inintial_model_directory = Path(curr_directory + "/CNN_Model_Parameters_Initial") 
    # Create the directories if they do not exist
    if not os.path.isdir(inintial_model_directory):
        inintial_model_directory.mkdir(parents=True, exist_ok=True)     
    initial_model_path = Path(inintial_model_directory / f"{str(model).split('(')[0]}_{CNN_Models.index(model)}")
    torch.save(model.state_dict(), initial_model_path)


# Determine the hyperparameters for the training
batch_size = 32
epochs = 20
learning_rates = [(i)/100 for i in range(1,10,2)]

# The data set is the MNIST data set
train_dataset = torchvision.datasets.MNIST(root='./data', train = True, transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train = False, transform=transforms.ToTensor())

train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Define the training loop for a model
def training(model: torch.nn.Module, fc= False,epochs=10, lr=1e-4, dataloader = train_dataloader, test_dataloader = None, model_path = None, model_loss_path = None, model_accuracy_path = None):

    # Define the optimizer and the loss function and create a loss tracking list
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_tracker = []
    test_accuracy_tracker = []

    for epoch in range(epochs):
        model.train()
        for i, (images, targets) in enumerate(dataloader):
            optimizer.zero_grad()

            # Push the images and targets to the device
            if fc:
                images = images.reshape(-1,28*28).to(device)
            else:
                images = images.to(device)

            targets = targets.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_function(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            loss_tracker.append(loss.cpu().detach().numpy())
            del images, outputs, targets, loss

        if test_dataloader != None:
            test_accuracy = testing(model, test_dataloader, fc = fc)
            test_accuracy_tracker.append(test_accuracy)
            print(f"{str(model).split('(')[0]}_: epoch {epoch+1} with loss {loss_tracker[-1]} and test accuracy {test_accuracy}. ")
        else:
            print(f"{str(model).split('(')[0]}: epoch {epoch+1} and the loss is {loss_tracker[-1]}.")

        # Save the parameters of the trained models if a path was provided
        if model_path != None:
            torch.save(model.state_dict(), model_path)
        
        # Save the losses of the trained models
        if model_loss_path != None:
            with open(Path(model_loss_path) , "w") as f:
                for item in loss_tracker:
                    f.write("%s\n" % item)
    
        # Save the losses of the trained models
        if model_accuracy_path != None:
            with open(Path(model_accuracy_path), "w") as f:
                for item in test_accuracy_tracker:
                    f.write("%s\n" % item)


# Define the test loop for a model
def testing(model: torch.nn.Module, dataloader,  fc= False):
    # test over the full rotated test set
    total = 0
    correct = 0

    with torch.no_grad():
        model.eval()
        for i, (images, targets) in enumerate(dataloader):
            if fc:
                images = images.reshape(-1,28*28).to(device)
            else:
                images = images.to(device)
            targets = targets.to(device)

            output = model(images)

            _, prediction = torch.max(output.data, 1)
            total += targets.shape[0]
            correct += (prediction == targets).sum().item()
    return correct/total*100.


# Define the training algorithm for the FCNNs
for model in FCNN_Models:
    # Start the training for different learning rates
    for lr in range(len(learning_rates)):
        # Reset the model parameters to train it with another learning rate
        inintial_model_directory = Path(curr_directory + "/FCNN_Model_Parameters_Initial")       # A directory for the model parameters
        initial_model_path = Path(inintial_model_directory / f"{str(model).split('(')[0]}_{FCNN_Models.index(model)}")
        model.load_state_dict(torch.load(initial_model_path))
        # Specify the directory to save the model
        model_directory = Path(curr_directory + "/FCNN_Model_Parameters")             # A directory for the model parameters
        loss_directory = Path(curr_directory + "/FCNN_Model_Loss")                    # A directory for the model losses
        accuracy_directory = Path(curr_directory + "/FCNN_Model_Accuracy")                    # A directory for the model accuracy
        current_model_path = Path(model_directory / f"{str(model).split('(')[0]}_{FCNN_Models.index(model)}_lr_{learning_rates[lr]}")
        current_model_loss_path = Path(loss_directory / f"{str(model).split('(')[0]}_{FCNN_Models.index(model)}_lr_{learning_rates[lr]}_loss")
        current_model_accuracy_path = Path(accuracy_directory / f"{str(model).split('(')[0]}_{FCNN_Models.index(model)}_lr_{learning_rates[lr]}_accuracy")

        # Create the directories if they do not exist
        if not os.path.isdir(model_directory):
            model_directory.mkdir(parents=True, exist_ok=True)
        
        if not os.path.isdir(loss_directory):
            loss_directory.mkdir(parents=True, exist_ok=True)
        
        if not os.path.isdir(accuracy_directory):
            accuracy_directory.mkdir(parents=True, exist_ok=True)

    
        # Check whether model parameters of a trained model already exist if not start training
        if not os.path.isfile(current_model_path):
            training(model, fc=True, epochs=epochs, lr=learning_rates[lr], dataloader=train_dataloader, test_dataloader=test_dataloader, model_path=current_model_path, model_loss_path=current_model_loss_path, model_accuracy_path=current_model_accuracy_path)
        

# Define the training algorithm for the CNNs
for model_CNN in CNN_Models:
    for l in range(len(learning_rates)):
        # Reset the parameters of the model
        inintial_model_directory = Path(curr_directory + "/CNN_Model_Parameters_Initial")       # A directory for the model parameters
        initial_model_path = Path(inintial_model_directory / f"{str(model_CNN).split('(')[0]}_{CNN_Models.index(model_CNN)}")
        model_CNN.load_state_dict(torch.load(initial_model_path))

        # Specify the directory to save the current model
        model_directory = Path(curr_directory + "/CNN_Model_Parameters")       # A directory for the model parameters
        loss_directory = Path(curr_directory + "/CNN_Model_Loss")                    # A directory for the model losses
        accuracy_directory = Path(curr_directory + "/CNN_Model_Accuracy")                    # A directory for the model accuracy
        current_model_path = Path(model_directory / f"{str(model_CNN).split('(')[0]}_{CNN_Models.index(model_CNN)}_lr_{learning_rates[l]}")
        current_model_loss_path = Path(loss_directory / f"{str(model_CNN).split('(')[0]}_{CNN_Models.index(model_CNN)}_lr_{learning_rates[l]}_loss")
        current_model_accuracy_path = Path(accuracy_directory / f"{str(model_CNN).split('(')[0]}_{CNN_Models.index(model_CNN)}_lr_{learning_rates[l]}_accuracy")

        # Create the directories if they do not exist
        if not os.path.isdir(model_directory):
            model_directory.mkdir(parents=True, exist_ok=True)
        
        if not os.path.isdir(loss_directory):
            loss_directory.mkdir(parents=True, exist_ok=True)
        
        if not os.path.isdir(accuracy_directory):
            accuracy_directory.mkdir(parents=True, exist_ok=True)

    
        # Check whether model parameters of a trained model already exist if not start training
        if not os.path.isfile(current_model_path):
            training(model_CNN, fc=False, epochs=epochs, lr=learning_rates[l], dataloader=train_dataloader, test_dataloader=test_dataloader, model_path=current_model_path, model_loss_path=current_model_loss_path, model_accuracy_path=current_model_accuracy_path)
        
