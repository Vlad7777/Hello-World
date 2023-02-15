#### 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import SGD

#Loading Dataset

# Defining transformation to convert images to tensor
transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
])


train_data = torchvision.datasets.MNIST(root = '/train', train = True, download = True, transform = transform)


test_data = torchvision.datasets.MNIST(root = './test', train = False, download = True, transform = transform)


valid_data, test_data = random_split(test_data, [5000, 5000])

# Creating Data Loaders

train_dataloader = DataLoader(train_data, batch_size = 32)
valid_dataloader = DataLoader(valid_data, batch_size = 32)
test_dataloader = DataLoader(test_data, batch_size = 32)

# Creating Neural Network

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.layer1=nn.Linear(784, 300)
        self.layer2=nn.Linear(300, 100)
        self.layer3=nn.Linear(100, 10)
    def forward(self, x):
        x=x.view(-1, 784)
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        return x


model = MyNet()


#Creating Optimizer and Loss Functions
optimizer = SGD(model.parameters(), lr = 0.001)


loss_function = torch.nn.CrossEntropyLoss()

# Performing Training and Validation

for epoch in range(10):
    
    # Performing Training for each epoch
    training_loss = 0.
    model.train()

    # The training loop
    for batch in train_dataloader:
        optimizer.zero_grad()
        input, label = batch
        output = model(input)
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()


    # Performing Validation for each epoch
    validation_loss = 0.
    model.eval()

    # The validation loop
    for batch in valid_dataloader:
        input, label = batch
        output = model(input)
        loss = loss_function(output, label)
        validation_loss += loss.item()

    # Calculating the average training and validation loss over epoch
    training_loss_avg = training_loss/len(train_dataloader)
    validation_loss_avg = validation_loss/len(valid_dataloader)

    # Printing average training and average validation losses
    print("Epoch: {}".format(epoch))
    print("Training loss: {}".format(training_loss_avg))
    print("Validation loss: {}".format(validation_loss_avg))
    
    
    
    
 #  Testing Accuracy
 
 
 # Setting the number of correct predictions to 0
num_correct_pred = 0

# Running the model over test dataset and calculating total correct predictions
for batch in test_dataloader:
        input, label = batch
        output = model(input)
        _, predictions = torch.max(output.data, 1)
        num_correct_pred += (predictions == label).sum().item()

# Calculating the accuracy of model on test dataset
accuracy = num_correct_pred/(len(test_dataloader)*test_dataloader.batch_size)

print(accuracy)


 # Making Prediction
 
prediction = model(test_data[1][0])

print(prediction.argmax().item())
# Outputs- 6

