import torch.nn as nn
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os, random
import csv
from tqdm.auto import tqdm 
from torch.utils.data import DataLoader

current_path = os.getcwd()

train_data_path = os.path.join(current_path, 'dataset_quality/train')
test_data_path = os.path.join(current_path, 'dataset_quality/test')

transforms = transforms.Compose([transforms.Resize(100),  # Shape 
                                 transforms.Grayscale(),  # Gray scale
                                 transforms.ToTensor()])  # Convert to tensor

train_dataset = datasets.ImageFolder(train_data_path, transform=transforms)
test_dataset = datasets.ImageFolder(test_data_path, transform=transforms)

#Load Data
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                               shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, 
                                              shuffle=True)

    
def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module):
    loss = 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Make predictions with the model
            y_pred = model(X)
            
            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
        loss /= len(data_loader)

        
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item()}


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    train_loss = 0
    for batch, (X, y) in enumerate(data_loader):

        X, y = X, y

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()
        
        

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    print(f"Train loss: {train_loss:.5f}")
def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module):
    test_loss = 0
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for X, y in data_loader:
            # Send data to GPU
            X, y = X, y
            
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)

            
        
        # Adjust metrics and print out
        test_loss /= len(data_loader)

        print(f"Test loss: {test_loss:.5f}\n")

def images_plot(features, labels):
    fig, ax = plt.subplots(2, 2, figsize=(25,25))
    ax = ax.flatten()
    random_sample = random.sample(range(16), 4)
    i = 0
    for sample in random_sample:
        label = 'OK' if labels[sample] == 1 else 'Failure'
        color = 'g' if labels[sample] == 1 else 'r'
        ax[i].imshow(features[sample].reshape(features.shape[2:]), cmap='gray')
        ax[i].set_title(label, color=color, size=35)
        i += 1
    plt.suptitle('Images quality detection', size=50)
    plt.show()
    
class Dimension(nn.Module):
    
    def __init__(self, input_shape: int, hidden_units: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                          out_channels=hidden_units, 
                          kernel_size=3, 
                          stride=1, 
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units, 
                          out_channels=hidden_units,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,
                             stride=2) # default stride value is same as kernel_size
            )
        self.block_2 = nn.Sequential(
                nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten()
            )
           
        
        
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)     
        return x.shape[1]


    
class newCNN(nn.Module):

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, linear_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=linear_shape, 
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)

        return x
    
torch.manual_seed(42)

get_dimension = Dimension(input_shape=1, hidden_units=10)
linear_size = get_dimension(torch.rand(1, 1, 100, 100))

model_2 = newCNN(input_shape=1, 
    hidden_units=10, 
    output_shape=2,
    linear_shape=linear_size)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(), 
                             lr=1e-3)

# Calculate model 0 results on test dataset
model_0_results = eval_model(model=model_2, data_loader=test_dataloader,
    loss_fn=loss_fn
)
model_0_results
epochs = 100
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader, 
        model=model_2, 
        loss_fn=loss_fn,
        optimizer=optimizer
    )
    test_step(data_loader=test_dataloader,
        model=model_2,
        loss_fn=loss_fn
    )
model_2_results = eval_model(
    model=model_2,
    data_loader=test_dataloader,
    loss_fn=loss_fn
)
model_2_results
