## -------------------------------------------------------------------------------------------------
## -- Project : A simple Linear model for FashionMINST dataset
## -------------------------------------------------------------------------------------------------

import torch
from torch import nn
import matplotlib.pyplot as plt 
from torchvision import datasets 
import torchvision
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

Batch_size = 32

train_data = datasets.FashionMNIST( 
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None)

test_data = datasets.FashionMNIST( 
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None)

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=Batch_size,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                              batch_size=Batch_size,
                              shuffle=True)

def acc_fn (y_true, y_pred):
    
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc



class_names = train_data.classes
class FashionMinstCNN(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.output_shape = output_shape
        
        self.layer_stack = nn.Sequential(nn.Flatten(), nn.Linear(input_shape, hidden_units),
                                         nn.Linear(hidden_units, output_shape))
        
    def forward(self, x):
        return self.layer_stack(x)
        
torch.manual_seed(42)

model = FashionMinstCNN(28*28, 128, len(class_names))

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) 

epochs = 3


for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")

    train_loss = 0

    for batch, (X, y) in enumerate(train_dataloader):
        model.train() 

        y_pred = model(X)


        loss = loss_fn(y_pred, y)
        train_loss += loss 

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")

    train_loss /= len(train_dataloader)
    
    test_loss, test_acc = 0, 0 
    model.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            
            test_pred = model(X)
           
            test_loss += loss_fn(test_pred, y) 

            test_acc += acc_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

        test_loss /= len(test_dataloader)

        test_acc /= len(test_dataloader)

    print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")
