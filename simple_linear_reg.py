## -------------------------------------------------------------------------------------------------
## -- Project : A simple linear regression model
## -------------------------------------------------------------------------------------------------

import torch
from torch import nn
import matplotlib.pyplot as plt 

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start,end,step).unsqueeze(dim=1)

y = weight*X + bias

# create train/ test set

train_split = int(0.8*len(X))
x_train,y_train = X[:train_split], y[:train_split]
x_test,y_test = X[train_split:],y[train_split:]


def plot_predictions (train_data = x_train, train_labels = y_train, test_data= x_test,test_labels = y_test, prediction = None):
    
    """
    a function to plot the training data
    """
    plt.figure( figsize = (10,7))
    
    plt.scatter (train_data, train_labels, c='b', s= 4, label = "training data")
    plt.scatter (test_data, test_labels, c='r', s= 4, label = 'test data')
    
    if prediction is not None:
        plt.scatter(test_data, prediction, c='g', s=4, label = "predictions")
        
    plt.legend(prop={"size":14});
    
class LinearRegression(nn.Module):
    def __init__ (self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)
        
    def forward(self, x : torch.tensor):
        return self.linear_layer(X)
    
torch.manual_seed(42)
model_1 = LinearRegression()

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=1e-4)
torch.manual_seed(42)
epochs = 10000
for epoch in range(epochs):
    model_1.train()
    y_pred = model_1(x_train)
    loss = loss_fn(y_pred[:40], y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(x_test)
        test_loss = loss_fn(test_pred[40:50], y_test)
        
    if epoch % 10 == 0:
        print(f"epoch: {epoch} | Loss: {loss} | test_loss: {test_loss}")
        
plot_predictions(prediction=y_pred[40:50].detach().numpy())
    
