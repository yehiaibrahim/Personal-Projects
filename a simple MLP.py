import sklearn
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import torch 
from sklearn.model_selection import train_test_split
from torch import nn


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device) 

n_samples = 1000

x, y = make_circles(n_samples, noise=0.3, random_state=42)

circles = pd.DataFrame({"X1":x[:,0],
                        "X2":x[:,1],
                        "label":y})

plt.scatter(x=x[:, 0],
            y=x[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu)

x = torch.tensor(x).type(torch.float32) 
y = torch.tensor(y).type(torch.float32) 

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,
                                                    random_state=42)

class circle_model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer_1 = nn.Linear(2,10)
        self.layer_2 = nn.Linear(10,20)
        self.layer_3 = nn.Linear(20,1)
        self.relu = nn.ReLU()
    def forward (self,x):
        return self.relu(self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x))))))

model = circle_model()
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

def acc_fn (y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)*100)
    
    return acc



epochs = 10000
for epoch in range(epochs):
    model.train()
    y_logits = model(x_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    loss = loss_fn(y_logits,
                   y_train)
    acc = acc_fn(y_train, y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.inference_mode():
        test_logits = model(x_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        
        test_loss = loss_fn(test_logits, y_test)
        test_acc = acc_fn(y_test, test_pred)
        
    if epoch % 10 == 0:
        print (f"epoch: {epoch} | loss: {loss:.5f}, acc: {acc:.2f}| test_loss: {test_loss:.5f}, test_acc: {test_loss:.5f}%" )

model.eval()
with torch.inference_mode():
    y_pred_1 = torch.round(torch.sigmoid(model(x_test))).squeeze()
    
print( acc_fn(y_test, y_pred_1))