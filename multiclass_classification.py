## -------------------------------------------------------------------------------------------------
## -- Project : A simple MLP for multiclass classification
## -------------------------------------------------------------------------------------------------

import torch
from torch import nn
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs 
from sklearn.model_selection import train_test_split

num_classes = 4
num_features = 2
random_seed = 42

x_blob, y_blob = make_blobs(n_samples=1000, n_features=num_features, 
                            centers=num_classes, cluster_std=1.5, 
                            random_state=random_seed)

x_blob = torch.from_numpy(x_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

x_blob_train, x_blob_test, y_blob_train, y_blob_test = train_test_split(x_blob,
                                                                        y_blob,
                                                                        test_size=0.2,
                                                                        random_state = random_seed)

plt.figure(figsize=(10,7))
plt.scatter(x_blob[:, 0], x_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)

class MultiClassModel(nn.Module):
    def __init__(self, inputs, outputs, hidden_units=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(inputs, hidden_units),
            nn.Linear(hidden_units, hidden_units),
            nn.Linear(hidden_units, outputs))
        
    def forward(self, x):
        return self.linear_layer_stack(x)
    
model = MultiClassModel(2, 4, 8)

Loss_fn = nn.CrossEntropyLoss()
otpimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)


torch.manual_seed(42)

epochs = 100

for epoch in range(epochs):
    model.train()
    y_preds = model(x_blob_train)
    y_preds_1 = torch.softmax(y_preds, 1).argmax(1)
    Loss = Loss_fn(y_preds, y_blob_train)
    
    otpimizer.zero_grad()
    Loss.backward()
    otpimizer.step()
        
model.eval()
with torch.inference_mode():
     y_test_preds = model(x_blob_test)
     
y_test_preds_1 = torch.softmax(y_test_preds,1).argmax(1)
print(y_test_preds_1[:10])
print(y_blob_test[:10])
     