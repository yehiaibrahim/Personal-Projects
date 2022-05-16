## got an accuracy of 0.0

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

ds = datasets.load_breast_cancer()
X,y = ds.data, ds.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

class LogisticRegression:
    def __init__(self,learning=.001, n_iterss=1000):
        self.lr = learning
        self.n_iters = n_iterss
        self.weights = 0
        self.bias = None
    def fit (self, X, y):
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range (1000):
            
            linear_model = np.dot (X, self.weights) + self.bias
            y_pred = self._sig(linear_model)
            
            dw = (1/n_samples) * np.dot (X.T, (y_pred - y))
            db = (1/n_samples) * np.sum (y_pred-y)
            
            self.weights -=self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias 
        y_pred = self._sig(linear_model)
        
        y_pred_cls = (1 if i> 0.5 else 0 for i in y_pred)
        return np.array(y_pred_cls)
    def _sig(self, X):
        return 1/(1+np.exp(-X))

def acc (y_true, y_pred):
    return np.sum(y_true == y_pred)/len(y_true)

LR = LogisticRegression(learning = .00001, n_iterss = 1000)
LR.fit(X_train, y_train)
pred = LR.predict(X_test)
print(acc (y_test,pred))
