from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

mnist = fetch_openml('mnist_784', version=1)

x = mnist.data.to_numpy().astype(np.float32)
y = mnist.target.to_numpy().astype(np.int64)

x /= 255.0

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

def ReLu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 10))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLu(Z):
    return Z > 0

def accuracy_report(y_pred, y_test):    
    """Generate accuracy report"""
    accuracy = np.mean(y_pred == y_test)
    print("="*50)
    print("ACCURACY REPORT")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Correct Predictions: {np.sum(y_pred == y_test)} / {len(y_test)}")
    print("="*50)


class recog:
    def __init__(self, x, y, lr, epochs):
        self.w1 = np.random.rand(10, 784)
        self.b1 = np.random.rand(10, 1)
        self.w2 = np.random.rand(10, 10)
        self.b2 = np.random.rand(10, 1)
        self.X = x
        self.y = y
        self.lr = lr
        self.epochs = epochs

    def forward_pass(self):
        self.Z1 = self.w1 @ self.X + self.b1
        self.A1 = ReLu(self.Z1)
        self.Z2 = self.w2 @ self.A1 + self.b2
        self.A2 = softmax(self.Z2)


    def back_prop(self):
        self.m = self.y.size
        self.one_hot_Y = one_hot(self.y)
        self.dZ2 = self.A2 - self.one_hot_Y
        self.dw2 = 1/self.m * self.dZ2.dot(self.A1.T)
        self.db2 = 1/self.m * np.sum(self.dZ2)
        self.dZ1 = self.w2.T @self.dZ2 * deriv_ReLu(self.Z1)
        self.dw1 = 1/self.m * self.dZ1.dot(self.X.T)
        self.db1 = 1/self.m * np.sum(self.dZ1)

    def update_param(self):
        self.w1 = self.w1 - self.lr * self.dw1
        self.b1 = self.b1 - self.lr * self.db1
        self.w2 = self.w2 - self.lr * self.dw2
        self.b2 = self.b2 - self.lr * self.db2
        
    def gradient_decent(self):
        for i in range(self.epochs):
            self.forward_pass()
            self.back_prop()
            self.update_param()
            if i % 10 ==0:
                print(i)
                
    def parameters(self):
        return self.w1, self.w2, self.b1, self.b2
    

    def predict(self, X):
        self.L1 = self.w1 @ X + self.b1
        self.AC1 = ReLu(self.L1)
        self.L2 = self.w2 @ self.AC1 + self.b2
        self.AC2 = softmax(self.L2)
        return self.AC2


model = recog(X_train.T, y_train, lr=0.1, epochs=200)
model.gradient_decent()
y_pred = np.argmax(model.predict(X_test.T), axis=0)

# Generate accuracy report
accuracy_report(y_pred, y_test)