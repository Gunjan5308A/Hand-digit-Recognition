from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

def ReLu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 10))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def deriv_ReLu(Z):
    return Z > 0

def accuracy_report(y_pred, y_test):    
    y_pred_labels = np.argmax(y_pred, axis=0)
    accuracy = np.mean(y_pred_labels == y_test)
    print("="*50)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Correct: {np.sum(y_pred_labels == y_test)} / {len(y_test)}")
    return accuracy

class recog:
    def __init__(self, x, y, lr=0.1, epochs=50, batch_size=64):
        # Improved Architecture: 784 -> 256 -> 128 -> 10
        # He Initialization for ReLu
        self.w1 = np.random.randn(256, 784) * np.sqrt(2./784)
        self.b1 = np.zeros((256, 1))
        self.w2 = np.random.randn(128, 256) * np.sqrt(2./256)
        self.b2 = np.zeros((128, 1))
        self.w3 = np.random.randn(10, 128) * np.sqrt(2./128)
        self.b3 = np.zeros((10, 1))
        
        self.X_all = x
        self.y_all = y
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def forward_pass(self, X):
        self.Z1 = self.w1 @ X + self.b1
        self.A1 = ReLu(self.Z1)
        self.Z2 = self.w2 @ self.A1 + self.b2
        self.A2 = ReLu(self.Z2)
        self.Z3 = self.w3 @ self.A2 + self.b3
        self.A3 = softmax(self.Z3)
        return self.A3

    def back_prop(self, X, Y):
        m = Y.size
        one_hot_Y = one_hot(Y)
        
        # Layer 3
        dZ3 = self.A3 - one_hot_Y
        self.dw3 = 1/m * dZ3 @ self.A2.T
        self.db3 = 1/m * np.sum(dZ3, axis=1, keepdims=True)
        
        # Layer 2
        dZ2 = (self.w3.T @ dZ3) * deriv_ReLu(self.Z2)
        self.dw2 = 1/m * dZ2 @ self.A1.T
        self.db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
        
        # Layer 1
        dZ1 = (self.w2.T @ dZ2) * deriv_ReLu(self.Z1)
        self.dw1 = 1/m * dZ1 @ X.T
        self.db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

    def update_param(self):
        self.w1 -= self.lr * self.dw1
        self.b1 -= self.lr * self.db1
        self.w2 -= self.lr * self.dw2
        self.b2 -= self.lr * self.db2
        self.w3 -= self.lr * self.dw3
        self.b3 -= self.lr * self.db3

    def gradient_decent(self):
        m = self.X_all.shape[1]
        for i in range(self.epochs):
            # Shuffle for Mini-batch
            permutation = np.random.permutation(m)
            X_shuffled = self.X_all[:, permutation]
            y_shuffled = self.y_all[permutation]
            
            for j in range(0, m, self.batch_size):
                X_batch = X_shuffled[:, j:j+self.batch_size]
                y_batch = y_shuffled[j:j+self.batch_size]
                
                self.forward_pass(X_batch)
                self.back_prop(X_batch, y_batch)
                self.update_param()
            
            if i % 1 == 0:
                print(f"Epoch: {i}")
                self.forward_pass(self.X_all)
                accuracy_report(self.A3, self.y_all)

    def predict(self, X):
        return self.forward_pass(X)

    def save_model(self):
        np.savez("digit_recog_model.npz", 
                 w1=self.w1, b1=self.b1, 
                 w2=self.w2, b2=self.b2,
                 w3=self.w3, b3=self.b3)

if __name__ == "__main__":
    print("Loading MNIST...")
    mnist = fetch_openml('mnist_784', version=1)
    x = mnist.data.to_numpy().astype(np.float32) / 255.0
    y = mnist.target.to_numpy().astype(np.int64)
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    
    # 30 epochs with mini-batch is usually enough for >95% accuracy
    model = recog(X_train.T, y_train, lr=0.1, epochs=20, batch_size=64)
    model.gradient_decent()
    model.save_model()
    
    print("\nFINAL TEST ACCURACY:")
    accuracy_report(model.predict(X_test.T), y_test)
