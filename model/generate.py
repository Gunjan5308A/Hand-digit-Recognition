
import numpy as np
from pathlib import Path

def ReLu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def singular_prediction(X):
    
    file_path = Path("digit_recog_model.npz")

    if file_path.is_file():   

        data = np.load(file_path)

        w1 = data["w1"]
        b1 = data["b1"]
        w2 = data["w2"]
        b2 = data["b2"]

    else:
        import model
        from sklearn.datasets import fetch_openml

        mnist = fetch_openml('mnist_784', version=1)

        x = mnist.data.to_numpy().astype(np.float32)
        y = mnist.target.to_numpy().astype(np.int64)

        x /= 255.0

        model = model.recog(x.T, y, lr=0.1, epochs=500)
        model.gradient_decent()
        model.save_model()

        data = np.load(file_path)

        w1 = data["w1"]
        b1 = data["b1"]
        w2 = data["w2"]
        b2 = data["b2"]



    X  = X.reshape(784, 1)
    Z1 =   w1 @ X + b1
    A1 = ReLu(Z1)
    Z2 = w2 @ A1 + b2
    A2 = softmax(Z2)
    return np.argmax(A2)

def train_model():
        import model
        from sklearn.datasets import fetch_openml

        mnist = fetch_openml('mnist_784', version=1)

        x = mnist.data.to_numpy().astype(np.float32)
        y = mnist.target.to_numpy().astype(np.int64)

        x /= 255.0

        model = model.recog(x.T, y, lr=0.1, epochs=500)
        model.gradient_decent()
        model.save_model()

