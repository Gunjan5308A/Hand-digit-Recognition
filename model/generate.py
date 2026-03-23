import numpy as np
from pathlib import Path

def ReLu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

class DigitPredictor:
    def __init__(self, model_path="digit_recog_model.npz"):
        self.model_path = Path(model_path)
        self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = self._load_model()

    def _load_model(self):
        if not self.model_path.is_file():
            print("Model not found. Run model/model.py to train.")
            return None, None, None, None, None, None
        
        data = np.load(self.model_path)
        return (data["w1"], data["b1"], 
                data["w2"], data["b2"],
                data["w3"], data["b3"])

    def predict(self, X):
        X = X.reshape(784, 1)
        # Layer 1
        Z1 = self.w1 @ X + self.b1
        A1 = ReLu(Z1)
        # Layer 2
        Z2 = self.w2 @ A1 + self.b2
        A2 = ReLu(Z2)
        # Layer 3
        Z3 = self.w3 @ A2 + self.b3
        A3 = softmax(Z3)
        return int(np.argmax(A3))

if __name__ == "__main__":
    predictor = DigitPredictor()
    dummy_input = np.zeros(784)
    print(f"Prediction: {predictor.predict(dummy_input)}")
