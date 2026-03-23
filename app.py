from flask import Flask, request, jsonify, render_template
import base64
import io
import numpy as np
from PIL import Image
from model.generate import DigitPredictor

app = Flask(__name__)

# Load model once on startup
predictor = DigitPredictor("digit_recog_model.npz")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_data = data["image"]

        # Remove metadata (data:image/png;base64,...)
        image_data = image_data.split(",")[1]

        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("L")

        # Resize to 28x28 using LANCZOS for better quality
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

        # Convert to numpy array and normalize
        img_array = np.array(image).astype(np.float32) / 255.0

        # Flatten and predict
        flattened = img_array.flatten() 
        prediction = predictor.predict(flattened)
        
        print(f"Prediction: {prediction}")

        return jsonify({
            "prediction": prediction,
            "status": "success"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

@app.route('/')
def home():
    return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)
