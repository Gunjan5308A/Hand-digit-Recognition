from flask import Flask, request, jsonify, render_template
import base64
import io
import numpy as np
from PIL import Image
from model.generate import singular_prediction

app = Flask(__name__)

number = None

@app.route("/predict", methods=["POST"])
def predict():
    global number
    data = request.get_json()
    image_data = data["image"]

    # Remove metadata (data:image/png;base64,...)
    image_data = image_data.split(",")[1]

    # Decode base64
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    # Ensure 28x28
    image = image.resize((28, 28))

    # Convert to numpy array
    img_array = np.array(image)

    # Normalize (0-255 → 0-1)
    img_array = img_array / 255.0

    # Flatten to 784
    flattened = img_array.flatten() 
    number = singular_prediction(flattened)

    print("\n\n ",number)


    return jsonify({
        "message": "Image received",
        "array_length": int(len(flattened))
    })

@app.route('/')
def home():
    global number
    return render_template('home.html', num=number)

if __name__ == "__main__":
    app.run(debug=True)