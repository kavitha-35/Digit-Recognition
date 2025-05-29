from flask import Flask,render_template,request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import base64



app = Flask(__name__)
model = load_model('model/Number_Identification_model.keras')

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L').resize((28, 28))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    return image_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    image_bytes = file.read()
    input_data = preprocess_image(image_bytes)
    prediction = model.predict(input_data)
    digit = int(np.argmax(prediction))
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    image_data = f'data:image/png;base64,{base64_image}'

    return render_template('prediction.html', predicted_digit=digit, image=image_data)



if __name__ == '__main__':
    app.run(debug=True)