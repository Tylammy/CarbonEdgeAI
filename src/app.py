# app.py
from flask import Flask, render_template, request, jsonify
import os
from evaluate_and_predict import predict_custom_image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML page

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the file temporarily
    image_path = os.path.join('static', file.filename)
    file.save(image_path)

    # Call the prediction function
    prediction = predict_custom_image(image_path)
    
    # Return the prediction result to the frontend
    return jsonify({'predicted_class': prediction})

if __name__ == '__main__':
    app.run(debug=True)
