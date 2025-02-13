from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
from model import IDSModel  # Ensure this file exists and defines IDSModel correctly

app = Flask(__name__)

# Load the trained model with weights_only=True to address the FutureWarning
try:
    model = IDSModel(input_size=41)  # Adjust input size if necessary
    model.load_state_dict(torch.load('ids_model.pth', weights_only=True))  # Updated to suppress warning
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON data from the request
        data = request.json.get('data', None)
        if data is None:
            return jsonify({'error': 'Missing "data" in request'}), 400

        # Validate input data length
        if len(data) != 41:  # Ensure 41 features as expected
            return jsonify({'error': 'Input data must have 41 features'}), 400

        # Convert input data into a tensor
        input_data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            outputs = model(input_data)
            _, predicted = torch.max(outputs.data, 1)

        prediction = predicted.item()
        return jsonify({'prediction': prediction})

    except Exception as e:
        # Handle unexpected errors
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Intrusion Detection System API is running. Use POST /predict to make predictions.'})

if __name__ == '__main__':
    app.run(debug=True)
