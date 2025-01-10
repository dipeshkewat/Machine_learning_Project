from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('car_sales_model.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get JSON input
    features = np.array(data['features']).reshape(1, -1)  # Convert to 2D array
    prediction = model.predict(features)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
