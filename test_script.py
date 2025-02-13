import requests

# The URL of your Flask app's /predict endpoint
url = "http://127.0.0.1:5000/predict"

# Corrected data with exactly 41 features
data = {
    "data": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
             1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
             2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
             3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
             4.1]  # Exactly 41 features
}

# Check the length of data before sending it
print(f"Length of input data: {len(data['data'])}")  # This should print 41

# Send the POST request with the data
response = requests.post(url, json=data)

# Print the response from the Flask app
print(response.json())
