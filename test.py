import requests
import json

# Define the input data
input_data = {
    "data": {
        "Age": 42.0,
        "Total_Purchase": 11066.8,
        "Account_Manager": 0.0,
        "Years": 7.22,
        "Num_Sites": 8.0
    }
}

# Send a POST request to the API
url = 'http://localhost:5000/predict'
headers = {'Content-type': 'application/json'}
response = requests.post(url, data=json.dumps(input_data), headers=headers)

# Print the prediction result
print(response.json())
