import requests
import json
url = "https://project3udemy.herokuapp.com"
# url= "http://0.0.0.0:8000"

request_data1 = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

request_data2 = {
    "age": 42,
    "workclass": "Private",
    "fnlgt": 159449,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec_managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 5178,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

response = requests.post("{}/inference".format(url), data=json.dumps(request_data1))
print("Response code: ", response.status_code)
print("Response from API: ",response.json())

response = requests.post("{}/inference".format(url), data=json.dumps(request_data2))
print("Response code: ", response.status_code)
print("Response from API: ",response.json())