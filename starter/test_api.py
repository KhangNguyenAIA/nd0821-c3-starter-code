from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

# Test data 1
test_data1 = {
    "age": 26,
    "workclass": "State-gov",
    "fnlgt": 11400,
    "education": "Bachelors",
    "education-num": 17,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "Black",
    "sex": "Male",
    "capital-gain": 3000,
    "capital-loss": 0,
    "hours-per-week": 50,
    "native-country": "Vietnam"
}

# Test data 2
test_data2 = {
    "age": 30,
    "workclass": "Private",
    "fnlgt": 159449,
    "education": "Bachelors",
    "education-num": 16,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec_managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Female",
    "capital-gain": 5178,
    "capital-loss": 0,
    "hours-per-week": 30,
    "native-country": "United-States"}


def test_get_root():
    '''
        Test that the root returns a message
    '''
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"Greeting": "Welcome!"}


def test_post_predict_less_than_50k():
    '''
        Test that predicted value is less than 50k
    '''
    r = client.post("/predict", json=test_data1)
    assert r.status_code == 200
    assert r.json() == {'prediction': 'Salary <= 50k'}


def test_post_predict_greater_than_50k():
    '''
        Test that predicted value is greater than 50k
    '''
    r = client.post("/predict", json=test_data2)
    assert r.status_code == 200
    assert r.json() == {'prediction': 'Salary > 50k'}