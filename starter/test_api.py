from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

# Test data 1
test_data1 = {
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

# Test data 2
test_data2 = {
    "age": 30,
    "workclass": "Private",
    "fnlgt": 159449,
    "education": "Bachelors",
    "education-num": 16,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
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
    assert r.json() == {"greeting": "Hello World!"}


def test_post_predict_less_than_50k():
    '''
        Test that predicted value is less than 50k
    '''
    r = client.post("/inference", json=test_data1)
    assert r.json() == {'result': 'Salary <= 50k'}


def test_post_predict_greater_than_50k():
    '''
        Test that predicted value is greater than 50k
    '''
    r = client.post("/inference", json=test_data2)
    assert r.json() == {'result': 'Salary > 50k'}