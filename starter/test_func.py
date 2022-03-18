import pandas as pd
import joblib
import pytest
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import train_model, inference

@pytest.fixture
def data():
    """ Simple function to generate some fake Pandas data."""
    df = pd.read_csv("./data/census.csv")
    return df

def test_data_shape(data):
    """ If your data is assumed to have no null values then this is a valid test. """
    assert data.shape[0] > 0, "Dropping null changes shape."

def test_preprocess_data(data):
    train, _ = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, _, _ = process_data(
        train, 
        categorical_features=cat_features, 
        label="salary", 
        training=True
    )

    assert X_train.shape[0] > 0
    assert y_train.shape[0] > 0


def test_train_model(data):
    train, _ = train_test_split(data, test_size=0.20)
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, _, _ = process_data(
        train, 
        categorical_features=cat_features, 
        label="salary", 
        training=True
    )
    lr_model = train_model(X_train, y_train)

    assert lr_model is not None

def test_inference(data):
    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    _, _, encoder, lb = process_data(
        train, 
        categorical_features=cat_features, 
        label="salary", 
        training=True
    )

    # Proces the test data with the process_data function.
    X_test, _, _, _ = process_data(
        test, 
        categorical_features=cat_features, 
        label="salary", 
        training=False,
        encoder=encoder,
        lb=lb
    )

    lr_model = joblib.load("./model/model.pkl") 

    preds = inference(lr_model, X_test)
    assert preds.shape[0] == len(X_test)
