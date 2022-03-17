import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

def test_data_shape(data: pd.DataFrame):
    """ If your data is assumed to have no null values then this is a valid test. """
    assert data.shape[0] > 0, "Dropping null changes shape."

def test_preprocess_data(data: pd.DataFrame):
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


def test_train_model(data: pd.Dataframe):
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

def test_inference(data: pd.DataFrame):
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

    with open('../model/model.pkl', 'r') as file:
        lr_model = pickle.load(file)

    preds = inference(lr_model, X_test.to_numpy())
    assert preds.shape[0] == len(X_test)

def test_metrisc(data: pd.DataFrame):
    pass