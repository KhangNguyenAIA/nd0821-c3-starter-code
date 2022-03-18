"""
This script will be used to load raw data, preprocess and
split those data into train and test sets before training model and
predicting.

Author: Khang Nguyen
Date: March 2022
"""
import pickle
import pandas as pd
import numpy as np
import logging
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from ml.data import (
    process_data
)
from ml.model import (
    train_model, 
    compute_model_metrics, 
    inference,
    slicing_inference_performance
)

logging.basicConfig(
    filename='log_file_name.log',
    level=logging.INFO, 
    format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def import_data(pth):
    '''
        Load data from csv path
    '''
    try:
        dataframe = pd.DataFrame(pd.read_csv(pth))
        logger.info("[SUCCESS]: Load data")
        return dataframe

    except FileNotFoundError as err:
        logger.error("ERROR: Failed to read file at %s", pth)
        raise err

if __name__ == "__main__":

    data = import_data("../data/census.csv")

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
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
    X_train, y_train, encoder, lb = process_data(
        train, 
        categorical_features=cat_features, 
        label="salary", 
        training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test, 
        categorical_features=cat_features, 
        label="salary", 
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Train and save a model.
    lr_model = train_model(X_train, y_train)
    joblib.dump(lr_model, "../model/model.pkl") 
    joblib.dump(encoder, "../model/encoder.enc")
    joblib.dump(lb, "../model/lb.enc")

    preds = inference(lr_model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    logger.info("Test Set Metrics:")
    logger.info(f"Test precision: {precision:.4f}")
    logger.info(f"Test recall: {recall:.4f}")
    logger.info(f"Test fbeta: {fbeta:.4f}\n")

    # # Load the model 
    model = joblib.load("../model/model.pkl") 
    enc = joblib.load("../model/encoder.enc")
    lb = joblib.load("../model/lb.enc")

    # Perform slice validation 
    slice_performance = slicing_inference_performance(
        model, test, enc, lb
    )
    logger.info(slice_performance)
