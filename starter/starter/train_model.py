# Script to train machine learning model.
import pickle
import pandas as pd
import boto3
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

# Add the necessary imports for the starter code.

# Add code to load in the data.
# Creating the low level functional client

data = pd.read_csv("../data/census.csv")

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

onehot_data = np.concatenate([X_train, X_test], axis=1)
onehot_data.to_csv("../data/onehot_data.csv", index=False)

# Train and save a model.
lr_model = train_model(X_train, y_train)
with open('../model/model.pkl', 'wb') as file:
    pickle.dump(lr_model, file)

preds = inference(lr_model, X_test.to_numpy())
precision, recall, fbeta = compute_model_metrics(y_test, preds)


