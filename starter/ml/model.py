from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score
from .data import process_data

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    lr = LogisticRegression(C=1.0)
    lr.fit(X_train, y_train)

    return lr


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def slicing_inference_performance(model, test, encoder, lb):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    slice_performance = {}
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
    for each in cat_features:
        for each_val in test[each].unique().tolist():
            test_cat = test[test[each] == each_val].reset_index(drop=True)
            X_test_cat, y_test_cat, _, _ = process_data(
                test_cat, 
                categorical_features=cat_features, 
                label="salary", 
                training=False,
                encoder=encoder,
                lb=lb
            )

            preds = model.predict(X_test_cat)
            precision, recall, fbeta = compute_model_metrics(y_test_cat, preds)

            slice_performance[f"{each}:{each_val}"] = {
                "precision": precision,
                "recall": recall,
                "fbeta": fbeta
            }
    return slice_performance
    