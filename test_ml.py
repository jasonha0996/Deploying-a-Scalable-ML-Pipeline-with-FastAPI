import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from ml.model import train_model, compute_model_metrics
from ml.data import process_data

# Test 1: Ensure train_model returns a scikit-learn LogisticRegression instance
def test_model_training_returns_logistic_regression():
    """
    Test that train_model returns a LogisticRegression model object.
    """
    data = pd.read_csv("data/census.csv")
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    X, y, _, _ = process_data(data, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X, y)
    assert isinstance(model, LogisticRegression)

# Test 2: Ensure compute_model_metrics outputs expected values for a known case
def test_model_metrics_computation():
    """
    Test compute_model_metrics returns correct metrics for known inputs.
    """
    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 0, 1]
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert round(precision, 2) == 1.0
    assert round(recall, 2) == 0.67
    assert round(fbeta, 2) == 0.80

# Test 3: Ensure processed data returns the correct shape
def test_processed_data_shapes_match():
    """
    Test that X and y from process_data have matching lengths.
    """
    data = pd.read_csv("data/census.csv")
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    X, y, _, _ = process_data(data, categorical_features=cat_features, label="salary", training=True)
    assert X.shape[0] == y.shape[0]
