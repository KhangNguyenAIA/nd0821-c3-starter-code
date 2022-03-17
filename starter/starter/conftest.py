import pandas as pd
import pytest


@pytest.fixture
def data():
    """ Simple function to generate some fake Pandas data."""
    df = pd.read_csv("../data/census.csv")
    return df

