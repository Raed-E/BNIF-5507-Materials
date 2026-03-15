import numpy as np
import pytest

from sklearn.datasets import make_blobs


@pytest.fixture
def simple_data():
    # small synthetic 2D blob dataset for testing
    X, y = make_blobs(n_samples=50, centers=3, random_state=0)
    return X, y
