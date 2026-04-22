# dataset.py
from sklearn.datasets import load_diabetes

# load load_diabetes dataset
def load_data():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    return X, y