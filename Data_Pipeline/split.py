import numpy as np


def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    n = len(X)
    indices = np.arange(n)

    if random_state is not None:
        np.random.seed(random_state)

    if shuffle:
        np.random.shuffle(indices)

    split_idx = int(n * (1 - test_size))

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
    )