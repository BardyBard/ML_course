import numpy as np


def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx @ w
    average = np.mean(e ** 2) / 2
    return average


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2,). The vector of model parameters.
        max_iters: int. The number of iterations to run the algorithm.
        gamma: float. The stepsize (learning rate).

    Returns:
        (w, loss): tuple of numpy array of shape (2,) w last weight and float.
    """
    w = initial_w.copy()
    for _ in range(max_iters):
        y_pred = tx @ w
        error = y_pred - y
        grad = tx.T @ error / len(y)
        w -= gamma * grad
    loss = compute_loss(y, tx, w)
    return w, loss
