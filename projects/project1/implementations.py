import numpy as np


def compute_loss(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D)
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx @ w
    average = np.mean(e**2) / 2
    return average


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N,)
        tx: numpy array of shape=(N, D)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        An numpy array of shape (D,) (same shape as w), containing the gradient of the loss at w.
    """
    N = tx.shape[0]
    y_pred = tx @ w
    error = y_pred - y
    return tx.T @ error / N


def compute_gradient_sgd(y, tx, w):
    """Computes the single-sample gradient at w.

    Args:
        y: numpy array of shape=(1,)
        tx: numpy array of shape=(D,)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        An numpy array of shape (D,) (same shape as w), containing the gradient of the loss at w.
    """
    y_pred = np.dot(w, tx)
    error = y_pred - y
    return error * tx


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Args:
        y: numpy array of shape=(N,)
        tx: numpy array of shape=(N, D)
        initial_w: numpy array of shape=(D,). The vector of model parameters.
        max_iters: int. The number of iterations to run the algorithm.
        gamma: float. The stepsize (learning rate).

    Returns:
        (w, loss): tuple of numpy array of shape (D,) w last weight and float.
    """
    w = initial_w.copy()
    for _ in range(max_iters):
        grad = compute_gradient(y, tx, w)
        # print("grad is", grad)
        w = w - gamma * grad
    loss = compute_loss(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D)
        initial_w: numpy array of shape=(D,). The vector of model parameters.
        max_iters: int. The number of iterations to run the algorithm.
        gamma: float. The stepsize (learning rate).

    Returns:
        (w, loss): tuple of numpy array of shape (D,) w last weight and float.
    """
    w = initial_w.copy()
    N = tx.shape[0]
    for _ in range(max_iters):
        random_index = np.random.randint(N)
        sampled_y = np.array([y[random_index]])
        sampled_tx = tx[random_index : random_index + 1, :].flatten()
        grad = compute_gradient_sgd(sampled_y, sampled_tx, w)
        w = w - gamma * grad
    loss = compute_loss(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D)

    Returns:
        (w, loss): tuple of numpy array of shape (D,) w last weight and float.
    """
    # Proceed carefully: naive impletentaion is ill-conditioned. Edit: solved by pseudoinverse.
    M = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.pinv(M) @ b
    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D)
        lambda_: float. The regularization parameter

    Returns:
        (w, loss): tuple of numpy array of shape (D,) w last weight and float.
    """
    # solve Mw = b
    N, D = tx.shape
    I = np.eye(D)
    A = tx.T @ tx + 2 * N * lambda_ * I
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    loss = compute_loss(y, tx, w)  # MSE only, no penalty term
    return w, loss


def sigmoid(x):
    """Activation function for logistic regression.
    Args:
        x: numpy array of shape=(N, )
    Returns:
        sigmoid(x): numpy array of shape=(N, )
    """
    return 1 / (1 + np.exp(-x))


def compute_logistic_loss(y, tx, w):
    """Calculate the loss using sigmoid activation function.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    y = (y == 1).astype(int)  # convert y from {-1, +1} to {0, 1}
    z = tx @ w
    epsilon = 1e-8  # epsilon trick to control values close to 0 and 1. Source: https://stackoverflow.com/questions/38125319/python-divide-by-zero-encountered-in-log-logistic-regression
    term1 = y * np.log(sigmoid(z) + epsilon)
    term2 = (1 - y) * np.log(1 - sigmoid(z) + epsilon)
    return -np.mean(term1 + term2)  # loss does not include the penalty term


def compute_logistic_gradient(y, tx, w, lambda_=None):
    """Computes the gradient of the logistic loss function at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        An numpy array of shape (D,) (same shape as w), containing the gradient of the loss at w.
    """
    N = tx.shape[0]
    e = sigmoid(tx @ w) - y
    average = (tx.T @ e) / N
    if lambda_:  # L2 regularization
        average += 2 * lambda_ * w  # derivation of ||w||**2
    return average


def logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_=None):
    """Logistic regression using gradient descent.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D)
        initial_w: numpy array of shape=(D,). The vector of model parameters.
        max_iters: int. The number of iterations to run the algorithm.
        gamma: float. The stepsize (learning rate).

    Returns:
        (w, loss): tuple of numpy array of shape (D,) w last weight and float.
    """
    w = initial_w.copy()
    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w, lambda_)
        w = w - gamma * grad
    loss = compute_logistic_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent.

    Args:
        y: numpy array of shape=(N, ). Target values.
        tx: numpy array of shape=(N, D). Input data.
        lambda_: float. Regularization parameter.
        initial_w: numpy array of shape=(D, ). Initial weights.
        max_iters: int. Number of iterations for gradient descent.
        gamma: float. Learning rate.

    Returns:
        tuple: A tuple (w, loss) where:
            - w: numpy array of shape=(D, ). Final weights.
            - loss: float. Final loss value.
    """
    return logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_)
