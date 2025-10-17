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
    # Proceed carefully: naive impletentaion is ill-conditioned.
    # Let np.linalg.solve do the hard work.
    # Note: tx.T @ tx seems to be singular, which is a problem for np.linalg.solve.
    # There are multiple ways to combat this:
    # - removing linearly dependent rows and columns
    # - adding regularization; but that would be equivalent to ridge/lasso
    # I suggest we try fixing it in the 1st way, but will check with the TA's as well. -M

    M = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.pinv(M) @ b
    # w = np.linalg.solve(M, b)
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
    I = np.eye(tx.shape[1])
    M = tx.T @ tx + lambda_ * I
    b = tx.T @ y
    w = np.linalg.solve(M, b)
    loss = compute_loss(y, tx, w)
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


def preprocess(x_train):
    """
    Preprocess training data (no test data yet).
    In particular this means:
        - remove 0-variance columns, they don't provide any useful information
        - standardize each column (feature)
        - clip extreme outliers after standardizations
        - prepend a bias column of 1s

    Args:
        x_train: numpy array of shape (N, D)

    Returns:
        tx: numpy array of shape (N, D'), where D' <= D + 1
    """
    # Remove 0-variance columns
    mask = x_train.std(axis=0) != 0
    x_train = x_train[:, mask]
    # Standardize
    x_mean = np.nanmean(x_train, axis=0)
    x_std = np.nanstd(x_train, axis=0)
    x_train = (x_train - x_mean) / x_std
    # Clip extreme outliers
    x_train = np.clip(x_train, -5, 5)
    # Prepend the 1s column
    ones = np.ones((len(x_train), 1), dtype=float)
    tx = np.hstack((ones, x_train.astype(float)))
    # Done!
    return tx


# ---------------- KNN IMPLEMENTATION
# Built with the help of https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
def get_neighbors(train, test_row, num_neighbors):
    """Find k nearest neighbors using vectorized operations.

    Args:
        train: numpy array of training data
        test_row: single test instance
        num_neighbors: number of neighbors to return

    Returns:
        numpy array of num_neighbors nearest training instances
    """
    # Vectorized distance computation
    distances = np.sqrt(np.sum((train[:, :-1] - test_row[:-1]) ** 2, axis=1))

    # Get indices of k nearest neighbors
    nearest_neighbor_indices = np.argsort(distances)[:num_neighbors]

    # Return the k nearest neighbors
    return train[nearest_neighbor_indices]


# def predict_classification(train, test_row, num_neighbors):
#     """Make a classification prediction with neighbors for a single test instance.
#
#     Args:
#         train: numpy array of training data
#         test_row: single test instance
#         num_neighbors: number of neighbors to use
#
#     Returns:
#         prediction for the test instance
#     """
#     neighbors = get_neighbors(train, test_row, num_neighbors)
#     output_values = [row[-1] for row in neighbors]
#     prediction = max(set(output_values), key=output_values.count)
#     return prediction


def predict_classification_batch(train_features, train_labels, test_features, num_neighbors):
    """Make classification predictions for multiple test instances using vectorized operations.

    Args:
        train_features: numpy array of training features (N_train, D)
        train_labels: numpy array of training labels (N_train,)
        test_features: numpy array of test features (N_test, D)
        num_neighbors: number of neighbors to use

    Returns:
        numpy array of shape (N_test,) containing predictions for all test instances
    """
    # Compute distances for all test samples at once
    # Shape: (N_test, N_train)
    # Using broadcasting: test_features[:, np.newaxis, :] creates shape (N_test, 1, D)
    # train_features creates shape (N_train, D) and broadcasts to (N_test, N_train, D)
    distances = np.sqrt(
        np.sum((test_features[:, np.newaxis, :] - train_features) ** 2, axis=2))

    # Get indices of k nearest neighbors for each test sample
    # Shape: (N_test, num_neighbors)
    nearest_indices = np.argsort(distances, axis=1)[:, :num_neighbors]

    # Get the labels of the k nearest neighbors
    # Shape: (N_test, num_neighbors)
    neighbor_labels = train_labels[nearest_indices]

    # For each test sample, find the most common label among its k neighbors
    predictions = np.array([
        np.bincount(neighbor_labels[i].astype(int)).argmax()
        for i in range(len(test_features))
    ])

    return predictions

