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


def preprocess_structural(x_train, ones = True):
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
        mask: 0-1 numpy array of shape (, D'), which columns were kept
    """
    # Remove 0-variance columns
    mask = x_train.std(axis=0) != 0
    tx = x_train[:, mask]
    if ones:
        # Prepend the 1s column
        ones = np.ones((len(x_train), 1), dtype=float)
        tx = np.hstack((ones, x_train.astype(float)))

    return tx, mask

def preprocess_unstructural(x_train):
    # Standardize
    x_mean = np.nanmean(x_train, axis=0)
    x_std = np.nanstd(x_train, axis=0)
    x_std[x_std == 0] = 1
    x_train = (x_train - x_mean) / x_std
    # Clip extreme outliers
    x_train = np.clip(x_train, -5, 5)
    return x_train

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

def knn_predict_streaming(train_X, train_y, test_X, k,
                          test_batch=1024, train_block=4096,
                          as_float32=True):
    """
    Memory-efficient exact KNN classification.
    - No (N_test x N_train x D) broadcast
    - Processes test and train in blocks
    - Uses squared distances via norms + GEMM
    """
    if as_float32:
        train_X = np.asarray(train_X, dtype=np.float32, order="C")
        test_X  = np.asarray(test_X,  dtype=np.float32, order="C")
    else:
        train_X = np.ascontiguousarray(train_X)
        test_X  = np.ascontiguousarray(test_X)

    # ensure labels are ints for bincount
    train_y = np.asarray(train_y)
    if not np.issubdtype(train_y.dtype, np.integer):
        # map arbitrary labels to 0..C-1
        uniq, inv = np.unique(train_y, return_inverse=True)
        y_int = inv.astype(np.int64)
        to_label = uniq
    else:
        y_int = train_y.astype(np.int64, copy=False)
        to_label = None

    N_train = train_X.shape[0]
    N_test  = test_X.shape[0]
    k = int(k)

    # Precompute train squared norms once
    train_norm = np.einsum('ij,ij->i', train_X, train_X)
    preds = np.empty(N_test, dtype=np.int64)

    for t0 in range(0, N_test, test_batch):
        t1 = min(t0 + test_batch, N_test)
        Xb = test_X[t0:t1]                                  # (B, D)
        B = Xb.shape[0]
        Xb_norm = np.einsum('ij,ij->i', Xb, Xb)             # (B,)

        # running top-k (best distances and indices) per test row
        best_d = np.full((B, k), np.inf, dtype=Xb.dtype)
        best_i = np.full((B, k), -1, dtype=np.int64)

        for s0 in range(0, N_train, train_block):
            s1 = min(s0 + train_block, N_train)
            Yb = train_X[s0:s1]                              # (T, D)
            Yb_norm = train_norm[s0:s1]                      # (T,)

            # squared distances via ||x||^2 + ||y||^2 - 2 x·y
            # allocates only a (B x T) matrix, no third dim
            cross = Xb @ Yb.T                                # (B, T)
            d2 = Xb_norm[:, None] + Yb_norm[None, :] - 2.0 * cross

            # local top-k per row in this block
            # (argpartition is O(n) and avoids full sort)
            if d2.shape[1] > k:
                part_idx = np.argpartition(d2, k-1, axis=1)[:, :k]
            else:
                part_idx = np.arange(d2.shape[1])[None, :].repeat(B, 0)
            local_d = np.take_along_axis(d2, part_idx, axis=1)
            local_i = part_idx + s0

            # merge current best with local best (size 2k) and keep k
            merged_d = np.concatenate([best_d, local_d], axis=1)
            merged_i = np.concatenate([best_i, local_i], axis=1)
            sel = np.argpartition(merged_d, k-1, axis=1)[:, :k]
            best_d = np.take_along_axis(merged_d, sel, axis=1)
            best_i = np.take_along_axis(merged_i, sel, axis=1)

        # vote among k neighbors
        neigh_labels = y_int[best_i]                         # (B, k)
        # bincount per row (works when labels are small non-negative ints)
        for r in range(B):
            preds[t0 + r] = np.bincount(neigh_labels[r]).argmax()

    # map back to original labels if needed
    if to_label is not None:
        return to_label[preds]
    return preds
