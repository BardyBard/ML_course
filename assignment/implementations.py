import numpy as np


def compute_loss(y, tx, w):
    """Calculate the loss using MSE.

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

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    # print(f"compute_gradient got y = {y}, tx = {tx}, w = {w}")

    N = len(y)
    y_pred = tx @ w
    error = y_pred - y
    return tx.T @ error / N
    

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
        grad = compute_gradient(y, tx, w)
        # print("grad is", grad)
        w = w - gamma * grad
    loss = compute_loss(y, tx, w)
    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent.

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
    N = len(tx)
    for _ in range(max_iters):
        random_index = np.random.randint(N)
        sampled_y = np.array([y[random_index]])
        sampled_tx = tx[random_index:random_index+1]
        grad = compute_gradient(sampled_y, sampled_tx, w)
        w = w - gamma * grad
    loss = compute_loss(y, tx, w)
    return w, loss
        
        

def least_squares(y, tx):
    """Least squares regression using normal equations.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        
    Returns:
        (w, loss): tuple of numpy array of shape (2,) w last weight and float.
    """
    # Proceed carefully: naive impletentaion is ill-conditioned.  
    # Let np.linalg do the hard work. -M
    # 
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)  
    loss = compute_loss(y, tx, w)
    return w, loss