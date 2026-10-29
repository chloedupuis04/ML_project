import numpy as np
from helpers import *

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm."""
    w = initial_w.copy()
    loss = compute_mse(y, tx, w)
    
    for _ in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient
        loss = compute_mse(y, tx, w)
        
    return w, loss
 

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD)."""
    w = initial_w.copy()
    loss = compute_mse(y, tx, w)
    
    for _ in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            gradient = compute_gradient(y_batch, tx_batch, w)
            w = w - gamma * gradient
            loss = compute_mse(y, tx, w)
            
    return w, loss


def least_squares(y, tx):
    """Calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = compute_mse(y, tx, w)
    return w, mse


def ridge_regression(y, tx, lambda_):
    """Implement ridge regression."""
    aI = 2 * lambda_ * tx.shape[0] * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent."""
    w = initial_w.copy()
    loss = calculate_loss(y, tx, w)
    
    for _ in range(max_iters):
        gradient = calculate_gradient(y, tx, w)
        w = w - gamma * gradient
        loss = calculate_loss(y, tx, w)

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent."""
    w = initial_w.copy()
    loss = calculate_loss(y, tx, w)
    
    for _ in range(max_iters):
        _, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        loss = calculate_loss(y, tx, w)

    return w, loss