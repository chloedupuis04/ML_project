"""Some helper functions for project 1."""

import csv
import numpy as np
import os


def load_csv_data(data_path, sub_sample=False):
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids


def create_csv_submission(ids, y_pred, name):
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def compute_gradient(y, tx, w):
    e = y - tx.dot(w)
    N = len(y) 
    gradient = -1/N * tx.T.dot(e)
    return gradient


def compute_mse(y, tx, w):
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    data_size = len(y)  
    batch_size = min(data_size, batch_size) 
    max_batches = int(data_size / batch_size)
    remainder = data_size - max_batches * batch_size

    if shuffle:
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start
        end_index = start_index + batch_size
        yield y[start_index:end_index], tx[start_index:end_index]


def sigmoid(t):
    t = np.clip(t, -250, 250)
    e = np.exp(-t)
    sig = 1/(1+e)
    return sig


def calculate_loss(y, tx, w):
    assert y.shape[0] == tx.shape[0]
    N = y.shape[0]
    sigma = sigmoid(np.dot(tx, w))
    eps = 1e-15
    loss = -(1/N) * (np.dot(y.T, np.log(sigma + eps)) + np.dot((1-y).T, np.log(1 - sigma + eps)))
    return np.array(np.squeeze(loss))


def calculate_gradient(y, tx, w):
    N = y.shape[0]
    sigma = sigmoid(np.dot(tx, w))    
    gradient = 1/N * (tx.T.dot(sigma-y))
    return gradient


def calculate_hessian(y, tx, w):
    N = y.shape[0]
    sigma = sigmoid(tx @ w)
    s = sigma * (1 - sigma)
    hessian = (tx.T @ (s[:, np.newaxis] * tx)) / N
    return hessian


def penalized_logistic_regression(y, tx, w, lambda_):
    loss = calculate_loss(y, tx, w) 
    gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, gradient


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w -= gamma * gradient
    return loss, w