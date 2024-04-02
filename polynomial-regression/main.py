import itertools
import json
import math
import os
import numpy as np
from numpy import ndarray
from tqdm import tqdm
from typing import Tuple

# create folders
os.makedirs('./output', exist_ok=True)

# list of variables
N = [2, 5, 10, 20, 50, 100, 200]  # sample size
degrees = [1, 2, 4, 8, 16, 32, 64]  # model complexity
sd = [0.05, 0.2]  # noise level
list_variables = list(itertools.product(N, degrees, sd))


def get_data(n: int, std: float) -> np.ndarray:
    mean_z = 0
    x = np.random.uniform(0, 1, n)
    z = np.random.normal(mean_z, std, n)
    y = np.cos(2 * math.pi * x) + z
    return np.array(list((zip(x, y))))


def get_mse(y: np.ndarray, y_hat: np.ndarray) -> ndarray:
    mse = np.mean((y - y_hat) ** 2)
    return mse


def fit_data(data: np.ndarray, degree: int, var: float, reg: bool) -> Tuple[ndarray, ndarray, ndarray]:
    x = data[:, 0].reshape(len(data), 1)
    y = data[:, 1].reshape(len(data), 1)
    x_matrix = _get_matrix(x, degree)

    # training
    theta_initial = np.random.random([degree + 1, 1])
    E, theta = gradient_descent(x_matrix, y, theta_initial, reg)
    y_hat = x_matrix.dot(theta)
    E_in = get_mse(y, y_hat)

    # testing
    test_data = get_data(2000, var)
    x_test = test_data[:, 0].reshape(len(test_data), 1)
    y_test = test_data[:, 1].reshape(len(test_data), 1)
    x_test_matrix = _get_matrix(x_test, degree)
    y_test_hat = x_test_matrix.dot(theta)
    E_out = get_mse(y_test, y_test_hat)

    return theta, E_in, E_out


def gradient_descent(x: np.ndarray, y: np.ndarray, theta: np.ndarray, reg: bool) -> Tuple[list, np.ndarray]:
    lr = 0.001  # learning rate
    lmb = 0.1 if reg else 0  # weight decay
    n_iter = 5000  # epochs

    error_list = []
    for i in range(n_iter):
        theta = (1 - lr * lmb) * theta + lr * 2 / len(x) * x.T.dot(y - x.dot(theta))
        y_hat = x.dot(theta)
        loss = get_mse(y, y_hat)
        error_list.append(loss)
    return error_list, theta


def experiment(n: int, d: int, std: float, reg: bool) -> Tuple[ndarray, ndarray]:
    E_in_list = []
    E_out_list = []
    for i in range(50):
        train_data = get_data(n, std)
        theta, E_in, E_out = fit_data(train_data, d, std, reg)
        E_in_list.append(E_in)
        E_out_list.append(E_out)
    return np.mean(E_in_list), np.mean(E_out_list)


def _get_matrix(x: np.ndarray, d: int) -> np.ndarray:
    """
    matrix of [x ** 0, ..., x ** d]
    """
    X = np.zeros([len(x), d + 1])
    for i in range(len(x)):
        X[i] = [np.array(x[i][0] ** j).T for j in range(d + 1)]
    return X


def save_json(path, res):
    json_object = json.dumps(res)
    with open(path, "w") as outfile:
        outfile.write(json_object)


if __name__ == '__main__':
    # non regularized
    print('=== Non Regularized ===')
    np.random.seed(42)
    results = {}
    for variables in tqdm(list_variables):
        _E_in, _E_out = experiment(variables[0], variables[1], variables[2], reg=False)
        results[str(variables)] = [_E_in, _E_out]
    save_json('output/non_regularized.json', results)

    # regularized
    print('=== Regularized ===')
    np.random.seed(42)
    results = {}
    for variables in tqdm(list_variables):
        _E_in, _E_out = experiment(variables[0], variables[1], variables[2], reg=True)
        results[str(variables)] = [_E_in, _E_out]
    save_json('output/regularized.json', results)
