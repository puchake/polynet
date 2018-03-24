"""
This module contains minimal realization of a neural network will try to fit a
polynomial to the dataset.
"""


import csv
import os

import numpy as np

from files_io import load_dataset, save_vars_dict, load_vars_dict

from gradients import *

from adam_optimizer import *
from neural_network import *


HYPER_PARAMS = {"learning_rate": 0.02, "max_epochs": 30000}


def normalize_mat(mat):
    """

    Args:
        mat:

    Returns:

    """
    mat_max = np.max(np.abs(mat))
    return mat / mat_max, mat_max


def denormalize_coeffs(coeffs, x_max, y_max, polynomial_degree):
    """

    Args:
        coeffs:
        x_max:
        y_max:
        polynomial_degree:

    Returns:

    """
    return coeffs * y_max / np.power(x_max, range(polynomial_degree + 1))


NORMALIZATION_PARAMS_PATH = os.path.join("data", "normalization_params.npz")
NET_PATH = os.path.join("data", "net.npz")
MAX_EPOCHS = 30000


def train(csv_path, polynomial_degree):
    """
    Fit polynomial of polynomial_degree to the dataset pointed by csv_path.
    """
    x_mat, y_mat = load_dataset(csv_path)
    x_mat, x_max = normalize_mat(x_mat)
    y_mat, y_max = normalize_mat(y_mat)
    x_powers_mat = np.power(x_mat, range(polynomial_degree + 1))
    placeholders = {"x_mat": x_mat, "y_mat": y_mat,
                    "x_powers_mat": x_powers_mat}
    normalization_params = {"x_max": x_max, "y_max": y_max}
    save_vars_dict(normalization_params, NORMALIZATION_PARAMS_PATH)
    net = NeuralNetwork(polynomial_degree)
    optimizer = AdamOptimizer(net)
    min_loss = None
    best_net_vars = {}
    best_coeffs = None
    for i in range(MAX_EPOCHS):
        vals = net.forward_pass(placeholders)
        if min_loss is None or vals["loss"] < min_loss:
            min_loss = vals["loss"]
            best_coeffs = vals["avg_coeffs"]
            net.backup(best_net_vars)
        net.backward_pass(placeholders, vals)
        optimizer.perform_update(net)
        if i % 1000 == 0:
            print(i, vals["loss"], vals["avg_coeffs"], vals["coeffs_variance"])
    save_vars_dict(best_net_vars, NET_PATH)
    coeffs = denormalize_coeffs(vals["avg_coeffs"], x_max,
                                y_max, polynomial_degree)
    return list(reversed(coeffs.tolist()))


def estimate(x):
    """

    Args:
        x:
        net_path:

    Returns:

    """
    net_vars = load_vars_dict(NET_PATH)
    polynomial_degree = net_vars["bias_1"].shape[0] - 1
    net = NeuralNetwork.from_net_vars(net_vars)
    normalization_params = load_vars_dict(NORMALIZATION_PARAMS_PATH)
    x_mat = np.array([[x / normalization_params["x_max"]]], dtype=np.float64)
    y_mat = np.array([[0.0]])
    x_powers_mat = np.power(x_mat, range(polynomial_degree + 1))
    placeholders = {"x_mat": x_mat, "y_mat": y_mat,
                    "x_powers_mat": x_powers_mat}
    vals = net.forward_pass(placeholders)
    y = vals["y_fit"] * normalization_params["y_max"]
    return y[0][0]
