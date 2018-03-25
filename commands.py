"""
This module contains realization of application commands along with some minor
functions which help in training and estimation.
"""


import os

import numpy as np

from files_io import load_dataset, save_vars_dict, load_vars_dict
from adam_optimizer import AdamOptimizer
from neural_network import NeuralNetwork


# Constant paths to NN variables and normalization parameters saved on the disk.
# These paths are relative to the place in which main script was launched.
NORMALIZATION_PARAMS_PATH = os.path.join("data", "normalization_params.npz")
NET_PATH = os.path.join("data", "net.npz")

# Training hyperparameters. These were found through trial and error.
# Training stops when that many iterations have passed.
MAX_ITERATIONS = 25000
# During the training there are attempts made to eliminate some coefficients of
# the modeled polynomial. These attempts are made after current loss falls
# below ELIMINATION_ATTEMPT_THRESHOLD value.
ELIMINATION_ATTEMPT_THRESHOLD = 1e-5
# Selected coefficient is eliminated from modeled polynomial when mean of
# absolute values of this coefficient in the current iteration is less than the
# COEFF_ELIMINATION_THRESHOLD value.
COEFF_ELIMINATION_THRESHOLD = 1e-5
# Training stops when this or smaller loss is reached.
TARGET_LOSS = 1e-25


def normalize_mat(mat):
    """
    Normalize given matrix of values and bring all of them to max range of
    <-1, 1>.

    Args:
        mat (np.ndarray): matrix of values that will be normalized

    Returns:
        (np.ndarray, float): first returned value is given matrix after
            normalization. Second returned value is maximal absolute value of
            this matrix before normalization.

    """
    mat_max = np.max(np.abs(mat))
    return mat / mat_max, mat_max


def denormalize_coeffs(coeffs, x_max, y_max, polynomial_degree):
    """
    Denormalize coefficients approximated by the NN to eliminate effect of
    normalization of x and y values. Return denormalized coefficients.

    Args:
        coeffs (np.ndarray): matrix of normalized coefficients calculated by NN.
        x_max (float): maximal value of x before normalization.
        y_max (float): maximal value of y before normalization.
        polynomial_degree (int): degree of the approximated polynomial.

    Returns:
        np.ndarray: matrix of denormalized coefficients.

    """
    return coeffs * y_max / np.power(x_max, range(polynomial_degree + 1))


def try_eliminating_coeffs(placeholders, normalization_params, vals, net,
                           optimizer, polynomial_degree):
    """
    Try to eliminate some coefficients from approximated polynomial.
    Coefficients are eliminated when mean of their absolute values from current
    iteration is less or equal than COEFF_ELIMINATION_THRESHOLD value. Start at
    the coefficient standing next to the biggest power of x and stop at the
    first coefficient that is not going to be eliminated. Return degree of the
    new polynomial after elimination.

    Args:
        placeholders (dict): collection of inputs to the NN which includes
            matrix of x values, matrix of y values and matrix of x powers.
        normalization_params (dict): dictionary which contains maximal x and y
            values.
        vals (dict): collection of values such as coefficients averages etc.
            computed during NN full forward pass.
        net (NeuralNetwork): NN object which approximates the polynomial.
        optimizer (AdamOptimizer): optimizer object which updates the NN during
            training.
        polynomial_degree (int): current degree of the modeled polynomial.

    Returns:
        int: degree of the modified polynomial. It might be the same as
            polynomial degree on input if no coefficient was eliminated.

    """
    denormalized_coeffs = denormalize_coeffs(
        vals["coeffs"], normalization_params["x_max"],
        normalization_params["y_max"], polynomial_degree
    )
    denormalized_coeffs_abs_avg = np.mean(np.abs(denormalized_coeffs), axis=0)

    # Try to eliminate coefficient standing next to the currently highest x
    # power in the modeled polynomial. Stop if elimination condition is not met
    # or if we try to eliminate last coefficient.
    while (
        denormalized_coeffs_abs_avg[polynomial_degree]
        < COEFF_ELIMINATION_THRESHOLD
        and polynomial_degree != 0
    ):
        net.eliminate_polynomial_degree(polynomial_degree)
        optimizer.eliminate_polynomial_degree(polynomial_degree)

        # Delete x powers corresponding to the eliminated coefficient from the
        # x_powers_mat inside the placeholders dict.
        x_powers_mask = np.ones_like(placeholders["x_powers_mat"][0, :],
                                     dtype=bool)
        x_powers_mask[polynomial_degree] = False
        placeholders["x_powers_mat"] = (placeholders["x_powers_mat"]
                                        [:, x_powers_mask])

        polynomial_degree -= 1
    return polynomial_degree


def train(csv_path, polynomial_degree):
    """
    Train NN to model points contained in the pointed dataset with a polynomial
    of a degree less than or equal to the given degree. Return coefficients of
    found polynomial.

    Args:
        csv_path (str): path to a .csv file which contains dataset of points.
        polynomial_degree (int): maximal degree of the modeled polynomial.

    Returns:
        list: list of coefficients of found polynomial. Most significant
            coefficient is first on this list.

    """
    x_mat, y_mat = load_dataset(csv_path)
    x_mat, x_max = normalize_mat(x_mat)
    y_mat, y_max = normalize_mat(y_mat)
    x_powers_mat = np.power(x_mat, range(polynomial_degree + 1))
    placeholders = {"x_mat": x_mat, "y_mat": y_mat,
                    "x_powers_mat": x_powers_mat}

    # Save maximal x and y values as they will be needed during estimation.
    normalization_params = {"x_max": x_max, "y_max": y_max}
    save_vars_dict(normalization_params, NORMALIZATION_PARAMS_PATH)

    net = NeuralNetwork(polynomial_degree)
    optimizer = AdamOptimizer(net)
    min_loss = None
    best_net_vars = {}
    best_coeffs = None
    for i in range(MAX_ITERATIONS):
        vals = net.forward_pass(placeholders)
        vals = net.loss_forward_pass(placeholders, vals)

        # Keep track of the best network approximation so far.
        if min_loss is None or vals["loss"] < min_loss:
            min_loss = vals["loss"]
            best_coeffs = vals["avg_coeffs"]
            net.backup(best_net_vars)
            if min_loss <= TARGET_LOSS:
                break

        net.backward_pass(placeholders, vals)

        # Attempt coefficients elimination after loss hits certain threshold.
        # The idea is to let the coefficients settle a bit to find out which
        # ones of them are truly unnecessary.
        if vals["loss"] < ELIMINATION_ATTEMPT_THRESHOLD:
            polynomial_degree = try_eliminating_coeffs(
                placeholders, normalization_params, vals, net, optimizer,
                polynomial_degree
            )
            best_net_vars = {}
            net.backup(best_net_vars)

        optimizer.perform_update(net)
    save_vars_dict(best_net_vars, NET_PATH)
    coeffs = denormalize_coeffs(best_coeffs, x_max,
                                y_max, polynomial_degree)
    return list(reversed(coeffs.tolist()))


def estimate(x):
    """
    Estimate y value for given x using previously trained NN to approximate
    this polynomial coefficients.

    Args:
        x (float): x value which will be used to calculate y with approximated
            polynomial coefficients.

    Returns:
        float: y value estimated for given x.

    """
    net_vars = load_vars_dict(NET_PATH)
    polynomial_degree = net_vars["bias_1"].shape[0] - 1
    net = NeuralNetwork.from_net_vars(net_vars)
    normalization_params = load_vars_dict(NORMALIZATION_PARAMS_PATH)
    x_powers_mat = np.power(x, range(polynomial_degree + 1))

    # Normalize input to NN as it was not trained to handle raw x values and
    # unnormalized x might skew the output coefficients.
    x_mat = np.array([[x / normalization_params["x_max"]]], dtype=np.float64)

    placeholders = {"x_mat": x_mat}
    vals = net.forward_pass(placeholders)
    coeffs = denormalize_coeffs(vals["coeffs"], normalization_params["x_max"],
                                normalization_params["y_max"],
                                polynomial_degree)
    y = np.sum(np.multiply(x_powers_mat, coeffs))
    return y
