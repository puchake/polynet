"""
This module contains minimal realization of a neural network will try to fit a
polynomial to the dataset.
"""


import csv

import numpy as np


HYPER_PARAMS = {"learning_rate": 0.01, "max_epochs": 10000}


def load_dataset(csv_path):
    """ Load dataset of points from the .csv file pointed by csv_path. """
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        x_list = []
        y_list = []
        for (x_str, y_str) in csv_reader:
            x_list.append(float(x_str))
            y_list.append(float(y_str))

        # Create matrices from x_list and y_list with one example per row.
        x_mat = np.array(x_list).reshape([-1, 1])
        y_mat = np.array(y_list).reshape([-1, 1])

        return x_mat, y_mat


def init_net_vars(polynomial_degree):
    """ Create dict which contains all NN variables. """
    net_vars = {"weights_1": np.random.normal(size=[1, polynomial_degree + 1]),
                "bias_1": np.zeros([polynomial_degree + 1, ])}
    return net_vars


def forward_pass(x_mat, y_mat, x_powers_mat, net_vars):
    """
    Perform forward pass with provided net_vars and return loss, average
    coefficients, their standard deviation and other indirect values in a dict.
    """
    vals = {}
    vals["coeffs"] = (np.matmul(x_mat, net_vars["weights_1"])
                      + net_vars["bias_1"])
    vals["avg_coeffs"] = np.mean(vals["coeffs"], axis=0)
    vals["std_coeffs"] = np.std(vals["coeffs"], axis=0)
    vals["y_fit"] = np.sum(np.multiply(x_powers_mat, vals["coeffs"]), axis=1,
                           keepdims=True)
    # TODO: add deviation of coefficients to the loss
    square_losses = np.power(vals["y_fit"] - y_mat, 2)
    vals["loss"] = np.mean(square_losses)
    return vals


def backward_pass(learning_rate, net_vars, x_mat, x_powers_mat, y_mat, vals):
    """ Perform backward pass and update net_vars. """
    # TODO: add deviation of coefficients to the gradient
    dloss_dmean = vals["loss"]
    dmean_dsquare_losses = np.ones_like(y_mat) / y_mat.shape[0]
    dloss_dsquare_losses = dloss_dmean * dmean_dsquare_losses
    dsquare_losses_dy_fit = 2 * (vals["y_fit"] - y_mat)
    dloss_dy_fit = np.multiply(dloss_dsquare_losses, dsquare_losses_dy_fit)
    dy_fit_dcoeffs = x_powers_mat
    dloss_dcoeffs = dloss_dy_fit * dy_fit_dcoeffs
    dloss_dbias_1 = np.sum(dloss_dcoeffs, axis=0)
    dloss_dweights_1 = np.matmul(x_mat.T, dloss_dcoeffs)
    net_vars["weights_1"] -= learning_rate * dloss_dweights_1
    net_vars["bias_1"] -= learning_rate * dloss_dbias_1


def fit_polynomial(csv_path, polynomial_degree):
    """
    Fit polynomial of polynomial_degree to the dataset pointed by csv_path.
    """
    x_mat, y_mat = load_dataset(csv_path)

    # Create matrix of x powers up to x^polynomial_degree.
    x_powers_mat = np.power(x_mat, range(polynomial_degree + 1))

    # Normalize data matrices.
    x_mat = (x_mat - np.mean(x_mat)) / np.std(x_mat)
    y_mat = (y_mat - np.mean(y_mat)) / np.std(y_mat)
    x_powers_mat[:, 1:] = (
        (x_powers_mat[:, 1:] - np.mean(x_powers_mat[:, 1:], axis=0))
        / np.std(x_powers_mat[:, 1:], axis=0)
    )

    net_vars = init_net_vars(polynomial_degree)
    for i in range(HYPER_PARAMS["max_epochs"]):
        vals = forward_pass(x_mat, y_mat, x_powers_mat, net_vars)
        backward_pass(HYPER_PARAMS["learning_rate"], net_vars,
                      x_mat, x_powers_mat, y_mat, vals)
        if i % 1000 == 0:
            print(vals["loss"], vals["avg_coeffs"], vals["std_coeffs"])

    # TODO: rescale coefficients with normalization params.
    return vals["avg_coeffs"]
