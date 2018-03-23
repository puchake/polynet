""" This module contains definition of the neural network class. """


import numpy as np

from gradients import (mean_grad, matmul_grad_of_mat_2, bias_add_grad,
                       square_loss_grad, coeffs_variance_grad,
                       element_wise_mul_grad, sum_grad)


class NeuralNetwork:

    def __init__(self, polynomial_degree):
        """
        Create NN variable matrices and holders for their gradients.

        Args:
            polynomial_degree: max polynomial degree with which the network
                should estimate points from the dataset.

        """
        self.vars = {
            "weights_1": np.random.normal(size=[1, polynomial_degree + 1])
                         * 0.001,
            "bias_1": np.zeros([polynomial_degree + 1, ])
        }
        self.grads = {key: np.zeros_like(self.vars[key]) for key in self.vars}

    def forward_pass(self, placeholders):
        """
        Perform forward pass through the network using given placeholders and
        return dictionary containing all intermediate values such as calculated
        coefficients, their average, variance, model loss etc.

        Args:
            placeholders (dict): collection of inputs to the network. It has to
                contain matrix of x values, matrix of y values and matrix of
                powers of x up to the polynomial degree which was used to create
                this network.

        Returns:
            dict: collection of matrices which were calculated during the
                forward pass. This includes matrix of coefficients, estimated
                ys, coefficients averages and variances, square loss of
                estimated ys and sum of the mean square loss and mean variances
                of coefficients..

        """
        vals = dict()
        vals["coeffs"] = (
            np.matmul(placeholders["x_mat"], self.vars["weights_1"])
            + self.vars["bias_1"]
        )
        vals["avg_coeffs"] = np.mean(vals["coeffs"], axis=0)
        vals["coeffs_variance"] = np.var(vals["coeffs"], axis=0)
        vals["y_fit_components"] = np.multiply(placeholders["x_powers_mat"],
                                               vals["coeffs"])
        vals["y_fit"] = np.sum(vals["y_fit_components"], axis=1, keepdims=True)
        vals["square_losses"] = np.power(vals["y_fit"] - placeholders["y_mat"],
                                         2)
        vals["loss"] = np.mean(vals["square_losses"]) + np.mean(
            vals["coeffs_variance"]
        )
        return vals

    def backward_pass(self, placeholders, vals):
        """
        Perform backward pass through the network and calculate gradients of
        the main loss with respect to the NN variables.

        Args:
            placeholders (dict): collection of inputs to the network. It has to
                contain matrix of x values, matrix of y values and matrix of
                powers of x up to the polynomial degree which was used to create
                this network.
            vals (dict): collection of intermediate values calculated during the
                forward pass.

        Returns:
            None

        """
        grad_of_mean_square_loss = 1
        grad_of_variance_loss = 1
        grad_of_square_losses = mean_grad(vals["square_losses"],
                                          grad_of_mean_square_loss)
        grad_of_coeffs_variance = mean_grad(vals["coeffs_variance"],
                                            grad_of_variance_loss)
        grad_of_coeffs_variance_loss = coeffs_variance_grad(
            vals["coeffs"], grad_of_coeffs_variance
        )
        grad_of_y_fit = square_loss_grad(vals["y_fit"], placeholders["y_mat"],
                                         grad_of_square_losses)
        grad_of_y_fit_components = sum_grad(vals["y_fit_components"],
                                            grad_of_y_fit)
        grad_of_coeffs_mean_square_loss = element_wise_mul_grad(
            placeholders["x_powers_mat"], grad_of_y_fit_components
        )
        grad_of_coeffs = (grad_of_coeffs_mean_square_loss
                          + grad_of_coeffs_variance_loss)
        grad_of_matmul, self.grads["bias_1"] = bias_add_grad(grad_of_coeffs)
        self.grads["weights_1"] = matmul_grad_of_mat_2(placeholders["x_mat"],
                                                       grad_of_matmul)
