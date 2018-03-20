""" This module contains NN saving and loading along with dataset loading. """


import csv
import os

import numpy as np


def save_net(net_vars, net_path):
    """
    Save neural net variables dictionary to an .npz archive.

    Args:
        net_vars (dict): dictionary which contains NN variables - np.ndarrays.
        net_path (str): path to the .npz archive in which variables will be
            stored.

    Returns:
        None

    Raises:
        OSError: If directory part of net_path could not be created.

    """
    dir_path, _ = os.path.split(net_path)
    if dir_path != "" and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    np.savez(net_path, **net_vars)


def load_net(net_path):
    """
    Load network's variables contained in an .npz archive and return them
    wrapped in a dictionary.

    Args:
        net_path (str): path to the .npz archive from which variables will be
            loaded.

    Returns:
        dict: collection of NN variables where dictionary key is a variable
            label and value associated with this key is an numpy array.

    Raises:
        IOError: If the net_path does not point to any existing file, or this
            file can't be read.
        OSError: If the loaded file cannot be interpreted as .npz archive or
            .npy file.
        ValueError: If the loaded file was single numpy array in .npy format
            and cannot be converted to a dict of NN's variables.

    """
    net_vars_npz = np.load(net_path)
    if isinstance(net_vars_npz, np.ndarray):
        raise ValueError("Loaded file is a single numpy array instead of npz "
                         "archive.")
    net_vars = {key: net_vars_npz[key] for key in net_vars_npz.files}
    return net_vars


def load_dataset(csv_path):
    """
    Load dataset of points from the .csv file and return it as pair of matrices.

    Args:
        csv_path (str): path to the .csv file which contains dataset.

    Returns:
        (np.ndarray, np.ndarray): pair of 2d matrices of shape = [num_rows, 1]
            which separately contain x and y values of points from the dataset.
            Matrix containing x values is first in the tuple.

    Raises:
        IOError: If file pointed by csv_path does not exists.
        ValueError: If the .csv dataset file is ill-formatted.

    """
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        x_list = []
        y_list = []
        for i, row in enumerate(csv_reader):
            try:
                x_str, y_str = row
                x_list.append(float(x_str))
                y_list.append(float(y_str))
            except ValueError:
                raise ValueError("Row {} in the .csv file is ill-formatted. It "
                                 "should contain two float values separated by "
                                 "a comma.".format(i))

        # Create matrices from x_list and y_list with one example per row.
        x_mat = np.array(x_list).reshape([-1, 1])
        y_mat = np.array(y_list).reshape([-1, 1])

        return x_mat, y_mat
