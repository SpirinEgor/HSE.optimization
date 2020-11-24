from os.path import exists
from typing import Tuple

import numpy
from sklearn.datasets import load_svmlight_file


def labels_to_zero_one_format(labels: numpy.ndarray) -> numpy.ndarray:
    unique_labels = numpy.unique(labels)
    if unique_labels.shape != 2:
        raise RuntimeError(f"Can't compress labels to 0 and 1, find {unique_labels} unique labels")
    zero_mask = labels == unique_labels[0]
    labels[zero_mask] = 0
    labels[~zero_mask] = 1
    return labels


def add_bias_column(data: numpy.ndarray) -> numpy.ndarray:
    new_data = numpy.ones((data.shape[0], data.shape[1] + 1), dtype=data.dtype)
    new_data[:, :-1] = data
    return new_data


def read_libsvm(data_path: str, zero_one_label: bool = True, add_bias: bool = True) -> Tuple[numpy.ndarray, numpy.ndarray]:
    if not exists(data_path):
        raise ValueError(f"{data_path} does not exist")
    x, y = load_svmlight_file(data_path)
    if zero_one_label:
        y = labels_to_zero_one_format(y)
    if add_bias:
        x = add_bias_column(x)
    return x, y


def read_tsv(data_path: str, zero_one_label: bool = True, add_bias: bool = True) -> Tuple[numpy.ndarray, numpy.ndarray]:
    if not exists(data_path):
        raise ValueError(f"{data_path} does not exist")
    data = numpy.loadtxt(data_path, delimiter="\t")
    x, y = data[:, 1:], data[:, 0]
    if zero_one_label:
        y = labels_to_zero_one_format(y)
    if add_bias:
        x = add_bias_column(x)
    return x, y
