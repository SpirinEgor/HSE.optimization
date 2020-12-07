from os.path import exists
from typing import Tuple, Union

import numpy
from scipy.sparse import hstack, csr_matrix
from sklearn.datasets import load_svmlight_file


def add_bias_column(data: Union[numpy.ndarray, csr_matrix]) -> Union[numpy.ndarray, csr_matrix]:
    bias = numpy.ones((data.shape[0], 1), dtype=data.dtype)
    if isinstance(data, numpy.ndarray):
        return numpy.append(data, bias, axis=1)
    else:
        return hstack((data, bias))


def labels_to_zero_one_format(labels: numpy.ndarray) -> numpy.ndarray:
    unique_labels = numpy.unique(labels)
    if unique_labels.shape[0] != 2:
        raise RuntimeError(f"Can't compress labels to 0 and 1, find {unique_labels} unique labels")
    zero_mask = labels == unique_labels[0]
    labels[zero_mask] = 0
    labels[~zero_mask] = 1
    return labels.astype(numpy.short)


def read_libsvm(data_path: str) -> Tuple[csr_matrix, numpy.ndarray]:
    if not exists(data_path):
        raise ValueError(f"{data_path} does not exist")
    x, y = load_svmlight_file(data_path)
    x = csr_matrix(x)
    x = add_bias_column(x)
    y = labels_to_zero_one_format(y)
    return x, y


def read_tsv(data_path: str) -> Tuple[numpy.ndarray, numpy.ndarray]:
    if not exists(data_path):
        raise ValueError(f"{data_path} does not exist")
    data = numpy.loadtxt(data_path, delimiter="\t")
    x, y = data[:, 1:], data[:, 0]
    x = add_bias_column(x)
    y = labels_to_zero_one_format(y)
    return x, y
