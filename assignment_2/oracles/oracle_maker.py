from assignment_2.oracles import AbstractOracle, LogisticRegressionOracle
from assignment_2.utils import read_libsvm, read_tsv


def make_oracle(data_path: str, data_format: str = "libsvm") -> AbstractOracle:
    if data_format == "libsvm":
        x_train, y_train = read_libsvm(data_path)
    elif data_format == "tsv":
        x_train, y_train = read_tsv(data_path)
    else:
        raise ValueError(f"Unknown format {data_format}")

    return LogisticRegressionOracle(x_train, y_train)
