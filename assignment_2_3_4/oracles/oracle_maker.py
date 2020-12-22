from assignment_2_3_4.oracles import AbstractOracle, LogisticRegressionOracle
from assignment_2_3_4.data_readers import read_libsvm, read_tsv


def make_oracle(
    data_path: str, data_format: str = "libsvm", oracle_name: str = "logistic regression"
) -> AbstractOracle:
    data_readers = {"libsvm": read_libsvm, "tsv": read_tsv}
    if data_format not in data_readers:
        raise ValueError(f"Unknown format {data_format}")
    x_train, y_train = data_readers[data_format](data_path)

    oracles = {LogisticRegressionOracle.name: LogisticRegressionOracle}
    if oracle_name not in oracles:
        raise ValueError(f"Unknown oracle {oracle_name}")
    return oracles[oracle_name](x_train, y_train)
