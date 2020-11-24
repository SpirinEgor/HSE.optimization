import unittest
from abc import ABC

import numpy

from assignment_2.oracles import LogisticRegressionOracle, AbstractOracle


class TestOracleDerivatives(unittest.TestCase, ABC):

    N_TRIALS = 10
    N_SAMPLES = 500
    N_FEATURES = 15
    SEED = 7
    ATOL = 1e-7

    oracles: AbstractOracle

    def setUp(self):
        numpy.random.seed(self.SEED)
        x = numpy.random.randn(self.N_SAMPLES, self.N_FEATURES)
        y = numpy.random.randint(0, high=2, size=(self.N_SAMPLES,))
        self.oracle = LogisticRegressionOracle(x, y)

    def test_gradient_calculation(self):
        w = numpy.random.randn(self.N_FEATURES)
        eps_machine = numpy.finfo(w.dtype).eps
        for i in range(self.N_TRIALS):
            with self.subTest(i=i):
                d = numpy.random.rand(self.N_FEATURES)
                eps = numpy.sqrt(eps_machine) * (1 + numpy.linalg.norm(w)) / numpy.linalg.norm(d)

                gradient = d @ self.oracle.grad(w)
                approx_value = (self.oracle.value(w + eps * d) - self.oracle.value(w - eps * d)) / (2 * eps)

                numpy.testing.assert_allclose(approx_value, gradient[0], atol=self.ATOL)

    def test_hessian_calculation(self):
        w = numpy.random.randn(self.N_FEATURES)
        eps_machine = numpy.finfo(w.dtype).eps
        for i in range(self.N_TRIALS):
            with self.subTest(i=i):
                d = numpy.random.rand(self.N_FEATURES)
                eps = numpy.sqrt(eps_machine) * (1 + numpy.linalg.norm(w)) / numpy.linalg.norm(d)

                hessian_vec_products = self.oracle.hessian_vec_product(w, d)
                approx_value = (self.oracle.grad(w + eps * d) - self.oracle.grad(w - eps * d)) / (2 * eps)

                numpy.testing.assert_allclose(approx_value, hessian_vec_products, atol=self.ATOL)


if __name__ == '__main__':
    unittest.main()
