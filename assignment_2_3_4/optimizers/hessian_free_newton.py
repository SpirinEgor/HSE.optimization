import numpy

from assignment_2_3_4.optimizers import AbstractOptimizer, OptimizationStep
from assignment_2_3_4.oracles import AbstractOracle


class HessianFreeNewtonOptimizer(AbstractOptimizer):
    name: str = "hessian free newton"

    def _get_direction(self, oracle: AbstractOracle, last_point: OptimizationStep) -> numpy.ndarray:
        x = -last_point.grad
        b = -last_point.grad
        g_k = oracle.hessian_vec_product(last_point.point, x) - b
        g_t_g = (g_k * g_k).sum()
        d_k = -g_k
        for _ in range(x.shape[0]):
            if numpy.linalg.norm(g_k) < self._config.tol:
                break
            a_d_k = oracle.hessian_vec_product(last_point.point, d_k)
            alpha = g_t_g / (d_k * a_d_k).sum()
            x += alpha * d_k
            g_next = g_k + alpha * a_d_k
            g_next_t_g_next = (g_next * g_next).sum()
            beta = g_next_t_g_next / g_t_g
            d_k = -g_next + beta * d_k
            g_k, g_t_g = g_next, g_next_t_g_next
        return x

    def _aggregate_optimization_step(
        self,
        oracle: AbstractOracle,
        new_point: numpy.ndarray,
        passed_time: float,
    ) -> OptimizationStep:
        new_value, new_grad, new_hessian = oracle.fuse_value_grad_hessian(new_point)
        return OptimizationStep(new_point, new_value, new_grad, passed_time, oracle.n_calls, hessian=new_hessian)
