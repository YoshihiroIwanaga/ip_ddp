import math

import numpy as np

from problem_setting.abstract.Config import Config


class CarConfig(Config):
    def __init__(
        self,
        x_ini: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        x_ref: np.ndarray = np.array([6.0, 6.0, math.pi / 2, 0.0, 0.0]),
        horizon: int = 500,
        dt: float = 0.015,
    ) -> None:
        self.n: int = 5  # state simension
        self.m: int = 2  # control input dimension
        self.x_ini: np.ndarray = x_ini
        self.x_ref: np.ndarray = x_ref
        self.dt: float = dt
        self.horizon: int = horizon
        self.Q: np.ndarray = np.diag(
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
        )
        self.R: np.ndarray = np.diag(
            np.array(
                [
                    0.1,
                    0.1 / math.radians(60) / math.radians(60),
                ]
            )
        )
        self.Q_terminal: np.ndarray = np.diag(
            np.array(
                [
                    250.0,
                    250.0,
                    250.0 / math.radians(10) / math.radians(10),
                    250.0,
                    250.0 / math.radians(20) / math.radians(20),
                ]
            )
        )
        self.Q_ini = self.Q

        self.x_max = np.array(
            [
                [np.inf],
                [np.inf],
                [np.inf],
                [40 * 1000 / 3600],
                [math.radians(40)],
            ]
        )
        self.x_min = np.array(
            [
                [-np.inf],
                [-np.inf],
                [-np.inf],
                [-40 * 1000 / 3600],
                [-math.radians(40)],
            ]
        )
        self.u_max = np.array([[5.0], [math.radians(60)]])
        self.u_min = np.array([[-5.0], [-math.radians(60)]])
        self.idx_x_min = [3, 4]
        self.idx_x_max = [3, 4]
        self.idx_u_min = [0, 1]
        self.idx_u_max = [0, 1]

        self.free_state_idx = []
        self.terminal_const_state_idx = []
        self.tol = 1e-6
        self.max_iter: int = 1000
        self.run_ddp: bool = True
        self.automatic_initilization_barrier_param_is_enabled: bool = True
        self.barrier_param_ini: float = 0.001
        self.step_size_num: int = 21
        self.cost_change_ratio_convergence_criteria: float = 0.99
        self.u_inis = np.zeros((self.step_size_num, self.m, self.horizon))
        self.knows_optimal_cost: bool = False
        self.optimal_cost: float = 0.0
