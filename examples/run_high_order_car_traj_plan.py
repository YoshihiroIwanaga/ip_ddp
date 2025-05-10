import math

import numpy as np

from ip_ddp.InteriorPointDdpSolver import InteriorPointDdpSolver
from ip_ddp.utils.OptimalControlProblem import OptimalControlProblem
from ip_ddp.utils.TrajInfo import TrajInfo
from problem_setting.high_order_car.CarConfig import CarConfig
from problem_setting.high_order_car.CarConstraint import CarConstraint
from problem_setting.high_order_car.CarCost import CarCost
from problem_setting.high_order_car.CarDynamics import CarDynamics
from utils.plot_traj_lib import plot_us, plot_xs

if __name__ == "__main__":
    x_ref = np.array([1.0, 20.0, math.pi / 2, 0.0, 0.0, 0.0, 0.0])
    cfg = CarConfig(x_ref=x_ref)
    dynamics = CarDynamics(cfg)
    const = CarConstraint(cfg)
    traj_info = TrajInfo(cfg, const.lc, const.lc_terminal, cfg.step_size_num)
    cost = CarCost(cfg)
    ocp = OptimalControlProblem(dynamics, const, cost, cfg)
    solver = InteriorPointDdpSolver(ocp, traj_info, cfg)
    solver.prints_info = True
    solver.run()
    xs_opt, us_opt = solver.get_optimal_traj()

    plot_xs(
        xs_opt,
        cfg.x_ref,
        cfg.x_min,
        cfg.x_max,
        cfg.horizon,
    )

    plot_us(
        us_opt,
        cfg.u_min,
        cfg.u_max,
        cfg.horizon,
    )
