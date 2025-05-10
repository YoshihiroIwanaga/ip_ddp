import math

import numpy as np

from ip_ddp.InteriorPointDdpSolver import InteriorPointDdpSolver
from ip_ddp.utils.OptimalControlProblem import OptimalControlProblem
from ip_ddp.utils.TrajInfo import TrajInfo
from problem_setting.car.CarConfig import CarConfig
from problem_setting.car.CarConstraint import CarConstraint
from problem_setting.car.CarCost import CarCost
from problem_setting.car.CarDynamics import CarDynamics
from problem_setting.car.utils.plot_car_lib import create_gif
from utils.plot_traj_lib import plot_us, plot_xs

if __name__ == "__main__":
    x_ref = np.array([5.0, -1.0, math.pi / 2, 0.0, 0.0])
    cfg = CarConfig(x_ref=x_ref)
    dynamics = CarDynamics(cfg)
    const = CarConstraint(cfg)
    traj_info = TrajInfo(cfg, const.lc, const.lc_terminal, cfg.step_size_num)
    cost = CarCost(cfg)
    ocp = OptimalControlProblem(dynamics, const, cost, cfg)
    solver = InteriorPointDdpSolver(ocp, traj_info, cfg)
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

    # crearte gif
    xs = []
    us = []
    for traj in solver.history.traj:
        xs.append(traj.xs[traj.traj_idx, :, :])
        us.append(traj.us[traj.traj_idx, :, :])

    create_gif(
        "./car_traj_plan.gif",
        xs,
        us,
        xs_opt,
        cfg.horizon,
        cfg.x_ref,
        cfg.x_min,
        cfg.x_max,
        cfg.u_min,
        cfg.u_max,
        3.0,
    )
