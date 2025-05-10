from ip_ddp.InteriorPointDdpSolver import InteriorPointDdpSolver
from ip_ddp.utils.OptimalControlProblem import OptimalControlProblem
from ip_ddp.utils.TrajInfo import TrajInfo
from problem_setting.inverted_pendulum.InvertedPendulumConfig import (
    InvertedPendulumConfig,
)
from problem_setting.inverted_pendulum.InvertedPendulumConstraint import (
    InvertedPendulumConstraint,
)
from problem_setting.inverted_pendulum.InvertedPendulumCost import InvertedPendulumCost
from problem_setting.inverted_pendulum.InvertedPendulumDynamics import (
    InvertedPendulumDynamics,
)
from utils.plot_traj_lib import plot_us, plot_xs

if __name__ == "__main__":
    cfg = InvertedPendulumConfig()
    dynamics = InvertedPendulumDynamics(cfg)
    const = InvertedPendulumConstraint(cfg)
    traj_info = TrajInfo(cfg, const.lc, const.lc_terminal, cfg.step_size_num)
    cost = InvertedPendulumCost(cfg)
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
