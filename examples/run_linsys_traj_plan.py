from ip_ddp.InteriorPointDdpSolver import InteriorPointDdpSolver
from ip_ddp.utils.OptimalControlProblem import OptimalControlProblem
from ip_ddp.utils.TrajInfo import TrajInfo
from problem_setting.lin_system.LinearSystemConfig import LinearSystemConfig
from problem_setting.lin_system.LinearSystemConstraint import LinearSystemConstraint
from problem_setting.lin_system.LinearSystemCost import LinearSystemCost
from problem_setting.lin_system.LinearSystemDynamics import LinearSystemDynamics
from utils.plot_traj_lib import plot_us, plot_xs

if __name__ == "__main__":
    cfg = LinearSystemConfig()
    dynamics = LinearSystemDynamics(cfg)
    const = LinearSystemConstraint(cfg)
    traj_info = TrajInfo(cfg, const.lc, const.lc_terminal, cfg.step_size_num)
    cost = LinearSystemCost(cfg)
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
