import numpy as np

from ip_ddp.utils.TrajInfo import TrajInfo
from problem_setting.abstract.Config import Config
from problem_setting.abstract.Constraint import Constraint
from problem_setting.abstract.Cost import Cost
from problem_setting.abstract.Dynamics import Dynamics


class OptimalControlProblem:
    def __init__(
        self,
        dynamics: Dynamics,
        const: Constraint,
        cost: Cost,
        cfg: Config,
    ) -> None:
        self.dynamics: Dynamics = dynamics
        self.const: Constraint = const
        self.cost: Cost = cost
        self.n: int = cfg.n
        self.m: int = cfg.m
        self.lc: int = const.lc
        self.lc_terminal: int = const.lc_terminal

    def set_constant(
        self,
        traj_info: TrajInfo,
    ) -> None:
        self.dynamics.set_constant(traj_info)
        self.const.set_constant(traj_info)
        self.cost.set_constant(traj_info)

    def transit(
        self,
        traj_info: TrajInfo,
        t: int,
    ) -> None:
        self.dynamics.transit(traj_info, t)

    def calc_const(
        self,
        traj_info: TrajInfo,
    ) -> None:
        self.const.calc(traj_info)

    def calc_cost(self, traj_info: TrajInfo) -> None:
        self.cost.calc_stage_cost(traj_info)
        self.cost.calc_terminal_cost(traj_info)
        traj_info.costs = (
            np.sum(traj_info.stage_costs, axis=1) + traj_info.terminal_costs
        )

    def diff(self, traj_info: TrajInfo, calc_dynamics_hessian: bool = False) -> None:
        self.dynamics.calc_jacobian(traj_info)
        if calc_dynamics_hessian:
            self.dynamics.calc_hessian(traj_info)
        self.const.calc_grad(traj_info)
        self.cost.calc_grad(traj_info)
        self.cost.calc_hessian(traj_info)
