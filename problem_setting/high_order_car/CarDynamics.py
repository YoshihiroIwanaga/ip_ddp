import numpy as np

from ip_ddp.utils.TrajInfo import TrajInfo
from problem_setting.abstract.Config import Config
from problem_setting.abstract.Dynamics import Dynamics


class CarDynamics(Dynamics):
    def __init__(self, config: Config) -> None:
        super(CarDynamics, self).__init__(config)
        self.wheel_base: float = 2.7

    def set_constant(
        self,
        traj_info: TrajInfo,
    ) -> None:
        # fxs: np.ndarray,  # ∈R^{n,n,te}
        traj_info.fxs[0, 0, :] = 1
        traj_info.fxs[1, 1, :] = 1
        traj_info.fxs[2, 2, :] = 1
        traj_info.fxs[3, 3, :] = 1
        traj_info.fxs[3, 5, :] = self.dt
        traj_info.fxs[4, 4, :] = 1
        traj_info.fxs[4, 6, :] = self.dt
        traj_info.fxs[5, 5, :] = 1
        traj_info.fxs[6, 6, :] = 1
        # fus: np.ndarray,  # ∈R^{n,m,te}
        traj_info.fus[5, 0, :] = self.dt
        traj_info.fus[6, 1, :] = self.dt

    def transit(
        self,
        traj_info: TrajInfo,
        t: int,  # time step
    ) -> None:
        # xs: np.ndarray,  # ∈R^{K,n,te+1} K:number of step sizes
        # us: np.ndarray,  # ∈R^{K,m,te} K:number of step sizes

        # x[k+1] = x[k] + dt*v[k]*cos(φ[k])
        traj_info.xs[:, 0, t + 1 : t + 2] = traj_info.xs[
            :, 0, t : t + 1
        ] + self.dt * traj_info.xs[:, 3, t : t + 1] * np.cos(
            traj_info.xs[:, 2, t : t + 1]
        )
        # y[k+1] = y[k] + dt*v[k]*sin(φ[k])
        traj_info.xs[:, 1, t + 1 : t + 2] = traj_info.xs[
            :, 1, t : t + 1
        ] + self.dt * traj_info.xs[:, 3, t : t + 1] * np.sin(
            traj_info.xs[:, 2, t : t + 1]
        )
        # φ[k+1] = φ[k] + dt*v[k]*tan(δ[k])/L
        traj_info.xs[:, 2, t + 1 : t + 2] = (
            traj_info.xs[:, 2, t : t + 1]
            + self.dt
            * traj_info.xs[:, 3, t : t + 1]
            * np.tan(traj_info.xs[:, 4, t : t + 1])
            / self.wheel_base
        )
        # v[k+1] = v[k] + dt*a[k]
        traj_info.xs[:, 3, t + 1 : t + 2] = (
            traj_info.xs[:, 3, t : t + 1] + self.dt * traj_info.xs[:, 5, t : t + 1]
        )
        # δ[k+1] = δ[k] + dt*ω[k]
        traj_info.xs[:, 4, t + 1 : t + 2] = (
            traj_info.xs[:, 4, t : t + 1] + self.dt * traj_info.xs[:, 6, t : t + 1]
        )
        # a[k+1] = a[k] + dt*u0[k]
        traj_info.xs[:, 5, t + 1 : t + 2] = (
            traj_info.xs[:, 5, t : t + 1] + self.dt * traj_info.us[:, 0, t : t + 1]
        )
        # ω[k+1] = ω[k] + dt*u1[k]
        traj_info.xs[:, 6, t + 1 : t + 2] = (
            traj_info.xs[:, 6, t : t + 1] + self.dt * traj_info.us[:, 1, t : t + 1]
        )

    def calc_jacobian(
        self,
        traj_info: TrajInfo,
    ) -> None:
        sin_yaw = np.sin(traj_info.xs[traj_info.traj_idx, 2, :-1])
        cos_yaw = np.cos(traj_info.xs[traj_info.traj_idx, 2, :-1])
        traj_info.fxs[0, 2, :] = (
            -self.dt * traj_info.xs[traj_info.traj_idx, 3, :-1] * sin_yaw
        )
        traj_info.fxs[0, 3, :] = self.dt * cos_yaw
        traj_info.fxs[1, 2, :] = (
            self.dt * traj_info.xs[traj_info.traj_idx, 3, :-1] * cos_yaw
        )
        traj_info.fxs[1, 3, :] = self.dt * sin_yaw
        traj_info.fxs[2, 3, :] = (
            self.dt * np.tan(traj_info.xs[traj_info.traj_idx, 4, :-1]) / self.wheel_base
        )
        traj_info.fxs[2, 4, :] = (
            self.dt
            * (
                traj_info.xs[traj_info.traj_idx, 3, :-1]
                / np.cos(traj_info.xs[traj_info.traj_idx, 4, :-1]) ** 2
            )
            / self.wheel_base
        )

    def calc_hessian(
        self,
        traj_info: TrajInfo,
    ) -> None:
        dt_sin_yaw = self.dt * np.sin(traj_info.xs[traj_info.traj_idx, 2, :-1])
        dt_cos_yaw = self.dt * np.cos(traj_info.xs[traj_info.traj_idx, 2, :-1])
        tan_steer = np.tan(traj_info.xs[traj_info.traj_idx, 4, :-1])

        # ∂2x[k+1]/∂φ[k]∂φ[k]
        traj_info.fxxs[0, 2, 2, :] = (
            -traj_info.xs[traj_info.traj_idx, 3, :-1] * dt_cos_yaw
        )
        # ∂2x[k+1]/∂φ[k]∂v[k]
        traj_info.fxxs[0, 2, 3, :] = -dt_sin_yaw
        traj_info.fxxs[0, 3, 2, :] = -dt_sin_yaw
        # ∂2y[k+1]/∂φ[k]∂φ[k]
        traj_info.fxxs[1, 2, 2, :] = (
            -traj_info.xs[traj_info.traj_idx, 3, :-1] * dt_sin_yaw
        )
        # ∂2y[k+1]/∂φ[k]∂v[k]
        traj_info.fxxs[1, 2, 3, :] = dt_cos_yaw
        traj_info.fxxs[1, 3, 2, :] = dt_cos_yaw
        # ∂2φ[k+1]/∂δ[k]∂δ[k]
        traj_info.fxxs[2, 4, 4, :] = (
            self.dt
            / self.wheel_base
            * traj_info.xs[traj_info.traj_idx, 3, :-1]
            * (2 * tan_steer**2 + 2)
            * tan_steer
        )
        # ∂2φ[k+1]/∂v[k]∂δ[k]
        traj_info.fxxs[2, 3, 4, :] = (
            self.dt
            / self.wheel_base
            * 1
            / (np.cos(traj_info.xs[traj_info.traj_idx, 4, :-1]) ** 2)
        )
        traj_info.fxxs[2, 4, 3, :] = (
            self.dt
            / self.wheel_base
            * 1
            / (np.cos(traj_info.xs[traj_info.traj_idx, 4, :-1]) ** 2)
        )
