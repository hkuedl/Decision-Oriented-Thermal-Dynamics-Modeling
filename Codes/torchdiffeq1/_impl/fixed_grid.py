from .solvers import FixedGridODESolver
from .rk_common import rk4_alt_step_func
from .misc import Perturb
import torch

class Euler(FixedGridODESolver):
    order = 1
    def _step_func(self, func, t0, dt, t1, y0, y11_i, y_input, y_phy):
        if y_input == 1:
            y0_new = torch.cat((y0, y11_i), dim=-1)
        elif y_input == 0:
            y0_new = y0
        if y_phy == 1:
            f0, ff, phy, phy_ori = func(t0, y0_new, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
            return dt*f0, ff, phy, phy_ori
            #f0, phy = func(t0, y0_new, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
            #return dt*f0, phy
        elif y_phy == 0:
            f0 = func(t0, y0_new, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
            return dt*f0

class Midpoint(FixedGridODESolver):
    order = 2
    def _step_func(self, func, t0, dt, t1, y0):
        half_dt = 0.5 * dt
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        y_mid = y0 + f0 * half_dt
        return dt * func(t0 + half_dt, y_mid), f0

class RK4(FixedGridODESolver):
    order = 4
    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        return rk4_alt_step_func(func, t0, dt, t1, y0, f0=f0, perturb=self.perturb), f0
