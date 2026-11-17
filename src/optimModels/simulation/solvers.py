import numpy as np
from scipy.integrate import solve_ivp

from optimModels.utils.configurations import SolverConfigurations
from optimModels.utils.constantes import solverMethod


_SCIPY_METHOD_MAP = {
    "LSODA": "LSODA",
    "LSODAR": "LSODA",
    "LSODE": "BDF",
    "HEUN": "RK23",
    "EULER": "RK23",
    "RK4": "RK45",
    "DORMAN_PRINCE": "RK45",
    "RKFehlberg": "RK45",
    "Dopri5": "RK45",
    "Dop853": "DOP853",
    "Vode": "BDF",
    "AdamsBashforth2": "LSODA",
    "Radau5": "Radau",
    "AdamsBashMoulton2": "LSODA",
}


class _ScipyOdeSolver:
    """Adapter mimicking an odespy solver instance using scipy.integrate.solve_ivp."""

    def __init__(self, func, method="LSODA", rtol=1e-6, atol=1e-8, nsteps=500):
        self.func = func
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.nsteps = nsteps
        self.u0 = None

    def set_initial_condition(self, u0):
        self.u0 = np.asarray(u0, dtype=float)

    def _rhs(self, t, y):
        # odespy expects f(u, t), whereas solve_ivp calls fun(t, y)
        return self.func(y, t)

    def solve(self, time_points, terminate=None):
        if self.u0 is None:
            raise ValueError("Initial condition not set. Call set_initial_condition() first.")

        t_eval = np.asarray(time_points, dtype=float)

        if terminate is None:
            sol = solve_ivp(
                fun=self._rhs,
                t_span=(t_eval[0], t_eval[-1]),
                y0=self.u0,
                method=self.method,
                t_eval=t_eval,
                rtol=self.rtol,
                atol=self.atol,
            )
            if not sol.success:
                raise RuntimeError(f"ODE integration failed ({self.method}): {sol.message}")
            # odespy returns (u, t) where u has shape (len(t), n_states)
            return sol.y.T, sol.t

        # Step-by-step evaluation when a steady-state terminate(u, t, step) callback is provided
        u_hist = [self.u0.copy()]
        t_hist = [t_eval[0]]

        if terminate(np.asarray(u_hist), np.asarray(t_hist), 0):
            return np.asarray(u_hist), np.asarray(t_hist)

        y_curr = self.u0.copy()
        for step in range(1, len(t_eval)):
            sol = solve_ivp(
                fun=self._rhs,
                t_span=(t_eval[step - 1], t_eval[step]),
                y0=y_curr,
                method=self.method,
                t_eval=[t_eval[step]],
                rtol=self.rtol,
                atol=self.atol,
            )
            if not sol.success:
                raise RuntimeError(f"ODE integration failed ({self.method}): {sol.message}")

            y_curr = sol.y[:, -1]
            u_hist.append(y_curr.copy())
            t_hist.append(t_eval[step])

            u_arr = np.asarray(u_hist)
            t_arr = np.asarray(t_hist)
            if terminate(u_arr, t_arr, step):
                return u_arr, t_arr

        return np.asarray(u_hist), np.asarray(t_hist)


class odespySolver:
    """
    ODE solver wrapper backed by scipy.integrate.solve_ivp (replacing legacy odespy).
    """

    def __init__(self, solver_method):
        self.solverMethod = solver_method

    def get_solver(self, func):
        """
        Returns a configured solver adapter compatible with the legacy odespy interface.

        Args:
            func: ODE right-hand side function with signature func(u, t).

        Returns:
            _ScipyOdeSolver
        """
        method_key = getattr(self.solverMethod, "name", str(self.solverMethod))
        scipy_method = _SCIPY_METHOD_MAP.get(method_key, "LSODA")

        return _ScipyOdeSolver(
            func=func,
            method=scipy_method,
            rtol=SolverConfigurations.RELATIVE_TOL,
            atol=SolverConfigurations.ABSOLUTE_TOL,
            nsteps=SolverConfigurations.N_STEPS,
        )

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)