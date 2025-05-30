import casadi as ca
import numpy as np
from numpy.typing import ArrayLike


class OCP:
    FATROP_OPTIONS = {
        "expand": True,
        "fatrop": {"print_level": 0},
        "structure_detection": "auto",
        "print_time": False,
        "jit": False,
        "jit_temp_suffix": False,
        "jit_cleanup": False,
        "jit_options": {"flags": ["-O7", "-march=native"], "compiler": "ccache gcc"},
    }

    dynamics: ca.Function
    lagrange_objective: ca.Function
    mayer_objective: ca.Function
    state_to_goal: ca.Function | None
    time_bounds: tuple[float, float]
    state_bounds: tuple[np.ndarray, np.ndarray]
    control_bounds: tuple[np.ndarray, np.ndarray]

    def __init__(
        self,
        dynamics: ca.Function,
        lagrange_objective: ca.Function | None,
        mayer_objective: ca.Function | None,
        time_bounds: tuple[float, float],
        state_bounds: tuple[ArrayLike, ArrayLike],
        control_bounds: tuple[ArrayLike, ArrayLike],
        state_to_goal: ca.Function | None = None,
    ) -> None:
        state_dim, control_dim = [dynamics.size_in(i)[0] for i in range(2)]
        if (
            np.asarray(state_bounds[0]).shape != np.asarray(state_bounds[1]).shape
            or np.asarray(state_bounds[0]).size != state_dim
        ):
            raise ValueError("State bound shapes do not match")
        if (
            np.asarray(control_bounds[0]).shape != np.asarray(control_bounds[1]).shape
            or np.asarray(control_bounds[0]).size != control_dim
        ):
            raise ValueError("Control bound shapes do not match")
        if lagrange_objective is None and mayer_objective is None:
            raise ValueError(
                "At least one of lagrange_objective and mayer_objective should be not None"
            )
        self.dynamics = dynamics
        x, u, t = ca.MX.sym("x", state_dim), ca.MX.sym("u", control_dim), ca.MX.sym("t")
        self.lagrange_objective = (
            lagrange_objective
            if lagrange_objective is not None
            else ca.Function("L", [x, u], [0])
        )
        self.mayer_objective = (
            mayer_objective
            if mayer_objective is not None
            else ca.Function("L_f", [x, t], [0])
        )
        self._nx = state_dim
        self._nu = control_dim
        self.time_bounds = time_bounds
        self.state_bounds = np.asarray(state_bounds[0]), np.asarray(state_bounds[1])
        self.control_bounds = (
            np.asarray(control_bounds[0]),
            np.asarray(control_bounds[1]),
        )
        self.state_to_goal = state_to_goal
        if state_to_goal is None:
            self._ng = self._nx
        else:
            self._ng = state_to_goal.size_out(0)[0]

    def _integrate(self, n_intervals: int, n_rk_steps_per_interval: int) -> ca.Function:
        time = ca.MX.sym("t")
        dt = time / n_intervals / n_rk_steps_per_interval
        state_var, control_var = ca.MX.sym("x", self._nx), ca.MX.sym("u", self._nu)
        dynamics_and_cost = ca.Function(
            "dynamics_and_cost",
            [state_var, control_var],
            [
                self.dynamics(state_var, control_var),
                self.lagrange_objective(state_var, control_var),
            ],
        )
        initial_state = ca.MX.sym("x_0", self._nx)
        x = initial_state
        running_cost = 0
        u = ca.MX.sym("u", self._nu)
        for _ in range(n_rk_steps_per_interval):
            k1, k1_q = dynamics_and_cost(x, u)
            k2, k2_q = dynamics_and_cost(x + dt / 2 * k1, u)
            k3, k3_q = dynamics_and_cost(x + dt / 2 * k2, u)
            k4, k4_q = dynamics_and_cost(x + dt * k3, u)
            x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            running_cost = running_cost + dt / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
        return ca.Function("F", [initial_state, u, time], [x, running_cost])

    def transcribe(
        self,
        initial_state: ArrayLike,
        final_state: ArrayLike,
        n_intervals: int = 30,
        n_rk_steps_per_interval: int = 4,
        override_fatrop_options: dict = {},
    ) -> tuple[ca.Function, dict]:
        initial_state, final_state = (
            np.asarray(initial_state).ravel(),
            np.asarray(final_state).ravel(),
        )
        if initial_state.size != self._nx or final_state.size != self._nx:
            raise ValueError(
                "Initial and final state sizes don't match state dimensions"
            )
        integrate = self._integrate(n_intervals, n_rk_steps_per_interval)
        states, controls, time = [], [], []
        decision_vars, initial_guess, lb, ub = [], [], [], []
        g, lbg, ubg, is_equality = [], [], [], []
        for k in range(n_intervals + 1):
            state = ca.MX.sym(f"x_{k}", self._nx)
            states.append(state)
            decision_vars.append(state)
            initial_guess.append(
                initial_state + k * (final_state - initial_state) / n_intervals
            )
            if k == 0:
                lb.append(initial_state)
                ub.append(initial_state)
            elif k == n_intervals and self.state_to_goal is None:
                lb.append(final_state)
                ub.append(final_state)
            else:
                lb.append(self.state_bounds[0])
                ub.append(self.state_bounds[1])
            t = ca.MX.sym("t")
            time.append(t)
            decision_vars.append(t)
            initial_guess.append(sum(self.time_bounds) / 2)
            lb.append(self.time_bounds[0])
            ub.append(self.time_bounds[1])
            if k < n_intervals:
                control = ca.MX.sym(f"u_{k}", self._nu)
                controls.append(control)
                decision_vars.append(control)
                initial_guess.append(ca.DM.zeros(self._nu))
                lb.append(self.control_bounds[0])
                ub.append(self.control_bounds[1])
        cost = 0
        for k in range(n_intervals):
            predicted_state, interval_cost = integrate(
                states[k], controls[k], time[k]
            )  # Integrate to the end of the interval
            cost = cost + interval_cost
            g.append(
                states[k + 1] - predicted_state
            )  # Multiple shooting gap closing constraint
            lbg += self._nx * [0]
            ubg += self._nx * [0]
            is_equality += self._nx * [True]
            g.append(time[k + 1] - time[k])
            lbg.append(0)
            ubg.append(0)
            is_equality.append(True)
        cost = cost + self.mayer_objective(states[-1], time[-1])
        if self.state_to_goal is not None:
            goal_diff = self.state_to_goal(states[-1]) - self.state_to_goal(final_state)
            g.append(goal_diff)
            lbg += goal_diff.numel() * [0]
            ubg += goal_diff.numel() * [0]
            is_equality += goal_diff.numel() * [True]
        problem = {"f": cost, "x": ca.vcat(decision_vars), "g": ca.vertcat(*g)}
        fatrop_options = self.FATROP_OPTIONS.copy()
        for key, value in override_fatrop_options.items():
            fatrop_options[key] = value
        fatrop_options["equality"] = is_equality
        solver = ca.nlpsol("solver", "fatrop", problem, fatrop_options)
        return solver, dict(
            x0=ca.vcat(initial_guess),
            lbx=ca.vcat(lb),
            ubx=ca.vcat(ub),
            lbg=ca.vcat(lbg),
            ubg=ca.vcat(ubg),
        )

    def _set_initial_final_state_constraint(
        self,
        initial_state: ArrayLike,
        final_state: ArrayLike,
        lbx: ca.DM,
        ubx: ca.DM,
        lbg: ca.DM,
        ubg: ca.DM,
    ) -> None:
        nx = self._nx
        lbx[:nx] = ubx[:nx] = np.asarray(initial_state)
        if self.state_to_goal is None:
            lbx[-nx - 1 : -1] = ubx[-nx - 1 : -1] = final_state
            return
        lbx[-nx - 1 : -1], ubx[-nx - 1 : -1] = self.state_bounds
        goal = self.state_to_goal(final_state)
        goal_size = goal.numel()
        lbg[-goal_size:] = ubg[-goal_size:] = goal - self.state_to_goal(ca.DM.zeros(nx))

    def _get_initial_guess(
        self,
        initial_state: np.ndarray,
        final_state: np.ndarray,
        n_intervals: int,
    ) -> np.ndarray:
        nu = self._nu
        time_initial_guess = sum(self.time_bounds) / 2
        states_lerp = np.linspace(
            np.asarray(initial_state), np.asarray(final_state), n_intervals + 1
        )
        x0 = np.hstack(
            [
                states_lerp,
                np.full((n_intervals + 1, 1), time_initial_guess),
                np.zeros((n_intervals + 1, nu)),
            ],
        )
        return x0.ravel()[:-nu]
