import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np

from .experiment import Experiment
from gcimopt.ocp import OCP

# Create symbolic variables for the state and control.
p, p_dot = ca.MX.sym("p"), ca.MX.sym("p_dot")
x = ca.vertcat(p, p_dot)
u = ca.MX.sym("u")

# Specify the dynamics as a function of the state and control.
xdot = ca.vertcat(p_dot, u)
dynamics = ca.Function("xdot", [x, u], [xdot])

# Lagrangian and Mayer term of the cost functional.
lagrangian = ca.Function("L", [x, u], [u**2])
t_f = ca.MX.sym("t_f")
mayer = ca.Function("L_f", [x, t_f], [t_f])


# Task distribution: sample an initial state / final state pair.
def sample_initial_final_states(
    gen: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    initial = gen.uniform(low=-1, high=1, size=2)
    final = gen.uniform(low=-1, high=1, size=2)
    return initial, final


time_bounds = 0.01, 6
state_bounds = (-10, -10), (10, 10)
control_bounds = (-2,), (2,)

ocp = OCP(dynamics, lagrangian, mayer, time_bounds, state_bounds, control_bounds)


@jax.jit
def transform_state_goal(
    state: jax.Array, goal: jax.Array, params: jax.Array
) -> jax.Array:
    return jax.nn.standardize(
        jnp.concatenate([state, goal]),
        mean=params[0],
        variance=params[1],
        axis=0,
    )


@jax.jit
def model_output_to_control(model_output: jax.Array, params: jax.Array) -> jax.Array:
    return model_output * jnp.sqrt(params[1]) + params[0]


def buffer_transform_states_goals(
    states: jax.Array, goals: jax.Array, params: jax.Array | None, fit: bool
) -> tuple[jax.Array, jax.Array | None]:
    X = jnp.hstack([states, goals])
    if fit:
        X_mean, X_var = jnp.mean(X, axis=0), jnp.var(X, axis=0)
        params = jnp.vstack([X_mean, X_var])
    assert params is not None
    return jax.nn.standardize(X, mean=params[0], variance=params[1], axis=0), params


def buffer_transform_controls(
    controls: jax.Array, params: jax.Array | None, fit: bool
) -> tuple[jax.Array, jax.Array | None]:
    if fit:
        y_mean, y_var = jnp.mean(controls, axis=0), jnp.var(controls, axis=0)
        params = jnp.vstack([y_mean, y_var])
    assert params is not None
    return (
        jax.nn.standardize(controls, mean=params[0], variance=params[1], axis=0),
        params,
    )


@jax.jit
def goal_reached(state: jax.Array, goal: jax.Array) -> jax.Array:
    diff = jnp.abs(state - goal)
    return jnp.logical_and(diff[0] <= 0.01, diff[1] <= 0.04)


experiment = Experiment(
    ocp,
    sample_initial_final_states,
    transform_state_goal,
    model_output_to_control,
    buffer_transform_states_goals,
    buffer_transform_controls,
    goal_reached,
)
