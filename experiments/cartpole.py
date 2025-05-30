import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np
from safe_control_gym.utils.registration import make

from .experiment import Experiment
from gcimopt.ocp import OCP

env = make(
    "cartpole",
    physics="pyb",
)

_, obs = env.reset()
model = obs["symbolic_model"]
x, u, x_dot = model.x_sym, model.u_sym, model.x_dot
t_f = ca.MX.sym("t_f")

dynamics = ca.Function("xdot", [x, u], [x_dot])

ALPHA = 0.05
lagrangian = ca.Function("L", [x, u], [ALPHA * ca.sum1(u**2)])
mayer = ca.Function("L_f", [x, t_f], [(1 - ALPHA) * t_f])

time_bounds = 0.5, 20
state_bounds = x.numel() * (-10,), x.numel() * (10,)
control_bounds = env.physical_action_bounds

ocp = OCP(dynamics, lagrangian, mayer, time_bounds, state_bounds, control_bounds)


def sample_initial_final_states(
    gen: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    initial = gen.uniform([-3, -1, 0, -1], [3, 1, 2 * np.pi, 1])
    final = [
        gen.uniform(-3, 3),
        0,
        np.pi * gen.integers(0, 2),
        0,
    ]
    return np.asarray(initial), np.asarray(final)


@jax.jit
def transform_state_goal(
    state: jax.Array, goal: jax.Array, params: jax.Array
) -> jax.Array:
    state_sin, state_cos = jnp.sin(state[2]), jnp.cos(state[2])
    goal_sin, goal_cos = jnp.sin(goal[2]), jnp.cos(goal[2])
    X = jnp.hstack(
        [
            goal[0] - state[0],
            state[jnp.asarray([1, 3])],
            state_sin,
            state_cos,
            goal[jnp.asarray([1, 3])],
            goal_sin,
            goal_cos,
        ]
    )
    return jax.nn.standardize(
        X,
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
    state_sin, state_cos = jnp.sin(states[:, [2]]), jnp.cos(states[:, [2]])
    goal_sin, goal_cos = jnp.sin(goals[:, [2]]), jnp.cos(goals[:, [2]])
    X = jnp.hstack(
        [
            goals[:, [0]] - states[:, [0]],
            states[:, [1, 3]],
            state_sin,
            state_cos,
            goals[:, [1, 3]],
            goal_sin,
            goal_cos,
        ]
    )
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
    TOL_X = 0.05
    TOL_X_DOT = 0.1
    TOL_THETA = 0.1
    TOL_THETA_DOT = 0.1
    return jnp.logical_and(
        jnp.logical_and(
            jnp.logical_and(diff[0] <= TOL_X, diff[1] <= TOL_X_DOT),
            diff[2] <= TOL_THETA,
        ),
        diff[3] <= TOL_THETA_DOT,
    )


experiment = Experiment(
    ocp,
    sample_initial_final_states,
    transform_state_goal,
    model_output_to_control,
    buffer_transform_states_goals,
    buffer_transform_controls,
    goal_reached,
)
