import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np
from gcimopt.ocp import OCP
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType
from safe_control_gym.utils.registration import make

from .experiment import Experiment

env = make(
    "quadrotor",
    quad_type=QuadType.TWO_D,
    physics="pyb",
)

_, obs = env.reset()
env.close()
model = obs["symbolic_model"]
x, u, x_dot = model.x_sym, model.u_sym, model.x_dot
t_f = ca.MX.sym("t_f")

dynamics = ca.Function("xdot", [x, u], [x_dot])

ALPHA = 1
lagrangian = ca.Function("L", [x, u], [ALPHA * ca.sum1(u**2)])
mayer = ca.Function("L_f", [x, t_f], [(1 - ALPHA) * t_f])

time_bounds = 0.01, 10
state_bounds = x.numel() * [-30], x.numel() * [30]
state_bounds[0][4] = -np.pi
state_bounds[1][4] = np.pi
control_bounds = env.physical_action_bounds

ocp = OCP(dynamics, lagrangian, mayer, time_bounds, state_bounds, control_bounds)


def sample_initial_final_states(
    gen: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    initial = gen.uniform([-5, -5, -5, -5, -np.pi, -1], [5, 5, 5, 5, np.pi, 1])
    final = gen.uniform([-5, 0, -5, 0, 0, 0], [5, 0, 5, 0, 0, 0])
    return initial, final


@jax.jit
def transform_state_goal(
    state: jax.Array, goal: jax.Array, params: jax.Array
) -> jax.Array:
    state_sin, state_cos = jnp.sin(state[4]), jnp.cos(state[4])
    goal_sin, goal_cos = jnp.sin(goal[4]), jnp.cos(goal[4])
    X = jnp.hstack(
        [
            goal[jnp.asarray([0, 2])] - state[jnp.asarray([0, 2])],
            state[jnp.asarray([1, 3, 5])],
            state_sin,
            state_cos,
            goal[jnp.asarray([1, 3, 5])],
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
    state_sin, state_cos = jnp.sin(states[:, [4]]), jnp.cos(states[:, [4]])
    goal_sin, goal_cos = jnp.sin(goals[:, [4]]), jnp.cos(goals[:, [4]])
    X = jnp.hstack(
        [
            goals[:, [0, 2]] - states[:, [0, 2]],
            states[:, [1, 3, 5]],
            state_sin,
            state_cos,
            goals[:, [1, 3, 5]],
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
    distance = jnp.linalg.norm(state[jnp.asarray([0, 2])] - goal[jnp.asarray([0, 2])])
    vel_distance = jnp.linalg.norm(
        state[jnp.asarray([1, 3])] - goal[jnp.asarray([1, 3])]
    )
    theta_diff = jnp.abs(state[4] - goal[4])
    theta_dot_diff = jnp.abs(state[5] - goal[5])
    TOL_DIST = 0.05
    TOL_VEL = 0.05
    TOL_THETA = 0.1
    TOL_THETA_DOT = 0.1
    return jnp.logical_and(
        jnp.logical_and(
            jnp.logical_and(distance <= TOL_DIST, vel_distance <= TOL_VEL),
            theta_diff <= TOL_THETA,
        ),
        theta_dot_diff <= TOL_THETA_DOT,
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
