import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType
from safe_control_gym.utils.registration import make

from .experiment import Experiment
from gcimopt.ocp import OCP

env = make(
    "quadrotor",
    quad_type=QuadType.THREE_D,
    physics="pyb",
    task_info=dict(stabilization_goal=[0, 0, 0]),
)

_, obs = env.reset()
model = obs["symbolic_model"]
x, u, x_dot = model.x_sym, model.u_sym, model.x_dot

t_f = ca.MX.sym("t_f")

dynamics = ca.Function("xdot", [x, u], [x_dot])

ALPHA = 1
lagrangian = ca.Function("L", [x, u], [ALPHA * ca.sum1(u**2)])
mayer = ca.Function("L_f", [x, t_f], [(1 - ALPHA) * t_f])

time_bounds = 0.01, 6
state_bounds = (
    x.numel()
    * [
        -15,
    ],
    x.numel()
    * [
        15,
    ],
)
control_bounds = env.physical_action_bounds

ocp = OCP(dynamics, lagrangian, mayer, time_bounds, state_bounds, control_bounds)


def sample_initial_final_states(gen: np.random.Generator) -> tuple[list, list]:
    RADIUS = 2
    x, y, z = gen.uniform(3 * [-RADIUS], 3 * [RADIUS])
    xx, yy, zz = gen.uniform(3 * [-RADIUS], 3 * [RADIUS])

    return [
        x,
        0,
        y,
        0,
        z,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ], [
        xx,
        0,
        yy,
        0,
        zz,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]


@jax.jit
def transform_state_goal(
    state: jax.Array, goal: jax.Array, params: jax.Array
) -> jax.Array:
    state_sin, state_cos = jnp.sin(state[6:9]), jnp.cos(state[6:9])
    goal_sin, goal_cos = jnp.sin(goal[6:9]), jnp.cos(goal[6:9])
    X = jnp.hstack(
        [
            state_sin,
            state_cos,
            goal_sin,
            goal_cos,
            state[9:],
            goal[9:],
            goal[jnp.asarray([0, 2, 4])] - state[jnp.asarray([0, 2, 4])],
            state[jnp.asarray([1, 3, 5])],
            goal[jnp.asarray([1, 3, 5])],
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
    state_sin, state_cos = jnp.sin(states[:, 6:9]), jnp.cos(states[:, 6:9])
    goal_sin, goal_cos = jnp.sin(goals[:, 6:9]), jnp.cos(goals[:, 6:9])
    X = jnp.hstack(
        [
            state_sin,
            state_cos,
            goal_sin,
            goal_cos,
            states[:, 9:],
            goals[:, 9:],
            goals[:, [0, 2, 4]] - states[:, [0, 2, 4]],
            states[:, [1, 3, 5]],
            goals[:, [1, 3, 5]],
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
    distance = jnp.linalg.norm(
        state[jnp.asarray([0, 2, 4])] - goal[jnp.asarray([0, 2, 4])]
    )
    vel_distance = jnp.linalg.norm(
        state[jnp.asarray([1, 3, 5])] - goal[jnp.asarray([1, 3, 5])]
    )
    angles_diff = jnp.abs(state[jnp.asarray([6, 7, 8])] - goal[jnp.asarray([6, 7, 8])])
    angles_dot_diff = jnp.abs(
        state[jnp.asarray([9, 10, 11])] - goal[jnp.asarray([9, 10, 11])]
    )
    TOL_DIST = 0.05
    TOL_VEL = 0.05
    TOL_ANGLES = 0.1
    TOL_ANGLES_DOT = 0.1
    return jnp.logical_and(
        jnp.logical_and(
            jnp.logical_and(distance <= TOL_DIST, vel_distance <= TOL_VEL),
            jnp.all(angles_diff <= TOL_ANGLES),
        ),
        angles_dot_diff <= TOL_ANGLES_DOT,
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
