from pathlib import Path

import casadi as ca
import jax
import jax.numpy as jnp
import jaxadi
import numpy as np

import urdf2casadi.urdfparser as u2c
from .experiment import Experiment
from gcimopt.ocp import OCP

location = Path(__file__).parent
parser = u2c.URDFparser()
parser.from_file(location / "resources/fer_franka_hand.urdf")

fk_dict = parser.get_forward_kinematics("fer_link0", "fer_hand")
forward_kinematics = fk_dict["T_fk"]

# The last joint in the chain (hand) is not controlled since it does not affect
# the end effector position.
n_joints = parser.get_n_joints("fer_link0", "fer_hand") - 1

joint_lower = np.array(fk_dict["lower"])[:-1]
joint_upper = np.array(fk_dict["upper"])[:-1]

vel_limit = np.asarray(parser.get_dynamics_limits("fer_link0", "fer_hand")[1][:-1])
max_vel, min_vel = vel_limit, -vel_limit

x, u = ca.MX.sym("x", n_joints), ca.MX.sym("u", n_joints)
x_dot = u

state_to_goal = ca.Function("fk", [x], [forward_kinematics(ca.vertcat(x, 0))[:-1, -1]])
t_f = ca.MX.sym("tf")

dynamics = ca.Function("xdot", [x, u], [x_dot])

ALPHA = 0.1
lagrangian = ca.Function("L", [x, u], [ALPHA * ca.sum1(u**2)])
mayer = ca.Function("L_f", [x, t_f], [(1 - ALPHA) * t_f])

time_bounds = 0.01, 10

ocp = OCP(
    dynamics,
    lagrangian,
    mayer,
    time_bounds,
    (joint_lower, joint_upper),
    (min_vel, max_vel),
    state_to_goal,
)

jax_fk = jaxadi.convert(state_to_goal.expand())


def sample_initial_final_states(
    gen: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    return gen.uniform(joint_lower, joint_upper), gen.uniform(joint_lower, joint_upper)


@jax.jit
def transform_state_goal(
    state: jax.Array, goal: jax.Array, params: jax.Array
) -> jax.Array:
    ee_position = jax_fk(state)[0].ravel()
    sin, cos = jnp.sin(state), jnp.cos(state)
    X = jnp.hstack([sin, cos, goal.ravel() - ee_position])
    return jax.nn.standardize(X, mean=params[0], variance=params[1], axis=0)


@jax.jit
def model_output_to_control(model_output: jax.Array, params: jax.Array) -> jax.Array:
    return model_output * jnp.sqrt(params[1]) + params[0]


def buffer_transform_states_goals(
    states: jax.Array, goals: jax.Array, params: jax.Array | None, fit: bool
) -> tuple[jax.Array, jax.Array | None]:
    fk = state_to_goal.map(states.shape[0])
    ee_positions = jnp.asarray(fk(np.asarray(states.T)).T)
    sines, cosines = jnp.sin(states), jnp.cos(states)
    X = jnp.hstack([sines, cosines, goals - ee_positions])
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
    diff = jnp.linalg.norm(jax_fk(state)[0].ravel() - goal.ravel())
    TOL = 0.02
    return diff <= TOL


experiment = Experiment(
    ocp,
    sample_initial_final_states,
    transform_state_goal,
    model_output_to_control,
    buffer_transform_states_goals,
    buffer_transform_controls,
    goal_reached,
)
