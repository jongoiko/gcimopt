from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any
from typing import Callable

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jaxadi
import numpy as np
import optax
from numpy.typing import ArrayLike
from tensorboardX import SummaryWriter

from .ocp import OCP
from .trajectory_buffer import TrajectoryBuffer


@eqx.filter_jit
def _mse_loss(model: Any, X: jax.Array, y: jax.Array) -> jax.Array:
    y_pred = jax.vmap(model)(X)
    diff = ((y - y_pred) ** 2).sum(axis=1)
    return jnp.mean(diff)


_mse_loss_grad = eqx.filter_value_and_grad(_mse_loss)


@eqx.filter_jit
def _make_mse_optim_step(
    model: Any,
    optim: optax.GradientTransformation,
    x: jax.Array,
    y: jax.Array,
    opt_state: optax.OptState,
) -> tuple[Any, jax.Array, optax.OptState]:
    loss, grads = _mse_loss_grad(model, x, y)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, loss, opt_state


def _simulation_ode(
    t: float,
    y: jax.Array,
    args: tuple[
        jax.Array,
        tuple[jax.Array, jax.Array] | None,
        tuple[jax.Array, jax.Array] | None,
        Callable[[jax.Array, jax.Array], jax.Array],
        Callable[[jax.Array, jax.Array], jax.Array],
        Callable[[jax.Array, jax.Array], jax.Array],
    ],
) -> jax.Array:
    goal, state_bounds, control_bounds, dynamics, lagrangian, policy_func = args
    y = y[:-1]
    above_bound, below_bound = y, y
    if state_bounds is not None:
        above_bound = y > state_bounds[1]
        below_bound = y < state_bounds[0]
        y = jnp.clip(y, *state_bounds)
    u = policy_func(y, goal)
    if control_bounds is not None:
        u = jnp.clip(u, *control_bounds)
    lagrange = lagrangian(y, u)
    xdot = dynamics(y, u).ravel()
    if state_bounds is not None:
        xdot = jnp.where(jnp.logical_and(above_bound, xdot > 0), 0, xdot)
        xdot = jnp.where(jnp.logical_and(below_bound, xdot < 0), 0, xdot)
    return jnp.concatenate([xdot, lagrange])


@eqx.filter_jit
def _simulate(
    initial: jax.Array,
    goal: jax.Array,
    t_0: float,
    t_f: float,
    dt: float,
    args: tuple[
        tuple[jax.Array, jax.Array] | None,
        tuple[jax.Array, jax.Array] | None,
        Callable[[jax.Array, jax.Array], jax.Array],
        Callable[[jax.Array, jax.Array], jax.Array],
        Callable[[jax.Array, jax.Array], jax.Array],
    ],
    n_samples: int,
) -> diffrax.Solution:
    term = diffrax.ODETerm(_simulation_ode)  # type: ignore
    solver = diffrax.Dopri8()
    saveat = diffrax.SaveAt(ts=jnp.linspace(t_0, t_f, n_samples))
    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=t_0,
        t1=t_f,
        dt0=dt,
        y0=initial,
        saveat=saveat,
        args=(goal,) + args,
    )
    return solution


class Policy:
    _ARCHIVE_MODEL = "model.eqx"
    _ARCHIVE_TRANSFORM_STATE_GOAL_PARAMS = "transform.npy"
    _ARCHIVE_MODEL_OUTPUT_TO_CONTROL_PARAMS = "postprocess.npy"

    _transform_state_goal_params: jax.Array | None
    _model_output_to_control_params: jax.Array | None

    _transform_state_goal: Callable[[jax.Array, jax.Array, jax.Array | None], jax.Array]
    _model_output_to_control: Callable[[jax.Array, jax.Array | None], jax.Array] | None

    _model: eqx.Module

    def __init__(
        self,
        model: eqx.Module,
        transform_state_goal: Callable[
            [jax.Array, jax.Array, jax.Array | None], jax.Array
        ],
        model_output_to_control: (
            Callable[[jax.Array, jax.Array | None], jax.Array] | None
        ),
        transform_state_goal_params: jax.Array | None = None,
        model_output_to_control_params: jax.Array | None = None,
    ) -> None:
        self._model = model
        self._transform_state_goal = transform_state_goal
        self._model_output_to_control = model_output_to_control
        self._transform_state_goal_params = transform_state_goal_params
        self._model_output_to_control_params = model_output_to_control_params
        self.set_inference_mode(True)

    def action_from_state_and_goal(
        self, state: jax.Array, goal: jax.Array
    ) -> jax.Array:
        model_input = self._transform_state_goal(
            state, goal, self._transform_state_goal_params
        )
        model_output = self._model(model_input)  # type: ignore
        if self._model_output_to_control is None:
            return model_output
        return self._model_output_to_control(
            model_output, self._model_output_to_control_params
        )

    def set_inference_mode(self, inference_mode: bool) -> None:
        self._model = eqx.nn.inference_mode(self._model, value=inference_mode)

    @property
    def num_model_params(self) -> int:
        return sum(
            leaf.size
            for leaf in jax.tree.leaves(self._model)
            if isinstance(leaf, jax.Array)
        )

    def train(
        self,
        buffer: TrajectoryBuffer,
        n_epochs: int,
        batch_size: int = 512,
        adam_lr: float = 5e-4,
        tb_writer: SummaryWriter | None = None,
        print_loss_every_n_epochs: int | None = None,
        epoch_callback: Callable[[Policy, int], None] | None = None,
        callback_every_n_epochs: int = 1,
        random_seed: int | None = None,
    ) -> None:
        batch_size = min((batch_size, buffer._X_train.shape[0]))
        optim = optax.adam(adam_lr)
        opt_state = optim.init(eqx.filter(self._model, eqx.is_array))
        random_gen = np.random.default_rng(random_seed)
        for epoch in range(n_epochs):
            self.set_inference_mode(False)
            # Train for one epoch
            data_iter = buffer._training_set_iterate_one_epoch(batch_size, random_gen)
            for x, y in data_iter:
                self._model, _, opt_state = _make_mse_optim_step(
                    self._model, optim, x, y, opt_state
                )
            data_iter = buffer._training_set_iterate_one_epoch(batch_size, random_gen)
            # Evaluate loss in training and validation sets
            self.set_inference_mode(True)
            train_loss = 0
            for x, y in data_iter:
                batch_loss = _mse_loss(self._model, x, y)
                train_loss += batch_loss * (x.shape[0] / buffer._X_train.shape[0])
            data_iter = buffer._validation_set_iterate(batch_size, random_gen)
            val_loss = 0
            for x, y in data_iter:
                batch_loss = _mse_loss(self._model, x, y)
                val_loss += batch_loss * (x.shape[0] / buffer._X_val.shape[0])
            if tb_writer is not None:
                tb_writer.add_scalar("epoch_loss/train", train_loss, epoch)
                tb_writer.add_scalar("epoch_loss/val", val_loss, epoch)
            if (
                print_loss_every_n_epochs is not None
                and epoch % print_loss_every_n_epochs == 0
            ):
                print(f"Epoch {epoch} loss: train {train_loss}, val {val_loss}")
            if epoch_callback is not None and epoch % callback_every_n_epochs == 0:
                epoch_callback(self, epoch)
        self.set_inference_mode(True)

    def evaluate(
        self,
        ocp: OCP,
        n_tasks: int,
        sample_initial_final_states: Callable[
            [np.random.Generator], tuple[ArrayLike, ArrayLike]
        ],
        goal_reached: Callable[[jax.Array, jax.Array], bool],
        simulation_bound_controls: bool,
        simulation_bound_states: bool,
        simulation_dt: float = 0.01,
        simulation_n_samples: int = 400,
        trajopt_results_path: str | None = None,
        opt_n_intervals: int = 30,
        override_fatrop_options: dict = {},
        random_seed: int | None = None,
    ) -> tuple[float, jax.Array]:
        initial_states, goals, optimal_costs = self._sample_tasks_and_optimal_costs(
            ocp,
            n_tasks,
            random_seed,
            opt_n_intervals,
            sample_initial_final_states,
            trajopt_results_path,
            override_fatrop_options,
        )
        # We use an additional state variable to store the integrated Lagrange term
        initial_states = np.hstack(
            [initial_states, np.zeros((initial_states.shape[0], 1))]
        )
        rollouts = self._run_rollouts(
            ocp,
            simulation_bound_states,
            simulation_bound_controls,
            initial_states,
            goals,
            simulation_dt,
            simulation_n_samples,
        )
        success_indices = self._get_success_indices(
            rollouts, goals, goal_reached, ocp.time_bounds[0]
        )
        success_rate = (
            100 * ((success_indices >= 0).sum() / success_indices.size).item()
        )
        relative_opt_errors = self._calculate_relative_opt_errors(
            rollouts, ocp, success_indices, jnp.asarray(optimal_costs)
        )
        return success_rate, relative_opt_errors

    def _get_success_indices(
        self,
        rollouts: diffrax.Solution,
        goals: np.ndarray,
        goal_reached: Callable[[jax.Array, jax.Array], bool],
        min_time: float,
    ) -> jax.Array:
        assert rollouts.ts is not None and rollouts.ys is not None
        times, states = rollouts.ts, rollouts.ys[..., :-1]
        vmap_goal_reached = jax.vmap(goal_reached)
        reshaped_goals = jnp.repeat(
            goals.reshape(goals.shape[0], 1, -1), states.shape[1], axis=1
        )

        def success(states: jax.Array, goal: jax.Array) -> tuple[jax.Array, jax.Array]:
            reached = vmap_goal_reached(states, goal)
            return reached.argmax(), jnp.any(reached)  # type: ignore

        success_indices, successes = jax.jit(jax.vmap(success))(states, reshaped_goals)
        minimum_success_indices = (times >= min_time).argmax(axis=1)
        success_indices = success_indices.at[
            success_indices < minimum_success_indices
        ].set(-1)
        success_indices = success_indices.at[jnp.logical_not(successes)].set(-1)
        return success_indices

    def _calculate_relative_opt_errors(
        self,
        rollouts: diffrax.Solution,
        ocp: OCP,
        success_indices: jax.Array,
        optimal_costs: jax.Array,
    ) -> jax.Array:
        assert rollouts.ts is not None and rollouts.ys is not None
        times, states, lagrange_costs = (
            rollouts.ts,
            rollouts.ys[..., :-1],
            rollouts.ys[..., -1],
        )
        success_mask = success_indices >= 0
        times = times[success_mask]
        states = states[success_mask]
        success_indices = success_indices[success_mask]
        optimal_costs = optimal_costs[success_mask]
        lagrange_costs = lagrange_costs[success_mask]
        lagrange_costs = lagrange_costs[
            jnp.arange(lagrange_costs.shape[0]), success_indices
        ]
        final_states = states[jnp.arange(states.shape[0]), success_indices, ...]
        final_times = times[jnp.arange(times.shape[0]), success_indices]
        jax_mayer = jaxadi.convert(ocp.mayer_objective)
        mayer = jax.jit(jax.vmap(lambda x, t: jax_mayer(x, t)[0].ravel()))
        mayer_costs = mayer(final_states, final_times)
        simulation_costs = lagrange_costs.ravel() + mayer_costs.ravel()
        relative_errors = 100 * (simulation_costs - optimal_costs) / optimal_costs
        return relative_errors

    def _run_rollouts(
        self,
        ocp: OCP,
        bound_states: bool,
        bound_controls: bool,
        initial_states: np.ndarray,
        goals: np.ndarray,
        dt: float,
        n_samples: int,
    ) -> diffrax.Solution:
        simulate = jax.vmap(_simulate, in_axes=(0, 0, None, None, None, None, None))
        state_bounds = (
            (jnp.asarray(ocp.state_bounds[0]), jnp.asarray(ocp.state_bounds[1]))
            if bound_states
            else None
        )
        control_bounds = (
            (jnp.asarray(ocp.control_bounds[0]), jnp.asarray(ocp.control_bounds[1]))
            if bound_controls
            else None
        )
        jax_dynamics = jaxadi.convert(ocp.dynamics)
        jax_lagrangian = jaxadi.convert(ocp.lagrange_objective)
        dynamics = jax.jit(lambda x, u: jax_dynamics(x, u)[0].ravel())
        lagrangian = jax.jit(lambda x, u: jax_lagrangian(x, u)[0].ravel())
        rollouts = simulate(
            jnp.asarray(initial_states),
            jnp.asarray(goals),
            0,
            ocp.time_bounds[1],
            dt,
            (
                state_bounds,
                control_bounds,
                dynamics,
                lagrangian,
                self.action_from_state_and_goal,
            ),
            n_samples,
        )
        return rollouts

    def _sample_tasks_and_optimal_costs(
        self,
        ocp: OCP,
        n_tasks: int,
        random_seed: int | None,
        opt_n_intervals: int,
        sample_initial_final_states: Callable[
            [np.random.Generator], tuple[ArrayLike, ArrayLike]
        ],
        trajopt_results_path: str | None,
        override_fatrop_options: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if trajopt_results_path is not None and Path(trajopt_results_path).exists():
            opts = np.load(trajopt_results_path)
            initial_states, goals, optimal_costs = (
                opts[:, : ocp._nx],
                opts[:, ocp._nx : -1],
                opts[:, -1],
            )
            return initial_states, goals, optimal_costs
        initial_states, goals = (
            np.zeros((n_tasks, ocp._nx)),
            np.zeros((n_tasks, ocp._ng)),
        )
        optimal_costs = np.zeros(n_tasks)
        rng = np.random.default_rng(random_seed)
        n_optimized_tasks = 0
        solver, solver_args = ocp.transcribe(
            np.zeros(ocp._nx),
            np.zeros(ocp._nx),
            n_intervals=opt_n_intervals,
            override_fatrop_options=override_fatrop_options,
        )
        while n_optimized_tasks < n_tasks:
            initial_state, final_state = sample_initial_final_states(rng)
            ocp._set_initial_final_state_constraint(
                initial_state,
                final_state,
                solver_args["lbx"],
                solver_args["ubx"],
                solver_args["lbg"],
                solver_args["ubg"],
            )
            solver_args["x0"] = ocp._get_initial_guess(
                np.asarray(initial_state),
                np.asarray(final_state),
                opt_n_intervals,
            )
            opt_result = solver(**solver_args)  # type: ignore
            if solver.stats()["unified_return_status"] == "SOLVER_RET_SUCCESS":
                optimal_costs[n_optimized_tasks] = opt_result["f"]  # type: ignore
                initial_states[n_optimized_tasks] = np.asarray(initial_state)
                goals[n_optimized_tasks] = (
                    final_state
                    if ocp.state_to_goal is None
                    else np.asarray(ocp.state_to_goal(final_state)).ravel()
                )
                n_optimized_tasks += 1
        if trajopt_results_path is not None and not Path(trajopt_results_path).exists():
            path = Path(trajopt_results_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            opts = np.hstack([initial_states, goals, optimal_costs.reshape(-1, 1)])
            np.save(trajopt_results_path, opts)
        return initial_states, goals, optimal_costs

    def save(self, path: str, format: str = "zip") -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            with open(temp_dir_path / self._ARCHIVE_MODEL, "wb") as model_file:
                eqx.tree_serialise_leaves(model_file, self._model)
            if self._transform_state_goal_params is not None:
                np.save(
                    temp_dir_path / self._ARCHIVE_TRANSFORM_STATE_GOAL_PARAMS,
                    self._transform_state_goal_params,
                )
            if self._model_output_to_control_params is not None:
                np.save(
                    temp_dir_path / self._ARCHIVE_MODEL_OUTPUT_TO_CONTROL_PARAMS,
                    self._model_output_to_control_params,
                )
            shutil.make_archive(path, format, temp_dir)

    @staticmethod
    def load(
        path: str,
        model: eqx.Module,
        transform_state_goal: Callable[
            [jax.Array, jax.Array, jax.Array | None], jax.Array
        ],
        model_output_to_control: (
            Callable[[jax.Array, jax.Array | None], jax.Array] | None
        ) = None,
    ) -> Policy:
        policy = Policy(model, transform_state_goal, model_output_to_control)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            shutil.unpack_archive(path, temp_dir)
            policy._model = eqx.tree_deserialise_leaves(
                temp_dir_path / Policy._ARCHIVE_MODEL, model
            )
            if Path.exists(
                temp_dir_path / Policy._ARCHIVE_MODEL_OUTPUT_TO_CONTROL_PARAMS
            ):
                policy._model_output_to_control_params = np.load(
                    temp_dir_path / Policy._ARCHIVE_MODEL_OUTPUT_TO_CONTROL_PARAMS
                )
            if Path.exists(temp_dir_path / Policy._ARCHIVE_TRANSFORM_STATE_GOAL_PARAMS):
                policy._transform_state_goal_params = np.load(
                    temp_dir_path / Policy._ARCHIVE_TRANSFORM_STATE_GOAL_PARAMS
                )
        return policy
