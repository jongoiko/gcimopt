import os
from typing import Callable
from typing import Iterator

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.model_selection import train_test_split

from .ocp import OCP


class TrajectoryBuffer:
    _ocp: OCP
    _X_train: jax.Array
    _X_val: jax.Array
    _y_train: jax.Array
    _y_val: jax.Array

    transform_states_goals_params: jax.Array | None
    transform_controls_params: jax.Array | None

    def __init__(
        self,
        ocp: OCP,
        trajectories_dir: str,
        transform_states_goals: Callable[
            [jax.Array, jax.Array, jax.Array | None, bool],
            tuple[jax.Array, jax.Array | None],
        ],
        transform_controls: Callable[
            [jax.Array, jax.Array | None, bool], tuple[jax.Array, jax.Array | None]
        ],
        validation_proportion: float = 0.1,
        goal_relabeling_augment: bool = True,
        random_seed: int | None = None,
    ) -> None:
        if validation_proportion < 0 or validation_proportion > 1:
            raise ValueError("validation_proportion must be a float between 0 and 1")
        self._ocp = ocp
        states, controls, _ = self._read_trajectories_from_dir(trajectories_dir)
        goals = self._get_goals_from_states(states)
        (
            states_train,
            states_val,
            controls_train,
            controls_val,
            goals_train,
            goals_val,
        ) = train_test_split(
            states,
            controls,
            goals,
            test_size=validation_proportion,
            random_state=random_seed,
        )
        if goal_relabeling_augment:
            states_train, controls_train, goals_train = (
                self._augment_by_intermediate_goals(
                    states_train, controls_train, goals_train
                )
            )
        else:
            # We discard the final state since it has no associated control; the
            # goal in every point in each trajectory is then the final state of
            # the trajectory.
            final_state_goals_train = goals_train[:, [-1], :]
            states_train, controls_train, goals_train = [
                array[:, :-1, :]
                for array in [states_train, controls_train, goals_train]
            ]
            goals_train = np.repeat(
                final_state_goals_train, goals_train.shape[1], axis=1
            )
        final_state_goals_val = goals_val[:, [-1], :]
        states_val, controls_val, goals_val = [
            array[:, :-1, :] for array in [states_val, controls_val, goals_val]
        ]
        goals_val = np.repeat(final_state_goals_val, goals_val.shape[1], axis=1)
        states_train = states_train.reshape(-1, ocp._nx)
        controls_train = controls_train.reshape(-1, ocp._nu)
        goals_train = goals_train.reshape(-1, goals_train.shape[-1])
        states_val = states_val.reshape(-1, ocp._nx)
        controls_val = controls_val.reshape(-1, ocp._nu)
        goals_val = goals_val.reshape(-1, goals_val.shape[-1])
        # First transform call: fit and transform
        self._X_train, self.transform_states_goals_params = transform_states_goals(
            states_train, goals_train, None, True
        )
        self._y_train, self.transform_controls_params = transform_controls(
            controls_train, None, True
        )
        # Second transform call: only transform. This will typically use
        # information from the first call (e.g. mean and variance of the
        # training set)
        self._X_val, _ = transform_states_goals(
            states_val, goals_val, self.transform_states_goals_params, False
        )
        self._y_val, _ = transform_controls(
            controls_val, self.transform_controls_params, False
        )
        assert (
            self._X_train.shape[0] == self._y_train.shape[0]
            and self._X_val.shape[0] == self._y_val.shape[0]
        )

    @property
    def train_split_size(self) -> int:
        return self._X_train.shape[0]

    @property
    def val_split_size(self) -> int:
        return self._X_val.shape[0]

    def _read_trajectories_from_dir(
        self, trajectories_dir: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        file_paths = [
            full_path
            for file in os.listdir(trajectories_dir)
            if os.path.isfile(full_path := os.path.join(trajectories_dir, file))
        ]
        raw_data = np.vstack([np.load(path) for path in file_paths])
        nx, nu = self._ocp._nx, self._ocp._nu
        states, controls, time = (
            raw_data[..., :nx],
            raw_data[..., -nu:],
            raw_data[..., [nx]],
        )
        return states, controls, time

    def _get_goals_from_states(self, states: np.ndarray) -> np.ndarray:
        if self._ocp.state_to_goal is None:
            return states
        nx = self._ocp._nx
        intermediate_states = states.reshape(-1, nx)
        state_to_goal = self._ocp.state_to_goal.map(intermediate_states.shape[0])
        intermediate_goals = np.asarray(state_to_goal(intermediate_states.T)).T
        return intermediate_goals.reshape(states.shape[0], states.shape[1], -1)

    def _augment_by_intermediate_goals(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        goals: np.ndarray,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # The last state in each trajectory has no associated control, so we
        # discard it.
        final_state_goals = goals[:, [-1], :]
        states, controls, goals = [
            array[:, :-1, :] for array in [states, controls, goals]
        ]
        # Convert the arrays to Jax arrays, moving them into the GPU if there is
        # one.
        jax.clear_caches()
        jnp_states = jnp.asarray(states)
        jnp_controls = jnp.asarray(controls)
        jnp_goals = jnp.asarray(goals)
        jnp_final_state_goals = jnp.asarray(final_state_goals)
        # Create state-goal/action pairs where each subsequent intermediate
        # state in the trajectory is treated as a goal.
        n_segments = states.shape[1]
        jnp_states, jnp_controls = [
            jnp.repeat(array, jnp.arange(n_segments, 0, -1), axis=1)
            for array in [jnp_states, jnp_controls]
        ]
        jnp_goals = jnp.hstack(
            [
                jnp.hstack([jnp_goals[:, i:, :], jnp_final_state_goals])
                for i in range(1, n_segments + 1)
            ]
        )
        assert jnp_goals.shape[:2] == jnp_goals.shape[:2] == jnp_controls.shape[:2]
        return jnp_states, jnp_controls, jnp_goals

    def _training_set_iterate_one_epoch(
        self, batch_size: int, random_generator: np.random.Generator
    ) -> Iterator[tuple[jax.Array, jax.Array]]:
        return self._iterate_over_split(
            self._X_train, self._y_train, batch_size, random_generator
        )

    def _validation_set_iterate(
        self, batch_size: int, random_generator: np.random.Generator
    ) -> Iterator[tuple[jax.Array, jax.Array]]:
        return self._iterate_over_split(
            self._X_val, self._y_val, batch_size, random_generator
        )

    def _iterate_over_split(
        self,
        X: jax.Array,
        y: jax.Array,
        batch_size: int,
        random_generator: np.random.Generator,
    ) -> Iterator[tuple[jax.Array, jax.Array]]:
        num_samples = X.shape[0]
        batch_size = min(batch_size, num_samples)
        indices = np.arange(num_samples)
        perm = random_generator.permutation(indices)
        start = 0
        end = batch_size
        while end <= num_samples:
            batch_perm = perm[start:end]
            yield (X[batch_perm], y[batch_perm])
            start = end
            end = start + batch_size
