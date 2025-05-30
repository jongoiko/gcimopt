from dataclasses import dataclass
from typing import Callable

import jax
import numpy as np
from numpy.typing import ArrayLike

from gcimopt.ocp import OCP


@dataclass
class Experiment:
    ocp: OCP
    sample_initial_final_states: Callable[
        [np.random.Generator], tuple[ArrayLike, ArrayLike]
    ]
    transform_state_goal: Callable[[jax.Array, jax.Array, jax.Array | None], jax.Array]
    model_output_to_control: Callable[[jax.Array, jax.Array | None], jax.Array] | None
    buffer_transform_states_goals: Callable[
        [jax.Array, jax.Array, jax.Array | None, bool],
        tuple[jax.Array, jax.Array | None],
    ]
    buffer_transform_controls: Callable[
        [jax.Array, jax.Array | None, bool], tuple[jax.Array, jax.Array | None]
    ]
    goal_reached: Callable[[jax.Array, jax.Array], bool]
