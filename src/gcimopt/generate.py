from collections.abc import Callable
from pathlib import Path

import casadi as ca
import numpy as np
import pathos
from numpy.typing import ArrayLike

from .ocp import OCP


class DatasetGenerator:
    _MIN_SUCCESS_RATE_WARNING = 0.6

    _ocp: OCP
    _sample_initial_final_states: Callable[
        [np.random.Generator], tuple[ArrayLike, ArrayLike]
    ]
    _random_seed: int | None

    def __init__(
        self,
        ocp: OCP,
        sample_initial_final_states: Callable[
            [np.random.Generator], tuple[ArrayLike, ArrayLike]
        ],
        random_seed: int | None = None,
    ) -> None:
        self._ocp = ocp
        self._sample_initial_final_states = sample_initial_final_states
        self._random_seed = random_seed

    def generate(
        self,
        n_trajectories: int,
        output_dir: str,
        n_intervals: int,
        n_processes: int | None = None,
        override_fatrop_options: dict = {},
    ) -> None:
        """
        Solve a large number of OCPs with different initial and final states, and store the results.

        Arguments:
        n_trajectories -- the total number of trajectories to optimize
        output_dir -- output directory of optimized trajectories
        n_intervals -- multiple shooting interval number
        n_processes -- number of processes to spawn; if None, as many as the number of CPUs will be spawned
        """
        n_processes = pathos.helpers.cpu_count() if n_processes is None else n_processes
        path = self._get_dir_path(output_dir)
        nx = self._ocp._nx
        solver, solver_args = self._ocp.transcribe(
            ca.DM.zeros(nx),
            ca.DM.zeros(nx),
            n_intervals,
            override_fatrop_options=override_fatrop_options,
        )
        solution_size = solver(**solver_args)["x"].numel()

        def worker(proc_num: int, n_trajs: int, generator: np.random.Generator) -> None:
            def log(msg: str) -> None:
                print(f"Process {proc_num + 1}/{n_processes}: {msg}")

            log(f"optimizing {n_trajs} trajectories.")
            if n_trajs == 0:
                log("no trajectories to optimize, exiting.")
                return
            x0, lbx, ubx, lbg, ubg = [
                solver_args[key] for key in ["x0", "lbx", "ubx", "lbg", "ubg"]
            ]
            trajectories = np.empty((solution_size, n_trajs))
            n_fails = 0
            n_optimized_trajectories = 0
            while n_optimized_trajectories < n_trajs:
                initial_state, final_state = self._sample_initial_final_states(
                    generator
                )
                self._ocp._set_initial_final_state_constraint(
                    initial_state, final_state, lbx, ubx, lbg, ubg
                )
                x0 = self._ocp._get_initial_guess(
                    np.asarray(initial_state),
                    np.asarray(final_state),
                    n_intervals,
                )
                success = True
                try:
                    sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
                    result = np.array(sol["x"].full()).ravel()
                    success = (
                        solver.stats()["unified_return_status"] == "SOLVER_RET_SUCCESS"
                    )
                except Exception:
                    success = False
                if success:
                    trajectories[:, n_optimized_trajectories] = result
                    n_optimized_trajectories += 1
                else:
                    n_fails += 1
                if n_fails + n_optimized_trajectories > 0:
                    success_percentage = (
                        100
                        * n_optimized_trajectories
                        / (n_optimized_trajectories + n_fails)
                    )
                    if success_percentage <= self._MIN_SUCCESS_RATE_WARNING * 100:
                        log(
                            f"only {success_percentage:.2f}% of the optimizations "
                            + "so far have succeeded."
                        )
            log(f"optimized {n_trajs} trajectories.")
            self._save_optimized_trajectories(trajectories, path, f"{proc_num + 1}")

        n_trajectories_per_process = n_trajectories // n_processes
        n_trajs = n_processes * [n_trajectories_per_process]
        n_trajs[-1] += n_trajectories - (n_trajectories // n_processes) * n_processes
        generator = np.random.default_rng(self._random_seed)
        random_generators = generator.spawn(n_processes)
        with pathos.multiprocessing.ProcessPool(n_processes) as pool:
            pool.map(worker, range(n_processes), n_trajs, random_generators)

    def _get_dir_path(self, output_dir: str) -> Path:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _save_optimized_trajectories(
        self, w: np.ndarray, output_dir_path: Path, file_name: str
    ) -> None:
        nx, nu = self._ocp._nx, self._ocp._nu
        n_trajectories = w.shape[1]
        # The last state in the trajectory is special since it doesn't have an
        # associated control: it represents the final state (goal) with which we
        # condition the policy
        w = np.concatenate([w, np.zeros((nu, n_trajectories))]).T
        block_size = nx + nu + 1
        w = w.reshape(n_trajectories, -1, block_size)
        # The second axis of w is now the time step; we record the time instant
        # at each step
        times = w[:, 0, nx]
        w[:, :, nx] = np.linspace(np.zeros(n_trajectories), times, w.shape[1]).T
        np.save(output_dir_path / f"{file_name}.npy", w)
