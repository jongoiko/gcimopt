import datetime
import sys
import time

import hydra
import jax
from gcimopt.generate import DatasetGenerator
from gcimopt.policies import Policy
from gcimopt.trajectory_buffer import TrajectoryBuffer
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter

from .cartpole import experiment as cartpole_experiment
from .double_integrator import experiment as double_integrator_experiment
from .drone2d import experiment as drone2d_experiment
from .drone3d import experiment as drone3d_experiment
from .experiment import Experiment
from .robot_reach import experiment as robot_reach_experiment

EXPERIMENTS = {
    "double_integrator": double_integrator_experiment,
    "robot_reach": robot_reach_experiment,
    "cartpole": cartpole_experiment,
    "drone2d": drone2d_experiment,
    "drone3d": drone3d_experiment,
}


def generate_dataset(exp: Experiment, cfg: DictConfig) -> None:
    generate_cfg = cfg.experiment.generate
    print("Generating dataset of optimal trajectories.")
    print(OmegaConf.to_yaml(generate_cfg, resolve=True))
    start = time.time()
    gen = DatasetGenerator(exp.ocp, exp.sample_initial_final_states, cfg.seed)
    fatrop_options = (
        OmegaConf.to_container(generate_cfg.fatrop_options)
        if "fatrop_options" in generate_cfg
        else {}
    )
    gen.generate(
        generate_cfg.n_trajectories,
        generate_cfg.trajectories_dir,
        generate_cfg.n_grid_points,
        generate_cfg.n_processes,
        fatrop_options,  # type: ignore
    )
    end = time.time()
    print(f"Dataset generation finished in {datetime.timedelta(seconds=end - start)}")


def _get_evaluate_args(exp: Experiment, cfg: DictConfig) -> dict:
    eval_cfg = cfg.experiment.evaluate
    fatrop_options = (
        OmegaConf.to_container(eval_cfg.fatrop_options)
        if "fatrop_options" in eval_cfg
        else {}
    )
    return dict(
        ocp=exp.ocp,
        n_tasks=eval_cfg.n_tasks,
        sample_initial_final_states=exp.sample_initial_final_states,
        goal_reached=exp.goal_reached,
        simulation_bound_controls=eval_cfg.simulation_bound_controls,
        simulation_bound_states=eval_cfg.simulation_bound_states,
        simulation_dt=eval_cfg.simulation_dt,
        simulation_n_samples=eval_cfg.simulation_n_samples,
        trajopt_results_path=eval_cfg.trajopt_results_path,
        opt_n_intervals=eval_cfg.opt_n_intervals,
        random_seed=cfg.seed,
        override_fatrop_options=fatrop_options,
    )


def train_policy(experiment_name: str, exp: Experiment, cfg: DictConfig) -> None:
    train_cfg = cfg.experiment.train
    print(f"Training policy on trajectories in {train_cfg.trajectories_dir}")
    print(OmegaConf.to_yaml(train_cfg, resolve=True))
    print("Policy model:")
    print(OmegaConf.to_yaml(cfg.experiment.policy_model))
    trajectory_buffer = TrajectoryBuffer(
        exp.ocp,
        train_cfg.trajectories_dir,
        exp.buffer_transform_states_goals,
        exp.buffer_transform_controls,
        train_cfg.validation_proportion,
        train_cfg.goal_relabeling_augment,
        random_seed=cfg.seed,
    )
    print("Dataset size:")
    print(f"    Training:   {trajectory_buffer.train_split_size}")
    print(f"    Validation: {trajectory_buffer.val_split_size}")
    print(
        f"    Total:      {trajectory_buffer.train_split_size + trajectory_buffer.val_split_size}"
    )
    key = jax.random.PRNGKey(cfg.seed)
    key, subkey = jax.random.split(key, 2)
    policy_factory = instantiate(cfg.experiment.policy_model.module, _partial_=True)
    policy_model = policy_factory(key=subkey)
    policy = Policy(
        policy_model,
        exp.transform_state_goal,
        exp.model_output_to_control,
        trajectory_buffer.transform_states_goals_params,
        trajectory_buffer.transform_controls_params,
    )
    print(f"The policy model has {policy.num_model_params} trainable parameters.")
    tb_writer_logdir = (
        f"{train_cfg.tensorboard_dir}/{experiment_name}_"
        + f"{cfg.experiment.policy_model.name}_{datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    tb_writer = SummaryWriter(tb_writer_logdir)
    epoch_callback, evaluate_every_n_epochs = None, 1
    if "evaluate_every_n_epochs" in train_cfg:
        eval_args = _get_evaluate_args(exp, cfg)

        def evaluate_callback(policy: Policy, epoch: int) -> None:
            success_rate, relative_opt_errors = policy.evaluate(**eval_args)
            tb_writer.add_scalar("epoch_policy_eval/success_rate", success_rate, epoch)
            tb_writer.add_scalar(
                "epoch_policy_eval/mean_relative_opt_error",
                relative_opt_errors.mean(),
                epoch,
            )

        epoch_callback = evaluate_callback
        evaluate_every_n_epochs = train_cfg.evaluate_every_n_epochs
    start = time.time()
    policy.train(
        trajectory_buffer,
        train_cfg.n_epochs,
        train_cfg.batch_size,
        train_cfg.adam_lr,
        tb_writer=tb_writer,
        epoch_callback=epoch_callback,
        callback_every_n_epochs=evaluate_every_n_epochs,
        print_loss_every_n_epochs=train_cfg.print_loss_every_n_epochs,
        random_seed=cfg.seed,
    )
    end = time.time()
    policy.save(train_cfg.policy_save_path)
    print(f"Policy training finished in {datetime.timedelta(seconds=end - start)}")


def evaluate_policy(exp: Experiment, cfg: DictConfig) -> None:
    eval_cfg = cfg.experiment.evaluate
    print(f"Evaluating policy {eval_cfg.policy_path}")
    print(OmegaConf.to_yaml(eval_cfg))
    eval_args = _get_evaluate_args(exp, cfg)
    key = jax.random.PRNGKey(cfg.seed)
    key, subkey = jax.random.split(key, 2)
    policy_factory = instantiate(eval_cfg.policy_model.module, _partial_=True)
    policy_model = policy_factory(key=subkey)
    policy = Policy.load(
        eval_cfg.policy_path,
        policy_model,
        exp.transform_state_goal,
        exp.model_output_to_control,
    )
    print(f"The policy model has {policy.num_model_params} trainable parameters.")
    start = time.time()
    success_rate, relative_opt_errors = policy.evaluate(**eval_args)
    end = time.time()
    print(f"Success rate: {success_rate:.3f}%")
    print(f"Average relative cost error: {relative_opt_errors.mean():.3f}%")
    print(f"Policy evaluation finished in {datetime.timedelta(seconds=end - start)}")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    if "experiment" not in cfg or cfg.experiment.name not in EXPERIMENTS:
        sys.exit(
            "it is required to specify an experiment: +experiment=EXP "
            + f"where EXP is one of {list(EXPERIMENTS.keys())}"
        )
    exp = EXPERIMENTS[cfg.experiment.name]
    generate = "generate" in cfg and cfg.generate
    train = "train" in cfg and cfg.train
    eval = "evaluate" in cfg and cfg.evaluate
    if not generate and not train and not eval:
        sys.exit("at least one of [generate, train, eval] must be set")
    OmegaConf.register_new_resolver("jax_nn", lambda f: getattr(jax.nn, f))
    if generate:
        generate_dataset(exp, cfg)
    if train:
        train_policy(cfg.experiment.name, exp, cfg)
    if eval:
        evaluate_policy(exp, cfg)


if __name__ == "__main__":
    main()
