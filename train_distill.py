"""Train RGB student policy via distillation from a state-based teacher checkpoint."""

from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

import mjlab.tasks  # noqa: F401 — populate task registry
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends

from xarm_lite6_mjlab.env_cfgs import lite6_lift_cube_vision_distillation_env_cfg
from xarm_lite6_mjlab.rl_cfg import lite6_lift_cube_vision_distillation_runner_cfg
from xarm_lite6_mjlab.runners import MjlabDistillationRunner


@dataclass(frozen=True)
class DistillConfig:
  teacher_checkpoint: str
  """Path to a PPO checkpoint trained on Mjlab-Lift-Cube-Lite6 (state-based)."""
  num_envs: int = 1024
  device: str = "cuda:0"


def main() -> None:
  cfg = tyro.cli(DistillConfig)
  configure_torch_backends()

  teacher_path = Path(cfg.teacher_checkpoint)
  if not teacher_path.exists():
    raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")

  env_cfg = lite6_lift_cube_vision_distillation_env_cfg(cam_type="rgb")
  env_cfg.scene.num_envs = cfg.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  env = RslRlVecEnvWrapper(env)

  train_cfg = lite6_lift_cube_vision_distillation_runner_cfg()
  log_dir = f"logs/rsl_rl/{train_cfg['experiment_name']}"

  runner = MjlabDistillationRunner(env, train_cfg, log_dir=log_dir, device=cfg.device)

  # Load teacher weights from the PPO checkpoint.
  # Distillation.load() automatically maps actor_state_dict -> teacher when
  # load_cfg={"teacher": True} is passed.
  runner.load(str(teacher_path), load_cfg={"teacher": True}, strict=True, map_location=cfg.device)

  runner.learn(num_learning_iterations=train_cfg["max_iterations"])

  env.close()


if __name__ == "__main__":
  main()
