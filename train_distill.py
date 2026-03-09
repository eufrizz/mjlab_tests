"""Train RGB student policy via distillation from a state-based teacher checkpoint."""

import re
from dataclasses import asdict, dataclass
from pathlib import Path

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
  teacher_checkpoint: str | None = None
  """Path to a local PPO checkpoint trained on Mjlab-Lift-Cube-Lite6."""
  teacher_wandb_run_path: str | None = None
  """WandB run path to download the teacher checkpoint from, e.g. 'entity/project/run_id'."""
  num_envs: int = 1024
  device: str = "cuda:0"


def _resolve_teacher_checkpoint(cfg: DistillConfig) -> Path:
  """Return a local path to the teacher checkpoint, downloading from WandB if needed."""
  if cfg.teacher_checkpoint is not None and cfg.teacher_wandb_run_path is not None:
    raise ValueError("Specify either --teacher-checkpoint or --teacher-wandb-run-path, not both.")
  if cfg.teacher_checkpoint is None and cfg.teacher_wandb_run_path is None:
    raise ValueError("One of --teacher-checkpoint or --teacher-wandb-run-path is required.")

  if cfg.teacher_checkpoint is not None:
    path = Path(cfg.teacher_checkpoint)
    if not path.exists():
      raise FileNotFoundError(f"Teacher checkpoint not found: {path}")
    return path

  # Download latest checkpoint from WandB.
  import wandb

  run_path = cfg.teacher_wandb_run_path
  run_id = run_path.split("/")[-1]
  cache_dir = Path("logs/wandb_cache") / run_id

  api = wandb.Api()
  run = api.run(run_path)
  names = [f.name for f in run.files() if re.match(r"^model_\d+\.pt$", f.name)]
  if not names:
    raise FileNotFoundError(f"No model_*.pt checkpoints found in wandb run {run_path}")

  latest = max(names, key=lambda x: int(x.split("_")[1].split(".")[0]))
  local_path = cache_dir / latest

  if local_path.exists():
    print(f"[INFO]: Using cached teacher checkpoint: {local_path}")
  else:
    cache_dir.mkdir(parents=True, exist_ok=True)
    run.file(latest).download(str(cache_dir), replace=True)
    print(f"[INFO]: Downloaded teacher checkpoint: {latest} from {run_path}")

  return local_path


def main() -> None:
  cfg = tyro.cli(DistillConfig)
  configure_torch_backends()

  teacher_path = _resolve_teacher_checkpoint(cfg)

  env_cfg = lite6_lift_cube_vision_distillation_env_cfg(cam_type="rgb")
  env_cfg.scene.num_envs = cfg.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  env = RslRlVecEnvWrapper(env)

  train_cfg = lite6_lift_cube_vision_distillation_runner_cfg()
  log_dir = f"logs/rsl_rl/{train_cfg.experiment_name}"

  runner = MjlabDistillationRunner(env, asdict(train_cfg), log_dir=log_dir, device=cfg.device)

  # Load teacher weights from the PPO checkpoint.
  # Distillation.load() automatically maps actor_state_dict -> teacher when
  # load_cfg={"teacher": True} is passed.
  # strict=False: the PPO actor is stochastic (has "std" in state_dict) but the
  # teacher model is non-stochastic, so strict=True would raise an unexpected-key error.
  runner.load(str(teacher_path), load_cfg={"teacher": True}, strict=False, map_location=cfg.device)

  runner.learn(num_learning_iterations=train_cfg.max_iterations)

  env.close()


if __name__ == "__main__":
  main()
