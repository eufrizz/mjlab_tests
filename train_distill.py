"""Train RGB student policy via distillation from a state-based teacher checkpoint."""

import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import tyro

import mjlab
import mjlab.tasks  # noqa: F401 — populate task registry
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags

from xarm_lite6_mjlab.env_cfgs import lite6_lift_cube_vision_distillation_env_cfg
from xarm_lite6_mjlab.rl_cfg import (
  RslRlDistillationRunnerCfg,
  lite6_lift_cube_vision_distillation_runner_cfg,
)
from xarm_lite6_mjlab.runners import MjlabDistillationRunner


@dataclass(frozen=True)
class DistillConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlDistillationRunnerCfg
  teacher_checkpoint: str | None = None
  """Path to a local PPO checkpoint trained on Mjlab-Lift-Cube-Lite6."""
  teacher_wandb_run_path: str | None = None
  """WandB run path to download the teacher checkpoint from, e.g. 'entity/project/run_id'."""
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000


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


def run_distill(cfg: DistillConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"

  configure_torch_backends()

  teacher_path = _resolve_teacher_checkpoint(cfg)

  print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  if cfg.video:
    from mjlab.utils.wrappers import VideoRecorder

    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner = MjlabDistillationRunner(env, agent_cfg, str(log_dir), device)

  add_wandb_tags(cfg.agent.wandb_tags)
  dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
  dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  # Load teacher weights from the PPO checkpoint.
  # Distillation.load() automatically maps actor_state_dict -> teacher when
  # load_cfg={"teacher": True} is passed.
  # strict=False: the PPO actor is stochastic (has "std" in state_dict) but the
  # teacher model is non-stochastic, so strict=True would raise an unexpected-key error.
  runner.load(str(teacher_path), load_cfg={"teacher": True}, strict=False, map_location=device)

  runner.learn(num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True)

  env.close()


def main() -> None:
  default = DistillConfig(
    env=lite6_lift_cube_vision_distillation_env_cfg(cam_type="rgb"),
    agent=lite6_lift_cube_vision_distillation_runner_cfg(),
  )
  cfg = tyro.cli(DistillConfig, default=default, config=mjlab.TYRO_FLAGS)

  log_root_path = Path("logs") / "rsl_rl" / cfg.agent.experiment_name
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if cfg.agent.run_name:
    log_dir_name += f"_{cfg.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  selected_gpus, _ = select_gpus(cfg.gpu_ids)
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  run_distill(cfg, log_dir)


if __name__ == "__main__":
  main()
