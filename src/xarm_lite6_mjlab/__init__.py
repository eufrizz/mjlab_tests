from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  lite6_lift_cube_env_cfg,
  lite6_lift_cube_vision_distillation_env_cfg,
  lite6_lift_cube_vision_env_cfg,
)
from .rl_cfg import (
  lite6_lift_cube_ppo_runner_cfg,
  lite6_lift_cube_vision_distillation_runner_cfg,
  lite6_lift_cube_vision_ppo_runner_cfg,
)
from .runners import MjlabDistillationRunner, MjlabStudentOnPolicyRunner

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-Lite6",
  env_cfg=lite6_lift_cube_env_cfg(),
  play_env_cfg=lite6_lift_cube_env_cfg(play=True),
  rl_cfg=lite6_lift_cube_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-Lite6-Rgb",
  env_cfg=lite6_lift_cube_vision_env_cfg(cam_type="rgb"),
  play_env_cfg=lite6_lift_cube_vision_env_cfg(cam_type="rgb", play=True),
  rl_cfg=lite6_lift_cube_vision_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-Lite6-Depth",
  env_cfg=lite6_lift_cube_vision_env_cfg(cam_type="depth"),
  play_env_cfg=lite6_lift_cube_vision_env_cfg(cam_type="depth", play=True),
  rl_cfg=lite6_lift_cube_vision_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-Lite6-Rgb-Distill",
  env_cfg=lite6_lift_cube_vision_distillation_env_cfg(cam_type="rgb"),
  play_env_cfg=lite6_lift_cube_vision_distillation_env_cfg(cam_type="rgb", play=True),
  rl_cfg=lite6_lift_cube_vision_distillation_runner_cfg(),
  runner_cls=MjlabDistillationRunner,
)

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-Lite6-Rgb-Student",
  env_cfg=lite6_lift_cube_vision_env_cfg(cam_type="rgb"),
  play_env_cfg=lite6_lift_cube_vision_env_cfg(cam_type="rgb", play=True),
  rl_cfg=lite6_lift_cube_vision_ppo_runner_cfg(),
  runner_cls=MjlabStudentOnPolicyRunner,
)
