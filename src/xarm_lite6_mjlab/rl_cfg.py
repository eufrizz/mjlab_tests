from dataclasses import dataclass, field

from mjlab.rl import (
  RslRlBaseRunnerCfg,
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

_CNN_CFG = {
  "output_channels": [16, 32],
  "kernel_size": [5, 3],
  "stride": [2, 2],
  "padding": "zeros",
  "activation": "elu",
  "max_pool": False,
  "global_pool": "none",
  "spatial_softmax": True,
  "spatial_softmax_temperature": 1.0,
}
_SPATIAL_SOFTMAX_CLASS = "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"


@dataclass
class RslRlDistillationAlgorithmCfg:
  num_learning_epochs: int = 1
  gradient_length: int = 15
  learning_rate: float = 1e-3
  max_grad_norm: float = 1.0
  loss_type: str = "mse"
  optimizer: str = "adam"
  class_name: str = "Distillation"


@dataclass
class RslRlDistillationRunnerCfg(RslRlBaseRunnerCfg):
  """Runner config for distillation (student/teacher) training."""

  class_name: str = "DistillationRunner"
  student: RslRlModelCfg = field(default_factory=lambda: RslRlModelCfg(stochastic=True))
  teacher: RslRlModelCfg = field(default_factory=lambda: RslRlModelCfg(stochastic=False))
  algorithm: RslRlDistillationAlgorithmCfg = field(default_factory=RslRlDistillationAlgorithmCfg)


def lite6_lift_cube_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      stochastic=True,
      init_noise_std=1.0,
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      stochastic=False,
      init_noise_std=1.0,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="lite6_lift_cube",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=2_000,
  )


def lite6_lift_cube_vision_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
      stochastic=True,
      init_noise_std=1.0,
      cnn_cfg=_CNN_CFG,
      class_name=_SPATIAL_SOFTMAX_CLASS,
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
      stochastic=False,
      cnn_cfg=_CNN_CFG,
      class_name=_SPATIAL_SOFTMAX_CLASS,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="lite6_lift_cube_vision",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=3_000,
    obs_groups={
      "actor": ("actor", "camera"),
      "critic": ("critic", "camera"),
    },
  )


def lite6_lift_cube_vision_distillation_runner_cfg() -> RslRlDistillationRunnerCfg:
  """Distillation runner config: RGB student distilled from a state-based teacher.

  The teacher uses the 'privileged' obs group (full state: joint_pos, joint_vel,
  ee_to_cube, cube_to_goal, actions) and its weights are loaded from a PPO
  checkpoint trained on Mjlab-Lift-Cube-Lite6.

  The student uses ('actor', 'camera') — limited proprioception + RGB — and
  is trained by minimising MSE against the teacher's actions.
  """
  return RslRlDistillationRunnerCfg(
    student=RslRlModelCfg(
      class_name=_SPATIAL_SOFTMAX_CLASS,
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
      stochastic=True,
      cnn_cfg=_CNN_CFG,
    ),
    teacher=RslRlModelCfg(
      class_name="MLPModel",
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      stochastic=False,
    ),
    algorithm=RslRlDistillationAlgorithmCfg(
      num_learning_epochs=1,
      gradient_length=15,
      learning_rate=1e-3,
      max_grad_norm=1.0,
      loss_type="mse",
      optimizer="adam",
    ),
    obs_groups={
      "student": ("actor", "camera"),
      "teacher": ("privileged",),
    },
    experiment_name="lite6_lift_cube_rgb_distill",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=3_000,
  )
