from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
import mujoco

from mjlab.entity import Entity, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import CameraSensorCfg, ContactMatch, ContactSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg

from .lite6_constants import LITE6_ACTION_SCALE, get_lite6_robot_cfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def get_cube_spec(cube_size: float = 0.015, mass: float = 0.05) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(cube_size,) * 3,
    mass=mass,
    rgba=(0.8, 0.2, 0.2, 1.0),
    solimp=(0.95, 0.99, 0.001, 0.5, 2),
    solref=(0.005, 1),
    friction=(2.0, 5e-3, 1e-4)
  )
  return spec


def gripper_close_reward(
  env: ManagerBasedRlEnv,
  object_name: str,
  std: float,
  gripper_actuator_name: str = "gripper",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward applying a closing (pinching) force to the gripper when the EE is near the cube.

  Shape: reaching(ee, cube) × normalised_pinch_force

  - reaching: Gaussian kernel on EE-to-cube distance, peaks at 1 within ~std of cube.
  - normalised_pinch_force: the motor ctrl signal divided by its max force, clamped to
    [0, 1]. For the motor actuator, ctrl IS the commanded force directly — positive
    values close the gripper, negative values open it.

  asset_cfg must specify site_names for the EE site.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]

  # Reaching: Gaussian over squared EE-to-cube distance.
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)  # (B, 3)
  obj_pos_w = obj.data.root_link_pos_w  # (B, 3)
  reach_sq = torch.sum(torch.square(ee_pos_w - obj_pos_w), dim=-1)  # (B,)
  reaching = torch.exp(-reach_sq / std**2)  # (B,)

  # Pinch force: for the motor actuator, ctrl == commanded force in Newtons.
  # Positive = closing, negative = opening. Normalise by max force and clamp to [0, 1].
  sim = env.unwrapped.sim
  act_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, gripper_actuator_name)
  ctrl_max = float(sim.mj_model.actuator_forcerange[act_id, 1])  # e.g. 10 N
  gripper_ctrl = sim.data.ctrl[:, act_id]  # (B,)
  pinch = gripper_ctrl.clamp(-1, 1.) #(gripper_ctrl / ctrl_max).clamp(0.0, 1.0)
  reward = reaching * pinch
  return reward


def lite6_lift_cube_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_lift_cube_env_cfg()

  cfg.scene.entities = {
    "robot": get_lite6_robot_cfg(),
    "cube": EntityCfg(spec_fn=get_cube_spec),
  }

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = LITE6_ACTION_SCALE

  cfg.observations["actor"].terms["ee_to_cube"].params["asset_cfg"].site_names = (
    "end_effector",
  )
  cfg.rewards["lift"].params["asset_cfg"].site_names = ("end_effector",)
  cfg.rewards["lift"].params["reaching_std"] = 0.1
  cfg.rewards["lift"].weight = 2

  cfg.rewards["lift_precise"].weight = 2

  # Reward closing the gripper when the EE is near the cube.
  # Weight starts at 0 so the policy first learns to reach before being
  # nudged to close the gripper.  Activated after ~100 PPO iterations.
  cfg.rewards["gripper_close"] = RewardTermCfg(
    func=gripper_close_reward,
    weight=0.0,
    params={
      "object_name": "cube",
      "std": 0.1,
      "gripper_actuator_name": "gripper",
      "asset_cfg": SceneEntityCfg("robot", site_names=("end_effector",)),
    },
  )
  cfg.curriculum["gripper_close_weight"] = CurriculumTermCfg(
    func=manipulation_mdp.reward_weight,
    params={
      "reward_name": "gripper_close",
      "weight_stages": [
        {"step": 0, "weight": 0.0},
        {"step": 200 * 24, "weight": 0.05},
      ],
    },
  )

  # Fingertip geom names in the Lite6 XML.
  fingertip_geoms = "(gripper_left_finger|gripper_right_finger)"
  cfg.events["fingertip_friction_slide"].params[
    "asset_cfg"
  ].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  # Configure collision sensor: match the link6 subtree (wrist + gripper).
  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "link6"

  # Restrict joint_pos_limits and joint_vel_hinge to arm joints only.
  # The gripper fingers are not part of the action space and their limits are
  # tiny (±8mm), so including them would spuriously penalise normal gripper motion.
  arm_joints = "(joint1|joint2|joint3|joint4|joint5|joint6)"
  cfg.rewards["joint_pos_limits"].params["asset_cfg"].joint_names = (arm_joints,)
  cfg.rewards["joint_vel_hinge"].params["asset_cfg"].joint_names = (arm_joints,)

  cfg.viewer.body_name = "link_base"

  # Self-collision sensor: fire when any non-EE arm link contacts another
  # non-adjacent robot body.  The secondary is filtered to the robot entity's
  # own subtree, so cube/terrain contacts do NOT trigger this sensor.
  # MuJoCo already suppresses parent-child contacts at the physics level, so
  # only genuinely dangerous self-collisions reach the sensor.
  # Use body-mode primaries for arm links only — gripper_left_finger_base and
  # gripper_right_finger_base are siblings so MuJoCo doesn't filter their contact,
  # meaning the full-subtree approach fires whenever the gripper opens/closes.
  # By excluding the gripper bodies from the primary, only dangerous arm-arm
  # self-collisions terminate the episode.
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(
      mode="body",
      pattern="(link.*|gripper_lite_body_c)",
      entity="robot",
    ),
    secondary=ContactMatch(mode="subtree", pattern="link_base", entity="robot"),
    secondary_policy="first",
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (self_collision_cfg,)

  cfg.terminations["self_collision"] = TerminationTermCfg(
    func=manipulation_mdp.illegal_contact,
    params={"sensor_name": "self_collision"},
  )

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}

    # Higher command resampling frequency for more dynamic play.
    assert cfg.commands is not None
    cfg.commands["lift_height"].resampling_time_range = (4.0, 4.0)

  return cfg


def lite6_lift_cube_vision_env_cfg(
  cam_type: Literal["rgb", "depth"],
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = lite6_lift_cube_env_cfg(play=play)

  camera_names = ["robot/gripper_cam"]
  cam_kwargs = {
    "robot/gripper_cam": {
      "height": 32,
      "width": 32,
    },
  }
  shared_cam_kwargs = dict(
    data_types=(cam_type,),
    enabled_geom_groups=(0, 3),
    use_shadows=False,
    use_textures=True,
  )

  cam_terms = {}
  for cam_name in camera_names:
    cam_cfg = CameraSensorCfg(
      name=cam_name.split("/")[-1],
      camera_name=cam_name,
      **cam_kwargs[cam_name],  # type: ignore[invalid-argument-type]
      **shared_cam_kwargs,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (cam_cfg,)
    param_kwargs: dict[str, Any] = {"sensor_name": cam_cfg.name}
    if cam_type == "depth":
      param_kwargs["cutoff_distance"] = 0.5
      func = manipulation_mdp.camera_depth
    else:
      func = manipulation_mdp.camera_rgb
    cam_terms[f"{cam_name.split('/')[-1]}_{cam_type}"] = ObservationTermCfg(
      func=func, params=param_kwargs
    )

  camera_obs = ObservationGroupCfg(
    terms=cam_terms, enable_corruption=False, concatenate_terms=True
  )
  cfg.observations["camera"] = camera_obs

  # Pop privileged info from actor observations.
  actor_obs = cfg.observations["actor"]
  actor_obs.terms.pop("ee_to_cube")
  actor_obs.terms.pop("cube_to_goal")

  # Add goal_position to actor observations.
  actor_obs.terms["goal_position"] = ObservationTermCfg(
    func=manipulation_mdp.target_position,
    params={
      "command_name": "lift_height",
      "asset_cfg": SceneEntityCfg("robot", site_names=("end_effector",)),
    },
    # NOTE: No noise for goal position.
  )

  return cfg
