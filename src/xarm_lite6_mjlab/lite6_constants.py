"""UFactory Lite6 constants for mjlab."""

import mujoco
import numpy as np

from mjlab.actuator import XmlPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from gym_xarmlite6.model import get_spec

##
# Collision config.
#
# The Lite6 collision geoms are named: link_base_c, link1_c ... link6_c,
# gripper_lite_body_c, gripper_left_finger, gripper_right_finger.
##

# Gripper-only: arm link geoms disabled from contact solver entirely.
# Faster simulation but no self-collision detection.
GRIPPER_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(".*",),
  contype={
    "(gripper_lite_body_c|gripper_left_finger|gripper_right_finger)": 1,
    ".*": 0,
  },
  conaffinity={
    "(gripper_lite_body_c|gripper_left_finger|gripper_right_finger)": 1,
    ".*": 0,
  },
  condim={
    "(gripper_left_finger|gripper_right_finger)": 4,
    ".*": 3,
  },
  friction={
    "(gripper_left_finger|gripper_right_finger)": (1.0, 5e-3, 1e-4),
    ".*": (0.6,),
  },
)

# Full collision: all geoms active, enabling self-collision detection.
# Slower simulation but more physically accurate.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*",),
  # contype/conaffinity left at XML defaults (all 1).
  condim={
    "(gripper_left_finger|gripper_right_finger)": 4,
    ".*": 3,
  },
  friction={
    "(gripper_left_finger|gripper_right_finger)": (1.0, 5e-3, 1e-4),
    ".*": (0.6,),
  },
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={},
  joint_vel={".*": 0.0},
)


def _get_spec() -> mujoco.MjSpec:
  return get_spec("lite6_gripper_wide.xml")


##
# Articulation.
#
# XmlPositionActuatorCfg wraps the existing group 1 position actuators from the
# XML without adding new ones. This registers the joints as actuated so mjlab's
# action manager can find them, without duplicating any actuator definitions.
##

ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    XmlPositionActuatorCfg(target_names_expr=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")),
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_lite6_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=_get_spec,
    articulation=ARTICULATION,
  )


##
# Action scale.
#
# Map actions from [-1, 1] to full joint range
# Only group 1 (position) actuators are included; group 2 (velocity) and
# group 0 (gripper motor) are excluded as they are not part of the action space.
##

def _compute_action_scale() -> dict[str, float]:
  m = _get_spec().compile()
  scale = {}
  for i in range(m.nu):
    if m.actuator_group[i] != 1:
      continue
    joint_id = m.actuator_trnid[i, 0]
    joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    # TODO: have cut down to half range (/4) instead of full range (/4) for better resolution, less jitter
    joint_range = (m.actuator_ctrlrange[i, 1] - m.actuator_ctrlrange[i, 0]) / 4
    scale[joint_name] = np.float32(joint_range)
  return scale


LITE6_ACTION_SCALE: dict[str, float] = _compute_action_scale()
