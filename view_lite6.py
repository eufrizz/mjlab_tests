"""Quick passive viewer for the Lite6 model — no mjlab overhead."""

import mujoco
import mujoco.viewer

from gym_xarmlite6.model import get_spec

spec = get_spec("lite6_gripper_wide.xml")
model = spec.compile()
data = mujoco.MjData(model)

model.opt.disableactuator = 2**2


mujoco.mj_resetDataKeyframe(model, data, 0)

with mujoco.viewer.launch_passive(model, data) as v:
    while v.is_running():
        mujoco.mj_step(model, data)
        v.sync()
