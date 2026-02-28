"""Quick passive viewer for the Lite6 model — no mjlab overhead."""

import argparse
import mujoco
import mujoco.viewer
import pathlib
import numpy as np

from gym_xarmlite6.model import get_spec
from xarm_lite6_mjlab.env_cfgs import get_cube_spec


def geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    return name if name else f"geom{geom_id}"

def print_contacts(model, data, step):
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = geom_name(model, c.geom1)
        g2 = geom_name(model, c.geom2)

        if g1 == "terrain" or g2 == "terrain":
            continue
        f = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, f)

        # contact frame axes in world coords
        normal    = c.frame[0:3]   # points from geom2 into geom1
        tangent1  = c.frame[3:6]
        tangent2  = c.frame[6:9]

        # friction vector in world frame
        friction_world = f[1]*tangent1 + f[2]*tangent2
        
        print(f"{step} {g1} ↔ {g2}  dist={c.dist:+.4f} normal={f[0]:.3f}N  |friction|={np.linalg.norm(friction_world):.3f}N")
        print(f"  friction_Z={friction_world[2]:.3f}N")   # vertical component


parser = argparse.ArgumentParser()
parser.add_argument("model_path", nargs="?", default=None, help="Path to compiled .mjb model")
parser.add_argument("--no-cube", action="store_true", help="Don't add the cube to the scene")
parser.add_argument("--contacts", action="store_true", help="Print contacts")
parser.add_argument("--lift", action="store_true", help="Execute lift and quit")
parser.add_argument("--video", metavar="PATH", default=None, help="Record to video (e.g. --video out.mp4)")
parser.add_argument("--fps", type=int, default=25, help="Video frame rate (default: 25)")
parser.add_argument("--res", nargs=2, type=int, default=[1280, 720], metavar=("W", "H"), help="Video resolution (default: 1280 720)")
args = parser.parse_args()

if args.model_path is None:
    spec = get_spec("lite6_gripper_wide.xml")
    terrain = spec.worldbody.add_geom(
        name="terrain", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.01)
    )
    terrain.material = "groundplane"
    ground_tex = spec.add_texture()
    ground_tex.name = "groundplane"
    ground_tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    ground_tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    ground_tex.mark = mujoco.mjtMark.mjMARK_EDGE
    ground_tex.rgb1 = [0.2, 0.3, 0.4]
    ground_tex.rgb2 = [0.1, 0.2, 0.3]
    ground_tex.markrgb = [0.8, 0.8, 0.8]
    ground_tex.width = 300
    ground_tex.height = 300
    ground_mat = spec.add_material()
    ground_mat.name = "groundplane"
    ground_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"
    ground_mat.texuniform = True
    ground_mat.texrepeat = [5.0, 5.0]
    ground_mat.reflectance = 0.2
    sky = spec.add_texture()
    sky.name = "skybox"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1 = [0.2, 0.4, 0.8]   # deep blue at top
    sky.rgb2 = [0.8, 0.9, 1.0]   # pale at horizon
    sky.width = 512
    sky.height = 512
    if not args.no_cube:
        cube_spec = get_cube_spec()
        frame = spec.worldbody.add_frame()
        spec.attach(child=cube_spec, prefix="cube/", frame=frame)
    model = spec.compile()
    data = mujoco.MjData(model)
    model.vis.headlight.ambient = [0.3, 0.3, 0.3]
    model.vis.headlight.diffuse = [0.6, 0.6, 0.6]
else:
    model_path = pathlib.Path.cwd() / args.model_path
    model = mujoco.MjModel.from_binary_path(str(model_path))
    data = mujoco.MjData(model)

model.opt.disableactuator = 2**2

mujoco.mj_resetDataKeyframe(model, data, 0)

cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube/cube")
if cube_body_id >= 0:
    jnt_adr = model.body_jntadr[cube_body_id]
    if jnt_adr >= 0:
        qpos_adr = model.jnt_qposadr[jnt_adr]
        data.qpos[qpos_adr : qpos_adr + 3] = [0.3, 0.0, 0.007]  # rest on ground in front of robot

print("Gripper actuator id:", mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper"))
# for i in range(model.nbody):
#     name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
#     jnt_adr = model.body_jntadr[i]
#     if jnt_adr >= 0:
#         qpos_adr = model.jnt_qposadr[jnt_adr]
#         print(i, name, jnt_adr, qpos_adr)=

home_pos = [0, 0.9, 1.15, 0, 0.18, 0, -1]
data.qpos[:6] = home_pos[:6]
data.ctrl[:7] = home_pos
steps = 20000
control = np.tile(home_pos, (steps, 1))
# turn gripper on
control[500:, -1] = [1]
# lift
control[1000:, 2] = 1.5

renderer = None
frames = []
frame_interval = 1  # steps between captured frames
if args.video:
    import mediapy as media
    w, h = args.res
    model.vis.global_.offwidth = w
    model.vis.global_.offheight = h
    renderer = mujoco.Renderer(model, height=h, width=w)
    frame_interval = max(1, round(1.0 / (args.fps * model.opt.timestep)))
    print(f"Recording {w}×{h} @ {args.fps} fps → {args.video}  (every {frame_interval} steps)")

with mujoco.viewer.launch_passive(model, data) as v:
    # Track the end-effector body
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    if site_id >= 0:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = model.site_bodyid[site_id]
        v.cam.distance = 0.3
        v.cam.azimuth = 170.0
        v.cam.elevation = 0.0

    step = 0
    while v.is_running():
        if args.lift:
            data.ctrl[:7] = control[min(step, steps-1)]
        mujoco.mj_step(model, data)
        v.sync()
        if renderer is not None and step % frame_interval == 0:
            renderer.update_scene(data, camera=v.cam)
            frames.append(renderer.render())
        if args.contacts:
            print_contacts(model, data, step)
        if args.lift and step >= steps - 1:
            break
        step += 1

    if frames:
        print(f"Writing {len(frames)} frames...")
        media.write_video(args.video, frames, fps=args.fps)
        renderer.close()
        print(f"Saved → {args.video}")
