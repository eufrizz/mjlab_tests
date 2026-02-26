"""Replay a NaN guard dump using the bundled .mjb model.

Usage:
    python replay_nan_dump.py [dump.npz]   # defaults to nan_dump_latest.npz

The model is loaded from the path stored in the dump's _metadata['model_file'].
If that path is relative, the script searches next to the .npz first, then in
/tmp/mjlab/nan_dumps/ (the default NanGuard output directory).

Viewer controls:
    Space       pause / resume
    [ / ]       slow down / speed up playback
    r           restart from beginning
    Esc         quit
"""

import sys
import time
import pathlib
import numpy as np
import mujoco
import mujoco.viewer

# ── Config ────────────────────────────────────────────────────────────────────

DUMP_PATH   = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "nan_dump_latest.npz")
DEFAULT_FPS = 25

# ── Load dump ─────────────────────────────────────────────────────────────────

data = np.load(DUMP_PATH, allow_pickle=True)
meta = data["_metadata"].item()

print("=== NaN Dump ===")
for k, v in meta.items():
    print(f"  {k}: {v}")
print()

step_keys = sorted(k for k in data.keys() if k.startswith("states_step_"))
states    = np.stack([data[k][0] for k in step_keys])   # (T, state_dim)
step_nums = [int(k.split("_")[-1]) for k in step_keys]
T         = len(states)

print(f"Loaded {T} states  (steps {step_nums[0]} → {step_nums[-1]})")
print(f"NaN detected at step {meta['detection_step']}  "
      f"(last captured step is {step_nums[-1]})\n")

# ── Locate and load the .mjb model ───────────────────────────────────────────

model_name = pathlib.Path(meta["model_file"]).name
search_dirs = [
    pathlib.Path.cwd(),
    DUMP_PATH.parent,
    pathlib.Path("/tmp/mjlab/nan_dumps"),
]
model_path = None
for d in search_dirs:
    candidate = d / model_name
    if candidate.exists():
        model_path = candidate
        break

if model_path is None:
    sys.exit(f"ERROR: could not find {model_name!r} in {[str(d) for d in search_dirs]}")

print(f"Loading model: {model_path}")
mj_model = mujoco.MjModel.from_binary_path(str(model_path))
mj_data  = mujoco.MjData(mj_model)
print(f"Model: nq={mj_model.nq}  nv={mj_model.nv}  na={mj_model.na}\n")

assert mj_model.nq + mj_model.nv + mj_model.na == states.shape[1], (
    f"State dim mismatch: model nq+nv+na="
    f"{mj_model.nq}+{mj_model.nv}+{mj_model.na}={mj_model.nq+mj_model.nv+mj_model.na}"
    f", dump has {states.shape[1]}"
)

# ── Re-centre to single-env origin ───────────────────────────────────────────
# The training grid places each env at a fixed world-frame offset.  All robot
# joint angles are in joint-space (no offset needed), but the cube's freejoint
# qpos stores absolute world position = env_origin + local_pos.
# We recover env_origin from the env index and grid layout, then subtract it
# so the cube appears at the correct local position relative to the robot.
#
# Grid formula (row-major, y-negated convention used by mjlab):
#   n_cols    = ceil(sqrt(num_envs_total))
#   row, col  = env_id // n_cols, env_id % n_cols
#   env_origin_x = -(row - (n_cols-1)/2) * env_spacing
#   env_origin_y =  (col - (n_cols-1)/2) * env_spacing

ENV_SPACING = 1.0   # metres — from SceneCfg(env_spacing=1.0)

n_envs   = meta["num_envs_total"]
env_id   = meta["nan_env_ids"][0]
n_cols   = int(np.ceil(np.sqrt(n_envs)))
row, col = env_id // n_cols, env_id % n_cols
center   = (n_cols - 1) / 2.0
env_origin = np.array([-(row - center) * ENV_SPACING,
                         (col - center) * ENV_SPACING])

print(f"Env grid: {n_cols}×{n_cols},  env_id={env_id}  (row={row}, col={col})")
print(f"Env origin: x={env_origin[0]:.2f}  y={env_origin[1]:.2f}")

def find_freejoint_qposadr(model: mujoco.MjModel) -> int | None:
    """Return the qpos start index of the first free joint, or None."""
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_qposadr[j])
    return None

free_adr = find_freejoint_qposadr(mj_model)
if free_adr is not None:
    print(f"Free joint qpos at index {free_adr}")
    states_local = states.copy()
    states_local[:, free_adr]     -= env_origin[0]
    states_local[:, free_adr + 1] -= env_origin[1]
    cube_z = states_local[:, free_adr + 2].mean()
    print(f"Cube Z: {cube_z:.4f} m  "
          f"({'on floor — cube fell!' if cube_z < 0.02 else 'at table height'})\n")
else:
    print("No free joint found — skipping env-origin recentre\n")
    states_local = states

# ── Interactive replay ────────────────────────────────────────────────────────

paused     = False
do_restart = False
step_delay = 1.0 / DEFAULT_FPS

def key_cb(keycode):
    global paused, do_restart, step_delay
    ch = chr(keycode) if 0 < keycode < 128 else ""
    if ch == " ":
        paused = not paused
        print("  [paused]" if paused else "  [resumed]")
    elif ch in ("r", "R"):
        do_restart = True
        print("  [restart]")
    elif ch == "[":
        step_delay = min(step_delay * 1.5, 2.0)
        print(f"  [slower → {1/step_delay:.1f} fps]")
    elif ch == "]":
        step_delay = max(step_delay / 1.5, 1 / 120)
        print(f"  [faster → {1/step_delay:.1f} fps]")

def geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    return name if name else f"geom{geom_id}"

def print_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    if data.ncon == 0:
        return
    lines = []
    for k in range(data.ncon):
        c = data.contact[k]
        # contact force in contact frame; element 0 is normal force
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, k, force)
        normal_force = abs(force[0])
        g1 = geom_name(model, c.geom1)
        g2 = geom_name(model, c.geom2)
        lines.append(f"    {g1} ↔ {g2}  dist={c.dist:+.4f}  F={normal_force:.2f}N")
    print(f"  contacts ({data.ncon}):")
    print("\n".join(lines))

print("Controls:  Space=pause  r=restart  [=slower  ]=faster  Esc=quit")
print("Opening viewer…\n")

with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_cb) as viewer:
    i = 0
    prev_ncon = 0
    while viewer.is_running():
        if do_restart:
            i, do_restart = 0, False

        if not paused:
            mujoco.mj_setState(mj_model, mj_data, states_local[i],
                               mujoco.mjtState.mjSTATE_PHYSICS)
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            suffix = "  ← last before NaN!" if i == T - 1 else ""
            # Print contact info whenever it changes or on the last step
            if mj_data.ncon != prev_ncon or i == T - 1:
                print(f"\nstep {step_nums[i]:>6d}  [{i+1:>3d}/{T}]{suffix}")
                print_contacts(mj_model, mj_data)
                prev_ncon = mj_data.ncon
            else:
                print(f"\r  step {step_nums[i]:>6d}  [{i+1:>3d}/{T}]  {mj_data.ncon} contacts{suffix}   ",
                      end="", flush=True)

            i = (i + 1) % T
            time.sleep(step_delay)
        else:
            viewer.sync()
            time.sleep(0.016)

print("\nDone.")
