"""
Task 5 - Workspace analysis (UR5, MuJoCo)

Task workspace      : TCP positions reachable by the arm (position-only IK).
Tool-down workspace : subset where the tool can point straight down onto the
                      table for at least one yaw (pick / place feasibility).
Dexterous workspace : subset where the tool points down for EVERY sampled yaw.

Fixes vs. the previous version
  * the ur5.xml has no <site>, so the TCP falls back to the "EE" body
  * the yaw target is now "tool pointing down, rotated by yaw about world Z"
    (before it was Rz(yaw) of the raw EE frame, which is not a tool-down pose)
  * multi-start IK (random restarts + warm start across yaw samples)
  * links are kept above the table (no solutions through the floor)
  * the table rectangle is highlighted and table coverage is reported

Run from the scripts/ folder:   python workspace_analysis.py
"""
import os
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.patches import Rectangle
from scipy.spatial.transform import Rotation

# ============================================================
# 1. CONFIGURATION
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(PROJECT_DIR, "robot_descriptions", "ur5.xml")
ASSET_DIR = os.path.join(PROJECT_DIR, "robot_descriptions")

OUTPUT_PATH = os.path.join(SCRIPT_DIR, "workspace_analysis_results.png")
DATA_PATH = os.path.join(SCRIPT_DIR, "workspace_analysis_data.npz")

# Scan grid (base frame). Increase resolution for the final report.
X_RANGE = np.linspace(-0.8, 0.8, 20)
Y_RANGE = np.linspace(-0.8, 0.8, 20)
Z_RANGE = np.arange(0.0, 0.6001, 0.05)

# Yaw samples about the vertical axis.
YAW_ANGLES = np.linspace(0, 2 * np.pi, 12, endpoint=False)

# Table: rectangle in the base frame and its surface height.
# If the loaded XML contains a box geom whose name contains "table", its position/size
# OVERRIDE these values (so they match the scene). Otherwise these are PLACEHOLDERS:
# set them to the real assignment setup.
TABLE_FROM_MODEL = True
TABLE_X = (-0.5, 0.5)
TABLE_Y = (-0.5, 0.5)
TABLE_HEIGHT = 0.0
TABLE_SLICE_OFFSETS = (0.05, 0.20)   # TCP heights above the table to plot (pick / hover)

# IK settings
MAX_IK_ITERATIONS = 100
POSITION_TOLERANCE = 1e-3
ORIENTATION_TOLERANCE = 1e-2
DAMPING = 0.05
MAX_JOINT_STEP = 0.3
N_POSITION_STARTS = 30       # random restarts for position-only IK (only failures pay the full cost)
N_YAW_STARTS = 3             # random restarts (besides warm start) per yaw
JOINT_FALLBACK_LIMIT = 2 * np.pi   # used when the XML joint is unlimited
LINK_CLEARANCE = 0.04        # approx. link radius [m]: elbow/wrist segments must clear the table by this
N_VALIDATE = 300             # random FK poses used to measure IK recall on known-reachable positions
RETEST_N = 150               # failed table points re-tried with many more IK starts
N_RETEST_STARTS = 40
MAX_REACH = 1.0              # skip points farther than this from the shoulder
SHOULDER_POS = np.array([0.0, 0.0, 0.09])
RNG_SEED = 0

TCP_CANDIDATES = ["attachment_site", "pinch", "tcp", "tool", "ee_site", "gripper_site", "EE"]

# Tool axis in the TCP frame (the axis that must point DOWN when picking).
# None = auto: EE body of this ur5.xml -> -x (flange sits at -0.04 along link6 x),
# a site (e.g. gripper attachment_site) -> +z. Override with a 3-vector if needed.
TOOL_AXIS_OVERRIDE = None
TOOL_DOWN = np.array([0.0, 0.0, -1.0])

rng = np.random.default_rng(RNG_SEED)

# ============================================================
# 2. LOAD MUJOCO MODEL AND ASSETS
# ============================================================

def build_kinematic_model(path):
    """
    Workspace analysis only needs kinematics, so meshes, textures and materials
    are stripped before compiling. This avoids missing-STL errors and the
    "Repeated file name in assets dict" error (e.g. the two 2f85.png files).
    """
    root = ET.parse(path).getroot()

    for comp in root.iter("compiler"):
        for attr in ("meshdir", "texturedir", "assetdir"):
            comp.attrib.pop(attr, None)

    for asset in root.findall("asset"):
        root.remove(asset)

    # Remove mesh geoms and any reference to materials/textures.
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag == "geom" and ("mesh" in child.attrib or child.get("type") == "mesh"):
                parent.remove(child)
    for el in root.iter():
        for attr in ("material", "texture"):
            el.attrib.pop(attr, None)

    # A moving body needs mass; add a tiny dummy inertial if its geoms were stripped.
    for body in root.iter("body"):
        has_joint = any(c.tag in ("joint", "freejoint") for c in body)
        has_mass = any(c.tag in ("inertial", "geom") for c in body)
        if has_joint and not has_mass:
            ET.SubElement(body, "inertial", pos="0 0 0", mass="0.1",
                          diaginertia="0.001 0.001 0.001")

    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


model = build_kinematic_model(MODEL_PATH)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
print("MuJoCo model loaded successfully (kinematics only, meshes stripped).")

TABLE_SOURCE = "placeholder defaults (NOT verified against the assignment scene)"
if TABLE_FROM_MODEL:
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]) or ""
        if "table" in (gname + bname).lower() and int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_BOX):
            c, h = data.geom_xpos[gid], model.geom_size[gid]
            TABLE_X = (float(c[0] - h[0]), float(c[0] + h[0]))
            TABLE_Y = (float(c[1] - h[1]), float(c[1] + h[1]))
            TABLE_HEIGHT = float(c[2] + h[2])
            TABLE_SOURCE = f"model geom '{gname or bname}'"
            break
print(f"Table: x={TABLE_X}, y={TABLE_Y}, top z={TABLE_HEIGHT:.3f}  <- {TABLE_SOURCE}")
if TABLE_SOURCE.startswith("placeholder"):
    print("WARNING: no table geom found in the XML; check the table size/position against the assignment setup.")

# ============================================================
# 3. IDENTIFY TCP (site if available, otherwise body "EE")
# ============================================================

tcp_is_site = False
tcp_id = -1
tcp_name = None

for name in TCP_CANDIDATES:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid >= 0:
        tcp_is_site, tcp_id, tcp_name = True, sid, name
        break
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid >= 0:
        tcp_is_site, tcp_id, tcp_name = False, bid, name
        break

if tcp_id < 0:
    raise RuntimeError(f"TCP not found. Update TCP_CANDIDATES: {TCP_CANDIDATES}")

print(f"TCP: {tcp_name} ({'site' if tcp_is_site else 'body'})")



def tcp_position():
    return (data.site_xpos[tcp_id] if tcp_is_site else data.xpos[tcp_id]).copy()


def tcp_rotation():
    m = data.site_xmat[tcp_id] if tcp_is_site else data.xmat[tcp_id]
    return m.reshape(3, 3).copy()


# ============================================================
# 4. IDENTIFY UR5 ARM JOINTS
# ============================================================

excluded_tokens = ("finger", "gripper", "robotiq", "mimic", "coupling")
hinge_type = int(mujoco.mjtJoint.mjJNT_HINGE)

arm_joints = []
for joint_id in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    if name is None or int(model.jnt_type[joint_id]) != hinge_type:
        continue
    if any(tok in name.lower() for tok in excluded_tokens):
        continue
    arm_joints.append(joint_id)

if len(arm_joints) < 6:
    raise RuntimeError("Could not identify six UR5 arm joints. Inspect the joint names in the XML.")

arm_joints = arm_joints[:6]
qpos_addresses = np.array([model.jnt_qposadr[j] for j in arm_joints], dtype=int)
dof_addresses = np.array([model.jnt_dofadr[j] for j in arm_joints], dtype=int)
joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in arm_joints]

print("\nSelected arm joints:")
for jid, name in zip(arm_joints, joint_names):
    print(name, "limited =", bool(model.jnt_limited[jid]), "range =", model.jnt_range[jid])

# ---- Tool axis, derived from the kinematics (not assumed) -------------------
# The tool axis is the direction from the wrist-3 body origin to the TCP, expressed in the
# TCP frame. For a correct TCP it must be parallel to the wrist-3 joint axis.
mujoco.mj_forward(model, data)
wrist_body = int(model.jnt_bodyid[arm_joints[-1]])
offset_world = tcp_position() - data.xpos[wrist_body]
R_tcp0 = tcp_rotation()
wrist_axis_local = R_tcp0.T @ data.xaxis[arm_joints[-1]]

if TOOL_AXIS_OVERRIDE is not None:
    TOOL_AXIS_LOCAL = np.asarray(TOOL_AXIS_OVERRIDE, dtype=float)
    src_txt = "manual override"
elif np.linalg.norm(offset_world) > 1e-6:
    TOOL_AXIS_LOCAL = R_tcp0.T @ offset_world
    src_txt = "wrist-3 body -> TCP direction"
else:
    TOOL_AXIS_LOCAL = wrist_axis_local
    src_txt = "wrist-3 joint axis (TCP coincides with wrist body)"
TOOL_AXIS_LOCAL = TOOL_AXIS_LOCAL / np.linalg.norm(TOOL_AXIS_LOCAL)

parallel = abs(float(TOOL_AXIS_LOCAL @ wrist_axis_local))
print(f"Tool axis (TCP frame): {np.round(TOOL_AXIS_LOCAL, 4)}  [{src_txt}]")
print(f"|cos| between tool axis and wrist-3 joint axis: {parallel:.4f}"
      + ("" if parallel > 0.99 else "   <-- WARNING: TCP is not on the wrist-3 axis; check TCP / TOOL_AXIS_OVERRIDE"))

MIN_LINK_Z = TABLE_HEIGHT   # TCP / wrist end must not go below the table surface

# Bodies whose frames define the arm chain (link2 .. wrist3) used for the table clearance check.
chain_bodies = [int(model.jnt_bodyid[j]) for j in arm_joints[1:]]

# ============================================================
# 5. JOINT LIMITS
# ============================================================

joint_lower = np.array([
    model.jnt_range[j, 0] if model.jnt_limited[j] else -JOINT_FALLBACK_LIMIT for j in arm_joints
])
joint_upper = np.array([
    model.jnt_range[j, 1] if model.jnt_limited[j] else JOINT_FALLBACK_LIMIT for j in arm_joints
])

initial_qpos = data.qpos.copy()


def above_floor():
    """
    Approximate table-clearance check (kinematic skeleton only, NOT mesh collision):
    sample points along each link segment (link2 -> ... -> wrist3 -> TCP). All segments except the
    last must clear the table by LINK_CLEARANCE; the last one (wrist -> TCP) must stay above the table.
    Self-collision and collisions with other objects are not checked.
    """
    pts = np.array([data.xpos[b] for b in chain_bodies] + [tcp_position()])
    for i in range(len(pts) - 1):
        seg = np.linspace(pts[i], pts[i + 1], 5)
        lim = TABLE_HEIGHT + (LINK_CLEARANCE if i < len(pts) - 2 else 0.0)
        if np.any(seg[:, 2] < lim - 1e-9):
            return False
    return True


def random_config():
    return rng.uniform(np.maximum(joint_lower, -np.pi), np.minimum(joint_upper, np.pi))


def tool_down_rotation(yaw):
    """World rotation with the tool axis pointing down, rotated by `yaw` about world Z."""
    a = TOOL_AXIS_LOCAL
    v = np.cross(a, TOOL_DOWN)
    s, c = np.linalg.norm(v), float(a @ TOOL_DOWN)
    if s < 1e-9:
        R0 = np.eye(3) if c > 0 else Rotation.from_euler("x", np.pi).as_matrix()
    else:
        R0 = Rotation.from_rotvec(v / s * np.arctan2(s, c)).as_matrix()
    return Rotation.from_euler("z", yaw).as_matrix() @ R0


# ============================================================
# 6. INVERSE KINEMATICS (damped least squares, multi-start)
# ============================================================

jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))


def ik_single(target_position, target_rotation, q_start):
    data.qpos[:] = initial_qpos
    data.qpos[qpos_addresses] = np.clip(q_start, joint_lower, joint_upper)

    for _ in range(MAX_IK_ITERATIONS):
        mujoco.mj_fwdPosition(model, data)

        position_error = target_position - tcp_position()
        if target_rotation is None:
            error = position_error
            orientation_ok = True
        else:
            orientation_error = Rotation.from_matrix(target_rotation @ tcp_rotation().T).as_rotvec()
            error = np.concatenate([position_error, orientation_error])
            orientation_ok = np.linalg.norm(orientation_error) < ORIENTATION_TOLERANCE

        if np.linalg.norm(position_error) < POSITION_TOLERANCE and orientation_ok:
            if above_floor():
                return True, data.qpos[qpos_addresses].copy()
            return False, None

        if tcp_is_site:
            mujoco.mj_jacSite(model, data, jacp, jacr, tcp_id)
        else:
            mujoco.mj_jacBody(model, data, jacp, jacr, tcp_id)

        if target_rotation is None:
            J = jacp[:, dof_addresses]
        else:
            J = np.vstack([jacp[:, dof_addresses], jacr[:, dof_addresses]])

        A = J @ J.T + DAMPING**2 * np.eye(J.shape[0])
        try:
            dq = J.T @ np.linalg.solve(A, error)
        except np.linalg.LinAlgError:
            return False, None

        dq = np.clip(dq, -MAX_JOINT_STEP, MAX_JOINT_STEP)
        data.qpos[qpos_addresses] = np.clip(data.qpos[qpos_addresses] + dq, joint_lower, joint_upper)

    return False, None


def solve_ik(target_position, target_rotation=None, seeds=()):
    """Try each seed in turn; return the first converged solution."""
    for q0 in seeds:
        ok, q = ik_single(target_position, target_rotation, q0)
        if ok:
            return True, q
    return False, None


# ============================================================
# 7. SCAN: 0 = unreachable, 1 = position only, 2 = tool-down (some yaw), 3 = dexterous (all yaws)
# ============================================================

nx, ny, nz = len(X_RANGE), len(Y_RANGE), len(Z_RANGE)
status = np.zeros((nx, ny, nz), dtype=np.int8)
yaw_ok_count = np.zeros((nx, ny, nz), dtype=np.int16)
skipped_far = np.zeros((nx, ny, nz), dtype=bool)

print("\nStarting workspace scan...")
print(f"Candidate positions: {nx * ny * nz}  |  yaw samples: {len(YAW_ANGLES)}")

home_q = initial_qpos[qpos_addresses].copy()

for ix, x in enumerate(X_RANGE):
    for iy, y in enumerate(Y_RANGE):
        for iz, z in enumerate(Z_RANGE):
            target = np.array([x, y, z])

            if np.linalg.norm(target - SHOULDER_POS) > MAX_REACH:
                skipped_far[ix, iy, iz] = True      # counted as unreachable
                continue

            seeds = [home_q] + [random_config() for _ in range(N_POSITION_STARTS)]
            reachable, q_pos = solve_ik(target, None, seeds)
            if not reachable:
                continue
            status[ix, iy, iz] = 1

            n_ok, n_tried, warm = 0, 0, q_pos
            for yaw in YAW_ANGLES:
                seeds = [warm] + [random_config() for _ in range(N_YAW_STARTS)]
                ok, q = solve_ik(target, tool_down_rotation(yaw), seeds)
                n_tried += 1
                if ok:
                    n_ok += 1
                    warm = q
                elif n_ok == 0 and n_tried >= 3:
                    break       # tool-down looks infeasible here; stop early
            yaw_ok_count[ix, iy, iz] = n_ok
            if n_ok == len(YAW_ANGLES):
                status[ix, iy, iz] = 3
            elif n_ok > 0:
                status[ix, iy, iz] = 2
    print(f"  x row {ix + 1}/{nx}: reachable={int((status >= 1).sum())} "
          f"tool-down={int((status >= 2).sum())} dexterous={int((status == 3).sum())}", end="\r")
print()


def points_where(mask):
    idx = np.argwhere(mask)
    return np.c_[X_RANGE[idx[:, 0]], Y_RANGE[idx[:, 1]], Z_RANGE[idx[:, 2]]] if len(idx) else np.zeros((0, 3))


task_workspace = points_where(status >= 1)
tool_down_workspace = points_where(status >= 2)
dexterous_workspace = points_where(status == 3)

print("Workspace scan finished.")
print("Task workspace points (position reachable):", len(task_workspace))
print("Tool-down workspace points (>=1 yaw):      ", len(tool_down_workspace))
print("Dexterous workspace points (all yaws):     ", len(dexterous_workspace))
if len(tool_down_workspace):
    print(f"Fraction of tool-down points that are dexterous: "
          f"{100 * len(dexterous_workspace) / len(tool_down_workspace):.1f}%")

# ============================================================
# 8. TABLE COVERAGE (denominator = ALL sampled grid points inside the table,
#    unreachable / skipped points count as failures)
# ============================================================

in_x = (X_RANGE >= TABLE_X[0] - 1e-9) & (X_RANGE <= TABLE_X[1] + 1e-9)
in_y = (Y_RANGE >= TABLE_Y[0] - 1e-9) & (Y_RANGE <= TABLE_Y[1] + 1e-9)
table_mask = in_x[:, None] & in_y[None, :]
n_table = int(table_mask.sum())
print(f"\nTable region x={TABLE_X}, y={TABLE_Y}: {n_table} sampled grid points per height "
      f"(grid spacing {X_RANGE[1] - X_RANGE[0]:.3f} x {Y_RANGE[1] - Y_RANGE[0]:.3f} m)")
if n_table < 25:
    print("WARNING: very few grid points inside the table; refine X_RANGE / Y_RANGE.")

slice_indices = [int(np.argmin(np.abs(Z_RANGE - (TABLE_HEIGHT + off)))) for off in TABLE_SLICE_OFFSETS]
coverage = {}
for iz in slice_indices:
    s = status[:, :, iz]
    c = {
        "reachable": int(((s >= 1) & table_mask).sum()),
        "tool_down": int(((s >= 2) & table_mask).sum()),
        "dexterous": int(((s == 3) & table_mask).sum()),
    }
    coverage[iz] = c
    print(f"  z={Z_RANGE[iz]:.2f} m: reachable {c['reachable']}/{n_table} ({100 * c['reachable'] / n_table:.1f}%), "
          f"tool-down {c['tool_down']}/{n_table} ({100 * c['tool_down'] / n_table:.1f}%), "
          f"dexterous {c['dexterous']}/{n_table} ({100 * c['dexterous'] / n_table:.1f}%)")

# ============================================================
# 9. VALIDATION OF THE IK-BASED CLASSIFICATION
# ============================================================

print("\nValidation")

# (a) IK recall on positions that are reachable BY CONSTRUCTION (forward kinematics of random configs).
n_ok = n_tot = 0
for _ in range(N_VALIDATE):
    q = random_config()
    data.qpos[:] = initial_qpos
    data.qpos[qpos_addresses] = q
    mujoco.mj_fwdPosition(model, data)
    if not above_floor():
        continue
    p = tcp_position()
    n_tot += 1
    ok, _ = solve_ik(p, None, [home_q] + [random_config() for _ in range(N_POSITION_STARTS)])
    n_ok += int(ok)
if n_tot:
    print(f"  (a) IK recall on known-reachable positions: {n_ok}/{n_tot} ({100 * n_ok / n_tot:.1f}%)  "
          f"-> misses are IK convergence failures, not true unreachability")

# (b) Re-test failed table points with many more random starts.
rng_idx = np.random.default_rng(RNG_SEED + 1)
for label, want, rot_fn in [("unreachable -> position", 0, False), ("position-only -> tool-down", 1, True)]:
    cand = []
    for iz in slice_indices:
        for ix, iy in np.argwhere(table_mask & (status[:, :, iz] == want) & ~skipped_far[:, :, iz]):
            cand.append((ix, iy, iz))
    if not cand:
        print(f"  (b) {label}: no failed table points to re-test")
        continue
    pick = rng_idx.permutation(len(cand))[:RETEST_N]
    flips = 0
    for k in pick:
        ix, iy, iz = cand[k]
        target = np.array([X_RANGE[ix], Y_RANGE[iy], Z_RANGE[iz]])
        Rd = tool_down_rotation(0.0) if rot_fn else None
        ok, _ = solve_ik(target, Rd, [home_q] + [random_config() for _ in range(N_RETEST_STARTS)])
        flips += int(ok)
    print(f"  (b) {label}: {flips}/{len(pick)} previously failed points succeeded with {N_RETEST_STARTS} more starts"
          + ("  <- increase N_*_STARTS" if flips else ""))

# (c) Tool-down check: the tool axis really points down and the tool tip is below the wrist.
cand = np.argwhere(status >= 2)
checked = good = 0
for k in rng_idx.permutation(len(cand))[:30]:
    ix, iy, iz = cand[k]
    target = np.array([X_RANGE[ix], Y_RANGE[iy], Z_RANGE[iz]])
    ok, q = solve_ik(target, tool_down_rotation(0.0), [home_q] + [random_config() for _ in range(N_RETEST_STARTS)])
    if not ok:
        continue
    data.qpos[qpos_addresses] = q
    mujoco.mj_fwdPosition(model, data)
    axis_world = tcp_rotation() @ TOOL_AXIS_LOCAL
    checked += 1
    good += int(axis_world @ TOOL_DOWN > 0.999 and tcp_position()[2] < data.xpos[wrist_body][2])
if checked:
    print(f"  (c) tool-down poses with tool axis down and TCP below wrist: {good}/{checked}")

# ============================================================
# 10. SAVE NUMERICAL DATA
# ============================================================

np.savez_compressed(
    DATA_PATH,
    status=status, yaw_ok_count=yaw_ok_count,
    task_workspace=task_workspace, tool_down_workspace=tool_down_workspace,
    dexterous_workspace=dexterous_workspace,
    yaw_angles=YAW_ANGLES, x_range=X_RANGE, y_range=Y_RANGE, z_range=Z_RANGE,
    table_x=TABLE_X, table_y=TABLE_Y, table_height=TABLE_HEIGHT,
)
print("\nNumerical results saved:", DATA_PATH)

# ============================================================
# 11. VISUALIZE (disjoint categories so nothing is hidden by overplotting)
# ============================================================

COLORS = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
LABELS = {1: "Reachable, tool-down infeasible", 2: "Tool-down, only some yaws", 3: "Dexterous (all yaws)"}

n_slices = len(slice_indices)
fig = plt.figure(figsize=(6.5 * (n_slices + 1), 6.8))

ax1 = fig.add_subplot(1, n_slices + 1, 1, projection="3d")
for code, size, alpha in [(1, 4, 0.2), (2, 8, 0.8), (3, 6, 0.6)]:
    pts = points_where(status == code)
    if len(pts):
        ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=size, alpha=alpha, c=COLORS[code],
                    label=f"{LABELS[code]} ({len(pts)})")
ax1.scatter([0], [0], [0], c="k", marker="^", s=50, label="Robot base")
ax1.set_title("UR5 workspace")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.set_zlabel("Z (m)")
ax1.legend(fontsize=7)

dx = X_RANGE[1] - X_RANGE[0]
dy = Y_RANGE[1] - Y_RANGE[0]
GX, GY = np.meshgrid(X_RANGE, Y_RANGE, indexing="ij")
for k, iz in enumerate(slice_indices):
    ax = fig.add_subplot(1, n_slices + 1, k + 2)
    s = status[:, :, iz]
    unreachable = s == 0
    ax.scatter(GX[unreachable], GY[unreachable], s=18, marker="x", c="lightgray", label="Unreachable")
    for code in (1, 2, 3):
        m = s == code
        if m.any():
            ax.scatter(GX[m], GY[m], s=45, marker="s", c=COLORS[code], alpha=0.85, label=LABELS[code])
    ax.add_patch(Rectangle((TABLE_X[0], TABLE_Y[0]), TABLE_X[1] - TABLE_X[0], TABLE_Y[1] - TABLE_Y[0],
                           fill=False, ec="red", lw=2, label="Table"))
    ax.plot(0, 0, "k^", ms=10, label="Robot base")
    c = coverage[iz]
    ax.set_title(f"TCP z = {Z_RANGE[iz]:.2f} m (table top + {Z_RANGE[iz] - TABLE_HEIGHT:.2f} m)\n"
                 f"table points: reachable {c['reachable']}/{n_table}, tool-down {c['tool_down']}/{n_table}, "
                 f"dexterous {c['dexterous']}/{n_table}", fontsize=9)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(X_RANGE[0] - dx, X_RANGE[-1] + dx)
    ax.set_ylim(Y_RANGE[0] - dy, Y_RANGE[-1] + dy)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend(fontsize=7, loc="upper right")

fig.text(0.5, 0.005,
         f"Table: {TABLE_SOURCE}. Clearance check = link skeleton only (no mesh/self collision). "
         f"IK is numerical: 'unreachable' means no solution found from the tested starts.",
         ha="center", fontsize=7)
plt.tight_layout(rect=(0, 0.02, 1, 1))
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
print("Plot saved:", OUTPUT_PATH)
plt.show()