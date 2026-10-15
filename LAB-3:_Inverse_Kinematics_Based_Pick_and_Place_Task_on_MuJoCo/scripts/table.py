"""
Task 6 - Table design for the UR5 pick-and-place cell (MuJoCo)

For every candidate table height h (top surface, measured from the robot base plane z = 0) the
script checks, at each (x, y) grid cell in front of the robot, a complete vertical pick/place
approach with the tool pointing down:

    grasp pose (TCP at h + GRASP_OFFSET)  ->  vertical lift  ->  pre-grasp (TCP at h + APPROACH_OFFSET)

A cell is "good" only if, for EVERY sampled yaw, all of the following hold:
    1. IK has a solution at the grasp pose and at every pose on the vertical path
       (the path is followed by continuation, so a good cell is one smooth approach, no branch flip)
    2. no large joint jump between consecutive path poses (MAX_JOINT_JUMP)
    3. the arm is not near a singularity along the path (sigma_min(J) >= SIGMA_MIN)
    4. the elbow/wrist links keep CLEARANCE_MIN above the table along the path

The table is then the largest rectangle, centred on y = 0 in front of the robot, that contains only
good cells. Heights are compared by area and by the worst-case margins inside that rectangle.

Run from the scripts/ folder:   python table_design.py   (add --fast for a quick check)
Outputs: table_design_results.png, table_design_data.npz
"""
import argparse
import os
import time
import xml.etree.ElementTree as ET
from multiprocessing import Pool

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.patches import Rectangle
from scipy.spatial.transform import Rotation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_DIR, "robot_descriptions", "ur5.xml")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "table_design_results.png")
DATA_PATH = os.path.join(SCRIPT_DIR, "table_design_data.npz")

# ---------------------------------------------------------------- design inputs
HEIGHTS = [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30]   # table top relative to base plane [m]
# The TCP used here is the tool FLANGE (EE body). The gripper hangs below it, so the flange must stay
# TOOL_LENGTH above the fingertips. TOOL_LENGTH is an ASSUMPTION (Robotiq 2F-85 ~0.15 m): set your own.
TOOL_LENGTH = 0.15         # flange -> fingertip pinch point [m]
GRASP_HEIGHT = 0.025       # fingertip height above table when grasping / placing (mid-height of a 5 cm part)
PREGRASP_LIFT = 0.15       # vertical retreat / approach distance above the grasp pose [m]
GRASP_OFFSET = TOOL_LENGTH + GRASP_HEIGHT           # flange height above table at grasp [m]
APPROACH_OFFSET = GRASP_OFFSET + PREGRASP_LIFT      # flange height above table at pre-grasp [m]
N_PATH = 4                 # poses along the vertical approach (grasp ... pre-grasp)
N_YAW = 4                  # yaw samples (0, 90, 180, 270 deg) - the UR5 wrist is continuous
X_GRID = np.arange(0.15, 0.85 + 1e-9, 0.05)          # in front of the robot
Y_GRID = np.arange(-0.70, 0.70 + 1e-9, 0.05)
BASE_KEEPOUT = 0.15        # radius around the base axis kept free of table [m]
HEIGHT_RULE = 0.80         # choose the HIGHEST table whose feasible area is >= this fraction of the best area
SAFETY_MARGIN = 0.05       # final table is the feasible rectangle shrunk by this on every free side [m]
VERIFY_STEP = 0.025        # grid step of the final verification [m]
VERIFY_YAW = 8             # yaw samples in the final verification

# comfort / safety thresholds
MAX_JOINT_JUMP = 0.8       # rad, largest single-joint change between neighbouring path poses
SIGMA_MIN = 0.05           # smallest singular value of the 6x6 Jacobian along the path
LINK_CLEARANCE_IK = 0.04   # hard clearance enforced inside IK (approx. link radius) [m]
CLEARANCE_MIN = 0.08       # comfortable clearance between elbow/wrist links and table [m]

# IK settings
MAX_IK_ITERATIONS = 100
POSITION_TOLERANCE = 1e-3
ORIENTATION_TOLERANCE = 1e-2
DAMPING = 0.05
MAX_JOINT_STEP = 0.3
N_STARTS = 12
JOINT_FALLBACK_LIMIT = 2 * np.pi
SHOULDER_POS = np.array([0.0, 0.0, 0.09])
MAX_REACH = 1.0
TCP_CANDIDATES = ["attachment_site", "pinch", "tcp", "tool", "ee_site", "gripper_site", "EE"]
TOOL_DOWN = np.array([0.0, 0.0, -1.0])
SEED = 0


# ---------------------------------------------------------------- model (kinematics only)
def build_kinematic_model(path):
    root = ET.parse(path).getroot()
    for comp in root.iter("compiler"):
        for attr in ("meshdir", "texturedir", "assetdir"):
            comp.attrib.pop(attr, None)
    for asset in root.findall("asset"):
        root.remove(asset)
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag == "geom" and ("mesh" in child.attrib or child.get("type") == "mesh"):
                parent.remove(child)
    for el in root.iter():
        for attr in ("material", "texture"):
            el.attrib.pop(attr, None)
    for body in root.iter("body"):
        has_joint = any(c.tag in ("joint", "freejoint") for c in body)
        has_mass = any(c.tag in ("inertial", "geom") for c in body)
        if has_joint and not has_mass:
            ET.SubElement(body, "inertial", pos="0 0 0", mass="0.1", diaginertia="0.001 0.001 0.001")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


class Robot:
    def __init__(self, path):
        self.m = build_kinematic_model(path)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        self.rng = np.random.default_rng(SEED)

        self.tcp_is_site, self.tcp_id = False, -1
        for name in TCP_CANDIDATES:
            sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid >= 0:
                self.tcp_is_site, self.tcp_id = True, sid
                break
            bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self.tcp_is_site, self.tcp_id = False, bid
                break
        if self.tcp_id < 0:
            raise RuntimeError("TCP not found; update TCP_CANDIDATES")

        excluded = ("finger", "gripper", "robotiq", "mimic", "coupling")
        joints = []
        for j in range(self.m.njnt):
            nm = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if nm is None or int(self.m.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                continue
            if any(t in nm.lower() for t in excluded):
                continue
            joints.append(j)
        self.joints = joints[:6]
        if len(self.joints) < 6:
            raise RuntimeError("Could not identify six arm joints")
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in self.joints], dtype=int)
        self.dadr = np.array([self.m.jnt_dofadr[j] for j in self.joints], dtype=int)
        self.lo = np.array([self.m.jnt_range[j, 0] if self.m.jnt_limited[j] else -JOINT_FALLBACK_LIMIT
                            for j in self.joints])
        self.hi = np.array([self.m.jnt_range[j, 1] if self.m.jnt_limited[j] else JOINT_FALLBACK_LIMIT
                            for j in self.joints])
        self.q_init = self.d.qpos.copy()
        self.home = self.q_init[self.qadr].copy()
        self.jacp = np.zeros((3, self.m.nv))
        self.jacr = np.zeros((3, self.m.nv))
        self.chain = [int(self.m.jnt_bodyid[j]) for j in self.joints[1:]]

        # tool axis from the kinematics: wrist-3 body -> TCP, expressed in the TCP frame
        mujoco.mj_forward(self.m, self.d)
        wrist = int(self.m.jnt_bodyid[self.joints[-1]])
        off = self.tcp_pos() - self.d.xpos[wrist]
        R0 = self.tcp_rot()
        axis = R0.T @ off if np.linalg.norm(off) > 1e-6 else R0.T @ self.d.xaxis[self.joints[-1]]
        self.tool_axis = axis / np.linalg.norm(axis)

    # --- kinematics helpers
    def tcp_pos(self):
        return (self.d.site_xpos[self.tcp_id] if self.tcp_is_site else self.d.xpos[self.tcp_id]).copy()

    def tcp_rot(self):
        return (self.d.site_xmat[self.tcp_id] if self.tcp_is_site else self.d.xmat[self.tcp_id]).reshape(3, 3).copy()

    def random_config(self):
        return self.rng.uniform(np.maximum(self.lo, -np.pi), np.minimum(self.hi, np.pi))

    def tool_down_rotation(self, yaw):
        a, v = self.tool_axis, np.cross(self.tool_axis, TOOL_DOWN)
        s, c = np.linalg.norm(v), float(self.tool_axis @ TOOL_DOWN)
        R0 = (np.eye(3) if c > 0 else Rotation.from_euler("x", np.pi).as_matrix()) if s < 1e-9 else \
            Rotation.from_rotvec(v / s * np.arctan2(s, c)).as_matrix()
        return Rotation.from_euler("z", yaw).as_matrix() @ R0

    def skeleton_min_clearance(self, h):
        """
        (min clearance of elbow..wrist links above the table top, TCP height above the table).
        Only link points that lie over the table area (in front of the base: x >= BASE_KEEPOUT) count;
        the shoulder column sits beside the table, not under it.
        """
        pts = np.array([self.d.xpos[b] for b in self.chain] + [self.tcp_pos()])
        zmin = np.inf
        for i in range(len(pts) - 2):                       # all segments except wrist->TCP
            seg = np.linspace(pts[i], pts[i + 1], 7)
            over_table = seg[seg[:, 0] >= BASE_KEEPOUT]
            if len(over_table):
                zmin = min(zmin, over_table[:, 2].min())
        return zmin - h, pts[-1, 2] - h

    def above_table(self, h):
        c_links, c_last = self.skeleton_min_clearance(h)
        return c_links >= LINK_CLEARANCE_IK - 1e-9 and c_last >= -1e-9   # inf (no link over table) passes

    # --- IK
    def ik_single(self, pos, Rd, q0, h):
        self.d.qpos[:] = self.q_init
        self.d.qpos[self.qadr] = np.clip(q0, self.lo, self.hi)
        for _ in range(MAX_IK_ITERATIONS):
            mujoco.mj_fwdPosition(self.m, self.d)
            e_p = pos - self.tcp_pos()
            e_r = Rotation.from_matrix(Rd @ self.tcp_rot().T).as_rotvec()
            if np.linalg.norm(e_p) < POSITION_TOLERANCE and np.linalg.norm(e_r) < ORIENTATION_TOLERANCE:
                return (self.above_table(h), self.d.qpos[self.qadr].copy())
            if self.tcp_is_site:
                mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.tcp_id)
            else:
                mujoco.mj_jacBody(self.m, self.d, self.jacp, self.jacr, self.tcp_id)
            J = np.vstack([self.jacp, self.jacr])[:, self.dadr]
            dq = J.T @ np.linalg.solve(J @ J.T + DAMPING**2 * np.eye(6), np.concatenate([e_p, 0.5 * e_r]))
            self.d.qpos[self.qadr] = np.clip(self.d.qpos[self.qadr] + np.clip(dq, -MAX_JOINT_STEP, MAX_JOINT_STEP),
                                             self.lo, self.hi)
        return False, None

    def solve(self, pos, Rd, seeds, h):
        for q0 in seeds:
            ok, q = self.ik_single(pos, Rd, q0, h)
            if ok:
                return q
        return None

    def sigma_min(self):
        if self.tcp_is_site:
            mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.tcp_id)
        else:
            mujoco.mj_jacBody(self.m, self.d, self.jacp, self.jacr, self.tcp_id)
        return float(np.linalg.svd(np.vstack([self.jacp, self.jacr])[:, self.dadr], compute_uv=False).min())


# ---------------------------------------------------------------- per-cell evaluation
_robot = None


def _init_worker(worker_seed):
    global _robot
    _robot = Robot(MODEL_PATH)
    _robot.rng = np.random.default_rng(worker_seed)


def follow_path(r, x, y, Rd, q_grasp, levels, h):
    """Follow the vertical approach by continuation from a grasp solution.
    Returns (feasible, min_sigma, min_clearance, max_jump)."""
    min_sigma, min_clear, max_jump = np.inf, np.inf, 0.0
    q_prev = q_grasp
    for z in levels:
        q_z = r.solve(np.array([x, y, z]), Rd, [q_prev], h)       # continuation only: no restarts
        if q_z is None:
            return False, 0.0, 0.0, 0.0
        max_jump = max(max_jump, float(np.abs(q_z - q_prev).max()))
        q_prev = q_z
        r.d.qpos[r.qadr] = q_z
        mujoco.mj_fwdPosition(r.m, r.d)
        min_sigma = min(min_sigma, r.sigma_min())
        min_clear = min(min_clear, r.skeleton_min_clearance(h)[0])
    return True, min_sigma, min_clear, max_jump


def is_comfortable(sg, cl, jp):
    return jp <= MAX_JOINT_JUMP and sg >= SIGMA_MIN and cl >= CLEARANCE_MIN


def _evaluate(x, y, h, n_yaw, n_starts):
    """Return (status, min_sigma, min_clearance, max_jump). status: 0 unreachable, 1 approach infeasible
    (IK / continuity), 2 approach possible but below comfort thresholds, 3 good.
    For every yaw several IK branches are tried and the best complete approach is kept, so a cell is only
    rejected if NO tested branch gives a comfortable approach."""
    r = _robot
    if np.linalg.norm(np.array([x, y, h + GRASP_OFFSET]) - SHOULDER_POS) > MAX_REACH:
        return 0, 0.0, 0.0, 0.0
    levels = h + np.linspace(GRASP_OFFSET, APPROACH_OFFSET, N_PATH)
    yaws = np.linspace(0, 2 * np.pi, n_yaw, endpoint=False)

    cell = [np.inf, np.inf, 0.0]            # worst sigma, worst clearance, worst jump over yaws
    all_comfortable = True
    warm = None
    for k, yaw in enumerate(yaws):
        Rd = r.tool_down_rotation(yaw)
        seeds = ([warm] if warm is not None else [r.home]) + [r.random_config() for _ in range(n_starts)]
        best, best_score, found_any_grasp = None, -np.inf, False
        for q0 in seeds:
            ok, q = r.ik_single(np.array([x, y, levels[0]]), Rd, q0, h)
            if not ok:
                continue
            found_any_grasp = True
            feas, sg, cl, jp = follow_path(r, x, y, Rd, q, levels, h)
            if not feas:
                continue
            if is_comfortable(sg, cl, jp):
                best, best_score = (q, sg, cl, jp, True), np.inf
                break
            score = sg - jp                  # prefer high sigma_min and small joint jumps
            if score > best_score:
                best, best_score = (q, sg, cl, jp, False), score
        if best is None:
            if not found_any_grasp:
                return (0 if k == 0 else 1), 0.0, 0.0, 0.0
            return 1, 0.0, 0.0, 0.0
        q, sg, cl, jp, comf = best
        warm = q
        all_comfortable &= comf
        cell = [min(cell[0], sg), min(cell[1], cl), max(cell[2], jp)]
    return (3 if all_comfortable else 2), cell[0], cell[1], cell[2]


def evaluate_cell(args):
    """Deterministic per-cell evaluation (seed derived from the cell), with escalation: a cell that is
    not 'good' after N_STARTS random IK branches is re-tried with 3x as many before it is rejected."""
    x, y, h = args[:3]
    n_yaw = args[3] if len(args) > 3 else N_YAW
    seed = (int(round(x * 1000)) * 73856093 ^ int(round(y * 1000)) * 19349663
            ^ int(round(h * 1000)) * 83492791 ^ n_yaw * 2654435761) % (2**32)
    _robot.rng = np.random.default_rng(seed)
    out = _evaluate(x, y, h, n_yaw, N_STARTS)
    if out[0] in (1, 2):
        retry = _evaluate(x, y, h, n_yaw, 3 * N_STARTS)
        if retry[0] > out[0]:
            out = retry
    return out


# ---------------------------------------------------------------- rectangle search
def largest_centered_rectangle(good):
    """good[ix, iy]; rectangle spans x indices i0..i1 and y symmetric about y = 0 (centre index ic)."""
    ic = int(np.argmin(np.abs(Y_GRID)))
    half = np.zeros(len(X_GRID), dtype=int)      # largest symmetric half-width (in cells) per x column
    for ix in range(len(X_GRID)):
        w = 0
        while ic - (w + 1) >= 0 and ic + (w + 1) < len(Y_GRID) and good[ix, ic - (w + 1):ic + (w + 2)].all():
            w += 1
        half[ix] = w if good[ix, ic] else -1
    best = (0.0, None)
    for i0 in range(len(X_GRID)):
        w = None
        for i1 in range(i0, len(X_GRID)):
            if half[i1] < 0:
                break
            w = half[i1] if w is None else min(w, half[i1])
            dx = (i1 - i0) * (X_GRID[1] - X_GRID[0])
            dy = 2 * w * (Y_GRID[1] - Y_GRID[0])
            if dx * dy > best[0] and i1 > i0 and w > 0:
                best = (dx * dy, (i0, i1, ic - w, ic + w))
    return best


def main():
    global X_GRID, Y_GRID, N_STARTS, N_YAW, HEIGHTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    a = ap.parse_args()
    if a.fast:
        X_GRID, Y_GRID = np.arange(0.15, 0.85 + 1e-9, 0.10), np.arange(-0.70, 0.70 + 1e-9, 0.10)
        N_STARTS, N_YAW, HEIGHTS = 6, 2, [-0.10, 0.0, 0.10]

    probe = Robot(MODEL_PATH)
    print(f"TCP is {'site' if probe.tcp_is_site else 'body'}; tool axis (TCP frame) = {np.round(probe.tool_axis, 4)}")
    print(f"Heights {HEIGHTS}; flange at +{GRASP_OFFSET:.3f} m (grasp) -> +{APPROACH_OFFSET:.3f} m (pre-grasp), "
          f"tool length {TOOL_LENGTH} m, "
          f"{len(X_GRID)}x{len(Y_GRID)} cells, {N_YAW} yaws, {a.workers} workers")

    status = np.zeros((len(HEIGHTS), len(X_GRID), len(Y_GRID)), dtype=np.int8)
    sigma = np.zeros(status.shape)
    clear = np.zeros(status.shape)
    jump = np.zeros(status.shape)

    t0 = time.time()
    with Pool(a.workers, initializer=_init_worker, initargs=(SEED,)) as pool:
        for ih, h in enumerate(HEIGHTS):
            jobs = [(x, y, h) for x in X_GRID for y in Y_GRID]
            res = pool.map(evaluate_cell, jobs, chunksize=8)
            for (ix, iy), (s, sg, cl, jp) in zip([(i, j) for i in range(len(X_GRID)) for j in range(len(Y_GRID))], res):
                status[ih, ix, iy], sigma[ih, ix, iy], clear[ih, ix, iy], jump[ih, ix, iy] = s, sg, cl, jp
            print(f"  h={h:+.2f} m done ({time.time() - t0:.0f} s): good cells {int((status[ih] == 3).sum())}")

    # keep the base footprint free of table
    XX, YY = np.meshgrid(X_GRID, Y_GRID, indexing="ij")
    keepout = np.hypot(XX, YY) < BASE_KEEPOUT

    results = []
    print("\nHeight comparison (largest y-centred rectangle of 'good' cells in front of the robot):")
    print(f"{'h [m]':>7} {'x range [m]':>16} {'y +- [m]':>9} {'size LxW [m]':>14} {'area [m2]':>10} "
          f"{'sigma_min':>10} {'clear [m]':>10} {'jump [rad]':>11}")
    for ih, h in enumerate(HEIGHTS):
        good = (status[ih] == 3) & ~keepout
        area, rect = largest_centered_rectangle(good)
        if rect is None:
            results.append(dict(h=h, area=0.0, rect=None))
            print(f"{h:7.2f}   no feasible rectangle")
            continue
        i0, i1, j0, j1 = rect
        sub = (slice(i0, i1 + 1), slice(j0, j1 + 1))
        res = dict(h=h, area=area, rect=rect, x0=X_GRID[i0], x1=X_GRID[i1], w=Y_GRID[j1],
                   sigma=float(sigma[ih][sub].min()), clear=float(clear[ih][sub].min()),
                   jump=float(jump[ih][sub].max()))
        results.append(res)
        print(f"{h:7.2f} {res['x0']:7.2f}..{res['x1']:<7.2f} {res['w']:9.2f} "
              f"{res['x1'] - res['x0']:6.2f}x{2 * res['w']:<6.2f} {area:10.3f} "
              f"{res['sigma']:10.3f} {res['clear']:10.3f} {res['jump']:11.3f}")

    areas = np.array([r["area"] for r in results])
    best = int(np.argmax(areas))
    print(f"\nLargest comfortable table: h = {HEIGHTS[best]:+.2f} m "
          f"({results[best]['x1'] - results[best]['x0']:.2f} x {2 * results[best]['w']:.2f} m)"
          if results[best]["rect"] else "\nNo feasible table found.")

    # ---------------- height selection and final verification
    best_area = float(areas.max())
    sel = max((i for i, r_ in enumerate(results) if r_["rect"] and r_["area"] >= HEIGHT_RULE * best_area),
              key=lambda i: HEIGHTS[i])
    rs = results[sel]
    x0 = rs["x0"] + SAFETY_MARGIN if rs["x0"] > X_GRID[0] + 1e-9 else rs["x0"]   # no margin against base side
    x1 = rs["x1"] - SAFETY_MARGIN
    w = rs["w"] - SAFETY_MARGIN
    h_sel = HEIGHTS[sel]
    print(f"\nSelected height: h = {h_sel:+.2f} m (highest table keeping >= {int(HEIGHT_RULE * 100)}% of the best area "
          f"{best_area:.3f} m2)")
    print(f"Table after {SAFETY_MARGIN:.2f} m safety margin: x = {x0:.2f}..{x1:.2f} m, y = +-{w:.2f} m "
          f"-> {x1 - x0:.2f} x {2 * w:.2f} m")

    vx = np.arange(x0, x1 + 1e-9, VERIFY_STEP)
    vy = np.arange(-w, w + 1e-9, VERIFY_STEP)
    with Pool(a.workers, initializer=_init_worker, initargs=(SEED + 1,)) as pool:
        vres = pool.map(evaluate_cell, [(x, y, h_sel, VERIFY_YAW) for x in vx for y in vy], chunksize=8)
    vstat = np.array([r_[0] for r_ in vres]).reshape(len(vx), len(vy))
    n_good = int((vstat == 3).sum())
    print(f"Verification on a {VERIFY_STEP * 100:.1f} cm grid with {VERIFY_YAW} yaws: {n_good}/{vstat.size} cells good "
          f"({100 * n_good / vstat.size:.1f}%), reachable-but-uncomfortable {int((vstat == 2).sum())}, "
          f"infeasible {int((vstat <= 1).sum())}")
    vs = np.array([r_[1] for r_ in vres])
    vc = np.array([r_[2] for r_ in vres])
    vj = np.array([r_[3] for r_ in vres])
    ok_mask = (np.array([r_[0] for r_ in vres]) == 3)
    if ok_mask.any():
        print(f"  over the good cells: worst sigma_min {vs[ok_mask].min():.3f}, worst link clearance over table {vc[ok_mask].min():.3f} m, "
              f"largest joint jump {vj[ok_mask].max():.2f} rad")
    print("\nUse in workspace_analysis.py:")
    print(f"  TABLE_X = ({x0:.2f}, {x1:.2f}); TABLE_Y = ({-w:.2f}, {w:.2f}); TABLE_HEIGHT = {h_sel:.2f}")

    # ---------------- figure
    n = len(HEIGHTS)
    fig, axes = plt.subplots(2, (n + 1) // 2 + (1 if n % 2 else 0) if False else 4, figsize=(20, 9))
    axes = axes.ravel()
    cmap = {0: "lightgray", 1: "tab:red", 2: "tab:orange", 3: "tab:green"}
    for ih, h in enumerate(HEIGHTS[:7]):
        ax = axes[ih]
        for code, lab in [(0, "unreachable"), (1, "approach infeasible"), (2, "reachable, not comfortable"),
                          (3, "good")]:
            m = status[ih] == code
            if code == 3:
                m = m & ~keepout
            if m.any():
                ax.scatter(XX[m], YY[m], s=30, marker="s", c=cmap[code], label=lab)
        r = results[ih]
        if r.get("rect"):
            ax.add_patch(Rectangle((r["x0"], -r["w"]), r["x1"] - r["x0"], 2 * r["w"], fill=False, ec="red", lw=2.5,
                                   label=f"table {r['x1'] - r['x0']:.2f}x{2 * r['w']:.2f} m"))
        if ih == sel:
            ax.add_patch(Rectangle((x0, -w), x1 - x0, 2 * w, fill=False, ec="blue", lw=2, ls="--",
                                   label="selected table (with margin)"))
        ax.add_patch(plt.Circle((0, 0), BASE_KEEPOUT, fc="none", ec="k", ls="--", label="base keep-out"))
        ax.plot(0, 0, "k^", ms=9)
        ax.set_title(f"Table top h = {h:+.2f} m" + ("  [SELECTED]" if ih == sel else ""), fontsize=10)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        ax.grid(True)
        if ih == 0 or ih == sel:
            ax.legend(fontsize=6, loc="upper right")
    ax = axes[min(n, 7)]
    ax.bar([f"{h:+.2f}" for h in HEIGHTS], areas, color="tab:blue")
    ax.set_xlabel("table top height h (m)")
    ax.set_ylabel("largest comfortable table area (m$^2$)")
    ax.set_title("Area vs height")
    for k in range(min(n, 7) + 1, len(axes)):
        axes[k].axis("off")
    fig.suptitle(f"Table design: tool-down vertical approach (flange {GRASP_OFFSET:.2f} -> {APPROACH_OFFSET:.2f} m above table, "
                 f"tool length {TOOL_LENGTH} m), "
                 f"all {N_YAW} yaws, sigma_min>={SIGMA_MIN}, clearance>={CLEARANCE_MIN} m", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    np.savez_compressed(DATA_PATH, heights=HEIGHTS, x_grid=X_GRID, y_grid=Y_GRID, status=status, sigma_min=sigma,
                        clearance=clear, max_jump=jump, areas=areas)
    print("Saved:", OUTPUT_PATH, "and", DATA_PATH)
    plt.show()


if __name__ == "__main__":
    main()