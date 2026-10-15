#!/usr/bin/env python3
"""Shared Franka Panda kinematics utilities for Lab 3 workspace/table scripts."""
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SCENE = PROJECT_DIR / "robot_descriptions" / "franka" / "pickandplace_scene.xml"
ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
TCP_SITE_NAME = "gripper"
TOOL_DOWN = np.array([0.0, 0.0, -1.0])


def align_vector(a, b):
    """Return rotation matrix mapping unit vector a to unit vector b."""
    a = np.asarray(a, float); a /= np.linalg.norm(a)
    b = np.asarray(b, float); b /= np.linalg.norm(b)
    v = np.cross(a, b); s = np.linalg.norm(v); c = float(np.dot(a, b))
    if s < 1e-10:
        if c > 0:
            return np.eye(3)
        trial = np.array([1.0, 0.0, 0.0])
        if abs(a @ trial) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, trial); axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(np.pi * axis).as_matrix()
    axis = v / s
    return Rotation.from_rotvec(axis * np.arctan2(s, c)).as_matrix()


class PandaKinematics:
    def __init__(self, scene_path=DEFAULT_SCENE):
        self.scene_path = Path(scene_path).expanduser().resolve()
        if not self.scene_path.exists():
            raise FileNotFoundError(f"Scene XML not found: {self.scene_path}")
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.joint_ids = np.array([self._required_id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINT_NAMES])
        self.qadr = self.model.jnt_qposadr[self.joint_ids].astype(int)
        self.dadr = self.model.jnt_dofadr[self.joint_ids].astype(int)
        self.site_id = self._required_id(mujoco.mjtObj.mjOBJ_SITE, TCP_SITE_NAME)
        self.lower = np.array([self.model.jnt_range[j, 0] if self.model.jnt_limited[j] else -2*np.pi for j in self.joint_ids])
        self.upper = np.array([self.model.jnt_range[j, 1] if self.model.jnt_limited[j] else 2*np.pi for j in self.joint_ids])
        mujoco.mj_forward(self.model, self.data)
        self.q_home = self.data.qpos[self.qadr].copy()
        self.q_full_home = self.data.qpos.copy()
        self.jacp = np.zeros((3, self.model.nv)); self.jacr = np.zeros((3, self.model.nv))
        self._tool_axis_local = np.array([0.0, 0.0, 1.0])

        # Table geometry is detected from the scene when possible.
        self.table_geoms = []
        for gid in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            body = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[gid])) or ""
            if "table" in (name + " " + body).lower() and int(self.model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_BOX):
                self.table_geoms.append(gid)
        self.table_top_z = None
        self.table_bounds = None
        if self.table_geoms:
            # Use the largest table-like box as the work surface.
            gid = max(self.table_geoms, key=lambda g: float(np.prod(self.model.geom_size[g][:2])))
            mujoco.mj_forward(self.model, self.data)
            c = self.data.geom_xpos[gid]; half = self.model.geom_size[gid]
            self.table_top_z = float(c[2] + half[2])
            self.table_bounds = (float(c[0]-half[0]), float(c[0]+half[0]), float(c[1]-half[1]), float(c[1]+half[1]))

    def _required_id(self, typ, name):
        idx = mujoco.mj_name2id(self.model, typ, name)
        if idx < 0:
            raise RuntimeError(f"Required MuJoCo object '{name}' not found in {self.scene_path}")
        return idx

    def set_q(self, q):
        self.data.qpos[:] = self.q_full_home
        self.data.qpos[self.qadr] = np.clip(np.asarray(q, float), self.lower, self.upper)
        mujoco.mj_forward(self.model, self.data)

    def tcp_pos(self):
        return self.data.site_xpos[self.site_id].copy()

    def tcp_rot(self):
        return self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def target_rotation(self, yaw):
        # Align the TCP site's local +Z axis downward, then rotate around world Z.
        R0 = align_vector(self._tool_axis_local, TOOL_DOWN)
        return Rotation.from_euler("z", yaw).as_matrix() @ R0

    def jacobian(self):
        mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.site_id)
        return np.vstack((self.jacp[:, self.dadr], self.jacr[:, self.dadr]))

    def ik(self, position, rotation=None, q_start=None, method="dls", max_iter=120, damping=0.06,
           pos_tol=0.004, rot_tol=0.025, max_step=0.20):
        """DLS IK; QP-like mode adds a bounded least-squares step for joint limits."""
        q = self.q_home.copy() if q_start is None else np.asarray(q_start, float).copy()
        q = np.clip(q, self.lower, self.upper)
        target = np.asarray(position, float)
        for it in range(1, max_iter + 1):
            self.set_q(q)
            ep = target - self.tcp_pos()
            er = np.zeros(3) if rotation is None else Rotation.from_matrix(rotation @ self.tcp_rot().T).as_rotvec()
            if np.linalg.norm(ep) <= pos_tol and (rotation is None or np.linalg.norm(er) <= rot_tol):
                return True, q.copy(), it, self.jacobian()
            J6 = self.jacobian()
            if rotation is None:
                J, e = J6[:3], ep
            else:
                J, e = J6, np.r_[ep, 0.6 * er]
            if method == "qp":
                # A small strictly convex QP: min ||J dq-e||^2 + damping^2||dq||^2
                # subject to joint position and per-iteration step bounds. Solved as bounded least squares.
                from scipy.optimize import lsq_linear
                A = np.vstack((J, damping*np.eye(7)))
                b = np.r_[e, np.zeros(7)]
                lo = np.maximum(self.lower-q, -max_step)
                hi = np.minimum(self.upper-q, max_step)
                try: dq = lsq_linear(A, b, bounds=(lo, hi), max_iter=40).x
                except Exception: return False, None, it, J6
            else:
                try: dq = J.T @ np.linalg.solve(J @ J.T + damping**2*np.eye(J.shape[0]), e)
                except np.linalg.LinAlgError: return False, None, it, J6
                dq = np.clip(dq, -max_step, max_step)
                dq = np.clip(dq, self.lower-q, self.upper-q)
            q = np.clip(q + dq, self.lower, self.upper)
        self.set_q(q)
        return False, None, max_iter, self.jacobian()

    def sampled_link_clearance(self, table_z, x_bounds=None):
        """Approximate link/table clearance using joint-body origins and TCP; not mesh collision checking."""
        body_points = []
        for jid in self.joint_ids:
            bid = int(self.model.jnt_bodyid[jid])
            body_points.append(self.data.xpos[bid].copy())
        body_points.append(self.tcp_pos())
        pts = np.asarray(body_points)
        best = np.inf
        for a, b in zip(pts[:-1], pts[1:]):
            seg = np.linspace(a, b, 8)
            if x_bounds is not None:
                seg = seg[(seg[:, 0] >= x_bounds[0]) & (seg[:, 0] <= x_bounds[1])]
            if len(seg): best = min(best, float(np.min(seg[:, 2] - table_z)))
        return float(best if np.isfinite(best) else np.inf)