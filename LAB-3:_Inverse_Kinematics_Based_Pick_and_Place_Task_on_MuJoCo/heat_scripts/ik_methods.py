#!/usr/bin/env python3
"""Pluggable inverse-kinematics backends for the HEAL pick-and-place task.

Three backends share one interface so they can be compared fairly:

    mink     off-the-shelf differential IK (Stage 1 baseline)
    dls      Damped Least Squares, closed loop, own implementation
    qp       Quadratic-program IK with joint-position and joint-velocity
             limits as hard constraints, own implementation
    dls_raw  (optional) DLS with NO limit handling, to show what the
             constraints in the other methods buy you

Every backend iterates from the scene's *current* configuration towards a
target TCP pose and reports iterations, residual error and constraint
violations.  `IKPlanner` adds the grasp-yaw search used in Stage 1
(try yaw + k*90 deg, keep the cheapest solution, fall back to position-only).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# Single source of truth for the shared IK settings.
from pickandplace import (
    DT_IK,
    IK_MAX_STEPS,
    IK_POS_TOL,
    IK_ROT_TOL,
    ORIENTATION_COST,
    target_pose,
)

# Joint speed allowed *inside the IK loop* (rad/s).  A step is counted as a
# velocity violation when |dq| / DT_IK exceeds this.  DLS and QP enforce it,
# mink (as configured in Stage 1) does not, so its violations are measured.
IK_VEL_LIMIT = 1.5

LIMIT_TOL = 1e-6            # rad, tolerance when counting limit violations


# ============================================================
# RESULT TYPES
# ============================================================

@dataclass
class IKResult:
    q: np.ndarray | None        # arm joint solution (None if not converged)
    converged: bool
    iters: int
    pos_err: float
    rot_err: float
    vel_viol: int = 0           # iterations with |dq|/dt > IK_VEL_LIMIT
    lim_viol: int = 0           # iterations where q left the joint range
    max_speed: float = 0.0      # max |dq|/dt seen (rad/s)


@dataclass
class SolveInfo:
    """Aggregate over all IK runs needed for one waypoint."""
    mode: str = "none"          # oriented | position-only | failed
    runs: int = 0
    total_iters: int = 0        # every iteration of every run
    accepted_iters: int = 0     # iterations of the run that was used
    time_s: float = 0.0
    vel_viol: int = 0
    lim_viol: int = 0
    max_speed: float = 0.0
    pos_err: float = float("nan")
    rot_err: float = float("nan")


def target_pos_rot(position, yaw):
    """Target TCP position and rotation matrix (gripper pointing down)."""
    R = (
        Rotation.from_euler("z", yaw).as_matrix()
        @ Rotation.from_euler("x", np.pi).as_matrix()
    )
    return np.asarray(position, dtype=float), R


# ============================================================
# BASE CLASS
# ============================================================

class IKBackend:
    name = "base"

    def __init__(self, scene):
        self.scene = scene
        self.model = scene.model
        self.arm_qadr = scene.arm_qadr
        self.arm_dadr = scene.arm_dadr

        joint_ids = [self.model.joint(f"joint_{i}").id for i in range(1, 7)]
        self.q_lo = self.model.jnt_range[joint_ids, 0].copy()
        self.q_hi = self.model.jnt_range[joint_ids, 1].copy()

    def run(self, position, yaw, use_orientation) -> IKResult:
        raise NotImplementedError

    # helpers shared by all backends
    def _count(self, q_new, dq):
        speed = np.abs(dq).max() / DT_IK
        vel = int(speed > IK_VEL_LIMIT * (1 + 1e-9))
        lim = int(np.any(q_new < self.q_lo - LIMIT_TOL)
                  or np.any(q_new > self.q_hi + LIMIT_TOL))
        return vel, lim, speed


# ============================================================
# 1) MINK (off-the-shelf baseline)
# ============================================================

class MinkIK(IKBackend):
    name = "mink"

    def __init__(self, scene, solver="daqp", vel_limit=False):
        super().__init__(scene)
        import mink

        # vel_limit=True adds mink.VelocityLimit (method "mink_vl") so mink
        # can be compared on equal terms with the constrained DLS / QP.
        if vel_limit:
            self.name = "mink_vl"

        self.mink = mink
        self.solver = solver
        self.cfg = mink.Configuration(self.model)

        def frame_task(orientation_cost):
            return mink.FrameTask(
                frame_name="gripper/pinch",
                frame_type="site",
                position_cost=1.0,
                orientation_cost=orientation_cost,
                lm_damping=1e-4,
                gain=0.5,
            )

        self.task_oriented = frame_task(ORIENTATION_COST)
        self.task_position = frame_task(0.0)

        arm_dofs = set(map(int, scene.arm_dadr))
        frozen = [i for i in range(self.model.nv) if i not in arm_dofs]
        self.freeze = mink.DofFreezingTask(
            model=self.model, dof_indices=frozen
        )
        self.posture = mink.PostureTask(self.model, cost=1e-3)
        self.limits = [mink.ConfigurationLimit(self.model)]
        if vel_limit:
            self.limits.append(
                mink.VelocityLimit(
                    self.model,
                    {f"joint_{i}": np.array([IK_VEL_LIMIT])
                     for i in range(1, 7)},
                )
            )

    def run(self, position, yaw, use_orientation):
        mink = self.mink
        task = self.task_oriented if use_orientation else self.task_position

        self.cfg.update(self.scene.data.qpos)
        self.posture.set_target_from_configuration(self.cfg)
        task.set_target(target_pose(position, yaw))

        vel_v = lim_v = 0
        max_speed = 0.0

        for it in range(IK_MAX_STEPS + 1):
            err = task.compute_error(self.cfg)
            pe, re = np.linalg.norm(err[:3]), np.linalg.norm(err[3:])
            ok = pe < IK_POS_TOL and (not use_orientation or re < IK_ROT_TOL)
            if ok:
                return IKResult(
                    self.cfg.q[self.arm_qadr].copy(), True, it, pe, re,
                    vel_v, lim_v, max_speed,
                )
            if it == IK_MAX_STEPS:
                break

            q_before = self.cfg.q[self.arm_qadr].copy()
            velocity = mink.solve_ik(
                self.cfg,
                [task, self.posture],
                DT_IK,
                self.solver,
                damping=1e-3,
                limits=self.limits,
                constraints=[self.freeze],
            )
            self.cfg.integrate_inplace(velocity, DT_IK)

            q_new = self.cfg.q[self.arm_qadr]
            v, l, s = self._count(q_new, q_new - q_before)
            vel_v += v
            lim_v += l
            max_speed = max(max_speed, s)

        return IKResult(None, False, IK_MAX_STEPS, pe, re,
                        vel_v, lim_v, max_speed)


# ============================================================
# Shared machinery for the two hand-written methods
# ============================================================

class _JacobianIK(IKBackend):
    """Kinematics + weighted task residual used by DLS and QP."""

    def __init__(self, scene, gain=0.5):
        super().__init__(scene)
        self.gain = gain
        self.d = mujoco.MjData(self.model)       # private scratch data
        self.site = scene.tcp_site
        self._jp = np.zeros((3, self.model.nv))
        self._jr = np.zeros((3, self.model.nv))

    def _task(self, q, p_t, R_t, use_orientation):
        """Return weighted Jacobian Jw, weighted residual ew, raw errors."""
        d = self.d
        d.qpos[self.arm_qadr] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)

        p = d.site_xpos[self.site]
        R = d.site_xmat[self.site].reshape(3, 3)

        e_pos = p_t - p
        e_rot = Rotation.from_matrix(R_t @ R.T).as_rotvec()   # world frame

        mujoco.mj_jacSite(self.model, d, self._jp, self._jr, self.site)
        Jp = self._jp[:, self.arm_dadr]
        Jr = self._jr[:, self.arm_dadr]

        if use_orientation:
            w = ORIENTATION_COST
            Jw = np.vstack([Jp, w * Jr])
            ew = np.concatenate([e_pos, w * e_rot])
        else:
            Jw, ew = Jp, e_pos

        return Jw, ew, float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot))

    def _begin(self):
        self.d.qpos[:] = self.scene.data.qpos
        return self.scene.data.qpos[self.arm_qadr].copy()

    def _step_fn(self, q, Jw, ew):
        raise NotImplementedError

    def run(self, position, yaw, use_orientation):
        p_t, R_t = target_pos_rot(position, yaw)
        q = self._begin()
        vel_v = lim_v = 0
        max_speed = 0.0

        for it in range(IK_MAX_STEPS + 1):
            Jw, ew, pe, re = self._task(q, p_t, R_t, use_orientation)
            ok = pe < IK_POS_TOL and (not use_orientation or re < IK_ROT_TOL)
            if ok:
                return IKResult(q.copy(), True, it, pe, re,
                                vel_v, lim_v, max_speed)
            if it == IK_MAX_STEPS:
                break

            q_new = self._step_fn(q, Jw, ew)
            v, l, s = self._count(q_new, q_new - q)
            vel_v += v
            lim_v += l
            max_speed = max(max_speed, s)
            q = q_new

        return IKResult(None, False, IK_MAX_STEPS, pe, re,
                        vel_v, lim_v, max_speed)


# ============================================================
# 2) DAMPED LEAST SQUARES (closed loop)
# ============================================================

class DLSIK(_JacobianIK):
    """dq = gain * J^T (J J^T + lambda^2 I)^-1 e, with adaptive damping.

    Damping follows Chiaverini / Maciejewski: lambda grows smoothly as the
    smallest singular value of J drops below `eps`, so the solver stays
    robust near singularities without slowing down far from them.

    constrained=True   -> the step is uniformly scaled to respect the joint
                          speed limit and the result is clamped to the joint
                          range (the usual practical DLS).
    constrained=False  -> no limit handling at all ("dls_raw").
    """

    def __init__(self, scene, gain=0.5, lam_min=0.01, lam_max=0.1,
                 eps=0.05, constrained=True):
        super().__init__(scene, gain)
        self.lam_min, self.lam_max, self.eps = lam_min, lam_max, eps
        self.constrained = constrained
        self.name = "dls" if constrained else "dls_raw"

    def _step_fn(self, q, Jw, ew):
        U, S, Vt = np.linalg.svd(Jw, full_matrices=False)

        lam2 = self.lam_min ** 2
        if S[-1] < self.eps:
            lam2 += self.lam_max ** 2 * (1.0 - (S[-1] / self.eps) ** 2)

        # J^+ e via the SVD: sum_i sigma_i / (sigma_i^2 + lam^2) * v_i u_i^T e
        coeff = S / (S ** 2 + lam2) * (U.T @ ew)
        dq = self.gain * (Vt.T @ coeff)

        if not self.constrained:
            return q + dq

        peak = np.abs(dq).max()
        limit = IK_VEL_LIMIT * DT_IK
        if peak > limit:
            dq *= limit / peak                      # keep direction
        return np.clip(q + dq, self.q_lo, self.q_hi)


# ============================================================
# 3) QP-BASED IK with joint and velocity limits
# ============================================================

class QPIK(_JacobianIK):
    """Each iteration solves

        min_dq  0.5 || W (J dq - gain * e) ||^2 + 0.5 * mu * ||dq||^2
        s.t.    q_lo - q <= dq <= q_hi - q          (joint position limits)
               -vmax*dt  <= dq <= vmax*dt           (joint velocity limits)

    so both limits are hard constraints of the optimisation rather than
    after-the-fact clipping.
    """

    name = "qp"

    def __init__(self, scene, gain=0.5, mu=1e-3, solver="daqp"):
        super().__init__(scene, gain)
        self.mu = mu
        self.solver = solver
        self.qp_failures = 0
        import qpsolvers
        self._qp = qpsolvers

    def _step_fn(self, q, Jw, ew):
        n = len(q)
        P = Jw.T @ Jw + self.mu * np.eye(n)
        c = -Jw.T @ (self.gain * ew)

        vmax = IK_VEL_LIMIT * DT_IK
        lb = np.maximum(self.q_lo - q, -vmax)
        ub = np.minimum(self.q_hi - q, vmax)
        # numerical safety if q sits marginally outside its range
        lb = np.minimum(lb, 0.0)
        ub = np.maximum(ub, 0.0)

        try:
            dq = self._qp.solve_qp(P, c, lb=lb, ub=ub, solver=self.solver)
        except Exception:
            dq = None

        if dq is None:
            self.qp_failures += 1
            return q.copy()
        return q + dq


# ============================================================
# PLANNER: yaw search + fallback (same strategy for every backend)
# ============================================================

class IKPlanner:

    YAW_OFFSETS = (0, 1, -1, 2)         # multiples of 90 deg

    def __init__(self, backend: IKBackend):
        self.backend = backend
        self.scene = backend.scene

    def solve(self, position, yaw):
        """Return (q or None, SolveInfo)."""
        t0 = time.perf_counter()
        q_now = self.scene.data.qpos[self.scene.arm_qadr].copy()
        info = SolveInfo()

        def account(res):
            info.runs += 1
            info.total_iters += res.iters
            info.vel_viol += res.vel_viol
            info.lim_viol += res.lim_viol
            info.max_speed = max(info.max_speed, res.max_speed)

        best, best_cost = None, np.inf
        for k in self.YAW_OFFSETS:
            res = self.backend.run(position, yaw + k * np.pi / 2, True)
            account(res)
            if res.converged:
                cost = float(np.sum(np.abs(res.q - q_now)))
                if cost < best_cost:
                    best, best_cost = res, cost

        if best is not None:
            info.mode = "oriented"
        else:
            res = self.backend.run(position, yaw, False)
            account(res)
            if res.converged:
                best, info.mode = res, "position-only"
            else:
                info.mode = "failed"

        if best is not None:
            info.accepted_iters = best.iters
            info.pos_err, info.rot_err = best.pos_err, best.rot_err

        info.time_s = time.perf_counter() - t0
        return (best.q if best is not None else None), info


# ============================================================
# FACTORY
# ============================================================

METHODS = ("mink", "mink_vl", "dls", "qp", "dls_raw")


def make_backend(name, scene, solver="daqp"):
    if name == "mink":
        return MinkIK(scene, solver=solver)
    if name == "mink_vl":
        return MinkIK(scene, solver=solver, vel_limit=True)
    if name == "dls":
        return DLSIK(scene, constrained=True)
    if name == "dls_raw":
        return DLSIK(scene, constrained=False)
    if name == "qp":
        return QPIK(scene, solver=solver)
    raise ValueError(f"unknown IK method '{name}', choose from {METHODS}")