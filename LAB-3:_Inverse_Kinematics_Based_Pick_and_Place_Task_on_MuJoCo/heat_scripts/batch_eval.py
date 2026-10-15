#!/usr/bin/env python3
"""Batch pick-and-place evaluation and IK-method comparison.

Runs the Stage 1 pipeline (approach, descend, grasp, lift, transit, place,
release) over many identical seeds with one or more IK backends, logs every
episode, and writes a summary with failure analysis.

Examples
--------
    # Section 10: >=25 episodes with the Mink baseline, run overnight
    python3 scripts/batch_eval.py --episodes 30 --methods mink --screenshots

    # Sections 12-13: compare Mink against the custom DLS and QP solvers
    python3 scripts/batch_eval.py --episodes 30 --methods mink dls qp --screenshots

    # Resume an interrupted overnight run (already finished seeds are skipped)
    python3 scripts/batch_eval.py --episodes 30 --methods mink dls qp \
        --out results/overnight --resume

    # Watch a single method live
    python3 scripts/batch_eval.py --episodes 2 --methods qp --viewer --speed 2

Outputs (in --out, default results/batch_<timestamp>/)
----------------------------------------------------
    episodes.jsonl   one JSON record per (method, seed), written immediately
    episodes.csv     the same records, flat, for spreadsheets / plotting
    summary.md       metrics table, constraint violations, failure analysis
    summary.json     the same numbers, machine readable
    config.json      every setting that influenced the run
    failures/        <method>_seed<N>.png screenshot + .npz simulator state
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

from envsetup import (
    PickPlaceScene,
    CUBE_SIZE,
    TABLE_TOP_Z,
    TRAY_CENTER,
    TRAY_INNER,
    GRIPPER_OPEN,
    GRIPPER_CLOSE,
    MAX_JOINT_SPEED,
    TRAY_FLOOR,
)
from pickandplace import (
    APPROACH_HEIGHT,
    LIFT_HEIGHT,
    TRANSIT_HEIGHT,
    DESCENT_SUBSTEPS,
    GRIP_STEPS,
    SETTLE_STEPS,
    GRASP_Z_OFFSET,
    PLACE_CLEARANCE,
    CARRY_SPEED,
    DT_IK,
    IK_MAX_STEPS,
    IK_POS_TOL,
    IK_ROT_TOL,
    ORIENTATION_COST,
)
from ik_methods import IK_VEL_LIMIT, METHODS, IKPlanner, make_backend


# ============================================================
# EVALUATION SETTINGS
# ============================================================

TRACK_TOL = 0.04            # m, TCP error that still counts as "arrived"
SIM_VEL_LIMIT = 1.05 * MAX_JOINT_SPEED      # rad/s, simulation speed check
CLEARANCE_EVERY = 20        # sim steps between clearance samples
CLEARANCE_MAX = 0.5         # m, distances beyond this are not computed
COLLISION_TOL = -0.002      # m, penetration deeper than this = collision
SLIP_JUMP = 0.04            # m, growth of |cube - tcp| that means a slip
LIMIT_MARGIN = 0.05         # rad, "close to a joint limit"
STRUCTURAL_DIST = 0.01      # m, geoms this close at home are ignored

MITIGATIONS = {
    "ik_infeasible": (
        "Better pre-grasp: add a higher intermediate waypoint, seed IK from "
        "a pre-grasp posture, try other grasp yaws (+/-90 deg, already on), "
        "relax ORIENTATION_COST, or move the cube spawn region towards the base."
    ),
    "tracking_error": (
        "The arm did not arrive: slow the move, allow a longer convergence "
        "wait, raise DESCENT_SUBSTEPS so each IK step is smaller."
    ),
    "joint_limit": (
        "Solution sits at a joint limit: choose the yaw candidate farthest "
        "from the limits, use a different approach direction (tilt the "
        "gripper 10-20 deg), or re-seed IK from the home posture."
    ),
    "collision": (
        "Raise APPROACH_HEIGHT / TRANSIT_HEIGHT, widen the tray "
        "(TRAY_INNER) or lower its walls, add a collision-margin task to the IK."
    ),
    "grasp_slip": (
        "Raise CUBE_FRICTION or grip force, lower CARRY_SPEED, grasp lower "
        "(GRASP_Z_OFFSET) or pick the yaw so the pads meet flat faces."
    ),
    "place_miss": (
        "Cube left the gripper away from the tray: add a settle before "
        "release, lower the release height, widen the tray."
    ),
    "place_unstable": (
        "Cube rests on a wall or tilted: lower the release height and add a "
        "longer settle after release."
    ),
}


# ============================================================
# SCENE WITH MONITORING
# ============================================================

class MonitoredScene(PickPlaceScene):
    """PickPlaceScene that records clearance, speed and limit statistics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase = "init"
        self._renderer = None
        self._setup_monitor()
        self.reset_monitor()

    # ---------------- monitor setup ----------------

    def _setup_monitor(self):
        m, d = self.model, self.data
        base = m.body("base_link").id
        parent = m.body_parentid

        def under_arm(body):
            while body != 0:
                if body == base:
                    return True
                body = parent[body]
            return False

        def gname(g):
            return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""

        collidable = [
            g for g in range(m.ngeom)
            if m.geom_contype[g] or m.geom_conaffinity[g]
        ]
        arm = [
            g for g in collidable
            if under_arm(m.geom_bodyid[g]) and m.geom_bodyid[g] != base
        ]
        self.table_geoms = [g for g in range(m.ngeom) if gname(g) == "lab_table_top"]
        self.tray_geoms = [g for g in range(m.ngeom) if gname(g).startswith("tray_")]

        # Drop geoms that are structurally next to the table/tray at the home
        # pose (e.g. a mesh that always touches), they would pin the minimum.
        self.arm_geoms, self.ignored_geoms = [], []
        for g in arm:
            near = min(
                mujoco.mj_geomDistance(m, d, g, o, CLEARANCE_MAX, None)
                for o in self.table_geoms + self.tray_geoms
            )
            (self.ignored_geoms if near < STRUCTURAL_DIST else self.arm_geoms).append(g)

        joint_ids = [m.joint(f"joint_{i}").id for i in range(1, 7)]
        self.jr_lo = m.jnt_range[joint_ids, 0].copy()
        self.jr_hi = m.jnt_range[joint_ids, 1].copy()

    def reset_monitor(self):
        self.mon = {
            "min_clear_table": CLEARANCE_MAX,
            "min_clear_tray": CLEARANCE_MAX,
            "clear_phase": "",
            "collision_samples": 0,
            "vel_viol_steps": 0,
            "max_speed": 0.0,
            "jl_viol_max": 0.0,
            "steps": 0,
        }
        self._k = 0
        self._t0 = float(self.data.time)

    # ---------------- stepping ----------------

    def _sample_clearance(self):
        m, d, mon = self.model, self.data, self.mon
        overall = min(mon["min_clear_table"], mon["min_clear_tray"])
        sample_min = CLEARANCE_MAX
        for g in self.arm_geoms:
            for o in self.table_geoms:
                dist = mujoco.mj_geomDistance(m, d, g, o, CLEARANCE_MAX, None)
                mon["min_clear_table"] = min(mon["min_clear_table"], dist)
                sample_min = min(sample_min, dist)
            for o in self.tray_geoms:
                dist = mujoco.mj_geomDistance(m, d, g, o, CLEARANCE_MAX, None)
                mon["min_clear_tray"] = min(mon["min_clear_tray"], dist)
                sample_min = min(sample_min, dist)
        if sample_min < overall:
            mon["clear_phase"] = self.phase         # phase of the new minimum
        if sample_min < COLLISION_TOL:
            mon["collision_samples"] += 1

    def step(self, n=1):
        mon = self.mon
        for _ in range(n):
            self.apply_control()
            mujoco.mj_step(self.model, self.data)

            self._k += 1
            mon["steps"] += 1

            speed = float(np.abs(self.data.qvel[self.arm_dadr]).max())
            mon["max_speed"] = max(mon["max_speed"], speed)
            if speed > SIM_VEL_LIMIT:
                mon["vel_viol_steps"] += 1

            q = self.data.qpos[self.arm_qadr]
            excess = max(float((self.jr_lo - q).max()), float((q - self.jr_hi).max()), 0.0)
            mon["jl_viol_max"] = max(mon["jl_viol_max"], excess)

            if self._k % CLEARANCE_EVERY == 0:
                self._sample_clearance()

            if self.viewer is not None:
                self._step_count += 1
                if self._step_count % self.sync_every == 0:
                    if not self.viewer.is_running():
                        raise SystemExit
                    self.viewer.sync()
                    time.sleep(
                        self.model.opt.timestep * self.sync_every / self.realtime_factor
                    )

    # ---------------- failure artefacts ----------------

    def save_state(self, path, extra):
        np.savez(
            path,
            qpos=self.data.qpos.copy(),
            qvel=self.data.qvel.copy(),
            ctrl=self.data.ctrl.copy(),
            q_target=self.q_target.copy(),
            time=self.data.time,
            tcp=self.tcp_pos(),
            cube=self.cube_pos(),
            **extra,
        )

    def save_screenshot(self, path):
        try:
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, 480, 640)
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [0.30, 0.0, 0.15]
            cam.distance, cam.azimuth, cam.elevation = 1.8, 135, -25
            self._renderer.update_scene(self.data, camera=cam)
            image = self._renderer.render()
            from PIL import Image
            Image.fromarray(image).save(path)
            return True
        except Exception as exc:                     # headless GL, no PIL...
            if not getattr(self, "_warned_shot", False):
                print(f"  (screenshot unavailable: {exc}; "
                      "try MUJOCO_GL=egl or MUJOCO_GL=osmesa)")
                self._warned_shot = True
            return False


# ============================================================
# EPISODE RUNNER
# ============================================================

class EpisodeRunner:

    def __init__(self, scene, method, planner, out_dir, screenshots):
        self.scene = scene
        self.method = method
        self.planner = planner
        self.out_dir = Path(out_dir)
        self.screenshots = screenshots

    # ---------------- motion primitives ----------------

    def _move(self, stage, position, yaw):
        sc = self.scene
        q, info = self.planner.solve(position, yaw)
        self.calls.append({
            "stage": stage,
            "mode": info.mode,
            "runs": info.runs,
            "iters_accepted": info.accepted_iters,
            "iters_total": info.total_iters,
            "time_s": round(info.time_s, 5),
            "vel_viol": info.vel_viol,
            "lim_viol": info.lim_viol,
            "max_speed": round(info.max_speed, 3),
        })

        if q is None:
            self.fail = ("ik_infeasible", stage)
            return False

        sc.q_target = q
        for _ in range(80):
            sc.step(50)
            q_err = np.max(np.abs(q - sc.data.qpos[sc.arm_qadr]))
            if q_err < 0.01:
                break

        err = float(np.linalg.norm(sc.tcp_pos() - np.asarray(position)))
        self.track_errs.append(err)
        if err >= TRACK_TOL:
            near_limit = bool(
                np.any(q - sc.jr_lo < LIMIT_MARGIN) or np.any(sc.jr_hi - q < LIMIT_MARGIN)
            )
            self.fail = ("joint_limit" if near_limit else "tracking_error", stage)
            return False
        return True

    def _move_linear(self, stage, p0, p1, yaw, n=DESCENT_SUBSTEPS):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        for a in np.linspace(0.0, 1.0, n + 1)[1:]:
            if not self._move(stage, (1 - a) * p0 + a * p1, yaw):
                return False
        return True

    def _check(self, label):
        sc = self.scene
        d = sc.cube_pos() - sc.tcp_pos()
        self.checks[label] = {
            "dist": round(float(np.linalg.norm(d)), 4),
            "cube_z": round(float(sc.cube_pos()[2]), 4),
        }

    # ---------------- one episode ----------------

    def run(self, seed):
        sc = self.scene
        wall0 = time.perf_counter()

        self.calls = []
        self.checks = {}
        self.track_errs = []
        self.fail = None

        cube_x, cube_y, cube_yaw = sc.reset(seed=seed)
        sc.reset_monitor()
        qp_fail0 = getattr(self.planner.backend, "qp_failures", 0)

        grasp = np.array([cube_x, cube_y, TABLE_TOP_Z + CUBE_SIZE / 2 + GRASP_Z_OFFSET])
        approach = grasp + np.array([0, 0, APPROACH_HEIGHT])
        lift = grasp + np.array([0, 0, LIFT_HEIGHT])

        sc.phase = "settle"
        sc.step(200)

        ok = True
        sc.phase = "approach"
        ok = ok and self._move("approach", approach, cube_yaw)

        if ok:
            sc.phase = "descend"
            ok = self._move_linear("descend", approach, grasp, cube_yaw)
        if ok:
            sc.step(SETTLE_STEPS)
            sc.phase = "grasp"
            sc.grip_cmd = GRIPPER_CLOSE
            sc.step(GRIP_STEPS)
            sc.max_speed = CARRY_SPEED
            self._check("grasped")
            held = sc.cube_pos() - sc.tcp_pos()
            place_center = np.array([
                TRAY_CENTER[0], TRAY_CENTER[1],
                TABLE_TOP_Z + TRAY_FLOOR + CUBE_SIZE / 2 + PLACE_CLEARANCE,
            ])
            place = place_center - held
            pre_place = place + np.array([0, 0, TRANSIT_HEIGHT])

            sc.phase = "lift"
            ok = self._move_linear("lift", grasp, lift, cube_yaw)
            if ok:
                self._check("lifted")
        if ok:
            sc.phase = "transit"
            ok = self._move("transit", pre_place, 0.0)
            if ok:
                self._check("transit")
        if ok:
            sc.phase = "place"
            ok = self._move_linear("place", pre_place, place, 0.0)
            if ok:
                sc.step(SETTLE_STEPS)
                self._check("placed")
        if ok:
            sc.phase = "release"
            sc.grip_cmd = GRIPPER_OPEN
            sc.max_speed = MAX_JOINT_SPEED
            sc.step(GRIP_STEPS)
            sc.step(250)

        in_tray = bool(ok and sc.cube_in_tray())
        reason, stage = self._classify(ok, in_tray)

        rec = self._record(seed, (cube_x, cube_y, cube_yaw), in_tray, reason, stage,
                           wall0, qp_fail0)

        if not in_tray:
            self._save_failure(rec, seed)

        # leave the scene in a clean state for the next method / seed
        sc.grip_cmd = GRIPPER_OPEN
        sc.max_speed = MAX_JOINT_SPEED
        return rec

    # ---------------- classification ----------------

    def _classify(self, moves_ok, in_tray):
        if in_tray:
            return "", ""
        mon = self.scene.mon
        if self.fail is not None:
            return self.fail
        if min(mon["min_clear_table"], mon["min_clear_tray"]) < COLLISION_TOL:
            return "collision", mon["clear_phase"]

        g = self.checks.get("grasped", {}).get("dist")
        if g is not None:
            later = [v for k, v in self.checks.items() if k != "grasped"]
            jump = max((v["dist"] - g for v in later), default=0.0)
            low = self.checks.get("lifted", {}).get("cube_z", 1.0) < 0.08
            if jump > SLIP_JUMP or low:
                return "grasp_slip", "carry"

        p = self.scene.cube_pos()
        limit = TRAY_INNER / 2 - CUBE_SIZE / 2
        inside = (abs(p[0] - TRAY_CENTER[0]) <= limit
                  and abs(p[1] - TRAY_CENTER[1]) <= limit)
        return ("place_unstable" if inside else "place_miss"), "release"

    # ---------------- logging ----------------

    def _record(self, seed, cube_pose, success, reason, stage, wall0, qp_fail0):
        sc, mon = self.scene, self.scene.mon
        calls = self.calls
        n = len(calls)

        acc = [c["iters_accepted"] for c in calls if c["mode"] != "failed"]
        t_tot = sum(c["time_s"] for c in calls)
        cube = sc.cube_pos()
        place_err = float(np.hypot(cube[0] - TRAY_CENTER[0], cube[1] - TRAY_CENTER[1]))
        clr_t, clr_r = mon["min_clear_table"], mon["min_clear_tray"]
        grasp = self.checks.get("grasped", {})
        slip = 0.0
        if grasp:
            slip = max((v["dist"] - grasp["dist"] for k, v in self.checks.items()
                        if k != "grasped"), default=0.0)

        return {
            "method": self.method,
            "seed": seed,
            "success": bool(success),
            "failure_reason": reason,
            "failure_stage": stage,
            "cube_x": round(float(cube_pose[0]), 4),
            "cube_y": round(float(cube_pose[1]), 4),
            "cube_yaw_deg": round(math.degrees(float(cube_pose[2])), 2),
            "final_cube_x": round(float(cube[0]), 4),
            "final_cube_y": round(float(cube[1]), 4),
            "final_cube_z": round(float(cube[2]), 4),
            "place_err_xy_m": round(place_err, 4),
            "ik_calls": n,
            "ik_time_total_s": round(t_tot, 4),
            "ik_time_mean_call_s": round(t_tot / n, 5) if n else float("nan"),
            "ik_iters_mean": round(float(np.mean(acc)), 2) if acc else float("nan"),
            "ik_iters_max": int(max(acc)) if acc else 0,
            "ik_iters_total_all_runs": int(sum(c["iters_total"] for c in calls)),
            "ik_fallback_calls": sum(c["mode"] == "position-only" for c in calls),
            "ik_failed_calls": sum(c["mode"] == "failed" for c in calls),
            "ik_vel_viol_iters": sum(c["vel_viol"] for c in calls),
            "ik_lim_viol_iters": sum(c["lim_viol"] for c in calls),
            "ik_max_speed_rad_s": round(max((c["max_speed"] for c in calls), default=0.0), 3),
            "qp_solver_failures": getattr(self.planner.backend, "qp_failures", 0) - qp_fail0,
            "track_err_max_m": round(max(self.track_errs), 4) if self.track_errs else float("nan"),
            "min_clearance_m": round(min(clr_t, clr_r), 5),
            "min_clearance_table_m": round(clr_t, 5),
            "min_clearance_tray_m": round(clr_r, 5),
            "min_clearance_phase": mon["clear_phase"],
            "collision_samples": mon["collision_samples"],
            "sim_vel_viol_steps": mon["vel_viol_steps"],
            "sim_max_joint_speed_rad_s": round(mon["max_speed"], 3),
            "sim_joint_limit_excess_rad": round(mon["jl_viol_max"], 5),
            "grasp_dist_m": grasp.get("dist", float("nan")),
            "max_slip_m": round(slip, 4),
            "sim_time_s": round(float(sc.data.time) - sc._t0, 2),
            "wall_time_s": round(time.perf_counter() - wall0, 2),
            "screenshot": "",
            "state_file": "",
            "ik_calls_detail": calls,
            "checkpoints": self.checks,
        }

    def _save_failure(self, rec, seed):
        fdir = self.out_dir / "failures"
        fdir.mkdir(parents=True, exist_ok=True)
        stem = f"{self.method}_seed{seed:03d}"
        state = fdir / f"{stem}.npz"
        self.scene.save_state(state, {"seed": seed})
        rec["state_file"] = str(state.relative_to(self.out_dir))
        if self.screenshots:
            png = fdir / f"{stem}.png"
            if self.scene.save_screenshot(png):
                rec["screenshot"] = str(png.relative_to(self.out_dir))


# ============================================================
# STATISTICS AND REPORTING
# ============================================================

def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / den, (centre + half) / den


def _mean_std(values):
    v = np.array([x for x in values if x == x], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def summarize(records, methods):
    out = {}
    for method in methods:
        rs = [r for r in records if r["method"] == method]
        n = len(rs)
        if n == 0:
            continue
        k = sum(r["success"] for r in rs)
        lo, hi = wilson(k, n)
        ik_t = _mean_std([r["ik_time_total_s"] for r in rs])
        ik_it = _mean_std([r["ik_iters_mean"] for r in rs])
        calls = sum(r["ik_calls"] for r in rs)
        reasons = Counter(
            f"{r['failure_reason']}@{r['failure_stage']}" if r["failure_stage"]
            else r["failure_reason"]
            for r in rs if not r["success"]
        )
        out[method] = {
            "episodes": n,
            "successes": k,
            "success_rate": k / n,
            "success_ci95": [lo, hi],
            "ik_time_per_episode_mean_s": ik_t[0],
            "ik_time_per_episode_std_s": ik_t[1],
            "ik_time_per_call_mean_s": (
                sum(r["ik_time_total_s"] for r in rs) / calls if calls else float("nan")),
            "ik_iters_mean": ik_it[0],
            "ik_iters_std": ik_it[1],
            "ik_iters_max": max(r["ik_iters_max"] for r in rs),
            "fallback_call_fraction": (
                sum(r["ik_fallback_calls"] for r in rs) / calls if calls else float("nan")),
            "ik_failed_calls": sum(r["ik_failed_calls"] for r in rs),
            "min_clearance_min_m": min(r["min_clearance_m"] for r in rs),
            "min_clearance_mean_m": _mean_std([r["min_clearance_m"] for r in rs])[0],
            "collision_episodes": sum(r["collision_samples"] > 0 for r in rs),
            "ik_vel_viol_iters": sum(r["ik_vel_viol_iters"] for r in rs),
            "ik_vel_viol_episodes": sum(r["ik_vel_viol_iters"] > 0 for r in rs),
            "ik_lim_viol_iters": sum(r["ik_lim_viol_iters"] for r in rs),
            "ik_lim_viol_episodes": sum(r["ik_lim_viol_iters"] > 0 for r in rs),
            "ik_max_speed_rad_s": max(r["ik_max_speed_rad_s"] for r in rs),
            "sim_vel_viol_steps": sum(r["sim_vel_viol_steps"] for r in rs),
            "sim_max_joint_speed_rad_s": max(r["sim_max_joint_speed_rad_s"] for r in rs),
            "sim_joint_limit_excess_max_rad": max(r["sim_joint_limit_excess_rad"] for r in rs),
            "place_err_mean_m": _mean_std([r["place_err_xy_m"] for r in rs if r["success"]])[0],
            "failure_reasons": dict(reasons),
        }
    return out


def fmt(x, spec=".3f"):
    return "n/a" if x is None or x != x else format(x, spec)


def build_markdown(records, summary, methods, config):
    L = []
    seeds = sorted({r["seed"] for r in records})
    L.append("# Pick-and-place batch evaluation\n")
    L.append(f"- Episodes per method: **{len(seeds)}** (seeds {seeds[0]}..{seeds[-1]}, "
             "identical for every method)")
    L.append(f"- Methods: {', '.join(methods)}")
    L.append(f"- Date: {config['started']}\n")

    L.append("## Metrics\n")
    L.append("| Metric | " + " | ".join(summary) + " |")
    L.append("|---|" + "---|" * len(summary))

    def row(label, key, spec=".3f", scale=1.0):
        cells = [fmt(summary[m][key] * scale, spec) if summary[m][key] == summary[m][key] else "n/a"
                 for m in summary]
        L.append(f"| {label} | " + " | ".join(cells) + " |")

    L.append("| Success rate | " + " | ".join(
        f"{summary[m]['successes']}/{summary[m]['episodes']} "
        f"({summary[m]['success_rate']*100:.0f}%)" for m in summary) + " |")
    L.append("| Success 95% CI (Wilson) | " + " | ".join(
        f"[{summary[m]['success_ci95'][0]*100:.0f}, {summary[m]['success_ci95'][1]*100:.0f}]%"
        for m in summary) + " |")
    L.append("| Mean IK time per episode (s) | " + " | ".join(
        f"{fmt(summary[m]['ik_time_per_episode_mean_s'])} +/- "
        f"{fmt(summary[m]['ik_time_per_episode_std_s'])}" for m in summary) + " |")
    row("Mean IK time per call (ms)", "ik_time_per_call_mean_s", ".1f", 1e3)
    L.append("| IK iterations to converge (mean +/- std, max) | " + " | ".join(
        f"{fmt(summary[m]['ik_iters_mean'], '.1f')} +/- {fmt(summary[m]['ik_iters_std'], '.1f')}, "
        f"{summary[m]['ik_iters_max']}" for m in summary) + " |")
    row("Position-only fallback calls (fraction)", "fallback_call_fraction", ".2f")
    L.append("| Min clearance, worst episode (mm) | " + " | ".join(
        fmt(summary[m]["min_clearance_min_m"] * 1e3, ".1f") for m in summary) + " |")
    row("Min clearance, mean over episodes (mm)", "min_clearance_mean_m", ".1f", 1e3)
    L.append("| Episodes with collision | " + " | ".join(
        str(summary[m]["collision_episodes"]) for m in summary) + " |")
    row("Mean final placement error, successes (mm)", "place_err_mean_m", ".1f", 1e3)

    L.append("\n## Constraint violations\n")
    L.append(f"Joint speed limit inside IK: {IK_VEL_LIMIT} rad/s; joint range from the model. "
             "A violation is an IK iteration whose step exceeds the speed limit or leaves "
             "the joint range.\n")
    L.append("| Quantity | " + " | ".join(summary) + " |")
    L.append("|---|" + "---|" * len(summary))
    L.append("| IK iterations over speed limit (episodes affected) | " + " | ".join(
        f"{summary[m]['ik_vel_viol_iters']} ({summary[m]['ik_vel_viol_episodes']})"
        for m in summary) + " |")
    L.append("| IK iterations outside joint range (episodes affected) | " + " | ".join(
        f"{summary[m]['ik_lim_viol_iters']} ({summary[m]['ik_lim_viol_episodes']})"
        for m in summary) + " |")
    row("Peak joint speed requested by IK (rad/s)", "ik_max_speed_rad_s", ".1f")
    L.append("| Simulation steps over speed limit | " + " | ".join(
        str(summary[m]["sim_vel_viol_steps"]) for m in summary) + " |")
    row("Peak simulated joint speed (rad/s)", "sim_max_joint_speed_rad_s", ".2f")
    row("Max simulated joint-limit excess (rad)", "sim_joint_limit_excess_max_rad", ".4f")

    L.append("\n## Failure analysis\n")
    any_fail = False
    for m in summary:
        reasons = summary[m]["failure_reasons"]
        if not reasons:
            L.append(f"**{m}**: no failures.\n")
            continue
        any_fail = True
        L.append(f"**{m}**\n")
        L.append("| Failure | Count | Seeds | Evidence |")
        L.append("|---|---|---|---|")
        for label, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            rs = [r for r in records if r["method"] == m and not r["success"]
                  and (f"{r['failure_reason']}@{r['failure_stage']}" if r["failure_stage"]
                       else r["failure_reason"]) == label]
            ev = ", ".join(f"`{r['screenshot'] or r['state_file']}`" for r in rs[:3])
            L.append(f"| {label} | {count} | {', '.join(str(r['seed']) for r in rs)} | {ev} |")
        L.append("")

    if any_fail:
        seen = {lbl.split("@")[0] for s in summary.values() for lbl in s["failure_reasons"]}
        L.append("### Suggested mitigations\n")
        for reason in MITIGATIONS:
            if reason in seen:
                L.append(f"- **{reason}**: {MITIGATIONS[reason]}")
        L.append("")

    L.append("## Per-seed outcomes (same seed = same cube pose)\n")
    L.append("| Seed | Cube x,y,yaw | " + " | ".join(summary) + " |")
    L.append("|---|---|" + "---|" * len(summary))
    by = {(r["method"], r["seed"]): r for r in records}
    for s in seeds:
        any_r = next(by[(m, s)] for m in summary if (m, s) in by)
        cells = []
        for m in summary:
            r = by.get((m, s))
            if r is None:
                cells.append("-")
            elif r["success"]:
                cells.append("ok")
            else:
                cells.append("FAIL " + r["failure_reason"] +
                             (f"@{r['failure_stage']}" if r["failure_stage"] else ""))
        L.append(f"| {s} | {any_r['cube_x']:.2f}, {any_r['cube_y']:.2f}, "
                 f"{any_r['cube_yaw_deg']:.0f} | " + " | ".join(cells) + " |")

    if len(summary) > 1:
        L.append("\n## Paired comparison (same seeds)\n")
        names = list(summary)
        L.append("| A vs B | both ok | only A | only B | both fail |")
        L.append("|---|---|---|---|---|")
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                both = onlya = onlyb = neither = 0
                for s in seeds:
                    ra, rb = by.get((a, s)), by.get((b, s))
                    if ra is None or rb is None:
                        continue
                    if ra["success"] and rb["success"]:
                        both += 1
                    elif ra["success"]:
                        onlya += 1
                    elif rb["success"]:
                        onlyb += 1
                    else:
                        neither += 1
                L.append(f"| {a} vs {b} | {both} | {onlya} | {onlyb} | {neither} |")

    return "\n".join(L) + "\n"


CSV_SKIP = {"ik_calls_detail", "checkpoints"}


def write_csv(path, records):
    if not records:
        return
    fields = [k for k in records[0] if k not in CSV_SKIP]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in fields})


def load_records(path):
    if not path.exists():
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=25, help="episodes per method")
    ap.add_argument("--seed0", type=int, default=0, help="first seed")
    ap.add_argument("--methods", nargs="+", default=["mink"], choices=METHODS)
    ap.add_argument("--solver", default="daqp", help="QP solver for mink / qp")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--resume", action="store_true",
                    help="continue an existing --out directory, skipping finished seeds")
    ap.add_argument("--screenshots", action="store_true",
                    help="save a PNG of every failure (needs offscreen GL)")
    ap.add_argument("--viewer", action="store_true", help="show the MuJoCo viewer")
    ap.add_argument("--speed", type=float, default=1.0, help="viewer speed factor")
    args = ap.parse_args()

    project = Path(__file__).resolve().parent.parent
    if args.out:
        out_dir = Path(args.out)
        if not out_dir.is_absolute():
            out_dir = Path.cwd() / out_dir
    else:
        out_dir = project / "results" / f"batch_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl = out_dir / "episodes.jsonl"
    if jsonl.exists() and not args.resume:
        sys.exit(f"{jsonl} already exists. Use --resume to continue it, "
                 "or choose a new --out directory.")

    records = load_records(jsonl) if args.resume else []
    done = {(r["method"], r["seed"]) for r in records}

    seeds = list(range(args.seed0, args.seed0 + args.episodes))

    config = {
        "started": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        "command": " ".join(sys.argv),
        "methods": args.methods,
        "seeds": seeds,
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "ik": {"dt": DT_IK, "max_steps": IK_MAX_STEPS, "pos_tol": IK_POS_TOL,
               "rot_tol": IK_ROT_TOL, "orientation_cost": ORIENTATION_COST,
               "vel_limit_rad_s": IK_VEL_LIMIT},
        "task": {"approach_h": APPROACH_HEIGHT, "lift_h": LIFT_HEIGHT,
                 "transit_h": TRANSIT_HEIGHT, "grasp_z_offset": GRASP_Z_OFFSET,
                 "carry_speed": CARRY_SPEED, "track_tol": TRACK_TOL},
        "eval": {"collision_tol_m": COLLISION_TOL, "slip_jump_m": SLIP_JUMP,
                 "clearance_every_steps": CLEARANCE_EVERY, "sim_vel_limit": SIM_VEL_LIMIT},
    }

    scene = MonitoredScene(with_gripper=True, seed=args.seed0)
    config["clearance_geoms"] = {
        "arm": len(scene.arm_geoms),
        "ignored_structural": [
            mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
            for g in scene.ignored_geoms],
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"Output directory: {out_dir}")
    print(f"Methods: {args.methods} | seeds {seeds[0]}..{seeds[-1]} "
          f"({len(seeds)} each) | already done: {len(done)}")
    if scene.ignored_geoms:
        print(f"Clearance: ignoring {len(scene.ignored_geoms)} geom(s) that already "
              "touch the table/tray at the home pose")

    runners = {}
    for m in args.methods:
        backend = make_backend(m, scene, solver=args.solver)
        runners[m] = EpisodeRunner(scene, m, IKPlanner(backend), out_dir, args.screenshots)

    def run_all(viewer=None):
        t_start = time.perf_counter()
        total = len(seeds) * len(args.methods)
        count = 0
        for seed in seeds:
            for m in args.methods:
                count += 1
                if (m, seed) in done:
                    continue
                rec = runners[m].run(seed)
                records.append(rec)
                with open(jsonl, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")

                tag = "SUCCESS" if rec["success"] else (
                    f"FAIL {rec['failure_reason']}"
                    + (f"@{rec['failure_stage']}" if rec["failure_stage"] else ""))
                elapsed = time.perf_counter() - t_start
                print(f"[{count:3d}/{total}] {m:8s} seed {seed:3d}  {tag:34s} "
                      f"ik={rec['ik_time_total_s']:6.2f}s it={rec['ik_iters_mean']:6.1f} "
                      f"clr={rec['min_clearance_m']*1e3:6.1f}mm  ({elapsed/60:.1f} min)",
                      flush=True)

    if args.viewer:
        import mujoco.viewer as mjv
        with mjv.launch_passive(scene.model, scene.data) as viewer:
            viewer.cam.lookat[:] = [0.30, 0.0, 0.15]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.8, 135, -25
            scene.viewer = viewer
            scene.realtime_factor = args.speed
            try:
                run_all()
            except SystemExit:
                pass
    else:
        run_all()

    # final reports from everything on disk (includes resumed episodes)
    records = load_records(jsonl)
    records.sort(key=lambda r: (r["seed"], args.methods.index(r["method"])
                                if r["method"] in args.methods else 99))
    methods_present = [m for m in args.methods if any(r["method"] == m for r in records)]
    summary = summarize(records, methods_present)

    write_csv(out_dir / "episodes.csv", records)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    md = build_markdown(records, summary, methods_present, config)
    (out_dir / "summary.md").write_text(md)

    print("\n" + md)
    print(f"Saved: {out_dir}/summary.md, summary.json, episodes.csv, episodes.jsonl")


if __name__ == "__main__":
    main()