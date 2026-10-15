#!/usr/bin/env python3
"""HEAL + Robotiq 2F-85 pick and place with mink IK."""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

try:
    import mink
except ImportError as exc:
    raise SystemExit(
        "Install dependencies with:\n"
        "python3 -m pip install mink daqp"
    ) from exc

from envsetup import (
    PickPlaceScene,
    CUBE_SIZE,
    TABLE_TOP_Z,
    TRAY_CENTER,
    TRAY_FLOOR,
    GRIPPER_OPEN,
    GRIPPER_CLOSE,
    MAX_JOINT_SPEED,
)


# ============================================================
# CONFIGURATION
# ============================================================

DT_IK = 0.02
IK_MAX_STEPS = 300
IK_POS_TOL = 0.006
IK_ROT_TOL = 0.15

ORIENTATION_COST = 0.2          # used for the oriented IK attempt

APPROACH_HEIGHT = 0.12
LIFT_HEIGHT = 0.16
TRANSIT_HEIGHT = 0.18

DESCENT_SUBSTEPS = 4            # linear interpolation for descend / lift
GRIP_STEPS = 160
SETTLE_STEPS = 100

GRASP_Z_OFFSET = 0.012          # grasp this far above the cube center
PLACE_CLEARANCE = 0.004         # cube bottom clearance above tray floor
CARRY_SPEED = 0.5               # joint speed limit while holding the cube

VERBOSE = False


# ============================================================
# TARGET POSE
# ============================================================

def target_pose(position, yaw):
    """Gripper pointing straight down, rotated by `yaw` about world z."""
    position = np.asarray(position, dtype=float)

    rotation_matrix = (
        Rotation.from_euler("z", yaw).as_matrix()
        @ Rotation.from_euler("x", np.pi).as_matrix()
    )

    return (
        mink.SE3.from_translation(position)
        @ mink.SE3.from_rotation(mink.SO3.from_matrix(rotation_matrix))
    )


# ============================================================
# MINK INVERSE KINEMATICS
# ============================================================

class MinkPickPlace:

    def __init__(self, scene, solver="daqp"):
        self.scene = scene
        self.model = scene.model
        self.solver = solver
        self.last_mode = "none"

        self.cfg = mink.Configuration(self.model)

        # Oriented task and position-only task (fallback).
        self.task_oriented = mink.FrameTask(
            frame_name="gripper/pinch",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=ORIENTATION_COST,
            lm_damping=1e-4,
            gain=0.5,
        )
        self.task_position = mink.FrameTask(
            frame_name="gripper/pinch",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=1e-4,
            gain=0.5,
        )

        # Freeze every non-arm DOF (cube, gripper fingers).
        arm_dofs = set(map(int, scene.arm_dadr))
        frozen = [i for i in range(self.model.nv) if i not in arm_dofs]
        self.freeze = mink.DofFreezingTask(
            model=self.model, dof_indices=frozen
        )

        self.posture = mink.PostureTask(self.model, cost=1e-3)
        self.limits = [mink.ConfigurationLimit(self.model)]

    # --------------------------------------------------------

    def _run_ik(self, task, position, yaw, check_rotation):
        self.cfg.update(self.scene.data.qpos)
        self.posture.set_target_from_configuration(self.cfg)
        task.set_target(target_pose(position, yaw))

        for _ in range(IK_MAX_STEPS):
            err = task.compute_error(self.cfg)
            pos_ok = np.linalg.norm(err[:3]) < IK_POS_TOL
            rot_ok = (not check_rotation) or (
                np.linalg.norm(err[3:]) < IK_ROT_TOL
            )
            if pos_ok and rot_ok:
                return self.cfg.q[self.scene.arm_qadr].copy()

            vel = mink.solve_ik(
                self.cfg,
                [task, self.posture],
                DT_IK,
                self.solver,
                damping=1e-3,
                limits=self.limits,
                constraints=[self.freeze],
            )
            self.cfg.integrate_inplace(vel, DT_IK)

        return None

    def solve(self, position, yaw):
        q_now = self.scene.data.qpos[self.scene.arm_qadr].copy()

        # 1. Oriented IK. A square cube with a parallel gripper can be
        #    grasped at yaw + k*90 deg, so try all four and keep the
        #    solution needing the least joint motion.
        best_q, best_cost = None, np.inf
        for k in (0, 1, -1, 2):
            q = self._run_ik(
                self.task_oriented,
                position,
                yaw + k * np.pi / 2,
                check_rotation=True,
            )
            if q is not None:
                cost = np.sum(np.abs(q - q_now))
                if cost < best_cost:
                    best_q, best_cost = q, cost
        if best_q is not None:
            self.last_mode = "oriented"
            return best_q

        # 2. Fallback: position-only IK.
        self.last_mode = "FALLBACK(position-only)"
        return self._run_ik(
            self.task_position, position, yaw, check_rotation=False
        )

    # --------------------------------------------------------

    def move(self, position, yaw):
        q_solution = self.solve(position, yaw)

        if q_solution is None:
            print("IK failed for target:", np.round(position, 3))
            return False

        self.scene.q_target = q_solution

        for _ in range(80):                 # up to 4000 steps
            self.scene.step(50)
            q_err = np.max(np.abs(
                self.scene.q_target
                - self.scene.data.qpos[self.scene.arm_qadr]
            ))
            if q_err < 0.01:
                break

        error = np.linalg.norm(
            self.scene.tcp_pos() - np.asarray(position)
        )

        if VERBOSE:
            print(
                "Target:", np.round(position, 3),
                "| TCP:", np.round(self.scene.tcp_pos(), 3),
                "| Error:", round(error, 4),
                "|", self.last_mode,
            )

        return error < 0.04

    def move_linear(self, p_start, p_end, yaw, n=DESCENT_SUBSTEPS):
        """Short Cartesian steps keep IK on one joint-space branch."""
        p_start = np.asarray(p_start, dtype=float)
        p_end = np.asarray(p_end, dtype=float)
        for a in np.linspace(0.0, 1.0, n + 1)[1:]:
            if not self.move((1 - a) * p_start + a * p_end, yaw):
                return False
        return True


# ============================================================
# PICK AND PLACE EPISODE
# ============================================================

def report_held(scene, label):
    d = scene.cube_pos() - scene.tcp_pos()
    print(f"  [{label}] cube-tcp = {np.round(d, 3)} "
          f"| dist = {np.linalg.norm(d):.3f}")


def run_episode(scene, controller, seed):
    cube_pose = scene.reset(seed=seed)
    x, y, cube_yaw = cube_pose

    grasp_position = np.array(
        [x, y, TABLE_TOP_Z + CUBE_SIZE / 2 + GRASP_Z_OFFSET]
    )
    approach_position = grasp_position + np.array([0, 0, APPROACH_HEIGHT])
    lift_position = grasp_position + np.array([0, 0, LIFT_HEIGHT])

    scene.step(200)     # let the cube settle

    print(f"\n--- Episode seed {seed} | cube at "
          f"{np.round(scene.cube_pos(), 3)} ---")

    # 1. Approach above the cube.
    if not controller.move(approach_position, cube_yaw):
        print("FAILED: approach")
        return False
    print("  IK mode (approach):", controller.last_mode)

    # 2. Descend in small Cartesian steps.
    if not controller.move_linear(approach_position, grasp_position, cube_yaw):
        print("FAILED: grasp positioning")
        return False
    print("  IK mode (descent):", controller.last_mode)
    scene.step(SETTLE_STEPS)

    # 3. Close the gripper.
    scene.grip_cmd = GRIPPER_CLOSE
    scene.step(GRIP_STEPS)
    scene.max_speed = CARRY_SPEED          # slow while carrying
    report_held(scene, "grasped")

    # Where the cube sits relative to the TCP while held; used so the
    # CUBE (not the TCP) ends up at the tray center.
    held_offset = scene.cube_pos() - scene.tcp_pos()

    cube_place_center = np.array([
        TRAY_CENTER[0],
        TRAY_CENTER[1],
        TABLE_TOP_Z + TRAY_FLOOR + CUBE_SIZE / 2 + PLACE_CLEARANCE,
    ])
    place_position = cube_place_center - held_offset
    pre_place_position = place_position + np.array([0, 0, TRANSIT_HEIGHT])

    # 4. Lift.
    if not controller.move_linear(grasp_position, lift_position, cube_yaw):
        print("FAILED: lift")
        return False
    report_held(scene, "lifted")

    # 5. Transit above the tray.
    if not controller.move(pre_place_position, 0.0):
        print("FAILED: transit")
        return False
    report_held(scene, "transit")

    # 6. Lower into the tray.
    if not controller.move_linear(pre_place_position, place_position, 0.0):
        print("FAILED: place")
        return False
    scene.step(SETTLE_STEPS)
    report_held(scene, "placed")

    # 7. Release.
    scene.grip_cmd = GRIPPER_OPEN
    scene.max_speed = MAX_JOINT_SPEED
    scene.step(GRIP_STEPS)
    scene.step(250)

    success = scene.cube_in_tray()
    print("Final cube position:", np.round(scene.cube_pos(), 3))
    print("Episode result:", "SUCCESS" if success else "NOT SUCCESSFUL")
    return success


# ============================================================
# MAIN
# ============================================================

def main():
    global VERBOSE

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--solver", default="daqp")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="viewer playback speed (1.0 = real time, 2.0 = twice as fast)",
    )
    args = parser.parse_args()

    VERBOSE = args.verbose

    scene = PickPlaceScene(with_gripper=True, seed=args.seed)
    print("HEAL + Robotiq model loaded.")
    print("Bodies:", scene.model.nbody,
          "| Joints:", scene.model.njnt,
          "| Actuators:", scene.model.nu)

    controller = MinkPickPlace(scene, solver=args.solver)

    if args.viewer:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(
            scene.model, scene.data
        ) as viewer:
            viewer.cam.lookat[:] = [0.30, 0.0, 0.15]
            viewer.cam.distance = 1.8
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25

            # step() now syncs the viewer and paces to real time.
            scene.viewer = viewer
            scene.realtime_factor = args.speed

            for episode in range(args.episodes):
                run_episode(scene, controller, args.seed + episode)

                if episode + 1 < args.episodes:
                    scene.reset(seed=args.seed + episode + 1)
                    time.sleep(0.5)

            # Keep the window open until it is closed.
            while viewer.is_running():
                scene.step(5)
    else:
        successes = 0
        for episode in range(args.episodes):
            successes += int(
                run_episode(scene, controller, args.seed + episode)
            )
        print(f"\nSummary: {successes}/{args.episodes} successful episodes")


if __name__ == "__main__":
    main()