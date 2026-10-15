#!/usr/bin/env python3
"""MuJoCo scene setup and control for Franka Panda."""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DESC_DIR = PROJECT_DIR / "robot_descriptions"

SCENE_XML = (
    DESC_DIR
    / "franka"
    / "pickandplace_scene.xml"
)
ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
FINGER_JOINT_NAMES = ("finger_joint1", "finger_joint2")

EE_SITE_CANDIDATES = ("gripper",)
CUBE_BODY_CANDIDATES = ("box", "cube")
TRAY_SITE_CANDIDATES = ("place_target", "plate_center", "target")

CUBE_SIZE = 0.05

# Panda finger positions are normally specified in metres.
GRIP_OPEN = 0.04
GRIP_CLOSE = 0.0

# Initial values for joint-space PD torque control.
ARM_KP = np.array([500., 500., 400., 300., 180., 120., 100.])
ARM_KD = np.array([45., 45., 35., 25., 16., 12., 10.])

# Conservative torque limits for simulation.
ARM_TORQUE_LIMIT = np.array([87., 87., 87., 87., 12., 12., 12.])


def object_id(model, obj_type, name):
    return mujoco.mj_name2id(model, obj_type, name)


def object_name(model, obj_type, idx):
    if idx < 0:
        return None
    return mujoco.mj_id2name(model, obj_type, idx)


def find_first(model, obj_type, candidates):
    for name in candidates:
        idx = object_id(model, obj_type, name)
        if idx >= 0:
            return idx, name
    return -1, None


class FrankaScene:

    def __init__(
        self,
        seed=0,
        scene_xml=SCENE_XML,
        viewer=None,
        realtime_factor=1.0,
    ):
        self.scene_xml = Path(scene_xml).expanduser().resolve()

        if not self.scene_xml.exists():
            raise FileNotFoundError(
                f"Scene XML not found: {self.scene_xml}"
            )

        self.model = mujoco.MjModel.from_xml_path(str(self.scene_xml))
        self.data = mujoco.MjData(self.model)

        self.rng = np.random.default_rng(seed)
        self.viewer = viewer
        self.realtime_factor = max(float(realtime_factor), 0.01)

        self._step_count = 0
        self.sync_every = 5

        # ----------------------------------------------------
        # Arm joints
        # ----------------------------------------------------

        self.arm_jids = np.array([
            self.required_joint(name)
            for name in ARM_JOINT_NAMES
        ])

        self.arm_qadr = self.model.jnt_qposadr[self.arm_jids].copy()
        self.arm_dadr = self.model.jnt_dofadr[self.arm_jids].copy()

        # ----------------------------------------------------
        # Finger joints
        # ----------------------------------------------------

        self.finger_jids = np.array([
            self.required_joint(name)
            for name in FINGER_JOINT_NAMES
        ])

        self.finger_qadr = self.model.jnt_qposadr[
            self.finger_jids
        ].copy()

        self.finger_dadr = self.model.jnt_dofadr[
            self.finger_jids
        ].copy()

        # ----------------------------------------------------
        # End effector
        # ----------------------------------------------------

        (
            self.ee_site_id,
            self.ee_site_name,
        ) = find_first(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            EE_SITE_CANDIDATES,
        )

        if self.ee_site_id < 0:
            raise RuntimeError(
                f"No end-effector site found: {EE_SITE_CANDIDATES}"
            )

        # ----------------------------------------------------
        # Cube
        # ----------------------------------------------------

        (
            self.cube_body_id,
            self.cube_body_name,
        ) = find_first(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            CUBE_BODY_CANDIDATES,
        )

        if self.cube_body_id < 0:
            raise RuntimeError(
                f"No cube body found: {CUBE_BODY_CANDIDATES}"
            )

        self.cube_joint_id = self.find_free_joint(
            self.cube_body_id
        )

        if self.cube_joint_id < 0:
            raise RuntimeError(
                f"Cube body '{self.cube_body_name}' "
                "must have a free joint."
            )

        self.cube_qadr = int(
            self.model.jnt_qposadr[self.cube_joint_id]
        )

        self.cube_dadr = int(
            self.model.jnt_dofadr[self.cube_joint_id]
        )

        # ----------------------------------------------------
        # Tray or placement target
        # ----------------------------------------------------

        (
            self.tray_site_id,
            self.tray_site_name,
        ) = find_first(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            TRAY_SITE_CANDIDATES,
        )

        # ----------------------------------------------------
        # Actuators
        # ----------------------------------------------------

        self.arm_actuator_ids = self.actuators_for_joints(
            self.arm_jids
        )

        self.finger_actuator_ids = self.actuators_for_joints(
            self.finger_jids
        )

        self.position_actuator_ids = self.find_arm_position_actuators()

        self.use_position_actuators = (
            len(self.position_actuator_ids) == 7
        )

        # A common Panda configuration. This must respect joint limits.
        self.home_q = np.array([
            0.0, -0.45, 0.0, -2.0, 0.0, 1.65, 0.75
        ])

        self.q_target = self.home_q.copy()
        self.grip_cmd = GRIP_OPEN

        self.reset(seed=seed)

    # ========================================================
    # MODEL HELPERS
    # ========================================================

    def required_joint(self, name):
        jid = object_id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if jid < 0:
            raise RuntimeError(
                f"Required joint '{name}' not found."
            )

        return jid

    def find_free_joint(self, body_id):
        for jid in range(self.model.njnt):
            if (
                self.model.jnt_bodyid[jid] == body_id
                and self.model.jnt_type[jid]
                == mujoco.mjtJoint.mjJNT_FREE
            ):
                return jid

        return -1

    def actuators_for_joints(self, joint_ids):
        wanted = set(map(int, joint_ids))
        result = []

        for aid in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[aid, 0])

            if joint_id in wanted:
                result.append(aid)

        return np.asarray(result, dtype=int)

    def find_arm_position_actuators(self):
        """Find position actuators named actuator1 through actuator7."""

        ids = []

        for i in range(1, 8):
            aid = object_id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"actuator{i}",
            )

            if aid < 0:
                return np.array([], dtype=int)

            ids.append(aid)

        return np.asarray(ids, dtype=int)

    # ========================================================
    # RESET
    # ========================================================

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[self.arm_qadr] = self.home_q
        self.q_target = self.home_q.copy()

        self.grip_cmd = GRIP_OPEN

        # Randomize cube position in a restricted region.
        x = self.rng.uniform(0.38, 0.52)
        y = self.rng.uniform(-0.28, -0.12)

        # The supplied scene places the robot base around z = 0.83.
        # This initial height assumes a table near z = 0.83.
        z = 0.83 + CUBE_SIZE / 2

        self.data.qpos[
            self.cube_qadr:self.cube_qadr + 3
        ] = [x, y, z]

        # Free-joint quaternion: w, x, y, z.
        self.data.qpos[
            self.cube_qadr + 3:self.cube_qadr + 7
        ] = [1, 0, 0, 0]

        self.data.qvel[
            self.cube_dadr:self.cube_dadr + 6
        ] = 0

        mujoco.mj_forward(self.model, self.data)

        return self.cube_pos().copy()

    # ========================================================
    # POSITIONS
    # ========================================================

    def tcp_pos(self):
        return self.data.site_xpos[self.ee_site_id].copy()

    def tcp_rot(self):
        return (
            self.data.site_xmat[self.ee_site_id]
            .reshape(3, 3)
            .copy()
        )

    def cube_pos(self):
        return self.data.xpos[self.cube_body_id].copy()

    def tray_pos(self):
        if self.tray_site_id >= 0:
            return self.data.site_xpos[
                self.tray_site_id
            ].copy()

        for name in ("plate", "plate_body", "tray"):
            bid = object_id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                name,
            )

            if bid >= 0:
                return self.data.xpos[bid].copy()

        raise RuntimeError(
            "No placement site or plate body was found. "
            "Add a site named 'place_target' to your scene XML."
        )

    # ========================================================
    # ARM CONTROL
    # ========================================================

    def apply_arm_control(self):
        q = self.data.qpos[self.arm_qadr]
        qd = self.data.qvel[self.arm_dadr]

        if self.use_position_actuators:
            for i, aid in enumerate(self.position_actuator_ids):
                lo, hi = self.model.actuator_ctrlrange[aid]

                self.data.ctrl[aid] = np.clip(
                    self.q_target[i],
                    lo,
                    hi,
                )

            return

        # Fallback for a model without active position actuators.
        # This assumes direct torque control is appropriate for the model.
        bias = self.data.qfrc_bias[self.arm_dadr]

        tau = (
            bias
            + ARM_KP * (self.q_target - q)
            - ARM_KD * qd
        )

        tau = np.clip(
            tau,
            -ARM_TORQUE_LIMIT,
            ARM_TORQUE_LIMIT,
        )

        self.data.qfrc_applied[self.arm_dadr] = tau

    # ========================================================
    # GRIPPER CONTROL
    # ========================================================

    def apply_gripper_control(self):
        if len(self.finger_actuator_ids) == 0:
            raise RuntimeError(
                "No actuators were found for the Panda finger joints."
            )

        for aid in self.finger_actuator_ids:
            lo, hi = self.model.actuator_ctrlrange[aid]

            # Standard Panda models usually use metres.
            # Some models instead use a 0..255 command scale.
            if hi > 2.0:
                ctrl = 255.0 if self.grip_cmd >= GRIP_OPEN else 0.0
            else:
                ctrl = self.grip_cmd

            self.data.ctrl[aid] = np.clip(ctrl, lo, hi)

    def apply_control(self):
        self.data.qfrc_applied[self.arm_dadr] = 0.0

        self.apply_arm_control()
        self.apply_gripper_control()

    # ========================================================
    # SIMULATION
    # ========================================================

    def step(self, n=1):
        for _ in range(n):
            self.apply_control()

            mujoco.mj_step(self.model, self.data)

            if self.viewer is not None:
                self._step_count += 1

                if self._step_count % self.sync_every == 0:
                    if not self.viewer.is_running():
                        raise SystemExit

                    self.viewer.sync()

                    time.sleep(
                        self.model.opt.timestep
                        * self.sync_every
                        / self.realtime_factor
                    )

    def move_joints(
        self,
        target,
        steps=800,
        tolerance=0.025,
    ):
        self.q_target = np.asarray(
            target,
            dtype=float,
        ).copy()

        for _ in range(steps // 5 + 1):
            self.step(5)

            error = np.max(
                np.abs(
                    self.data.qpos[self.arm_qadr]
                    - self.q_target
                )
            )

            if error < tolerance:
                break

    def open_gripper(self):
        self.grip_cmd = GRIP_OPEN

    def close_gripper(self):
        self.grip_cmd = GRIP_CLOSE

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    def print_model_summary(self):
        print("Scene:", self.scene_xml)
        print(
            f"Bodies={self.model.nbody}, "
            f"joints={self.model.njnt}, "
            f"actuators={self.model.nu}"
        )

        print(
            "Arm control:",
            "position actuators"
            if self.use_position_actuators
            else "PD torque fallback",
        )

        print("End-effector site:", self.ee_site_name)
        print("Cube body:", self.cube_body_name)

        actuator_names = [
            object_name(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                i,
            )
            for i in range(self.model.nu)
        ]

        print("Actuators:", actuator_names)
        print("Finger actuators:", self.finger_actuator_ids)