#!/usr/bin/env python3

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DESC_DIR = PROJECT_DIR / "robot_descriptions"

ARM_XML = DESC_DIR / "single_arm_heal_effort_actuation_velocity.xml"
GRIPPER_XML = DESC_DIR / "robotiq_2f85_v4" / "2f85.xml"
HEAL_MESH_DIR = DESC_DIR / "heal_meshes"


# ============================================================
# SCENE CONFIGURATION
# ============================================================

TABLE_TOP_Z = 0.0
FLOOR_Z = -0.75

CUBE_SIZE = 0.05
CUBE_MASS = 0.05
CUBE_FRICTION = (1.5, 0.02, 0.002)

SPAWN_X = (0.28, 0.58)
SPAWN_Y = (0.08, 0.34)
SPAWN_YAW = (-np.pi / 4, np.pi / 4)

TRAY_CENTER = (0.43, -0.22)
TRAY_INNER = 0.16
TRAY_WALL = 0.01
TRAY_WALL_H = 0.03
TRAY_FLOOR = 0.004

GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 255.0

# HEAL uses velocity actuators.
VELOCITY_KP = 3.0
MAX_JOINT_SPEED = 1.0

ARM_JOINT_NAMES = tuple(f"joint_{i}" for i in range(1, 7))

ARM_ACTUATOR_NAMES = (
    "turret",
    "shoulder",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)


# ============================================================
# XML HELPERS
# ============================================================

def add_element(parent, tag, **attributes):
    element = ET.SubElement(parent, tag)

    for key, value in attributes.items():
        if isinstance(value, (tuple, list, np.ndarray)):
            value = " ".join(str(float(x)) for x in value)

        element.set(key.rstrip("_"), str(value))

    return element


# ============================================================
# BUILD HEAL + ROBOTIQ MODEL
# ============================================================

def build_model(with_gripper=True, **kwargs):

    if not ARM_XML.exists():
        raise FileNotFoundError(f"HEAL XML not found: {ARM_XML}")

    if not GRIPPER_XML.exists():
        raise FileNotFoundError(f"Robotiq XML not found: {GRIPPER_XML}")

    if not HEAL_MESH_DIR.exists():
        raise FileNotFoundError(
            f"HEAL mesh directory not found: {HEAL_MESH_DIR}"
        )

    if not with_gripper:
        raise ValueError("This configuration expects the Robotiq gripper.")

    # Load HEAL XML.
    root = ET.parse(ARM_XML).getroot()

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")

    # Use an absolute mesh directory.
    compiler.set("meshdir", str(HEAL_MESH_DIR.resolve()))

    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("HEAL XML does not contain a worldbody.")

    # Verify the HEAL attachment site.
    attachment_site = root.find(".//site[@name='right_center']")
    if attachment_site is None:
        raise RuntimeError("Could not find HEAL site 'right_center'.")

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    add_element(
        world,
        "geom",
        name="lab_table_top",
        type="box",
        pos=[0.20, 0.0, TABLE_TOP_Z - 0.02],
        size=[0.50, 0.50, 0.02],
        rgba=[0.68, 0.53, 0.36, 1],
        friction=[1, 0.005, 0.0001],
    )

    leg_height = (TABLE_TOP_Z - 0.04 - FLOOR_Z) / 2

    leg_positions = [
        (-0.25, -0.45),
        (-0.25, 0.45),
        (0.65, -0.45),
        (0.65, 0.45),
    ]

    for i, (x, y) in enumerate(leg_positions):
        leg_body = add_element(
            world,
            "body",
            name=f"table_leg_body_{i}",
            pos=[x, y, FLOOR_Z + leg_height],
        )

        add_element(
            leg_body,
            "geom",
            name=f"table_leg_{i}",
            type="box",
            pos=[0, 0, 0],
            size=[0.02, 0.02, leg_height],
            rgba=[0.35, 0.35, 0.38, 1],
        )

    # --------------------------------------------------------
    # TRAY
    # --------------------------------------------------------

    tray = add_element(
        world,
        "body",
        name="tray",
        pos=[TRAY_CENTER[0], TRAY_CENTER[1], TABLE_TOP_Z],
    )

    half_out = TRAY_INNER / 2 + TRAY_WALL

    add_element(
        tray,
        "geom",
        name="tray_floor",
        type="box",
        pos=[0, 0, TRAY_FLOOR / 2],
        size=[half_out, half_out, TRAY_FLOOR / 2],
        rgba=[0.25, 0.55, 0.85, 1],
        friction=[1, 0.005, 0.0001],
    )

    wall_z = TRAY_FLOOR + TRAY_WALL_H / 2
    wall_offset = TRAY_INNER / 2 + TRAY_WALL / 2

    walls = [
        (
            "tray_wall_xp",
            [wall_offset, 0, wall_z],
            [TRAY_WALL / 2, half_out, TRAY_WALL_H / 2],
        ),
        (
            "tray_wall_xn",
            [-wall_offset, 0, wall_z],
            [TRAY_WALL / 2, half_out, TRAY_WALL_H / 2],
        ),
        (
            "tray_wall_yp",
            [0, wall_offset, wall_z],
            [TRAY_INNER / 2, TRAY_WALL / 2, TRAY_WALL_H / 2],
        ),
        (
            "tray_wall_yn",
            [0, -wall_offset, wall_z],
            [TRAY_INNER / 2, TRAY_WALL / 2, TRAY_WALL_H / 2],
        ),
    ]

    for name, pos, size in walls:
        add_element(
            tray,
            "geom",
            name=name,
            type="box",
            pos=pos,
            size=size,
            rgba=[0.25, 0.55, 0.85, 1],
        )

    add_element(
        tray,
        "site",
        name="place_target",
        pos=[0, 0, TRAY_FLOOR + CUBE_SIZE / 2],
        size=0.008,
        rgba=[0, 1, 0, 0.8],
    )

    # --------------------------------------------------------
    # PICK CUBE
    # --------------------------------------------------------

    cube = add_element(
        world,
        "body",
        name="cube",
        pos=[
            sum(SPAWN_X) / 2,
            sum(SPAWN_Y) / 2,
            TABLE_TOP_Z + CUBE_SIZE / 2 + 0.0005,
        ],
    )

    # Free joint allows the cube to move in 3D.
    add_element(cube, "freejoint", name="cube_free")

    # Explicit mass and diagonal inertia for a uniform cube.
    cube_inertia = CUBE_MASS * CUBE_SIZE**2 / 6.0

    add_element(
        cube,
        "inertial",
        pos=[0, 0, 0],
        mass=CUBE_MASS,
        diaginertia=[cube_inertia, cube_inertia, cube_inertia],
    )

    # Collision and visual geometry.
    add_element(
        cube,
        "geom",
        name="cube_geom",
        type="box",
        size=[CUBE_SIZE / 2] * 3,
        mass=0,
        rgba=[0.9, 0.12, 0.12, 1],
        friction=CUBE_FRICTION,
        condim=4,
    )

    # --------------------------------------------------------
    # ATTACH ROBOTIQ GRIPPER TO HEAL
    # --------------------------------------------------------

    arm_spec = mujoco.MjSpec.from_string(
        ET.tostring(root, encoding="unicode")
    )

    gripper_spec = mujoco.MjSpec.from_file(str(GRIPPER_XML))

    arm_spec.attach(
        gripper_spec,
        prefix="gripper/",
        site=arm_spec.site("right_center"),
    )

    return arm_spec.compile()


# ============================================================
# PICK AND PLACE SCENE
# ============================================================

class PickPlaceScene:

    def __init__(self, with_gripper=True, seed=0, **kwargs):

        self.model = build_model(with_gripper=with_gripper)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        # HEAL joint addresses.
        self.arm_qadr = np.array([
            self.model.jnt_qposadr[self.model.joint(name).id]
            for name in ARM_JOINT_NAMES
        ], dtype=int)

        self.arm_dadr = np.array([
            self.model.jnt_dofadr[self.model.joint(name).id]
            for name in ARM_JOINT_NAMES
        ], dtype=int)

        self.arm_act = np.array([
            self.model.actuator(name).id
            for name in ARM_ACTUATOR_NAMES
        ], dtype=int)

        arm_actuator_set = set(self.arm_act.tolist())

        self.grip_act = np.array([
            actuator_id
            for actuator_id in range(self.model.nu)
            if actuator_id not in arm_actuator_set
        ], dtype=int)

        # Object and target identifiers.
        self.cube_body = self.model.body("cube").id

        cube_joint = self.model.joint("cube_free").id
        self.cube_qadr = int(self.model.jnt_qposadr[cube_joint])
        self.cube_dadr = int(self.model.jnt_dofadr[cube_joint])

        self.place_site = self.model.site("place_target").id

        pinch_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "gripper/pinch",
        )

        if pinch_id < 0:
            raise RuntimeError(
                "The Robotiq model does not contain "
                "the expected site 'pinch'."
            )

        self.tcp_site = pinch_id

        self.home_q = np.array([0.0, -0.8, 1.2, -0.4, 0.0, 0.0])

        self.q_target = self.home_q.copy()
        self.grip_cmd = GRIPPER_OPEN

        # Viewer / real-time pacing (viewer stays None when headless).
        self.viewer = None
        self.realtime_factor = 1.0      # 2.0 = twice real time
        self.sync_every = 10            # sync viewer every N sim steps
        self._step_count = 0

        self.max_speed = MAX_JOINT_SPEED

        self.reset(seed=seed)

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(self, seed=None, randomize_cube=True):

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[self.arm_qadr] = self.home_q

        self.q_target = self.home_q.copy()
        self.grip_cmd = GRIPPER_OPEN
        self.max_speed = MAX_JOINT_SPEED

        pose = self.randomize_cube() if randomize_cube else None

        mujoco.mj_forward(self.model, self.data)

        return pose

    # --------------------------------------------------------
    # CUBE
    # --------------------------------------------------------

    def randomize_cube(self, rng=None):

        rng = self.rng if rng is None else rng

        x = rng.uniform(*SPAWN_X)
        y = rng.uniform(*SPAWN_Y)
        yaw = rng.uniform(*SPAWN_YAW)

        qadr = self.cube_qadr

        self.data.qpos[qadr:qadr + 3] = [
            x,
            y,
            TABLE_TOP_Z + CUBE_SIZE / 2 + 0.0005,
        ]

        self.data.qpos[qadr + 3:qadr + 7] = [
            np.cos(yaw / 2),
            0,
            0,
            np.sin(yaw / 2),
        ]

        self.data.qvel[self.cube_dadr:self.cube_dadr + 6] = 0

        mujoco.mj_forward(self.model, self.data)

        return np.array([x, y, yaw])

    # --------------------------------------------------------
    # POSITION HELPERS
    # --------------------------------------------------------

    def tcp_pos(self):
        return self.data.site_xpos[self.tcp_site].copy()

    def cube_pos(self):
        return self.data.xpos[self.cube_body].copy()

    def place_pos(self):
        return self.data.site_xpos[self.place_site].copy()

    def cube_in_tray(self):

        p = self.cube_pos()

        limit = TRAY_INNER / 2 - CUBE_SIZE / 2

        return bool(
            abs(p[0] - TRAY_CENTER[0]) <= limit
            and abs(p[1] - TRAY_CENTER[1]) <= limit
            and p[2] <= (
                TABLE_TOP_Z + TRAY_FLOOR + CUBE_SIZE / 2 + 0.015
            )
        )

    # --------------------------------------------------------
    # VELOCITY ACTUATOR CONTROL
    # --------------------------------------------------------

    def apply_control(self):
        q = self.data.qpos[self.arm_qadr]
        kv = self.model.actuator_gainprm[self.arm_act, 0]

        v_cmd = np.clip(
            VELOCITY_KP * (self.q_target - q),
            -self.max_speed,
            self.max_speed,
        )

        # gravity/Coriolis feed-forward as a velocity offset
        ff = self.data.qfrc_bias[self.arm_dadr] / kv

        self.data.ctrl[self.arm_act] = v_cmd + ff

        if len(self.grip_act):
            self.data.ctrl[self.grip_act] = self.grip_cmd

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

    def body_name(self, geom_id):
        body_id = self.model.geom_bodyid[int(geom_id)]
        return (
            mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(body_id),
            )
            or "<unnamed>"
        )