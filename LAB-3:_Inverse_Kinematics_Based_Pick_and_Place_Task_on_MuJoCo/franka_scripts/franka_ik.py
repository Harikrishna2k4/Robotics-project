#!/usr/bin/env python3
"""Mink inverse kinematics for Franka Panda."""

from __future__ import annotations

import numpy as np
import mink

from scipy.spatial.transform import Rotation


IK_DT = 0.02
IK_STEPS = 250

POSITION_TOL = 0.008
ROTATION_TOL = 0.18


def make_target_pose(position, yaw=0.0):
    """Create a target pose with the gripper pointing approximately down."""

    position = np.asarray(position, dtype=float)

    rotation_matrix = (
        Rotation.from_euler("z", yaw).as_matrix()
        @ Rotation.from_euler("x", np.pi).as_matrix()
    )

    return (
        mink.SE3.from_translation(position)
        @ mink.SE3.from_rotation(
            mink.SO3.from_matrix(rotation_matrix)
        )
    )


class FrankaIK:

    def __init__(self, scene, solver="daqp"):
        self.scene = scene
        self.model = scene.model
        self.solver = solver

        self.configuration = mink.Configuration(self.model)

        # Position and orientation task.
        self.task_pose = mink.FrameTask(
            frame_name=scene.ee_site_name,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.15,
            lm_damping=1e-4,
            gain=0.5,
        )

        # Position-only fallback task.
        self.task_position = mink.FrameTask(
            frame_name=scene.ee_site_name,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=1e-4,
            gain=0.5,
        )

        # Freeze every DOF except the seven arm DOFs.
        arm_dofs = set(map(int, scene.arm_dadr))

        frozen_dofs = [
            i
            for i in range(self.model.nv)
            if i not in arm_dofs
        ]

        self.freeze = mink.DofFreezingTask(
            model=self.model,
            dof_indices=frozen_dofs,
        )

        self.posture = mink.PostureTask(
            self.model,
            cost=1e-3,
        )

        self.limits = [
            mink.ConfigurationLimit(self.model)
        ]

        self.last_mode = "none"

    def _solve_task(
        self,
        task,
        position,
        yaw,
        require_rotation,
    ):
        self.configuration.update(self.scene.data.qpos)

        self.posture.set_target_from_configuration(
            self.configuration
        )

        task.set_target(
            make_target_pose(position, yaw)
        )

        for _ in range(IK_STEPS):
            error = task.compute_error(
                self.configuration
            )

            position_ok = (
                np.linalg.norm(error[:3])
                < POSITION_TOL
            )

            rotation_ok = (
                not require_rotation
                or np.linalg.norm(error[3:])
                < ROTATION_TOL
            )

            if position_ok and rotation_ok:
                return self.configuration.q[
                    self.scene.arm_qadr
                ].copy()

            velocity = mink.solve_ik(
                self.configuration,
                [task, self.posture],
                IK_DT,
                self.solver,
                damping=1e-3,
                limits=self.limits,
                constraints=[self.freeze],
            )

            self.configuration.integrate_inplace(
                velocity,
                IK_DT,
            )

        return None

    def solve(self, position, yaw=0.0):
        current_q = self.scene.data.qpos[
            self.scene.arm_qadr
        ].copy()

        best_solution = None
        best_cost = np.inf

        # Try several grasp orientations for a square object.
        for offset in (
            0.0,
            np.pi / 2,
            -np.pi / 2,
            np.pi,
        ):
            candidate = self._solve_task(
                self.task_pose,
                position,
                yaw + offset,
                require_rotation=True,
            )

            if candidate is None:
                continue

            cost = np.sum(
                np.abs(candidate - current_q)
            )

            if cost < best_cost:
                best_solution = candidate
                best_cost = cost

        if best_solution is not None:
            self.last_mode = "oriented"
            return best_solution

        self.last_mode = "position-only"

        return self._solve_task(
            self.task_position,
            position,
            yaw,
            require_rotation=False,
        )

    def move(
        self,
        position,
        yaw=0.0,
        max_steps=2000,
        tolerance=0.012,
    ):
        solution = self.solve(position, yaw)

        if solution is None:
            print(
                "IK failed for target:",
                np.round(position, 3),
            )
            return False

        self.scene.move_joints(
            solution,
            steps=max_steps,
            tolerance=tolerance,
        )

        actual_position = self.scene.tcp_pos()

        error = np.linalg.norm(
            actual_position
            - np.asarray(position, dtype=float)
        )

        print(
            f"Target={np.round(position, 3)} "
            f"TCP={np.round(actual_position, 3)} "
            f"error={error:.3f} m "
            f"mode={self.last_mode}"
        )

        return error < 0.05

    def move_linear(
        self,
        start,
        end,
        yaw=0.0,
        segments=5,
    ):
        """Move along a straight Cartesian path using intermediate IK targets."""

        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)

        for alpha in np.linspace(
            0.0,
            1.0,
            segments + 1,
        )[1:]:
            target = (
                (1.0 - alpha) * start
                + alpha * end
            )

            if not self.move(target, yaw):
                return False

        return True