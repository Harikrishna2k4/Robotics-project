#!/usr/bin/env python3
"""Franka Panda pick-and-place demo using MuJoCo and Mink."""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from envsetup import FrankaScene, CUBE_SIZE
from franka_ik import FrankaIK


APPROACH_Z = 0.12
LIFT_Z = 0.16
GRASP_EXTRA_Z = 0.005

SETTLE_STEPS = 150
GRIP_STEPS = 300
LIFT_SUCCESS_THRESHOLD = 0.03
PLACE_SUCCESS_THRESHOLD = 0.10


def print_cube_contacts(scene):
    """Print all current contacts involving the cube."""

    print("\nContacts involving cube:")

    cube_geom_id = mujoco.mj_name2id(
        scene.model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "box_geom",
    )

    if cube_geom_id < 0:
        print("Could not find geometry named 'box_geom'.")
        return

    found = False

    for i in range(scene.data.ncon):
        contact = scene.data.contact[i]

        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)

        if cube_geom_id not in (geom1, geom2):
            continue

        other_geom = (
            geom2 if geom1 == cube_geom_id else geom1
        )

        other_name = mujoco.mj_id2name(
            scene.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            other_geom,
        )

        if other_name is None:
            other_name = f"geometry_{other_geom}"

        print(
            f"  box_geom <-> {other_name}: "
            f"distance={contact.dist:.6f} m"
        )

        found = True

    if not found:
        print("  No contacts involving the cube.")


def run_episode(scene, ik, seed):
    """Run one pick-and-place episode."""

    scene.reset(seed=seed)
    scene.step(250)

    cube = scene.cube_pos().copy()
    tray = scene.tray_pos().copy()

    print(
        f"\nEpisode {seed}: "
        f"cube={np.round(cube, 4)}, "
        f"tray={np.round(tray, 4)}"
    )

    # Calculate TCP targets.
    grasp = cube + np.array([
        0.0,
        0.0,
        CUBE_SIZE / 2 + GRASP_EXTRA_Z - 0.02,
    ])

    approach = grasp + np.array([
        0.0,
        0.0,
        APPROACH_Z,
    ])

    lift = grasp + np.array([
        0.0,
        0.0,
        LIFT_Z,
    ])

    # --------------------------------------------------------
    # 1. Approach
    # --------------------------------------------------------

    print("\n1. Moving above cube")

    if not ik.move(approach, yaw=0.0):
        print("FAILED: approach")
        return False

    # --------------------------------------------------------
    # 2. Descend
    # --------------------------------------------------------

    print("\n2. Descending to cube")

    if not ik.move_linear(
        approach,
        grasp,
        yaw=0.0,
        segments=4,
    ):
        print("FAILED: descent")
        return False

    scene.step(SETTLE_STEPS)

    print("Cube at grasp:", np.round(scene.cube_pos(), 4))
    print("TCP at grasp:", np.round(scene.tcp_pos(), 4))

    # --------------------------------------------------------
    # 3. Close gripper
    # --------------------------------------------------------

    print("\n3. Closing gripper")

    scene.close_gripper()
    scene.step(GRIP_STEPS)

    cube_after_grasp = scene.cube_pos().copy()
    tcp_after_grasp = scene.tcp_pos().copy()

    finger_after_grasp = scene.data.qpos[
        scene.finger_qadr
    ].copy()

    held_offset = cube_after_grasp - tcp_after_grasp

    print("Cube after grasp:", np.round(cube_after_grasp, 4))
    print("TCP after grasp:", np.round(tcp_after_grasp, 4))
    print(
        "Finger positions after grasp:",
        np.round(finger_after_grasp, 5),
    )
    print("Cube relative to TCP:", np.round(held_offset, 4))

    # Inspect cube contacts after closing.
    print_cube_contacts(scene)

    if np.all(finger_after_grasp >= 0.035):
        print(
            "WARNING: Finger joints remain near their open "
            "positions."
        )

    # --------------------------------------------------------
    # 4. Lift
    # --------------------------------------------------------

    print("\n4. Lifting cube")

    if not ik.move_linear(
        grasp,
        lift,
        yaw=0.0,
        segments=4,
    ):
        print("FAILED: lift")
        return False

    cube_after_lift = scene.cube_pos().copy()
    tcp_after_lift = scene.tcp_pos().copy()

    finger_after_lift = scene.data.qpos[
        scene.finger_qadr
    ].copy()

    cube_displacement = (
        cube_after_lift - cube_after_grasp
    )

    print("Cube after lifting:", np.round(cube_after_lift, 4))
    print("TCP after lifting:", np.round(tcp_after_lift, 4))
    print(
        "Finger positions after lifting:",
        np.round(finger_after_lift, 5),
    )
    print(
        "Cube displacement during lift:",
        np.round(cube_displacement, 4),
    )

    if cube_displacement[2] <= LIFT_SUCCESS_THRESHOLD:
        print(
            "\nGRASP FAILED: the cube did not rise by at least "
            f"{LIFT_SUCCESS_THRESHOLD:.2f} m."
        )

        print_cube_contacts(scene)

        print(
            "Stopping before tray transit. Inspect the contact "
            "diagnostics before changing the grasp geometry."
        )

        return False

    # --------------------------------------------------------
    # 5. Calculate placement target
    # --------------------------------------------------------

    place_center = tray.copy()
    place_center[2] += CUBE_SIZE / 2 + 0.015

    # Compensate for the measured cube-to-TCP offset.
    place_tcp = place_center - held_offset

    transit = place_tcp + np.array([
        0.0,
        0.0,
        0.15,
    ])

    # --------------------------------------------------------
    # 6. Move above tray
    # --------------------------------------------------------

    print("\n5. Moving above tray")

    if not ik.move(transit, yaw=0.0):
        print("FAILED: transit")
        return False

    # --------------------------------------------------------
    # 7. Lower cube
    # --------------------------------------------------------

    print("\n6. Lowering cube")

    if not ik.move_linear(
        transit,
        place_tcp,
        yaw=0.0,
        segments=4,
    ):
        print("FAILED: lowering")
        return False

    scene.step(SETTLE_STEPS)

    print("Cube before release:", np.round(scene.cube_pos(), 4))
    print("TCP before release:", np.round(scene.tcp_pos(), 4))

    # --------------------------------------------------------
    # 8. Release
    # --------------------------------------------------------

    print("\n7. Releasing cube")

    scene.open_gripper()
    scene.step(GRIP_STEPS)
    scene.step(300)

    final_cube = scene.cube_pos().copy()

    xy_error = np.linalg.norm(
        final_cube[:2] - place_center[:2]
    )

    success = xy_error < PLACE_SUCCESS_THRESHOLD

    print("Final cube:", np.round(final_cube, 4))
    print("Target centre:", np.round(place_center, 4))
    print("XY error:", round(xy_error, 4), "m")
    print(
        "Result:",
        "SUCCESS" if success else "NOT SUCCESSFUL",
    )

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Franka Panda pick-and-place simulation"
    )

    parser.add_argument(
        "--scene",
        default=None,
        help="Path to the MuJoCo scene XML",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--solver",
        default="daqp",
    )

    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Display the MuJoCo viewer",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Simulation playback speed",
    )

    args = parser.parse_args()

    kwargs = {
        "seed": args.seed,
        "realtime_factor": args.speed,
    }

    if args.scene:
        kwargs["scene_xml"] = args.scene

    scene = FrankaScene(**kwargs)
    scene.print_model_summary()

    ik = FrankaIK(
        scene,
        solver=args.solver,
    )

    if args.viewer:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(
            scene.model,
            scene.data,
        ) as viewer:

            viewer.cam.lookat[:] = [
                0.35,
                -0.1,
                0.85,
            ]

            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25

            scene.viewer = viewer

            successes = 0

            for episode in range(args.episodes):
                if run_episode(
                    scene,
                    ik,
                    args.seed + episode,
                ):
                    successes += 1

                if episode + 1 < args.episodes:
                    time.sleep(0.5)

            print(
                f"\nSummary: {successes}/{args.episodes} "
                "successful episodes"
            )

            # Keep the viewer open after the episode.
            while viewer.is_running():
                scene.step(5)

    else:
        successes = 0

        for episode in range(args.episodes):
            if run_episode(
                scene,
                ik,
                args.seed + episode,
            ):
                successes += 1

        print(
            f"\nSummary: {successes}/{args.episodes} "
            "successful episodes"
        )


if __name__ == "__main__":
    main()