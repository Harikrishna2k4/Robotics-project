import time
import numpy as np
import mujoco
import mujoco.viewer

from utils import Rx, Ry, Rz, ELEMENTARY_ROTATIONS, set_body_orientation

MODEL_PATH = "asymmetric_body.xml"


# ---------------------------------------------------------------
# TODO(student): edit this sequence to try different experiments.
# Each entry is (axis, angle_radians, frame) where frame is either
# "current" (compose on the RIGHT: R_new = R_old @ R_step) or
# "fixed" (compose on the LEFT: R_new = R_step @ R_old).
# Compare the two frame choices for the SAME list of (axis, angle).
# ---------------------------------------------------------------
rotation_sequence = [
    ("z", np.deg2rad(90), "current"),   # TODO: try "fixed" here instead
    ("x", np.deg2rad(90), "current"),   # TODO: try "fixed" here instead
]


def compose_sequence(sequence):
    """TODO(student): implement this.

    Given a list of (axis, angle, frame) tuples, return the final
    3x3 rotation matrix R obtained by applying them in order,
    starting from R = identity.

    Hint: ELEMENTARY_ROTATIONS["x"](angle) gives you R_x(angle) etc.
    Hint: think about *why* current-frame composition is a right-
    multiply and fixed-frame composition is a left-multiply -- this
    is exactly the "Current Frame" vs. "Fixed Frame" distinction
    from the Composition-of-Rotations lecture. Don't just guess;
    check it against your Problem 3/4 reasoning.
    """
    R = np.eye(3)
    for axis, angle, frame in sequence:
        R_step = ELEMENTARY_ROTATIONS[axis](angle)
        # TODO: replace the next line with the correct composition
        # rule depending on `frame`.
        raise NotImplementedError("Implement current- vs fixed-frame composition")
    return R


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    R_final = compose_sequence(rotation_sequence)
    set_body_orientation(data, R_final)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer open. Close the window to exit.")
        print(f"Applied sequence: {rotation_sequence}")
        while viewer.is_running():
            viewer.sync()
            time.sleep(1 / 60)

    # TODO(student, optional): instead of a static final pose,
    # animate the sequence step by step (e.g. interpolate each
    # elemental rotation over ~1 second) so the recording clearly
    # shows each step happening, not just the end result.


if __name__ == "__main__":
    main()
