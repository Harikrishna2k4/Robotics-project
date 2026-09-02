"""
01_rotation_sandbox.py -- HW1 Part 2, Task 1: does rotation order matter?

Interactive version:
  - 3 buttons rotate the dart +90 deg about X/Y/Z
  - 2 buttons toggle WORLD (fixed) vs BODY (current) frame composition
  - Object is offset from the world origin (and lifted above the floor)
    so both axis triads are clearly visible without overlapping or
    clipping into the checkerboard
  - World axes (solid RGB) are fixed near the origin, lifted slightly
    off the floor to avoid z-fighting
  - Body axes (pastel RGB) are attached to the object and rotate with it
  - Floor reflection is disabled so there's no "ghost" duplicate axis
    reflected in the glossy checkerboard material
  - Each button press ANIMATES the 90 deg step over ~0.6s instead of
    snapping instantly, so the rotation is visible in a recording

Keyboard shortcuts: 1/2/3 = rotate X/Y/Z, R = reset, F = toggle frame.
Mouse: drag = orbit camera, scroll = zoom.
"""

import numpy as np
import glfw
import mujoco

from utils import ELEMENTARY_ROTATIONS, set_body_orientation

MODEL_PATH = "../model/asymmetric_body.xml"

WINDOW_W, WINDOW_H = 1200, 900
BTN_W, BTN_H = 160, 50
BTN_MARGIN = 15

# How far to push the object away from the world origin, and how high
# to lift it off the floor, so the world-frame axis triad (drawn near
# the origin) and the body-frame axis triad (drawn on the object)
# don't overlap or clip into the checkerboard plane.
OBJECT_OFFSET = np.array([1.4, 0.0, 0.35])

# Small lift for the world axis triad so it doesn't z-fight with the
# ground plane geometry.
WORLD_AXIS_HEIGHT = 0.03

AXIS_LEN = 0.6
AXIS_WIDTH = 0.015

# Solid RGB for the fixed WORLD axes (always near the origin, never rotate)
WORLD_AXIS_RGBA = {
    "x": np.array([1.0, 0.0, 0.0, 1.0]),
    "y": np.array([0.0, 1.0, 0.0, 1.0]),
    "z": np.array([0.0, 0.2, 1.0, 1.0]),
}
# Pastel RGB for the BODY axes (attached to the object, rotate with it)
BODY_AXIS_RGBA = {
    "x": np.array([1.0, 0.55, 0.55, 1.0]),
    "y": np.array([0.55, 1.0, 0.55, 1.0]),
    "z": np.array([0.55, 0.75, 1.0, 1.0]),
}

ANIM_DURATION = 0.6  # seconds per 90 deg step


class ButtonUI:
    """Simple clickable-button overlay drawn with mjr_label."""

    def __init__(self):
        self.buttons = {
            "rot_x": {"label": "Rotate X +90", "x": BTN_MARGIN, "y": BTN_MARGIN},
            "rot_y": {"label": "Rotate Y +90", "x": BTN_MARGIN, "y": BTN_MARGIN + (BTN_H + 10)},
            "rot_z": {"label": "Rotate Z +90", "x": BTN_MARGIN, "y": BTN_MARGIN + 2 * (BTN_H + 10)},
            "reset": {"label": "Reset", "x": BTN_MARGIN, "y": BTN_MARGIN + 3 * (BTN_H + 10)},
            "frame_world": {"label": "Frame: WORLD", "x": BTN_MARGIN, "y": BTN_MARGIN + 4 * (BTN_H + 10) + 20},
            "frame_body": {"label": "Frame: BODY", "x": BTN_MARGIN, "y": BTN_MARGIN + 5 * (BTN_H + 10) + 20},
        }
        self.frame_mode = "body"  # "body" (current/moving) or "world" (fixed)

    def _rect_bottom_left(self, key, win_h, fb_scale_x, fb_scale_y):
        b = self.buttons[key]
        x_fb = b["x"] * fb_scale_x
        w_fb = BTN_W * fb_scale_x
        h_fb = BTN_H * fb_scale_y
        y_top_fb = b["y"] * fb_scale_y
        y_fb = win_h - y_top_fb - h_fb
        return mujoco.MjrRect(int(x_fb), int(y_fb), int(w_fb), int(h_fb))

    def draw(self, ctx, win_h_px, fb_scale_x, fb_scale_y, animating):
        for key, b in self.buttons.items():
            rect = self._rect_bottom_left(key, win_h_px, fb_scale_x, fb_scale_y)
            is_active_frame = (key == "frame_world" and self.frame_mode == "world") or \
                               (key == "frame_body" and self.frame_mode == "body")
            if key.startswith("frame_"):
                bg = (0.2, 0.6, 0.2) if is_active_frame else (0.25, 0.25, 0.25)
            else:
                bg = (0.1, 0.25, 0.55) if not animating else (0.4, 0.4, 0.4)
            mujoco.mjr_label(
                rect, mujoco.mjtFont.mjFONT_NORMAL, b["label"],
                bg[0], bg[1], bg[2], 0.9,
                1.0, 1.0, 1.0,
                ctx,
            )

    def hit_test(self, key, mx, my):
        b = self.buttons[key]
        return (b["x"] <= mx <= b["x"] + BTN_W) and (b["y"] <= my <= b["y"] + BTN_H)

    def click(self, mx, my):
        for key in self.buttons:
            if self.hit_test(key, mx, my):
                return key
        return None


def find_free_joint_qposadr(model):
    """Locate the qpos address of the (first) free joint in the model,
    so we can set the object's world position directly."""
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return model.jnt_qposadr[j]
    raise RuntimeError("No free joint found in model -- can't set position.")


def add_axis_triad(scene, origin, R, length, width, rgba_dict):
    """Append 3 arrow geoms (x,y,z) to the mjvScene, starting at `origin`
    and oriented by rotation matrix R (R's columns are the axis directions
    expressed in world coordinates)."""
    for i, axis in enumerate(("x", "y", "z")):
        direction = R[:, i]
        end = origin + length * direction
        if scene.ngeom >= scene.maxgeom:
            return
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
            np.zeros(3), np.eye(3).flatten(), rgba_dict[axis].astype(np.float32),
        )
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, origin, end)
        scene.ngeom += 1


def apply_step(R, axis, frame_mode):
    """Full (non-animated) +90 deg step, used to compute the animation
    target."""
    R_step = ELEMENTARY_ROTATIONS[axis](np.deg2rad(90))
    if frame_mode == "body":
        return R @ R_step
    elif frame_mode == "world":
        return R_step @ R
    else:
        raise ValueError(f"Unknown frame mode: {frame_mode}")


def partial_step(R_base, axis, theta, frame_mode):
    """Rotation matrix at an intermediate angle `theta` (0..90 deg) of the
    CURRENT animated step, composed onto R_base with the correct frame
    convention. At theta=90deg this equals apply_step(R_base, axis, ...)."""
    R_step = ELEMENTARY_ROTATIONS[axis](theta)
    if frame_mode == "body":
        return R_base @ R_step
    else:
        return R_step @ R_base


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    qadr = find_free_joint_qposadr(model)

    R_base = np.eye(3)      # orientation BEFORE the in-progress animation
    R_display = np.eye(3)   # orientation actually rendered this frame

    def push_pose(R):
        set_body_orientation(data, R)
        data.qpos[qadr:qadr + 3] = OBJECT_OFFSET  # keep position fixed/offset
        mujoco.mj_forward(model, data)

    push_pose(R_display)

    if not glfw.init():
        raise RuntimeError("Could not initialize GLFW")

    window = glfw.create_window(WINDOW_W, WINDOW_H, "Rotation Sandbox", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultCamera(cam)
    mujoco.mjv_defaultOption(opt)
    opt.frame = mujoco.mjtFrame.mjFRAME_NONE  # hide MuJoCo built-in body/frame axes
    cam.lookat[:] = np.array([OBJECT_OFFSET[0] * 0.5, 0.0, OBJECT_OFFSET[2] * 0.5])
    cam.distance = 3.5
    cam.azimuth = 120
    cam.elevation = -20

    scene = mujoco.MjvScene(model, maxgeom=10000)
    scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0  # kill floor reflection ghosting
    ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    ui = ButtonUI()

    anim = {"active": False, "axis": None, "frame_mode": None, "t0": 0.0}
    drag = {"on": False, "x": 0, "y": 0}

    def reset_pose():
        nonlocal R_base, R_display
        R_base = np.eye(3)
        R_display = np.eye(3)
        anim["active"] = False
        anim["axis"] = None
        anim["frame_mode"] = None
        anim["t0"] = 0.0

    def start_animation(axis):
        if anim["active"]:
            return  # ignore input mid-animation, keep it simple/clean
        anim["active"] = True
        anim["axis"] = axis
        anim["frame_mode"] = ui.frame_mode
        anim["t0"] = glfw.get_time()

    def mouse_button_callback(win, button, action, mods):
        if button != glfw.MOUSE_BUTTON_LEFT:
            return
        mx, my = glfw.get_cursor_pos(win)
        if action == glfw.PRESS:
            hit = ui.click(mx, my)
            if hit in ("rot_x", "rot_y", "rot_z"):
                start_animation(hit.split("_")[1])
            elif hit == "reset":
                reset_pose()
            elif hit == "frame_world":
                ui.frame_mode = "world"
            elif hit == "frame_body":
                ui.frame_mode = "body"
            else:
                drag["on"] = True
                drag["x"], drag["y"] = mx, my
        elif action == glfw.RELEASE:
            drag["on"] = False

    def cursor_pos_callback(win, xpos, ypos):
        if drag["on"]:
            dx = xpos - drag["x"]
            dy = ypos - drag["y"]
            cam.azimuth -= dx * 0.3
            cam.elevation = np.clip(cam.elevation - dy * 0.3, -89, 89)
            drag["x"], drag["y"] = xpos, ypos

    def scroll_callback(win, xoffset, yoffset):
        cam.distance = max(0.1, cam.distance * (1 - 0.1 * yoffset))

    def key_callback(win, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_1:
            start_animation("x")
        elif key == glfw.KEY_2:
            start_animation("y")
        elif key == glfw.KEY_3:
            start_animation("z")
        elif key == glfw.KEY_F:
            if not anim["active"]:
                ui.frame_mode = "world" if ui.frame_mode == "body" else "body"
        elif key == glfw.KEY_R:
            reset_pose()

    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_key_callback(window, key_callback)

    print("Buttons: Rotate X/Y/Z (animated +90 deg), Reset, Frame WORLD/BODY toggle.")
    print("Keyboard: 1/2/3 = rotate X/Y/Z, R = reset, F = toggle frame. Drag = orbit, scroll = zoom.")

    while not glfw.window_should_close(window):
        # --- advance animation ---
        if anim["active"]:
            t = (glfw.get_time() - anim["t0"]) / ANIM_DURATION
            if t >= 1.0:
                R_base = apply_step(R_base, anim["axis"], anim["frame_mode"])
                R_display = R_base
                anim["active"] = False
            else:
                # ease-in-out for a nicer-looking rotation
                s = t * t * (3 - 2 * t)
                theta = np.deg2rad(90) * s
                R_display = partial_step(R_base, anim["axis"], theta, anim["frame_mode"])
        push_pose(R_display)

        win_w, win_h = glfw.get_window_size(window)
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        scale_x, scale_y = fb_w / win_w, fb_h / win_h

        viewport = mujoco.MjrRect(0, 0, fb_w, fb_h)
        mujoco.mjv_updateScene(
            model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene
        )

        # World axes: fixed near the origin (lifted slightly off the floor)
        add_axis_triad(
            scene, np.array([0.0, 0.0, WORLD_AXIS_HEIGHT]), np.eye(3),
            AXIS_LEN, AXIS_WIDTH, WORLD_AXIS_RGBA,
        )
        # Body axes: at the (fixed) offset position, oriented by R_display
        add_axis_triad(scene, OBJECT_OFFSET, R_display, AXIS_LEN, AXIS_WIDTH, BODY_AXIS_RGBA)

        mujoco.mjr_render(viewport, scene, ctx)

        ui.draw(ctx, win_h, scale_x, scale_y, anim["active"])

        legend = (
            "World axes (solid, at origin): X=red Y=green Z=blue\n"
            "Body axes (pastel, on object): X=pink Y=mint Z=sky\n"
            f"Frame mode: {ui.frame_mode.upper()}"
            + ("   [animating...]" if anim["active"] else "")
        )
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
            viewport, legend, "", ctx,
        )

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    main()