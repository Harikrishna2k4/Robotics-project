import time
import numpy as np
import mujoco
import mujoco.viewer


# ============================================================
# LOAD MODEL
# ============================================================

modelPath = "scene_turtlebot3_waffle_pi.xml"

model = mujoco.MjModel.from_xml_path(modelPath)
data = mujoco.MjData(model)


# ============================================================
# ROBOT BODY
# ============================================================

body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "base"
)

if body_id < 0:
    raise RuntimeError("Body 'base' not found")


# ============================================================
# SPEED
# ============================================================

linear = 0.0
angular = 0.0


# ============================================================
# KEYBOARD
# ============================================================

def native_keyboard_callback(keycode):

    global linear, angular

    if keycode == 265:       # UP
        linear = 4.0
        angular = 0.0

    elif keycode == 264:     # DOWN
        linear = -4.0
        angular = 0.0

    elif keycode == 263:     # LEFT
        angular = 2.5

    elif keycode == 262:     # RIGHT
        angular = -2.5

    elif keycode == 32:      # SPACE
        linear = 0.0
        angular = 0.0


# ============================================================
# DRAW ONE AXIS
# ============================================================

def draw_arrow(scene, origin, direction, color):

    if scene.ngeom + 1 >= scene.maxgeom:
        return

    direction = np.asarray(direction, dtype=float)

    norm = np.linalg.norm(direction)

    if norm < 1e-8:
        return

    direction = direction / norm

    # --------------------------------------------------------
    # AXIS SIZE
    # --------------------------------------------------------

    axis_length = 0.80

    start = np.asarray(origin, dtype=float).copy()

    end = start + direction * axis_length


    # ========================================================
    # SHAFT
    # ========================================================

    shaft = scene.geoms[scene.ngeom]

    mujoco.mjv_initGeom(
        shaft,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).flatten(),
        np.asarray(color, dtype=np.float32)
    )

    mujoco.mjv_connector(
        shaft,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        0.0035,
        start,
        end
    )

    scene.ngeom += 1


    # ========================================================
    # ARROW HEAD
    # ========================================================

    if scene.ngeom >= scene.maxgeom:
        return

    head_length = 0.10
    head_radius = 0.012

    head_center = (
        end -
        direction * (head_length / 2.0)
    )


    # --------------------------------------------------------
    # Build rotation matrix
    # --------------------------------------------------------

    if abs(direction[2]) < 0.9:

        reference = np.array([
            0.0,
            0.0,
            1.0
        ])

    else:

        reference = np.array([
            0.0,
            1.0,
            0.0
        ])


    x_axis = np.cross(
        reference,
        direction
    )

    x_axis /= np.linalg.norm(x_axis)


    y_axis = np.cross(
        direction,
        x_axis
    )

    y_axis /= np.linalg.norm(y_axis)


    arrow_matrix = np.column_stack(
        (
            x_axis,
            y_axis,
            direction
        )
    ).flatten()


    head = scene.geoms[scene.ngeom]

    mujoco.mjv_initGeom(
        head,

        mujoco.mjtGeom.mjGEOM_ARROW,

        np.array([
            head_radius,
            head_radius,
            head_length / 2.0
        ]),

        head_center,

        arrow_matrix,

        np.asarray(color, dtype=np.float32)
    )

    scene.ngeom += 1


# ============================================================
# DRAW ORIGIN
# ============================================================

def draw_origin(scene, origin):

    if scene.ngeom >= scene.maxgeom:
        return

    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],

        mujoco.mjtGeom.mjGEOM_SPHERE,

        np.array([
            0.012,
            0.0,
            0.0
        ]),

        origin,

        np.eye(3).flatten(),

        np.array([
            1.0,
            1.0,
            1.0,
            1.0
        ])
    )

    scene.ngeom += 1


# ============================================================
# DRAW COMPLETE FRAME
# ============================================================

def draw_frame(scene, origin, rotation, alpha):

    # X = RED

    draw_arrow(
        scene,
        origin,
        rotation[:, 0],
        [1.0, 0.0, 0.0, alpha]
    )


    # Y = GREEN

    draw_arrow(
        scene,
        origin,
        rotation[:, 1],
        [0.0, 1.0, 0.0, alpha]
    )


    # Z = BLUE

    draw_arrow(
        scene,
        origin,
        rotation[:, 2],
        [0.0, 0.0, 1.0, alpha]
    )


# ============================================================
# MATRIX TEXT
# ============================================================

def make_matrix_text(R):

    return (
        "BASE ROTATION MATRIX\n"
        f"[ {R[0,0]: .3f}  {R[0,1]: .3f}  {R[0,2]: .3f} ]\n"
        f"[ {R[1,0]: .3f}  {R[1,1]: .3f}  {R[1,2]: .3f} ]\n"
        f"[ {R[2,0]: .3f}  {R[2,1]: .3f}  {R[2,2]: .3f} ]"
    )


# ============================================================
# START
# ============================================================

print()
print("=======================================================")
print("          TURTLEBOT 3-AXIS REFERENCE FRAMES")
print("=======================================================")
print()
print("UP       -> Forward")
print("DOWN     -> Backward")
print("LEFT     -> Rotate Left")
print("RIGHT    -> Rotate Right")
print("SPACE    -> Stop")
print()


# ============================================================
# VIEWER
# ============================================================

with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=native_keyboard_callback
) as viewer:

    while viewer.is_running():

        start_time = time.time()


        # ====================================================
        # MOTOR CONTROL
        # ====================================================

        data.actuator(
            "wheel_left"
        ).ctrl = linear - angular

        data.actuator(
            "wheel_right"
        ).ctrl = linear + angular


        # ====================================================
        # PHYSICS
        # ====================================================

        mujoco.mj_step(
            model,
            data
        )


        # ====================================================
        # BASE POSITION
        # ====================================================

        base_position = data.xpos[
            body_id
        ].copy()


        # ====================================================
        # BASE ROTATION
        # ====================================================

        rotation_matrix = data.xmat[
            body_id
        ].reshape(3, 3).copy()


        # ====================================================
        # PHYSICAL CHASSIS CENTER
        # ====================================================

        body_center_offset = np.array([
            -0.064,
             0.0,
             0.01
        ])


        robot_position = (
            base_position
            +
            rotation_matrix @ body_center_offset
        )


        # ====================================================
        # COMMON ORIGIN
        # ====================================================
        #
        # Both coordinate frames originate at exactly
        # this same point.
        #
        # ====================================================

        common_origin = robot_position


        # ====================================================
        # WORLD FRAME ROTATION
        # ====================================================

        world_rotation = np.eye(3)


        # ====================================================
        # CLEAR CUSTOM GEOMETRY
        # ====================================================

        viewer.user_scn.ngeom = 0


        # ====================================================
        # EXACT COMMON ORIGIN
        # ====================================================

        draw_origin(
            viewer.user_scn,
            common_origin
        )


        # ====================================================
        # WORLD FRAME
        # ====================================================

        draw_frame(
            viewer.user_scn,
            common_origin,
            world_rotation,
            1.0
        )


        # ====================================================
        # ROBOT BODY FRAME
        # ====================================================

        draw_frame(
            viewer.user_scn,
            common_origin,
            rotation_matrix,
            0.55
        )


        # ====================================================
        # MATRIX IN MUJOCO WINDOW
        # ====================================================

        matrix_text = make_matrix_text(
            rotation_matrix
        )

        viewer.set_texts(
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                matrix_text,
                ""
            )
        )


        # ====================================================
        # UPDATE
        # ====================================================

        viewer.sync()


        # ====================================================
        # TIMING
        # ====================================================

        rest_time = (
            model.opt.timestep
            -
            (time.time() - start_time)
        )

        if rest_time > 0:
            time.sleep(rest_time)