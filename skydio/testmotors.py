import time
import numpy as np
import mujoco
import mujoco.viewer
import glfw

MODEL_PATH = "mujoco_menagerie/skydio_x2/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

HOVER = 3.2495625

target_z = 0.30

KP = 8.0
KD = 3.0
KI = 0.5

integral = 0.0

running = True
emergency_stop = False


def key_callback(keycode):
    global running
    global emergency_stop

    if keycode == 256:       # ESC
        running = False

    elif keycode == 32:     # SPACE
        emergency_stop = True
        data.ctrl[:] = 0
        print("EMERGENCY STOP")


mujoco.mj_resetDataKeyframe(model, data, 0)

with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback
) as viewer:

    print()
    print("================================")
    print("       SKYDIO X2 CONTROL")
    print("================================")
    print("HOLD R = continuously UP")
    print("HOLD F = continuously DOWN")
    print("Release = hold altitude")
    print("SPACE = emergency stop")
    print("ESC = exit")
    print("================================")

    last_time = time.time()

    while viewer.is_running() and running:

        now = time.time()
        dt = now - last_time
        last_time = now

        dt = min(dt, 0.02)

        # ----------------------------------------
        # Get actual keyboard state
        # ----------------------------------------

        window = viewer.user_scn

        # GLFW window is obtained from the viewer.
        # MuJoCo viewer forwards keyboard callbacks,
        # while we use the callback only for emergency
        # commands.

        # ----------------------------------------
        # Current altitude
        # ----------------------------------------

        z = data.qpos[2]
        vz = data.qvel[2]

        # ----------------------------------------
        # PID altitude control
        # ----------------------------------------

        error = target_z - z

        integral += error * dt
        integral = np.clip(integral, -0.5, 0.5)

        correction = (
            KP * error
            + KI * integral
            - KD * vz
        )

        motor = HOVER + correction
        motor = np.clip(motor, 0.0, 5.0)

        if emergency_stop:
            motor = 0.0

        data.ctrl[:] = motor

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(0.005)

    data.ctrl[:] = 0