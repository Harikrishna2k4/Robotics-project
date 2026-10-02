import time
import math
import numpy as np
import mujoco
import mujoco.viewer

MODEL_PATH = "mujoco_menagerie/skydio_x2/scene.xml"

# ============================================================
# MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# ============================================================
# X2 CONSTANTS
# ============================================================

HOVER = 3.2495625

TARGET_Z_MIN = 0.15
TARGET_Z_MAX = 5.0

target_z = 0.30

# ============================================================
# ALTITUDE PID
# ============================================================

KP_Z = 10.0
KD_Z = 4.0
KI_Z = 1.0

z_integral = 0.0

# ============================================================
# ATTITUDE CONTROL
# ============================================================

pitch_cmd = 0.0
roll_cmd = 0.0

yaw_cmd = 0.0

MAX_TILT = math.radians(10.0)
MAX_YAW = math.radians(45.0)

# ============================================================
# ROTOR POSITIONS
# From your x2.xml
# ============================================================

rotor_x = np.array([
    -0.14,
    -0.14,
     0.14,
     0.14
])

rotor_y = np.array([
    -0.18,
     0.18,
     0.18,
    -0.18
])

# Actual X2 yaw directions
yaw_sign = np.array([
    -1.0,
     1.0,
    -1.0,
     1.0
])

YAW_GEAR = 0.0201

# ============================================================
# ATTITUDE PID
# ============================================================

KP_ROLL = 1.2
KD_ROLL = 0.20

KP_PITCH = 1.2
KD_PITCH = 0.20

KP_YAW = 0.5
KD_YAW = 0.10

# ============================================================
# KEY INPUT
# ============================================================

keys = {
    "w": False,
    "a": False,
    "s": False,
    "d": False,
    "r": False,
    "f": False,
    "q": False,
    "e": False,
}

# Time of the most recent event for each key
last_event = {
    "w": 0.0,
    "a": 0.0,
    "s": 0.0,
    "d": 0.0,
    "r": 0.0,
    "f": 0.0,
    "q": 0.0,
    "e": 0.0,
}

# Keyboard repeat timeout.
#
# As long as the OS keeps sending repeated events,
# the key remains active.
#
# When you release the key, repeat events stop and
# the key becomes inactive after this timeout.
KEY_TIMEOUT = 0.12

running = True
emergency_stop = False


# ============================================================
# KEYBOARD CALLBACK
# ============================================================

def key_callback(keycode):

    global running
    global emergency_stop

    # ESC
    if keycode == 256:
        running = False
        return

    # SPACE
    if keycode == 32:

        if not emergency_stop:

            reset_drone()

        else:

            # Resume at original X2 hover state
            mujoco.mj_resetDataKeyframe(
                model,
                data,
                0
            )

            # Reset all commands
            for key in keys:
                keys[key] = False
                last_event[key] = 0.0

            emergency_stop = False

            print()
            print("EMERGENCY STOP OFF")
            print("Drone resumed")
            print()

        return

    try:
        key = chr(keycode).lower()
    except (ValueError, OverflowError):
        return

    if key in keys:

        now = time.monotonic()

        # Mark this key active
        keys[key] = True

        # Remember the latest key event
        last_event[key] = now


# ============================================================
# RESET
# ============================================================

def reset_drone():

    global target_z
    global pitch_cmd
    global roll_cmd
    global yaw_cmd
    global z_integral
    global emergency_stop

    # Reset to original X2 hover keyframe
    mujoco.mj_resetDataKeyframe(
        model,
        data,
        0
    )

    target_z = 0.30

    pitch_cmd = 0.0
    roll_cmd = 0.0
    yaw_cmd = 0.0

    z_integral = 0.0

    for key in keys:
        keys[key] = False
        last_event[key] = 0.0

    emergency_stop = True

    data.ctrl[:] = 0.0

    print()
    print("==========================================")
    print("             EMERGENCY STOP")
    print("==========================================")
    print("Position reset")
    print("Orientation reset")
    print("Velocity reset")
    print("Altitude = 0.30 m")
    print("Motors OFF")
    print("Press SPACE to resume")
    print("==========================================")


# ============================================================
# HELD-KEY TEST
# ============================================================

def is_held(key):

    now = time.monotonic()

    if now - last_event[key] < KEY_TIMEOUT:
        return True

    keys[key] = False

    return False


# ============================================================
# QUATERNION -> ROTATION MATRIX
# ============================================================

def quat_to_rotmat(q):

    R = np.zeros((3, 3))

    mujoco.mju_quat2Mat(
        R.reshape(-1),
        q
    )

    return R


# ============================================================
# ROLL / PITCH / YAW
# ============================================================

def get_rpy(R):

    roll = math.atan2(
        R[2, 1],
        R[2, 2]
    )

    pitch = math.atan2(
        -R[2, 0],
        math.sqrt(
            R[2, 1] ** 2 +
            R[2, 2] ** 2
        )
    )

    yaw = math.atan2(
        R[1, 0],
        R[0, 0]
    )

    return roll, pitch, yaw


# ============================================================
# ANGLE WRAP
# ============================================================

def wrap_angle(angle):

    return (
        angle + math.pi
    ) % (
        2.0 * math.pi
    ) - math.pi


# ============================================================
# VIEWER
# ============================================================

with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback
) as viewer:

    print()
    print("==============================================")
    print("            SKYDIO X2 CONTROL")
    print("==============================================")
    print()
    print("HOLD R = climb")
    print("HOLD F = descend")
    print()
    print("HOLD W = forward")
    print("HOLD S = backward")
    print("HOLD A = left")
    print("HOLD D = right")
    print()
    print("HOLD Q = yaw left")
    print("HOLD E = yaw right")
    print()
    print("SPACE = reset + emergency stop")
    print("SPACE = resume")
    print("ESC = exit")
    print()
    print("A click by itself does NOT remain active.")
    print("==============================================")

    last_time = time.monotonic()

    while viewer.is_running() and running:

        now = time.monotonic()

        dt = now - last_time

        last_time = now

        dt = np.clip(
            dt,
            0.001,
            0.02
        )

        # ====================================================
        # EMERGENCY STOP
        # ====================================================

        if emergency_stop:

            data.ctrl[:] = 0.0

            viewer.sync()

            time.sleep(0.005)

            continue

        # ====================================================
        # CHECK WHICH KEYS ARE STILL HELD
        # ====================================================

        w = is_held("w")
        a = is_held("a")
        s = is_held("s")
        d = is_held("d")

        r = is_held("r")
        f = is_held("f")

        q = is_held("q")
        e = is_held("e")

        # ====================================================
        # ALTITUDE
        # ====================================================

        if r and not f:

            target_z += 0.6 * dt

        elif f and not r:

            target_z -= 0.6 * dt

        target_z = np.clip(
            target_z,
            TARGET_Z_MIN,
            TARGET_Z_MAX
        )

        # ====================================================
        # PITCH
        # ====================================================

        if w and not s:

            pitch_cmd += math.radians(35.0) * dt

        elif s and not w:

            pitch_cmd -= math.radians(35.0) * dt

        else:

            # Return toward level
            if pitch_cmd > 0:

                pitch_cmd -= math.radians(50.0) * dt

                pitch_cmd = max(
                    0.0,
                    pitch_cmd
                )

            elif pitch_cmd < 0:

                pitch_cmd += math.radians(50.0) * dt

                pitch_cmd = min(
                    0.0,
                    pitch_cmd
                )

        pitch_cmd = np.clip(
            pitch_cmd,
            -MAX_TILT,
            MAX_TILT
        )

        # ====================================================
        # ROLL
        # ====================================================

        if a and not d:

            roll_cmd -= math.radians(35.0) * dt

        elif d and not a:

            roll_cmd += math.radians(35.0) * dt

        else:

            # Return toward level
            if roll_cmd > 0:

                roll_cmd -= math.radians(50.0) * dt

                roll_cmd = max(
                    0.0,
                    roll_cmd
                )

            elif roll_cmd < 0:

                roll_cmd += math.radians(50.0) * dt

                roll_cmd = min(
                    0.0,
                    roll_cmd
                )

        roll_cmd = np.clip(
            roll_cmd,
            -MAX_TILT,
            MAX_TILT
        )

        # ====================================================
        # YAW
        # ====================================================

        if q and not e:

            yaw_cmd -= math.radians(45.0) * dt

        elif e and not q:

            yaw_cmd += math.radians(45.0) * dt

        # ====================================================
        # CURRENT STATE
        # ====================================================

        z = data.qpos[2]

        vz = data.qvel[2]

        quat = data.qpos[3:7]

        R = quat_to_rotmat(quat)

        roll, pitch, yaw_angle = get_rpy(R)

        gyro = data.qvel[3:6]

        # ====================================================
        # ALTITUDE PID
        # ====================================================

        z_error = target_z - z

        z_integral += z_error * dt

        z_integral = np.clip(
            z_integral,
            -0.5,
            0.5
        )

        vertical_acc = (
            KP_Z * z_error
            + KI_Z * z_integral
            - KD_Z * vz
        )

        vertical_acc = np.clip(
            vertical_acc,
            -3.0,
            3.0
        )

        # ====================================================
        # TILT COMPENSATION
        # ====================================================

        tilt_factor = (
            math.cos(roll)
            * math.cos(pitch)
        )

        tilt_factor = max(
            tilt_factor,
            0.75
        )

        hover_total = HOVER * 4.0

        total_vertical_thrust = (
            hover_total
            + vertical_acc
            * hover_total
            / 9.81
        )

        total_thrust = (
            total_vertical_thrust
            / tilt_factor
        )

        base_thrust = total_thrust / 4.0

        base_thrust = np.clip(
            base_thrust,
            0.0,
            5.0
        )

        # ====================================================
        # ATTITUDE PID
        # ====================================================

        roll_error = roll_cmd - roll

        pitch_error = pitch_cmd - pitch

        yaw_error = wrap_angle(
            yaw_cmd - yaw_angle
        )

        roll_torque = (
            KP_ROLL * roll_error
            - KD_ROLL * gyro[0]
        )

        pitch_torque = (
            KP_PITCH * pitch_error
            - KD_PITCH * gyro[1]
        )

        yaw_torque = (
            KP_YAW * yaw_error
            - KD_YAW * gyro[2]
        )

        # ====================================================
        # MOTOR MIXER
        # ====================================================

        B = np.array([
            [
                1.0,
                1.0,
                1.0,
                1.0
            ],
            [
                rotor_y[0],
                rotor_y[1],
                rotor_y[2],
                rotor_y[3]
            ],
            [
                -rotor_x[0],
                -rotor_x[1],
                -rotor_x[2],
                -rotor_x[3]
            ],
            [
                yaw_sign[0] * YAW_GEAR,
                yaw_sign[1] * YAW_GEAR,
                yaw_sign[2] * YAW_GEAR,
                yaw_sign[3] * YAW_GEAR
            ]
        ])

        desired = np.array([
            total_thrust,
            roll_torque,
            pitch_torque,
            yaw_torque
        ])

        motor = np.linalg.solve(
            B.T @ B + 1e-6 * np.eye(4),
            B.T @ desired
        )

        # ====================================================
        # MOTOR LIMIT
        # ====================================================

        motor = np.clip(
            motor,
            0.0,
            13.0
        )

        # ====================================================
        # APPLY
        # ====================================================

        data.ctrl[:] = motor

        # ====================================================
        # PHYSICS
        # ====================================================

        mujoco.mj_step(
            model,
            data
        )

        viewer.sync()

        time.sleep(0.002)

    data.ctrl[:] = 0.0