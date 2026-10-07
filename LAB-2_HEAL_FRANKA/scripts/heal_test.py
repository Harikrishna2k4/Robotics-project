from pathlib import Path
import time
import mujoco
import mujoco.viewer
import numpy as np

# --------------------------------------------------
# Find XML relative to this Python file
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
XML_PATH = (
    PROJECT_ROOT
    / "robot_descriptions"
    / "single_arm_heal_effort_actuation_rs.xml"
)

# --------------------------------------------------
# Load MuJoCo model
# --------------------------------------------------
model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)

print("======================================")
print("HEAL 6-DOF MuJoCo Model Initialized")
print("======================================")

# ... Keep your existing imports and model loading setup above ...

# Define customized PD parameters to hold the weight of the HEAL links
# We assign higher stiffness (Kp) to the lower, heavier joints (shoulder/elbow)
Kp = np.array([350.0, 600.0, 400.0, 150.0, 60.0, 15.0])
Kd = np.array([35.0,  60.0,  40.0,  15.0,  6.0,  1.5])

# ... Keep your existing imports, model load, and Kp/Kd definitions above ...

# Define two separate distinct poses for the HEAL arm (in Radians)
heal_home_pose  = np.array([0.0, -0.6,  1.2, 0.0,  0.5, 0.0])
heal_reach_pose = np.array([0.8, -0.2,  0.7, 0.4, -0.3, 0.5])

print("🚀 Launching HEAL dynamic standalone controller...")
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    start_time = time.time()

    while viewer.is_running():
        step_start = time.time()
        elapsed_time = time.time() - start_time

        # --- Dynamic Trajectory Generation ---
        # Smoothly oscillate back and forth between home and reach poses using a sine wave
        # Oscillator ranges from 0.0 to 1.0 continuously over time
        oscillator = 0.5 * (1.0 + np.sin(elapsed_time * 1.5)) 
        q_target = (1.0 - oscillator) * heal_home_pose + oscillator * heal_reach_pose

        # 1. Read current angles (qpos) and angular speeds (qvel)
        q_current = data.qpos[0:6]
        v_current = data.qvel[0:6]

        # 2. Compute PD Control Equation: τ = Kp * (error) - Kd * (velocity)
        torques = Kp * (q_target - q_current) - Kd * v_current

        # 3. Apply calculated control efforts to the motor actuators
        data.ctrl[0:6] = torques

        # 4. Step the physics engine forward
        mujoco.mj_step(model, data)

        # 5. Sync frames to update your active GUI window 
        viewer.sync()

        # 6. Ensure real-time playback speed matching
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
# Define a stable test pose (in Radians) so it moves away from its zero-line
# Let's flex the shoulder and elbow slightly forward
q_target = np.array([0.0, -0.6, 1.2, 0.0, 0.5, 0.0])

print("🚀 Launching HEAL standalone controller...")
with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():
        step_start = time.time()

        # 1. Read current angles (qpos) and angular speeds (qvel) for all 6 joints
        q_current = data.qpos[0:6]
        v_current = data.qvel[0:6]

        # 2. Compute PD Control Equation: τ = Kp * (error) - Kd * (velocity)
        torques = Kp * (q_target - q_current) - Kd * v_current

        # 3. Apply calculated control efforts to the motor actuators
        data.ctrl[0:6] = torques

        # 4. Step the physics engine forward
        mujoco.mj_step(model, data)

        # 5. Sync frames to update your active GUI window 
        viewer.sync()
# Define a stable test pose (in Radians) so it moves away from its zero-line
# Let's flex the shoulder and elbow slightly forward
q_target = np.array([0.0, -0.6, 1.2, 0.0, 0.5, 0.0])

print("🚀 Launching HEAL standalone controller...")
with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():
        step_start = time.time()

        # 1. Read current angles (qpos) and angular speeds (qvel) for all 6 joints
        q_current = data.qpos[0:6]
        v_current = data.qvel[0:6]

        # 2. Compute PD Control Equation: τ = Kp * (error) - Kd * (velocity)
        torques = Kp * (q_target - q_current) - Kd * v_current

        # 3. Apply calculated control efforts to the motor actuators
        data.ctrl[0:6] = torques

        # 4. Step the physics engine forward
        mujoco.mj_step(model, data)

        # 5. Sync frames to update your active GUI window 
        viewer.sync()

        # 6. Ensure real-time playback speed matching
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        # 6. Ensure real-time playback speed matching
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
