from pathlib import Path
import time
import mujoco
import mujoco.viewer
import numpy as np

# --------------------------------------------------
# Find XML relative to this Python file
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
XML_PATH = PROJECT_ROOT / "robot_descriptions" / "franka_scene.xml"

# --------------------------------------------------
# Load MuJoCo model
# --------------------------------------------------
model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)

print("==================================================")
print("Franka 7-DOF Pick-and-Place Sequence Operational")
print("==================================================")

# --------------------------------------------------
# Define Proportional-Derivative (PD) Gains for Torque
# --------------------------------------------------
Kp_franka = np.array([600.0, 800.0, 600.0, 500.0, 250.0, 150.0, 50.0])
Kd_franka = np.array([50.0,  70.0,  50.0,  40.0,  20.0,  10.0,  5.0])

# --------------------------------------------------
# Define Key Poses (Joint Angles in Radians for 7 Joints)
# --------------------------------------------------
# These pre-computed joint configurations align the arm positions over your table
POSE_HOME        = np.array([0.0,  -0.785,  0.0,   -2.356,  0.0,   1.571,  0.785])
POSE_HOVER_BOX   = np.array([0.52,  0.22,  -0.34,  -2.10,   0.02,  2.32,   0.785])
POSE_GRASP_BOX   = np.array([0.52,  0.34,  -0.34,  -1.92,   0.02,  2.26,   0.785])
POSE_LIFT_BOX    = np.array([0.52,  0.10,  -0.34,  -2.20,   0.02,  2.30,   0.785])
POSE_HOVER_DROP  = np.array([-0.40, 0.22,   0.25,  -2.10,  -0.05,  2.32,   0.785])
POSE_DROP_BOX    = np.array([-0.40, 0.34,   0.25,  -1.92,  -0.05,  2.26,   0.785])

# --------------------------------------------------
# Pick and Place State Machine Setup
# --------------------------------------------------
# Each state defines: (target_joint_pose, gripper_command, duration_in_seconds)
# Gripper command: 0.04 = Open fully, -20.0 = High torque close grasp
states = [
    {"pose": POSE_HOME,       "gripper": 0.04,  "duration": 2.0, "desc": "1. Resetting to Home Pose"},
    {"pose": POSE_HOVER_BOX,  "gripper": 0.04,  "duration": 2.5, "desc": "2. Hovering above Red Box"},
    {"pose": POSE_GRASP_BOX,  "gripper": 0.04,  "duration": 1.5, "desc": "3. Descending to Box level"},
    {"pose": POSE_GRASP_BOX,  "gripper": -20.0, "duration": 1.0, "desc": "4. Activating Gripper Grasp"},
    {"pose": POSE_LIFT_BOX,   "gripper": -20.0, "duration": 1.5, "desc": "5. Lifting Box off Table"},
    {"pose": POSE_HOVER_DROP, "gripper": -20.0, "duration": 2.5, "desc": "6. Moving to Destination Zone"},
    {"pose": POSE_DROP_BOX,   "gripper": -20.0, "duration": 1.5, "desc": "7. Descending to Delivery Level"},
    {"pose": POSE_DROP_BOX,   "gripper": 0.04,  "duration": 1.0, "desc": "8. Releasing Gripper Fingers"},
    {"pose": POSE_HOVER_DROP, "gripper": 0.04,  "duration": 1.5, "desc": "9. Retracting Arm upward"},
    {"pose": POSE_HOME,       "gripper": 0.04,  "duration": 2.0, "desc": "10. Task Complete - Returning Home"}
]

# Set start frame configuration
data.qpos[0:7] = POSE_HOME
mujoco.mj_forward(model, data)

# --------------------------------------------------
# Start Simulation Loop
# --------------------------------------------------
print("\n🚀 Launching Pick and Place Sequence Automation...")
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    current_state_idx = 0
    state_start_time = time.time()
    print(f"\n👉 Current Action: {states[current_state_idx]['desc']}")

    while viewer.is_running():
        step_start = time.time()
        elapsed_in_state = time.time() - state_start_time

        # Fetch parameters for the active state step
        current_state = states[current_state_idx]
        q_start = POSE_HOME if current_state_idx == 0 else states[current_state_idx - 1]["pose"]
        q_target_pose = current_state["pose"]
        gripper_cmd = current_state["gripper"]
        duration = current_state["duration"]

        # --- Smooth Trajectory Interpolation ---
        blend = min(1.0, elapsed_in_state / duration)
        s = 3 * (blend ** 2) - 2 * (blend ** 3)  # Cubic ease-in-out formula
        q_target = (1.0 - s) * q_start + s * q_target_pose

        # 1. Read live states for the 7 arm joints
        q_current = data.qpos[0:7]
        v_current = data.qvel[0:7]

        # 2. Compute PD Torques: τ = Kp * (error) - Kd * (velocity)
        torques = Kp_franka * (q_target - q_current) - Kd_franka * v_current

        # 3. Inject calculated torques into the first 7 motor actuators
        data.ctrl[0:7] = torques
        
        # 4. Command the general gripper actuator (index 7)
        # Uses the built-in <general> template properties from your XML script
        if model.nu > 7:
            data.ctrl[7] = gripper_cmd

        # Advance physics simulation step
        mujoco.mj_step(model, data)
        viewer.sync()

        # --- State Machine Transition Verification ---
        if elapsed_in_state >= duration:
            current_state_idx += 1
            if current_state_idx >= len(states):
                print("\n🏁 Sequence fully executed. Restarting sequence loop...")
                current_state_idx = 0  # Loop back indefinitely for visualization
            
            state_start_time = time.time()
            print(f"👉 Current Action: {states[current_state_idx]['desc']}")

        # Lock loop cycles to real-world simulation step time bounds
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
