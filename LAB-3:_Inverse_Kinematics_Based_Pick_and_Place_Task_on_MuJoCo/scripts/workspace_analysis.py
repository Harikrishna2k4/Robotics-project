import numpy as np
import mujoco
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ==========================================
# 1. LOAD MUJOCO MODEL & SETUP CONFIG
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # This points to your Lab_3 folder

# Define the absolute paths to your assets
model_path = os.path.join(PROJECT_DIR, "robot_descriptions", "ur5.xml")
mesh_dir = os.path.join(PROJECT_DIR, "robot_descriptions")

# 1. Read the raw XML content as a text string
with open(model_path, "r") as f:
    xml_text = f.read()

# 2. Dynamically scan your directory to bundle all meshes into a virtual memory dictionary
assets = {}
for root, dirs, files in os.walk(mesh_dir):
    for file in files:
        if file.endswith(('.stl', '.png', '.xml')):
            full_path = os.path.join(root, file)
            # Create relative keys matching how your XML addresses them
            rel_path_1 = os.path.relpath(full_path, PROJECT_DIR)                        # e.g. robot_descriptions/ur5/link1.stl
            rel_path_2 = os.path.relpath(full_path, os.path.join(mesh_dir, "ur5"))       # e.g. link1.stl
            rel_path_3 = f"../robot_descriptions/ur5/{file}"                             # e.g. legacy relative paths
            
            with open(full_path, "rb") as asset_file:
                content = asset_file.read()
                assets[file] = content
                assets[rel_path_1] = content
                assets[rel_path_2] = content
                assets[rel_path_3] = content

# 3. Change directory to Lab_3 so baseline relative lookups are clean
os.chdir(PROJECT_DIR)

# 4. Compile directly from the string using the fully resolved virtual asset map
model = mujoco.MjModel.from_xml_string(xml_text, assets=assets)
data = mujoco.MjData(model)

print("SUCCESS: MuJoCo model and all STL meshes compiled perfectly!")

# Define your Tool Center Point (TCP) site name from your XML.
# Fallback strategy if 'attachment_site' isn't explicitly defined in the XML.
try:
    tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if tcp_site_id == -1:
        raise ValueError
except:
    # Try looking for common alternatives in default MuJoCo scenes (e.g., 'pinch', 'wrist_3_link', 'tool')
    for alternative in ["pinch", "tool", "wrist_3_link"]:
        tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, alternative)
        if tcp_site_id != -1:
            print(f"Tracking TCP via site: '{alternative}'")
            break
    else:
        # Fallback to index 0 if absolutely no name matches
        tcp_site_id = 0
        print("Warning: Could not find named site. Defaulting to site index 0.")

# Define grid bounds covering your robotic station table surface
x_range = np.linspace(-0.8, 0.8, 40)
y_range = np.linspace(-0.8, 0.8, 40)
z_range = np.linspace(0.0, 0.6, 25) # Table height to max vertical reach

task_workspace = []
dexterous_workspace = []

# ==========================================
# 2. NUMERICAL INVERSE KINEMATICS SOLVER
# ==========================================
def check_ik_feasibility(target_pos, target_quat=None):
    """
    Attempts to solve numerical inverse kinematics for a target position.
    If target_quat is provided, it checks if that specific orientation is reachable.
    """
    # Reset to a neutral starting joint configuration
    data.qpos[:] = 0
    mujoco.mj_forward(model, data)
    
    # Simple Newton-Raphson IK loop
    max_steps = 100
    tol = 1e-4
    for _ in range(max_steps):
        mujoco.mj_forward(model, data)
        
        # Get current TCP position
        current_pos = data.site_xpos[tcp_site_id]
        pos_err = target_pos - current_pos
        
        # Handle orientation error if computing dexterous workspace
        if target_quat is not None:
            current_quat = np.zeros(4)
            mujoco.mj_mat2Quat(current_quat, data.site_xmat[tcp_site_id])
            # Quaternion mathematical alignment check
            quat_err = target_quat[1:] * current_quat[0] - current_quat[1:] * target_quat[0] + np.cross(current_quat[1:], target_quat[1:])
            error = np.concatenate([pos_err, quat_err])
        else:
            error = pos_err
            
        if np.linalg.norm(error) < tol:
            # Check joint position limits to ensure it's truly feasible
            for i in range(model.njnt):
                if data.qpos[i] < model.jnt_range[i][0] or data.qpos[i] > model.jnt_range[i][1]:
                    return False
            return True
            
        # Get Jacobian matrix
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, tcp_site_id)
        
        J = jacp if target_quat is None else np.vstack([jacp, jacr])
        
        # Compute joint update using pseudo-inverse
        dq = np.linalg.pinv(J) @ error
        data.qpos[:] += dq
        
    return False

# ==========================================
# 3. WORKSPACE SCANNING LOOP
# ==========================================
print("Scanning workspace points... Please wait.")
for x in x_range:
    for y in y_range:
        for z in z_range:
            point = np.array([x, y, z])
            
            # A. Check Task Workspace (At least one orientation works)
            if check_ik_feasibility(point):
                task_workspace.append(point)
                
                # B. Check Dexterous Workspace (Feasible across 4 arbitrary Yaw angles)
                yaw_angles = [0, np.pi/2, np.pi, -np.pi/2]
                dexterous_valid = True
                for yaw in yaw_angles:
                    # Convert target yaw to quaternion format [w, x, y, z]
                    target_quat = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
                    if not check_ik_feasibility(point, target_quat):
                        dexterous_valid = False
                        break
                
                if dexterous_valid:
                    dexterous_workspace.append(point)

task_workspace = np.array(task_workspace)
dexterous_workspace = np.array(dexterous_workspace)

# ==========================================
# 4. PLOTTING & VISUALIZATION
# ==========================================
fig = plt.figure(figsize=(12, 5))

# Plot 1: 3D Point Cloud Representation
ax1 = fig.add_subplot(121, projection='3d')
if len(task_workspace) > 0:
    ax1.scatter(task_workspace[:,0], task_workspace[:,1], task_workspace[:,2], c='lightblue', alpha=0.3, label='Task Workspace', s=2)
if len(dexterous_workspace) > 0:
    ax1.scatter(dexterous_workspace[:,0], dexterous_workspace[:,1], dexterous_workspace[:,2], c='darkblue', alpha=0.8, label='Dexterous Workspace', s=5)
ax1.set_title("3D Reachable Workspaces")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.set_zlabel("Z (m)")
ax1.legend()

# Plot 2: 2D Table Surface Occupancy Grid Map (Slice at Z = 0.1m above table)
ax2 = fig.add_subplot(122)
z_slice_height = 0.1
if len(task_workspace) > 0:
    task_slice = task_workspace[np.abs(task_workspace[:,2] - z_slice_height) < 0.02]
    ax2.scatter(task_slice[:,0], task_slice[:,1], c='lightcoral', label='Feasible Task Region', s=15)
if len(dexterous_workspace) > 0:
    dex_slice = dexterous_workspace[np.abs(dexterous_workspace[:,2] - z_slice_height) < 0.02]
    ax2.scatter(dex_slice[:,0], dex_slice[:,1], c='crimson', label='Feasible Dexterous Region', s=25)

ax2.set_title(f"Feasible Table Work Regions (Slice at Z = {z_slice_height}m)")
ax2.set_xlabel("X (m)")
ax2.set_ylabel("Y (m)")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
output_image = os.path.join(PROJECT_DIR, "scripts", "workspace_analysis_results.png")
plt.savefig(output_image)
plt.show()
print(f"Analysis complete! Results successfully saved to: {output_image}")
