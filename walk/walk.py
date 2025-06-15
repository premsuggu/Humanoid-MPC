import pinocchio
import numpy as np
import crocoddyl
import os
from robot_gait_generator.crocoddyl import CrocoddylGaitProblemsInterface
import csv
import pinocchio.visualize
from scipy.spatial.transform import Rotation as R

os.environ["ROS_PACKAGE_PATH"] = "/home/prem/wb_mpc/robots:" + os.environ.get("ROS_PACKAGE_PATH", "")
# Path to your URDF
urdf_path = "/home/prem/wb_mpc/robots/h1_description/urdf/h1.urdf"

# Load the robot model
model = pinocchio.buildModelFromUrdf(urdf_path, pinocchio.JointModelFreeFlyer())

print(f"model.nq: {model.nq}")
print(f"model.nv: {model.nv}")
# Initial state (neutral pose)
q0 = pinocchio.neutral(model)
x0 = np.concatenate([q0, np.zeros(model.nv)])

# Foot link names as in your URDF
LEG_ORDER = ["left_ankle_link", "right_ankle_link"] 

# Gait parameters for bipedal walking
GAIT_PARAMETERS = {
    "step_frequencies": [1.0, 1.0],
    "duty_cycles": [0.7, 0.7],
    "phase_offsets": [0.0, 0.5],
    "relative_feet_targets": [[0.15, 0.0, 0.0], [0.15, 0.0, 0.0]],  # stride length for each foot
    "foot_lift_height": [0.05, 0.05],
}

# Instantiate the gait problem interface
gait_problem_interface = CrocoddylGaitProblemsInterface(
    pinocchio_robot_model=model,
    default_standing_configuration=q0,
    ee_names=LEG_ORDER,
)

# Create the Crocoddyl shooting problem for walking gait
problem = gait_problem_interface.create_generic_gait_problem(
    x0=x0,
    starting_feet_heights=[0.0, 0.0],
    duration=5.0,                       #Duration
    time_step=0.05,                     #Time Step
    **GAIT_PARAMETERS
)

print(f"created problem successfully for time steps {len(problem.runningModels)}")

# Solve for the trajectory using DDP
solver = crocoddyl.SolverFDDP(problem)
solver.solve([], [], 100)

# Extract reference trajectories
reference_trajectory = solver.xs
reference_controls = solver.us

#------------------------------------------------------------------------------------
# duration = 10.0
dt = 0.05
num_steps = 119
step_frequencies = [1.0, 1.0]
duty_cycles = [0.7, 0.7]
phase_offsets = [0.0, 0.5]  # left, right

contact_schedule = []
for t in range(num_steps):
    time = t * dt
    contacts = []
    for freq, duty, phase_offset in zip(step_frequencies, duty_cycles, phase_offsets):
        phase = ((freq * time + phase_offset) % 1.0)
        in_contact = phase < duty
        contacts.append(in_contact)
    contact_schedule.append(contacts)

print("Contact schedule (per time step, per foot):")
for t, contacts in enumerate(contact_schedule):
    print(f"Step {t}: {{'left': {contacts[0]}, 'right': {contacts[1]}}}")

#-------------------------Visualize----------------------------------

# Load the visual model (if you have a mesh directory)
mesh_dir = "/home/prem/wb_mpc/robots/h1_description/meshes"  # or set to your mesh directory if you have visuals
model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(urdf_path, mesh_dir, pinocchio.JointModelFreeFlyer())

viz = pinocchio.visualize.MeshcatVisualizer(model, collision_model, visual_model)

# Initialize the viewer
viz.initViewer(open=True)
viz.loadViewerModel()

# Animate the trajectory
import time

for x in reference_trajectory:
    viz.display(x[:model.nq])
    time.sleep(0.02)  # Adjust for smoother/faster animation


#------------- Save Files ---------------------
# 1. Save q_ref and v_ref (N+1 x nq/nv)
data = model.createData()
q_ref = np.array([x[:model.nq] for x in reference_trajectory])
v_ref = np.array([x[model.nq:] for x in reference_trajectory])

np.savetxt("q_ref.csv", q_ref, delimiter=',')
np.savetxt("v_ref.csv", v_ref, delimiter=',')

# 2. Save a_ref (N x nv)
a_ref = np.diff(v_ref, axis=0) / dt  # finite difference
np.savetxt("a_ref.csv", a_ref, delimiter=',')

# 3. Save contact_schedule (N x n_ee)
contact_schedule_np = np.array(contact_schedule, dtype=int)  # or bool
np.savetxt("contact_schedule.csv", contact_schedule_np, fmt='%d', delimiter=',')

# 4. Save end-effector positions/orientations if available
# (You need to compute these using forward kinematics for each q_ref)
# Example for positions:
ee_names = ["left_ankle_link", "right_ankle_link"]  # or your actual foot frame names
ee_pos_ref = []
for q in q_ref:
    pinocchio.forwardKinematics(model, data, q)
    pinocchio.updateFramePlacements(model, data)
    ee_pos_ref.append([data.oMf[model.getFrameId(name)].translation for name in ee_names])
ee_pos_ref = np.array(ee_pos_ref)  # shape (N+1, n_ee, 3)
# Save as flattened CSV for each timestep
ee_pos_ref_flat = ee_pos_ref.reshape((ee_pos_ref.shape[0], -1))
np.savetxt("ee_pos_ref.csv", ee_pos_ref_flat, delimiter=',')

# EE orientations
ee_ori_ref = []
for q in q_ref:
    pinocchio.forwardKinematics(model, data, q)
    pinocchio.updateFramePlacements(model, data)
    quats = []
    for name in ee_names:
        rot = data.oMf[model.getFrameId(name)].rotation
        quat = R.from_matrix(rot).as_quat()  # [x, y, z, w]
        # Reorder to [w, x, y, z] for consistency with C++
        quat_wxyz = np.roll(quat, 1)
        quats.append(quat_wxyz)
    ee_ori_ref.append(np.concatenate(quats))
ee_ori_ref = np.array(ee_ori_ref)  # shape (N+1, n_ee*4)
np.savetxt("ee_ori_ref.csv", ee_ori_ref, delimiter=',')

# 5. Save CoM and ZMP references
com_ref = np.array([pinocchio.centerOfMass(model, data, q) for q in q_ref]).squeeze()
np.savetxt("com_ref.csv", com_ref, delimiter=',')

# 6. ZMP formula: ZMP = CoM_xy - (CoM_z / (CoM_ddot_z + g)) * CoM_ddot_xy
g = 9.81
zmp_ref = []
for k in range(len(a_ref)):
    q = q_ref[k]
    v = v_ref[k]
    a = a_ref[k]
    pinocchio.forwardKinematics(model, data, q, v, a)
    pinocchio.centerOfMass(model, data, q, v, a)
    com = data.com[0].copy()
    com_acc = data.vcom[0].copy()  # This is actually velocity, so you may need to use data.acom[0] if available
    if hasattr(data, 'acom'):
        com_ddot = data.acom[0].copy()
    else:
        # Fallback: finite difference (less accurate)
        if k < len(a_ref)-1:
            com_ddot = (data.vcom[0] - prev_vcom) / dt
        else:
            com_ddot = np.zeros(3)
    prev_vcom = data.vcom[0].copy()
    denom = com_ddot[2] + g if abs(com_ddot[2] + g) > 1e-4 else 1e-4
    zmp_x = com[0] - (com[2] / denom) * com_ddot[0]
    zmp_y = com[1] - (com[2] / denom) * com_ddot[1]
    zmp_ref.append([zmp_x, zmp_y])
zmp_ref = np.array(zmp_ref)
np.savetxt("zmp_ref.csv", zmp_ref, delimiter=',')
