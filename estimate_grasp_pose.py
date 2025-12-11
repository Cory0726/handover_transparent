import json
import numpy as np
import math


def pose_to_matrix(x, y, z, rx, ry, rz):
    """
    Convert 6-DoF pose (x, y, z, rx, ry, rz) into a 4x4 homogeneous
    transformation matrix.

    Parameters
    ----------
    x, y, z : float
        Translation components (in meters or robot units).
    rx, ry, rz : float
        Rotation angles in degrees.
        Rotation convention: R = Rz * Ry * Rx (ZYX Euler angles).

    Returns
    -------
    T : (4, 4) ndarray
        Homogeneous transformation matrix.
    """
    # Convert degrees to radians
    rx, ry, rz = map(np.deg2rad, [rx, ry, rz])

    # Rotation around X-axis
    Rx = np.array([
        [1,          0,           0],
        [0,  np.cos(rx), -np.sin(rx)],
        [0,  np.sin(rx),  np.cos(rx)]
    ], dtype=float)

    # Rotation around Y-axis
    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [ 0,          1,         0],
        [-np.sin(ry), 0, np.cos(ry)]
    ], dtype=float)

    # Rotation around Z-axis
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0,                   0, 1]
    ], dtype=float)

    # Rotation order: R = Rz * Ry * Rx  (ZYX / yaw-pitch-roll)
    R = Rz @ Ry @ Rx

    # Build 4x4 homogeneous transform
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    return T

def matrix_to_pose(T):
    """
    Convert a 4x4 homogeneous transformation matrix into a 6-DoF pose
    (x, y, z, rx, ry, rz) with ZYX Euler angles.

    Parameters
    ----------
    T : (4, 4) ndarray
        Homogeneous transformation matrix.

    Returns
    -------
    x, y, z : float
        Translation components.
    rx, ry, rz : float
        Rotation angles in degrees.
        Rotation convention: R = Rz * Ry * Rx (ZYX Euler angles).
    """
    # Translation
    x, y, z = T[0, 3], T[1, 3], T[2, 3]

    # Rotation matrix
    R = T[:3, :3]

    # Recover Euler angles for R = Rz * Ry * Rx (ZYX)
    # ry = -asin(R[2, 0])
    ry = -np.arcsin(R[2, 0])
    cy = np.cos(ry)

    if abs(cy) > 1e-6:
        # Normal case (no gimbal lock)
        rx = np.arctan2(R[2, 1] / cy, R[2, 2] / cy)
        rz = np.arctan2(R[1, 0] / cy, R[0, 0] / cy)
    else:
        # Gimbal lock case: cos(ry) ~ 0
        rx = 0.0
        rz = np.arctan2(-R[0, 1], R[1, 1])

    # Convert radians back to degrees
    rx, ry, rz = map(np.rad2deg, [rx, ry, rz])

    return [x, y, z, rx, ry, rz]

def invert_transform(T):
    """
    Invert a 4x4 rigid transform matrix.
    T = [[R, t],
         [0, 1]]
    return T_inv = [[R.T, -R.T @ t],
                    [0,    1        ]]
    """
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    return T_inv

def get_T_cam_grasp(cam_matrix_file, grasp_result_file):
    # Load the tof camera matrix
    with open(cam_matrix_file, 'r') as f:
        tof_cam_data = json.load(f)
    fx = tof_cam_data['fx']  # unit: pixel
    fy = tof_cam_data['fy']
    cx = tof_cam_data['cx']
    cy = tof_cam_data['cy']
    # Load the grasp point from result of GR-ConvNet
    with open(grasp_result_file, 'r') as f:
        grasp_result_data = json.load(f)
    v, u = grasp_result_data['center']  # unit: pixel
    z = grasp_result_data['depth']  # unit: mm
    angle = math.degrees(grasp_result_data['angle'])  # unit: degree
    # width = grasp_result_data['width']  # unit: mm
    # Transform the grasp point into tof camera frame
    cam_x = (u - cx) * z / fx  # unit: mm
    cam_y = (v - cy) * z / fy
    cam_z = z
    # # Generate the transformation metrix
    return pose_to_matrix(cam_x, cam_y, cam_z, 0, 0, angle)

def get_T_flange_cam(file):
    # Load the file
    with open(file, 'r') as f:
        flange_in_cam = json.load(f)
    x = flange_in_cam['x']  * 1000  # unit: mm
    y = flange_in_cam['y'] * 1000
    z = flange_in_cam['z'] * 1000
    rx = flange_in_cam['rx']  # unit: deg
    ry = flange_in_cam['ry']
    rz = flange_in_cam['rz']
    # Create the transformation matrix
    return invert_transform(pose_to_matrix(x, y, z, rx, ry, rz))

def get_grasp_pose():
    # File path
    file_tof_cam_flange_t = 'data/tof_cam_flange_transfer.json'
    file_cam_matrix = 'data/tof_cam_matrix.json'
    file_grasp_result = 'data/grconv_grasp_result.json'
    # Set grasp offset (gripper length)
    tool_size = 220  # unit: mm
    # Estimate grasp pose
    T_flage_grasp = get_T_flange_cam(file_tof_cam_flange_t) @ get_T_cam_grasp(file_cam_matrix, file_grasp_result)
    pose_flange_grasp = matrix_to_pose(T_flage_grasp)
    pose_flange_grasp[2] = pose_flange_grasp[2] - tool_size
    # Save the grasp pose as .json
    grasp_pose_dict = {
        'x': pose_flange_grasp[0],
        'y': pose_flange_grasp[1],
        'z': pose_flange_grasp[2],
        'rx': pose_flange_grasp[3],
        'ry': pose_flange_grasp[4],
        'rz': pose_flange_grasp[5]
    }
    with open('data/grasp_estimate_pose.json', 'w') as f:
        json.dump(grasp_pose_dict, f, indent=4)

    return pose_flange_grasp


if __name__ == '__main__':
    grasp_point = get_grasp_pose()
    print(grasp_point)