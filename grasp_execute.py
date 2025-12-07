import json

import cv2
import numpy as np

def main():
    # Load the tof camera matrix
    with open('data/tof_cam_matrix.json', 'r') as f:
        tof_cam_data = json.load(f)
    fx = tof_cam_data['fx']  # unit: pixel
    fy = tof_cam_data['fy']
    cx = tof_cam_data['cx']
    cy = tof_cam_data['cy']
    # Load the grasp point from result of GR-ConvNet
    with open('data/grconv_grasp_result.json', 'r') as f:
        grasp_result_data = json.load(f)
    v, u = grasp_result_data['center']  # unit: pixel
    z = grasp_result_data['depth']  # unit: mm
    angle = grasp_result_data['angle']  # unit: rad
    width = grasp_result_data['width']  # unit: mm
    print(fx, fy, cx, cy, u, v, z, angle, width)
    # Transform the grasp point into tof camera frame
    cam_x = (u - cx) * z / fx  # unit: mm
    cam_y = (v - cy) * z / fy
    cam_z = z
    cam_p = np.array(
        [cam_x,
        cam_y,
        cam_z,
        1]
    )
    print(cam_p)
    with open('data/tof_cam_flange_transfer.json', 'r') as f:
        flange_in_cam = json.load(f)
    x = flange_in_cam['x']  # unit: m
    y = flange_in_cam['y']
    z = flange_in_cam['z']
    rx = flange_in_cam['rx']  # unit: deg
    ry = flange_in_cam['ry']
    rz = flange_in_cam['rz']
    print(x, y, z, rx, ry, rz)
    r_deg = np.array([rx, ry, rz], dtype=np.float32)
    t = np.array([x, y, z], dtype=np.float32)
    r_rad = np.deg2rad(r_deg)
    R_cam_flange, _ = cv2.Rodrigues(r_rad)  # 3x3 matrix
    T_cam_flange = np.eye(4, dtype=np.float32)
    T_cam_flange[:3, :3] = R_cam_flange
    T_cam_flange[:3, 3] = t

    print(T_cam_flange)  # unit: m, rad

    T_flange_cam = np.linalg.inv(T_cam_flange)
    print(T_flange_cam)
if __name__ == '__main__':
    main()