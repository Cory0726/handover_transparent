import json
import numpy as np

def main():
    # Load the tof camera matrix
    with open('data/tof_cam_matrix.json', 'r') as f:
        tof_cam_data = json.load(f)
    fx = tof_cam_data['fx']
    fy = tof_cam_data['fy']
    cx = tof_cam_data['cx']
    cy = tof_cam_data['cy']
    # Load the grasp point from result of GR-ConvNet
    with open('data/grconv_grasp_result.json', 'r') as f:
        grasp_result_data = json.load(f)
    v, u = grasp_result_data['center']
    z = grasp_result_data['depth']
    angle = grasp_result_data['angle']
    width = grasp_result_data['width']
    print(fx, fy, cx, cy, u, v, z, angle, width)
    # Transform the grasp point into tof camera frame
    cam_x = (u - cx) * z / fx
    cam_y = (v - cy) * z / fy
    cam_z = z
    cam_p = np.array(
        [[cam_x],
        [cam_y],
        [cam_z],
        [1]]
    )
    print(cam_p)

if __name__ == '__main__':
    main()