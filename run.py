import sys
import subprocess
import logging
import time
import cv2
import numpy as np
from estimate_grasp_pose import get_grasp_pose
from techman_tools.robot_control import TMRobot, show_pose

def run_tof_data_grab_process():
    subprocess.run(
        ['python', 'tof_cam/tof_data_grab.py',
         '--intensity', 'data/tof_intensity.png',
         '--depth', 'data/tof_depth',
         '--heatmap', 'data/tof_depth_heatmap.png'], check=True
    )

def run_rgb_data_grab_process():
    subprocess.run(
        ['python', 'rgb_cam/rgb_data_grab.py',
         '--rgb', 'data/rgb_img.png'], check=True
    )

def save_intensity_to_grayscale(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    output_img_path = 'data/tof_intensity_grayscale.png'
    cv2.imwrite(output_img_path, img)
    print(f'Saved {output_img_path}')

def run_unet_predict_process():
    subprocess.run(
        ['python', 'u_net/predict.py',
         # '--input', 'data/tof_intensity_grayscale.png',
         '--input', 'data/temp_test.png',
         '--output', 'data/unet_hand_mask.png',
         '--model', 'u_net/Hand_Seg_HandWithArms_S480480G_F0240_S1_Ep36_Score090095_20251220.pth'], check=True
    )

def run_da3_predict_process():
    subprocess.run(
        ['python', 'da3/predict.py',
         '--img', 'data/tof_intensity_grayscale.png',
         '--depth', 'data/tof_depth.npy',
         '--hand', 'data/unet_hand_mask.png',
         '--cam-matrix', 'data/tof_cam_matrix.json'], check=True
    )

def run_transcg_predict_process():
    subprocess.run(
        ['python', 'transcg/predict.py',
         '--img', 'data/tof_intensity_grayscale.png',
         '--depth', 'data/da3_cal_depth.npy'], check=True
    )

def run_gr_convnet_predict_process():
    subprocess.run(
        ['python', 'gr_convnet/predict.py',
         '--depth', 'data/da3_cal_depth.npy',
         '--mask', 'data/unet_hand_mask.png'], check=True
    )

def main():
    # Initialize the log.
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    # Wait 5 second for start
    logging.info('Wait 5 seconds for start...')
    time.sleep(5)  #  unit : second

    start_time = time.time()

    # Grab ToF and RGB Camera data
    logging.info('Grabbing Image data...')
    run_tof_data_grab_process()
    run_rgb_data_grab_process()
    save_intensity_to_grayscale('data/tof_intensity.png')  # Grayscale

    # U-Net hand segmentation
    logging.info('Running U-Net model...')
    run_unet_predict_process()

    # # Depth-Anything-3
    # logging.info('Running DA3 model...')
    # run_da3_predict_process()
    #
    # # TransCG
    # # logging.info('Running TransCG model...')
    # # run_transcg_predict_process()
    #
    # # GR-ConvNet
    # logging.info('Running GR-ConvNet model...')
    # run_gr_convnet_predict_process()
    #
    # # Get the grasp point pose (Pick point)
    # logging.info('Start grasping...')
    # grasp_pose = get_grasp_pose()
    # show_pose('Grasp Pose', grasp_pose)
    # # Set the place point
    # origin_point = [-400.1218, 12.36882, 636.417, -176.5101, 51.12951, 19.41987]
    #
    # # Execute grasping
    # # tmrobot = TMRobot('192.168.50.49')
    # # tmrobot.pick_and_place(pick_point=grasp_pose, place_point=origin_point)
    # end_time = time.time()
    # logging.info(f'Finished in {end_time - start_time} seconds')

def reset_robot_state():
    robot = TMRobot('192.168.50.49')
    robot.move2origin()
    robot.gripper_open()


if __name__ == '__main__':
    run_unet_predict_process()