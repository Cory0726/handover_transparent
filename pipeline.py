import sys
import subprocess
import logging
import cv2


def run_tof_data_grab_process():
    subprocess.run(
        ['python', 'tof_cam/tof_data_grab.py',
         '--intensity', 'data/tof_intensity.png',
         '--depth', 'data/tof_depth',
         '--heatmap', 'data/tof_depth_heatmap.png'], check=True
    )

def save_intensity_to_grayscale(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    output_img_path = 'data/tof_intensity_grayscale.png'
    cv2.imwrite(output_img_path, img)
    print(f'Saved {output_img_path}')

def run_unet_predict_process():
    subprocess.run(
        ['python', 'u_net/predict.py',
         '--input', 'data/tof_intensity_grayscale.png',
         '--output', 'data/unet_hand_mask.png',
         '--model', 'u_net/Hand_Seg_EGTEA_plus_S640480G_Scale05_Score08994_20251123.pth',
         '--process', 'u_net/img_process_temp'], check=True
    )

if __name__ == '__main__':
    # Initialize the log.
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    # ==================================================
    # ToF intensity and depth data
    # ==================================================
    # Grab the intensity image and depth image from ToF camera.
    # logging.info('Grabbing ToF data...')
    # run_tof_data_grab_process()
    # Save intensity image to grayscale image.
    # save_intensity_to_grayscale('data/tof_intensity.png')

    # ==================================================
    # U-Net hand segmentation
    # ==================================================
    logging.info('Running U-Net model...')
    run_unet_predict_process()

