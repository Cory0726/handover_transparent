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
         '--output', 'data/unet_hand_mask.png'], check=True
    )

def run_da3_predict_process():
    subprocess.run(
        ['python', 'da3/predict.py',
         '--img', 'data/tof_intensity_grayscale.png',
         '--depth', 'data/tof_depth.npy',
         '--hand', 'data/unet_hand_mask.png'], check=True
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

    # ==================================================
    # Depth-Anything-3
    # ==================================================
    # logging.info('Running DA3 model...')
    # run_da3_predict_process()

    # ==================================================
    # TransCG
    # ==================================================
    # logging.info('Running TransCG model...')
    # run_transcg_predict_process()

    # ==================================================
    # GR-ConvNet
    # ==================================================
    # logging.info('Running GR-ConvNet model...')
    # run_gr_convnet_predict_process()

if __name__ == '__main__':
    main()
