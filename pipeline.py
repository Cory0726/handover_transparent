import subprocess
import logging

import cv2


def run_tof_data_grab_process():
    subprocess.run(
        ['python', 'tof_data_grab.py',
         '--intensity', 'data/tof_intensity.png',
         '--depth', 'data/tof_depth',
         '--heatmap', 'data/tof_depth_heatmap.png', ], check=True
    )

def save_intensity_to_grayscale(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('data/tof_intensity_grayscale.png', img)


if __name__ == '__main__':
    # Initialize the log.
    logging.basicConfig(level=logging.INFO)

    # ==================================================
    # ToF intensity and depth data
    # ==================================================
    # Grab the intensity image and depth image from ToF camera.
    logging.info('Grabbing ToF data...')
    run_tof_data_grab_process()
    # Save intensity image to grayscale image.
    save_intensity_to_grayscale('data/tof_intensity.png')
    print('Saved data/tof_intensity_grayscale.png')

    
