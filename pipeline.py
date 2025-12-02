import subprocess
import logging


def run_tof_data_grab():
    subprocess.run(
        ['python',
         'tof_data_grab.py',
         '--intensity',
         'result_tof_data/intensity.png',
         '--depth',
         'result_tof_data/depth',
         '--heatmap',
         'result_tof_data/heatmap.png', ], check=True
    )

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('Grabbing ToF data')
    # Grab the intensity image and depth image from ToF camera.
    run_tof_data_grab()
    logging.info('Saved ToF data')
