import subprocess


if __name__ == '__main__':
    # Grab the intensity image and depth image from ToF camera.
    subprocess.run(
        ['python',
         'tof_data_grab.py',
         '--intensity',
         'result_tof_data/intensity.png',
         '--depth',
         'result_tof_data/depth',
         '--heatmap',
         'result_tof_data/heatmap.png',]
    )
    subprocess.run(['python', 'tof_data_grab.py'])