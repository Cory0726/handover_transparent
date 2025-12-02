import subprocess


if __name__ == '__main__':
    # Grab the intensity image and depth image from ToF camera.
    subprocess.run(
        ['python',
         'tof_data_grab.py',
         '--intensity',
         'intensity.png',
         '--depth',
         'depth',
         '--heatmap',
         'heatmap.png',]
    )
    subprocess.run(['python', 'tof_data_grab.py'])