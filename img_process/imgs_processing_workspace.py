import os
import shutil
from pathlib import Path
import tools_box
from matchcheck_imgs_masks import sync_imgs_masks


def dataset_processing():
    dir_home = 'Origin'
    dir = ['Origin_EgoHands', 'Origin_EgoYouTubeHands', 'Origin_EGTEA_GAZE_PLUS', 'Origin_GTEA',
           'Origin_GTEA_GAZE_PLUS', 'Origin_HandOverFace']
    for i in range(6):
        dir[i] = f'{dir_home}/{dir[i]}'
        # Crop(Imgs and Masks)
        tools_box.center_crop_to_square(f'{dir[i]}/imgs', f'{dir[i]}/imgs_cropped')
        tools_box.center_crop_to_square(f'{dir[i]}/masks', f'{dir[i]}/masks_cropped')
        # Masks binarize
        tools_box.binarize_images(f'{dir[i]}/masks_cropped', f'{dir[i]}/masks_binarized')
        # Filter
        tools_box.dataset_mask_filter(
            image_dir=f'{dir[i]}/imgs_cropped',
            mask_dir=f'{dir[i]}/masks_binarized',
            mask_ratio_range=(0.02, 0.4)
        )
        # Resize
        tools_box.resize_and_convert_image(
            input_folder=f'{dir[i]}/imgs_cropped',
            output_folder=f'{dir[i]}/imgs_resized',
            target_size=(480, 480),
        )
        tools_box.resize_and_convert_mask(
            input_folder=f'{dir[i]}/masks_binarized',
            output_folder=f'{dir[i]}/masks_resized',
            target_size=(480, 480),
        )
        # GrayScale
        tools_box.convert_folder_to_grayscale(f'{dir[i]}/imgs_resized', f'{dir[i]}/imgs_grayscale')
        # Remove temp dir
        shutil.rmtree(f'{dir[i]}/imgs')
        shutil.rmtree(f'{dir[i]}/imgs_cropped')
        shutil.rmtree(f'{dir[i]}/imgs_resized')
        shutil.rmtree(f'{dir[i]}/masks')
        shutil.rmtree(f'{dir[i]}/masks_cropped')
        shutil.rmtree(f'{dir[i]}/masks_binarized')
        # Rename
        os.rename(f'{dir[i]}/imgs_grayscale', f'{dir[i]}/imgs')
        os.rename(f'{dir[i]}/masks_resized', f'{dir[i]}/masks')

def move2integation():
    dir = ['Process_EgoHands_480_480_MaskFilter_02_40_GrayScale',
           'Process_EgoYouTubeHands_480_480_MaskFilter_02_40_GrayScale'
    ]

    dir_output = 'Process_Hands_withoutHandOverFace_480_480_MaskFilter_02_40_GrayScale'

    for i in range (2):
        dir[i] = f'{dir[i]}'
        dir_items = os.listdir(f'{dir_output}/imgs')
        total_dir_items = len(dir_items)
        print(f'Total dir items: {total_dir_items}')
        num = total_dir_items + 1

        tools_box.rename_and_move_files(
            input_dir=f'{dir[i]}/imgs',
            output_dir=f'{dir_output}/imgs',
            start_num=num,
        )
        tools_box.rename_and_move_files(
            input_dir=f'{dir[i]}/masks',
            output_dir=f'{dir_output}/masks',
            start_num=num,
            suffix='_mask'
        )

if __name__ == '__main__':

    # tools_box.rename_and_move_files(
    #     input_dir='temp_output',
    #     output_dir='out',
    #     start_num=124,
    # )

    # tools_box.resize_and_convert_image(
    #     input_folder='temp_temp',
    #     output_folder='temp_out',
    #     target_size=(640, 480),
    # )

    tools_box.center_crop_to_square(
        input_folder='temp',
        output_folder='temp_output',
    )

    # tools_box.convert_folder_to_grayscale(
    #     input_dir='temp',
    #     output_dir='temp_out',
    # )
    # tools_box.binarize_images(
    #     input_dir='temp',
    #     output_dir='temp_out',
    #     threshold=254
    # )