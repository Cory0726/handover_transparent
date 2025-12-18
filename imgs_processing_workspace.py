import cv2

import tools_box


def main():
    # dir = ['Origin_EgoHands', 'Origin_EgoYouTubeHands', 'Origin_EGTEA_GAZE_PLUS', 'Origin_GTEA', 'Origin_GTEA_GAZE_PLUS', 'Origin_HandOverFace']
    # for i in range(6):
        # tools_box.center_crop_to_square(f'{dir[i]}/imgs', f'{dir[i]}/imgs_cropped')
        # tools_box.center_crop_to_square(f'{dir[i]}/masks', f'{dir[i]}/masks_cropped')

        # tools_box.convert_folder_to_grayscale(f'{dir[i]}/imgs_cropped', f'{dir[i]}/imgs_grayscale')
        # tools_box.binarize_images(f'{dir[i]}/masks_cropped', f'{dir[i]}/masks_binarized')

        # tools_box.resize_and_convert_image(
        #     input_folder=f'{dir[i]}/imgs_grayscale',
        #     output_folder=f'{dir[i]}/imgs_resized',
        #     target_size=(480, 480),
        # )
        # tools_box.resize_and_convert_mask(
        #     input_folder=f'{dir[i]}/masks_binarized',
        #     output_folder=f'{dir[i]}/masks_resized',
        #     target_size=(480, 480),
        # )

        # tools_box.dataset_mask_filter(
        #     image_dir=f'{dir[i]}/imgs_resized',
        #     mask_dir=f'{dir[i]}/masks_resized',
        #     mask_ratio_range=(0.1, 0.4)
        # )
        # dir = 'Process_HandOverFace_480_480_MaskFilter_10_40_GrayScale'
        # dir_output = 'Process_All_Hands_480_480_MaskFilter_10_40_GrayScale'
        # num = 6302
        # tools_box.rename_and_move_files(
        #     input_dir=f'{dir}/imgs_resized',
        #     output_dir=f'{dir_output}/imgs',
        #     start_num=num,
        # )
        # tools_box.rename_and_move_files(
        #     input_dir=f'{dir}/masks_resized',
        #     output_dir=f'{dir_output}/masks',
        #     start_num=num,
        #     suffix='_mask'
        # )

    img = cv2.imread('Process_All_Hands_480_480_MaskFilter_10_40_GrayScale/imgs_grayscale/img_00031.png', cv2.IMREAD_UNCHANGED)
    print(img.shape)
    img_mask = cv2.imread('Process_All_Hands_480_480_MaskFilter_10_40_GrayScale/masks/img_00001_mask.png', cv2.IMREAD_UNCHANGED)
    print(img_mask.shape)
if __name__ == '__main__':
    main()