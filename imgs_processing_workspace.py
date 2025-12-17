import cv2

import tools_box


def main():
    dir = 'C:/Users/lkfu5/PycharmProjects/Dataset/Dataset_Hand/Process_GTEA_GAZE_PLUS_640_480_MaskFilter_05_40_Grayscale/'
    #
    # output_imgs_dir = dir + 'imgs'
    input_imgs_dir = dir + 'imgs'
    # imgs_gray_dir = dir + 'imgs_temp'
    #
    # output_mask_dir = dir + 'masks'
    input_mask_dir = dir + 'masks'
    # masks_bi_dir = dir + 'masks_temp'
    #
    tools_box.dataset_mask_filter(
        input_imgs_dir,
        input_mask_dir,
        mask_ratio_range=(0.05, 0.4),
    )

if __name__ == '__main__':
    main()