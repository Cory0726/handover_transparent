import tools_box


def main():
    dir = 'C:/Users/lkfu5/PycharmProjects/Dataset/Dataset_Hand/Process_GTEA_GAZE_PLUS/'

    output_imgs_dir = dir + 'imgs'
    input_imgs_dir = dir + 'imgs_backup'
    imgs_gray_dir = dir + 'imgs_temp'

    output_mask_dir = dir + 'masks'
    input_mask_dir = dir + 'masks_backup'
    masks_bi_dir = dir + 'masks_temp'

    tools_box.convert_folder_to_grayscale(input_imgs_dir, imgs_gray_dir)
    tools_box.resize_and_convert_image(imgs_gray_dir,output_imgs_dir, (640, 480))

    tools_box.binarize_images(input_mask_dir, masks_bi_dir)
    tools_box.resize_and_convert_mask(masks_bi_dir, output_mask_dir, (640, 480))




if __name__ == '__main__':
    main()