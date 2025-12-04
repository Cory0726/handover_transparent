import argparse
import cv2
import numpy as np
from inference import Inferencer


def array_info(arr):
    return f'Shape{arr.shape}, Max: {arr.max():f}, Min: {arr.min():f}, Avg: {arr.mean():f}, {arr.dtype}'

def rawdepth_to_heatmap(rawdepth):
    gray_img = cv2.normalize(rawdepth, None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(255 - gray_img, cv2.COLORMAP_TURBO)
    # heatmap = cv2.applyColorMap(255 - gray_img, cv2.COLORMAP_JET)
    return heatmap

def main(img_path, depth_path):
    # Load grayscale image
    grayscale_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    print('grayscale_img : ' + array_info(grayscale_img))
    # Raw depth in mm
    depth = np.load(depth_path)  # Unit : mm
    print('raw_depth_mm : ' + array_info(depth) + '(mm)')

    # Convert depth unit mm to m
    depth_m = depth / 1000  # Unit : m

    # Create a fake RGB image by grayscale image
    fake_rgb = np.stack([grayscale_img, grayscale_img, grayscale_img], axis=2)  # (H, W, 3)
    print('fake_rgb : ' + array_info(fake_rgb))

    # Normalize the fake RGB image to 0 - 1
    norm_fake_rgb = fake_rgb / 255
    print('norm_fake_rgb : ' + array_info(norm_fake_rgb))

    # Initialize the inference, specify the configuration file.
    inferencer = Inferencer(cfg_path='transcg/configs/inference.yaml')

    # # Call inferencer for refined depth
    depth_refine, depth_ori = inferencer.inference(
        rgb=norm_fake_rgb,
        depth=depth_m,
        target_size=(640, 480),
        depth_coefficient=5.0,
        inpainting=True,
    )
    # Save the depth as .npy
    predict_depth = (depth_refine * 1000).astype(np.float32)  # Unit : mm
    print('Predict depth : ' + array_info(predict_depth) + '(mm)')
    np.save('data/transcg_predict_depth.npy', predict_depth)
    predict_depth_heatmap = rawdepth_to_heatmap(predict_depth)
    cv2.imwrite('data/transcg_predict_depth_heatmap.png', predict_depth_heatmap)


def parse_args():
    parser = argparse.ArgumentParser(description='U-Net segmentation model')
    parser.add_argument('--img', type=str)
    parser.add_argument('--depth', type=str)
    parser.parse_args()
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.img, args.depth)
