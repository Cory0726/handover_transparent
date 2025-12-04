import argparse

import cv2
import numpy as np
import torch.utils.data

from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import plot_depth_with_grasp

def vis_heatmap(img:np.ndarray):
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    return img

def main(depth_path, mask_path):
    # Load depth image
    depth = np.load(depth_path)  # shape (H, W)
    depth = np.expand_dims(depth, axis=2)  # shape (H, W, 1)
    # Load hand segmentation mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    hand_mask = (mask == 255)
    # Load pre-trained model
    net = torch.load(
        "gr_convnet/trained-models/jacquard-d-grconvnet3-drop0-ch32/epoch_50_iou_0.94",
        weights_only=False
    )
    # Get the compute device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    # Pre-process the depth data to tensor for the model
    img_data = CameraData()
    x, crop_depth_img = img_data.get_data(depth=depth)
    x = x.unsqueeze(0)  # Increase the dimension of batch
    # Predict the grasp pose by GR-ConvNet model
    with torch.no_grad():
        xc = x.to(device)
        pred = net.predict(xc)
    q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
    # Save the result of the predict grasp
    cv2.imwrite('data/grconv_crop_depth.png', vis_heatmap(crop_depth_img))
    cv2.imwrite('data/grconv_q_img.png', vis_heatmap(q_img))
    cv2.imwrite('data/grconv_ang_img.png', vis_heatmap(ang_img))
    cv2.imwrite('data/grconv_width_img.png', vis_heatmap(width_img))
    # Plot the grasp rectangle on the depth image
    fig, final_grasps= plot_depth_with_grasp(crop_depth_img, q_img, ang_img, width_img, no_grasps=1)
    fig.savefig('data/grconv_grasp_result.pdf')

    for i in range(3):
        if hand_mask[final_grasps[i].center[0], final_grasps[i].center[1]]:
            continue
        print(final_grasps[i].center, type(final_grasps[i].center))

def parse_args():
    parser = argparse.ArgumentParser(description='GR-ConvNet model')
    parser.add_argument('--depth', type=str)
    parser.add_argument('--mask', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args.depth, args.mask)