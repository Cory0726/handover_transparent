import argparse
import json
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
    # ==================================================
    # Load the data
    # ==================================================
    depth = np.load(depth_path)  # shape (H, W)
    depth = np.expand_dims(depth, axis=2)  # shape (H, W, 1)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # shape (H, W)
    # ==================================================
    # Pre-process the data
    # ==================================================
    crop_size = 224  # crop_size = 224
    depth_data = CameraData(output_size=crop_size)
    x, crop_depth = depth_data.get_data(depth)
    print(f'Crop depth : {crop_depth.shape}, {crop_depth.dtype}, {crop_depth.max()}, {crop_depth.min()}')
    x = x.unsqueeze(0)  # Increase the dimension of batch
    mask_data = CameraData(output_size=crop_size)
    _, crop_mask = mask_data.get_data(mask)
    # ==================================================
    # Predict stage
    # ==================================================
    net = torch.load(
        "gr_convnet/trained-models/jacquard-d-grconvnet3-drop0-ch32/epoch_50_iou_0.94",
        weights_only=False
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    with torch.no_grad():
        xc = x.to(device)
        pred = net.predict(xc)
    q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'], crop_size)
    # ==================================================
    # Save the result
    # ==================================================
    cv2.imwrite('data/grconv_crop_depth.png', vis_heatmap(crop_depth))
    cv2.imwrite('data/grconv_crop_mask.png', crop_mask)
    cv2.imwrite('data/grconv_q_img.png', vis_heatmap(q_img))
    cv2.imwrite('data/grconv_ang_img.png', vis_heatmap(ang_img))
    cv2.imwrite('data/grconv_width_img.png', vis_heatmap(width_img))
    # Plot the grasp rectangle on the depth image
    depth = depth.squeeze()
    fig, grasp= plot_depth_with_grasp(depth, mask, q_img, ang_img, width_img, no_grasps=1)
    grasp_depth = depth[grasp.center[0], grasp.center[1]]
    print(f'Q max: {q_img.max()}, Q min: {q_img.min()}')
    print(f'Grasp depth : {grasp_depth} mm')
    fig.savefig('data/grconv_grasp_result.pdf')
    print('Saved : data/grconv_grasp_result.pdf')
    final_grasp = {
        'center' : [int(grasp.center[0]), int(grasp.center[1])],
        'depth' : float(grasp_depth),
        'angle' : float(grasp.angle),
        'width' : float(grasp.width),
    }
    with open('data/grconv_grasp_result.json', 'w') as f:
        json.dump(final_grasp, f, indent=4)
        print('Saved : data/grconv_grasp_result.json')

def parse_args():
    parser = argparse.ArgumentParser(description='GR-ConvNet model')
    parser.add_argument('--depth', type=str)
    parser.add_argument('--mask', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args.depth, args.mask)