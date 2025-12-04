import cv2
import torch
from skimage.filters import gaussian


def post_process_output(q_img, cos_img, sin_img, width_img):
    """
    Post-process the raw output of the network, convert to numpy arrays, apply filtering.
    :param q_img: Q output of network (as torch Tensors)
    :param cos_img: cos output of network
    :param sin_img: sin output of network
    :param width_img: Width output of network
    :return: Filtered Q output, Filtered Angle output, Filtered Width output
    """
    # Convert tensor -> numpy (H,W)
    q_img = q_img.cpu().numpy().squeeze()
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
    width_img = width_img.cpu().numpy().squeeze() * 150.0
    # Smooth maps by Gaussian
    q_img = gaussian(q_img, 2.0, preserve_range=True)
    ang_img = gaussian(ang_img, 2.0, preserve_range=True)
    width_img = gaussian(width_img, 1.0, preserve_range=True)
    # Resize the image from (224, 224) to (480, 480)
    q_img = cv2.resize(q_img, (480, 480), interpolation=cv2.INTER_NEAREST)
    ang_img = cv2.resize(ang_img, (480, 480), interpolation=cv2.INTER_NEAREST)
    width_img = cv2.resize(width_img, (480, 480), interpolation=cv2.INTER_LINEAR)
    return q_img, ang_img, width_img
