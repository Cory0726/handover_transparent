import cv2
import numpy as np
import torch

from utils.dataset_processing import image


class CameraData:
    """
    Dataset wrapper for the camera data.
    """
    def __init__(self, width=640, height=480, output_size=480):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        """
        self.output_size = output_size

        left = (width - output_size) // 2
        top = (height - output_size) // 2
        right = (width + output_size) // 2
        bottom = (height + output_size) // 2

        self.bottom_right = (bottom, right)
        self.top_left = (top, left)

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_data(self, depth):
        depth_img = image.Image(depth)
        # Crop the image
        depth_img.crop(bottom_right=self.bottom_right, top_left=self.top_left)
        crop_img = depth_img.img.squeeze()
        # Resize the image
        resized_img = cv2.resize(depth_img.img, (224, 224), interpolation=cv2.INTER_NEAREST)
        depth_img.img = np.expand_dims(resized_img, 2)
        # Normalise
        depth_img.normalise()
        # Change the order of the dimension
        depth_img.img = depth_img.img.transpose((2, 0, 1))

        x = self.numpy_to_torch(depth_img.img)
        return x, crop_img
