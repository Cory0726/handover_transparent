import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
import cv2
from utils.data_loading import BasicDataset
from unet import UNet

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()  # Set model to evaluation mode
    # Preprocess input image: resize, normalize, convert to numpy array
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    # Add batch dimension (1, C, H, W)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()  # Forward pass, move output to CPU

        # Resize prediction to original image size
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')

        # For multi-class segmentation:
        # Each pixel has n class scores, take  the index (class ID) with the highest score per pixel.
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            # For binary segmentation:
            # Apply sigmoid to convert logits to probabilities (0-1 range),
            # then threshold to produce a boolean mask (foreground/background)
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

def run_predict(
        img,
        model,
        scale=0.5,
        num_of_channels = 1,
        num_of_classes = 2,
        mask_threshold = 0.5,
        bilinear = False,
):
    net = UNet(n_channels=num_of_channels, n_classes=num_of_classes, bilinear=bilinear)  # n_channel=3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net.to(device=device)
    state_dict = torch.load(model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    mask = predict_img(net=net,
                        full_img=img,
                        scale_factor=scale,
                        out_threshold=mask_threshold,
                        device=device)
    # Return mask
    return mask_to_image(mask, mask_values)

def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    mask: shape (H,W), pixel values = 0 or 255
    return: shape (H,W), pixel values = 0 or 255, keeping only largest connected component
    """

    # Convert 255 to 1 for connected component processing
    mask_bin = (mask > 0).astype('uint8')

    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)

    # If no foreground region exists, just return original
    if num_labels <= 1:
        return mask

    # Foreground areas (skip label 0 which is background)
    areas = stats[1:, cv2.CC_STAT_AREA]

    # The largest connected component label index (need +1 because we skipped background)
    largest_label = 1 + np.argmax(areas)

    # Build the output mask
    largest_mask = (labels == largest_label).astype('uint8') * 255

    return largest_mask

def main(input_img_path, output_img_path, model_path):
    # Load image
    img = Image.open(input_img_path)

    # Final mask
    final_mask_np = None

    # Adjust the brightness of the image 50 - 95 %
    brightness_levels = [i/100 for i in range(50, 96, 5)]
    print(f'brightness_levels: {brightness_levels} %')
    print(f'Loaded model : {model_path}, U-Net model predicting...')
    for b in brightness_levels:
        enhancer = ImageEnhance.Brightness(img)
        img_bright = enhancer.enhance(b)

        # Predict the mask by U-Net model
        result_mask = run_predict(
            img=img_bright,
            model= model_path,
            scale=0.5,
            num_of_channels=1,
            num_of_classes=2,
            mask_threshold=0.5,
            bilinear=False,
        )

        # Save images and masks at different brightness levels
        temp_img_path = f'u_net/img_process_temp/img_brightness_{int(b*100):3d}.png'
        temp_mask_path = f'u_net/img_process_temp/mask_brightness_{int(b*100):3d}.png'
        img_bright.save(temp_img_path)
        # print(f'Saved: {temp_img_path}')
        result_mask.save(temp_mask_path)
        # print(f'Saved: {temp_mask_path}')

        # Mask convert to numpy type
        mask_np = np.array(result_mask)
        # Pixel-wise OR merging
        if final_mask_np is None:
            final_mask_np = mask_np.copy()
        else:
            final_mask_np = np.maximum(final_mask_np, mask_np)

    # Keep the largest component of the final mask
    final_mask_np = keep_largest_component(final_mask_np)
    # Save the final mask
    final_mask = Image.fromarray(final_mask_np)
    final_mask.save(output_img_path)
    print(f'Saved {output_img_path}')

def parse_args():
    parser = argparse.ArgumentParser(description='U-Net segmentation model')
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args.input, args.output, 'u_net/Hand_Seg_EGTEA_plus_S640480G_Scale05_Score08994_20251123.pth')