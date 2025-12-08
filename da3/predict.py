
import argparse
import torch
import cv2
import numpy as np
from depth_anything_3.api import DepthAnything3
from sklearn.linear_model import LinearRegression, RANSACRegressor

def array_info(arr):
    return f'Shape{arr.shape}, Max: {arr.max():f}, Min: {arr.min():f}, Avg: {arr.mean():f}, {arr.dtype}'

def da3_model_initial():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained('depth-anything/da3metric-large')  # DA3METRIC-LARGE Model
    model = model.to(device)
    return model

def predict_da3_depth(da3_model, img):
    prediction = da3_model.inference([img])
    # Re-size to the input resolution
    raw_depth = cv2.resize(prediction.depth[0],(640,480),interpolation=cv2.INTER_LINEAR)
    return raw_depth

def calibrate_depth_ransac(
    D_da3,                      # DA3 depth map, shape (H, W), (mm)
    D_tof,                      # ToF depth map, shape (H, W), mm
    mask,                       # Mask image, 255 = use pixel, 0 = ignore
    min_depth=200,             # Minimum valid depth (mm)
    max_depth=600,              # Maximum valid depth (mm)
    residual_threshold=1,    # RANSAC inlier threshold (mm)
    min_samples=3000,            # Minimum valid samples for calibration
    max_trials=1000              # RANSAC iterations
):
    """
    Calibrate DA3 depth using ToF depth in masked regions (mask = 255).
    Uses RANSAC to robustly fit:
        d_tof ≈ a * d_da3 + b

    Returns:
        D_da3_calibrated : np.ndarray (H, W), calibrated DA3 depth (mm)
    """

    # Build valid-pixel mask
    valid = (
        (mask == 255) &
        np.isfinite(D_da3) & np.isfinite(D_tof) &  # Remove NaNs
        (D_da3 > min_depth) & (D_da3 < max_depth) &
        (D_tof > min_depth) & (D_tof < max_depth)
    )
    save_file_name = 'data/da3_valid_pixel_mask.png'
    cv2.imwrite(save_file_name, valid.astype(np.uint8) * 255)
    print(f'Saved {save_file_name}')
    # Check the number of valid pixels
    print('Number of valid pixels: ', valid.sum())
    if valid.sum() < min_samples:
        print('[WARNING] Not enough valid pixels for RANSAC calibration. Skipping calibration.')
        return D_da3.copy()
    # Extract valid pixels >> flatten to 1D arrays
    d_da3 = D_da3[valid].astype(np.float32)  # shape (N,)
    d_tof = D_tof[valid].astype(np.float32)  # shape (N,)
    # RANSAC robust regression
    ransac_model = RANSACRegressor(
        LinearRegression(),
        max_trials=max_trials,
        min_samples=min_samples,
        residual_threshold=residual_threshold,
    )
    ransac_model.fit(d_da3.reshape(-1, 1), d_tof)
    # Extract scale and offset
    a = float(ransac_model.estimator_.coef_[0])
    b = float(ransac_model.estimator_.intercept_)
    print(f"[Calibration] d_tof ≈ {a:.4f} * d_da3 + {b:.4f}")
    # Apply calibration to the entire predicted depth map of DA3
    D_da3_calibrated = a * D_da3 + b
    return D_da3_calibrated

def depth_to_color(depth):
    """Normalize depth to 0~255 and apply a colormap for visualization."""
    depth_vis = depth.copy()

    # Handle NaN / inf
    depth_vis = np.where(np.isfinite(depth_vis), depth_vis, 0)

    # Use percentiles to avoid extreme outliers affecting contrast
    d_min = np.percentile(depth_vis, 1)
    d_max = np.percentile(depth_vis, 99)

    if d_max <= d_min:  # fallback
        d_min = float(depth_vis.min())
        d_max = float(depth_vis.max())

    if d_max == d_min:
        d_max = d_min + 1e-6

    depth_norm = np.clip((depth_vis - d_min) / (d_max - d_min), 0, 1)
    depth_8u = (depth_norm * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
    return depth_color

def main(input_img_file, tof_depth_file, hand_seg_mask_file):
    """
    Predict Depth-anything 3 depth with RANSAC calibration.
    :param input_img_file: Path to grayscale or intensity input image for DA3 prediction.
    :param tof_depth_file: Path to .file containing tof depth in millimeters.
    :param hand_seg_mask_file: Path to segmentation mask (uint8 image with values {255 = hand, 0 = background}).
    :param output_depth_file: Output path (.npy) for saving the calibrated depth map (millimeters).
    """
    # ==================================================
    # Predict the raw depth by Depth-Anything-3 model
    # ==================================================
    # Initialize the DA3 model
    model = da3_model_initial()
    # Load original image for DA3 model predict
    img = cv2.imread(input_img_file, cv2.IMREAD_GRAYSCALE)
    print('Original image for DA3 predict : ' + array_info(img))
    # Predicted raw depth
    # predicted_depth = predict_da3_depth(model, img)
    predicted_depth = predict_da3_depth(model, img) * 1000  # Unit : mm
    print('Predicted depth by DA3 model : ' + array_info(predicted_depth) + '(mm)')
    # Save the predicted depth as .npy
    save_file_name = 'data/da3_predicted_depth'
    np.save(save_file_name, predicted_depth)
    print(f'Saved : {save_file_name}')
    # Save the predicted depth heatmap
    save_file_name = 'data/da3_predicted_depth_heatmap.png'
    cv2.imwrite(save_file_name, depth_to_color(predicted_depth))
    print(f'Saved : {save_file_name}')

    # ==================================================
    # Calibrate the predicted raw depth with ToF raw depth and Hand Segmentation mask
    # ==================================================
    # Load the ToF raw depth
    tof_depth = np.load(tof_depth_file)  # Unit : mm
    print('ToF raw depth : ' + array_info(tof_depth) + '(mm)')
    # Load the mask of hand segmentation
    hand_seg_mask = cv2.imread(hand_seg_mask_file, cv2.IMREAD_UNCHANGED)  # [255, 0]
    print('Hand Seg mask : ' + array_info(hand_seg_mask))
    # Calibrate the predicted raw depth
    calibrate_depth = calibrate_depth_ransac(predicted_depth, tof_depth, hand_seg_mask)  # Unit : mm
    print('Calibrated depth : ' + array_info(calibrate_depth) + '(mm)')
    # Save the calibrated depth as .npy
    np.save('data/da3_cal_depth', calibrate_depth)
    print(f'Save : data/da3_cal_depth')
    # Save the calibrated depth heatmap
    save_file_name = 'data/da3_cal_depth_heatmap.png'
    cv2.imwrite(save_file_name, depth_to_color(calibrate_depth))
    print(f'Saved : {save_file_name}')

def parse_args():
    parser = argparse.ArgumentParser(description='Depth-Anythin-3 model')
    parser.add_argument('--img', type=str)
    parser.add_argument('--depth', type=str)
    parser.add_argument('--hand', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args.img, args.depth, args.hand)
