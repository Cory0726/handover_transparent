import cv2
import numpy as np
import os
import subprocess
from glob import glob
import json

def run_unet_predict_process_folder(input_folder, output_folder):
    """
    批次執行 U-Net 預測，並將輸出檔名改為 原檔名_mask.副檔名
    :param input_folder: 原始圖片的資料夾 (例如 'data/test_images')
    :param output_folder: 預測結果要存放的資料夾 (例如 'data/pred_masks')
    """

    # 1. 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)

    # 2. 抓取資料夾內所有圖片 (支援 png, jpg, jpeg)
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob(os.path.join(input_folder, ext)))

    image_paths.sort()

    print(f"==> 準備處理 {len(image_paths)} 張圖片...")

    # 3. 迴圈處理每一張圖
    for index, img_path in enumerate(image_paths):

        # 取得檔名和副檔名
        filename_with_ext = os.path.basename(img_path)  # 例如: 'hand_01.png'
        filename_base, file_ext = os.path.splitext(filename_with_ext)  # 'hand_01', '.png'

        # 組合新的輸出檔名 (加上 _mask)
        # 範例: 'hand_01' + '_mask' + '.png' -> 'hand_01_mask.png'
        new_filename = filename_base + '_mask' + file_ext

        # 組合輸出路徑
        save_path = os.path.join(output_folder, new_filename)

        print(f"[{index + 1}/{len(image_paths)}] Processing: {filename_with_ext} -> Saving as: {new_filename}")

        try:
            # 呼叫原本的 predict.py
            subprocess.run(
                args=[
                    'python', 'test_predict.py',
                    '--input', img_path,  # 傳入單張圖片路徑
                    '--output', save_path,
                    '--model', 'Hand_Seg_EGTEA_plus_S640480G_Scale05_Score08994_20251123.pth',
                ],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"!! Error processing {filename_with_ext}: {e}")

    print("==> 全部處理完成！")

def calculate_unet_metrics(pred_dir, gt_dir, result_file_name, threshold=127):
    """
    計算手部分割的 IoU, Dice, Pixel Accuracy
    輸入:
        pred_dir: 模型預測出的 Mask 資料夾路徑
        gt_dir: 真實標籤 (Ground Truth) Mask 資料夾路徑
        threshold: 二值化的閾值 (通常為 127 或 0.5)
    """

    # 獲取檔案列表 (假設檔名是匹配的)
    # 如果你的預測圖檔名跟真實圖檔名不同，這裡需要額外處理
    pred_files = sorted(glob(os.path.join(pred_dir, "*.png")))
    gt_files = sorted(glob(os.path.join(gt_dir, "*.png")))

    # 檢查檔案數量
    if len(pred_files) == 0 or len(gt_files) == 0:
        print("錯誤：找不到影像，請檢查路徑或副檔名 (例如是否為 .jpg)")
        return

    if len(pred_files) != len(gt_files):
        print(f"警告：檔案數量不一致 (預測: {len(pred_files)}, 真實: {len(gt_files)})")
        print("將只計算檔名匹配的部分...")

    # 初始化總分
    total_iou = []
    total_dice = []
    total_acc = []

    print(f"開始評估 {len(pred_files)} 張影像...\n")

    for pred_path, gt_path in zip(pred_files, gt_files):
        # 1. 讀取影像 (讀取為灰階)
        pred = cv2.imread(pred_path, 0)
        gt = cv2.imread(gt_path, 0)

        # 確保尺寸一致
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 2. 二值化 (0 為背景, 1 為手)
        # 這裡假設 Mask 是 0-255，大於 127 的視為手
        pred_bin = (pred > threshold).astype(np.uint8)
        gt_bin = (gt > threshold).astype(np.uint8)

        # 3. 計算基本元素 (TP, FP, FN)
        # Intersection (交集): 兩者都是 1 的地方
        intersection = (pred_bin & gt_bin).sum()

        # Union (聯集): 其中一個是 1 的地方
        union = (pred_bin | gt_bin).sum()

        # Ground Truth 和 Prediction 的總像素數 (用於 Dice)
        gt_sum = gt_bin.sum()
        pred_sum = pred_bin.sum()

        # 4. 計算指標

        # --- IoU ---
        # 避免分母為 0 (如果兩張圖都是全黑背景，IoU 設為 1)
        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union

        # --- Dice Coefficient ---
        if (gt_sum + pred_sum) == 0:
            dice = 1.0
        else:
            dice = 2 * intersection / (gt_sum + pred_sum)

        # --- Pixel Accuracy ---
        # 相等的像素 / 總像素
        height, width = pred_bin.shape
        total_pixels = height * width
        correct_pixels = (pred_bin == gt_bin).sum()
        acc = correct_pixels / total_pixels

        # 加入列表
        total_iou.append(iou)
        total_dice.append(dice)
        total_acc.append(acc)

    # 5. 計算平均值 (Mean Metrics)
    mIoU = np.mean(total_iou)
    mDice = np.mean(total_dice)
    mAcc = np.mean(total_acc)

    print("-" * 30)
    print("評估結果 (Evaluation Results):")
    print("-" * 30)
    print(f"Images Processed : {len(total_iou)}")
    print(f"mIoU (Mean IoU)  : {mIoU:.4f}  <-- 最重要指標")
    print(f"Dice Coefficient : {mDice:.4f}")
    print(f"Pixel Accuracy   : {mAcc:.4f}")
    print("-" * 30)

    result = {
        'imgs_number': len(total_iou),
        "mIoU": mIoU,
        'Dice_Coefficient': mDice,
        'Pixel_Accuracy': mAcc,
    }
    with open(result_file_name, 'w') as f:
        json.dump(result, f, indent=4)
        print(f'Saved : {result_file_name}\n\n')

def run_unet_predict_set():
    # ==================================================
    # Run U-Net predict
    # ==================================================
    # 1
    run_unet_predict_process_folder(
        input_folder='C:/Users/lkfu5/PycharmProjects/Dataset/Dataset_Hand/Process_EgoHands_640_480_MaskFilter_05_40_GrayScale/imgs',
        output_folder='C:/Users/lkfu5/PycharmProjects/Dataset/Dataset_Hand/Process_EgoHands_640_480_MaskFilter_05_40_GrayScale/predict',
    )
    # 2
    run_unet_predict_process_folder(
        input_folder='C:/Users/lkfu5/PycharmProjects/Dataset/Dataset_Hand/Process_EgoYouTubeHands_640_480_MaskFilter_05_40_GrayScale/imgs',
        output_folder='C:/Users/lkfu5/PycharmProjects/Dataset/Dataset_Hand/Process_EgoYouTubeHands_640_480_MaskFilter_05_40_GrayScale/predict',
    )
    # 3
    run_unet_predict_process_folder(
        input_folder='C:/Users/lkfu5/PycharmProjects/Dataset/Dataset_Hand/Process_EGTEA_Gaze_plus_640_480_MaskFilter_05_40_GrayScale/imgs',
        output_folder='C:/Users/lkfu5/PycharmProjects/Dataset/Dataset_Hand/Process_EGTEA_Gaze_plus_640_480_MaskFilter_05_40_GrayScale/predict',
    )
    # 4
    run_unet_predict_process_folder(
        input_folder='C:/Users/lkfu5/PycharmProjects/Dataset/Dataset_Hand/Process_GTEA_640_480_MaskFilter_05_40_GrayScale/imgs',
        output_folder='C:/Users/lkfu5/PycharmProjects/Dataset/Dataset_Hand/Process_GTEA_640_480_MaskFilter_05_40_GrayScale/predict',
    )
    # 5
    run_unet_predict_process_folder(
        input_folder='C:/Users/lkfu5/PycharmProjects/Dataset/Dataset_Hand/Process_GTEA_GAZE_PLUS_640_480_MaskFilter_05_40_GrayScale/imgs',
        output_folder='C:/Users/lkfu5/PycharmProjects/Dataset/Dataset_Hand/Process_GTEA_GAZE_PLUS_640_480_MaskFilter_05_40_GrayScale/predict',
    )
    # 6
    run_unet_predict_process_folder(
        input_folder='C:/Users/lkfu5/PycharmProjects/Dataset/Dataset_Hand/Process_HandOverFace_640_480_MaskFilter_05_40_GrayScale/imgs',
        output_folder='C:/Users/lkfu5/PycharmProjects/Dataset/Dataset_Hand/Process_HandOverFace_640_480_MaskFilter_05_40_GrayScale/predict',
    )

def cal_unet_metric_set():
    # ==================================================
    # Calculate the U-Net metrics
    # ==================================================
    # 1
    calculate_unet_metrics(
        pred_dir='C:/Users/lkfu5/PycharmProjects/Dataset/Dataset_Hand/Process_EgoHands_640_480_MaskFilter_05_40_GrayScale/predict',
        gt_dir='C:/Users/lkfu5/PycharmProjects/Dataset/Dataset_Hand/Process_EgoHands_640_480_MaskFilter_05_40_GrayScale/masks',
        result_file_name='test_result/EgoHands_result.json'
    )
    # 2
    calculate_unet_metrics(
        pred_dir='C:/Users/lkfu5/PycharmProjects/Dataset/Dataset_Hand/Process_EgoYouTubeHands_640_480_MaskFilter_05_40_GrayScale/predict',
        gt_dir='C:/Users/lkfu5/PycharmProjects/Dataset/Dataset_Hand/Process_EgoYouTubeHands_640_480_MaskFilter_05_40_GrayScale/masks',
        result_file_name='test_result/EgoYouTubeHands_result.json'
    )
    # 3
    calculate_unet_metrics(
        pred_dir='C:/Users/lkfu5/PycharmProjects/Dataset/Dataset_Hand/Process_EGTEA_Gaze_plus_640_480_MaskFilter_05_40_GrayScale/predict',
        gt_dir='C:/Users/lkfu5/PycharmProjects/Dataset/Dataset_Hand/Process_EGTEA_Gaze_plus_640_480_MaskFilter_05_40_GrayScale/masks',
        result_file_name='test_result/EGTEA_Gaze_plus_result.json'
    )
    # 4
    calculate_unet_metrics(
        pred_dir='C:/Users/lkfu5/PycharmProjects/Dataset/Dataset_Hand/Process_GTEA_640_480_MaskFilter_05_40_GrayScale/predict',
        gt_dir='C:/Users/lkfu5/PycharmProjects/Dataset/Dataset_Hand/Process_GTEA_640_480_MaskFilter_05_40_GrayScale/masks',
        result_file_name='test_result/GTEA_result.json'
    )
    # 5
    calculate_unet_metrics(
        pred_dir='C:/Users/lkfu5/PycharmProjects/Dataset/Dataset_Hand/Process_GTEA_GAZE_PLUS_640_480_MaskFilter_05_40_GrayScale/predict',
        gt_dir='C:/Users/lkfu5/PycharmProjects/Dataset/Dataset_Hand/Process_GTEA_GAZE_PLUS_640_480_MaskFilter_05_40_GrayScale/masks',
        result_file_name='test_result/GTEA_GAZE_PLUS_result.json'
    )
    # 6
    calculate_unet_metrics(
        pred_dir='C:/Users/lkfu5/PycharmProjects/Dataset/Dataset_Hand/Process_HandOverFace_640_480_MaskFilter_05_40_GrayScale/predict',
        gt_dir='C:/Users/lkfu5/PycharmProjects/Dataset/Dataset_Hand/Process_HandOverFace_640_480_MaskFilter_05_40_GrayScale/masks',
        result_file_name='test_result/HandOverFace_result.json'
    )

if __name__ == "__main__":
    cal_unet_metric_set()