import os
import cv2
import numpy as np
from glob import glob


def calculate_unet_metrics(pred_dir, gt_dir, threshold=127):
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


if __name__ == "__main__":
    # 假設你的真實 Mask 放在這裡
    ground_truth_folder = 'test_data/ground_truth'

    # 假設你的模型預測出來的 Mask 放在這裡
    prediction_folder = 'test_data/predict'

    # 注意：請確保兩個資料夾內的圖片檔名是一樣的，或者順序是一樣的
    calculate_unet_metrics(prediction_folder, ground_truth_folder)