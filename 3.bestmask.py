import os
import torch
import gc
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, euler_number
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def select_target_mask(masks):
    """根据多条件筛选最优掩码"""
    candidates = []
    for mask_data in masks:
        mask = mask_data["segmentation"]
        h, w = mask.shape

        # 边界检查
        if np.any(mask[0, :]) or np.any(mask[-1, :]) or np.any(mask[:, 0]) or np.any(mask[:, -1]):
            continue

        # 计算特征
        area_ratio = np.sum(mask) / (h * w)
        euler = euler_number(mask)

        # 硬性条件过滤
        if not (0.02 <= area_ratio <= 0.07 and euler == 1):
            continue

        # 区域属性分析
        labeled = label(mask)
        regions = regionprops(labeled)
        if not regions:
            continue
        main_region = max(regions, key=lambda x: x.area)

        # 特征计算
        y_centroid, x_centroid = main_region.centroid
        x_offset = abs(x_centroid - (w / 2))
        acura = (x_offset / 20000) + abs(area_ratio - 0.035)

        candidates.append({
            'mask_data': mask_data,
            'features': {
                'area_ratio': area_ratio,
                'euler': euler,
                'x_offset': x_offset,
                'acura': acura
            }
        })

    if not candidates:
        return None

    # 按acura值排序
    sorted_candidates = sorted(candidates, key=lambda x: x['features']['acura'])
    best = sorted_candidates[0]

    print(f"最优掩码 | Acura：{best['features']['acura']:.3f} | 偏移：{best['features']['x_offset']:.1f}px")
    return best['mask_data']

def calculate_integral_ratio(hist):
    """计算特征积分比值"""
    x = np.arange(256)
    weights = 256 - x

    # 计算积分项
    terms = hist * weights

    # 计算积分值
    total_integral = np.sum(terms)
    partial_integral = np.sum(terms[:150])

    return partial_integral / total_integral if total_integral != 0 else 0

# 配置参数
device = torch.device('cpu')
MODEL_TYPE = "vit_l"
CHECKPOINT_PATH = r"D:\BaiduNetdiskDownload\sam\sam_vit_l_0b3195.pth"
INPUT_FOLDER = r"C:\Users\my0131\Desktop\test_data"
OUTPUT_FOLDER = r"C:\Users\my0131\Desktop\test_data\out"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 初始化模型
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=16,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.90,
    crop_n_layers=0,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=10000,
)

# 处理图像
image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for idx, filename in enumerate(image_files):
    print(f"\nProcessing {idx + 1}/{len(image_files)}: {filename}")

    try:
        # --- 图像加载和掩码生成 ---
        image_path = os.path.join(INPUT_FOLDER, filename)
        orig_image = Image.open(image_path).convert("RGB")
        w, h = orig_image.size

        scale = 1024 / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        image = np.array(orig_image.resize(new_size))

        masks = mask_generator.generate(image)
        print(f"Generated {len(masks)} masks")

        if not masks:
            print("未检测到任何掩码")
            continue

        target_mask = select_target_mask(masks)
        if not target_mask:
            print("没有符合条件的掩码")
            continue

        # --- 保存掩码 ---
        base_name = os.path.splitext(filename)[0]
        output_subdir = os.path.join(OUTPUT_FOLDER, base_name)
        os.makedirs(output_subdir, exist_ok=True)

        mask_img = Image.fromarray(target_mask["segmentation"]).resize((w, h), Image.NEAREST)
        mask_img.save(os.path.join(output_subdir, "best_mask.png"))

        # --- 直方图分析 ---
        gray_image = orig_image.convert("L")
        gray_array = np.array(gray_image)
        mask_array = np.array(mask_img)

        if np.sum(mask_array) == 0:
            print(f"掩码区域为空：{filename}")
            continue

        target_pixels = gray_array[mask_array]
        if len(target_pixels) == 0:
            print(f"目标像素为空：{filename}")
            continue

        hist, _ = np.histogram(target_pixels, bins=256, range=(0, 255))
        hist = hist / hist.sum()

        # 调用修复后的函数
        ratio = calculate_integral_ratio(hist)
        threshold = 150

        # 保存直方图和分析结果
        plt.figure(figsize=(10, 6))
        plt.bar(range(256), hist, width=1.0)
        plt.axvline(threshold, color='r', linestyle='--')
        plt.savefig(os.path.join(output_subdir, "histogram.png"))
        plt.close()

        with open(os.path.join(output_subdir, "ratio.txt"), "w") as f:
            f.write(f"{ratio:.4f}")

        # --- 清理内存 ---
        del image, masks, orig_image
        gc.collect()

    except Exception as e:
        print(f"处理失败：{filename} - {str(e)}")

