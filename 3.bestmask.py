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

        # 条件：严格边界检查（排除任何接触边界的掩码）
        if (np.any(mask[0, :]) or  # 上边界
                np.any(mask[-1, :]) or  # 下边界
                np.any(mask[:, 0]) or  # 左边界
                np.any(mask[:, -1])):  # 右边界
            continue

        # 计算特征
        area_ratio = np.sum(mask) / (h * w)
        euler = euler_number(mask)

        # 计算区域属性
        labeled = label(mask)
        regions = regionprops(labeled)
        if not regions:
            continue
        main_region = max(regions, key=lambda x: x.area)

        # 计算横向中心偏移量
        y_centroid, x_centroid = main_region.centroid
        x_offset = abs(x_centroid - (w / 2))  # 横向中心偏移量（像素单位）

        if x_offset > 160:
            continue

        # 计算固体度
        labeled = label(mask)
        regions = regionprops(labeled)
        if not regions:
            continue
        main_region = max(regions, key=lambda x: x.area)
        solidity = main_region.solidity

        # 条件判断
        conditions = {
            'area': 0.05 <= area_ratio <= 0.28,
            'euler': euler == 1,
            'solidity': solidity >= 0.6
        }
        score = sum(conditions.values())

        # 面积接近度（用于平局排序）
        area_diff = abs(area_ratio - 0.2)

        candidates.append({
            'mask_data': mask_data,
            'score': score,
            'area_diff': area_diff,
            'features': {
                'area_ratio': area_ratio,
                'euler': euler,
                'solidity': solidity,
                'x_offset': x_offset
            }
        })
    if not candidates:
        return None

    # 排序逻辑
    sorted_candidates = sorted(candidates,
                               key=lambda x: (-x['score'],
                                              x['area_diff'],
                                              x['features']['x_offset']))
    best = sorted_candidates[0]

    print(f"最优掩码得分：{best['score']}/3 面积比：{best['features']['area_ratio']:.3f} "
          f"欧拉数：{best['features']['euler']} 横轴偏移度：{best['features']['x_offset']:.1f}")

    return best['mask_data'] if best['score'] > 0 else None


# 配置参数
device = torch.device('cpu')
MODEL_TYPE = "vit_l"
CHECKPOINT_PATH = r"sam_vit_l_0b3195.pth"
INPUT_FOLDER = r""
OUTPUT_FOLDER = r""

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
        # 加载图像
        image_path = os.path.join(INPUT_FOLDER, filename)
        orig_image = Image.open(image_path).convert("RGB")
        w, h = orig_image.size

        # 调整尺寸
        scale = 1024 / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        image = np.array(orig_image.resize(new_size))

        # 生成掩码
        masks = mask_generator.generate(image)
        print(f"Generated {len(masks)} masks")

        # 筛选和保存
        if not masks:
            print("未检测到任何掩码")
            continue

        target_mask = select_target_mask(masks)
        if not target_mask:
            print("没有符合条件的掩码")
            continue

        base_name = os.path.splitext(filename)[0]
        output_subdir = os.path.join(OUTPUT_FOLDER, base_name)
        os.makedirs(output_subdir, exist_ok=True)

        mask_img = Image.fromarray(target_mask["segmentation"]).resize((w, h), Image.NEAREST)
        output_path = os.path.join(output_subdir, "best_mask.png")
        mask_img.save(output_path)
        print(f"已保存最优掩码：{output_path}")

        # 清理内存
        del image, masks, orig_image
        gc.collect()

    except Exception as e:
        print(f"处理失败：{filename} - {str(e)}")

print("\n处理完成！输出目录：", OUTPUT_FOLDER)
def calculate_integral_ratio(hist):
    """计算特征积分比值"""
    x = np.arange(256)
    weights = 256 - x

    # 计算积分项
    terms = hist * weights

    # 计算最高像素值和最低像素值
    min_pixel = np.min(np.where(hist > 0)[0])  # 最低像素值
    max_pixel = np.max(np.where(hist > 0)[0])  # 最高像素值

    # 计算阈值
    threshold = int((max_pixel - min_pixel) * 0.7)

    # 计算积分值
    total_integral = np.sum(terms)
    partial_integral = np.sum(terms[:threshold])  # 使用动态阈值

    ratio = partial_integral / total_integral if total_integral != 0 else 0
    return ratio, threshold  # 返回两个值

for idx, filename in enumerate(image_files):
    print(f"\nProcessing {idx + 1}/{len(image_files)}: {filename}")

    try:
        # 加载图像
        image_path = os.path.join(INPUT_FOLDER, filename)
        orig_image = Image.open(image_path).convert("RGB")
        w, h = orig_image.size

        # 调整尺寸
        scale = 1024 / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        image = np.array(orig_image.resize(new_size))

        # 生成掩码
        masks = mask_generator.generate(image)
        print(f"Generated {len(masks)} masks")

        # 筛选和保存
        if not masks:
            print("未检测到任何掩码")
            continue

        target_mask = select_target_mask(masks)

        if not target_mask:
            print("没有符合条件的掩码")
            continue

        if target_mask:
            base_name = os.path.splitext(filename)[0]
            output_subdir = os.path.join(OUTPUT_FOLDER, base_name)
            os.makedirs(output_subdir, exist_ok=True)

        # 调整掩码尺寸并保存
        mask_img = Image.fromarray(target_mask["segmentation"]).resize((w, h), Image.NEAREST)
        mask_img.save(os.path.join(output_subdir, "best_mask.png"))

        # 新增灰度直方图分析
        try:
            # 转换原始图像为灰度
            gray_image = orig_image.convert("L")
            gray_array = np.array(gray_image)

            # 获取掩码区域像素
            mask_array = np.array(mask_img)
            if np.sum(mask_array) == 0:  # 检查掩码区域是否为空
                print(f"掩码区域为空：{filename}")
                continue

            target_pixels = gray_array[mask_array]

            # 检查目标像素是否有效
            if len(target_pixels) == 0:
                print(f"目标像素为空：{filename}")
                continue

            # 计算直方图
            hist, _ = np.histogram(target_pixels, bins=256, range=(0, 255))
            hist = hist / hist.sum()

            # 计算积分比值
            ratio, threshold = calculate_integral_ratio(hist)  # 解包两个值

            # 可视化直方图
            plt.figure(figsize=(10, 6))
            plt.bar(range(256), hist, width=1.0)  # 绘制直方图
            plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold}')  # 添加阈值线
            plt.title(f"Gray Histogram\nIntegral Ratio: {ratio:.4f}")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")
            plt.xlim(0, 255)
            plt.legend()  # 显示图例
            plt.savefig(os.path.join(output_subdir, "histogram.png"))
            plt.close()

            # 保存比值结果
            with open(os.path.join(output_subdir, "ratio.txt"), "w") as f:
                f.write(f"{ratio:.4f}")

            print(f"分析完成：积分比值 {ratio:.4f}")

        except Exception as e:
            print(f"直方图分析失败：{filename} - {str(e)}")

        # 清理内存
        finally:
            del image, masks, orig_image
            gc.collect()

    except Exception as e:
        print(f"处理失败：{filename} - {str(e)}")

print("\n处理完成！输出目录：", OUTPUT_FOLDER)

