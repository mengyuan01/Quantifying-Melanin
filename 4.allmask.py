import os
import torch
import gc
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, euler_number
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# 强制使用CPU配置
device = torch.device('cpu')
MODEL_TYPE = "vit_l"
CHECKPOINT_PATH = r"sam_vit_l_0b3195.pth"
INPUT_FOLDER = r""
OUTPUT_FOLDER = r""

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def save_mask_visualization(image, masks, output_dir="mask_visualizations"):
    """
    保存所有掩码的可视化结果和特征日志
    """
    os.makedirs(output_dir, exist_ok=True)

    # 创建日志文件
    log_file = open(f"{output_dir}/mask_features.log", "w")
    log_file.write("MaskID | AreaRatio | Euler | Solidity | XOffset\n")
    log_file.write("-" * 50 + "\n")

    # 灰度化处理原图
    gray_image = np.mean(image, axis=2).astype(np.uint8) if len(image.shape) == 3 else image

    for idx, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]

        if not np.any(mask):  # 检查掩码是否为空
            continue

        # 创建可视化图像
        plt.figure(figsize=(10, 6))

        # 显示原图
        plt.imshow(gray_image, cmap='gray')

        # 用半透明红色显示掩码区域
        plt.imshow(np.where(mask, 255, 0), alpha=0.3, cmap='Reds')

        # 计算特征
        h, w = mask.shape
        area_ratio = np.sum(mask) / (h * w)
        euler = euler_number(mask)
        labeled = label(mask)
        regions = regionprops(labeled)
        main_region = max(regions, key=lambda x: x.area)
        x_offset = abs(main_region.centroid[1] - w / 2)
        solidity = main_region.solidity

        # 添加特征标注
        plt.title(f"Mask {idx:03d}\nArea: {area_ratio:.3f}  Euler: {euler}  XOffset: {x_offset:.1f}")
        plt.axis('off')

        # 保存可视化文件
        filename = f"{output_dir}/mask_{idx:03d}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=100)
        plt.close()

        # 记录特征日志
        log_file.write(f"{idx:03d} | {area_ratio:.4f} | {euler} | {solidity:.3f} | {x_offset:.1f}\n")

    log_file.close()
    print(f"可视化结果已保存至 {output_dir} 目录")

def manual_mask_selector(masks, visual_dir="mask_visualizations"):
    """
    交互式掩码选择器
    """
    while True:
        user_input = input("请输入掩码编号(000-999)或文件名（输入q退出）: ").strip()

        if user_input.lower() == 'q':
            return None

        # 尝试解析数字输入
        if user_input.isdigit():
            mask_id = int(user_input)
            if 0 <= mask_id < len(masks):
                return masks[mask_id]
            print(f"错误：编号超出范围 (0-{len(masks) - 1})")
            continue

        # 尝试解析文件名输入
        if user_input.endswith('.png'):
            try:
                mask_id = int(user_input.split('_')[1].split('.')[0])
                if 0 <= mask_id < len(masks):
                    return masks[mask_id]
            except:
                pass
            print(f"错误：无法解析文件名 {user_input}")
            continue

        print("无效输入，请重新输入")

def calculate_integral_ratio(hist):
    """计算特征积分比值"""
    x = np.arange(256)
    weights = 256 - x

    # 计算积分项
    terms = hist * weights

    # 计算积分值
    total_integral = np.sum(terms)
    partial_integral = np.sum(terms[:150])  # 0-150包含端点

    return partial_integral / total_integral if total_integral != 0 else 0

# 加载模型到CPU
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

image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for idx, filename in enumerate(image_files):
    print(f"\nProcessing {idx + 1}/{len(image_files)}: {filename}")

    try:
        # 构建完整路径
        image_path = os.path.join(INPUT_FOLDER, filename)

        # 加载并调整图像尺寸
        orig_image = Image.open(image_path).convert("RGB")
        w, h = orig_image.size
        scale = 1024 / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        image = np.array(orig_image.resize(new_size))

        # 生成掩码
        masks = mask_generator.generate(image)
        print(f"Generated {len(masks)} masks")

        # 创建输出子目录
        base_name = os.path.splitext(filename)[0]
        output_subdir = os.path.join(OUTPUT_FOLDER, base_name)
        os.makedirs(output_subdir, exist_ok=True)

        # 保存掩码可视化结果和日志
        save_mask_visualization(image, masks, output_dir=os.path.join(output_subdir, "visualizations"))

        # 手动选择掩码
        selected_mask = manual_mask_selector(masks, visual_dir=os.path.join(output_subdir, "visualizations"))
        if selected_mask is not None:
            print(f"Selected mask: {selected_mask}")

            # 新增灰度直方图分析
            try:
                # 转换原始图像为灰度，并调整尺寸
                gray_image = orig_image.convert("L").resize(new_size)
                gray_array = np.array(gray_image)

                # 获取掩码区域像素
                mask_array = selected_mask["segmentation"]  # 掩码已经是调整尺寸后的
                target_pixels = gray_array[mask_array]

                # 计算直方图
                hist, _ = np.histogram(target_pixels, bins=256, range=(0, 255))
                hist = hist / hist.sum()

                # 计算积分比值
                ratio = calculate_integral_ratio(hist)

                # 可视化直方图
                plt.figure(figsize=(10, 6))
                plt.bar(range(256), hist, width=1.0)
                plt.title(f"Gray Histogram\nIntegral Ratio: {ratio:.4f}")
                plt.xlabel("Pixel Value")
                plt.ylabel("Frequency")
                plt.xlim(0, 255)
                plt.savefig(os.path.join(output_subdir, "histogram.png"))
                plt.close()

                # 保存比值结果
                with open(os.path.join(output_subdir, "ratio.txt"), "w") as f:
                    f.write(f"{ratio:.4f}")

                print(f"分析完成：积分比值 {ratio:.4f}")

            except Exception as e:
                print(f"直方图分析失败：{str(e)}")

    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

    # 清理内存
    finally:
        del image, masks, orig_image
        gc.collect()

print("\nAll images processed! Check output folder:", OUTPUT_FOLDER)
