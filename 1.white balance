import os
import cv2
import numpy as np
from tqdm import tqdm


def gray_world_balance(img):

    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])

    avg_gray = (avg_b + avg_g + avg_r) / 3
    scale_b = avg_gray / avg_b
    scale_g = avg_gray / avg_g
    scale_r = avg_gray / avg_r

    balanced = img.copy()
    balanced[:, :, 0] = np.clip(img[:, :, 0] * scale_b, 0, 255)
    balanced[:, :, 1] = np.clip(img[:, :, 1] * scale_g, 0, 255)
    balanced[:, :, 2] = np.clip(img[:, :, 2] * scale_r, 0, 255)

    return balanced.astype(np.uint8)


def process_folder(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)


    valid_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    files = [f for f in os.listdir(input_folder)
             if os.path.splitext(f)[1].lower() in valid_ext]


    for filename in tqdm(files, desc="Processing Images"):
        try:
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

            balanced = gray_world_balance(img)

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path,cv2.cvtColor(balanced, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"\n处理 {filename} 时出错: {str(e)}")


if __name__ == "__main__":
    INPUT_FOLDER = r"" 
    OUTPUT_FOLDER = r"" 

    process_folder(INPUT_FOLDER, OUTPUT_FOLDER)
    print("\n处理完成！输出目录：", OUTPUT_FOLDER)

