import os
import pandas as pd


def collect_ratios(main_folder, output_excel):
    """
    从主文件夹下的子文件夹收集ratio.txt数据
    生成包含子文件夹名称和比值的Excel文件
    """
    data = []

    # 遍历主文件夹下的所有子文件夹
    for folder_name in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, folder_name)

        # 确保是目录
        if not os.path.isdir(folder_path):
            continue

        ratio_file = os.path.join(folder_path, "ratio.txt")

        # 检查ratio.txt是否存在
        if not os.path.exists(ratio_file):
            print(f"警告：{folder_name} 中未找到ratio.txt")
            continue

        try:
            # 读取文件内容
            with open(ratio_file, 'r') as f:
                ratio_value = float(f.read().strip())

            data.append({
                "子文件夹名称": folder_name,
                "比值": ratio_value
            })

        except Exception as e:
            print(f"处理 {folder_name} 时出错：{str(e)}")

    # 创建DataFrame并保存
    if data:
        df = pd.DataFrame(data)
        df.to_excel(output_excel, index=False)
        print(f"成功生成Excel文件：{output_excel}")
        print(f"共处理 {len(data)} 条有效数据")
    else:
        print("未找到有效数据")


if __name__ == "__main__":
    # 配置路径
    MAIN_FOLDER = r""  # 修改为你的主文件夹路径
    OUTPUT_EXCEL = r"ratios_results.xlsx"  # 输出Excel文件名

    # 执行收集
    collect_ratios(MAIN_FOLDER, OUTPUT_EXCEL)
