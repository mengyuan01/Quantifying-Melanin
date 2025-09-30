import os
import pandas as pd


def collect_ratios(main_folder, output_excel):

    data = []

    for folder_name in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, folder_name)

        if not os.path.isdir(folder_path):
            continue

        ratio_file = os.path.join(folder_path, "ratio.txt")

        if not os.path.exists(ratio_file):
            print(f"error：{folder_name} exclude ratio.txt")
            continue

        try:
            with open(ratio_file, 'r') as f:
                ratio_value = float(f.read().strip())

            data.append({
                "子文件夹名称": folder_name,
                "比值": ratio_value
            })

        except Exception as e:
            print(f" {folder_name} error：{str(e)}")

    if data:
        df = pd.DataFrame(data)
        df.to_excel(output_excel, index=False)
        print(f"success：{output_excel}")
        print(f"{len(data)}")
    else:
        print("error")


if __name__ == "__main__":
    MAIN_FOLDER = 
    OUTPUT_EXCEL = 
    collect_ratios(MAIN_FOLDER, OUTPUT_EXCEL)

