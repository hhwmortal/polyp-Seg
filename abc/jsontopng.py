import os
import subprocess
import shutil
from PIL import Image
import numpy as np

# 输入输出文件夹路径
input_folder = "C:/Users/gzh/Desktop/12121"  # 输入文件夹路径，包含多个JSON标记文件
final_output_folder = "C:/Users/gzh/Desktop/333"  # 最终输出路径

# 确保输出文件夹存在
os.makedirs(final_output_folder, exist_ok=True)

# 遍历输入文件夹中的JSON标记文件
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        # 构建输入JSON文件的完整路径
        input_path = os.path.join(input_folder, filename)

        # 使用labelme_json_to_dataset命令行工具进行转换
        command = f"labelme_json_to_dataset {input_path} -o {final_output_folder}/{filename[:-5]}"
        subprocess.run(command, shell=True)

        # 构建label.png的路径
        label_path = os.path.join(final_output_folder, f"{filename[:-5]}/label.png")

        # 移动label.png到输出文件夹并处理颜色
        if os.path.exists(label_path):
            # 重命名为原始文件名并移动到最终输出文件夹
            new_label_path = os.path.join(final_output_folder, f"{os.path.splitext(filename)[0]}_label.png")

            # 处理颜色，将非黑色区域替换为白色
            img = Image.open(label_path).convert("RGB")
            data = np.array(img)

            # 创建掩码，找到所有非黑色区域
            mask = (data[:, :, 0] != 0) | (data[:, :, 1] != 0) | (data[:, :, 2] != 0)
            data[mask] = [255, 255, 255]  # 将非黑色区域替换为白色

            # 保存处理后的图像
            processed_img = Image.fromarray(data)
            processed_img.save(new_label_path)

            print(f"Successfully generated {new_label_path}")
        else:
            print(f"Failed to generate label.png for {filename}")

        # 删除临时文件夹
        shutil.rmtree(os.path.join(final_output_folder, filename[:-5]))  # 删除临时文件夹及其内容