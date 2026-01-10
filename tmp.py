import os
import csv
from pathlib import Path


def find_and_save_symlinks(folder_path: str, output_csv: str = "symlinks.csv"):
    """
    查找并保存文件夹中的所有符号链接到CSV文件

    Args:
        folder_path: 要扫描的文件夹路径
        output_csv: 输出CSV文件路径
    """
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹不存在: {folder_path}")
        return

    symlinks_data = []

    # 遍历文件夹中的所有项
    for root, dirs, files in os.walk(folder_path):
        # 检查目录中的符号链接
        for dir_name in dirs[:]:  # 使用副本遍历，避免修改原列表的问题
            full_path = os.path.join(root, dir_name)
            if os.path.islink(full_path):
                try:
                    target = os.readlink(full_path)
                    symlinks_data.append({
                        'type': '目录',
                        'name': dir_name,
                        'path': full_path,
                        'target': target,
                        'target_exists': os.path.exists(target)
                    })
                except OSError as e:
                    print(f"无法读取链接 {full_path}: {e}")

        # 检查文件中的符号链接
        for file_name in files:
            full_path = os.path.join(root, file_name)
            if os.path.islink(full_path):
                try:
                    target = os.readlink(full_path)
                    symlinks_data.append({
                        'type': '文件',
                        'name': file_name,
                        'path': full_path,
                        'target': target,
                        'target_exists': os.path.exists(target)
                    })
                except OSError as e:
                    print(f"无法读取链接 {full_path}: {e}")

    # 保存到CSV
    if symlinks_data:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['type', 'name', 'path', 'target', 'target_exists']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(symlinks_data)

        print(f"找到 {len(symlinks_data)} 个符号链接")
        print(f"已保存到: {output_csv}")

        # 打印简要统计
        dir_links = sum(1 for item in symlinks_data if item['type'] == '目录')
        file_links = sum(1 for item in symlinks_data if item['type'] == '文件')
        valid_links = sum(1 for item in symlinks_data if item['target_exists'])

        print(f"  目录链接: {dir_links}")
        print(f"  文件链接: {file_links}")
        print(f"  有效链接: {valid_links}")
        print(f"  失效链接: {len(symlinks_data) - valid_links}")
    else:
        print("没有找到任何符号链接")


# 使用示例
if __name__ == "__main__":
    target_folder = "../workdata/val"
    find_and_save_symlinks(target_folder, "symlinks_report.csv")