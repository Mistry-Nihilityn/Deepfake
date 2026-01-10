import os
import json
import argparse
from asyncio import Lock
from collections import defaultdict
import concurrent.futures

import tqdm
from PIL import Image


def is_broken_image(path):
    try:
        Image.open(path).verify()
        return False
    except Exception as e:
        print(f"Image at {path} is broken! {e}")
        return True

def collect_single_dataset(folder_path):
    if folder_path is None or not os.path.isdir(folder_path):
        return {}, 0

    max_workers = os.cpu_count()
    all_png_files = []
    base_path = os.path.abspath(folder_path)

    for root, dirs, files in tqdm.tqdm(os.walk(folder_path), desc="扫描文件夹"):
        for file in files:
            if file.lower().endswith('.png') or file.lower().endswith('.jpg') or file.lower().endswith('.jpeg'):
                full_path = os.path.join(root, file)
                all_png_files.append(os.path.abspath(full_path))

    if not all_png_files:
        return {}, 0

    hierarchical_data = {}
    cnt = 0

    def process_file(file_path):
        nonlocal cnt, hierarchical_data
        if not is_broken_image(file_path):
            rel_path = os.path.relpath(file_path, base_path)
            path_parts = rel_path.split(os.sep)
            current_level = hierarchical_data
            for part in path_parts[:-1]:
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]
            if not isinstance(current_level, dict):
                current_level = {}
            if not isinstance(current_level.get('__files__', None), list):
                current_level['__files__'] = []
            current_level['__files__'].append(os.path.abspath(file_path))
            cnt += 1
        return 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, file_path) for file_path in all_png_files]
        for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures),
                total=len(all_png_files),
                desc="处理文件"
        ):
            try:
                future.result()
            except Exception as e:
                print(f"处理文件时出错: {e}")

    return hierarchical_data, cnt

def collect(path):
    file = os.path.join(path, "label.json")
    if os.path.exists(file):
        print(f"JSON文件已存在: {file}")
        return

    label_dict, cnt = collect_single_dataset(path)
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(label_dict, f, indent=4, ensure_ascii=False)
    print(f"JSON文件已保存到: {file} 一共 {cnt} 个样本")


def main():
    ROOT_PATH = "../dataset"
    DATASET_NAMES = ["DiT", "lia", "heygen", "heygen_new", "deepfacelab", "StyleGAN2", "fsgan", "faceswap", "FaceForensics++", "Celeb-DF-v2",
                     "SageMaker", "simswap", "stargan", "starganv2", "styleclip"]

    for name in DATASET_NAMES:
        collect(os.path.join(ROOT_PATH, name))

if __name__ == "__main__":
    main()