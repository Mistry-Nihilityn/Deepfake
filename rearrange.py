import os
import json
import argparse
from asyncio import Lock
from collections import defaultdict
import concurrent.futures

import tqdm
from PIL import Image

dataset_names = [
    "deepfacelab",
    "DiT",
    "FaceForensics++",
    "Celeb-DF-v2"
]

root_path = r"E:\DeepFake\Dataset\frames"


def is_broken_image(path):
    try:
        Image.open(path).verify()
        return False
    except:
        return True

def get_second_last_dir(path):
    # 将路径标准化并分割成各个部分
    path_parts = os.path.normpath(path).split(os.sep)
    # 检查是否有倒数第二层
    if len(path_parts) >= 2:
        return path_parts[-2]
    else:
        return ""


def get_folder(dataset_name):
    if dataset_name == "FaceForensics++":
        real_train = [os.path.join(root_path, dataset_name, "original_sequences")]
        fake_train = [os.path.join(root_path, dataset_name, "manipulated_sequences")]
        real_test = []
        fake_test = []
    elif dataset_name == "Celeb-DF-v2":
        real_train = []
        fake_train = []
        real_test = [os.path.join(root_path, dataset_name, "Celeb-real"),
                     os.path.join(root_path, dataset_name, "YouTube-real")]
        fake_test = [os.path.join(root_path, dataset_name, "Celeb-synthesis")]
    elif dataset_name in ["deepfacelab", "heygen_new", "stargan", "starganv2", "styleclip"]:
        real_train = [os.path.join(root_path, dataset_name, "real")]
        fake_train = [os.path.join(root_path, dataset_name, "fake")]
        real_test = []
        fake_test = []
    elif dataset_name in ["DiT", "fsgan", "lia", "StyleGAN2", "faceswap", "heygen", "simswap"]:
        real_train = []
        fake_train = [os.path.join(root_path, dataset_name, "ff")]
        real_test = []
        fake_test = [os.path.join(root_path, dataset_name, "cdf")]
    else:
        real_train = []
        fake_train = []
        real_test = []
        fake_test = []
        print(f"未知数据集 {dataset_name}")
        return None, None, None, None

    for folder in real_train + fake_train + real_test + fake_test:
        if not os.path.exists(folder):
            print(f"错误: 文件夹 '{folder}' 不存在！")
            return None, None, None, None

        if not os.path.isdir(folder):
            print(f"错误: '{folder}' 不是文件夹！")
            return None, None, None, None

    return real_train, fake_train, real_test, fake_test


def collect_single_dataset(folders):
    if folders is None:
        return {}, 0

    max_workers = os.cpu_count()

    all_png_files = []
    for folder in folders:
        for root, dirs, files in tqdm.tqdm(os.walk(folder), desc="扫描文件夹"):
            for file in files:
                if file.lower().endswith('.png'):
                    full_path = os.path.join(root, file)
                    all_png_files.append(os.path.abspath(full_path))

    if not all_png_files:
        return {}, 0

    png_files = defaultdict(list)
    cnt = 0

    def process_file(file_path):
        nonlocal cnt
        if not is_broken_image(file_path):
            second_last_dir = get_second_last_dir(file_path)
            png_files[second_last_dir].append(file_path)
            cnt += 1
        return 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, file_path) for file_path in all_png_files]
        for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures),
                total=len(all_png_files)
        ):
            try:
                future.result()
            except Exception as e:
                print(e)

    return dict(png_files), cnt


def collect(dataset_name):
    real_tain, fake_train, real_test, fake_test = get_folder(dataset_name)
    file = os.path.join(root_path, dataset_name, "label.json")
    if os.path.exists(file):
        print(f"JSON文件已存在: {file}")
        return

    # 修改扫描函数以支持绝对路径选项
    real_train_files, real_train_cnt = collect_single_dataset(real_tain)
    fake_train_files, fake_train_cnt = collect_single_dataset(fake_train)
    real_test_files, real_test_cnt = collect_single_dataset(real_test)
    fake_test_files, fake_test_cnt = collect_single_dataset(fake_test)
    # 将结果写入JSON文件
    with open(file, 'w', encoding='utf-8') as f:
        json.dump({
            "train": {
            "real": real_train_files,
            "fake": fake_train_files
        },
            "test": {
            "real": real_test_files,
            "fake": fake_test_files
        }
        }, f, indent=4, ensure_ascii=False)

    print(f"label 0 rael: {real_train_cnt} 个文件")
    print(f"label 1 fake: {fake_train_cnt} 个文件")
    print(f"JSON文件已保存到: {file}")


def main():

    dataset_type = "train"
    for dataset_name in dataset_names:
        collect(dataset_name, dataset_type)

if __name__ == "__main__":
    main()