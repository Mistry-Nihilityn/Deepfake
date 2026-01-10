import json
import random
import os

import torch
from matplotlib import pyplot as plt
from torchvision import transforms

from rearrange import collect
import random
from typing import Dict, List, Tuple
import numpy as np


def split_parts(hierarchical_data: Dict, ratio: Tuple[float, ...] = (0.7, 0.1, 0.2),
                strict: bool = True) -> List[List[str]]:
    if strict:
        return _split_strict(hierarchical_data, ratio)
    else:
        return _split_relaxed(hierarchical_data, ratio)


def _split_strict(hierarchical_data: Dict, ratio: Tuple[float, ...]) -> List[List[str]]:
    video_folders = []
    def extract_video_folders(node: Dict, current_path: str = ""):
        files = node.get('__files__', [])
        has_subdirs = False
        for key, value in node.items():
            if key != '__files__' and isinstance(value, dict):
                has_subdirs = True
                new_path = f"{current_path}/{key}" if current_path else key
                extract_video_folders(value, new_path)
        if files and not has_subdirs:
            video_folders.append({
                'path': current_path,
                'files': files,
                'count': len(files)
            })
        elif files:
            video_folders.append({
                'path': current_path if current_path else 'root',
                'files': files,
                'count': len(files)
            })

    extract_video_folders(hierarchical_data)

    random.shuffle(video_folders)
    total_videos = len(video_folders)
    cumulative_ratios = np.cumsum(ratio[:-1])
    split_indices = [int(r * total_videos) for r in cumulative_ratios]

    video_parts = []
    start_idx = 0
    for end_idx in split_indices:
        video_parts.append(video_folders[start_idx:end_idx])
        start_idx = end_idx
    video_parts.append(video_folders[start_idx:])

    result = []
    for part in video_parts:
        file_list = []
        for video in part:
            file_list.extend(video['files'])
        result.append(sorted(file_list))
    return result


def _split_relaxed(hierarchical_data: Dict, ratio: Tuple[float, ...]) -> List[List[str]]:
    all_files = []
    def collect_files(node: Dict):
        files = node.get('__files__', [])
        all_files.extend(files)

        for key, value in node.items():
            if key != '__files__' and isinstance(value, dict):
                collect_files(value)

    collect_files(hierarchical_data)

    random.shuffle(all_files)
    total_files = len(all_files)
    cumulative_ratios = np.cumsum(ratio[:-1])
    split_indices = [int(r * total_files) for r in cumulative_ratios]

    result = []
    start_idx = 0
    for end_idx in split_indices:
        result.append(sorted(all_files[start_idx:end_idx]))
        start_idx = end_idx
    result.append(sorted(all_files[start_idx:]))
    return result


def get_folder(root, dataset_name, data_domain):

    with open(os.path.join(root, dataset_name, "label.json"), 'r') as f:
        images = json.load(f)

    if dataset_name == "FaceForensics++":
        real = {k:images.get(k, {}) for k in ["original_sequences"]}
        fake = {k:images.get(k, {}) for k in ["manipulated_sequences"]}
    elif dataset_name == "Celeb-DF-v2":
        real = {k:images.get(k, {}) for k in ["Celeb-real", "YouTube-real"]}
        fake = {k:images.get(k, {}) for k in ["Celeb-synthesis"]}
    elif dataset_name in ["deepfacelab", "heygen_new", "stargan", "starganv2", "styleclip", "SageMaker"]:
        real = {k:images.get(k, {}) for k in ["real"]}
        fake = {k:images.get(k, {}) for k in ["fake"]}
    elif dataset_name in ["DiT", "fsgan", "lia", "StyleGAN2", "faceswap", "heygen", "simswap"]:
        real = {}
        fake = {k:images.get(k, {}) for k in data_domain}
    else:
        real = {}
        fake = {}
        print(f"未知数据集 {dataset_name}")
    return real, fake


def load_train(config, logger):
    train_images = {"fake": [], "real": []}
    val_images = {"fake": [], "real": []}
    test_images = {"fake": [], "real": []}

    for label in ["real", "fake"]:
        for dataset_name in config["dataset"]["train"][f"{label}_dataset_names"]:
            collect(os.path.join(config["dataset"]["train"]["root"], dataset_name))
            folder_dict = get_folder(config["dataset"]["train"]["root"],
                                     dataset_name,
                                     config["dataset"]["train"]["domain"])[0 if label == "real" else 1]
            if config["dataset"]["train"]["type"] == "in-domain":
                sub_train, sub_val, sub_test = split_parts(folder_dict, (
                    config["dataset"]["train"]["split"]["train"],
                    config["dataset"]["train"]["split"]["val"],
                    config["dataset"]["train"]["split"]["test"]
                ), strict=True)
                train_images[label].extend(sub_train)
                val_images[label].extend(sub_val)
                test_images[label].extend(sub_test)
                logger.info(f"{dataset_name} {label} images: {len(sub_train)}/{len(sub_val)}/{len(sub_test)}")
            else:
                sub_train, sub_val = split_parts(folder_dict, (
                    config["dataset"]["train"]["split"]["train"],
                    config["dataset"]["train"]["split"]["val"]
                ), strict=True)
                train_images[label].extend(sub_train)
                val_images[label].extend(sub_val)
                logger.info(f"{dataset_name} {label} images: {len(sub_train)}/{len(sub_val)}")
    return train_images, val_images, test_images


def load_test(config, logger):
    test_images = {"fake": [], "real": []}
    for label in ["real", "fake"]:
        for dataset_name in config["dataset"]["test"][f"{label}_dataset_names"]:
            collect(os.path.join(config["dataset"]["test"]["root"], dataset_name))
            folder_dict = get_folder(config["dataset"]["test"]["root"],
                                     dataset_name,
                                     config["dataset"]["test"]["domain"])[0 if label == "real" else 1]
            sub_test, = split_parts(folder_dict, (
                config["dataset"]["test"]["split"]["test"],
            ), strict=True)
            test_images[label].extend(sub_test)
            logger.info(f"{dataset_name} {label} images: {len(sub_test)}")
    return test_images


def show_samples(dataset, n=5):
    """
    显示数据集中的样本

    Args:
        dataset: AbstractDataset 实例
        n: 要显示的样本数量
    """
    indices = random.sample(range(len(dataset)), n)
    samples = [dataset[i] for i in indices]

    fig, axes = plt.subplots(1, n, figsize=(15, 3))

    if hasattr(dataset, 'config') and 'mean' in dataset.config and 'std' in dataset.config:
        mean = torch.tensor(dataset.config['mean']).view(3, 1, 1)
        std = torch.tensor(dataset.config['std']).view(3, 1, 1)

    for ax, (image_tensor, label) in zip(axes, samples):
        if 'mean' in locals() and 'std' in locals():
            image_tensor = image_tensor * std + mean

        image_tensor = torch.clamp(image_tensor, 0, 1)

        to_pil = transforms.ToPILImage()
        img = to_pil(image_tensor)

        ax.imshow(img)
        ax.set_title(f"Label: {'Fake' if label == 1 else 'Real'}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()
