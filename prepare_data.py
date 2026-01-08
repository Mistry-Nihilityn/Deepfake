import json
import random
import os

import torch
import yaml
from matplotlib import pyplot as plt
from torchvision import transforms

from dataset.AbstractDataset import AbstractDataset
from rearrange import collect

def load_train_val(config):
    train_images = {"fake": [], "real": []}
    val_images = {"fake": [], "real": []}
    
    for dataset_name in config["dataset"]["train"]["real_dataset_names"]:

        collect(dataset_name)

        with open(os.path.join(config["dataset"]["train"]["root"], dataset_name, "label.json"), 'r') as f:
            images = json.load(f)
        real_folders = list(images["train"]["real"].keys())
        random.shuffle(real_folders)

        real_train_cnt = int(len(real_folders) * config["dataset"]["train"]["split"]["train"])
        real_val_cnt = int(len(real_folders) * config["dataset"]["train"]["split"]["val"])
        for folder in real_folders[:real_train_cnt]:
            train_images["real"].extend(images["train"]["real"][folder])
        for folder in real_folders[real_train_cnt:real_train_cnt+real_val_cnt]:
            val_images["real"].extend(images["train"]["real"][folder])

    for dataset_name in config["dataset"]["train"]["fake_dataset_names"]:

        collect(dataset_name)

        with open(os.path.join(config["dataset"]["train"]["root"], dataset_name, "label.json"), 'r') as f:
            images = json.load(f)
        fake_folders = list(images["train"]["fake"].keys())
        random.shuffle(fake_folders)

        fake_train_cnt = int(len(fake_folders) * config["dataset"]["train"]["split"]["train"])
        fake_val_cnt = int(len(fake_folders) * config["dataset"]["train"]["split"]["val"])
        for folder in fake_folders[:fake_train_cnt]:
            train_images["fake"].extend(images["train"]["fake"][folder])
        for folder in fake_folders[fake_train_cnt:fake_train_cnt+fake_val_cnt]:
            val_images["fake"].extend(images["train"]["fake"][folder])
    return train_images, val_images


def load_test(config):
    test_images = {"fake": [], "real": []}
    for dataset_name in config["dataset"]["test"]["real_dataset_names"]:

        collect(dataset_name)

        with open(os.path.join(config["dataset"]["test"]["root"], dataset_name, "label.json"), 'r') as f:
            images = json.load(f)
        real_folders = list(images["test"]["real"].keys())
        real_test_cnt = int(len(real_folders) * config["dataset"]["test"]["split"]["test"])
        for folder in real_folders[:real_test_cnt]:
            test_images["real"].extend(images["test"]["real"][folder])

    for dataset_name in config["dataset"]["test"]["fake_dataset_names"]:

        collect(dataset_name)

        with open(os.path.join(config["dataset"]["test"]["root"], dataset_name, "label.json"), 'r') as f:
            images = json.load(f)
        fake_folders = list(images["test"]["fake"].keys())
        fake_test_cnt = int(len(fake_folders) * config["dataset"]["test"]["split"]["test"])
        for folder in fake_folders[:fake_test_cnt]:
            test_images["fake"].extend(images["test"]["fake"][folder])

    print(f"\nTest set:")
    print(f"  - fake: {len(test_images['fake']):,}")
    print(f"  - real: {len(test_images['real']):,}")
    print(f"  - total: {len(test_images['fake']) + len(test_images['real']):,}")

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


if __name__ == "__main__":
    with open("config/run_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    train_images, val_images = load_train_val(config)
    test_images = load_test(config)
    train_loader = AbstractDataset(train_images, config, mode="train")
    test_loader = AbstractDataset(test_images, config, mode="test")

    show_samples(train_loader, n=5)
    show_samples(train_loader, n=5)
    show_samples(test_loader, n=5)
    show_samples(test_loader, n=5)
