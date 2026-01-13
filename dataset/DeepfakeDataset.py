import sys

import torch

sys.path.append('.')
import os

import numpy as np
import cv2
import random
from PIL import Image
from torch.utils import data
from torchvision import transforms as T, transforms
from abc import abstractmethod, ABCMeta


class AbstractDataset(data.Dataset, metaclass=ABCMeta):
    def __init__(self, images, resolution=224, balance=True, augment_config=None, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.balance = balance
        self.augment_config = augment_config
        self.resolution = resolution
        self.mean = mean
        self.std = std

        self.real_imgs = {name: images[name]["real"] for name in images.keys() if len(images[name]["real"])}
        self.fake_imgs = {name: images[name]["fake"] for name in images.keys() if len(images[name]["fake"])}
        self.real_cnt = self.fake_cnt = None
        self.transform = self.init_data_aug_method()

    def init_data_aug_method(self):
        if self.augment_config:
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.resolution),
                transforms.CenterCrop(self.resolution),
                transforms.RandomHorizontalFlip(p=self.augment_config['flip_prob']),
                transforms.ColorJitter(
                    brightness=self.augment_config['brightness_limit'],
                    contrast=self.augment_config['contrast_limit'],
                    saturation=self.augment_config['saturation_limit'],
                    hue=self.augment_config['hue_limit']
                ),
            ])
        else:
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.resolution),
                transforms.CenterCrop(self.resolution),
            ])
        return trans

    def load_rgb(self, file_path):
        size = self.resolution
        assert os.path.exists(file_path), f"{file_path} does not exist"
        img = cv2.imread(file_path)
        if img is None:
            img = Image.open(file_path)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            if img is None:
                raise ValueError('Loaded image is None: {}'.format(file_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return np.array(img, dtype=np.float32)/255

    def to_tensor(self, img):
        if isinstance(img, torch.Tensor):
            return img
        return T.ToTensor()(img)

    def normalize(self, img):
        normalize = T.Normalize(mean=self.mean, std=self.std)
        return normalize(img)

    def data_aug(self, img, augmentation_seed=None):
        if augmentation_seed is not None:
            random.seed(augmentation_seed)
            np.random.seed(augmentation_seed)

        augmented_img = self.transform(img)

        if augmentation_seed is not None:
            random.seed()
            np.random.seed()

        return augmented_img

    @abstractmethod
    def __getitem__(self, index, no_norm=False):
        ...

    def __len__(self):
        return self.real_cnt + self.fake_cnt


def inplace_sample(lst:list, num_sample):
    samples = sorted(random.sample(list(range(len(lst))), num_sample))
    res = [lst[i] for i in samples]
    lst.clear()
    lst.extend(res)


class TrainDataset(AbstractDataset):
    def __init__(self, images, sample_per_class, *args, **kwargs):
        super().__init__(images, *args, **kwargs)
        if self.balance:
            self.sample_per_class = sample_per_class
            tot = 0
            for sub in self.fake_imgs.values():
                if len(sub) == 0:
                    continue
                tot += self.sample_per_class
                if self.sample_per_class < len(sub):
                    inplace_sample(sub, self.sample_per_class)

            lengths = sorted([len(sub) for sub in self.real_imgs.values()])

            l = 1
            ans = r = lengths[-1]

            def check(m):
                return sum(map(lambda x: min(x, m), lengths)) > tot

            while l <= r:
                mid = (l+r)//2
                if check(mid):
                    r = mid-1
                    ans = mid
                else:
                    l = mid+1

            classes = len(self.real_imgs)
            prefix = 0
            for i, sub in enumerate(self.real_imgs.values()):
                num_sample = min(ans if prefix + (classes-i)*ans <= tot else ans - 1, len(sub))
                prefix += num_sample
                if num_sample < len(sub):
                    inplace_sample(sub, num_sample)
            self.real_cnt = self.fake_cnt = tot
        else:
            self.sample_per_class = None

            self.real_cnt = 0
            for sub in self.real_imgs.values():
                self.real_cnt += len(sub)

            self.fake_cnt = 0
            for sub in self.fake_imgs.values():
                self.fake_cnt += len(sub)

        # show_samples(self, 5)

    def __getitem__(self, index, no_norm=False):
        if index < self.real_cnt:
            img = None
            label = 0
            clazz = -1
            path = "Unkown"
            for name, sub in self.real_imgs.items():
                if index < len(sub):
                    path = sub[index]
                    img = self.load_rgb(path)
                    break
                else:
                    index -= len(sub)

            if img is None:
                path = random.choice(random.choice([sub for sub in self.real_imgs.values() if len(sub) > 0]))
                img = self.load_rgb(path)
        else:
            index -= self.real_cnt
            img = None
            label = 1
            clazz = None
            path = "Unkown"
            for cls_idx, sub in enumerate(self.fake_imgs.values()):
                clazz = cls_idx
                if index < len(sub):
                    path = sub[index]
                    img = self.load_rgb(path)
                    break
                else:
                    if self.balance:
                        if index < self.sample_per_class:
                            path1, path2 = random.sample(sub, 2)
                            path = f"{path1} | {path2}"
                            ratio = random.random()
                            img = self.load_rgb(path1) * ratio + self.load_rgb(path2) * (1 - ratio)
                            break
                        else:
                            index -= self.sample_per_class
                    else:
                        index -= len(sub)
        img = self.data_aug(img)
        if img is None or label is None or clazz is None:
            print(index)
            raise Exception
        return img, label, clazz, path


class TestDataset(AbstractDataset):
    def __init__(self, images, *args, **kwargs):
        super().__init__(images, *args, **kwargs)
        self.real_list = []
        for sub in self.real_imgs.values():
            for img in sub:
                self.real_list.append({"clazz": -1, "img": img})
        self.fake_list = []
        for idx, sub in enumerate(self.fake_imgs.values()):
            for img in sub:
                self.fake_list.append({"clazz": idx, "img": img})
        if self.balance:
            cnt = min(len(self.real_list), len(self.fake_list))
            if cnt < len(self.real_list):
                inplace_sample(self.real_list, cnt)
            if cnt < len(self.fake_list):
                inplace_sample(self.fake_list, cnt)
            self.real_cnt = self.fake_cnt = cnt
        else:
            self.real_cnt = len(self.real_list)
            self.fake_cnt = len(self.fake_list)
            self.sample_per_class = None

    def __getitem__(self, index, no_norm=False):
        if index < len(self.real_list):
            path = self.real_list[index]["img"]
            img = self.load_rgb(path)
            clazz = self.real_list[index]["clazz"]
            label = 0
        else:
            index -= len(self.real_list)
            path = self.fake_list[index]["img"]
            img = self.load_rgb(path)
            clazz = self.fake_list[index]["clazz"]
            label = 1
        img = self.data_aug(img)
        return img, label, clazz, path
