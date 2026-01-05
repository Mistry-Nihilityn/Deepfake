import sys

sys.path.append('.')
import os

import numpy as np
from copy import deepcopy
import cv2
import random
from PIL import Image

from torch.utils import data
from torchvision import transforms as T

import albumentations as A

from dataset.albu import IsotropicResize


class AbstractDataset(data.Dataset):
    def __init__(self, images, config, mode='train'):
        self.config = config
        self.mode = mode

        if self.config['dataset']["train" if mode == "train" or mode == "val" else "test"]["balance"]:
            random.shuffle(images["real"])
            random.shuffle(images["fake"])
            l = min(len(images["real"]), len(images["fake"]))
            images["real"] = images["real"][:l]
            images["fake"] = images["fake"][:l]

        self.real_cnt = len(images["real"])
        self.fake_cnt = len(images["fake"])
        self.image_list = images["real"] + images["fake"]
        self.label_list = [0] * len(images["real"]) + [1] * len(images["fake"])
        self.transform = self.init_data_aug_method()

    def init_data_aug_method(self):
        if self.config["dataset"]["train"]["use_data_augmentation"]:
            aug_config = self.config["dataset"]["train"]['data_aug']
            trans = A.Compose([
                A.HorizontalFlip(p=aug_config['flip_prob']),
                A.Rotate(limit=aug_config['rotate_limit'],p=aug_config['rotate_prob']),
                A.GaussianBlur(blur_limit=aug_config['blur_limit'],p=aug_config['blur_prob']),
                A.OneOf([
                    IsotropicResize(
                        max_side=self.config['resolution'],
                        interpolation_down=cv2.INTER_AREA,
                        interpolation_up=cv2.INTER_CUBIC
                    ),
                    IsotropicResize(
                        max_side=self.config['resolution'],
                        interpolation_down=cv2.INTER_AREA,
                        interpolation_up=cv2.INTER_LINEAR
                    ),
                    IsotropicResize(
                        max_side=self.config['resolution'],
                        interpolation_down=cv2.INTER_LINEAR,
                        interpolation_up=cv2.INTER_LINEAR
                    ),
                ], p=1),
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=aug_config['brightness_limit'],
                        contrast_limit=aug_config['contrast_limit'],
                        p=1.0
                    ),
                    A.FancyPCA(p=1.0),
                    A.HueSaturationValue(p=1.0)
                ], p=0.5),
                A.ImageCompression(
                    quality_range=(aug_config['quality_lower'],
                                   aug_config['quality_upper']),
                    p=0.5
                )
            ],
                keypoint_params=None
            )
            # trans = A.Compose([])
        else:
            trans = A.Compose([])
        return trans

    def load_rgb(self, file_path):
        size = self.config['resolution']
        assert os.path.exists(file_path), f"{file_path} does not exist"
        img = cv2.imread(file_path)
        if img is None:
            img = Image.open(file_path)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            if img is None:
                raise ValueError('Loaded image is None: {}'.format(file_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(np.array(img, dtype=np.uint8))

    def to_tensor(self, img):
        return T.ToTensor()(img)

    def normalize(self, img):
        mean = self.config['mean']
        std = self.config['std']
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    def data_aug(self, img, augmentation_seed=None):
        if augmentation_seed is not None:
            random.seed(augmentation_seed)
            np.random.seed(augmentation_seed)

        kwargs = {'image': img}

        transformed = self.transform(**kwargs)

        augmented_img = transformed['image']

        if augmentation_seed is not None:
            random.seed()
            np.random.seed()

        return augmented_img

    def __getitem__(self, index, no_norm=False):
        image_path = self.image_list[index]
        label = self.label_list[index]
        augmentation_seed = None

        try:
            image = self.load_rgb(image_path)
        except Exception as e:
            print(f"Error loading image at index {index}: {e}")
            return self.__getitem__(0)
        image = np.array(image)

        if self.mode == 'train' and self.config["dataset"]["train"]['use_data_augmentation']:
            image_trans = self.data_aug(image, augmentation_seed)
        else:
            image_trans = deepcopy(image)

        if not no_norm:
            image_trans = self.normalize(self.to_tensor(image_trans))

        return image_trans, label

    def __len__(self):
        assert len(self.image_list) == len(self.label_list), 'Number of images and labels are not equal'
        return len(self.image_list)
