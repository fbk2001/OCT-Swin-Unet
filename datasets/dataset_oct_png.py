import os
import random

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def apply_label_map(label, label_map=None, max_valid_label=8):
    mapped = np.array(label, dtype=np.int64, copy=True)
    if label_map:
        for src, dst in label_map.items():
            mapped[mapped == int(src)] = int(dst)
    mapped[mapped == 9] = 0
    mapped[(mapped < 0) | (mapped > max_valid_label)] = 0
    return mapped


class RandomGenerator(object):
    def __init__(self, output_size, augment=True):
        self.output_size = output_size
        self.augment = augment

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if self.augment:
            if random.random() > 0.5:
                image, label = random_rot_flip(image, label)
            elif random.random() > 0.5:
                image, label = random_rotate(image, label)

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class OCT_PNG_dataset(Dataset):
    def __init__(self, base_dir='./data/OCTLayers', list_dir='./lists/OCTLayers', split='train', transform=None,
                 image_dir='image', mask_dir='masks', label_map=None, max_valid_label=8):
        self.transform = transform
        self.split = split
        self.data_dir = base_dir
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.label_map = label_map
        self.max_valid_label = max_valid_label
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case_name = self.sample_list[idx].strip('\n').split(",")[0]
        if case_name.endswith('.png'):
            case_stem = case_name[:-4]
        else:
            case_stem = case_name

        image_path = os.path.join(self.data_dir, self.image_dir, case_stem + '.png')
        label_path = os.path.join(self.data_dir, self.mask_dir, case_stem + '.png')

        image = np.array(Image.open(image_path).convert('L'), dtype=np.float32)
        label = np.array(Image.open(label_path), dtype=np.int64)
        label = apply_label_map(label, label_map=self.label_map, max_valid_label=self.max_valid_label)

        sample = {'image': image, 'label': label, 'case_name': case_stem}
        if self.transform:
            transformed = self.transform({'image': image, 'label': label})
            transformed['case_name'] = case_stem
            sample = transformed
        return sample
