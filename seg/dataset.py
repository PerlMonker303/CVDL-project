import cv2
import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt


def enhance_index(index):
    if index == 0:
        return '00000'
    digits = 0
    res = ''
    cp = index
    while cp > 0:
        digits += 1
        cp = (int)(cp / 10)

    while digits < 5:
        res += '0'
        digits += 1

    res += str(index)

    return res


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_dim, indices):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_dim = image_dim
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, f'rs{enhance_index(self.indices[index])}.jpg')
        mask_path = os.path.join(self.mask_dir, f'rs{enhance_index(self.indices[index])}.png')
        image = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))

        mask[mask == 18] = 17
        mask_processed = np.zeros_like(mask)
        rail_color = 17  # 18 for tramway
        mask_processed = np.where(mask == rail_color, 1, mask_processed)
        mask = mask_processed

        image = cv2.resize(image, dsize=self.image_dim, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=self.image_dim, interpolation=cv2.INTER_LINEAR)

        # plt.imshow(image, cmap='gray')
        # plt.show()
        # plt.imshow(mask, cmap='gray')
        # plt.show()

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        return image, mask
