import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)  # lists all the files that are in that directory

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        '''Instructions:
        - comment/uncomment the parts between [START]...[END] based on the problem you wish to solve
        '''
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index]).replace(".jpg", ".png")
        image = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))
        # RAILS - using train tracks and tramway tracks
        # [START]
        mask[mask == 18] = 17
        mask_processed = np.zeros_like(mask)
        rail_color=17  # 18 for tramway
        mask_processed = np.where(mask == rail_color, 1, mask_processed)
        mask = mask_processed
        # [END]
        '''
        # CELLS
        # [START]
        # all pixels > 0 are marked as 1
        mask = np.where(mask > 0, 1, mask)
        # [END]
        '''
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        else:
            mask = torch.from_numpy(mask)

        image = image.float()
        mask = mask.long()

        return image, mask
