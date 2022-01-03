import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from utils import enhance_index


class CustomDataset(Dataset):
    def __init__(self, image_dir, alpha, labels_path, indices):
        self.alpha = alpha
        self.image_dir = image_dir
        self.labels_path = labels_path
        self.indices = indices
        self.labels = self.read_labels()

    def __len__(self):
        return len(self.indices)

    def read_labels(self):
        labels = {}
        f = open(self.labels_path, 'r')
        lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            labels[int(line[0])] = int(line[1])
        f.close()
        return labels

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, f'{enhance_index(self.indices[index])}.png')
        image = np.array(Image.open(img_path).convert("L"))
        cls = self.labels[index]

        # plt.imshow(image, cmap='gray')
        # plt.show()

        image = torch.from_numpy(image)

        return image, cls

