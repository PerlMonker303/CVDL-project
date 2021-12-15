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
        self.offset = -1
        self.labels = self.read_labels()
        self.images = os.listdir(self.image_dir)  # lists all the files that are in that directory

    def __len__(self):
        return len(self.images)

    def read_labels(self):
        labels = {}
        f = open(self.labels_path, 'r')
        lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            labels[int(line[0])] = int(line[1])

            if self.offset == -1:
                self.offset = int(line[0])
        f.close()
        return labels

    def __getitem__(self, index):
        index = index + self.offset  # add an offset for the validation data. The offset is set to be the first entry in the validation labels file
        img_path = os.path.join(self.image_dir, f'{enhance_index(index)}.png')
        image = np.array(Image.open(img_path).convert("L"))
        cls = self.labels[index]

        # plt.imshow(image, cmap='gray')
        # plt.show()

        image = torch.from_numpy(image)

        return image, cls

