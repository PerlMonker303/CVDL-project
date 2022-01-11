import pickle
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import albumentations as A

from utils import format_alpha, enhance_index

ALPHA = 1.3
IMG_DIR = "../../dataset/jpgs/rs19_val/"
MASK_DIR = "../../dataset/uint8/rs19_val/"
BETA = 25  # keep only bounding boxes with size greater than this value
IMAGE_SIZE = (224, 224)
MAX_WIDTH = 1920
MAX_HEIGHT = 1080
GAMMA = 0.75  # percentage of white admitted in an image


class ImageGenerator:
    def __init__(self, image_dir, mask_dir, save_dir, alpha, beta, gamma, image_size=(224,224), write_labels=False):
        self._index = 0
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.image_size = image_size
        self.images = os.listdir(image_dir)
        self.save_dir = f'{save_dir}a_{format_alpha(self.alpha)}/'
        self.labels_path = f'{self.save_dir}labels.txt'
        self.write_labels = write_labels
        if self.write_labels:
            f = open(self.labels_path, 'w')
            f.write('')
            f.close()

        self.resize = A.Compose([A.Resize(width=image_size[0], height=image_size[1])])

        self._removed_images = 0

    def get_image(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index]).replace(".jpg", ".png")
        image = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path))

        mask[mask == 18] = 17
        mask_processed = np.zeros_like(mask)
        rail_color = 17  # 18 for tramway
        mask_processed = np.where(mask == rail_color, 1, mask_processed)
        mask = mask_processed

        return image, mask

    def write_image(self, image, mask):
        pth_image = f'{self.save_dir}image/{enhance_index(self._index)}.png'
        pth_mask = f'{self.save_dir}mask/{enhance_index(self._index)}.png'
        cv2.imwrite(pth_image, image)
        cv2.imwrite(pth_mask, 255 * mask)

    def generate_images(self):
        # alpha = 1 no change
        with open('bounding_boxes_dict.pickle', 'rb') as handle:
            bbs = pickle.load(handle)
            for key in bbs:
                if len(bbs[key]) > 0:
                    for bb in bbs[key]:
                        image, mask = self.get_image(index=key)
                        # plt.imshow(image, cmap='gray')
                        # plt.show()
                        # plt.imshow(mask, cmap='gray')
                        # plt.show()

                        # skip bbs that are too small
                        width = bb[2] - bb[0]
                        height = bb[3] - bb[1]
                        if width < self.beta or height < self.beta:
                            self._removed_images += 1
                            continue

                        cropped_image = self.crop_image(image, bb[0], bb[1], bb[2], bb[3], self.alpha)
                        cropped_mask = self.crop_image(mask, bb[0], bb[1], bb[2], bb[3], self.alpha)

                        # skip bbs that have too much white
                        white_pixels = np.sum(cropped_mask) / (cropped_mask.shape[0] * cropped_mask.shape[0])
                        if white_pixels >= self.gamma:
                            self._removed_images += 1
                            continue

                        # plt.imshow(cropped_image, cmap='gray')
                        # plt.show()
                        # plt.imshow(cropped_mask, cmap='gray')
                        # plt.show()

                        resized = self.resize(image=cropped_image, mask=cropped_mask)
                        cropped_image = resized['image']
                        cropped_mask = resized['mask']

                        # plt.imshow(cropped_image, cmap='gray')
                        # plt.show()
                        # plt.imshow(cropped_mask, cmap='gray')
                        # plt.show()

                        self.write_image(cropped_image, cropped_mask)
                        if self.write_labels:
                            f = open(self.labels_path, 'a')
                            f.write(f'{self._index} {bb[4]}\n')
                            f.close()
                        self._index += 1

        print(f'Removed images: {self._removed_images}')

    def crop_image(self, img, x1, y1, x2, y2, alpha):
        width = x2 - x1
        height = y2 - y1
        x_dif = width * (alpha - 1)
        y_dif = height * (alpha - 1)

        if x1 >= x_dif:
            x1 = int(x1 - x_dif)
        else:
            x1 = 0

        if x2 + x_dif <= MAX_WIDTH:
            x2 = int(x2 + x_dif)
        else:
            x2 = MAX_WIDTH

        if y1 >= y_dif:
            y1 = int(y1 - y_dif)
        else:
            y1 = 0

        if y2 + y_dif <= MAX_HEIGHT:
            y2 = int(y2 + y_dif)
        else:
            y2 = MAX_HEIGHT

        return img[y1:y2, x1:x2]


def generate_images():
    alphas = [1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35]
    betas = [25, 30, 35]
    gammas = [0.75]
    for alpha in alphas:
        for beta in betas:
            for gamma in gammas:
                save_dir = f"./data/b_{beta}/"
                print(f'alpha={alpha};beta={beta};gamma={gamma}')
                ig = ImageGenerator(IMG_DIR, MASK_DIR, save_dir, alpha, beta, gamma, IMAGE_SIZE, True)
                ig.generate_images()


def main():
    generate_images()
    return
    ig = ImageGenerator(IMG_DIR, MASK_DIR, SAVE_DIR, ALPHA, BETA, GAMMA, IMAGE_SIZE)
    ig.generate_images()


if __name__ == "__main__":
    main()
