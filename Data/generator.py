from glob import glob
from random import randint
from torch.utils.data import Dataset
import cv2
import numpy as np

class Transform:
    def __init__(self, crop_size=256):
        self.crop_size = crop_size

    def __call__(self, image):
        h, w, c = image.shape
        if h > self.crop_size and w > self.crop_size:
            start_height = randint(0, h - self.crop_size)
            start_width = randint(0, w - self.crop_size)
            img = np.transpose(image[start_height : start_height + self.crop_size, start_width : start_width + self.crop_size, :], (2, 0, 1))
        else:
            left_h_pad = np.zeros((0, min(w, self.crop_size), c), dtype=np.uint8)
            right_h_pad = np.zeros((0, min(w, self.crop_size), c), dtype=np.uint8)
            left_w_pad = np.zeros((self.crop_size, 0, c), dtype=np.uint8)
            right_w_pad = np.zeros((self.crop_size, 0, c), dtype=np.uint8)
            img = image
            if h < self.crop_size:
                diff = self.crop_size - h
                pad = int(diff / 2)
                left_h_pad = np.zeros((pad, min(w, self.crop_size), c), dtype=np.uint8)
                right_h_pad = np.zeros((diff - pad, min(w, self.crop_size), c), dtype=np.uint8)
            else:
                start_height = randint(0, h - self.crop_size)
                img = img[start_height : start_height + self.crop_size, :, :]

            if w < self.crop_size:
                diff = self.crop_size - w
                pad = int(diff / 2)
                left_w_pad = np.zeros((self.crop_size, pad, c), dtype=np.uint8)
                right_w_pad = np.zeros((self.crop_size, diff - pad, c), dtype=np.uint8)
            else:
                start_width = randint(0, w - self.crop_size)
                img = img[:, start_width : start_width + self.crop_size, :]
            img = np.concatenate((left_h_pad, img, right_h_pad), axis=0)
            img = np.concatenate((left_w_pad, img, right_w_pad), axis=1)
            img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)
        return img

class PreProcessor(Dataset):
    def __init__(self, folder="Data/images", transform=None):
        self.images = glob(f"{folder}/*.jpg")
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        img = cv2.imread(image)
        image = self.transform(img)
        return image