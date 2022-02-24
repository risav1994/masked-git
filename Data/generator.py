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
        start_height = randint(0, h - self.crop_size)
        start_width = randint(0, w - self.crop_size)
        return image[start_height : start_height + 256, start_width : start_width + 256, :]

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