from glob import glob
import cv2
import numpy as np

class DataGen:
	def __init__(self, folder="Data/images", batch_size=1):
		self.folders = glob(f"{folder}/*.jpg")
		self.batch_size = batch_size

	def __call__(self):
		images = []
		while True:
			for image in self.folders:
				img = cv2.imread(image)
				img = cv2.resize(img, (256, 256))
				images.append(img)
				if len(images) == self.batch_size:
					yield np.array(images).astype(np.float32)
					images = []