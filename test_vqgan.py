from torch.utils.data import DataLoader
from torchvision import transforms as T
from Models.vqgan import VQGan
from Models.discriminator import Discriminator
from Data.generator import PreProcessor, Transform
import cv2
import torch
import numpy as np

class Test:
    def __init__(self, base_channel=128, levels=[1, 1, 2, 2, 4], num_enc_residual_layers=2, num_dec_residual_layers=3, latent_dims=256, num_vectors=1024, num_layers=3, base_filters=64):
        self.device = torch.device("cpu")
        self.vqgan = VQGan(
            base_channel, levels, num_enc_residual_layers, num_dec_residual_layers, latent_dims, num_vectors
        ).to(self.device)
        transform = T.Compose([Transform(crop_size=256)])
        self.dataset = PreProcessor(folder="Data/images/train2014", transform=transform)
        self.generator = DataLoader(self.dataset, batch_size=8, shuffle=True, num_workers=8)
        self.iterator = iter(self.generator)
        self.vqgan.eval()
        self.model_loaded = False

    def load_model(self, path):
        if not self.model_loaded:
            self.vqgan.load_state_dict(torch.load(path))
            self.model_loaded = True

    def __call__(self, path):
        self.load_model(path)
        imgs = next(self.iterator).to(self.device)
        orig_imgs = imgs.permute(0, 2, 3, 1).detach().numpy().astype(np.uint8)
        with torch.no_grad():
            decoded, z_q_ma, z_q, z, inp_quant = self.vqgan(imgs)
            output = decoded.permute(0, 2, 3, 1).detach().numpy().astype(np.uint8)
            for i in range(len(output)):
                cv2.imwrite(f"Data/out_{i}.jpg", output[i, :, :, :])
                cv2.imwrite(f"Data/orig_{i}.jpg", orig_imgs[i, :, :, :])


if __name__ == "__main__":
    test = Test()
    test(path="Ckpts/saved-ckpts/v2/vqgan_98000.pt")
