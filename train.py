from Models.vqgan import VQGan
from Models.discriminator import Discriminator
from Data.generator import PreProcessor, Transform
from Models.losses import calc_quant_loss, adopt_weight, LPIPS, calc_lambda
from torch.utils.data import DataLoader
from torchvision import transforms as T
from collections import deque
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np

class Train:
    def __init__(self, base_channel=128, levels=[1, 1, 2, 2, 4], num_enc_residual_layers=2, num_dec_residual_layers=3, latent_dims=256, num_vectors=1024, num_layers=3, base_filters=64):
        self.device = torch.device("cuda")
        self.vqgan = nn.DataParallel(
            VQGan(base_channel, levels, num_enc_residual_layers, num_dec_residual_layers, latent_dims, num_vectors)
        ).to(self.device)
        self.discriminator = nn.DataParallel(
            Discriminator(num_layers, base_filters)
        ).to(self.device)
        self.opt_vq = torch.optim.Adam(
            self.vqgan.parameters(),
            lr=1e-4
        )
        self.opt_disc = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=1e-4
        )
        transform = T.Compose([Transform(crop_size=256)])
        self.dataset = PreProcessor(folder="Data/images/train2014", transform=transform)
        self.generator = DataLoader(self.dataset, batch_size=8, shuffle=True, num_workers=8)
        self.perc_loss = nn.DataParallel(LPIPS().eval()).to(self.device)

    def __call__(self):
        step = 0
        save_every = 1000
        max_to_keep = 5
        ckpt_dir = "Ckpts/"
        ckpt_queue = deque()
        num_epochs = 100
        for epoch in range(num_epochs):
            with tqdm(self.generator) as bar:
                for imgs in self.generator:
                    imgs = imgs.to(self.device)
                    step += 1
                    bar.update(1)
                    decoded, z_q_ma, z_q, z, inp_quant = self.vqgan(imgs)
                    quant_loss = calc_quant_loss(z_q, z)
                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoded)
                    perceptual_loss = self.perc_loss(imgs, decoded)
                    rec_loss = torch.abs(imgs - decoded)
                    nll_loss = (perceptual_loss + rec_loss).mean()
                    g_loss = -torch.mean(disc_fake)

                    disc_coeff = adopt_weight(step)
                    lmda = 0
                    if disc_coeff > 0:
                        lmda = calc_lambda(nll_loss, g_loss, self.vqgan)
                    vq_loss = nll_loss + quant_loss + disc_coeff * lmda * g_loss

                    d_loss_real = torch.mean(F.relu(1. - disc_real))
                    d_loss_fake = torch.mean(F.relu(1. + disc_fake))

                    gan_loss = disc_coeff * .5 * (d_loss_real + d_loss_fake)

                    self.opt_vq.zero_grad()
                    vq_loss.backward(retain_graph=True)

                    self.opt_disc.zero_grad()
                    gan_loss.backward()

                    self.opt_vq.step()
                    self.opt_disc.step()
                    torch.cuda.empty_cache()
                    bar.set_postfix({"VQ Loss": vq_loss.cpu().detach().numpy(), "Gan Loss": gan_loss.cpu().detach().numpy()})

                    if step % save_every == 0:
                        ckpt_name = f"{ckpt_dir}vqgan_{step}.pt"
                        torch.save(self.vqgan.state_dict(), f"{ckpt_dir}vqgan_{step}.pt")
                        ckpt_queue.append(ckpt_name)
                        if len(ckpt_queue) > max_to_keep:
                            os.remove(ckpt_queue.popleft())


if __name__ == "__main__":
    train = Train()
    train()
